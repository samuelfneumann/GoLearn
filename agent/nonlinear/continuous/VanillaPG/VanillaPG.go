package vanillapg

import (
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/agent/nonlinear/continuous/policy"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/network"
	ts "sfneuman.com/golearn/timestep"
)

// TODO:
// Config should take in architecutres and then CreateAgent will return
// an appropriate agent
//
//	Critic should be called ValueFn not Critic
//
// Can have:
//	Vanilla PG (V + Q + ER) -- Actor Critic
//	Vanilla PG with GAE --> Forward view which is equivalent to:
//					REINFORCE -> Vanilla PG with GAE lambda = 1
//
// ! EPOCH END --> ENV RESET
// ! 	INTERESTING: if we don't reset the env, the last epoch used to state value on the last timestep
// !					to estimate the rest of the rewards. If we do NOT reset the environment, then
// !					the next epoch will begin with an episode halfway through, but this may not be
// !					the worst thing in the world. We will still accurately estimate that state's value and the policy gradient.
// !	FIXES FOR ENDING ENV ON EPOCH END:
// !		1. Don't
// !		2. Let agents send the the experiment signals at every Observe()
//!			3. Create an OnlineEpochExperiment type that resets episodes at the end of an epoch
//
// ! SAVE COPIES OF THIS FILE, DO GAE FIRST FOLLOWING SPINNING UP THEN:
// ! REWORK THIS FILE TO BE VANILLA PG --> USE ER BUFFER --> Test with fully online
// !
// ! SPINNING UP IS COOL BECAUSE IT'S BASICALLY FANCY REINFORCE!!!!!!!!!!!!!!!!!!!!

// ! Currently only works with a batch size of 1 == 1 entire trajectory
type VPG struct {
	// Policy
	behaviour    agent.NNPolicy        // Has its own VM
	trainPolicy  agent.PolicyLogProber // Policy struct that is learned
	policySolver G.Solver
	policyVM     G.VM
	advantage    *G.Node

	buffer           *vpgBuffer
	epochLength      int
	currentEpochElem int
	completedEpochs  int
	eval             bool
	prevStep         ts.TimeStep
	actionDims       int

	// State value critic
	vCritic        network.NeuralNet
	vVM            G.VM
	vTrainCritic   network.NeuralNet
	vTrainVM       G.VM
	vNextVal       *G.Node
	vSolver        G.Solver
	valueGradSteps int
}

func New(env environment.Environment, c Config,
	seed int64) (*VPG, error) {
	if !c.ValidAgent(&VPG{}) {
		return nil, fmt.Errorf("new: invalid configuration type: %T", c)
	}

	config, ok := c.(*TreePolicyConfig)
	if !ok {
		return nil, fmt.Errorf("new: invalid configuration type: %T", c)
	}

	// Validate and adjust policy/critics as needed
	err := config.Validate()
	if err != nil {
		return nil, fmt.Errorf("new: %v", err)
	}

	features := env.ObservationSpec().Shape.Len()
	actionDims := env.ActionSpec().Shape.Len()
	epochLength := config.EpochLength
	buffer := newVPGBuffer(features, actionDims, epochLength, config.Lambda,
		config.Gamma)

	// Construct behaviour and training policy
	behaviour, err := config.policy.CloneWithBatch(1)
	if err != nil {
		return nil, fmt.Errorf("new: could not construct behaviour policy: %v",
			err)
	}
	trainPolicy := config.policy
	// fmt.Println("Train policy", trainPolicy.Actions())
	// fmt.Println("Behave policy", trainPolicy.Actions())

	// Monte-Carlo method: vNextVal == sum of rewards following current step
	vCritic := config.vCritic
	vVM := G.NewTapeMachine(vCritic.Graph())
	vTrainCritic, err := vCritic.CloneWithBatch(epochLength)
	if err != nil {
		return nil, fmt.Errorf("new: could not clone train critic: %v", err)
	}
	vTrainCritic, vNextVal, vTrainVM,
		err := addMSELoss(vTrainCritic, "stateNetwork")
	if err != nil {
		return nil, fmt.Errorf("new: could not create state critic: %v", err)
	}

	// Construct policy loss
	graph := trainPolicy.Network().Graph()
	statesPlaceholder := make([]float64, features*epochLength)
	actionsPlaceholder := make([]float64, actionDims*epochLength)
	logProb, err := trainPolicy.LogProbOf(statesPlaceholder, actionsPlaceholder)
	if err != nil {
		return nil, fmt.Errorf("new: could not compute log(∇π): %v", err)
	}
	advantage := G.NewVector(
		graph,
		tensor.Float64,
		G.WithShape(logProb.Shape()...),
		G.WithName("advantage"),
	)

	negative := G.NewConstant(-1.0)
	policyLoss := G.Must(G.HadamardProd(logProb, advantage))
	policyLoss = G.Must(G.Sum(policyLoss))
	policyLoss = G.Must(G.Mean(policyLoss))
	policyLoss = G.Must(G.Mul(negative, policyLoss))

	// Gradient of policy loss
	_, err = G.Grad(policyLoss, trainPolicy.Network().Learnables()...)
	if err != nil {
		panic(fmt.Sprintf("new: could not compute policy gradient: %v", err))
	}
	policyVM := G.NewTapeMachine(
		graph,
		G.BindDualValues(trainPolicy.Network().Learnables()...),
	)

	retVal := &VPG{
		trainPolicy:  trainPolicy,
		behaviour:    behaviour,
		policySolver: config.PolicySolver,
		advantage:    advantage,
		policyVM:     policyVM,

		buffer:           buffer,
		epochLength:      config.EpochLength,
		currentEpochElem: 0,
		completedEpochs:  0,

		eval:       false,
		actionDims: actionDims,
		prevStep:   ts.TimeStep{},

		vCritic:        vCritic,
		vVM:            vVM,
		vTrainCritic:   vTrainCritic,
		vNextVal:       vNextVal,
		vTrainVM:       vTrainVM,
		vSolver:        config.VSolver,
		valueGradSteps: config.ValueGradSteps,
	}

	return retVal, nil
}

// Returns network, next value, reward, and discount for creating the
// target plus the VM for running the network
func addMSELoss(net network.NeuralNet, name string) (network.NeuralNet,
	*G.Node, G.VM, error) {
	graph := net.Graph()

	if len(net.Prediction()) > 1 {
		// This should never happen
		msg := fmt.Sprintf("addMSELoss: illegal number of outputs for "+
			"critic \n\twant(1)\n\thave(%v)", len(net.Prediction()))
		panic(msg)
	}

	// Add critic update target input (next state value, reward, discount)
	nextVal := G.NewVector(
		graph,
		tensor.Float64,
		G.WithShape(net.BatchSize()),
		G.WithName(fmt.Sprintf("%vNextValue", name)),
	)

	// Critic loss
	loss := G.Must(G.Sub(net.Prediction()[0], nextVal))
	loss = G.Must(G.Square(loss))
	loss = G.Must(G.Mean(loss))

	// Gradient of loss
	_, err := G.Grad(loss, net.Learnables()...)
	if err != nil {
		// This should never happen
		msg := fmt.Sprintf("new: could not compute gradient: %v", err)
		panic(msg)
	}

	// Compute gradient on critic loss
	criticVM := G.NewTapeMachine(
		graph,
		G.BindDualValues(net.Learnables()...),
	)
	return net, nextVal, criticVM, nil
}

func (v *VPG) SelectAction(t ts.TimeStep) *mat.VecDense {
	if v.eval {
		switch p := v.behaviour.(type) {
		case *policy.GaussianTreeMLP:
			p.SelectAction(t)
			return mat.NewVecDense(len(p.Mean()), p.Mean())

		default:
			panic(fmt.Sprintf("selectAction: invalid behaviour policy "+
				"%T", v.behaviour))
		}
	}

	// return mat.NewVecDense(1, []float64{0.1})
	// time.Sleep(time.Millisecond * 10)
	action := v.behaviour.SelectAction(t)

	// fmt.Println("Action:", action)

	return action
}

func (v *VPG) EndEpisode() {}

func (v *VPG) Eval() {
	v.eval = true
}

func (v *VPG) Train() {
	v.eval = false
}

func (v *VPG) ObserveFirst(t ts.TimeStep) {
	if !t.First() {
		fmt.Fprintf(os.Stderr, "Warning: ObserveFirst() should only be"+
			"called on the first timestep (current timestep = %d)", t.Number)
	}
	v.prevStep = t
}

// Observe observes and records any timestep other than the first timestep
func (v *VPG) Observe(action mat.Vector, nextStep ts.TimeStep) {
	if action.Len() != v.actionDims {
		msg := fmt.Sprintf("observe: illegal action dimensions \n\twant(%v)"+
			"\n\thave(%v)", v.actionDims, action.Len())
		panic(msg)
	}

	// Store data in the buffer
	state := v.prevStep.Observation.RawVector().Data
	err := v.vCritic.SetInput(state)
	if err != nil {
		panic(fmt.Sprintf("observe: cannot set critic input for advantage "+
			"estimation: %v", err))
	}
	v.vVM.RunAll()
	stateVal := v.vCritic.Output()[0].Data().([]float64)[0]
	v.vVM.Reset()

	a := action.(*mat.VecDense).RawVector().Data
	reward := nextStep.Reward
	discount := nextStep.Discount
	v.buffer.store(state, a, reward, stateVal, discount)

	// If the episode has ended, finsih the path in the buffer by computing
	// the advantages for the episode
	v.currentEpochElem += 1
	if v.currentEpochElem >= v.epochLength {
		v.buffer.finishPath(stateVal)
	} else if v.prevStep.Last() {
		v.buffer.finishPath(0.0)
	}

	v.prevStep = nextStep
}

func (v *VPG) TdError(ts.Transition) float64 {
	panic("tderror: not implemented")
}

func (v *VPG) Step() {
	if v.currentEpochElem < v.epochLength {
		return
	}
	// fmt.Println("=== === Step() ")

	state, action, adv, ret, err := v.buffer.get()
	if err != nil {
		panic(fmt.Sprintf("could not sample from buffer: %v", err))
	}

	// Set the advantage in the trainPolicy's computational graph
	advantage := tensor.NewDense(tensor.Float64, v.advantage.Shape(),
		tensor.WithBacking(adv))
	err = G.Let(v.advantage, advantage)
	if err != nil {
		panic(fmt.Sprintf("step: could not set advantages in policy "+
			"gradient: %v", err))
	}

	// Set the states and actions to compute the log probability of
	// for the trainPolicy's gradient in its graph
	_, err = v.trainPolicy.LogProbOf(state, action)
	if err != nil {
		panic(fmt.Sprintf("step: could not calculate log probabilities: %v",
			err))
	}

	v.policyVM.RunAll()
	// fmt.Println(v.trainPolicy.LogProb().Value())
	// fmt.Println(v.trainPolicy.(*policy.GaussianTreeMLP).Std())
	// fmt.Println(v.advantage.Value())
	// fmt.Println(v.policyCost.Value())
	// time.Sleep(time.Millisecond * 1000)
	v.policySolver.Step(v.trainPolicy.Network().Model())
	v.policyVM.Reset()

	// v.trainPolicy.(*policy.GaussianTreeMLP).PrintVals()

	// Monte-Carlo method: nextValue = sum of rewards
	returns := tensor.NewDense(tensor.Float64, v.vNextVal.Shape(),
		tensor.WithBacking(ret))
	err = G.Let(v.vNextVal, returns)
	if err != nil {
		panic(fmt.Sprintf("step: could not set state critic update target: %v",
			err))
	}

	for i := 0; i < v.valueGradSteps; i++ {
		v.vVM.RunAll()
		v.vSolver.Step(v.vTrainCritic.Model())
		v.vVM.Reset()
	}

	v.currentEpochElem = 0
	v.completedEpochs++
	network.Set(v.behaviour.Network(), v.trainPolicy.Network())
	network.Set(v.vCritic, v.vTrainCritic)
}
