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

var globalFloats []float64

var flag bool = false

// var flag bool = false
var CLoss *G.Node
var CLossVal G.Value
var Prediction G.Value
var LogProbVal G.Value

var PolicyLoss G.Value
var ValueLoss G.Value

// TODO:
// Config should take in architecutres and then CreateAgent will return
// an appropriate agent
//
//	ValueFn should be called ValueFn not ValueFn
//
// Can have:
//	Vanilla PG (V + Q + ER) -- Actor ValueFn
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
//!			3. Create an OnlineEpochExperiment type that resets episodes at the end of an epoch !!!!!!!!!!!!!!!!!!!!
//

// !
// ! SPINNING UP IS COOL BECAUSE IT'S BASICALLY FANCY REINFORCE!!!!!!!!!!!!!!!!!!!!

// ! Currently only works with a batch size of 1 == 1 entire trajectory
type VPG struct {
	// Policy
	behaviour         agent.NNPolicy        // Has its own VM
	trainPolicy       agent.PolicyLogProber // Policy struct that is learned
	trainPolicySolver G.Solver
	trainPolicyVM     G.VM
	advantages        *G.Node
	logProb           *G.Node

	buffer           *vpgBuffer
	epochLength      int
	currentEpochStep int
	completedEpochs  int
	eval             bool

	prevStep   ts.TimeStep
	actionDims int

	// State value critic
	vValueFn             network.NeuralNet
	vVM                  G.VM
	vTrainValueFn        network.NeuralNet
	vTrainValueFnVM      G.VM
	vTrainValueFnTargets *G.Node
	vSolver              G.Solver
	valueGradSteps       int
}

func New(env environment.Environment, c Config,
	seed int64) (*VPG, error) {
	if !c.ValidAgent(&VPG{}) {
		return nil, fmt.Errorf("new: invalid configuration type: %T", c)
	}

	config, ok := c.(*CategoricalMLPConfig)
	if !ok {
		return nil, fmt.Errorf("new: invalid configuration type: %T", c)
	}

	// Validate and adjust policy/critics as needed
	err := config.Validate()
	if err != nil {
		return nil, fmt.Errorf("new: %v", err)
	}

	// Create the VPG buffer
	features := env.ObservationSpec().Shape.Len()
	actionDims := env.ActionSpec().Shape.Len()
	buffer := newVPGBuffer(features, actionDims, config.BatchSize(),
		config.Lambda, config.Gamma)

	// Create the prediction value function
	valueFn := config.vValueFn
	vVM := G.NewTapeMachine(valueFn.Graph())

	// Create the training value function
	trainValueFn := config.vTrainValueFn

	trainValueFnTargets := G.NewMatrix(
		trainValueFn.Graph(),
		tensor.Float64,
		G.WithShape(trainValueFn.Prediction()[0].Shape()...),
		G.WithName("Value Function Update Target"),
	)

	valueFnLoss := G.Must(G.Sub(trainValueFn.Prediction()[0], trainValueFnTargets))
	valueFnLoss = G.Must(G.Square(valueFnLoss))
	valueFnLoss = G.Must(G.Mean(valueFnLoss))
	G.Read(valueFnLoss, &ValueLoss)

	_, err = G.Grad(valueFnLoss, trainValueFn.Learnables()...)
	if err != nil {
		panic(err)
	}

	trainValueFnVM := G.NewTapeMachine(trainValueFn.Graph(), G.BindDualValues(trainValueFn.Learnables()...))

	// Create the prediction policy
	behaviour := config.behaviour

	// Create the training policy
	trainPolicy := config.policy
	logProb := trainPolicy.(*policy.CategoricalMLP).LogProbNode()
	advantages := G.NewVector(
		trainPolicy.Network().Graph(),
		tensor.Float64,
		G.WithName("Advantages"),
		G.WithShape(config.EpochLength),
	)

	policyLoss := G.Must(G.HadamardProd(logProb, advantages))
	policyLoss = G.Must(G.Mean(policyLoss))
	policyLoss = G.Must(G.Neg(policyLoss))

	_, err = G.Grad(policyLoss, trainPolicy.Network().Learnables()...)
	if err != nil {
		panic(err)
	}
	G.Read(trainPolicy.(*policy.CategoricalMLP).LogProbNode(), &LogProbVal)
	G.Read(policyLoss, &PolicyLoss)

	trainPolicyVM := G.NewTapeMachine(trainPolicy.Network().Graph(), G.BindDualValues(trainPolicy.Network().Learnables()...))

	vpg := &VPG{
		behaviour:         behaviour,
		trainPolicy:       trainPolicy,
		trainPolicyVM:     trainPolicyVM,
		trainPolicySolver: config.PolicySolver,
		advantages:        advantages,
		logProb:           logProb,

		vValueFn: valueFn,
		vVM:      vVM,

		vTrainValueFn:        trainValueFn,
		vTrainValueFnTargets: trainValueFnTargets,
		vTrainValueFnVM:      trainValueFnVM,
		vSolver:              config.VSolver,
		valueGradSteps:       config.ValueGradSteps,

		buffer:           buffer,
		epochLength:      config.EpochLength,
		currentEpochStep: 0,
		completedEpochs:  0,
		eval:             false,
	}

	return vpg, nil
}

func (v *VPG) SelectAction(t ts.TimeStep) *mat.VecDense {
	if t != v.prevStep {
		panic("selectAction: timestep is different from that previously " +
			"recorded")
	}
	action := v.behaviour.SelectAction(t)
	// fmt.Println(action)
	return action
}

func (v *VPG) EndEpisode() {}

func (v *VPG) Eval() {}

func (v *VPG) Train() {}

func (v *VPG) ObserveFirst(t ts.TimeStep) {
	if !t.First() {
		fmt.Fprintf(os.Stderr, "Warning: ObserveFirst() should only be"+
			"called on the first timestep (current timestep = %d)", t.Number)
	}
	flag = false
	v.prevStep = t
}

// Observe observes and records any timestep other than the first timestep
func (v *VPG) Observe(action mat.Vector, nextStep ts.TimeStep) {
	// Finish current episode to end epoch
	if flag {
		v.prevStep = nextStep
		return
	}

	// Calculate value of previous step
	o := v.prevStep.Observation.RawVector().Data
	err := v.vValueFn.SetInput(o)
	if err != nil {
		panic(err)
	}
	err = v.vVM.RunAll()
	if err != nil {
		panic(err)
	}
	vT := v.vValueFn.Output()[0].Data().([]float64)
	v.vVM.Reset()
	if len(vT) != 1 {
		panic("observe: multiple values predicted for state value")
	}
	r := nextStep.Reward
	a := action.(*mat.VecDense).RawVector().Data
	v.buffer.store(o, a, r, vT[0])

	// Update obs (critical!)
	v.prevStep = nextStep
	o = nextStep.Observation.RawVector().Data

	v.currentEpochStep++
	terminal := nextStep.Last() || v.currentEpochStep == v.epochLength
	if terminal {
		if nextStep.TerminalEnd() {
			v.buffer.finishPath(0.0)
		} else {
			err := v.vValueFn.SetInput(o)
			if err != nil {
				panic(err)
			}
			err = v.vVM.RunAll()
			if err != nil {
				panic(err)
			}
			lastVal := v.vValueFn.Output()[0].Data().([]float64)
			v.vVM.Reset()
			if len(lastVal) != 1 {
				panic("observe: multiple values predicted for next state value")
			}
			v.buffer.finishPath(lastVal[0])
			flag = v.currentEpochStep == v.epochLength
		}
	}
}

func (v *VPG) TdError(ts.Transition) float64 {
	panic("tderror: not implemented")
}

func (v *VPG) Step() {
	if v.currentEpochStep < v.epochLength {
		return
	}

	obs, act, adv, ret, err := v.buffer.get()
	if err != nil {
		panic(err)
	}

	// Policy gradient step
	advantagesTensor := tensor.NewDense( // * technically this needs to be called only once
		tensor.Float64,
		v.advantages.Shape(),
		tensor.WithBacking(adv),
	)
	err = G.Let(v.advantages, advantagesTensor)
	if err != nil {
		panic(err)
	}
	v.trainPolicy.(*policy.CategoricalMLP).LogProbOf(obs, act)
	if err := v.trainPolicyVM.RunAll(); err != nil {
		panic(err)
	}
	// fmt.Println("Policy Gradient")

	// grad, err := v.trainPolicy.Network().(*network.MultiHeadMLP).Layers()[3].Weights().Grad()
	// if err != nil {
	// 	panic(err)
	// }
	// fmt.Println(floatutils.NonZero(grad.Data().([]float64)))

	fmt.Println(v.trainPolicy.Network().(*network.MultiHeadMLP).Layers()[3].Weights().Grad())
	if err := v.trainPolicySolver.Step(v.trainPolicy.Network().Model()); err != nil {
		panic(err)
	}

	fmt.Println("Policy Loss:", PolicyLoss)
	// fmt.Println("Logits:", v.trainPolicy.(*policy.CategoricalMLP).Logits())
	fmt.Println("Log Prob:", v.trainPolicy.(*policy.CategoricalMLP).LogProbInputActionsVal)
	fmt.Println("Log Sum Exp:", v.trainPolicy.(*policy.CategoricalMLP).LogSumExp)

	// fmt.Println(v.trainPolicy.(*policy.GaussianTreeMLP).Std)
	// fmt.Println(v.trainPolicy.(*policy.GaussianTreeMLP).Mean)
	// fmt.Println(v.trainPolicy.(*policy.CategoricalMLP).LogProbNode().Value())
	// fmt.Println(LogProbVal)
	// fmt.Println(v.trainPolicy.(*policy.GaussianTreeMLP).ExternActionsLogProbVal)
	// fmt.Println(v.trainPolicy.(*policy.GaussianTreeMLP).ExternActionsVal)
	// fmt.Println("=============================")
	// fmt.Println()
	// fmt.Println()

	v.trainPolicyVM.Reset()

	// Value function update
	trainValueFnTargetsTensor := tensor.NewDense( // * this actually can be called once and saved
		tensor.Float64,
		v.vTrainValueFnTargets.Shape(),
		tensor.WithBacking(ret),
	)
	err = G.Let(v.vTrainValueFnTargets, trainValueFnTargetsTensor)
	if err != nil {
		panic(err)
	}
	if err := v.vTrainValueFnVM.RunAll(); err != nil {
		panic(err)
	}
	// fmt.Println("Value Function Gradient")
	// grad, err = v.vTrainValueFn.(*network.MultiHeadMLP).Layers()[3].Weights().Grad()
	// if err != nil {
	// 	panic(err)
	// }
	// fmt.Println(floatutils.NonZero(grad.Data().([]float64)))
	// fmt.Println(v.vTrainValueFn.(*network.MultiHeadMLP).Layers()[3].Weights().Grad())
	if err := v.vSolver.Step(v.vTrainValueFn.Model()); err != nil {
		panic(err)
	}

	// fmt.Println("Value Loss:", ValueLoss)
	v.vTrainValueFnVM.Reset()

	// Update behaviour policy and prediction value funcion
	network.Set(v.behaviour.Network(), v.trainPolicy.Network())
	network.Set(v.vValueFn, v.vTrainValueFn)
	v.completedEpochs++
	v.currentEpochStep = 0
}
