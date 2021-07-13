package vanillapg

import (
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/network"
	ts "sfneuman.com/golearn/timestep"
)

// TODO:
// Config should take in architecutres and then CreateAgent will return
// an appropriate agent
//
// Can have:
//	Vanilla PG (V + Q + ER) -- Actor Critic
//	Vanilla PG with GAE --> Forward view which is equivalent to:
//					REINFORCE -> Vanilla PG with GAE lambda = 1

//
// ! SAVE COPIES OF THIS FILE, DO GAE FIRST FOLLOWING SPINNING UP THEN:
// ! REWORK THIS FILE TO BE VANILLA PG --> USE ER BUFFER --> Test with fully online
// !
// ! SPINNING UP IS COOL BECAUSE IT'S BASICALLY FANCY REINFORCE!!!!!!!!!!!!!!!!!!!!

// ! Currently only works with a batch size of 1 == 1 entire trajectory
type VPG struct {
	// Policy
	behaviour    agent.PolicyLogProber // Has its own VM
	trainPolicy  agent.PolicyLogProber // Policy struct that is learned
	policySolver G.Solver
	policyVM     G.VM
	advantage    *G.Node

	buffer     *vpgBuffer
	eval       bool
	prevStep   ts.TimeStep
	actionDims int

	// Action value critic
	qCritic   network.NeuralNet
	qNextVal  *G.Node
	qDiscount *G.Node
	qReward   *G.Node
	qVM       G.VM
	qSolver   G.Solver

	// State value critic
	vCritic   network.NeuralNet
	vNextVal  *G.Node
	vDiscount *G.Node
	vReward   *G.Node
	vVM       G.VM
	vSolver   G.Solver
}

func New(env environment.Environment, config SeparateCriticConfig,
	seed int64) (*VPG, error) {

	// Validate and adjust policy/critics as needed
	err := config.Validate()
	if err != nil {
		return nil, fmt.Errorf("new: %v", err)
	}

	// Construct behaviour and training policy
	var behaviour, trainPolicy agent.PolicyLogProber
	if config.Policy.Network().BatchSize() == 1 {
		behaviour = config.Policy
		newPolicy, err := behaviour.CloneWithBatch(config.MaxEpisodeLength)
		if err != nil {
			return nil, fmt.Errorf("new: could not clone behaviour policy: %v",
				err)
		}
		trainPolicy = newPolicy.(agent.PolicyLogProber)
	} else {
		trainPolicy = config.Policy
		newBehaviour, err := trainPolicy.CloneWithBatch(1)
		if err != nil {
			return nil, fmt.Errorf("new: could not clone training policy: %v",
				err)
		}
		behaviour = newBehaviour.(agent.PolicyLogProber)
	}

	features := env.ObservationSpec().Shape.Len()
	batchSize := config.MaxEpisodeLength
	actionDims := env.ActionSpec().Shape.Len()
	buffer := newVPGBuffer(features, actionDims, batchSize)

	vCritic, vNextVal, vDiscount, vReward, vVM,
		err := addMSELoss(config.VCritic, "stateNetwork")
	if err != nil {
		return nil, fmt.Errorf("new: could not create state critic: %v", err)
	}

	qCritic, qNextVal, qDiscount, qReward, qVM,
		err := addMSELoss(config.QCritic, "actionNetwork")
	if err != nil {
		return nil, fmt.Errorf("new: could not create action critic: %v", err)
	}

	// Construct policy loss
	// ! later, we can call _, err := LogProbOf(states, actions) to input the states and actions to get log prob for learning
	graph := trainPolicy.Network().Graph()
	statesPlaceholder := make([]float64, features*batchSize)
	actionsPlaceholder := make([]float64, actionDims*batchSize)
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

	return &VPG{
		trainPolicy:  trainPolicy,
		behaviour:    behaviour,
		policySolver: config.PolicySolver,
		advantage:    advantage,
		policyVM:     policyVM,

		buffer:     buffer,
		eval:       false,
		actionDims: actionDims,
		prevStep:   ts.TimeStep{},

		qCritic:   qCritic,
		qNextVal:  qNextVal,
		qDiscount: qDiscount,
		qReward:   qReward,
		qVM:       qVM,
		qSolver:   config.QSolver,

		vCritic:   vCritic,
		vNextVal:  vNextVal,
		vDiscount: vDiscount,
		vReward:   vReward,
		vVM:       vVM,
		vSolver:   config.VSolver,
	}, nil
}

// ! This should be moved to the config.CreateAgent() function
// ! We need one for state and one for action critic
// func createCritic(features, batchSize int, layers []int,
// 	biases []bool, activations []*network.Activation,
// 	init G.InitWFn, name string) (network.NeuralNet, *G.Node, *G.Node,
// 	*G.Node, G.VM, error) {
// 	// Create the state value function critic
// 	graph := G.NewGraph()
// 	critic, err := network.NewMultiHeadMLP(features, batchSize, 1, graph,
// 		layers, biases, init, activations)
// 	if err != nil {
// 		return nil, nil, nil, nil, nil,
// 			fmt.Errorf("new: could not create critic: %v", err)
// 	}

// 	if len(critic.Prediction()) > 1 {
// 		// This should never happen
// 		msg := fmt.Sprintf("createCritic: illegal number of outputs for "+
// 			"critic \n\twant(1)\n\thave(%v)", len(critic.Prediction()))
// 		panic(msg)
// 	}

// 	// Add critic update target input (next state value, reward, discount)
// 	nextVal := G.NewVector(
// 		graph,
// 		tensor.Float64,
// 		G.WithShape(batchSize),
// 		G.WithName(fmt.Sprintf("%vNextValue", name)),
// 	)
// 	reward := G.NewVector(
// 		graph,
// 		tensor.Float64,
// 		G.WithShape(batchSize),
// 		G.WithName(fmt.Sprintf("%vReward", name)),
// 	)
// 	discount := G.NewVector(
// 		graph,
// 		tensor.Float64,
// 		G.WithShape(batchSize),
// 		G.WithName(fmt.Sprintf("%vDiscount", name)))
// 	target := G.Must(G.HadamardProd(nextVal, discount))
// 	target = G.Must(G.Add(reward, target))

// 	// Critic loss
// 	loss := G.Must(G.Sub(critic.Prediction()[0], target))
// 	loss = G.Must(G.Square(loss))
// 	loss = G.Must(G.Mean(loss))

// 	// Gradient of loss
// 	_, err = G.Grad(loss, critic.Learnables()...)
// 	if err != nil {
// 		// This should never happen
// 		msg := fmt.Sprintf("new: could not compute gradient: %v", err)
// 		panic(msg)
// 	}

// 	// Compute gradient on critic loss
// 	criticVM := G.NewTapeMachine(
// 		graph,
// 		G.BindDualValues(critic.Learnables()...),
// 	)

// 	return critic, nextVal, discount, reward, criticVM, nil
// }

// Returns network, next value, reward, and discount for creating the
// target plus the VM for running the network
func addMSELoss(net network.NeuralNet, name string) (network.NeuralNet, *G.Node, *G.Node,
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
	reward := G.NewVector(
		graph,
		tensor.Float64,
		G.WithShape(net.BatchSize()),
		G.WithName(fmt.Sprintf("%vReward", name)),
	)
	discount := G.NewVector(
		graph,
		tensor.Float64,
		G.WithShape(net.BatchSize()),
		G.WithName(fmt.Sprintf("%vDiscount", name)))
	target := G.Must(G.HadamardProd(nextVal, discount))
	target = G.Must(G.Add(reward, target))

	// Critic loss
	loss := G.Must(G.Sub(net.Prediction()[0], target))
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
	return net, nextVal, discount, reward, criticVM, nil
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

	state := v.prevStep.Observation.RawVector().Data
	a := action.(*mat.VecDense).RawVector().Data
	reward := nextStep.Reward
	discount := nextStep.Discount

	v.buffer.store(state, a, reward, discount)

	v.prevStep = nextStep
}

func (v *VPG) TdError(ts.Transition) float64 {
	panic("tderror: not implemented")
}

func (v *VPG) Step() {
	// 1. Calculate V for the full batch - size N
	// 2. Calculate Q for the full batch - size N
	// 3. Calculate advantage V - Q - size N
	// 4. Set log prob input
	// 5. Set advantage input
	// 6. Run policy VM to learn
	// 7. Calculate V(next state)
	// 8. Input v(next state), reward, discount
	// 9. Run V VM
	// 10. Calculate Q(next state, next action)
	// 11. Input Q(next state, next action), reward, discount
	// 12. Run Q VM
	// 13. Copy train policy -> behaviour policy

	state, action, reward, discount, err := v.buffer.get()
	if err != nil {
		panic(fmt.Sprintf("could not sample from buffer: %v", err))
	}

	v.vCritic.SetInput(state)
	v.vVM.RunAll()
	stateValues := v.vCritic.Output()[0].(*tensor.Dense)
	v.vVM.Reset()

	// Based on the type of critic network, actions and states may need
	// to be interleaved. RevTreeMLP expects two separate inputs, one
	// for actions and another for states. Other MLPs expect a single
	// input (which must already be interleaved).
	switch v.qCritic.(type) {
	case *network.RevTreeMLP:
		v.qCritic.SetInput(append(state, action...))
	default:
		// Here, we'd have to interleave the states and actions for a
		// multi head MLP e.g. or tree MLP
		panic("not implemented")
	}
	v.qVM.RunAll()
	actionValues := v.qCritic.Output()[0].(*tensor.Dense)
	v.qVM.Reset()

	advantages, err := tensor.Sub(actionValues, stateValues)
	if err != nil {
		panic(fmt.Sprintf("step: could not calculate advantages: %v", err))
	}

	err = G.Let(v.advantage, advantages)
	if err != nil {
		panic(fmt.Sprintf("step: could not set advantages in policy "+
			"gradient: %v", err))
	}

	_, err = v.trainPolicy.LogProbOf(state, action)
	if err != nil {
		panic(fmt.Sprintf("step: could not calculate log probabilities: %v",
			err))
	}

	v.policyVM.RunAll()
	v.policyVM.Reset()

	// Update state critic
	err = G.Let(v.vDiscount, discount)
	if err != nil {
		panic(fmt.Sprintf("step: could not set state critic discount: %v", err))
	}

	err = G.Let(v.vReward, reward)
	if err != nil {
		panic(fmt.Sprintf("step: could not set state critic rewards: %v", err))
	}

	nextStateVals :=
}
