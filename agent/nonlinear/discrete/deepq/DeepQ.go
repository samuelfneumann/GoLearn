package deepq

import (
	"fmt"
	"log"
	"os"

	"gonum.org/v1/gonum/mat"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"sfneuman.com/golearn/agent/linear/discrete/qlearning"
	"sfneuman.com/golearn/agent/nonlinear/discrete/policy"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/expreplay"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
)

// Config implements a configuration for a DeepQ agent
type Config struct {
	PolicyLayers []int
	Biases       []bool
	Activations  []policy.Activation
	InitWFn      G.InitWFn
	LearningRate float64
	Epsilon      float64

	// Experience replay parameters
	Remover         expreplay.Selector
	Sampler         expreplay.Selector
	MaximumCapacity int
	MinimumCapacity int

	// Target net updates
	Tau                  float64 // Polyak averaging constant
	TargetUpdateInterval int     // Number of steps target network updates
}

// BatchSize returns the batch size of the agent constructed using this
// Config
func (c Config) BatchSize() int {
	return c.Sampler.BatchSize()
}

// NewQLearning returns a DeepQ agent that uses Q-learning.
// That is, the algorithm uses linear function approximation to learn
// online with no target networks.
func NewQlearning(env environment.Environment, config qlearning.Config,
	seed int64, InitWFn G.InitWFn) (*DeepQ, error) {
	deepQConfig := Config{
		Epsilon:      config.Epsilon,
		LearningRate: config.LearningRate,
		PolicyLayers: []int{},
		Biases:       []bool{},
		Activations:  []policy.Activation{},
		InitWFn:      InitWFn,

		Tau:                  1.0,
		TargetUpdateInterval: 1,
	}
	return New(env, deepQConfig, seed)
}

// DeepQ implements the deep Q-learning algorithm. This algorithm is
// conceptually similar to DQN, but uses the MSE loss. Currently, DeepQ
// only works online.
type DeepQ struct {
	policy    *policy.MultiHeadEGreedyMLP // behaviour policy
	trainNet  *policy.MultiHeadEGreedyMLP
	targetNet *policy.MultiHeadEGreedyMLP // policy providing the update target

	selectedActions *G.Node // Actions selected at the previous states in the batch
	numActions      int

	replay expreplay.ExperienceReplayer

	// nextStateActionValues is the input node in the graph of policy that
	// is given the action values of the next state. For update:
	//
	// Q(s, a) <- Q(s, a) + α * (r + Q(s', a') - Q(s, a)) ∇Q(s, a)
	//
	// nextStateActionValues provides Q(s', a') for all a' in s'
	nextStateActionValues *G.Node
	rewards               *G.Node
	discounts             *G.Node

	// VMs and solver for running the computational graphs
	vm       G.VM
	trainVM  G.VM
	targetVM G.VM
	solver   G.Solver

	prevStep   ts.TimeStep
	prevAction int
	nextStep   ts.TimeStep

	learningRate float64
	batchSize    int

	// Target network updates
	tau                  float64 // Polyak averaging constant
	targetUpdateInterval int     // Steps between target updates
	gradientSteps        int
}

// New creates and returns a new DeepQ agent
func New(env environment.Environment, config Config,
	seed int64) (*DeepQ, error) {
	// Ensure environment has discrete actions
	if env.ActionSpec().Cardinality != spec.Discrete {
		return &DeepQ{}, fmt.Errorf("deepq: cannot use non-discrete " +
			"actions")
	}
	if env.ActionSpec().LowerBound.Len() > 1 {
		return &DeepQ{}, fmt.Errorf("deepq: actions must be " +
			"1-dimensional")
	}
	if env.ActionSpec().LowerBound.AtVec(0) != 0.0 {
		return &DeepQ{}, fmt.Errorf("deepq: actions must be " +
			"enumerated starting from 0")
	}

	// Configuration variables
	batchSize := config.BatchSize()
	numActions := int(env.ActionSpec().UpperBound.AtVec(0)) + 1
	hiddenSizes := config.PolicyLayers
	biases := config.Biases
	activations := config.Activations
	init := config.InitWFn
	learningRate := config.LearningRate
	ε := config.Epsilon

	// Error checking
	if len(hiddenSizes) != len(biases) {
		msg := fmt.Sprintf("new: invalid number of biases\n\twant(%v)"+
			"\n\thave(%v)", len(hiddenSizes), len(biases))
		return &DeepQ{}, fmt.Errorf(msg)
	}
	if len(hiddenSizes) != len(activations) {
		msg := fmt.Sprintf("new: invalid number of activations\n\twant(%v)"+
			"\n\thave(%v)", len(hiddenSizes), len(activations))
		return &DeepQ{}, fmt.Errorf(msg)
	}

	g := G.NewGraph()

	// Behaviour network
	policy, _ := policy.NewMultiHeadEGreedyMLP(
		ε,
		env,
		1, // For behaviour policy, we only need to select a single action
		g,
		hiddenSizes,
		biases,
		init,
		activations,
		seed,
	)

	// Clone the policy network to create a target net which gives the
	// update target
	targetNet, err := policy.CloneWithBatch(batchSize)
	targetNet.SetEpsilon(0.0) // Qlearning target policy is greedy
	if err != nil {
		msg := "new: could not create target network: %v"
		return &DeepQ{}, fmt.Errorf(msg, err)
	}
	gTarget := targetNet.Graph()

	// Target network updating schedule
	tau := config.Tau
	targetUpdateInterval := config.TargetUpdateInterval
	if targetUpdateInterval < 1 {
		err := fmt.Errorf("new: target networks must be updated at positive "+
			"timestep intervals \n\twant(>0) \n\thave(%v)",
			targetUpdateInterval)
		return &DeepQ{}, err
	}

	// Create a learning network which will learn the weights
	trainNet, err := policy.CloneWithBatch(batchSize)
	if err != nil {
		msg := "new: could not create learning network: %v"
		return &DeepQ{}, fmt.Errorf(msg, err)
	}
	gTrain := trainNet.Graph()

	// Create nodes to compute the update target: r + γ * max[Q(s', a')]
	nextStateActionValues := G.NewMatrix(gTrain, tensor.Float64,
		G.WithShape(batchSize, numActions), G.WithName("targetActionVals"))
	rewards := G.NewVector(gTrain, tensor.Float64, G.WithShape(batchSize),
		G.WithName("reward"))
	discounts := G.NewVector(gTrain, tensor.Float64, G.WithShape(batchSize),
		G.WithName("discount"))

	// Compute the update target
	updateTarget := G.Must(G.Max(nextStateActionValues, 1))
	updateTarget = G.Must(G.HadamardProd(updateTarget, discounts))
	updateTarget = G.Must(G.Add(updateTarget, rewards))

	// Action selected by the policy in the previous state. This is
	// needed to compute the loss using the correct action value since
	// the network outputs N action values, one for each environmental
	// action
	selectedActions := G.NewMatrix(
		gTrain,
		tensor.Float64,
		G.WithName("actionSelected"),
		G.WithShape(batchSize, numActions),
	)
	selectedActionsValue := G.Must(G.HadamardProd(trainNet.Prediction(), // ! This will change when ER is added
		selectedActions))
	selectedActionsValue = G.Must(G.Sum(selectedActionsValue, 1))

	// Compute the Mean Squarred TD error
	losses := G.Must(G.Sub(updateTarget, selectedActionsValue))
	losses = G.Must(G.Square(losses))
	cost := G.Must(G.Mean(losses))

	// Compute the gradient with respect to the Mean Squarred TD error
	_, err = G.Grad(cost, trainNet.Learnables()...)
	if err != nil {
		msg := fmt.Sprintf("new: could not compute gradient: %v", err)
		panic(msg)
	}

	// Create the VMs and solver for running the policy
	vm := G.NewTapeMachine(g)
	targetVM := G.NewTapeMachine(gTarget)
	trainVM := G.NewTapeMachine(gTrain, G.BindDualValues(trainNet.Learnables()...))
	solver := G.NewAdamSolver(G.WithLearnRate(learningRate))

	// Create the experience replay buffer. The replay buffer stores
	// actions selected as one-hot vectors
	numFeatures := env.ObservationSpec().Shape.Len()
	replay, err := expreplay.New(
		config.Remover,
		config.Sampler,
		config.MinimumCapacity,
		config.MaximumCapacity,
		numFeatures,
		numActions,
	)
	if err != nil {
		msg := "new: could not create experience replay buffer: %v"
		return &DeepQ{}, fmt.Errorf(msg, err)
	}

	return &DeepQ{
		policy:                policy,
		trainNet:              trainNet,
		targetNet:             targetNet,
		selectedActions:       selectedActions,
		numActions:            numActions,
		replay:                replay,
		nextStateActionValues: nextStateActionValues,
		rewards:               rewards,
		discounts:             discounts,
		vm:                    vm,
		trainVM:               trainVM,
		targetVM:              targetVM,
		solver:                solver,
		prevStep:              ts.TimeStep{},
		prevAction:            0,
		nextStep:              ts.TimeStep{},
		learningRate:          learningRate,
		batchSize:             batchSize,
		tau:                   tau,
		targetUpdateInterval:  targetUpdateInterval,
		gradientSteps:         0,
	}, nil
}

// ObserveFirst observes and records the first episodic timestep
func (d *DeepQ) ObserveFirst(t ts.TimeStep) {
	if !t.First() {
		fmt.Fprintf(os.Stderr, "Warning: ObserveFirst() should only be"+
			"called on the first timestep (current timestep = %d)", t.Number)
	}
	d.prevStep = ts.TimeStep{}
	d.nextStep = t
}

// Observe observes and records any timestep other than the first timestep
func (d *DeepQ) Observe(action mat.Vector, nextStep ts.TimeStep) {
	if action.Len() != 1 {
		fmt.Fprintf(os.Stderr, "Warning: value-based methods should not "+
			"have multi-dimensional actions (action dim = %d)", action.Len())
	}

	// Add to replay buffer
	// Do not add if the previous step was the first step. In this case,
	// the argument action is the action taken in the first state, and
	// d.prevAction and d.prevState are invalid since no action was
	// taken to get to the first state and no state existed before the
	// first state
	if !d.prevStep.First() {
		prevAction := mat.NewVecDense(d.numActions, nil)
		prevAction.SetVec(d.prevAction, 1.0)

		nextAction := mat.NewVecDense(d.numActions, nil)
		nextAction.SetVec(int(action.AtVec(0)), 1.0)

		transition := ts.NewTransition(d.prevStep, prevAction, d.nextStep, nextAction)
		d.replay.Add(transition)
	}

	// Update internal variables
	d.prevStep = d.nextStep
	d.nextStep = nextStep
	d.prevAction = int(action.AtVec(0))

}

// Step updates the weights of the Agent's Policies.
func (d *DeepQ) Step() {
	// Don't update if replay buffer is empty
	if d.replay.Capacity() == 0 {
		fmt.Fprintln(os.Stderr, "step: skipping update, replay buffer empty")
		return
	}

	S, A, R, discount, NextS, _, err := d.replay.Sample()
	if err != nil {
		msg := fmt.Sprintf("step: could not sample from replay buffer: %v",
			err)
		panic(msg)
	}

	// Previous action one-hot vectors
	prevActions := tensor.New(
		tensor.WithShape(d.batchSize, d.numActions),
		tensor.WithBacking(A),
	)
	G.Let(d.selectedActions, prevActions)

	// Predict the action values in state S
	err = d.trainNet.SetInput(S)
	if err != nil {
		msg := fmt.Sprintf("step: could not set trainNet input: %v", err)
		panic(msg)
	}

	// Predict the action values in the next state NextS
	err = d.targetNet.SetInput(NextS)
	if err != nil {
		msg := fmt.Sprintf("step: could not set target net input: %v", err)
		panic(msg)
	}

	// Compute the next state-action values
	d.targetVM.RunAll()

	// Set the action values for the actions in the next state
	err = G.Let(d.nextStateActionValues, d.targetNet.Output())
	if err != nil {
		panic(fmt.Sprintf("step: could not set next state-action values: %v",
			err))
	}

	// Set the reward for the current action
	// reward := d.nextStep.Reward
	rewardTensor := tensor.New(tensor.WithBacking(R),
		tensor.WithShape(d.batchSize))
	err = G.Let(d.rewards, rewardTensor)
	if err != nil {
		panic(fmt.Sprintf("step: could not set reward: %v", err))
	}

	// Set the discount for the next action value
	// discount := d.nextStep.Discount
	discountTensor := tensor.New(tensor.WithBacking(discount),
		tensor.WithShape(d.batchSize))
	err = G.Let(d.discounts, discountTensor)
	if err != nil {
		panic(fmt.Sprintf("step: could not set discount: %v", err))
	}

	d.targetVM.Reset()

	// Run the learning step
	d.trainVM.RunAll()
	d.solver.Step(d.trainNet.Model())
	d.trainVM.Reset()
	d.gradientSteps++

	// Update the target network by setting its weights to the newly learned
	// weights
	if d.gradientSteps%d.targetUpdateInterval == 0 {
		if d.tau == 1.0 {
			d.targetNet.Set(d.trainNet)
		} else {
			d.targetNet.Polyak(d.trainNet, d.tau)
		}
	}
	d.policy.Set(d.trainNet)
}

// SelectAction runs the necessary VMs and then returns an action
// selected by the behaviour policy.
func (d *DeepQ) SelectAction(t ts.TimeStep) *mat.VecDense {
	obs := t.Observation.RawVector().Data
	err := d.policy.SetInput(obs)
	if err != nil {
		log.Fatal(err)
	}

	// Run the policy's computational graph
	d.vm.RunAll()

	// Get the policy to select an action using the data generated by
	// running the computational graph
	action, _ := d.policy.SelectAction()

	d.vm.Reset()
	return action
}

// TdError calculates the TD error generated by the learner on some
// transition.
func (d *DeepQ) TdError(t ts.Transition) float64 {
	state := t.State
	d.policy.SetInput(state.RawVector().Data)
	d.vm.RunAll()
	_, actionValue := d.policy.SelectAction()
	d.vm.Reset()

	nextState := t.NextState
	d.policy.SetInput(nextState.RawVector().Data)
	d.vm.RunAll()
	_, nextActionValue := d.targetNet.SelectAction()
	d.vm.Reset()

	return t.Reward + t.Discount*nextActionValue - actionValue
}
