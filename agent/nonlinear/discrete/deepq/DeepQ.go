package deepq

import (
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/agent/linear/discrete/qlearning"
	"sfneuman.com/golearn/agent/nonlinear/discrete/policy"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/expreplay"
	"sfneuman.com/golearn/network"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
)

// DeepQ implements the deep Q-learning algorithm. This algorithm is
// conceptually similar to DQN, but uses the MSE loss.
type DeepQ struct {
	// Action selection policies
	behaviourPolicy   agent.EGreedyNNPolicy // Behaviour egreedy policy
	behaviourPolicyVM G.VM
	targetPolicy      agent.EGreedyNNPolicy // Target greedy policy
	targetPolicyVM    G.VM

	// Policy for learning weights that takes in batches of inputs
	trainNet   agent.EGreedyNNPolicy // Policy whose weights are adapted
	trainNetVM G.VM
	solver     G.Solver // Adapts the weights of trainNet

	// Policy that provides the update target for a batch of inputs
	// Note that this is a target network, providing the update target.
	// It is not the network for the target policy
	targetNet   agent.EGreedyNNPolicy
	targetNetVM G.VM

	// Variables to track target network updates
	tau                  float64 // Polyak averaging constant
	targetUpdateInterval int     // Steps between target updates
	gradientSteps        int

	selectedActions *G.Node // Actions taken at the previous states
	numActions      int

	replay expreplay.ExperienceReplayer

	// nextStateActionValues is the input node in the graph of trainNet that
	// is given the action values of the next state. For update:
	//
	// Q(s, a) <- Q(s, a) + α * (r + Q(s', a') - Q(s, a)) ∇Q(s, a)
	//
	// nextStateActionValues provides Q(s', a') for all a' in s' and is
	// computed by targetNet.
	nextStateActionValues *G.Node
	rewards               *G.Node
	discounts             *G.Node

	// Keep track of previous states and actions to add to replay buffer
	prevStep   ts.TimeStep
	prevAction int
	nextStep   ts.TimeStep

	batchSize int
	eval      bool // Whether or not in evaluation mode

}

// New creates and returns a new DeepQ agent
func New(env environment.Environment, config Config,
	seed int64) (*DeepQ, error) {
	// Ensure environment has discrete actions
	if env.ActionSpec().Cardinality != spec.Discrete {
		return &DeepQ{}, fmt.Errorf("deepq: cannot use non-discrete " +
			"actions")
	}

	// Ensure actions are one-dimensional
	if env.ActionSpec().LowerBound.Len() > 1 {
		return &DeepQ{}, fmt.Errorf("deepq: actions must be " +
			"1-dimensional")
	}

	// Ensure actions are enumerated from 0
	if env.ActionSpec().LowerBound.AtVec(0) != 0.0 {
		return &DeepQ{}, fmt.Errorf("deepq: actions must be " +
			"enumerated starting from 0")
	}

	// Ensure the configuration is valid
	err := config.Validate()
	if err != nil {
		return &DeepQ{}, err
	}

	// Extract configuration variables
	batchSize := config.BatchSize()
	numActions := int(env.ActionSpec().UpperBound.AtVec(0)) + 1
	hiddenSizes := config.PolicyLayers
	biases := config.Biases
	activations := config.Activations
	init := config.InitWFn
	ε := config.Epsilon

	// Behaviour network for selecting actions
	g := G.NewGraph()
	behaviourPolicy, err := policy.NewMultiHeadEGreedyMLP(
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
	if err != nil {
		return &DeepQ{}, err
	}
	behaviourPolicyVM := G.NewTapeMachine(g)

	// Create the target policy for selecting actions
	targetPolicyClone, err := behaviourPolicy.ClonePolicy()
	if err != nil {
		return &DeepQ{}, fmt.Errorf("new: could not create target policy")
	}
	targetPolicy := targetPolicyClone.(agent.EGreedyNNPolicy)
	targetPolicy.SetEpsilon(0.0)
	targetPolicyVM := G.NewTapeMachine(targetPolicy.Graph())

	// Create the target network which provides the update target
	targetNetClone, err := behaviourPolicy.ClonePolicyWithBatch(batchSize)
	if err != nil {
		msg := "new: could not create target network: %v"
		return &DeepQ{}, fmt.Errorf(msg, err)
	}
	targetNet := targetNetClone.(agent.EGreedyNNPolicy)
	targetNet.SetEpsilon(0.0) // Qlearning target policy is greedy
	targetNetVM := G.NewTapeMachine(targetNet.Graph())

	// Target network update schedule
	tau := config.Tau
	targetUpdateInterval := config.TargetUpdateInterval

	// Create a training network which learns the weights
	trainNetClone, err := behaviourPolicy.ClonePolicyWithBatch(batchSize)
	if err != nil {
		msg := "new: could not create learning network: %v"
		return &DeepQ{}, fmt.Errorf(msg, err)
	}
	trainNet := trainNetClone.(agent.EGreedyNNPolicy)
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

	// Action selected in the previous state. This is needed to compute
	// the loss using the correct action value since the network outputs N
	// action values, one for each environmental action
	selectedActions := G.NewMatrix(
		gTrain,
		tensor.Float64,
		G.WithName("actionSelected"),
		G.WithShape(batchSize, numActions),
	)
	selectedActionsValue := G.Must(G.HadamardProd(trainNet.Prediction(),
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

	// Compile the trainNet graph into a VM
	trainNetVM := G.NewTapeMachine(
		gTrain,
		G.BindDualValues(trainNet.Learnables()...),
	)
	solver := config.Solver

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
		behaviourPolicy:       behaviourPolicy,
		behaviourPolicyVM:     behaviourPolicyVM,
		targetPolicy:          targetPolicy,
		targetPolicyVM:        targetPolicyVM,
		trainNet:              trainNet,
		trainNetVM:            trainNetVM,
		solver:                solver,
		targetNet:             targetNet,
		targetNetVM:           targetNetVM,
		tau:                   tau,
		targetUpdateInterval:  targetUpdateInterval,
		gradientSteps:         0,
		selectedActions:       selectedActions,
		numActions:            numActions,
		replay:                replay,
		nextStateActionValues: nextStateActionValues,
		rewards:               rewards,
		discounts:             discounts,
		prevStep:              ts.TimeStep{},
		prevAction:            0,
		nextStep:              ts.TimeStep{},
		batchSize:             batchSize,
		eval:                  false,
	}, nil
}

// NewQlearning returns a DeepQ agent that uses Q-learning.
// That is, the algorithm uses linear function approximation to learn
// online with no target networks.
func NewQlearning(env environment.Environment, config qlearning.Config,
	seed int64, InitWFn G.InitWFn) (*DeepQ, error) {
	learningRate := config.LearningRate
	deepQConfig := Config{
		Epsilon:      config.Epsilon,
		PolicyLayers: []int{},
		Biases:       []bool{},
		Activations:  []network.Activation{},
		InitWFn:      InitWFn,
		Solver:       G.NewVanillaSolver(G.WithLearnRate(learningRate)),

		Tau:                  1.0,
		TargetUpdateInterval: 1,
	}
	return New(env, deepQConfig, seed)
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
	if !d.nextStep.First() {
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
	// Don't update if replay buffer is empty or has insufficient
	// samples to sample
	S, A, R, discount, NextS, _, err := d.replay.Sample()
	if expreplay.IsEmptyBuffer(err) || expreplay.IsInsufficientSamples(err) {
		return
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
	d.targetNetVM.RunAll()

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

	d.targetNetVM.Reset()

	// Run the learning step
	d.trainNetVM.RunAll()
	d.solver.Step(d.trainNet.Model())
	d.trainNetVM.Reset()
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

	d.targetPolicy.Set(d.trainNet)
	d.behaviourPolicy.Set(d.trainNet)
}

// SelectAction runs the necessary VMs and then returns an action
// selected by the behaviour policy.
func (d *DeepQ) SelectAction(t ts.TimeStep) *mat.VecDense {
	var policy agent.NNPolicy
	var vm G.VM

	// Select action from target or behaviour policy depending on if
	// in training or eval mode
	if d.eval {
		policy = d.targetPolicy
		vm = d.targetPolicyVM
	} else {
		policy = d.behaviourPolicy
		vm = d.behaviourPolicyVM

	}

	obs := t.Observation.RawVector().Data
	err := policy.SetInput(obs)
	if err != nil {
		panic(fmt.Sprintf("selectaction: %v", err))
	}

	// Run the policy's computational graph
	vm.RunAll()

	// Get the policy to select an action using the data generated by
	// running the computational graph
	action, _ := policy.SelectAction()

	vm.Reset()
	return action
}

// TdError calculates the TD error generated by the learner on some
// transition.
func (d *DeepQ) TdError(t ts.Transition) float64 {
	state := t.State
	d.behaviourPolicy.SetInput(state.RawVector().Data)
	d.behaviourPolicyVM.RunAll()
	_, actionValue := d.behaviourPolicy.SelectAction()
	d.behaviourPolicyVM.Reset()

	nextState := t.NextState
	d.targetPolicy.SetInput(nextState.RawVector().Data)
	d.targetPolicyVM.RunAll()
	_, nextActionValue := d.targetPolicy.SelectAction()
	d.targetPolicyVM.Reset()

	return t.Reward + t.Discount*nextActionValue - actionValue
}

// Eval sets the agent into evaluation mode
func (d *DeepQ) Eval() {
	d.eval = true
}

// Train sets the agent into training mode
func (d *DeepQ) Train() {
	d.eval = false
}

// Cleanup at the end of an episode
func (d *DeepQ) EndEpisode() {}
