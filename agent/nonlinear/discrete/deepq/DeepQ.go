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
	}
	return New(env, deepQConfig, seed)
}

// DeepQ implements the deep Q-learning algorithm. This algorithm is
// conceptually similar to DQN, but uses the MSE loss.
type DeepQ struct {
	policy *policy.MultiHeadEGreedyMLP // behaviour policy
	// trainNet  *policy.MultiHeadEGreedyMLp
	targetNet *policy.MultiHeadEGreedyMLP // policy providing the update target

	selectedAction   *G.Node // Actions selected at the previous states in the batch
	batchPrevActions *tensor.Dense
	numActions       int

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
	vm G.VM
	// trainVM  G.VM
	targetVM G.VM
	solver   G.Solver

	prevStep   ts.TimeStep
	prevAction int
	nextStep   ts.TimeStep

	learningRate float64
	batchSize    int
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

	// Policy construction
	batchSize := 1
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
		batchSize,
		g,
		hiddenSizes,
		biases,
		init,
		activations,
		seed,
	)

	// Clone the policy network to create a target net which gives the
	// update target
	targetNet, err := policy.Clone()
	targetNet.SetEpsilon(0.0) // Qlearning target net is greedy
	if err != nil {
		return &DeepQ{}, fmt.Errorf("new: could not create target network")
	}
	gTarget := targetNet.Graph()

	// Create nodes to compute the update target: r + γ * max[Q(s', a')]
	nextStateActionValues := G.NewMatrix(g, tensor.Float64,
		G.WithShape(batchSize, numActions), G.WithName("targetActionVals"))
	rewards := G.NewVector(g, tensor.Float64, G.WithShape(batchSize),
		G.WithName("reward"))
	discounts := G.NewVector(g, tensor.Float64, G.WithShape(batchSize),
		G.WithName("discount"))

	// Compute the update target
	updateTarget := G.Must(G.Max(nextStateActionValues, 1))
	updateTarget = G.Must(G.HadamardProd(updateTarget, discounts))
	updateTarget = G.Must(G.Add(updateTarget, rewards))

	// Action selected by the policy in the previous state. This is
	// needed to compute the loss using the correct action value since
	// the network outputs N action values, one for each environmental
	// action
	selectedAction := G.NewMatrix(
		g,
		tensor.Float64,
		G.WithName("actionSelected"),
		G.WithShape(batchSize, numActions),
	)
	selectedActionValue := G.Must(G.HadamardProd(policy.Prediction(), // ! This will change when ER is added
		selectedAction))
	selectedActionValue = G.Must(G.Sum(selectedActionValue)) // ! When ER is added: along = 1

	// prevAction is the tensor used to set the value of selectedAction
	// We store the tensor and backing slice for computational
	// efficiency so that we don't have to create a new tensor for the
	// actions selected in the batch each time the computational graph is run
	// selectedActionSlice := make([]float64, batchSize*numActions)
	prevAction := tensor.New(
		tensor.Of(tensor.Float64),
		tensor.WithShape(selectedAction.Shape()...),
	)

	// Compute the Mean Squarred TD error
	losses := G.Must(G.Sub(updateTarget, selectedActionValue))
	losses = G.Must(G.Square(losses))
	cost := G.Must(G.Mean(losses))

	// Compute the gradient with respect to the Mean Squarred TD error
	_, err = G.Grad(cost, policy.Learnables()...)
	if err != nil {
		msg := fmt.Sprintf("new: could not compute gradient: %v", err)
		panic(msg)
	}

	// Create the VMs and solver for running the policy
	vm := G.NewTapeMachine(g, G.BindDualValues(policy.Learnables()...))
	targetVM := G.NewTapeMachine(gTarget)
	solver := G.NewAdamSolver(G.WithLearnRate(learningRate))

	return &DeepQ{policy, targetNet, selectedAction, prevAction, numActions, nextStateActionValues, rewards, discounts, vm, targetVM, solver,
		ts.TimeStep{}, 0, ts.TimeStep{}, learningRate, batchSize}, nil
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
	d.prevStep = d.nextStep
	d.prevAction = int(action.AtVec(0))
	d.nextStep = nextStep
}

// Step updates the weights of the Agent's Policies.
func (d *DeepQ) Step() {
	// Manually adjust backing slice so that the input tensor does not
	// need to be re-initialized at each timestep
	selectedActionSlice := make([]float64, d.numActions*d.batchSize)
	selectedActionSlice[d.prevAction] = 1.0 // ! needs to be adjusted for batches
	copy(d.batchPrevActions.Data().([]float64), selectedActionSlice)
	G.Let(d.selectedAction, d.batchPrevActions)

	prevObs := d.prevStep.Observation.RawVector().Data
	err := d.policy.SetInput(prevObs)
	if err != nil {
		msg := fmt.Sprintf("step: could not set policy input: %v", err)
		panic(msg)
	}

	nextObs := d.nextStep.Observation.RawVector().Data
	err = d.targetNet.SetInput(nextObs)
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
	reward := d.nextStep.Reward
	rewardTensor := tensor.New(tensor.WithBacking([]float64{reward}),
		tensor.WithShape(d.batchSize))
	err = G.Let(d.rewards, rewardTensor)
	if err != nil {
		panic(fmt.Sprintf("step: could not set reward: %v", err))
	}

	// Set the discount for the next action value
	discount := d.nextStep.Discount
	discountTensor := tensor.New(tensor.WithBacking([]float64{discount}),
		tensor.WithShape(d.batchSize))
	err = G.Let(d.discounts, discountTensor)
	if err != nil {
		panic(fmt.Sprintf("step: could not set discount: %v", err))
	}

	d.targetVM.Reset()

	// Run the learning step
	d.vm.RunAll()
	d.solver.Step(d.policy.Model())
	d.vm.Reset()

	// Update the target network by setting its weights to the newly learned
	// weights
	d.targetNet.Set(d.policy)
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

	nextState := t.NextState
	d.targetNet.SetInput(nextState.RawVector().Data)

	d.targetVM.RunAll()
	_, nextActionValue := d.targetNet.SelectAction()
	d.targetVM.Reset()

	d.vm.RunAll()
	_, actionValue := d.policy.SelectAction()
	d.vm.Reset()

	return t.Reward + t.Discount*nextActionValue - actionValue
}
