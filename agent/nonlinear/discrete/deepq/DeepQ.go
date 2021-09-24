package deepq

import (
	"fmt"

	"github.com/samuelfneumann/golearn/agent"
	"github.com/samuelfneumann/golearn/agent/linear/discrete/qlearning"
	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/expreplay"
	"github.com/samuelfneumann/golearn/initwfn"
	"github.com/samuelfneumann/golearn/network"
	"github.com/samuelfneumann/golearn/solver"
	ts "github.com/samuelfneumann/golearn/timestep"
	"gonum.org/v1/gonum/mat"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// DeepQ implements the deep Q-learning algorithm. This algorithm is
// conceptually similar to DQN, but uses the MSE loss.
type DeepQ struct {
	// Action selection policy. We only need a single policy for both
	// target and behaviour policy. DeepQ's target policy is greedy
	// with respect to action values, which we can get by setting
	// the policy to evaluation mode.
	policy agent.EGreedyNNPolicy

	// Policy for learning weights that takes in batches of inputs
	trainNet   network.NeuralNet // Policy whose weights are adapted
	trainNetVM G.VM
	solver     G.Solver // Adapts the weights of trainNet

	// Policy that provides the update target for a batch of inputs
	// Note that this is a target network, providing the update target.
	// It is not the network for the target policy
	targetNet   network.NeuralNet
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
	prevStep ts.TimeStep

	batchSize int
}

// New creates and returns a new DeepQ agent
func New(env environment.Environment, c agent.Config,
	seed int64) (agent.Agent, error) {
	if !c.ValidAgent(&DeepQ{}) {
		return nil, fmt.Errorf("new: invalid configuration type: %T", c)
	}

	// Ensure environment has discrete actions
	if env.ActionSpec().Cardinality != environment.Discrete {
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
	err := c.Validate()
	if err != nil {
		return &DeepQ{}, err
	}

	config := c.(Config)

	// Extract configuration variables
	batchSize := config.BatchSize()
	numActions := int(env.ActionSpec().UpperBound.AtVec(0)) + 1

	// Create the target network which provides the update target
	targetNet := config.targetNet
	if layers := targetNet.OutputLayers(); layers != 1 {
		msg := "new: target net should return a single target prediction " +
			"\n\twant(1)\n\thave(%v)"
		return &DeepQ{}, fmt.Errorf(msg, layers)
	}
	targetNetVM := G.NewTapeMachine(targetNet.Graph())

	// Target network update schedule
	tau := config.Tau
	targetUpdateInterval := config.TargetUpdateInterval

	// Create the training network, which is the network whose weights
	// are learned
	trainNet := config.trainNet
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

	var cost *G.Node
	for _, pred := range trainNet.Prediction() {
		selectedActionsValue := G.Must(G.HadamardProd(pred,
			selectedActions))
		selectedActionsValue = G.Must(G.Sum(selectedActionsValue, 1))

		// Compute the Mean Squarred TD error
		loss := G.Must(G.Sub(updateTarget, selectedActionsValue))
		loss = G.Must(G.Square(loss))
		loss = G.Must(G.Mean(loss))
		if cost == nil {
			cost = loss
		} else {
			cost = G.Must(G.Add(cost, loss))
		}
	}

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
	replay, err := config.ExpReplay.Create(numFeatures, numActions, seed,
		false)
	if err != nil {
		msg := "new: could not create experience replay buffer: %v"
		return &DeepQ{}, fmt.Errorf(msg, err)
	}

	return &DeepQ{
		policy:                config.policy,
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
		batchSize:             batchSize,
	}, nil
}

// NewQlearning returns a DeepQ agent that uses Q-learning.
// That is, the algorithm uses linear function approximation to learn
// online with no target networks.
func NewQlearning(env environment.Environment, config qlearning.Config,
	seed int64, InitWFn *initwfn.InitWFn) (agent.Agent, error) {
	learningRate := config.LearningRate
	sol, err := solver.NewVanilla(learningRate, 1, -1.0)
	if err != nil {
		return nil, fmt.Errorf("newQLearning: cannot create solver: %v", err)
	}
	deepQConfig := &Config{
		Epsilon:     config.Epsilon,
		Layers:      []int{},
		Biases:      []bool{},
		Activations: []*network.Activation{},
		InitWFn:     InitWFn,
		Solver:      sol,

		Tau:                  1.0,
		TargetUpdateInterval: 1,
	}
	return New(env, deepQConfig, seed)
}

// ObserveFirst observes and records the first episodic timestep
func (d *DeepQ) ObserveFirst(t ts.TimeStep) error {
	if !t.First() {
		return fmt.Errorf("observeFirst: timestep is not first "+
			"(current timestep = %d)", t.Number)
	}
	d.prevStep = t
	return nil
}

// Observe observes and records any timestep other than the first timestep
func (d *DeepQ) Observe(a mat.Vector, nextStep ts.TimeStep) error {
	if a.Len() != 1 {
		return fmt.Errorf("observe: cannot observe "+
			"multi-dimensional action (action dim = %d) for DeepQ", a.Len())
	}

	// Add to replay buffer
	if !nextStep.First() {
		action := mat.NewVecDense(d.numActions, nil)
		action.SetVec(int(a.AtVec(0)), 1.0)
		nextAction := mat.NewVecDense(d.numActions, nil)

		transition := ts.NewTransition(d.prevStep, action, nextStep, nextAction)
		err := d.replay.Add(transition)
		if err != nil {
			return fmt.Errorf("observe: could not add to replay buffer: %v",
				err)
		}
	}

	d.prevStep = nextStep
	return nil
}

// Step updates the weights of the Agent's Policies.
func (d *DeepQ) Step() error {
	if d.IsEval() {
		return nil
	}

	// Don't update if replay buffer is empty or has insufficient
	// samples to sample
	S, A, R, discount, NextS, _, err := d.replay.Sample()
	if expreplay.IsEmptyBuffer(err) || expreplay.IsInsufficientSamples(err) {
		return nil
	}

	// Predict the action values in the next state NextS
	err = d.targetNet.SetInput(NextS)
	if err != nil {
		return fmt.Errorf("step: could not set target net input: %v", err)
	}
	err = d.targetNetVM.RunAll()
	if err != nil {
		return fmt.Errorf("step: could not run target vm: %v", err)
	}

	// Set the action values for the actions in the next state
	err = G.Let(d.nextStateActionValues, d.targetNet.Output()[0])
	if err != nil {
		return fmt.Errorf("step: could not set next state-action values: %v",
			err)
	}

	d.targetNetVM.Reset()

	// Set the reward for the current action
	rewardTensor := tensor.New(tensor.WithBacking(R),
		tensor.WithShape(d.batchSize))
	err = G.Let(d.rewards, rewardTensor)
	if err != nil {
		return fmt.Errorf("step: could not set reward: %v", err)
	}

	// Set the discount for the next action value
	discountTensor := tensor.New(tensor.WithBacking(discount),
		tensor.WithShape(d.batchSize))
	err = G.Let(d.discounts, discountTensor)
	if err != nil {
		return fmt.Errorf("step: could not set discount: %v", err)
	}

	// Previous action one-hot vectors
	prevActions := tensor.New(
		tensor.WithShape(d.batchSize, d.numActions),
		tensor.WithBacking(A),
	)
	err = G.Let(d.selectedActions, prevActions)
	if err != nil {
		return fmt.Errorf("step: could not set previous actions: %v", err)
	}

	// Predict the action values in state S
	err = d.trainNet.SetInput(S)
	if err != nil {
		return fmt.Errorf("step: could not set trainNet input: %v", err)
	}

	// Run the learning step
	err = d.trainNetVM.RunAll()
	if err != nil {
		return fmt.Errorf("step: could not run train vm: %v", err)
	}

	err = d.solver.Step(d.trainNet.Model())
	if err != nil {
		return fmt.Errorf("step: could not step solver: %v", err)
	}

	d.trainNetVM.Reset()
	d.gradientSteps++

	// Update the target network by setting its weights to the newly learned
	// weights
	if d.gradientSteps%d.targetUpdateInterval == 0 {
		if d.tau == 1.0 {
			err = network.Set(d.targetNet, d.trainNet)
			if err != nil {
				return fmt.Errorf("step: could not update target network")
			}
		} else {
			err = network.Polyak(d.targetNet, d.trainNet, d.tau)
			if err != nil {
				return fmt.Errorf("step: could not update target network")
			}
		}
	}

	err = network.Set(d.policy.Network(), d.trainNet)
	if err != nil {
		return fmt.Errorf("step: could not update target network")
	}

	return nil
}

// SelectAction runs the necessary VMs and then returns an action
// selected by the behaviour policy.
func (d *DeepQ) SelectAction(t ts.TimeStep) *mat.VecDense {
	// Select action from target or behaviour policy depending on if
	// in training or eval mode
	return d.policy.SelectAction(t)
}

// TdError calculates the TD error generated by the learner on some
// transition.
func (d *DeepQ) TdError(t ts.Transition) float64 {
	step := ts.TimeStep{Observation: t.State}
	action := int(d.policy.SelectAction(step).AtVec(0))
	actionValues := d.policy.Network().Output()[0].Data()
	actionValue := actionValues.([]float64)[action]

	d.policy.Eval()
	step.Observation = t.NextState
	nextAction := int(d.policy.SelectAction(step).AtVec(0))
	nextActionValues := d.policy.Network().Output()[0].Data()
	nextActionValue := nextActionValues.([]float64)[nextAction]
	d.policy.Train()

	return t.Reward + t.Discount*nextActionValue - actionValue
}

// Eval sets the agent into evaluation mode
func (d *DeepQ) Eval() {
	d.policy.Eval()
}

// Train sets the agent into training mode
func (d *DeepQ) Train() {
	d.policy.Train()
}

// IsEval returns whether the agent is in evaluation mode
func (d *DeepQ) IsEval() bool {
	return d.policy.IsEval()
}

// Cleanup at the end of an episode
func (d *DeepQ) EndEpisode() {}
