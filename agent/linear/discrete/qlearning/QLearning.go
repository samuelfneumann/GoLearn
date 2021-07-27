// Package qlearning implements the Q-Learning algorithm.
//
// The Q-Learning algorithm is a special case of the Expected Sarsa
// algorithm. This package implements the same functionality as the
// esarsa package, but with some minor performance improvements due to
// the nature of the Q-Learning target policy being known before-hand.
package qlearning

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/agent/linear/discrete/policy"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/matutils/initializers/weights"
)

// QLearning implements the online Q-Learning algorithm. Actions selected by
// this algorithm will always be enumerated as (0, 1, 2, ... N) where
// N is the maximum possible action.
type QLearning struct {
	agent.Learner
	agent.Policy // Behaviour
	Target       agent.Policy
	seed         uint64
	eval         bool // Whether or not in evaluation mode
}

// New creates a new QLearning struct. The agent spec agent should be
// a spec.QLearning. If the agent spec is not a spec.QLearning, New
// will panic.
func New(env environment.Environment, config agent.Config,
	init weights.Initializer, seed uint64) (agent.Agent, error) {
	// Ensure environment has discrete actions
	if env.ActionSpec().Cardinality != spec.Discrete {
		return nil, fmt.Errorf("qlearning: cannot use non-discrete " +
			"actions")
	}
	if env.ActionSpec().LowerBound.Len() > 1 {
		return nil, fmt.Errorf("qlearning: actions must be " +
			"1-dimensional")
	}
	if env.ActionSpec().LowerBound.AtVec(0) != 0.0 {
		return nil, fmt.Errorf("qlearning: actions must be " +
			"enumerated starting from 0")
	}
	if !config.ValidAgent(&QLearning{}) {
		return nil, fmt.Errorf("qlearning: invalid agent for configuration "+
			"type %T", config)
	}

	c := config.(Config)
	if !c.ValidAgent(&QLearning{}) {
		return nil, fmt.Errorf("qlearning: invalid config for agent QLearning")
	}
	err := c.Validate()
	if err != nil {
		return nil, fmt.Errorf("qlearning: %v", err)
	}

	// Get the behaviour policy
	e := c.Epsilon
	behaviour, err := policy.NewEGreedy(e, seed, env)
	if err != nil {
		return &QLearning{}, fmt.Errorf("qlearning: invalid behaviour "+
			"policy: %v", err)
	}

	// Get the target policy
	target, err := policy.NewGreedy(seed, env)
	if err != nil {
		return &QLearning{}, fmt.Errorf("qlearning: invalid target "+
			"policy: %v", err)
	}

	// Get the learning rate
	learningRate := c.LearningRate

	// Ensure both policies and learner reference the same weights
	weights := behaviour.Weights()
	target.SetWeights(weights)

	learner, err := NewQLearner(behaviour, learningRate)
	if err != nil {
		err := fmt.Errorf("qlearning: cannot create learner")
		return &QLearning{}, err
	}

	// Initialize weights
	for weight := range weights {
		init.Initialize(weights[weight])
	}

	return &QLearning{learner, behaviour, target, seed, false}, nil
}

// SelectAction selects an action from either the agent's behaviour or
// target policy. The policy depends on whether or not the agent is in
// evaluation mode or training mode.
func (q *QLearning) SelectAction(t timestep.TimeStep) *mat.VecDense {
	if !q.eval {
		return q.Policy.SelectAction(t)
	}
	return q.Target.SelectAction(t)
}

// Eval sets the agent into evaluation mode
func (q *QLearning) Eval() {
	q.eval = true
}

// Train sets the agent into training mode
func (q *QLearning) Train() {
	q.eval = false
}
