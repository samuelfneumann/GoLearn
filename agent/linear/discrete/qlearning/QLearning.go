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
	"sfneuman.com/golearn/utils/matutils/initializers/weights"
)

// QLearning implements the Q-Learning algorithm. Actions selected by
// this algorithm will always be enumerated as (0, 1, 2, ... N) where
// N is the maximum possible action.
type QLearning struct {
	agent.Learner
	agent.Policy // Behaviour
	Target       agent.Policy
	seed         uint64
}

// SetWeights sets the weights of the QLearning Learner and Policies
func (q *QLearning) SetWeights(weights map[string]*mat.Dense) error {
	// Learner and Policies share weights, so it is sufficient to call
	// SetWeights() on only one of these fields
	return q.Learner.SetWeights(weights)
}

// Weights gets the weights of the QLearning Learner and Policies
func (q *QLearning) Weights() map[string]*mat.Dense {
	// Learner and Policies share weights, so it is sufficient to call
	// Weights() on only one of these fields
	return q.Learner.Weights()
}

// New creates a new QLearning struct. The agent spec agent should be
// a spec.QLearning. If the agent spec is not a spec.QLearning, New
// will panic.
func New(env environment.Environment, agent spec.Agent,
	init weights.Initializer, seed uint64) (*QLearning, error) {
	// Ensure environment has discrete actions
	if env.ActionSpec().Cardinality != spec.Discrete {
		return &QLearning{}, fmt.Errorf("qlearning: cannot use non-discrete " +
			"actions")
	}
	if env.ActionSpec().LowerBound.Len() > 1 {
		return &QLearning{}, fmt.Errorf("qlearning: actions must be " +
			"1-dimensional")
	}
	if env.ActionSpec().LowerBound.AtVec(0) != 0.0 {
		return &QLearning{}, fmt.Errorf("qlearning: actions must be " +
			"enumerated starting from 0")
	}

	agent = agent.(spec.QLearning) // Ensure we have a QLearning spec

	// Get the behaviour policy
	agentSpec := agent.Spec()
	e, ok := agentSpec["behaviour epsilon"]
	if !ok {
		err := fmt.Errorf("qlearning: no behaviour epsilon specified")
		return &QLearning{}, err
	}
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
	learningRate, ok := agentSpec["learning rate"]
	if !ok {
		err := fmt.Errorf("qlearning: no learning rate specified")
		return &QLearning{}, err
	}

	// Ensure both policies and learner reference the same weights
	weights := behaviour.Weights()
	target.SetWeights(weights)

	learner := NewQLearner(weights, learningRate)

	// Initialize weights
	init.Initialize(weights["weights"])

	return &QLearning{learner, behaviour, target, seed}, nil
}
