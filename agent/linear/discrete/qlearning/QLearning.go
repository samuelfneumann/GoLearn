// Package qlearning implements the Q-Learning algorithm.
//
// The Q-Learning algorithm is a special case of the Expected Sarsa
// algorithm. This package implements the same functionality as the
// esarsa package, but with some minor performance improvements due to
// the nature of the Q-Learning target policy being known before-hand.
package qlearning

import (
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

// New creates a new QLearning struct. The agent spec agent should be
// a spec.QLearning
func New(env environment.Environment, agent spec.Agent,
	init weights.Initializer, seed uint64) *QLearning {
	// Ensure environment has discrete actions
	if env.ActionSpec().Cardinality != spec.Discrete {
		panic("Q-learning can only be used with discrete actions")
	}
	if env.ActionSpec().Shape.Len() > 1 {
		panic("Q-Learning cannot be used with multi-dimensional actions")
	}
	if env.ActionSpec().LowerBound.AtVec(0) != 0.0 {
		panic("Actions must be enumerated starting from 0 for" +
			"value-based methods")
	}

	agent = agent.(spec.QLearning) // Ensure we have a QLearning spec

	// Get the agent specifications
	agentSpec := agent.Spec()
	e, ok := agentSpec["behaviour epsilon"]
	if !ok {
		panic("no epsilon specified")
	}

	learningRate, ok := agentSpec["learning rate"]
	if !ok {
		panic("no learning rate specified")
	}

	// Get the environment specifications
	envSpec := env.ObservationSpec()
	features := envSpec.Shape.Len()

	// Calculate the number of actions == Upper Bound + 1
	// e.g. if there are 4 actions (0, 1, 2, 3), then the upper bound is 3
	actions := int(env.ActionSpec().UpperBound.AtVec(0) + 1.0)

	// Create algorithm components using previous specifications
	behaviour := policy.NewEGreedy(e, seed, features, actions)
	target := policy.NewGreedy(seed, features, actions)

	// Ensure both policies and learner reference the same weights
	weights := behaviour.Weights()
	target.SetWeights(weights)

	learner := NewQLearner(weights, learningRate)

	// Initialize weights
	init.Initialize(weights["weights"])

	return &QLearning{learner, behaviour, target, seed}
}
