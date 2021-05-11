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
)

// QLearning implements the Q-Learning algorithm
type QLearning struct {
	agent.Learner
	agent.Policy
	target agent.Policy
	seed   uint64
}

// New creates a new QLearning struct. The agent spec agent should be
// a spec.QLearning
func New(env environment.Environment, agent spec.Agent,
	seed uint64) *QLearning {

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

	// Get the environment specifrications
	envSpec := env.ObservationSpec()
	features := envSpec.Shape.Len()
	actions := env.ActionSpec().Shape.Len()

	// Create algorithm components using previous specifications
	behaviour := policy.NewEGreedy(e, seed, features, actions)
	target := policy.NewGreedy(seed, features, actions)
	weights := behaviour.Weights()["weights"]
	learner := NewQLearner(weights, learningRate)

	return &QLearning{learner, behaviour, target, seed}
}
