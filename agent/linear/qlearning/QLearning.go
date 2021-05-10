// Package qlearning implements the Q-Learning algorithm
package qlearning

import (
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/agent/linear/policy"
)

// QLearning implements the Q-Learning algorithm
type QLearning struct {
	agent.Learner
	agent.Policy
	target agent.Policy
	seed   uint64
}

// New creates a new QLearning struct
func New(e, learningRate float64, seed uint64, features,
	actions int) *QLearning {

	behaviour := policy.NewEGreedy(e, seed, features, actions)
	target := behaviour.GreedyPolicy
	weights := behaviour.Weights()["weights"]
	learner := NewQLearner(weights, learningRate)

	return &QLearning{learner, behaviour, target, seed}
}
