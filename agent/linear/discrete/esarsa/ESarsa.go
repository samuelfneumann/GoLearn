// Package esarsa implements the Expected Sarsa algorithm
package esarsa

import (
	"fmt"

	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/agent/linear/discrete/policy"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
)

// ESarsa implements the Q-Learning algorithm
type ESarsa struct {
	agent.Learner
	agent.Policy
	target agent.Policy
	seed   uint64
}

// New creates a new ESarsa struct. The agent spec agent should be a
// spec.ESarsa or spec.QLearning
func New(env environment.Environment, agent spec.Agent,
	seed uint64) *ESarsa {

	// Ensure we have either an ESarsa or QLearning spec
	_, okSarsa := agent.(spec.ESarsa)
	_, okQLearning := agent.(spec.QLearning)
	if !okSarsa && !okQLearning {
		msg := fmt.Sprintf("%T not spec.ESarsa or spec.QLearning", agent)
		panic(msg)
	}

	// Get the agent specifications
	agentSpec := agent.Spec()
	behaviourE, ok := agentSpec["behaviour epsilon"]
	if !ok {
		panic("no behaviour epsilon specified")
	}

	targetE, ok := agentSpec["target epsilon"]
	if !ok {
		panic("no target epsilon specified")
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
	behaviour := policy.NewEGreedy(behaviourE, seed, features, actions)
	target := policy.NewEGreedy(targetE, seed, features, actions)
	weights := behaviour.Weights()["weights"]
	learner := NewESarsaLearner(weights, learningRate, targetE)

	return &ESarsa{learner, behaviour, target, seed}
}
