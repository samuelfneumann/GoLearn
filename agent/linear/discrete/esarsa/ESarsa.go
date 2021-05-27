// Package esarsa implements the Expected Sarsa algorithm
package esarsa

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/agent/linear/discrete/policy"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/utils/matutils/initializers/weights"
)

// ESarsa implements the Expected Sarsa algorithm. Actions selected by
// this algorithm will always be enumerated as (0, 1, 2, ... N) where
// N is the maximum possible action.
type ESarsa struct {
	agent.Learner
	agent.Policy // Behaviour
	Target       agent.Policy
	seed         uint64
}

// SetWeights sets the weights of the ESarsa Learner and Policies
func (e *ESarsa) SetWeights(weights map[string]*mat.Dense) error {
	// Learner and Policies share weights, so it is sufficient to call
	// SetWeights() on only one of these fields
	return e.Learner.SetWeights(weights)
}

// Weights gets the weights of the ESarsa Learner and Policies
func (e *ESarsa) Weights() map[string]*mat.Dense {
	// Learner and Policies share weights, so it is sufficient to call
	// Weights() on only one of these fields
	return e.Learner.Weights()
}

// New creates a new ESarsa struct. The agent spec agent should be a
// spec.ESarsa or spec.QLearning
func New(env environment.Environment, agent spec.Agent,
	init weights.Initializer, seed uint64) *ESarsa {
	// Ensure environment has discrete actions
	if env.ActionSpec().Cardinality != spec.Discrete {
		panic("ESarsa can only be used with discrete actions")
	}
	if env.ActionSpec().LowerBound.Len() > 1 {
		panic("ESarsa cannot be used with multi-dimensional actions")
	}
	if env.ActionSpec().LowerBound.AtVec(0) != 0.0 {
		panic("Actions must be enumerated starting from 0 for" +
			"value-based methods")
	}

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

	// Create algorithm components using previous specifications
	behaviour := policy.NewEGreedy(behaviourE, seed, env)
	target := policy.NewEGreedy(targetE, seed, env)

	// Ensure both policies and learner reference the same weights
	weights := behaviour.Weights()
	target.SetWeights(weights)

	learner := NewESarsaLearner(weights, learningRate, targetE)

	// Initialize weights
	init.Initialize(weights["weights"])

	return &ESarsa{learner, behaviour, target, seed}
}
