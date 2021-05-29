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

// ESarsa implements the online Expected Sarsa algorithm. Actions selected by
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
// spec.ESarsa or spec.QLearning. If the agent spec is neither
// spec.ESarsa or spec.QLearing, New will panic.
func New(env environment.Environment, agent spec.Agent,
	init weights.Initializer, seed uint64) (*ESarsa, error) {
	// Ensure environment has discrete actions
	if env.ActionSpec().Cardinality != spec.Discrete {
		return &ESarsa{}, fmt.Errorf("esarsa: cannot use non-discrete actions")
	}
	if env.ActionSpec().LowerBound.Len() > 1 {
		return &ESarsa{}, fmt.Errorf("esarsa: actions must be 1-dimensional")
	}
	if env.ActionSpec().LowerBound.AtVec(0) != 0.0 {
		return &ESarsa{}, fmt.Errorf("esarsa: actions must be enumerated " +
			"starting from 0")
	}

	// Ensure we have either an ESarsa or QLearning spec
	_, okSarsa := agent.(spec.ESarsa)
	_, okQLearning := agent.(spec.QLearning)
	if !okSarsa && !okQLearning {
		err := fmt.Errorf("%T not spec.ESarsa or spec.QLearning", agent)
		return &ESarsa{}, err
	}

	// Get the behaviour policy
	agentSpec := agent.Spec()
	behaviourE, ok := agentSpec["behaviour epsilon"]
	if !ok {
		err := fmt.Errorf("esarsa: no behaviour epsilon specified")
		return &ESarsa{}, err
	}
	behaviour, err := policy.NewEGreedy(behaviourE, seed, env)
	if err != nil {
		return &ESarsa{}, fmt.Errorf("esarsa: invalid behaviour policy: %v",
			err)
	}

	// Get the target policy
	targetE, ok := agentSpec["target epsilon"]
	if !ok {
		panic("no target epsilon specified")
	}
	if !ok {
		err := fmt.Errorf("esarsa: no target epsilon specified")
		return &ESarsa{}, err
	}
	target, err := policy.NewEGreedy(targetE, seed, env)
	if err != nil {
		return &ESarsa{}, fmt.Errorf("esarsa: invalid target policy: %v", err)
	}

	// Get the learning rate
	learningRate, ok := agentSpec["learning rate"]
	if !ok {
		err := fmt.Errorf("esarsa: no learning rate specified")
		return &ESarsa{}, err
	}

	// Ensure both policies and learner reference the same weights
	weights := behaviour.Weights()
	target.SetWeights(weights)

	learner, err := NewESarsaLearner(weights, learningRate, targetE)
	if err != nil {
		err := fmt.Errorf("esarsa: cannot create learner")
		return &ESarsa{}, err
	}

	// Initialize weights
	init.Initialize(weights["weights"])

	return &ESarsa{learner, behaviour, target, seed}, nil
}
