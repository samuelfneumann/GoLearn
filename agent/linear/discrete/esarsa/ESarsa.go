// Package esarsa implements the Expected Sarsa algorithm
package esarsa

import (
	"fmt"

	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/agent/linear/discrete/policy"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/utils/matutils/initializers/weights"
)

// Config represents a configuration for the ESarsa agent
type Config struct {
	BehaviourE   float64 // epislon for behaviour policy
	TargetE      float64 // epsilon for target policy
	LearningRate float64
}

// ESarsa implements the online Expected Sarsa algorithm. Actions selected by
// this algorithm will always be enumerated as (0, 1, 2, ... N) where
// N is the maximum possible action.
type ESarsa struct {
	agent.Learner
	agent.Policy // Behaviour
	Target       agent.Policy
	seed         uint64
}

// New creates a new ESarsa struct. The agent spec agent should be a
// spec.ESarsa or spec.QLearning. If the agent spec is neither
// spec.ESarsa or spec.QLearing, New will panic.
func New(env environment.Environment, config Config,
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

	// Get the behaviour policy
	behaviourE := config.BehaviourE
	behaviour, err := policy.NewEGreedy(behaviourE, seed, env)
	if err != nil {
		return &ESarsa{}, fmt.Errorf("esarsa: invalid behaviour policy: %v",
			err)
	}

	// Get the target policy
	targetE := config.TargetE
	target, err := policy.NewEGreedy(targetE, seed, env)
	if err != nil {
		return &ESarsa{}, fmt.Errorf("esarsa: invalid target policy: %v", err)
	}

	// Get the learning rate
	learningRate := config.LearningRate

	// Ensure both policies and learner reference the same weights
	weights := behaviour.Weights()
	target.SetWeights(weights)

	learner, err := NewESarsaLearner(behaviour, learningRate, targetE)
	if err != nil {
		err := fmt.Errorf("esarsa: cannot create learner")
		return &ESarsa{}, err
	}

	// Initialize weights
	for weight := range weights {
		init.Initialize(weights[weight])
	}

	return &ESarsa{learner, behaviour, target, seed}, nil
}
