// Package actorcritic implements linear Actor-Critic algorithms
package actorcritic

import (
	"fmt"

	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/agent/linear/continuous/policy"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/utils/matutils/initializers/weights"
)

// LinearGaussian implements the Linear Gaussian Actor Critic algorithm:
//
// https://hal.inria.fr/hal-00764281/PDF/DegrisACC2012.pdf
//
// This algorithm is an actor critic algorithm which uses linear
// function approximation and eligibility traces. The critic learns the
// state value function to approximate the actor gradient.
type LinearGaussian struct {
	agent.Policy
	agent.Learner
	seed uint64
}

// NewLinearGaussian returns a new LinearGaussian agent. The weights for
// the linear function approximators (actor and critic) are initialized
// using the init Initializer argument. The elgibility traces for the
// algorithm are always initialized to 0.
func NewLinearGaussian(env environment.Environment, agent spec.Agent,
	init weights.Initializer, seed uint64) (*LinearGaussian, error) {
	// Ensure continuous action environment is used
	actionSpec := env.ActionSpec()
	if actionSpec.Cardinality != spec.Continuous {
		return nil, fmt.Errorf("actions must be continuous")
	}

	// Set up the policy
	p := policy.NewGaussian(seed, env)
	weights := p.Weights()

	// Ensure a compatible configuration was given
	agent = agent.(spec.LinearGaussianActorCritic)
	agentSpec := agent.Spec()

	// Get the actor learning rate
	actorLearningRate, ok := agentSpec[spec.ActorLearningRate]
	if !ok {
		err := fmt.Errorf("new: no actor learning rate specified")
		return &LinearGaussian{}, err
	}

	// Get the critic learning rate
	criticLearningRate, ok := agentSpec[spec.CriticLearningRate]
	if !ok {
		err := fmt.Errorf("new: no critic learning rate specified")
		return &LinearGaussian{}, err
	}

	// Get the eligibility trace decay rate
	decay, ok := agentSpec[spec.Decay]
	if !ok {
		err := fmt.Errorf("new: no decay rate specified")
		return &LinearGaussian{}, err
	}

	// Create the Gaussian learner ot learn the Gaussian policy
	l, err := NewGaussianLearner(p, actorLearningRate,
		criticLearningRate, decay)
	if err != nil {
		err := fmt.Errorf("new: could not create learner: %v", err)
		return &LinearGaussian{}, err
	}

	// Initialize learner weights
	init.Initialize(weights[policy.MeanWeightsKey])
	init.Initialize(weights[policy.StdWeightsKey])
	init.Initialize(weights[policy.CriticWeightsKey])

	return &LinearGaussian{p, l, seed}, nil
}
