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

// https://hal.inria.fr/hal-00764281/PDF/DegrisACC2012.pdf
type LinearGaussian struct {
	agent.Policy
	agent.Learner
	seed uint64
}

func NewLinearGaussian(env environment.Environment, agent spec.Agent,
	init weights.Initializer, seed uint64) (*LinearGaussian, error) {
	// Ensure continuous action environment is used
	actionSpec := env.ActionSpec()
	if actionSpec.Cardinality != spec.Continuous {
		return nil, fmt.Errorf("actions must be continuous")
	}

	p := policy.NewGaussian(seed, env)
	weights := p.Weights()

	agent = agent.(spec.LinearGaussianActorCritic)
	agentSpec := agent.Spec()
	actorLearningRate, ok := agentSpec[spec.ActorLearningRate]
	if !ok {
		err := fmt.Errorf("new: no actor learning rate specified")
		return &LinearGaussian{}, err
	}

	criticLearningRate, ok := agentSpec[spec.CriticLearningRate]
	if !ok {
		err := fmt.Errorf("new: no critic learning rate specified")
		return &LinearGaussian{}, err
	}

	decay, ok := agentSpec[spec.Decay]
	if !ok {
		err := fmt.Errorf("new: no decay rate specified")
		return &LinearGaussian{}, err
	}

	l, err := NewGaussianLearner(p, actorLearningRate,
		criticLearningRate, decay)
	if err != nil {
		err := fmt.Errorf("new: could not create learner: %v", err)
		return &LinearGaussian{}, err
	}

	init.Initialize(weights[policy.MeanWeightsKey])
	init.Initialize(weights[policy.StdWeightsKey])
	init.Initialize(weights[policy.CriticWeightsKey])

	return &LinearGaussian{p, l, seed}, nil
}
