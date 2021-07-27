// Package actorcritic implements linear Actor-Critic algorithms
package actorcritic

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/agent/linear/continuous/policy"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/matutils/initializers/weights"
)

// LinearGaussian implements the Linear Gaussian Actor Critic algorithm:
//
// https://hal.inria.fr/hal-00764281/PDF/DegrisACC2012.pdf
//
// This algorithm is an actor critic algorithm which uses linear
// function approximation and eligibility traces. The critic learns the
// state value function to approximate the actor gradient.
//
// So far LinearGaussian works only with single-dimensional actions.
type LinearGaussian struct {
	*policy.Gaussian
	*GaussianLearner
	seed uint64
	eval bool // Whether or not the agent is in evaluation mode
}

// NewLinearGaussian returns a new LinearGaussian agent. The weights for
// the linear function approximators (actor and critic) are initialized
// using the init Initializer argument. The elgibility traces for the
// algorithm are always initialized to 0.
func NewLinearGaussian(env environment.Environment, c agent.Config,
	init weights.Initializer, seed uint64) (agent.Agent, error) {
	// Ensure continuous action environment is used
	actionSpec := env.ActionSpec()
	if actionSpec.Cardinality != spec.Continuous {
		return nil, fmt.Errorf("newLinearGaussian: actions must be continuous")
	}
	if actionSpec.Shape.Len() != 1 {
		return nil, fmt.Errorf("newLinearGaussian: LinearGaussian does not " +
			"yet support multi-dimensional actions")
	}
	if !c.ValidAgent(&LinearGaussian{}) {
		return nil, fmt.Errorf("newLinearGaussian: invalid agent for "+
			"configuration type %T", c)
	}
	config, ok := c.(Config)
	if !ok {
		return nil, fmt.Errorf("newLinearGaussian: invalid config for agent " +
			"LinearGaussian")
	}
	err := config.Validate()
	if err != nil {
		return nil, fmt.Errorf("newLinearGaussian: %v", err)
	}

	// Set up the policy
	p := policy.NewGaussian(seed, env)
	weights := p.Weights()

	// Get the actor learning rate
	actorLearningRate := config.ActorLearningRate

	// Get the critic learning rate
	criticLearningRate := config.CriticLearningRate

	// Get the eligibility trace decay rate
	decay := config.Decay

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

	return &LinearGaussian{p, l, seed, false}, nil
}

// SelectAction selects an action from either the agent's behaviour or
// target policy. The policy depends on whether or not the agent is in
// evaluation mode or training mode.
func (l *LinearGaussian) SelectAction(t timestep.TimeStep) *mat.VecDense {
	if !l.eval {
		return l.Gaussian.SelectAction(t)
	}
	return l.Gaussian.Mean(t.Observation)
}

// Eval sets the agent into evaluation mode
func (l *LinearGaussian) Eval() {
	l.eval = true
}

// Train sets the agent into training mode
func (l *LinearGaussian) Train() {
	l.eval = false
}
