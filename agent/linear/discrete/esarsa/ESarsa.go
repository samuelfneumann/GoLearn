// Package esarsa implements the Expected Sarsa algorithm
package esarsa

import (
	"fmt"

	"github.com/samuelfneumann/golearn/agent"
	"github.com/samuelfneumann/golearn/agent/linear/discrete/policy"
	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/environment/wrappers"
	"github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/golearn/utils/matutils/initializers/weights"
	"gonum.org/v1/gonum/mat"
)

// ESarsa implements the online Expected Sarsa algorithm. Actions selected by
// this algorithm will always be enumerated as (0, 1, 2, ... N) where
// N is the maximum possible action.
type ESarsa struct {
	agent.Learner
	agent.Policy // Behaviour
	Target       agent.Policy
	seed         uint64
	eval         bool // Whether or not in evaluation mode

	// indexTileCoding represents whether the environment is using
	// tile coding and returning the non-zero indices as features
	indexTileCoding bool
}

// New creates a new ESarsa struct. The agent environment agent should be a
// environment.ESarsa or environment.QLearning. If the agent environment is neither
// environment.ESarsa or environment.QLearing, New will panic.
func New(env environment.Environment, c agent.Config,
	init weights.Initializer, seed uint64) (agent.Agent, error) {
	// Ensure environment has discrete actions
	if env.ActionSpec().Cardinality != environment.Discrete {
		return nil, fmt.Errorf("esarsa: cannot use non-discrete actions")
	}
	if env.ActionSpec().LowerBound.Len() > 1 {
		return nil, fmt.Errorf("esarsa: actions must be 1-dimensional")
	}
	if env.ActionSpec().LowerBound.AtVec(0) != 0.0 {
		return nil, fmt.Errorf("esarsa: actions must be enumerated " +
			"starting from 0")
	}
	if !c.ValidAgent(&ESarsa{}) {
		return nil, fmt.Errorf("esarsa: invalid agent for configuration "+
			"type %T", c)
	}
	config, ok := c.(Config)
	if !ok {
		return nil, fmt.Errorf("esarsa: invalid config for agent ESarsa")
	}
	err := config.Validate()
	if err != nil {
		return nil, fmt.Errorf("esarsa: %v", err)
	}

	// Get the behaviour policy
	behaviourE := config.BehaviourE
	behaviourPol, err := policy.NewEGreedy(behaviourE, seed, env)
	if err != nil {
		return &ESarsa{}, fmt.Errorf("esarsa: invalid behaviour policy: %v",
			err)
	}
	behaviour := behaviourPol.(*policy.EGreedy)

	// Get the target policy
	targetE := config.TargetE
	targetPol, err := policy.NewEGreedy(targetE, seed, env)
	if err != nil {
		return &ESarsa{}, fmt.Errorf("esarsa: invalid target policy: %v", err)
	}
	target := targetPol.(*policy.EGreedy)

	// Get the learning rate
	learningRate := config.LearningRate

	// Ensure both policies and learner reference the same weights
	weights := behaviour.Weights()
	target.SetWeights(weights)

	// Check if the environment uses tile coding and returns the
	// indices of non-zero elements of the tile-coded vectors as
	// state representations
	_, indexTileCoding := env.(*wrappers.IndexTileCoding)

	learner, err := NewESarsaLearner(behaviour, target, learningRate, targetE,
		indexTileCoding)
	if err != nil {
		err := fmt.Errorf("esarsa: cannot create learner")
		return &ESarsa{}, err
	}

	// Initialize weights
	for weight := range weights {
		init.Initialize(weights[weight])
	}

	return &ESarsa{
		Learner:         learner,
		Policy:          behaviour,
		Target:          target,
		seed:            seed,
		eval:            false,
		indexTileCoding: indexTileCoding,
	}, nil
}

// SelectAction selects an action from either the agent's behaviour or
// target policy. The policy depends on whether or not the agent is in
// evaluation mode or training mode.
func (e *ESarsa) SelectAction(t timestep.TimeStep) *mat.VecDense {
	if !e.eval {
		return e.Policy.SelectAction(t)
	}
	return e.Target.SelectAction(t)
}

// Step wraps the stepping operations of the ESarsaLearner so that
// stepping is not permitted when in evaluation mode.
func (e *ESarsa) Step() {
	if e.IsEval() {
		return
	}
	e.Learner.Step()
}
