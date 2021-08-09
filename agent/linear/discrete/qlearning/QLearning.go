// Package qlearning implements the Q-Learning algorithm.
//
// The Q-Learning algorithm is a environmential case of the Expected Sarsa
// algorithm. This package implements the same functionality as the
// esarsa package, but with some minor performance improvements due to
// the nature of the Q-Learning target policy being known before-hand.
package qlearning

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

// QLearning implements the online Q-Learning algorithm. Actions selected by
// this algorithm will always be enumerated as (0, 1, 2, ... N) where
// N is the maximum possible action.
type QLearning struct {
	agent.Learner
	agent.Policy // Behaviour
	Target       agent.Policy
	seed         uint64
	eval         bool // Whether or not in evaluation mode

	// indexTileCoding represents whether the environment is using
	// tile coding and returning the non-zero indices as features
	indexTileCoding bool
}

// New creates a new QLearning struct. The agent environment agent should be
// a environment.QLearning. If the agent environment is not a environment.QLearning, New
// will panic.
func New(env environment.Environment, config agent.Config,
	init weights.Initializer, seed uint64) (agent.Agent, error) {
	// Ensure environment has discrete actions
	if env.ActionSpec().Cardinality != environment.Discrete {
		return nil, fmt.Errorf("qlearning: cannot use non-discrete " +
			"actions")
	}
	if env.ActionSpec().LowerBound.Len() > 1 {
		return nil, fmt.Errorf("qlearning: actions must be " +
			"1-dimensional")
	}
	if env.ActionSpec().LowerBound.AtVec(0) != 0.0 {
		return nil, fmt.Errorf("qlearning: actions must be " +
			"enumerated starting from 0")
	}
	if !config.ValidAgent(&QLearning{}) {
		return nil, fmt.Errorf("qlearning: invalid agent for configuration "+
			"type %T", config)
	}

	c := config.(Config)
	if !c.ValidAgent(&QLearning{}) {
		return nil, fmt.Errorf("qlearning: invalid config for agent QLearning")
	}
	err := c.Validate()
	if err != nil {
		return nil, fmt.Errorf("qlearning: %v", err)
	}

	// Get the behaviour policy
	e := c.Epsilon
	behaviourPol, err := policy.NewEGreedy(e, seed, env)
	if err != nil {
		return &QLearning{}, fmt.Errorf("qlearning: invalid behaviour "+
			"policy: %v", err)
	}
	behaviour := behaviourPol.(*policy.EGreedy)

	// Get the target policy
	targetPol, err := policy.NewGreedy(seed, env)
	if err != nil {
		return &QLearning{}, fmt.Errorf("qlearning: invalid target "+
			"policy: %v", err)
	}
	target := targetPol.(*policy.EGreedy)

	// Get the learning rate
	learningRate := c.LearningRate

	// Ensure both policies and learner reference the same weights
	weights := behaviour.Weights()
	target.SetWeights(weights)

	// Check if the environment uses tile coding and returns the
	// indices of non-zero elements of the tile-coded vectors as
	// state representations
	_, indexTileCoding := env.(*wrappers.IndexTileCoding)

	learner, err := NewQLearner(behaviour, learningRate, indexTileCoding)
	if err != nil {
		err := fmt.Errorf("qlearning: cannot create learner")
		return &QLearning{}, err
	}

	// Initialize weights
	for weight := range weights {
		init.Initialize(weights[weight])
	}

	return &QLearning{
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
func (q *QLearning) SelectAction(t timestep.TimeStep) *mat.VecDense {
	if !q.eval {
		return q.Policy.SelectAction(t)
	}
	return q.Target.SelectAction(t)
}

// Step wraps the stepping operations of the QLearner so that stepping
// is not permitted when in evaluation mode.
func (q *QLearning) Step() {
	if q.IsEval() {
		return
	}
	q.Learner.Step()
}
