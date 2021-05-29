// Package agent defines an agent interface
package agent

import (
	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/timestep"
)

// Agent determines the implementation details of an agent or algorithm
//
// An Agent is composed of a Learner, which learns weights, and a Policy
// which chooses actions in each state. The Policy chooses which actions
// are taken, and the Learner uses these actions to update the Policy.
type Agent interface {
	Learner
	Policy
}

// Learner implements a learning algorithm that defines how weights are
// updated.
//
// A Learner determines how weights are changed, and therefore how a Policy
// changes over time. The Learner and Policy of an Agent should have pointers
// to the same weights so that the Learner can use the transitions chosen by
// the Policy to update the weights appropriately.
type Learner interface {
	Step() // Performs an update
	Observe(action mat.Vector, nextObs timestep.TimeStep)
	ObserveFirst(timestep.TimeStep)
	Weights() map[string]*mat.Dense
	SetWeights(map[string]*mat.Dense) error
	TdError(t timestep.Transition) float64
}

// Policy represents a policy that an agent can have.
//
// Policies determine how agents select actions. Agents usually have a
// target and behaviour policy. For a given agent, the Policy and Learner
// should have pointers to the same weights so that any changes the learner
// makes to the weights are reflected in the actions the Policy chooses
type Policy interface {
	SelectAction(t timestep.TimeStep) mat.Vector
	Weights() map[string]*mat.Dense
	SetWeights(map[string]*mat.Dense) error
}
