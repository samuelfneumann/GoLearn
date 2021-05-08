// Package environment outlines the interfaces and sturcts needed to implement
// concrete environments
package environment

// TODO: Create a start distribution type that each env has and samples start states from

import (
	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/timestep"
)

// Starter implements a distribution of starting states and samples starting
// states for environments
type Starter interface {
	Start() mat.Vector
}

// Cardinality indicates whether the associated type is continuous or discrete
type Cardinality int

const (
	Discrete Cardinality = iota
	Continuous
)

// SpecType determines what kind of specification a Spec is. A Spec can
// specify the layout of an acion, an observation, a discount, or a reward
type SpecType int

const (
	Action SpecType = iota
	Observation
	Discount
	Reward
)

// Spec implements a specification, which tells the type, shape, and bounds of
// an action, observation, discount, or reward
type Spec struct {
	Shape      mat.Vector
	Type       SpecType
	LowerBound mat.Vector
	UpperBound mat.Vector
	Cardinality
}

// Task implements the reward scheme for taking actions in some environment
type Task interface {
	// fmt.Stringer
	GetReward(t timestep.TimeStep, a mat.Vector) float64
	AtGoal(state mat.Matrix) bool
}

// Environment implements a simualted environment, which includes a Task to
// complete
type Environment interface {
	Task
	Starter
	// fmt.Stringer
	New() (Environment, timestep.TimeStep) // Environment starts ready to use
	Reset() timestep.TimeStep              // Resets between episodes
	Step(action mat.Vector) (timestep.TimeStep, bool)
	RewardSpec() Spec
	DiscountSpec() Spec
	ObservationSpec() Spec
	ActionSpec() Spec
}
