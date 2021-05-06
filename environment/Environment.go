// Package environment outlines the interfaces and sturcts needed to implement
// concrete environments
package environment

import (
	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/timestep"
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
}

// Task implements the reward scheme for taking actions in some environment
type Task interface {
	// fmt.Stringer
	GetReward(t timestep.TimeStep, action mat.Vector) float64
}

// Environment implements a simualted environment, which includes a Task to
// complete
type Environment interface {
	Task
	// fmt.Stringer
	Reset() timestep.TimeStep
	Step(action mat.Vector) (timestep.TimeStep, bool)
	RewardSpec() Spec
	DiscountSpec() Spec
	ObservationSpec() Spec
	ActionSpec() Spec
}
