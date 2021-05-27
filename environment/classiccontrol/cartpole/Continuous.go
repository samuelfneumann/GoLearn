// Package cartpole implements the Cartpole classic control environment
package cartpole

import (
	"gonum.org/v1/gonum/mat"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/floatutils"
)

// cartpole.Continuous implements the classic control environment
// Cartpole with continuous actions. In this environment, a pole is
// attached to a cart, which can move horizontally. Gravity pulls the
// pole downwards so that balancing it in an upright position is very
// difficult.
//
// The state features are continuous and consist of the cart's x
// position and speed, as well as the pole's angle from the positive
// y-axis and the pole's angular velocity. All state features are
// bounded by the constants defined in this package. For the position,
// speed, and angular velocity features, extreme values are clipped to
// within the legal ranges. For the pole's angle feature, extreme values
// are normalized so that all angles stay in the range (-π, π].
//
// Actions are continuous and 1-dimensional, consisting of the force
// to apply to the cart horizontally. Negative force moves the cart left
// and positive force moves the cart right. Actions are bounded to the
// interval [-1, 1]. Actions outside of this range are clipped to stay
// within this range.
type Continuous struct {
	*base
}

// NewContinuous constructs a new Cartpole environment with continuous
// actions
func NewContinuous(t env.Task, discount float64) (*Continuous, ts.TimeStep) {
	base, firstStep := newBase(t, discount)
	cartpole := Continuous{base}

	return &cartpole, firstStep
}

// ActionSpec returns the action specification of the environment
func (c *Continuous) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{float64(MinContinuousAction)})
	upperBound := mat.NewVecDense(1, []float64{float64(MaxContinuousAction)})

	return spec.NewEnvironment(shape, spec.Action, lowerBound,
		upperBound, spec.Continuous)
}

// Step takes one environmental step given action a and returns the next
// state as a timestep.TimeStep and a bool indicating whether or not the
// episode has ended. Actions are 1-dimensional, consisting of the
// horizontal force to apply to the cart. Actions outside the legal
// range of [-1, 1] are clipped to stay within this range.
func (c *Continuous) Step(a mat.Vector) (ts.TimeStep, bool) {
	// Ensure action is 1-dimensional
	if a.Len() > 1 {
		panic("Actions should be 1-dimensional")
	}

	// Continuous action in [-1, 1]
	directionMagnitude := a.AtVec(0)
	directionMagnitude = floatutils.Clip(directionMagnitude,
		MinContinuousAction, MaxContinuousAction)

	// Calculate the next state given the direction to apply force
	nextState := c.nextState(directionMagnitude)

	// Update the embedded base Cartpole environment
	return c.update(a, nextState)
}
