// Package cartpole implements the Cartpole classic control environment
package cartpole

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
)

// cartpole.Discrete implements the classic control environment
// Cartpole with discrete actions. In this environment, a pole is
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
// are normalized so that all angles stay in the range (-π, π]. Upon
// reaching a position boundary, the velocity of the cart is set to 0.
//
// Actions are discrete, consisting of the direction to apply
// horizontal force to the cart. Legal actions are in {0, 1, 2}:
//
//	Action		Meaning
//	  0			Apply force left
//	  1			Do nothing
//	  2			Apply force right
//
// Illegal actions will cause the environment to panic.
//
// Discrete implements the environment.Environment interface
type Discrete struct {
	*base
}

// NewDiscrete constructs a new Cartpole environment with discrete
// actions
func NewDiscrete(t env.Task, discount float64) (*Discrete, ts.TimeStep) {
	base, firstStep := newBase(t, discount)
	cartpole := Discrete{base}

	return &cartpole, firstStep
}

// ActionSpec returns the action specification of the environment
func (c *Discrete) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(ActionDims, nil)
	lowerBound := mat.NewVecDense(ActionDims, []float64{float64(MinDiscreteAction)})
	upperBound := mat.NewVecDense(ActionDims, []float64{float64(MaxDiscreteAction)})

	return spec.NewEnvironment(shape, spec.Action, lowerBound,
		upperBound, spec.Discrete)
}

// Step takes one environmental step given action a and returns the next
// timestep as a timestep.TimeStep and a bool indicating whether or not
// the episode has ended. Actions are discrete, consisting of the
// direction to apply horizontal force to the cart or whether to apply
// no force to the cart. Legal actions are in the set {0, 1, 2}.
// Actions outside this range will cause the environment to panic.
func (c *Discrete) Step(a *mat.VecDense) (ts.TimeStep, bool) {
	// Ensure action is 1-dimensional
	if a.Len() > ActionDims {
		panic("Actions should be 1-dimensional")
	}

	// Discrete action in {0, 1, 2}
	direction := a.AtVec(0)

	// Ensure a legal action was selected
	intAction := int(direction)
	if intAction < MinDiscreteAction || intAction > MaxDiscreteAction {
		panic(fmt.Sprintf("illegal action %v \u2209 (0, 1, 2)", intAction))
	}

	// Convert action (0, 1, 2) to a direction (-1, 0, 1)
	direction--

	// Calculate the next state given the direction to apply force
	nextState := c.nextState(direction)

	// Update the embedded base Cartpole environment
	return c.update(a, nextState)
}
