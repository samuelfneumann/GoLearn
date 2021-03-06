// Package cartpole implements the Cartpole classic control environment
package cartpole

import (
	"fmt"

	env "github.com/samuelfneumann/golearn/environment"
	ts "github.com/samuelfneumann/golearn/timestep"
	"gonum.org/v1/gonum/mat"
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
// Discrete implements the environment.Environment interface
type Discrete struct {
	*base
}

// NewDiscrete constructs a new Cartpole environment with discrete
// actions
func NewDiscrete(t env.Task, discount float64) (env.Environment,
	ts.TimeStep, error) {
	base, firstStep, err := newBase(t, discount)
	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("newDiscrete: %v", err)
	}
	cartpole := Discrete{base}

	return &cartpole, firstStep, nil
}

// ActionSpec returns the action specification of the environment
func (c *Discrete) ActionSpec() env.Spec {
	shape := mat.NewVecDense(ActionDims, nil)
	lowerBound := mat.NewVecDense(ActionDims, []float64{float64(MinDiscreteAction)})
	upperBound := mat.NewVecDense(ActionDims, []float64{float64(MaxDiscreteAction)})

	return env.NewSpec(shape, env.Action, lowerBound,
		upperBound, env.Discrete)
}

// Step takes one environmental step given action a and returns the next
// timestep as a timestep.TimeStep and a bool indicating whether or not
// the episode has ended. Actions are discrete, consisting of the
// direction to apply horizontal force to the cart or whether to apply
// no force to the cart. Legal actions are in the set {0, 1, 2}.
// Actions outside this range will cause an error to be returned.
func (c *Discrete) Step(a *mat.VecDense) (ts.TimeStep, bool, error) {
	// Ensure action is 1-dimensional
	if a.Len() > ActionDims {
		return ts.TimeStep{}, true, fmt.Errorf("step: actions should be " +
			"1-dimensional")
	}

	// Discrete action in {0, 1, 2}
	direction := a.AtVec(0)

	// Ensure a legal action was selected
	intAction := int(direction)
	if intAction < MinDiscreteAction || intAction > MaxDiscreteAction {
		return ts.TimeStep{}, true, fmt.Errorf("step: illegal action %v "+
			"\u2209 (0, 1, 2)", intAction)
	}

	// Convert action (0, 1, 2) to a direction (-1, 0, 1)
	direction--

	// Calculate the next state given the direction to apply force
	nextState := c.nextState(direction)

	// Update the embedded base Cartpole environment
	nextStep, done := c.update(a, nextState)
	return nextStep, done, nil
}
