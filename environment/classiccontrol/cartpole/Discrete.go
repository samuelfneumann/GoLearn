// Package cartpole implements the Cartpole classic control environment
package cartpole

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
)

type Discrete struct {
	*base
}

// New constructs a new Cartpole environment
func NewDiscrete(t env.Task, discount float64) (*Discrete, ts.TimeStep) {
	base, firstStep := newBase(t, discount)
	cartpole := Discrete{base}

	return &cartpole, firstStep
}

// ActionSpec returns the action specification of the environment
func (c *Discrete) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{float64(MinDiscreteAction)})
	upperBound := mat.NewVecDense(1, []float64{float64(MaxDiscreteAction)})

	return spec.NewEnvironment(shape, spec.Action, lowerBound,
		upperBound, spec.Discrete)
}

// Step takes one environmental step given action a and returns the next
// state as a timestep.TimeStep and a bool indicating whether or not the
// episode has ended
func (c *Discrete) Step(a mat.Vector) (ts.TimeStep, bool) {
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
