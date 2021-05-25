// Package cartpole implements the Cartpole classic control environment
package cartpole

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
)

type Continuous struct {
	*base
}

// New constructs a new Cartpole environment
func NewContinuous(t env.Task, discount float64) (*Continuous, ts.TimeStep) {
	base, firstStep := newBase(t, discount)
	cartpole := Continuous{base}

	return &cartpole, firstStep
}

// ActionSpec returns the action specification of the environment
func (c *Continuous) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{float64(MinDiscreteAction)})
	upperBound := mat.NewVecDense(1, []float64{float64(MaxDiscreteAction)})

	return spec.NewEnvironment(shape, spec.Action, lowerBound,
		upperBound, spec.Continuous)
}

// Step takes one environmental step given action a and returns the next
// state as a timestep.TimeStep and a bool indicating whether or not the
// episode has ended
func (c *Continuous) Step(a mat.Vector) (ts.TimeStep, bool) {
	// Continuous action in [-1, 1]
	directionMagnitude := a.AtVec(0)

	// Ensure a legal action was selected
	if directionMagnitude < MinContinuousAction ||
		directionMagnitude > MaxContinuousAction {
		panic(fmt.Sprintf("illegal action %v \u2209 [-1, 1]",
			directionMagnitude))
	}

	// Calculate the next state given the direction to apply force
	nextState := c.nextState(directionMagnitude)

	// Update the embedded base Cartpole environment
	return c.update(a, nextState)
}
