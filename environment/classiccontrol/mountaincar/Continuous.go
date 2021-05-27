package mountaincar

import (
	"gonum.org/v1/gonum/mat"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/floatutils"
)

// mountaincar.Continuous in a classic control environment where an agent must
// learn to drive an underpowered car up a hill. Actions are continuous in
// [-1.0, 1.0], representing the acceleration and direction of
// acceleration to be applied to the var. Actions outside of this range
// are clipped to within this range.
//
//For more information on the Mountain Car environment, see
// mountaincar.Discrete.
type Continuous struct {
	*base
}

// NewContinuous creates a new Continuous action Mountain Car
// environment with the argument task
func NewContinuous(t env.Task, discount float64) (*Continuous, ts.TimeStep) {
	// Create and store the base Mountain Car environment
	baseEnv, firstStep := newBase(t, discount)

	mountainCar := Continuous{baseEnv}

	return &mountainCar, firstStep

}

// ActionSpec returns the action specification of the environment
func (m *Continuous) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{MinContinuousAction})
	upperBound := mat.NewVecDense(1, []float64{MaxContinuousAction})

	return spec.NewEnvironment(shape, spec.Action, lowerBound,
		upperBound, spec.Continuous)

}

// Step takes one environmental step given action a and returns the next
// state as a timestep.TimeStep and a bool indicating whether or not the
// episode has ended
func (m *Continuous) Step(a mat.Vector) (ts.TimeStep, bool) {
	// Ensure action is 1-dimensional
	if a.Len() > 1 {
		panic("Actions should be 1-dimensional")
	}

	// Clip action to legal range
	force := floatutils.Clip(a.AtVec(0), MinContinuousAction,
		MaxContinuousAction)

	// Calculate the next state given the force/action
	newState := m.nextState(force)

	// Update embedded base Mountain Car environment
	nextStep, last := m.update(a, newState)

	return nextStep, last
}
