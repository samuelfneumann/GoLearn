package mountaincar

import (
	"math"

	"gonum.org/v1/gonum/mat"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/floatutils"
)

const (
	MinAction float64 = -1.0
	MaxAction float64 = 1.0
	Power     float64 = 0.0015 // The power of the engine
)

// mountaincar.Continuous in a classic control environment where an agent must
// learn to drive an underpowered car up a hill. Actions are continuous in
// [-1.0, 1.0]. Actions outside of this range are clipped to within this range.
//
//For more information on the Mountain Car environment, see
// mountaincar.Discrete.
type Continuous struct {
	*base
	power float64
}

// New creates a new Continuous environment with the argument task
func NewContinuous(t env.Task, discount float64) (*Continuous, ts.TimeStep) {
	// Create and store the base Mountain Car environment
	baseEnv, firstStep := newBase(t, discount)
	power := Power

	mountainCar := Continuous{baseEnv, power}

	return &mountainCar, firstStep

}

// ActionSpec returns the action specification of the environment
func (m *Continuous) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{MinAction})
	upperBound := mat.NewVecDense(1, []float64{MaxAction})

	return spec.NewEnvironment(shape, spec.Action, lowerBound,
		upperBound, spec.Continuous)

}

// NextState calculates the next state in the environment given action a
func (m *Continuous) NextState(a mat.Vector) mat.Vector {
	// Clip action to legal range
	force := floatutils.Clip(a.AtVec(0), MinAction, MaxAction)

	// Get the current state
	state := m.lastStep.Observation
	position, velocity := state.AtVec(0), state.AtVec(1)

	// Update the velocity
	velocity += force*m.power - 0.0025*math.Cos(3*position)
	velocity = floatutils.Clip(velocity, m.speedBounds.Min, m.speedBounds.Max)

	// Update the position
	position += velocity
	position = floatutils.Clip(position, m.positionBounds.Min,
		m.positionBounds.Max)

	// Ensure position stays within bounds
	if position <= m.positionBounds.Min && velocity < 0 {
		velocity = 0
	}

	// Create the new timestep
	newState := mat.NewVecDense(2, []float64{position, velocity})
	return newState

}

// Step takes one environmental step given action a and returns the next
// state as a timestep.TimeStep and a bool indicating whether or not the
// episode has ended
func (m *Continuous) Step(a mat.Vector) (ts.TimeStep, bool) {
	newState := m.NextState(a)

	// Update embedded base Mountain Car environment
	nextStep, last := m.update(a, newState)

	return nextStep, last
}
