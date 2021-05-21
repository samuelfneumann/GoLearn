// Package mountaincar implements the discrete action classic control
// environment "Mountain Car"
package mountaincar

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
)

const (
	MinPosition float64 = -1.2
	MaxPosition float64 = 0.6
	MaxSpeed    float64 = 0.07
	Force       float64 = 0.001
	Gravity     float64 = 0.0025
)

// mountaincar.Discrete in a classic control environment where an agent must
// learn to drive an underpowered car up a hill. Actions are discrete in
// (0, 1, 2) where:
//
//	Action	Meaning
//	  0		Accelerate left
//	  1		Do nothing
//	  2		Accelerate right
//
//  Actions other than 0, 1, or 2 result in a panic
//
// When designing a starter for this environment, care should be taken to
// ensure that the starting states are chosen within the environmental
// bounds. If the starter produces a state outside of the position and
// speed bounds, the environment will panic. This may happen near the
// end of training, resulting in significant data loss.
//
// Any taks may be used with the mountaincar.Discrete environment, but the
// classic control task is defined in the mountaincar.Goal struct, where the
// agent must learn to reach the goal at the top of the hill.
type Discrete struct {
	*base
}

// New creates a new Discrete environment with the argument task
func NewDiscrete(t env.Task, discount float64) (*Discrete, ts.TimeStep) {
	// Create and store the base Mountain Car environment
	baseEnv, firstStep := newBase(t, discount)

	mountainCar := Discrete{baseEnv}

	return &mountainCar, firstStep

}

// ActionSpec returns the action specification of the environment
func (m *Discrete) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{0})
	upperBound := mat.NewVecDense(1, []float64{2})

	return spec.NewEnvironment(shape, spec.Action, lowerBound,
		upperBound, spec.Discrete)

}

// NextState calculates the next state in the environment given action a
func (m *Discrete) NextState(a mat.Vector) mat.Vector {
	action := a.AtVec(0)

	// Ensure a legal action was selected
	intAction := int(action)
	if intAction != 0 && intAction != 1 && intAction != 2 {
		panic(fmt.Sprintf("illegal action %v \u2209 (0, 1, 2)", intAction))
	}

	// Get the current state
	state := m.lastStep.Observation
	position, velocity := state.AtVec(0), state.AtVec(1)

	// Update the velocity
	velocity += (action-1.0)*m.force + math.Cos(3*position)*(-m.gravity)
	velocity = math.Min(velocity, m.speedBounds.Max)
	velocity = math.Max(velocity, m.speedBounds.Min)

	// Update the position
	position += velocity
	position = math.Min(position, m.positionBounds.Max)
	position = math.Max(position, m.positionBounds.Min)

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
func (m *Discrete) Step(a mat.Vector) (ts.TimeStep, bool) {
	newState := m.NextState(a)

	// Update embedded base Mountain Car environment
	nextStep, last := m.update(a, newState)

	return nextStep, last

}
