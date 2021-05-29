// Package mountaincar implements the discrete action classic control
// environment "Mountain Car"
package mountaincar

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
)

// Discrete implements the classic control Mountain Car environment.
// In this environment, the agent controls a car in a valley between two
// hills. The car is underpowered and cannot drive up the hill unless
// it rocks back and forth from hill to hill, using its momentum to
// gradually climb higher.
//
// State features consist of the x position of the car and its velocity.
// These features are bounded by the MinPosition, MaxPosition, and
// MaxSpeed constants defined in this package. The sign of the velocity
// feature denotes direction, with negative meaning that the car is
// travelling left and positive meaning that the car is travelling
// right. Upon reaching the minimum or maximum position, the velocity
// of the car is set to 0.
//
// Actions are 1-dimensional and discrete in (0, 1, 2). Actions
// determine in which direction to apply full accelerating force to the
// car:
//
//	Action	Meaning
//	  0		Accelerate left
//	  1		Do nothing
//	  2		Accelerate right
//
// Actions other than 0, 1, or 2 result in a panic
//
// Discrete implements the environment.Environment interface

type Discrete struct {
	*base
}

// New creates a new Discrete action Mountain Car environment with the
// argument task
func NewDiscrete(t env.Task, discount float64) (env.Environment,
	ts.TimeStep) {
	// Create and store the base Mountain Car environment
	baseEnv, firstStep := newBase(t, discount)

	mountainCar := Discrete{baseEnv}

	return &mountainCar, firstStep

}

// ActionSpec returns the action specification of the environment
func (m *Discrete) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(ActionDims, nil)
	lowerBound := mat.NewVecDense(ActionDims, []float64{float64(MinDiscreteAction)})
	upperBound := mat.NewVecDense(ActionDims, []float64{float64(MaxDiscreteAction)})

	return spec.NewEnvironment(shape, spec.Action, lowerBound,
		upperBound, spec.Discrete)

}

// Step takes one environmental step given action a and returns the next
// timestep as a timestep.TimeStep and a bool indicating whether or not
// the episode has ended. Actions are discrete, consisting of the
// direction to accelerate the car or whether to apply no acceleration
// to the car. Legal actions are in the set {0, 1, 2}. Actions outside
// this range will cause the environment to panic.
func (m *Discrete) Step(a mat.Vector) (ts.TimeStep, bool) {
	// Ensure action is 1-dimensional
	if a.Len() > ActionDims {
		panic("Actions should be 1-dimensional")
	}

	// Discrete action in {0, 1, 2}
	action := a.AtVec(0)

	// Ensure a legal action was selected
	intAction := int(action)
	if intAction > MaxDiscreteAction || intAction < MinDiscreteAction {
		panic(fmt.Sprintf("illegal action %v \u2209 (0, 1, 2)", intAction))
	}

	// Calculate the force
	force := action - 1.0

	// Calculate the next state given the force/action
	newState := m.nextState(force)

	// Update embedded base Mountain Car environment
	return m.update(a, newState)
}
