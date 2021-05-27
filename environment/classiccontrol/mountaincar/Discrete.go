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

// New creates a new Discrete action Mountain Car environment with the
// argument task
func NewDiscrete(t env.Task, discount float64) (*Discrete, ts.TimeStep) {
	// Create and store the base Mountain Car environment
	baseEnv, firstStep := newBase(t, discount)

	mountainCar := Discrete{baseEnv}

	return &mountainCar, firstStep

}

// ActionSpec returns the action specification of the environment
func (m *Discrete) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{float64(MinDiscreteAction)})
	upperBound := mat.NewVecDense(1, []float64{float64(MaxDiscreteAction)})

	return spec.NewEnvironment(shape, spec.Action, lowerBound,
		upperBound, spec.Discrete)

}

// Step takes one environmental step given action a and returns the next
// state as a timestep.TimeStep and a bool indicating whether or not the
// episode has ended
func (m *Discrete) Step(a mat.Vector) (ts.TimeStep, bool) {
	// Ensure action is 1-dimensional
	if a.Len() > 1 {
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
