// Package mountaincar implements the discrete action classic control
// environment "Mountain Car"
package mountaincar

import (
	"fmt"

	env "github.com/samuelfneumann/golearn/environment"
	ts "github.com/samuelfneumann/golearn/timestep"
	"gonum.org/v1/gonum/mat"
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
// Discrete implements the environment.Environment interface

type Discrete struct {
	*base
}

// New creates a new Discrete action Mountain Car environment with the
// argument task
func NewDiscrete(t env.Task, discount float64) (env.Environment, ts.TimeStep,
	error) {
	// Create and store the base Mountain Car environment
	baseEnv, firstStep, err := newBase(t, discount)
	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("newDiscrete: %v", err)
	}

	mountainCar := Discrete{baseEnv}

	return &mountainCar, firstStep, nil

}

// ActionSpec returns the action specification of the environment
func (m *Discrete) ActionSpec() env.Spec {
	shape := mat.NewVecDense(ActionDims, nil)
	lowerBound := mat.NewVecDense(ActionDims, []float64{float64(MinDiscreteAction)})
	upperBound := mat.NewVecDense(ActionDims, []float64{float64(MaxDiscreteAction)})

	return env.NewSpec(shape, env.Action, lowerBound,
		upperBound, env.Discrete)

}

// Step takes one environmental step given action a and returns the next
// timestep as a timestep.TimeStep and a bool indicating whether or not
// the episode has ended. Actions are discrete, consisting of the
// direction to accelerate the car or whether to apply no acceleration
// to the car. Legal actions are in the set {0, 1, 2}. Actions outside
// this range will cause an error to be returned.
func (m *Discrete) Step(a *mat.VecDense) (ts.TimeStep, bool, error) {
	// Ensure action is 1-dimensional
	if a.Len() > ActionDims {
		return ts.TimeStep{}, true, fmt.Errorf("step: ctions should be " +
			"1-dimensional")
	}

	// Discrete action in {0, 1, 2}
	action := a.AtVec(0)

	// Ensure a legal action was selected
	intAction := int(action)
	if intAction > MaxDiscreteAction || intAction < MinDiscreteAction {
		return ts.TimeStep{}, true, fmt.Errorf("step: illegal action %v "+
			"\u2209 (0, 1, 2)", intAction)
	}

	// Calculate the force
	force := action - 1.0

	// Calculate the next state given the force/action
	newState := m.nextState(force)

	// Update embedded base Mountain Car environment
	nextStep, done := m.update(a, newState)
	return nextStep, done, nil
}
