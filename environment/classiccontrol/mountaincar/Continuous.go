package mountaincar

import (
	"fmt"

	env "github.com/samuelfneumann/golearn/environment"
	ts "github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/golearn/utils/floatutils"
	"gonum.org/v1/gonum/mat"
)

// Continuous implements the classic control Mountain Car environment.
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
// Actions are 1-dimensional and continuous. Actions determine the force
// to apply to the car and in which direction to apply this force.
// Actions are bounded between [-1, 1] = [MinContinuousAction,
// MaxContinuousAction], and actions outside of this range are clipped
// to stay within this range.
//
// Continuous implements the environment.Environment interface
type Continuous struct {
	*base
}

// NewContinuous creates a new Continuous action Mountain Car
// environment with the argument task
func NewContinuous(t env.Task, discount float64) (env.Environment,
	ts.TimeStep, error) {
	// Create and store the base Mountain Car environment
	baseEnv, firstStep, err := newBase(t, discount)
	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("newContinuous: %v", err)
	}

	mountainCar := Continuous{baseEnv}

	return &mountainCar, firstStep, nil

}

// ActionSpec returns the action specification of the environment
func (m *Continuous) ActionSpec() env.Spec {
	shape := mat.NewVecDense(ActionDims, nil)
	lowerBound := mat.NewVecDense(ActionDims, []float64{MinContinuousAction})
	upperBound := mat.NewVecDense(ActionDims, []float64{MaxContinuousAction})

	return env.NewSpec(shape, env.Action, lowerBound,
		upperBound, env.Continuous)

}

// Step takes one environmental step given action a and returns the next
// timestep as a timestep.TimeStep and a bool indicating whether or not
// the episode has ended. Actions are 1-dimensional and continuous,
// consisting of the horizontal force to apply to the cart. Actions
// outside the legal range of [-2, 2] are clipped to stay within this
// range.
func (m *Continuous) Step(a *mat.VecDense) (ts.TimeStep, bool, error) {
	// Ensure action is 1-dimensional
	if a.Len() > ActionDims {
		return ts.TimeStep{}, true, fmt.Errorf("Actions should be " +
			"1-dimensional")
	}

	// Clip action to legal range
	force := floatutils.Clip(a.AtVec(0), MinContinuousAction,
		MaxContinuousAction)

	// Calculate the next state given the force/action
	newState := m.nextState(force)

	// Update embedded base Mountain Car environment
	nextStep, last := m.update(a, newState)

	return nextStep, last, nil
}
