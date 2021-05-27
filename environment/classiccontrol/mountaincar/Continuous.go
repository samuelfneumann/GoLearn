package mountaincar

import (
	"gonum.org/v1/gonum/mat"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/floatutils"
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
func NewContinuous(t env.Task, discount float64) (*Continuous, ts.TimeStep) {
	// Create and store the base Mountain Car environment
	baseEnv, firstStep := newBase(t, discount)

	mountainCar := Continuous{baseEnv}

	return &mountainCar, firstStep

}

// ActionSpec returns the action specification of the environment
func (m *Continuous) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(ActionDims, nil)
	lowerBound := mat.NewVecDense(ActionDims, []float64{MinContinuousAction})
	upperBound := mat.NewVecDense(ActionDims, []float64{MaxContinuousAction})

	return spec.NewEnvironment(shape, spec.Action, lowerBound,
		upperBound, spec.Continuous)

}

// Step takes one environmental step given action a and returns the next
// state as a timestep.TimeStep and a bool indicating whether or not the
// episode has ended
func (m *Continuous) Step(a mat.Vector) (ts.TimeStep, bool) {
	// Ensure action is 1-dimensional
	if a.Len() > ActionDims {
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
