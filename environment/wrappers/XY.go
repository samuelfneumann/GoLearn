package wrappers

import (
	"fmt"

	env "github.com/samuelfneumann/golearn/environment"
	ts "github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/golearn/utils/floatutils"
	"gonum.org/v1/gonum/mat"
)

// XY converts one-hot state encodings of environment.RowColer's
// to the corresponding (x, y) == (col, row) coordinates.
type XY struct {
	env.RowColer

	currentTimeStep ts.TimeStep
}

// NewXY returns a new XY environment wrapper
func NewXY(e env.RowColer) env.Environment {
	return &XY{
		RowColer:        e,
		currentTimeStep: ts.TimeStep{},
	}
}

// Reset resets the environment to some starting state
func (x *XY) Reset() (ts.TimeStep, error) {
	// Reset embedded environment
	step, err := x.RowColer.Reset()
	if err != nil {
		return ts.TimeStep{}, err
	}

	// Convert the observation to (x, y) coordinates
	newObs, err := x.getObs(step.Observation)
	if err != nil {
		return ts.TimeStep{}, fmt.Errorf("reset: could not calculate "+
			"observation: %v", err)
	}

	step.Observation = newObs
	x.currentTimeStep = step

	return step, nil
}

// Step takes one environmental step given some action
func (x *XY) Step(action *mat.VecDense) (ts.TimeStep, bool, error) {
	// Take a step in the embedded environment
	// step will be the TimeStep with S_{t+1} and R_{t} for action A_{t}
	step, _, err := x.RowColer.Step(action)
	if err != nil {
		return ts.TimeStep{}, true, err
	}

	// Convert the observation to (x, y) coordinates
	newObs, err := x.getObs(step.Observation)
	if err != nil {
		return ts.TimeStep{}, true, fmt.Errorf("step: could not calculate "+
			"observation: %v", err)
	}

	step.Observation = newObs
	x.currentTimeStep = step

	return step, step.Last(), nil
}

// CurrentTimeStep returns the current time step in the environment
func (x *XY) CurrentTimeStep() ts.TimeStep {
	return x.currentTimeStep
}

// getObs returns the (x, y) version of a one-hot encoded vector
func (x *XY) getObs(obs *mat.VecDense) (*mat.VecDense, error) {
	// Get the index of the 1 in the one-hot encoding
	index := floatutils.Where(
		obs.RawVector().Data,
		func(v float64) bool {
			return v == 1.0
		},
	)
	if len(index) != 1 {
		return nil, fmt.Errorf("getObs: vector is not one-hot")
	}

	// Construct the (x, y) vector observation
	newObs := mat.NewVecDense(2, nil)
	row := x.Rows() * (index[0] / x.Cols())
	col := float64(index[0] - x.Cols()*row)
	newObs.SetVec(0, col)
	newObs.SetVec(1, float64(row))

	return newObs, nil
}

// ObservationSpec returns the observation specification of the
// environment
func (x *XY) ObservationSpec() env.Spec {
	shape := mat.NewVecDense(2, nil)
	low := mat.NewVecDense(2, []float64{0, 0})
	high := mat.NewVecDense(2, []float64{
		float64(x.Cols() - 1),
		float64(x.Rows() - 1),
	})

	return env.NewSpec(shape, env.Observation, low, high, env.Discrete)
}

// String returns the string representation of the environment
func (x *XY) String() string {
	return fmt.Sprintf("XY: %v", x.RowColer)
}
