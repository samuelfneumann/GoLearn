// Package wrappers provides wrappers for environments
package wrappers

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/matutils"
	"sfneuman.com/golearn/utils/matutils/tilecoder"
)

// TileCoding wraps an environment and returns tile-coded observations
// of the environment states. TileCoding itself implements the
// environment.Environment interface and is therefore itself an
// environment. All tile-coded representations will contain a bias unit
// as the first feature in the tile-coded representation.
type TileCoding struct {
	environment.Environment
	coder *tilecoder.TileCoder
}

// NewTileCoding creates and returns a new TileCoding environment
func NewTileCoding(env environment.Environment, numTilings int, bins []int,
	seed uint64) (*TileCoding, ts.TimeStep) {
	envSpec := env.ObservationSpec()
	minDims := envSpec.LowerBound
	maxDims := envSpec.UpperBound

	coder := tilecoder.New(numTilings, minDims, maxDims, bins, seed, true)

	// Reset the tile-coded environment
	step := env.Reset()
	step.Observation = coder.Encode(step.Observation)

	return &TileCoding{env, coder}, step
}

// Reset resets the environment to some starting state
func (t *TileCoding) Reset() ts.TimeStep {
	step := t.Environment.Reset()

	// Tile code first observation
	obs := t.coder.Encode(step.Observation)
	step.Observation = obs

	return step
}

// Step takes one environmental step given action a and returns the next
// state as a timestep.TimeStep and a bool indicating whether or not the
// episode has ended
func (t *TileCoding) Step(a mat.Vector) (ts.TimeStep, bool) {
	// Get the next step from the environment
	step, last := t.Environment.Step(a)

	// Tile code the observation
	obs := t.coder.Encode(step.Observation)
	step.Observation = obs

	return step, last
}

// ObservationSpec returns the observation specification of the
// environment
func (t *TileCoding) ObservationSpec() spec.Environment {
	length := t.coder.VecLength()
	shape := mat.NewVecDense(length, nil)

	lowerBound := mat.NewVecDense(length, nil)

	upperBound := matutils.VecOnes(length)

	return spec.NewEnvironment(shape, spec.Observation, lowerBound,
		upperBound, spec.Continuous)

}

// String returns a string representation of the TileCoding environment
func (t *TileCoding) String() string {
	return fmt.Sprintf("TileCoding: %v", t.Environment)
}
