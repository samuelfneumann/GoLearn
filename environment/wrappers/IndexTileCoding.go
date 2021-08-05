// Package wrappers provides wrappers for environments
package wrappers

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/spec"
	ts "github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/golearn/utils/matutils"
	"github.com/samuelfneumann/golearn/utils/matutils/tilecoder"
)

// IndexTileCoding wraps an environment and returns as observations
// of the environment states a vector of indices of non-zero components
// of the tile-coded representation of the environmental observation.
// For example, if the tile-coded representation of some environment
// state is [1 0 1 0 0 0 1], then this struct would return the vector
// [0 2 6] as the state observation.
//
// IndexTileCoding itself implements the
// environment.Environment interface and is therefore itself an
// environment. All tile-coded representations will contain a bias unit
// as the first feature in the tile-coded representation.
type IndexTileCoding struct {
	environment.Environment
	coder *tilecoder.TileCoder
}

// NewIndexTileCoding creates and returns a new IndexTileCoding environment,
// wrapping an existing environment. The wrapped environment
// is reset when wrapped by the IndexTileCoding environment by calling
// the wrapped environment's Reset() method.
//
// The bins parameter specifies both how many tilings to use as well
// as the number of tiles per tiling. The length of the outer-slice is
// the number of tilings. The lengths of the inner-slices are the
// number of bins per dimension for that tiling.
//
//
// See tilecoder.TileCoder for more details.
func NewIndexTileCoding(env environment.Environment, bins [][]int,
	seed uint64) (*IndexTileCoding, ts.TimeStep) {
	envSpec := env.ObservationSpec()
	minDims := envSpec.LowerBound
	maxDims := envSpec.UpperBound

	coder := tilecoder.New(minDims, maxDims, bins, seed, true)

	// Reset the tile-coded environment
	step := env.Reset()
	obs := coder.EncodeIndices(step.Observation)
	step.Observation = mat.NewVecDense(len(obs), obs)

	return &IndexTileCoding{env, coder}, step
}

// Reset resets the environment to some starting state
func (t *IndexTileCoding) Reset() ts.TimeStep {
	step := t.Environment.Reset()

	// Tile code first observation
	obs := t.coder.EncodeIndices(step.Observation)
	step.Observation = mat.NewVecDense(len(obs), obs)

	return step
}

// Step takes one environmental step given action a and returns the next
// state as a timestep.TimeStep and a bool indicating whether or not the
// episode has ended
func (t *IndexTileCoding) Step(a *mat.VecDense) (ts.TimeStep, bool) {
	// Get the next step from the environment
	step, last := t.Environment.Step(a)

	// Tile code the observation
	obs := t.coder.EncodeIndices(step.Observation)
	step.Observation = mat.NewVecDense(len(obs), obs)

	return step, last
}

// ObservationSpec returns the observation specification of the
// environment
func (t *IndexTileCoding) ObservationSpec() spec.Environment {
	length := t.coder.VecLength()
	shape := mat.NewVecDense(length, nil)

	lowerBound := mat.NewVecDense(length, nil)

	upperBound := matutils.VecOnes(length)

	return spec.NewEnvironment(shape, spec.Observation, lowerBound,
		upperBound, spec.Continuous)

}

// String returns a string representation of the IndexTileCoding environment
func (t *IndexTileCoding) String() string {
	return fmt.Sprintf("IndexTileCoding: %v", t.Environment)
}
