// Package wrappers provides wrappers for environments
package wrappers

import (
	"math"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/gonum/stat/samplemv"

	"sfneuman.com/golearn/utils/matutils"
)

// Controls tiling offsets. For each dimension, tilings are offset by
// randomly sampling from a uniform distribution with support
// [- tiling width/OffsetDiv, tiling width/OffsetDiv]
const OffsetDiv float64 = 1.5

// TileCoder implements functionality for tile coding a vector. Tile
// coding takes a low-dimensional vector and changes it into a large,
// sparse vector consisting of only 0's and 1's. Each 1 represents the
// coordinates of the original vector in some space of tilings. For
// example:
//
//		[0.5, 0.1] -> [0, 0, 0, 1, 0, 0, 1, 0]
//
//
// The number of nonzero elements in the tile-coded representation equals
// the number of tilings used to encode the vector. The number of total
// features in the tile-coded representation is the number of tilings
// times the number of tiles per tiling. Tile coding requires that the
// space to be tiled be bounded.
//
// This implementation of tile coding uses dense tilings over the entire
// state space. That is, each dimension of state space is fully tiled,
// and hash-based tile coding is not used. This implementation also
// uses multiple tilings, each of which consist of the name number
// of tiles per tiling.
type TileCoder struct {
	numTilings        int
	minDims           mat.Vector
	offsets           []*mat.Dense
	bins              []int
	binLengths        []float64
	seed              uint64
	featuresPerTiling int
}

// NewTileCoder creates and returns a new TileCoder struct. The minDims
// and maxDims arguments are the bounds on each dimension between which
// tilings will be placed. These arguments should have the same shape
// as vectors which will be tile coded. The bins argument determines
// how many tiles are placed (per tilings) along each dimension and
// should have the same number of elements as the minDims and maxDims
// arguments.
func NewTileCoder(numTilings int, minDims, maxDims mat.Vector, bins []int,
	seed uint64) TileCoder {
	// Calculate the length of bins and the tiling offset bounds
	var bounds []r1.Interval
	binLengths := make([]float64, len(bins))
	for i := 0; i < minDims.Len(); i++ {
		// Calculate the length of bins
		binLength := (maxDims.AtVec(i) - minDims.AtVec(i)) / float64(bins[i])
		bound := binLength / OffsetDiv // Bounds tiling offsets

		binLengths[i] = binLength
		bounds = append(bounds, r1.Interval{Min: -bound, Max: bound})
	}

	// Create RNG for uniform sampling of tiling offsets
	source := rand.NewSource(seed)
	u := distmv.NewUniform(bounds, source)
	sampler := samplemv.IID{Dist: u}

	// Calculate offsets
	var offsets []*mat.Dense
	for i := 0; i < numTilings; i++ {
		samples := mat.NewDense(1, len(bounds), nil)
		sampler.Sample(samples)

		offsets = append(offsets, samples)
	}

	// Number of features in each tiling
	featuresPerTiling := prod(bins)

	return TileCoder{numTilings, minDims, offsets, bins, binLengths, seed,
		featuresPerTiling}
}

func (t TileCoder) EncodeBatch(b *mat.Dense) *mat.Dense {
	rows, cols := b.Dims()
	tileCoded := mat.NewDense(t.VecLength(), cols, nil)

	// Create vector of ones
	oneSlice := make([]float64, cols)
	for i := 0; i < rows; i++ {
		oneSlice[i] = 1.0
	}
	ones := mat.NewVecDense(rows, oneSlice)

	// Vector that holds all the data that is manipulated
	data := mat.NewVecDense(rows, nil)

	// Tile code
	for j := 0; j < t.numTilings; j++ {
		indexOffset := j * t.featuresPerTiling
		index := mat.NewVecDense(rows, nil)

		for i := len(t.bins) - 1; i > -1; i-- {
			// Clone the next batch of features into the data vector
			data.CloneFromVec(b.RowView(i))

			data.AddScaledVec(data, t.offsets[j].At(0, i), ones)
			data.AddScaledVec(data, -t.minDims.AtVec(i), ones)

			matutils.VecFloor(data, t.binLengths[i])
			matutils.VecClip(data, 0.0, float64(t.bins[i]-1))

			if i == len(t.bins)-1 {
				index.AddVec(index, data)
			} else {
				index.AddScaledVec(index, float64(t.bins[i+1]), data)
			}
		}

		// Set the proper 1.0 values
		for i := 0; i < index.Len(); i++ {
			row := indexOffset + int(index.AtVec(i))
			tileCoded.Set(row, i, 1.0)
		}
	}
	return tileCoded
}

func (t TileCoder) Encode(v mat.Vector) *mat.VecDense {
	tileCoded := mat.NewVecDense(t.VecLength(), nil)

	for j := 0; j < t.numTilings; j++ {
		indexOffset := j * t.featuresPerTiling
		index := 0

		for i := len(t.bins) - 1; i > -1; i-- {
			// Offset the tiling
			data := v.AtVec(i) + t.offsets[j].At(0, i)

			tile := math.Floor((data - t.minDims.AtVec(i)) / t.binLengths[i])

			// Clip tile to within tiling bounds
			tile = math.Min(tile, float64(t.bins[i]-1))
			tile = math.Max(tile, 0)

			tileIndex := int(tile)
			if i == len(t.bins)-1 {
				index += tileIndex
			} else {
				index += tileIndex * t.bins[i+1]
			}
		}
		tileCoded.SetVec(indexOffset+index, 1.0)
	}
	return tileCoded
}

func (t TileCoder) VecLength() int {
	return t.numTilings * t.featuresPerTiling
}

func prod(i []int) int {
	prod := 1
	for _, v := range i {
		prod *= v
	}
	return prod
}
