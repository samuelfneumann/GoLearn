// Package wrappers provides wrappers for environments
package wrappers

import (
	"math"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/gonum/stat/samplemv"
)

const OffsetDiv float64 = 1.5

// For now, we just implement a plain tile coder

type TileCoder struct {
	numTilings   int
	minDims      mat.Vector
	offsets      []*mat.Dense
	bins         []int
	binLengths   []float64
	seed         uint64
	tilingLength int
}

func NewTileCoder(numTilings int, minDims, maxDims mat.Vector, bins []int,
	seed uint64) TileCoder {

	var bounds []r1.Interval
	binLengths := make([]float64, len(bins))
	for i := 0; i < minDims.Len(); i++ {
		// Calculate the length of bins
		binLength := (maxDims.AtVec(i) - minDims.AtVec(i)) / float64(bins[i])
		bound := binLength / OffsetDiv

		binLengths[i] = binLength
		bounds = append(bounds, r1.Interval{Min: -bound, Max: bound})
	}

	// Calculate offsets
	var offsets []*mat.Dense
	source := rand.NewSource(seed)
	u := distmv.NewUniform(bounds, source)
	sampler := samplemv.IID{Dist: u}
	for i := 0; i < numTilings; i++ {
		samples := mat.NewDense(1, len(bounds), nil)
		sampler.Sample(samples)

		offsets = append(offsets, samples)
	}

	tilingLength := prod(bins)

	return TileCoder{numTilings, minDims, offsets, bins, binLengths, seed, tilingLength}
}

func (t TileCoder) EncodeBatch(b *mat.Dense) *mat.Dense {
	rows, cols := b.Dims()
	tileCoded := mat.NewDense(rows, t.VecLength(), nil)

	// Create vector of ones
	oneSlice := make([]float64, cols)
	for i := 0; i < cols; i++ {
		oneSlice[i] = 1.0
	}
	ones := mat.NewVecDense(cols, oneSlice)

	// Vector that holds all the data that is manipulated
	data := mat.NewVecDense(cols, nil)

	// Tile code
	for j := 0; j < t.numTilings; j++ {
		indexOffset := j * t.tilingLength
		index := mat.NewVecDense(cols, nil)

		for i := len(t.bins) - 1; i > -1; i-- {
			// Clone the next batch of features into the data vector
			data.CloneFromVec(b.ColView(i))

			data.AddScaledVec(data, t.offsets[j].At(0, i), ones)
			data.AddScaledVec(data, -t.minDims.AtVec(i), ones)

			VecFloor(data, t.binLengths[i])
			VecClip(data, 0.0, float64(t.bins[i]-1))

			if i == len(t.bins)-1 {
				index.AddVec(index, data)
			} else {
				index.AddScaledVec(index, float64(t.bins[i+1]), data)
			}
		}

		// Set the proper 1.0 values
		for i := 0; i < index.Len(); i++ {
			col := indexOffset + int(index.AtVec(i))
			tileCoded.Set(i, col, 1.0)
		}
	}
	return tileCoded
}

func VecClip(a *mat.VecDense, min, max float64) {
	for i := 0; i < a.Len(); i++ {
		value := a.AtVec(i)

		if value < min {
			a.SetVec(i, min)
		} else if value > max {
			a.SetVec(i, max)
		}
	}
}

func VecFloor(a *mat.VecDense, b float64) {
	for i := 0; i < a.Len(); i++ {
		mod := math.Floor(a.AtVec(i) / b)
		a.SetVec(i, mod)
	}
}

func (t TileCoder) Encode(v mat.Vector) *mat.VecDense {
	tileCoded := mat.NewVecDense(t.VecLength(), nil)

	for j := 0; j < t.numTilings; j++ {
		indexOffset := j * t.tilingLength
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
	return t.numTilings * t.tilingLength
}

func prod(i []int) int {
	prod := 1
	for _, v := range i {
		prod *= v
	}
	return prod
}
