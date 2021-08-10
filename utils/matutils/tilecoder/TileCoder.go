// Package tilecoder implements tile coding of vectors
package tilecoder

import (
	"fmt"
	"math"
	"sync"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/gonum/stat/samplemv"

	"github.com/samuelfneumann/golearn/utils/floatutils"
	"github.com/samuelfneumann/golearn/utils/matutils"
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
	numTilings  int
	minDims     mat.Vector
	offsets     []*mat.Dense
	bins        [][]int
	binLengths  [][]float64
	seed        uint64
	includeBias bool

	// Concurrent index encoding
	wait    sync.WaitGroup
	indices chan int
}

// NewTileCoder creates and returns a new TileCoder struct. The minDims
// and maxDims arguments are the bounds on each dimension between which
// tilings will be placed. These arguments should have the same shape
// as vectors which will be tile coded.
//
// The bins argument determines both the number of tilings to use and
// the number of tiles per each tiling. This parameter is a [][]int.
// The number of elements in the outer slice determines the number of
// tilings to use. The sub-slices determine how many tiles are placed
// along each dimension for the respective tiling. For example, if
// bins := [][]int{{2, 2}, {4, 3}}, then the TileCoder uses two tilings.
// The first tiling is a 2x2 tiling. The second tiling uses 4 tiles
// along the first dimension and 3 tiles along the second dimension.
// The number of tiles along each dimension should equal the length of
// the minDims and maxDims parameters. That is, len(bins[i]) ==
// minDims.Len() == maxDims.Len() for any i in [0, len(bins)-1].
//
//The parameter includeBias determines whether or not a
// bias unit is kept as the first unit in the tile coded representation.
func New(minDims, maxDims mat.Vector, bins [][]int,
	seed uint64, includeBias bool) *TileCoder {
	// Error checking
	if minDims.Len() != maxDims.Len() {
		msg := fmt.Sprintf("cannot specify minimum with fewer dimensions "+
			"than maximum: %d != %d", minDims.Len(), maxDims.Len())
		panic(msg)
	}
	if len(bins) == 0 {
		panic("cannot have less than 1 bin per dimension")
	}
	if len(bins[0]) != minDims.Len() {
		msg := fmt.Sprintf("there should be a single number of bins for "+
			"each dimension: \n\thave(%d) \n\twant (%d)", len(bins[0]),
			minDims.Len())
		panic(msg)
	}

	// Calculate the length of bins and the tiling offset bounds
	var bounds []r1.Interval
	numTilings := len(bins)
	binLengths := make([][]float64, numTilings)

	for j := 0; j < numTilings; j++ {
		tilingBinLengths := make([]float64, minDims.Len())
		binLengths[j] = tilingBinLengths

		for i := 0; i < minDims.Len(); i++ {
			// Calculate the length of bins
			binLength := (maxDims.AtVec(i) - minDims.AtVec(i))
			binLength /= float64(bins[j][i])
			bound := binLength / OffsetDiv // Bounds tiling offsets

			binLengths[j][i] = binLength
			bounds = append(bounds, r1.Interval{Min: -bound, Max: bound})
		}
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

	// Channel along which encoded indices are sent
	indices := make(chan int, numTilings)

	return &TileCoder{numTilings, minDims, offsets, bins, binLengths, seed,
		includeBias, sync.WaitGroup{}, indices}
}

// Calculates how many features exist in the tile-coded representation
// before tiling number i
func (t *TileCoder) featuresBeforeTiling(i int) int {
	features := 0
	for j := 0; j < i; j++ {
		features += prod(t.bins[j])
	}
	return features
}

// EncodeBatch encodes a batch of vectors held in a Dense matrix. In
// this batch, each row should be a sequential feature, while each
// column should be a sequential sample in the batch. This function
// returns a new matrix which holds the tile coded representation of
// each vector in the batch. The returned matrix is of the size
// k x c, where k is the number of features in the tile coded
// representation and c is the number of samples in the batch (the
// number of columns in the input matrix)
func (t *TileCoder) EncodeBatch(b *mat.Dense) *mat.Dense {
	bias := 0
	if t.includeBias {
		bias = 1
	}

	rows, cols := b.Dims()
	tileCoded := mat.NewDense(t.VecLength(), cols, nil)

	// A vector of 1.0's will be needed for calculations later
	ones := matutils.VecOnes(rows)

	// Vector that holds all the data that is manipulated
	data := mat.NewVecDense(rows, nil)

	// Tile code for each sequential tiling
	for j := 0; j < t.numTilings; j++ {
		// indexOffset is the index into the tile-coded vector at which
		// the current tiling will start
		indexOffset := t.featuresBeforeTiling(j)
		index := mat.NewVecDense(rows, nil)

		// Tile code the batch based on the current tiling
		// We loop through each feature to calculate the tile index to
		// set to 1.0
		for i := len(t.bins[j]) - 1; i > -1; i-- {
			// Clone the next batch of features into the data vector
			data.CloneFromVec(b.RowView(i))

			// Offset the tiling
			data.AddScaledVec(data, t.offsets[j].At(0, i), ones)

			// Calculate which tile each feature is in along the current
			// dimension. Subtracting the minimum dimension will ensure that
			// the data is between [0, 1] before multiplying by the bin
			// length in VecFloor. The integer value of this * binLength
			// is the tile along the current dimension that the feature is in:
			//
			// binLengths[i] = max - min / binLength
			// (data - min) / ((max - min) / binLength) =
			// = ((data - min) / (max - min)) * binLength = IND
			// int(IND) == index into tiling along current dimension
			data.AddScaledVec(data, -t.minDims.AtVec(i), ones)
			matutils.VecFloor(data, t.binLengths[j][i])

			// If out-of-bounds, use the last tile
			matutils.VecClip(data, 0.0, float64(t.bins[j][i]-1))

			// Calculate the index into the tile-coded representation
			// that should be 1.0 for this tiling
			if i == len(t.bins[j])-1 {
				index.AddVec(index, data)
			} else {
				index.AddScaledVec(index, float64(t.bins[j][i+1]), data)
			}
		}

		// Set the proper 1.0 values
		for i := 0; i < index.Len(); i++ {
			// Offset the 1.0 based on which tiling was used for the previous
			// iteration of coding
			row := indexOffset + int(index.AtVec(i)) + bias

			tileCoded.Set(row, i, 1.0)
		}
	}
	if t.includeBias {
		biasUnits := make([]float64, cols)
		for i := 0; i < cols; i++ {
			biasUnits[i] = 1.0
		}
		tileCoded.SetRow(0, biasUnits)
	}
	return tileCoded
}

// encodeWithTiling returns the index of the tile coded feature vector
// which should be a 1.0 when the input vector v is encoded with tiling
// number tiling in the TileCoder.
func (t *TileCoder) encodeWithTiling(v mat.Vector, tiling int) int {
	bias := 0
	if t.includeBias {
		bias = 1
	}

	// indexOffset is the index into the tile-coded vector at which
	// the current tiling will start
	indexOffset := t.featuresBeforeTiling(tiling)
	index := 0

	// Tile code the vector based on the current tiling
	// We loop through each feature to calculate the tile index to
	// set to 1.0 along this feature dimension
	for i := len(t.bins[tiling]) - 1; i > -1; i-- {
		// Offset the tiling
		data := v.AtVec(i) + t.offsets[tiling].At(0, i)

		// Calculate the index of the tile along the current feature
		// dimension in which the feature falls
		tile := math.Floor((data - t.minDims.AtVec(i)) / t.binLengths[tiling][i])

		// Clip tile to within tiling bounds
		tile = floatutils.Clip(tile, 0.0, float64(t.bins[tiling][i]-1))

		// Calculate the index into the tile-coded representation
		// that should be 1.0 for this tiling
		tileIndex := int(tile)
		if i == len(t.bins[tiling])-1 {
			index += tileIndex
		} else {
			index += tileIndex * t.bins[tiling][i+1]
		}
	}
	return indexOffset + index + bias
}

// EncodedIndices returns a slice of the non-zero indices in the tile
// coded vector when v is tile coded with the receiving TileCoder t.
func (t *TileCoder) EncodeIndices(v mat.Vector) []float64 {
	// Check if using a bias unit
	bias := 0
	if t.includeBias {
		bias = 1
	}

	// Create the slice of non-zero indices
	indices := make([]float64, t.numTilings+bias)

	// Listen on the indices channel for indices to set non-zero
	t.wait.Add(1)
	go func() {
		for i := 0; i < t.numTilings; i++ {
			index := float64(<-t.indices)
			indices[i] = index
		}
		t.wait.Done()
	}()

	// Concurrently calculate the non-zero indices for each tiling
	t.wait.Add(t.numTilings)
	for i := 0; i < t.numTilings; i++ {
		go func(tiling int) {
			t.indices <- t.encodeWithTiling(v, tiling)
			t.wait.Done()
		}(i)
	}

	// If using a bias unit, add its index to the list of non-zero indices
	if t.includeBias {
		indices[len(indices)-1] = 0.0
	}

	// Ensure all goroutines have finished adding non-zero indices to
	// the indices slice before returning
	t.wait.Wait()
	return indices
}

// ToVector converts a vector of non-zero indices to a tile-coded
// vector
func (t *TileCoder) ToVector(v mat.Vector) *mat.VecDense {
	tileCoded := mat.NewVecDense(t.VecLength(), nil)
	for i := 0; i < v.Len(); i++ {
		tileCoded.SetVec(int(v.AtVec(i)), 1.0)
	}
	return tileCoded
}

// ToIndices converts a tile-coded vector to a vector of non-zero
// indices
func (t *TileCoder) ToIndices(v mat.Vector) *mat.VecDense {
	indices := make([]float64, 0, t.NumTilings())
	for i := 0; i < v.Len(); i++ {
		if v.AtVec(i) != 0.0 {
			indices = append(indices, float64(i))
		} else if v.AtVec(i) != 1.0 {
			panic("toIndices: vector is not a tile-coded vector")
		}
	}
	return mat.NewVecDense(t.NumTilings(), indices)
}

// Encode encodes a single vector as a tile-coded vector
func (t *TileCoder) Encode(v mat.Vector) *mat.VecDense {
	tileCoded := mat.NewVecDense(t.VecLength(), nil)

	for _, index := range t.EncodeIndices(v) {
		tileCoded.SetVec(int(index), 1.0)
	}
	return tileCoded
}

// String returns a string representation of a *TileCoder
func (t *TileCoder) String() string {
	return fmt.Sprintf("Tilings %d  |  Tiles: %v", t.numTilings, t.bins)
}

// VecLength returns the number of features in a tile-coded vector
func (t *TileCoder) VecLength() int {
	baseVec := 0
	for i := 0; i < t.numTilings; i++ {
		baseVec += prod(t.bins[i])
	}
	if t.includeBias {
		return baseVec + 1
	}
	return baseVec
}

// NumTilings returns the number of tilings the tile coder uses for
// encoding vectors
func (t *TileCoder) NumTilings() int {
	return t.numTilings
}

// prod calculates the product of all integers in a []int
func prod(i []int) int {
	prod := 1
	for _, v := range i {
		prod *= v
	}
	return prod
}
