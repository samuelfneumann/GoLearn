package weights

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/gonum/stat/distuv"
)

// LinearMV initializes a single linear layer or matrix using weights
// drawn from a multivariate distribution. The distribution should have
// as many dimensions as there are columns in the matrix. The
// initializer then initializes all columns of a single row of the
// matrix by  sampling from the distribution. This is repeated for each
// row.
//
// This function assumes that each column corresponds to weights that
// are multiplied by by a single feature. Each row is a new sample of
// weights. For example, if we had 10 features and 4 actions for a
// Q-learning agent, then the weight matrix should be (4 x 10), and
// this function will initialize all 10 weights on a row-by-row basis.
// For a given row, each column will be initialized from some
// distribution which may be different from distributions for
// initialization of subsequent columns in the same row. For a given
// column, all weights in all rows will be initialized from the same
// distribution.
type LinearMV struct {
	distmv.Rander
}

// NewLinear returns a new Linear Initializer, with weights drawn from
// the distribution defined by rand
func NewLinearMV(rand distmv.Rander) LinearMV {
	if rand == nil {
		panic("rand cannot be nil")
	}
	return LinearMV{rand}
}

// Initialize initializes a linear layer of weights
func (l LinearMV) Initialize(weights *mat.Dense) {
	if weights == nil {
		return
	}
	r, _ := weights.Dims()

	for i := 0; i < r; i++ {
		row := l.Rand(nil)
		weights.SetRow(i, row)
	}
}

// LinearUV initializes a single linear layer of weights, drawn from
// a univariate distribution
type LinearUV struct {
	distuv.Rander
}

// NewLinearUV  creates and returns a new LinearUV
func NewLinearUV(rand distuv.Rander) LinearUV {
	if rand == nil {
		panic("rand cannot be nil")
	}
	return LinearUV{rand}
}

// Initialize initializes a matrix of weights using values drawn from
// a univariate distribution
func (l LinearUV) Initialize(weights *mat.Dense) {
	if weights == nil {
		return
	}

	backingData := weights.RawMatrix().Data
	for i := 0; i < len(backingData); i++ {
		backingData[i] = l.Rand()
	}
}
