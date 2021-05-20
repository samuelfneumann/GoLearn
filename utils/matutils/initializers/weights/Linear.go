package weights

import (
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

// Linear initializes a single linear layer or matrix.
type Linear struct {
	rand distmv.Rander
}

// NewLinear returns a new Linear Initializer, with weights drawn from
// the distribution defined by rand
func NewLinear(rand distmv.Rander) Linear {
	if rand == nil {
		panic("rand cannot be nil")
	}
	return Linear{rand}
}

// Initialize initializes a linear layer of weights
func (l Linear) Initialize(weights *mat.Dense) {
	if weights == nil {
		panic("cannot pass nil weights")
	}
	r, _ := weights.Dims()

	for i := 0; i < r; i++ {
		row := l.rand.Rand(nil)
		weights.SetRow(i, row)
	}
}
