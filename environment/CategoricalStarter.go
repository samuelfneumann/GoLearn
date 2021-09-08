package environment

import (
	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
)

// Categorical starter returns starting states as vectors sampled from
// a multi-dimensional uniform categorical distribution. The categorical
// distributions sample values in (0, 1, 2, ... N).
type CategoricalStarter struct {
	features int
	seed     uint64
	rand     []distuv.Categorical
}

// NewCategoricalStarter returns a new CategoricalStarter, sampling
// dimension i from (0, 1, 2, ... bounds[i]-1)
func NewCategoricalStarter(bounds []int, seed uint64) CategoricalStarter {
	source := rand.NewSource(seed)

	rand := make([]distuv.Categorical, len(bounds))
	for i := range rand {
		// Create the weights for the uniform categorical distribution
		weights := make([]float64, bounds[i])
		for j := range weights {
			weights[j] = 1.0 / float64(len(weights))
		}

		rand[i] = distuv.NewCategorical(weights, source)
	}

	return CategoricalStarter{len(bounds), seed, rand}
}

// Start returns a starting state vector
func (c CategoricalStarter) Start() *mat.VecDense {
	start := make([]float64, c.features)
	for i := range start {
		start[i] = c.rand[i].Rand()
	}

	return mat.NewVecDense(c.features, start)
}
