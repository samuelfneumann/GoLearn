package environment

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// CategoricalStarter returns starting states as vectors sampled from
// a multi-dimensional uniform categorical distribution.
type CategoricalStarter struct {
	features      int
	seed          int64
	rng           []*rand.Rand
	startFeatures [][]int
}

// NewCategoricalStarter returns a new CategoricalStarter. The
// element at index i in the resulting starting state vector, will
// have a value drawn uniformly randmly from startFeatures[i].
func NewCategoricalStarter(startFeatures [][]int,
	seed int64) CategoricalStarter {
	source := rand.NewSource(seed)
	rng := make([]*rand.Rand, len(startFeatures))

	for i := range rng {
		// Create the weights for the uniform categorical distribution
		weights := make([]float64, len(startFeatures[i]))
		for j := range weights {
			weights[j] = 1.0 / float64(len(weights))
		}

		rng[i] = rand.New(source)
	}

	return CategoricalStarter{
		features:      len(startFeatures),
		seed:          seed,
		rng:           rng,
		startFeatures: startFeatures,
	}
}

// Start returns a starting state vector
func (c CategoricalStarter) Start() *mat.VecDense {
	start := make([]float64, c.features)

	for i := range start {
		// Get a random index for the current feature's starting value
		ind := c.rng[i].Intn(len(c.startFeatures[i]))

		// Set the current features starting value from the predeclared legal
		// starting values
		start[i] = float64(c.startFeatures[i][ind])
	}

	return mat.NewVecDense(c.features, start)
}
