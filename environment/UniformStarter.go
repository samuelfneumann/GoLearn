package environment

import (
	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	"gonum.org/v1/gonum/stat/distmv"
)

// UniformStarter returns starting states drawn from a (possibly
// multi-dimensional) uniform distribution.
type UniformStarter struct {
	features int
	seed     uint64
	rand     *distmv.Uniform
}

// NewUniformStarter returns a new UniformStarter. This Starter
// returns starting vectors drawn from a (possibly multi-dimensional)
// uniform distribution, with element at index i in the resulting
// starting vector taking values between [bounds[i].Min, bounds[i].Max).
func NewUniformStarter(bounds []r1.Interval, seed uint64) UniformStarter {
	source := rand.NewSource(seed)
	rand := distmv.NewUniform(bounds, source)

	return UniformStarter{len(bounds), seed, rand}
}

// Start returns a new starting vector
func (u UniformStarter) Start() *mat.VecDense {
	return mat.NewVecDense(u.features, u.rand.Rand(nil))
}
