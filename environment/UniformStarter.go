package environment

import (
	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	"gonum.org/v1/gonum/stat/distmv"
)

type UniformStarter struct {
	features int
	seed     uint64
	rand     *distmv.Uniform
}

func NewUniformStarter(bounds []r1.Interval, seed uint64) UniformStarter {
	source := rand.NewSource(seed)
	rand := distmv.NewUniform(bounds, source)

	return UniformStarter{len(bounds), seed, rand}
}

func (u UniformStarter) Start() mat.Vector {
	return mat.NewVecDense(u.features, u.rand.Rand(nil))
}
