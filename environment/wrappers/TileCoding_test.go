package wrappers

import (
	"testing"
	"time"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	"gonum.org/v1/gonum/stat/distuv"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/environment/box2d/lunarlander"
)

func BenchmarkLunarLander(b *testing.B) {
	seed := uint64(time.Now().UnixNano())
	src := rand.NewSource(seed)
	rng := distuv.Uniform{Min: -1.0, Max: 1.0, Src: src}

	s := environment.NewUniformStarter([]r1.Interval{
		{Min: lunarlander.InitialX, Max: lunarlander.InitialX},
		{Min: lunarlander.InitialY, Max: lunarlander.InitialY},
		{Min: lunarlander.InitialRandom, Max: lunarlander.InitialRandom},
	}, seed)
	task := lunarlander.NewLand(s, 500)
	l, _ := lunarlander.NewDiscrete(task, 0.99, seed)

	tc, _ := NewTileCoding(l, [][]int{{8, 8, 8, 8, 8, 8, 8, 8}}, seed)

	for i := 0; i < b.N; i++ {
		tc.Step(mat.NewVecDense(2, []float64{rng.Rand(), rng.Rand()}))
	}
}
