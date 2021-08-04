package tilecoder

import (
	"testing"

	"gonum.org/v1/gonum/mat"
)

func BenchmarkTileCoder(b *testing.B) {
	tc := New(
		mat.NewVecDense(8, []float64{0, 0, 0, 0, 0, 0, 0, 0}),
		mat.NewVecDense(8, []float64{1, 1, 1, 1, 1, 1, 1, 1}),
		[][]int{{8, 8, 8, 8, 8, 8, 8, 8}},
		12,
		true,
	)

	y := mat.NewVecDense(8, []float64{0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5})

	for i := 0; i < b.N; i++ {
		tc.Encode(y)
	}
}
