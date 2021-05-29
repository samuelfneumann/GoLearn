// Package matutils implements utility function for working with mat.Matrix
// structs
package matutils

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// Format formats a matrix for printing
func Format(X mat.Matrix) string {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	return fmt.Sprintf("%v", fa)
}

// MaxVec finds and returns the index of the maximum value in a vector.
// If multiple equal max values exist, only the first one is returned.
func MaxVec(values mat.Vector) int {
	max, idx := values.AtVec(0), 0
	numActions, _ := values.Dims()

	for i := 0; i < numActions; i++ {
		if values.AtVec(i) > max {
			max = values.AtVec(i)
			idx = i
		}
	}
	return idx
}

// RowMean compute and returns the mean of the rows of a matrix
func RowMean(matrix *mat.Dense) *mat.VecDense {
	r, _ := matrix.Dims()
	rowMeans := make([]float64, r)

	for i := 0; i < r; i++ {
		rowMeans[i] = stat.Mean(matrix.RawRowView(i), nil)
	}
	return mat.NewVecDense(r, rowMeans)
}

// VecClip performs an element-wise clipping of a vector's values such
// that each value is at least min and at most max
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

// VecFloor performs an element-wise floor division of a vector by some
// constant b
func VecFloor(a *mat.VecDense, b float64) {
	for i := 0; i < a.Len(); i++ {
		mod := math.Floor(a.AtVec(i) / b)
		a.SetVec(i, mod)
	}
}

// VecOnes returns a vector of 1.0's
func VecOnes(length int) *mat.VecDense {
	oneSlice := make([]float64, length)
	for i := 0; i < length; i++ {
		oneSlice[i] = 1.0
	}
	return mat.NewVecDense(length, oneSlice)
}
