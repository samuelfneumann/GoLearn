// Package matutils implements utility function for working with mat.Matrix
// structs
package matutils

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// Format formats a matrix for printing
func Format(X mat.Matrix) string {
	fa := mat.Formatted(X, mat.Prefix(""), mat.Squeeze())
	return fmt.Sprintf("%v", fa)
}

// MaxVec finds and returns the index of the maximum value in a vector
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
