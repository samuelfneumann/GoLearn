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
