// Package weights defines weight initializations for linear agents
package weights

import "gonum.org/v1/gonum/mat"

// Initializer initializes weights
type Initializer interface {
	Initialize(weights *mat.Dense) // initializes weights
}
