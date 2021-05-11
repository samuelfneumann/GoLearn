package weights

import "fmt"

// Zero implements the distmv.Rander interface so that zero initialization
// can be accomplished through the weight initialization structs which take
// a single distmv.Rander argument
type Zero struct {
	size int
}

// NewZero creates and returns a new *Zero
func NewZero(weights []float64) *Zero {
	return &Zero{len(weights)}
}

// Rand gets and returns a slice of 0's. If x is nil, then the function
// creates a new slice of the appropriate size and fills it with 0's.
// The function panics if the size of the argument slice does not equal
// the expected size that the Zero struct was initialized with.
func (z *Zero) Rand(x []float64) []float64 {
	// If nil, create a new slice
	if x == nil {
		x = make([]float64, z.size)
	}

	// If the size is different from the expected size, panic
	if len(x) != z.size {
		panic(fmt.Sprintf("incorrect size \n\twant: %d \n\thave: %d",
			z.size, len(x)))
	}

	// Fill with 0's
	for i := range x {
		x[i] = 0
	}
	return x
}
