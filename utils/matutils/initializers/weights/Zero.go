package weights

import "fmt"

// ZeroUV implements the distuv.Rander interface so that zero
// initialization can be accomplished thorugh the weight initialization
// structs which take a distuv.Rander argument
type ZeroUV struct{}

// NewZeroUV returns a new ZeroUV
func NewZeroUV() ZeroUV {
	return ZeroUV{}
}

// Rand draws a random number from the interval [0, 0]
func (z ZeroUV) Rand() float64 {
	return 0.0
}

// ZeroMV implements the distmv.Rander interface so that zero initialization
// can be accomplished through the weight initialization structs which take
// a distmv.Rander argument
type ZeroMV struct {
	size int
}

// NewZero creates and returns a new *Zero
func NewZeroMV(weights []float64) *ZeroMV {
	return &ZeroMV{len(weights)}
}

// Rand gets and returns a slice of 0's. If x is nil, then the function
// creates a new slice of the appropriate size and fills it with 0's.
// The function panics if the size of the argument slice does not equal
// the expected size that the Zero struct was initialized with.
func (z *ZeroMV) Rand(x []float64) []float64 {
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
