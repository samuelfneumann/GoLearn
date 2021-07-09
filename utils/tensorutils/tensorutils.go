package tensorutils

// Slice implements a struct that can be used for slicing tensors.
//
// Given a tensor T and a Slice S, T.Slice(..., S, ...) is equivalent to
// T[..., S.start:S.end:S.step, ...]
type Slice struct {
	start, end, step int
}

// Start returns the start index for the tensor slice
func (s Slice) Start() int {
	return s.start
}

// End returns the ending index for the tensor slice
func (s Slice) End() int {
	return s.end
}

// Step returns the step for the tensor slice
func (s Slice) Step() int {
	return s.step
}

// NewSlice returns a new Slice that can be used to slice tensors
func NewSlice(start, stop, step int) Slice {
	return Slice{start, stop, step}
}
