package environment

import (
	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/timestep"
)

// FunctionEnder ends an episode whenever a function of a vector
// (usually the underlying environment state) returns true.
type FunctionEnder struct {
	end     func(*mat.VecDense) bool
	endType timestep.EndType
}

// NewFunctionEnder returns a new FunctionEnder which ends episodes with
// end type endType when f returns true.
func NewFunctionEnder(f func(*mat.VecDense) bool, endType timestep.EndType) Ender {
	return &FunctionEnder{f, endType}
}

// End determines whether or not the current episode should be ended,
// returning a boolean to indicate episode temrination. If the episode
// should be ended, End() will modify the timestep so that its StepType
// field is timestep.Last and its EndType is the appropriate ending
// type.
func (f *FunctionEnder) End(t *timestep.TimeStep) bool {
	if f.end(t.Observation) {
		t.StepType = timestep.Last
		t.SetEnd(f.endType)
		return true
	}
	return false
}
