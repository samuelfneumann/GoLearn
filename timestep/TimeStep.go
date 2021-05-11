// Package timestep implements timesteps of the agent-environment interaction
package timestep

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// StepType denotes the type of step that a TimeStep can be, either  first
// environmental step, a middle step, or a last step
type StepType int

const (
	First StepType = iota
	Mid
	Last
)

// String converts a StepType into a string representation
func (s StepType) String() string {
	switch s {
	case First:
		return "First"
	case Last:
		return "Last"
	default:
		return "Mid"
	}
}

// TimeStep packages together a single timestep in an environment
type TimeStep struct {
	stepType    StepType
	Reward      float64
	Discount    float64
	Observation mat.Vector
	Number      int
}

// New constructs a new TimeStep
func New(t StepType, r, d float64, o mat.Vector, n int) TimeStep {
	return TimeStep{t, r, d, o, n}
}

// First returns whether a TimeStep is the first in an environment
func (t *TimeStep) First() bool {
	return t.stepType == First
}

// Mid returns whether a TimeStep is a middle step in an environment
func (t *TimeStep) Mid() bool {
	return t.stepType == Mid
}

// Last returns whether a TimeStep is the last step in an environment
func (t *TimeStep) Last() bool {
	return t.stepType == Last
}

// String converts a TimeStep into its string representation
func (t TimeStep) String() string {
	str := "TimeStep | Type: %v  |  Reward:  %.2f  |  Discount: %.2f  |  " +
		"Step Number:  %v"

	return fmt.Sprintf(str, t.stepType, t.Reward, t.Discount, t.Number)
}
