// Package timestep implements timesteps of the agent-environment interaction
package timestep

import "gonum.org/v1/gonum/mat"

// StepType denotes the type of step that a TimeStep can be, either  first
// environmental step, a middle step, or a last step
type StepType int

const (
	First StepType = iota
	Mid
	Last
)

// TimeStep packages together a single timestep in an environment
type TimeStep struct {
	stepType    StepType
	Reward      float64
	Discount    float64
	Observation mat.Matrix
}

func New(t StepType, r, d float64, o mat.Matrix) TimeStep {
	return TimeStep{t, r, d, o}
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
