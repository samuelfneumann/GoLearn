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

// EndType denotes how the last episode ended. The episode could end
// due to a time limit or due to reaching the goal.
type EndType string

const (
	// Terminal state (either goal, out-of-bounds, or ... reached)
	TerminalStateReached EndType = "terminal"

	// Episode was cutoff due to a timeout
	Timeout EndType = "timeout"

	// TimeStep was not last TimeStep or unspecified ending type
	nilEnd = "nil"
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

// Transition packages together a SARSA tuple (S_{t}, A_{t}, R_{t+1},
// S_{t+1}, A_{t+1})
type Transition struct {
	State      *mat.VecDense
	Action     *mat.VecDense
	Reward     float64
	Discount   float64
	NextState  *mat.VecDense
	NextAction *mat.VecDense
}

// NewTransition creates and returns a new transition struct
func NewTransition(step TimeStep, action *mat.VecDense, nextStep TimeStep,
	nextAction *mat.VecDense) Transition {
	state := step.Observation
	reward := nextStep.Reward // reward for the action argument
	discount := nextStep.Discount
	nextState := nextStep.Observation
	return Transition{state, action, reward, discount, nextState, nextAction}
}

// TimeStep packages together a single timestep in an environment.
// Given a SARSA tuple (S_{t}, A_{t}, R_{t+1}, S_{t+1}, A_{t+1}), a
// TimeStep packages together the (R_{t+1}, S_{t+1}) portion, together
// with the discount value, step number, and whether the step is the
// last environmental step.
type TimeStep struct {
	StepType    StepType
	Reward      float64
	Discount    float64
	Observation *mat.VecDense
	Number      int
	EndType
}

// New constructs a new TimeStep
func New(t StepType, r, d float64, o *mat.VecDense, n int) TimeStep {
	return TimeStep{t, r, d, o, n, nilEnd}
}

// SetEnd sets the ending type for the timestep
func (t *TimeStep) SetEnd(e EndType) error {
	if !t.Last() {
		return fmt.Errorf("setEnd: cannot set ending type of non-last " +
			"timestep")
	}
	t.EndType = e
	return nil
}

// TerminalEnd returns whether this timestep was the last timestep
// due to reaching the terminal state.
func (t *TimeStep) TerminalEnd() bool {
	return t.EndType == TerminalStateReached
}

// CutoffEnd returns whether this timestep was the last timestep due to
// an episode being cutoff
func (t *TimeStep) CutoffEnd() bool {
	return t.EndType == Timeout
}

// UnspecifiedEnd returns true if the ending type of the episode is
// unspecified. This may occurr if the TimeStep is not the last in the
// episode, or if it is the last but no end type was specified.
func (t *TimeStep) UnspecifiedEnd() bool {
	return t.EndType == nilEnd
}

// First returns whether a TimeStep is the first in an environment
func (t *TimeStep) First() bool {
	return t.StepType == First
}

// Mid returns whether a TimeStep is a middle step in an environment
func (t *TimeStep) Mid() bool {
	return t.StepType == Mid
}

// Last returns whether a TimeStep is the last step in an environment
func (t *TimeStep) Last() bool {
	return t.StepType == Last
}

// String converts a TimeStep into its string representation
func (t TimeStep) String() string {
	str := "TimeStep | Type: %v  |  Reward:  %.2f  |  Discount: %.2f  |  " +
		"Step Number:  %v"

	return fmt.Sprintf(str, t.StepType, t.Reward, t.Discount, t.Number)
}
