package environment

import "sfneuman.com/golearn/timestep"

// StepLimit implements the Ender interface to end episodes at specific
// timestep limits
type StepLimit struct {
	episodeSteps int
}

// NewStepLimit creates and returns a new step limit
func NewStepLimit(episodeSteps int) StepLimit {
	return StepLimit{episodeSteps}
}

// End determines whether or not the current episode should be ended,
// returning a boolean to indicate episode temrination. If the episode
// should be ended End() will modify the timestep so that its StepType
// field is timestep.Last
func (s StepLimit) End(t *timestep.TimeStep) bool {
	if t.Number >= s.episodeSteps {
		t.StepType = timestep.Last
		return true
	}
	return false
}
