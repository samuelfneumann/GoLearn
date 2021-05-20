package mountaincar

import (
	"sfneuman.com/golearn/timestep"
)

// GoalEnder implements functionality for ending an episode of the
// MountainCar environment once a goal position has been reached
type GoalEnder struct {
	goalX float64
}

// NewGoalEnder creates and returns a new GoalEnder given a goal x
// position
func NewGoalEnder(goalX float64) *GoalEnder {
	return &GoalEnder{goalX}
}

// End determines if a timestep is the last timestep in an episode by
// checking if the current state of the environment is the goal state.
// If so, End() changes the TimeStep's StepType to timestep.Last. This
// function returns true if the argument timestep is the last timestep
// in the episode and false otherwise.
func (g *GoalEnder) End(t *timestep.TimeStep) bool {
	if t.Observation.AtVec(0) >= g.goalX {
		t.StepType = timestep.Last
		return true
	}
	return false
}
