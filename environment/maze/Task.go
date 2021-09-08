package maze

import (
	"fmt"

	env "github.com/samuelfneumann/golearn/environment"
	ts "github.com/samuelfneumann/golearn/timestep"
	"gonum.org/v1/gonum/mat"
)

const (
	TimeStepReward float64 = -1.0
	TerminalReward float64 = 0
)

// Goal implements the task of reaching a goal in the maze. The task
// is solved once the agent reaches (one of) the goal position(s).
//
// The agent gets a TimeStepReward reward on each timestep, and a
// TerminalReward for reaching the goal state. Episodes end when the
// goal state is reached or when a step limit is reached.
type Goal struct {
	env.Starter

	goalCol []int
	goalRow []int

	stepLimit env.Ender
}

// NewGoal returns a new Goal. The goalCol and goalRow arguments
// should have the same length. The cutoff parameter determines the
// number of timesteps before an episode is cut off.
func NewGoal(starter env.CategoricalStarter, goalCol, goalRow []int,
	cutoff int) (env.Task, error) {
	if len(goalCol) != len(goalRow) {
		return nil, fmt.Errorf("newGoal: goal must have same number of " +
			"x positions as y positions")
	}

	stepLimit := env.NewStepLimit(cutoff)
	return &Goal{
		stepLimit: stepLimit,
		Starter:   starter,
		goalCol:   goalCol,
		goalRow:   goalRow,
	}, nil
}

// GetReward returns the reward for a given transition
func (s *Goal) GetReward(_, _, nextState mat.Vector) float64 {
	if s.AtGoal(nextState) {
		return TerminalReward
	}
	return TimeStepReward
}

// End ends a timestep if it is the last in an episode by changing its
// type to timestep.Last and setting its ending type
func (s *Goal) End(t *ts.TimeStep) bool {
	if last := s.stepLimit.End(t); last {
		return last
	}

	if s.AtGoal(t.Observation) {
		t.SetEnd(ts.TerminalStateReached)
		t.StepType = ts.Last
		return true
	}
	return false
}

// AtGoal returns if the argument state is the goal state
func (s *Goal) AtGoal(state mat.Matrix) bool {
	rows, cols := state.Dims()
	if rows != 2 || cols != 1 {
		return false
	}

	for i := range s.goalCol {

		goalCol := s.goalCol[i]
		goalRow := s.goalRow[i]

		if int(state.At(0, 0)) == goalRow && int(state.At(1, 0)) == goalCol {
			return true
		}
	}
	return false
}
