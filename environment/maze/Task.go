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

	rows int
	cols int

	stepLimit env.Ender
}

// NewGoal returns a new Goal. The goalCol and goalRow arguments
// should have the same length. The cutoff parameter determines the
// number of timesteps before an episode is cut off. The rows and cols
// define the size of the maze.
func NewGoal(starter env.CategoricalStarter, goalCol, goalRow []int,
	rows, cols int, cutoff int) (env.Task, error) {
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
		rows:      rows,
		cols:      cols,
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
func (g *Goal) AtGoal(obs mat.Matrix) bool {
	col, row := g.toCoord(obs.(mat.Vector))

	for i := range g.goalCol {

		goalCol := g.goalCol[i]
		goalRow := g.goalRow[i]

		if row == goalRow && col == goalCol {
			return true
		}
	}
	return false
}

// toCoord converts a one-hot encoding to an (x, y)/(col, row)
// coordinate
func (g *Goal) toCoord(oneHot mat.Vector) (col, row int) {
	var nonZeroInd int
	for i := 0; i < oneHot.Len(); i++ {
		if oneHot.AtVec(i) == 1.0 {
			nonZeroInd = i
			break
		}
	}

	row = nonZeroInd / g.cols
	col = nonZeroInd - (row * g.cols)

	return
}
