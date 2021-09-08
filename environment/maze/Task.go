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

// Todo: ignore goals etc. in GoMaze, and use goals in Solve task
// Then, use a uniform categorical starter or single starter, set a goal position, etc.
// Can have multiple goals like gridworld
type Solve struct {
	env.Starter

	goalCol []int
	goalRow []int

	stepLimit env.Ender
}

func NewSolve(starter env.CategoricalStarter, goalCol, goalRow []int,
	cutoff int) (env.Task, error) {
	if len(goalCol) != len(goalRow) {
		return nil, fmt.Errorf("newSolve: goal must have same number of " +
			"x positions as y positions")
	}

	stepLimit := env.NewStepLimit(cutoff)
	return &Solve{
		stepLimit: stepLimit,
		Starter:   starter,
		goalCol:   goalCol,
		goalRow:   goalRow,
	}, nil
}

func (s *Solve) GetReward(_, _, nextState mat.Vector) float64 {
	if s.AtGoal(nextState) {
		return TerminalReward
	}
	return TimeStepReward
}

func (s *Solve) End(t *ts.TimeStep) bool {
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

func (s *Solve) AtGoal(state mat.Matrix) bool {
	rows, cols := state.Dims()
	if rows != 2 || cols != 1 {
		return false
	}

	for i := range s.goalCol {

		goalCol := s.goalCol[i]
		goalRow := s.goalRow[i]

		if int(state.At(0, 0)) == goalRow && int(state.At(0, 1)) == goalCol {
			return true
		}
	}
	return false
}
