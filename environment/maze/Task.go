package maze

import (
	env "github.com/samuelfneumann/golearn/environment"
	ts "github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/gomaze"
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
	maze *gomaze.Maze

	// goal position [][2]int
	// start position

	stepLimit env.Ender

	registered bool
}

func NewSolve(cutoff int) env.Task {
	stepLimit := env.NewStepLimit(cutoff)
	return &Solve{
		stepLimit: stepLimit,
	}
}

func (s *Solve) Register(m *gomaze.Maze) {
	s.maze = m
	s.registered = true
}

func (s *Solve) Start() *mat.VecDense {
	row, col := s.maze.Start()
	return mat.NewVecDense(2, []float64{
		float64(row),
		float64(col),
	})
}

func (s *Solve) GetReward(_, _, _ mat.Vector) float64 {
	if s.maze.AtGoal() {
		return TerminalReward
	}
	return TimeStepReward
}

func (s *Solve) End(t *ts.TimeStep) bool {
	if last := s.stepLimit.End(t); last {
		return last
	}

	if s.maze.AtGoal() {
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

	goalRow, goalCol := s.maze.Goal()

	return int(state.At(0, 0)) == goalRow && int(state.At(0, 1)) == goalCol
}
