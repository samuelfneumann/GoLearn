// Package maze implements maze environments using GoMaze
package maze

import (
	"fmt"

	env "github.com/samuelfneumann/golearn/environment"
	ts "github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/gomaze"
	"gonum.org/v1/gonum/mat"
)

const (
	DefaultStartRow int = -1
	DefaultStartCol int = -1
	DefaultEndRow   int = -1
	DefaultEndCol   int = -1
)

type Maze struct {
	env.Task
	maze *gomaze.Maze

	discount    float64
	currentStep ts.TimeStep
}

func New(t env.Task, rows, cols int, init gomaze.Initer,
	discount float64) (env.Environment, ts.TimeStep, error) {

	start := t.Start()
	startRow := int(start.AtVec(1))
	startCol := int(start.AtVec(0))

	maze, err := gomaze.NewMaze(rows, cols, DefaultEndRow, DefaultEndCol,
		startRow, startCol, init)
	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("new: could not create maze: %v",
			err)
	}

	task, ok := t.(*Solve)
	if ok {
		task.Register(maze)
	}

	floatState := maze.Reset()
	state := mat.NewVecDense(len(floatState), floatState)
	step := ts.New(ts.First, 0, discount, state, 0)

	mazeEnv := &Maze{
		Task:        t,
		maze:        maze,
		discount:    discount,
		currentStep: step,
	}

	return mazeEnv, step, nil
}

func (m *Maze) Step(action *mat.VecDense) (ts.TimeStep, bool, error) {
	if action.Len() > 1 {
		return ts.TimeStep{}, false, fmt.Errorf("step: actions must be " +
			"1-dimensional")
	}

	a := int(action.AtVec(0))

	newPos, _, _, err := m.maze.Step(a)
	if err != nil {
		return ts.TimeStep{}, false, err
	}
	nextState := mat.NewVecDense(len(newPos), newPos)

	reward := m.GetReward(m.CurrentTimeStep().Observation, action, nextState)
	nextStep := ts.New(ts.Mid, reward, m.discount, nextState,
		m.CurrentTimeStep().Number+1)

	last := m.End(&nextStep)

	return nextStep, last, nil
}

func (m *Maze) Reset() (ts.TimeStep, error) {
	floatState := m.maze.Reset()

	// ! Adjust this for new tasks that don't use GoMaze's underlying goal/start etc.
	// start := m.Start()
	// if start.Len() != 2 {
	// 	return ts.TimeStep{}, fmt.Errorf("oh no")
	// }
	// m.maze.SetCell(int(start.AtVec(0)), int(start.AtVec(1)))
	// state := start
	// ! --------------------------------------------

	state := mat.NewVecDense(len(floatState), floatState)
	step := ts.New(ts.First, 0, m.discount, state, 0)

	m.currentStep = step

	return step, nil
}

func (m *Maze) CurrentTimeStep() ts.TimeStep {
	return m.currentStep
}

func (m *Maze) ActionSpec() env.Spec {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{0.0})
	upperBound := mat.NewVecDense(1, []float64{float64(gomaze.Actions - 1)})

	return env.NewSpec(shape, env.Action, lowerBound, upperBound, env.Discrete)
}

func (m *Maze) ObservationSpec() env.Spec {
	shape := mat.NewVecDense(2, nil)
	lowerBound := mat.NewVecDense(2, []float64{0., 0.})
	upperBound := mat.NewVecDense(1, []float64{
		float64(m.maze.Rows()),
		float64(m.maze.Cols()),
	})

	return env.NewSpec(shape, env.Observation, lowerBound, upperBound,
		env.Discrete)
}

func (m *Maze) DiscountSpec() env.Spec {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{m.discount})

	return env.NewSpec(shape, env.Discount, lowerBound, lowerBound,
		env.Discrete)
}
