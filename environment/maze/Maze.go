// Package maze implements maze environments using GoMaze
package maze

import (
	"fmt"

	env "github.com/samuelfneumann/golearn/environment"
	ts "github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/gomaze"
	"gonum.org/v1/gonum/mat"
)

// Assumes that Start() returns a vector or [row, col]
// State observations are [x, y] == [col, row], where origin is the
// top left cell of the grid

// Maze implements a maze environment.
//
// State observations are 2-dimensional vectors consisting of the
// [x, y] == [col, row] position of the agent. The origin is the top
// left cell. State observations are discrete.
//
// Actions are discrete in the set (0, 1, 2, 3). Actions outside this
// range will cause an error to be returned from Step(). Actions have
// the following meanings:
//
//	Action		Meaning
//	  0			Move north
//	  1			Move south
//	  2			Move west
//	  3			Move east
//
// The Maze environment expects a Task to return a 2-dimensional start
// state using its Start() method. This vector should be of the form
// [x, y] == [col, row].
//
// Maze satisfies the environment.Environment interface.
type Maze struct {
	env.Task
	maze *gomaze.Maze

	discount    float64
	currentStep ts.TimeStep
}

// New returns a new Maze environment.
func New(t env.Task, rows, cols int, init gomaze.Initer,
	discount float64) (env.Environment, ts.TimeStep, error) {
	// Create the underlying GoMaze maze
	maze, err := gomaze.NewMaze(rows, cols, -1, -1, -1, -1, init)
	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("new: could not create maze: %v",
			err)
	}

	// Get a starting state
	start := t.Start()
	err = validateState(rows, cols, start)
	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("new %v", err)
	}
	maze.SetCell(int(start.AtVec(0)), int(start.AtVec(1)))

	// Create the new step and maze environment
	step := ts.New(ts.First, 0, discount, start, 0)

	mazeEnv := &Maze{
		Task:        t,
		maze:        maze,
		discount:    discount,
		currentStep: step,
	}

	return mazeEnv, step, nil
}

// Step takes a single environmental step given some action
func (m *Maze) Step(action *mat.VecDense) (ts.TimeStep, bool, error) {
	if action.Len() > 1 {
		return ts.TimeStep{}, false, fmt.Errorf("step: actions must be " +
			"1-dimensional")
	}

	// Calculate the next position given the action
	a := int(action.AtVec(0))
	newPos, _, _, err := m.maze.Step(a)
	if err != nil {
		return ts.TimeStep{}, false, err
	}
	nextState := mat.NewVecDense(len(newPos), newPos)

	// Construct next timestep
	reward := m.GetReward(m.CurrentTimeStep().Observation, action, nextState)
	nextStep := ts.New(ts.Mid, reward, m.discount, nextState,
		m.CurrentTimeStep().Number+1)
	last := m.End(&nextStep)

	m.currentStep = nextStep
	return nextStep, last, nil
}

// Reset resets the environment to some starting state to begin a new
// episode
func (m *Maze) Reset() (ts.TimeStep, error) {
	_ = m.maze.Reset()

	// Get a starting position
	start := m.Start()
	if err := validateState(m.maze.Rows(), m.maze.Cols(), start); err != nil {
		return ts.TimeStep{}, fmt.Errorf("reset: %v", err)
	}

	// Set the starting position, and construct the first time step
	m.maze.SetCell(int(start.AtVec(0)), int(start.AtVec(1)))
	step := ts.New(ts.First, 0, m.discount, start, 0)

	m.currentStep = step
	return step, nil
}

// CurrentTimeStep returns the current time step of the environment
func (m *Maze) CurrentTimeStep() ts.TimeStep {
	return m.currentStep
}

// ActionSpec returns the action specification of the environment
func (m *Maze) ActionSpec() env.Spec {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{0.0})
	upperBound := mat.NewVecDense(1, []float64{float64(gomaze.Actions - 1)})

	return env.NewSpec(shape, env.Action, lowerBound, upperBound, env.Discrete)
}

// ObservationSpec returns the observation specification of the
// environment
func (m *Maze) ObservationSpec() env.Spec {
	shape := mat.NewVecDense(2, nil)
	lowerBound := mat.NewVecDense(2, []float64{0., 0.})
	upperBound := mat.NewVecDense(2, []float64{
		float64(m.maze.Rows()),
		float64(m.maze.Cols()),
	})

	return env.NewSpec(shape, env.Observation, lowerBound, upperBound,
		env.Discrete)
}

// DiscountSpec returns the discount specification of the environment
func (m *Maze) DiscountSpec() env.Spec {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{m.discount})

	return env.NewSpec(shape, env.Discount, lowerBound, lowerBound,
		env.Discrete)
}

// String returns a string representation of the environment
func (m *Maze) String() string {
	return m.maze.String()
}

// validateState returns an error if the given state is invalid. The
// rows and cols parameters are the number of rows and columns in the
// maze.
func validateState(rows, cols int, state mat.Vector) error {
	if state.Len() != 2 {
		return fmt.Errorf("illegal number of "+
			"start vector dimensions \n\thave(%v) \n\twant(2)", state.Len())
	}
	if startRow := int(state.AtVec(0)); startRow > rows || startRow < 0 {
		return fmt.Errorf("row index out of "+
			"range [%v] with length %v", startRow, rows)
	}
	if startCol := int(state.AtVec(1)); startCol > rows || startCol < 0 {
		return fmt.Errorf("row index out of "+
			"range [%v] with length %v", startCol, rows)
	}

	return nil
}
