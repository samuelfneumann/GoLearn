// Package gridworld implements 2D gridworld environments
package gridworld

// TODO: Starting positions should be a vector of positions, with a randomly
// chosen position each time Reset() is called

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/matutils"
)

type SingleStart struct {
	state mat.Vector
	r, c  int
}

func NewSingleStart(x, y, r, c int) (environment.Starter, error) {
	if x > c {
		return &SingleStart{}, fmt.Errorf("x = %d > cols = %d", x, c)
	} else if y > r {
		return &SingleStart{}, fmt.Errorf("y = %d > cols = %d", y, c)
	}

	start := cToV(x, y, r, c)
	return &SingleStart{start, r, c}, nil
}

func (s *SingleStart) Start() mat.Vector {
	return s.state
}

// GridWorld represents a gridworld environment
//
// A gridworld is represented as a flattened matrix, but in this implementation
// only the matrix dimensions and current agent position are tracked
type GridWorld struct {
	environment.Task
	environment.Starter
	r, c     int
	position int // *mat.VecDense // current position
	// start       int
	discount    float64
	currentStep timestep.TimeStep
}

// Dims gets the rows and columns of the GridWorld
func (g *GridWorld) Dims() (r, c int) {
	return g.r, g.c
}

// At checks the value at position (i, j) in the gridworld. A value of 1.0
// indicates that the agent is at position (i, j).
func (g *GridWorld) At(i, j int) float64 {
	if (i*g.c)+j == g.position {
		return 1.0
	}
	return 0.0
}

// New creates a new gridworld with starting position (x, y), r rows, and c
// columns, task t, and discount factor discount
func New(x, y, r, c int, t environment.Task, d float64, s environment.Starter) (*GridWorld, timestep.TimeStep) {
	// Set the starting position
	start := s.Start()
	startInd := cToInd(x, y, c)

	startStep := timestep.New(timestep.First, 0.0, d, start, 0)

	g := &GridWorld{t, s, r, c, startInd, d, startStep}

	return g, g.Reset()
}

func (g *GridWorld) Reset() timestep.TimeStep {
	startVec := g.Start()
	g.position = g.vToInd(startVec)
	obs := g.getObservation()

	startStep := timestep.New(timestep.First, 0, g.discount, obs, 0)
	g.currentStep = startStep
	return startStep
}

// func (g *GridWorld) Step(action mat.Vector) (timestep.TimeStep, bool) {
// 	if l := action.Len(); l > 0 {
// 		panic(fmt.Sprintf("action dimension too large - want 1, have %d", l))
// 	}

// 	direction := action.AtVec(0)
// 	//Left
// 	if direction == 0 {
// 		if
// 	}
// }

func (g *GridWorld) Step(action mat.Vector) (timestep.TimeStep, bool) {
	direction := action.AtVec(0)
	x, y := g.Coordinates()
	var newPosition mat.Vector

	// Move the current position
	switch direction {
	case 0: // Left
		if newX := x - 1; newX < 0 {
			newPosition = g.cToV(x, y)
		} else {
			newPosition = g.cToV(newX, y)
		}

	case 1: // Right
		if newX := x + 1; newX >= g.c {
			newPosition = g.cToV(x, y)
		} else {
			newPosition = g.cToV(newX, y)
		}

	case 2: // Up
		if newY := y + 1; newY >= g.r {
			newPosition = g.cToV(x, y)
		} else {
			newPosition = g.cToV(x, newY)
		}

	case 3: // Down
		if newY := y - 1; newY < 0 {
			newPosition = g.cToV(x, y)
		} else {
			newPosition = g.cToV(x, newY)
		}
	}
	g.position = g.vToInd(newPosition)

	// Get information to pass back
	reward := g.GetReward(g.currentStep, action)
	number := g.currentStep.Number + 1
	stepType := timestep.Mid

	// Check if this transition is to the end state
	if g.AtGoal(newPosition) {
		stepType = timestep.Last
	}

	// Set up the next timestep and update the gridworld's current step
	step := timestep.New(stepType, reward, g.discount, newPosition, number)
	g.currentStep = step

	return step, stepType == timestep.Last
}

// cToV converts coordinates (x, y) to a vector
func (g *GridWorld) cToV(x, y int) mat.Vector {
	return cToV(x, y, g.r, g.c)
}

func cToV(x, y, r, c int) mat.Vector {
	vec := mat.NewVecDense(r*c, nil)
	ind := cToInd(x, y, c)
	vec.SetVec(ind, 1.0)
	return vec
}

// vToC converts a one-hot vector into (x, y) coordinates in the GridWorld
func (g *GridWorld) vToC(v mat.Vector) (int, int) {
	return vToC(v, g.r, g.c)
}

// vToC converts a vector representation of a one-hot matrix into the (x, y)
// coordinates of the single 1.0 value in the matrix
func vToC(v mat.Vector, r, c int) (int, int) {
	for i := 0; i < v.Len(); i++ {
		if v.AtVec(i) != 0.0 {
			y := i / c
			x := i - (y * c)
			return x, y
		}
	}
	return -1, -1
}

func (g *GridWorld) cToInd(x, y int) int {
	return cToInd(x, y, g.c)
}

func cToInd(x, y, c int) int {
	return y*c + x
}

func vToInd(v mat.Vector, r, c int) int {
	x, y := vToC(v, r, c)
	return cToInd(x, y, c)
}

func (g *GridWorld) vToInd(v mat.Vector) int {
	return vToInd(v, g.r, g.c)
}

func (g *GridWorld) Coordinates() (int, int) {
	y := (g.position / g.c)
	x := g.position - (y * g.c)
	return x, y
}

func (g *GridWorld) String() string {
	str := "GridWorld | At: %v  |   Goal: %v  |  Bounds: (%d, %d)"
	position := matutils.Format(g.getCoordinates(g.position))

	return fmt.Sprintf(str, position, g.Task, g.r, g.c)
}

func (g *GridWorld) getObservation() *mat.VecDense {
	position := mat.NewVecDense(g.r*g.c, nil)
	position.SetVec(g.position, 1.0)
	return position
}

func (g *GridWorld) getCoordinates(v int) mat.Matrix {
	vec := mat.NewVecDense(g.r*g.c, nil)
	vec.SetVec(v, 1.0)
	coords, err := getCoordinates(vec, g.r, g.c)
	if err != nil {
		msg := fmt.Sprintf("Cannot reshape v: v is not of size %d", g.r*g.c)
		panic(msg)
	}
	return coords
}

// Task types
type Goal struct {
	goals *mat.Dense // one-hot encoding of goal states
	// goals          [][]int
	r, c           int // total rows and columns in environment
	timeStepReward float64
	goalReward     float64
}

func (g *Goal) GetReward(t timestep.TimeStep, a mat.Vector) float64 {
	// fmt.Println("Need to implement Goal.GetReward()")
	obs := t.Observation.(mat.Vector)
	x, y := vToC(obs, g.r, g.c)

	direction := a.AtVec(0)
	var newPosition mat.Vector

	// Move the current position
	switch direction {
	case 0: // Left
		if newX := x - 1; newX < 0 {
			newPosition = cToV(x, y, g.r, g.c)
		} else {
			newPosition = cToV(newX, y, g.r, g.c)
		}

	case 1: // Right
		if newX := x + 1; newX >= g.c {
			newPosition = cToV(x, y, g.r, g.c)
		} else {
			newPosition = cToV(newX, y, g.r, g.c)
		}

	case 2: // Up
		if newY := y + 1; newY >= g.r {
			newPosition = cToV(x, y, g.r, g.c)
		} else {
			newPosition = cToV(x, newY, g.r, g.c)
		}

	case 3: // Down
		if newY := y - 1; newY < 0 {
			newPosition = cToV(x, y, g.r, g.c)
		} else {
			newPosition = cToV(x, newY, g.r, g.c)
		}
	}

	nextX, nextY := vToC(newPosition, g.r, g.c)

	// Get the current coordinates
	numGoals, _ := g.goals.Dims()

	for i := 0; i < numGoals; i++ {
		ind := g.goals.RowView(i)
		goalX := int(ind.AtVec(0))
		goalY := int(ind.AtVec(1))
		if nextX == goalX && nextY == goalY {
			return g.goalReward
		}
	}

	return g.timeStepReward
}

// NewGoal creates and returns a new goal at position (x, y), given that the
// gridworld has r rows and c columns
func NewGoal(x, y []int, r, c int, tr, gr float64) (*Goal, error) {
	if len(x) != len(y) {
		return &Goal{}, fmt.Errorf("X length (%d) != Y length (%d)",
			len(x), len(y))
	}

	goals := mat.NewVecDense(r*c, nil)
	for i := range x {
		// Ensure that the goal is within the proper bounds
		if x[i] > c {
			return &Goal{}, fmt.Errorf("x[%d] = %d > cols = %d", i, x[i], c)
		} else if y[i] > r {
			return &Goal{}, fmt.Errorf("y[%d] = %d > cols = %d", i, y[i], c)
		}

		pos := y[i]*c + x[i]
		goals.SetVec(pos, 1.0)
	}

	goalCoords, err := getCoordinates(goals, r, c)
	if err != nil {
		panic("could not parse rewards")
	}

	return &Goal{goalCoords, r, c, tr, gr}, nil
}

func (g *Goal) String() string {
	// coords, err := getCoordinates(g.goals, g.r, g.c)
	// if err != nil {
	// 	panic("Cannot reshape goals")
	// }
	// return ""
	return matutils.Format(g.goals)
}

func (g *Goal) AtGoal(state mat.Matrix) bool {
	obs := state.(mat.Vector)
	x, y := vToC(obs, g.r, g.c)

	// Get the current coordinates
	numGoals, _ := g.goals.Dims()

	for i := 0; i < numGoals; i++ {
		ind := g.goals.RowView(i)
		goalX := int(ind.AtVec(0))
		goalY := int(ind.AtVec(1))
		if x == goalX && y == goalY {
			return true
		}
	}
	return false
}

// getCoordinates gets the (x, y) coordinates of non-zero elements in a
// VecDense if the VecDense were to be transformed to a matrix of shape (r, c)
func getCoordinates(v *mat.VecDense, r, c int) (*mat.Dense, error) {
	if r*c > v.Len() {
		return nil, fmt.Errorf("Cannot reshape v to shape (%d, %d)", r, c)
	}

	// var x, y []float64
	var coords []float64

	for i := 0; i < r*c; i++ {
		if v.AtVec(i) != 0.0 {
			var row int = i / c
			var col int = i - (row * c)

			// x = append(x, float64(col))
			// y = append(y, float64(row))
			coords = append(coords, float64(col), float64(row))
		}
	}
	// fmt.Println(x, y)
	// coords := append(x, y...)
	positions := mat.NewDense(len(coords)/2, 2, coords)
	return positions, nil
}
