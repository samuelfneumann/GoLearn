// Package gridworld implements 2D gridworld environments
package gridworld

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/matutils"
)

// GridWorld represents a gridworld environment
type GridWorld struct {
	environment.Task
	r, c     int
	position *mat.VecDense // current position
	start    *mat.VecDense
	discount float64
}

// Dims gets the rows and columns of the GridWorld
func (g *GridWorld) Dims() (r, c int) {
	return g.r, g.c
}

// At checks the value at position (i, j) in the gridworld. A value of 1.0
// indicates that the agent is at position (i, j).
func (g *GridWorld) At(i, j int) float64 {
	return g.position.At(i, j)
}

// New creates a new gridworld with starting position (x, y), r rows, and c
// columns, task t, and discount factor discount
func New(x, y, r, c int, t environment.Task, discount float64) *GridWorld {
	// Set the starting position
	startPos := make([]float64, r*c)
	startPos[c*y+x] = 1.0
	start := mat.NewVecDense(r*c, startPos)

	return &GridWorld{t, r, c, start, start, discount}
}

func (g *GridWorld) Reset() timestep.TimeStep {
	g.position.CloneFromVec(g.start)
	obs := g.getObservation()

	return timestep.New(timestep.First, 0, g.discount, obs)
}

// func (g *GridWorld) Step(action mat.Vector) (timestep.TimeStep, bool) {

// }

func (g *GridWorld) String() string {
	str := "GridWorld | At: %v  |  Start: %v | Goal: %v  |  Bounds: (%d, %d)"
	position := matutils.Format(g.getCoordinates(g.position))
	start := matutils.Format(g.getCoordinates(g.start))

	return fmt.Sprintf(str, position, start, g.Task, g.r, g.c)
}

func (g *GridWorld) getObservation() *mat.VecDense {
	return g.position
}

func (g *GridWorld) getCoordinates(v *mat.VecDense) mat.Matrix {
	coords, err := getCoordinates(v, g.r, g.c)
	if err != nil {
		msg := fmt.Sprintf("Cannot reshape v: v is not of size %d", g.r*g.c)
		panic(msg)
	}
	return coords
}

// Task types
type Goal struct {
	goals *mat.VecDense
	r, c  int // total rows and columns in environment
}

func (g *Goal) GetReward(t timestep.TimeStep, action mat.Vector) float64 {
	fmt.Println("Need to implement Goal.GetReward()")
	return 0.0
}

// NewGoal creates and returns a new goal at position (x, y), given that the
// gridworld has r rows and c columns
func NewGoal(x, y []int, r, c int) (*Goal, error) {
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

	return &Goal{goals, r, c}, nil
}

func (g *Goal) String() string {
	coords, err := getCoordinates(g.goals, g.r, g.c)
	if err != nil {
		panic("Cannot reshape goals")
	}
	// return ""
	return matutils.Format(coords)
}

// getCoordinates gets the (x, y) coordinates of non-zero elements in a
// VecDense if the VecDense were to be transformed to a matrix of shape (r, c)
func getCoordinates(v *mat.VecDense, r, c int) (mat.Matrix, error) {
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
			coords = append(coords, float64(row), float64(col))
		}
	}
	// fmt.Println(x, y)
	// coords := append(x, y...)
	positions := mat.NewDense(len(coords)/2, 2, coords)
	return positions, nil
}
