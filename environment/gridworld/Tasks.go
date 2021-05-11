package gridworld

import (
	"fmt"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/matutils"
)

// Goal represents the task of reaching goal states in a GridWorld
type Goal struct {
	goals *mat.Dense // one-hot encoding of goal states
	// goals          [][]int
	r, c           int // total rows and columns in environment
	timeStepReward float64
	goalReward     float64
}

// GetReward returns the reward for the current state and action
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
		return &Goal{}, fmt.Errorf("x length (%d) != y length (%d)",
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

// String returns the Goal as a string
func (g *Goal) String() string {
	return matutils.Format(g.goals)
}

// AtGoal represents if the goal state has been reached or not
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
		return nil, fmt.Errorf("cannot reshape v to shape (%d, %d)", r, c)
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

// Min returns the minimum reward attainable in the Task
func (g *Goal) Min() float64 {
	rewards := []float64{g.timeStepReward, g.goalReward}
	return floats.Min(rewards)
}

// Max returns the maximum reward attainable in the Task
func (g *Goal) Max() float64 {
	rewards := []float64{g.timeStepReward, g.goalReward}
	return floats.Max(rewards)
}
