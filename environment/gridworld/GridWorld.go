// Package gridworld implements 2D gridworld environments
package gridworld

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/matutils"
)

const numActions int = 4

// GridWorld represents a gridworld environment
//
// A gridworld is represented as a flattened matrix, but in this implementation
// only the matrix dimensions and current agent position are tracked
type GridWorld struct {
	environment.Task
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
func New(r, c int, t environment.Task, d float64) (*GridWorld,
	timestep.TimeStep) {
	// Set the starting position
	start := t.Start()
	// startInd := cToInd(x, y, c)
	startInd := vToInd(start, r, c)

	startStep := timestep.New(timestep.First, 0.0, d, start, 0)

	g := &GridWorld{t, r, c, startInd, d, startStep}

	return g, g.Reset()
}

// CurrentTimeStep returns the last TimeStep that occurred in the
// environment
func (g *GridWorld) CurrentTimeStep() timestep.TimeStep {
	return g.currentStep
}

// Reset resets the GridWorld in between episodes. It must explicitly
// be called between episodes.
func (g *GridWorld) Reset() timestep.TimeStep {
	startVec := g.Start()
	g.position = g.vToInd(startVec)
	obs := g.getObservation()

	startStep := timestep.New(timestep.First, 0, g.discount, obs, 0)
	g.currentStep = startStep
	return startStep
}

func (g *GridWorld) NextObs(action *mat.VecDense) *mat.VecDense {
	direction := action.AtVec(0)
	x, y := g.Coordinates()
	var newPosition *mat.VecDense

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

	return newPosition
}

// Step takes an action in the environemnt
func (g *GridWorld) Step(action *mat.VecDense) (timestep.TimeStep, bool) {
	newPosition := g.NextObs(action)
	g.position = g.vToInd(newPosition)

	// Get information to pass back
	reward := g.GetReward(g.currentStep.Observation, action, newPosition)
	number := g.currentStep.Number + 1
	stepType := timestep.Mid

	// Set up the next timestep and update the gridworld's current step
	step := timestep.New(stepType, reward, g.discount, newPosition, number)
	g.End(&step)

	g.currentStep = step

	return step, stepType == timestep.Last
}

// cToV converts coordinates (x, y) to a vector
func (g *GridWorld) cToV(x, y int) *mat.VecDense {
	return cToV(x, y, g.r, g.c)
}

// cToV converts a coordinate (x, y) into a vector
func cToV(x, y, r, c int) *mat.VecDense {
	vec := mat.NewVecDense(r*c, nil)
	ind := cToInd(x, y, c)
	vec.SetVec(ind, 1.0)
	return vec
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

// cToInd converts a coordinate (x, y) into the index of the 1.0 in a
// one-hot vector. The one-hot vector is a vector representation of a
// one-hot matrix with c columns
func cToInd(x, y, c int) int {
	return y*c + x
}

// vToInd gets the index of the 1.0 in a one-hot vector
func vToInd(v mat.Vector, r, c int) int {
	x, y := vToC(v, r, c)
	return cToInd(x, y, c)
}

// vToInd gets the index of the 1.0 in a GridWorld's one-hot vector
// representation
func (g *GridWorld) vToInd(v *mat.VecDense) int {
	return vToInd(v, g.r, g.c)
}

// Coordinates returns the current position in the gridworld as (x, y)
func (g *GridWorld) Coordinates() (int, int) {
	y := (g.position / g.c)
	x := g.position - (y * g.c)
	return x, y
}

// String converts a GridWorld into a string representation
func (g *GridWorld) String() string {
	str := "GridWorld | At: %v  |   Goal: %v  |  Bounds: (%d, %d)"
	position := matutils.Format(g.getCoordinates(g.position))

	return fmt.Sprintf(str, position, g.Task, g.r, g.c)
}

// getObservation returns the current GridWorld as a one-hot vector
// representation
func (g *GridWorld) getObservation() *mat.VecDense {
	position := mat.NewVecDense(g.r*g.c, nil)
	position.SetVec(g.position, 1.0)
	return position
}

// getCoordinates returns the current coordinates (x, y) in the GridWorld
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

// DiscountSpec generates the discount specification for the GridWorld
func (g *GridWorld) DiscountSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)

	min := mat.NewVecDense(1, []float64{g.discount})

	return spec.NewEnvironment(shape, spec.Discount, min, min,
		spec.Continuous)

}

// ObservationSpec generates the observation specification for the
// GridWorld
func (g *GridWorld) ObservationSpec() spec.Environment {
	shape := mat.NewVecDense(g.r*g.c, nil)

	min := mat.NewVecDense(g.r*g.c, nil)
	ones := make([]float64, g.r*g.c)
	for i := range ones {
		ones[i] = 1.0
	}
	max := mat.NewVecDense(g.r*g.c, ones)

	return spec.NewEnvironment(shape, spec.Observation, min, max,
		spec.Discrete)
}

// ActionSpec generates the action specification for the GridWorld
func (g *GridWorld) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)

	min := mat.NewVecDense(1, []float64{0})
	maxAction := float64(numActions - 1)
	max := mat.NewVecDense(1, []float64{maxAction})

	return spec.NewEnvironment(shape, spec.Action, min, max,
		spec.Discrete)
}
