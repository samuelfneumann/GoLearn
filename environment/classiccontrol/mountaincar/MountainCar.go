// Package mountaincar implements the discrete action classic control
// environment "Mountain Car"
package mountaincar

import (
	"fmt"
	"math"
	"os"
	"strings"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
)

const (
	MinPosition float64 = -1.2
	MaxPosition float64 = 0.6
	MaxSpeed    float64 = 0.07
	Force       float64 = 0.001
	Gravity     float64 = 0.0025
)

// Mountain Car in a classic control environment where an agent must
// learn to drive an underpowered car up a hill. Actions are discrete in
// (0, 1, 2) where:
//
//	Action	Meaning
//	  0		Accelerate left
//	  1		Do nothing
//	  2		Accelerate right
//
//  Actions other than 0, 1, or 2 result in a panic
//
// When designing a starter for this environment, care should be taken to
// ensure that the starting states are chosen within the environmental
// bounds. If the starter produces a state outside of the position and
// speed bounds, the environment will panic. This may happen near the
// end of training, resulting in significant data loss.
//
// Any taks may be used with the MountainCar environment, but the
// classic control task is defined in the Goal struct, where the agent
// must learn to reach the goal at the top of the hill.
type MountainCar struct {
	env.Task
	positionBounds r1.Interval
	speedBounds    r1.Interval
	lastStep       ts.TimeStep
	discount       float64
	force          float64
	gravity        float64
}

// New creates a new MountainCar environment with the argument task
func New(t env.Task, discount float64) (*MountainCar, ts.TimeStep) {
	positionBounds := r1.Interval{Min: MinPosition, Max: MaxPosition}
	speedBounds := r1.Interval{Min: -MaxSpeed, Max: MaxSpeed}

	state := t.Start()
	validateState(state, positionBounds, speedBounds)

	firstStep := ts.New(ts.First, 0.0, discount, state, 0)

	mountainCar := MountainCar{t, positionBounds, speedBounds, firstStep,
		discount, Force, Gravity}

	return &mountainCar, firstStep

}

// ObservationSpec returns the observation specification of the
// environment
func (m *MountainCar) ObservationSpec() spec.Environment {
	shape := mat.NewVecDense(2, nil)
	lowerBound := mat.NewVecDense(2, []float64{m.positionBounds.Min,
		m.speedBounds.Min})
	upperBound := mat.NewVecDense(2, []float64{m.positionBounds.Max,
		m.speedBounds.Max})

	return spec.NewEnvironment(shape, spec.Observation, lowerBound,
		upperBound, spec.Continuous)

}

// ActionSpec returns the action specification of the environment
func (m *MountainCar) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{0})
	upperBound := mat.NewVecDense(1, []float64{2})

	return spec.NewEnvironment(shape, spec.Action, lowerBound,
		upperBound, spec.Discrete)

}

// DiscountSpec returns the discounting specification of the environment
func (m *MountainCar) DiscountSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{m.discount})
	upperBound := mat.NewVecDense(1, []float64{m.discount})

	return spec.NewEnvironment(shape, spec.Discount, lowerBound,
		upperBound, spec.Continuous)

}

// Reset resets the environment and returns a starting state drawn from
// the environment Starter
func (m *MountainCar) Reset() ts.TimeStep {
	state := m.Start()
	validateState(state, m.positionBounds, m.speedBounds)
	startStep := ts.New(ts.First, 0, m.discount, state, 0)
	m.lastStep = startStep

	return startStep
}

// NextState calculates the next state in the environment given action a
func (m *MountainCar) NextState(a mat.Vector) mat.Vector {
	action := a.AtVec(0)

	// Ensure a legal action was selected
	intAction := int(action)
	if intAction != 0 && intAction != 1 && intAction != 2 {
		panic(fmt.Sprintf("illegal action %v \u2209 (0, 1, 2)", intAction))
	}

	// Get the current state
	state := m.lastStep.Observation
	position, velocity := state.AtVec(0), state.AtVec(1)

	// Update the velocity
	velocity += (action-1.0)*m.force + math.Cos(3*position)*(-m.gravity)
	velocity = math.Min(velocity, m.speedBounds.Max)
	velocity = math.Max(velocity, m.speedBounds.Min)

	// Update the position
	position += velocity
	position = math.Min(position, m.positionBounds.Max)
	position = math.Max(position, m.positionBounds.Min)

	// Ensure position stays within bounds
	if position <= m.positionBounds.Min && velocity < 0 {
		velocity = 0
	}

	// Create the new timestep
	newState := mat.NewVecDense(2, []float64{position, velocity})
	return newState

}

// Step takes one environmental step given action a and returns the next
// state as a timestep.TimeStep and a bool indicating whether or not the
// episode has ended
func (m *MountainCar) Step(a mat.Vector) (ts.TimeStep, bool) {
	// Create the new timestep
	newState := m.NextState(a)
	reward := m.GetReward(m.lastStep.Observation, a, newState)
	nextStep := ts.New(ts.Mid, reward, m.discount, newState,
		m.lastStep.Number+1)

	// Check if the step is the last in the episode and adjust step type
	// if necessary
	m.End(&nextStep)

	m.lastStep = nextStep
	return nextStep, nextStep.Last()

}

// calculateRow calculates what to draw for a single row of text-based
// rendering
func calculateRow(xIndices, width int) string {
	var builder strings.Builder

	// Starting "=" signs
	for i := 0; i < width; i++ {
		fmt.Fprintf(&builder, "=")
	}

	// Spaces
	for i := 0; i < xIndices-(2*width); i++ {
		fmt.Fprintf(&builder, " ")
	}

	// Ending "="
	for i := 0; i < width; i++ {
		fmt.Fprintf(&builder, "=")
	}
	return builder.String()
}

// Render renders a text-based version of the environment
func (m *MountainCar) Render() {
	xIndices := 16

	// Print the hill
	var hill strings.Builder
	for i := 1; i < xIndices/2+1; i++ {
		if i == 1 {
			fmt.Fprint(&hill, calculateRow(xIndices, i)+"🏁\n")
		} else {
			fmt.Fprintln(&hill, calculateRow(xIndices, i))
		}
	}
	fmt.Fprintln(&hill, "")

	// Calculate the x position at which to draw the car
	xPos := m.lastStep.Observation.AtVec(0)
	xPos = (xPos - m.positionBounds.Min) /
		(m.positionBounds.Max - m.positionBounds.Min)
	x := int(xPos * float64(xIndices))

	// Print the position bar
	var builder strings.Builder
	for i := 0; i < xIndices; i++ {
		if i == x {
			fmt.Fprintf(&builder, "🚗")
		} else if i == xIndices-1 {
			fmt.Fprintf(&builder, "🏁")
		} else {
			fmt.Fprintf(&builder, "=")
		}
	}

	// Clear screen and draw
	os.Stdout.WriteString("\x1b[3;J\x1b[H\x1b[2J")
	fmt.Printf("%v%v\n", &hill, &builder)

}

// String returns a string representation of the environment
func (m *MountainCar) String() string {
	str := "Mountain Car  |  Position: %v  |  Speed: %v"
	state := m.lastStep.Observation
	return fmt.Sprintf(str, state.AtVec(0), state.AtVec(1))
}

// validateState validates the state to ensure the position and speed
// are within the environmental limits
func validateState(s mat.Vector, positionBounds,
	speedBounds r1.Interval) {
	position := s.AtVec(0)
	if position < positionBounds.Min || position > positionBounds.Max {
		panic(fmt.Sprintf("illegal position %v \u2209 [%v, %v]", position,
			positionBounds.Min, positionBounds.Max))
	}

	speed := s.AtVec(1)
	if speed < speedBounds.Min || speed > speedBounds.Max {
		panic(fmt.Sprintf("illegal speed %v \u2209 [%v, %v]", speed,
			speedBounds.Min, speedBounds.Max))
	}
}
