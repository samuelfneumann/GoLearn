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
	"sfneuman.com/golearn/utils/floatutils"
)

const (
	MinPosition float64 = -1.2
	MaxPosition float64 = 0.6
	MaxSpeed    float64 = 0.07
	Power       float64 = 0.0015 // Engine power
	Gravity     float64 = 0.0025

	// Discrete Actions Env
	MinDiscreteAction int = 0
	MaxDiscreteAction int = 2

	// Continuous Actions Env
	MinContinuousAction float64 = -1.0
	MaxContinuousAction float64 = 1.0
)

// base implements the underlying Mountain Car environment. It tracks
// all the needed physical and environmental variables, but does not
// compute next states given actions. The Discrete and Continuous
// structs each embed a base environment and calculate the next states
// from actions. This class is only used to track the Task and current state.
//
// Note that this struct does not implement the environment.Environment
// interface and is used only to unify the Discrete and Continuous
// action versions of Mountain Car by storing and updating variables
// that are common to both environment. This reduces code duplication
// between the two environments.
//
// In Mountain Car, the environment state is continuous and consists of
// the car's x position and velocity. The x position and velocity are
// bounded by the constants defined in this package.
type base struct {
	env.Task
	positionBounds r1.Interval
	speedBounds    r1.Interval
	lastStep       ts.TimeStep
	discount       float64
	power          float64
	gravity        float64
}

// newBase creates a new base environment with the argument task
func newBase(t env.Task, discount float64) (*base, ts.TimeStep) {
	positionBounds := r1.Interval{Min: MinPosition, Max: MaxPosition}
	speedBounds := r1.Interval{Min: -MaxSpeed, Max: MaxSpeed}

	state := t.Start()
	validateState(state, positionBounds, speedBounds)

	firstStep := ts.New(ts.First, 0.0, discount, state, 0)

	mountainCar := base{t, positionBounds, speedBounds, firstStep,
		discount, Power, Gravity}

	return &mountainCar, firstStep

}

// ObservationSpec returns the observation specification of the
// environment
func (m *base) ObservationSpec() spec.Environment {
	shape := mat.NewVecDense(2, nil)
	lowerBound := mat.NewVecDense(2, []float64{m.positionBounds.Min,
		m.speedBounds.Min})
	upperBound := mat.NewVecDense(2, []float64{m.positionBounds.Max,
		m.speedBounds.Max})

	return spec.NewEnvironment(shape, spec.Observation, lowerBound,
		upperBound, spec.Continuous)
}

// DiscountSpec returns the discounting specification of the environment
func (m *base) DiscountSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{m.discount})
	upperBound := mat.NewVecDense(1, []float64{m.discount})

	return spec.NewEnvironment(shape, spec.Discount, lowerBound,
		upperBound, spec.Continuous)
}

// Reset resets the environment and returns a starting state drawn from
// the environment Starter
func (m *base) Reset() ts.TimeStep {
	state := m.Start()
	validateState(state, m.positionBounds, m.speedBounds)
	startStep := ts.New(ts.First, 0, m.discount, state, 0)
	m.lastStep = startStep

	return startStep
}

// NextState calculates the next state in the environment given action a
func (m *base) nextState(force float64) mat.Vector {
	// Get the current state
	state := m.lastStep.Observation
	position, velocity := state.AtVec(0), state.AtVec(1)

	// Update the velocity
	velocity += force*m.power - m.gravity*math.Cos(3*position)
	velocity = floatutils.Clip(velocity, m.speedBounds.Min, m.speedBounds.Max)

	// Update the position
	position += velocity
	position = floatutils.Clip(position, m.positionBounds.Min,
		m.positionBounds.Max)

	// Ensure position stays within bounds
	if position <= m.positionBounds.Min && velocity < 0 {
		velocity = 0
	}

	// Create the new timestep
	newState := mat.NewVecDense(2, []float64{position, velocity})
	return newState

}

// update updates the base environment to change the last state to newState.
// This function also checks whether or not a TimeStep is the last in the
// episode, adjusting it accordingly. This funciton also calculates the
// reward for the previous state and given action as defined by the
// Task. This function returns the next TimeStep and whether or not this
// TimeStep is the last in the episode.
//
// This function updates the underlying variables which are common to
// both the Discrete and Continuous action versions of Mountain Car.
func (m *base) update(action, newState mat.Vector) (ts.TimeStep, bool) {
	// Create the new timestep
	reward := m.GetReward(m.lastStep.Observation, action, newState)
	nextStep := ts.New(ts.Mid, reward, m.discount, newState,
		m.lastStep.Number+1)

	// Check if the step is the last in the episode and adjust step type
	// if necessary
	m.End(&nextStep)

	m.lastStep = nextStep
	return nextStep, nextStep.Last()

}

// Render renders a text-based version of the environment
func (m *base) Render() {
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
func (m *base) String() string {
	str := "Mountain Car  |  Position: %v  |  Speed: %v"
	state := m.lastStep.Observation
	return fmt.Sprintf(str, state.AtVec(0), state.AtVec(1))
}

// calculateRow calculates what to draw for a single row of text-based
// rendering of the hill in Mountain Car
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
