package mountaincar

import (
	"fmt"
	"math"
	"os"
	"strings"

	"github.com/samuelfneumann/golearn/environment"
	env "github.com/samuelfneumann/golearn/environment"
	ts "github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/golearn/utils/floatutils"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
)

const (
	MinPosition     float64 = -1.2
	MaxPosition     float64 = 0.6
	MaxSpeed        float64 = 0.07
	Power           float64 = 0.0015 // Engine power
	Gravity         float64 = 0.0025
	ActionDims      int     = 1
	ObservationDims int     = 2

	// Discrete Actions Env
	MinDiscreteAction int = 0
	MaxDiscreteAction int = 2

	// Continuous Actions Env
	MinContinuousAction float64 = -1.0
	MaxContinuousAction float64 = 1.0
)

// base implements the classic control Mountain Car environment. In this
// environment, the agent controls a car in a valley between two hills.
// The car is underpowered and cannot drive up the hill unless it rocks
// back and forth from hill to hill, using its momentum to gradually
// climb higher.
//
// State features consist of the x position of the car and its velocity.
// These features are bounded by the MinPosition, MaxPosition, and
// MaxSpeed constants defined in this package. The sign of the velocity
// feature denotes direction, with negative meaning that the car is
// travelling left and positive meaning that the car is travelling
// right. Upon reaching the minimum or maximum position, the velocity
// of the car is set to 0.
//
// Actions determine the force to apply to the car and in which
// direction to apply this force. Actions may be discrete or continuous.
// Environments that deal with discrete and continuous actions are the
// public mountaincar.Discrete and mountaincar.Continuous structs
// reenvironmenttively. These are the only public Mountain Car environments
// that can be used.
//
// base does not implement the environment.Environment interface
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

// CurrentTimeStep returns the last TimeStep that occurred in the
// environment
func (b *base) CurrentTimeStep() ts.TimeStep {
	return b.lastStep
}

// ObservationSpec returns the observation environmentification of the
// environment
func (m *base) ObservationSpec() environment.Spec {
	shape := mat.NewVecDense(ObservationDims, nil)
	lowerBound := mat.NewVecDense(ObservationDims, []float64{m.positionBounds.Min,
		m.speedBounds.Min})
	upperBound := mat.NewVecDense(ObservationDims, []float64{m.positionBounds.Max,
		m.speedBounds.Max})

	return environment.NewSpec(shape, environment.Observation, lowerBound,
		upperBound, environment.Continuous)
}

// DiscountSpec returns the discounting environmentification of the environment
func (m *base) DiscountSpec() environment.Spec {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{m.discount})
	upperBound := mat.NewVecDense(1, []float64{m.discount})

	return environment.NewSpec(shape, environment.Discount, lowerBound,
		upperBound, environment.Continuous)
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
func (m *base) nextState(force float64) *mat.VecDense {
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

// update calculates the next TimeStep in the environment given an
// action and the next state of the environment. This function then
// saves this TimeStep as the current step in the environment.
//
// This funciton is used so that the discrete and continuous action
// versions of Mountain Car can be deal with uniformly. Each calculates
// the force to apply and calls this struct's nextState() function.
// The result of that function is then passed to this function as
// well as the action taken, which is needed to calculate the reward
// for the action.
func (m *base) update(action, newState *mat.VecDense) (ts.TimeStep, bool) {
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
