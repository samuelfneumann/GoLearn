// Package cartpole implements the Cartpole classic control environment
package cartpole

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/floatutils"
)

const (
	// Physical constants
	Gravity        float64 = 9.8
	CartMass       float64 = 1.0
	PoleMass       float64 = 0.1
	TotalMass      float64 = CartMass + PoleMass
	HalfPoleLength float64 = 0.5  // half of pole length
	ForceMag       float64 = 10.0 // Magnification of force applied
	Dt             float64 = 0.02 // seconds between state updates

	// Bounds (+/-) on state variabels
	PositionBounds        float64 = 4.8
	SpeedBounds           float64 = math.MaxFloat64
	AngleBounds           float64 = math.Pi
	AngularVelocityBounds float64 = math.MaxFloat64

	// Discrete Actions
	MinDiscreteAction int = 0
	MaxDiscreteAction int = 2
)

// Cartpole implements the classic control environment Cartpole. In
// this environment, a pole is attached to a cart, which can move
// horizontally. The agent must get the pole to face straight up for
// as long as possible.
//
// The state features are continuous and consist of the cart's x
// position and speed, as well as the pole's angle from the positive
// y-axis and the pole's angular velocity. All state features are
// bounded by the constants defined in this file.
//
// Actions are discrete and consist of the force applied to the cart:
//
//	Action	Meaning
//	  0		Accelerate left
//	  1		Do nothing
//	  2		Accelerate right
type Cartpole struct {
	env.Task
	lastStep              ts.TimeStep
	discount              float64
	gravity               float64
	forceMag              float64
	poleMass              float64
	halfPoleLength        float64
	cartMass              float64
	dt                    float64
	positionBounds        r1.Interval
	speedBounds           r1.Interval
	angleBounds           r1.Interval
	angularVelocityBounds r1.Interval
}

// New constructs a new Cartpole environment
func New(t env.Task, discount float64) (*Cartpole, ts.TimeStep) {
	positionBounds := r1.Interval{Min: -PositionBounds, Max: PositionBounds}
	speedBounds := r1.Interval{Min: -SpeedBounds, Max: SpeedBounds}
	angleBounds := r1.Interval{Min: -AngleBounds, Max: AngleBounds}
	angularVelocityBounds := r1.Interval{Min: -AngularVelocityBounds,
		Max: AngularVelocityBounds}

	// Get the first state
	state := t.Start()
	validateState(state, positionBounds, speedBounds, angleBounds,
		angularVelocityBounds)

	// Construct first timestep
	firstStep := ts.New(ts.First, 0.0, discount, state, 0)

	cartpole := Cartpole{t, firstStep, discount, Gravity, ForceMag, PoleMass,
		HalfPoleLength, CartMass, Dt, positionBounds, speedBounds, angleBounds,
		angularVelocityBounds}

	return &cartpole, firstStep
}

// Reset resets the environment and returns a starting state drawn from
// the environment Starter
func (c *Cartpole) Reset() ts.TimeStep {
	state := c.Start()
	validateState(state, c.positionBounds, c.speedBounds, c.angleBounds,
		c.angularVelocityBounds)

	startStep := ts.New(ts.First, 0, c.discount, state, 0)
	c.lastStep = startStep

	return startStep

}

// ActionSpec returns the action specification of the environment
func (c *Cartpole) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{float64(MinDiscreteAction)})
	upperBound := mat.NewVecDense(1, []float64{float64(MaxDiscreteAction)})

	return spec.NewEnvironment(shape, spec.Action, lowerBound,
		upperBound, spec.Discrete)
}

// ObservationSpec returns the observation specification of the
// environment
func (c *Cartpole) ObservationSpec() spec.Environment {
	shape := mat.NewVecDense(4, nil)

	lower := []float64{c.positionBounds.Min, c.speedBounds.Min,
		c.angleBounds.Min, c.angularVelocityBounds.Min}
	lowerBound := mat.NewVecDense(4, lower)

	upper := []float64{c.positionBounds.Max, c.speedBounds.Max,
		c.angleBounds.Max, c.angularVelocityBounds.Max}
	upperBound := mat.NewVecDense(4, upper)

	return spec.NewEnvironment(shape, spec.Observation, lowerBound,
		upperBound, spec.Continuous)
}

// DiscountSpec returns the discounting specification of the environment
func (c *Cartpole) DiscountSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{c.discount})
	upperBound := mat.NewVecDense(1, []float64{c.discount})

	return spec.NewEnvironment(shape, spec.Discount, lowerBound,
		upperBound, spec.Continuous)
}

// Step takes one environmental step given action a and returns the next
// state as a timestep.TimeStep and a bool indicating whether or not the
// episode has ended
func (c *Cartpole) Step(a mat.Vector) (ts.TimeStep, bool) {
	// Discrete action in {0, 1, 2}
	action := a.AtVec(0)

	// Ensure a legal action was selected
	intAction := int(action)
	if intAction < MinDiscreteAction || intAction > MaxDiscreteAction {
		panic(fmt.Sprintf("illegal action %v \u2209 (0, 1, 2)", intAction))
	}

	// Get state variables
	state := c.lastStep.Observation
	x, xDot := state.AtVec(0), state.AtVec(1)
	th, thDot := state.AtVec(2), state.AtVec(3)

	// Magnify the action force in the appropriate direction
	var force float64
	if action == 0 {
		force = -c.forceMag
	} else if action == 2 {
		force = c.forceMag
	} else {
		force = 0.0 // No action taken
	}

	// Calculate physical variables to determine next state
	cosTheta := math.Cos(th)
	sinTheta := math.Sin(th)

	totalMass := c.poleMass + c.cartMass
	poleMassOverLength := c.poleMass / c.halfPoleLength

	temp := (force + poleMassOverLength*thDot*thDot*sinTheta) / totalMass
	thAcc := (c.gravity*sinTheta - cosTheta*temp) / (c.halfPoleLength *
		(4.0/3.0 - c.poleMass*cosTheta*cosTheta/totalMass))
	xAcc := temp - poleMassOverLength*thAcc*cosTheta/totalMass

	// Update state variables using Euler kinematic integration
	x += (c.dt * xDot)
	x = floatutils.Clip(x, c.positionBounds.Min, c.positionBounds.Max)

	xDot += (c.dt * xAcc)

	th += (c.dt * thDot)
	th = normalizeAngle(th, c.angleBounds)

	thDot += (c.dt * thAcc)

	// Create the new timestep
	newState := mat.NewVecDense(4, []float64{x, xDot, th, thDot})
	reward := c.GetReward(c.lastStep.Observation, a, newState)
	nextStep := ts.New(ts.Mid, reward, c.discount, newState,
		c.lastStep.Number+1)

	// Check if the step ends the episode
	c.End(&nextStep)

	c.lastStep = nextStep
	return nextStep, nextStep.Last()
}

// validateState ensures that a state observation is valid and between
// the physical bounds of the Cartpole environment
func validateState(obs mat.Vector, positionBounds, speedBounds, angleBounds,
	angularVelocityBounds r1.Interval) {
	// Che if the angle is within bounds
	positionWithinBounds := obs.AtVec(0) <= positionBounds.Max &&
		obs.AtVec(0) >= positionBounds.Min
	if !positionWithinBounds {
		panic(fmt.Sprintf("position is not within bounds %v",
			positionBounds))
	}

	speedWithinBounds := obs.AtVec(1) <= speedBounds.Max &&
		obs.AtVec(0) >= speedBounds.Min
	if !speedWithinBounds {
		panic(fmt.Sprintf("speed is not within bounds %v",
			speedBounds))
	}

	angleWithinBounds := obs.AtVec(2) <= angleBounds.Max &&
		obs.AtVec(0) >= angleBounds.Min
	if !angleWithinBounds {
		panic(fmt.Sprintf("angle is not within bounds %v",
			angleBounds))
	}

	angularVeloyWithinBounds := obs.AtVec(0) <=
		angularVelocityBounds.Max && obs.AtVec(0) >=
		angularVelocityBounds.Min
	if !angularVeloyWithinBounds {
		panic(fmt.Sprintf("angular veloty is not within bounds %v",
			angularVelocityBounds))
	}

}

func (c *Cartpole) String() string {
	msg := "Cartpole  |  Position: %v  | Speed: %v  |  Angle: %v" +
		"  |  Angular Velocity: %v"

	state := c.lastStep.Observation
	position, speed := state.AtVec(0), state.AtVec(1)
	angle, velocity := state.AtVec(2), state.AtVec(3)

	return fmt.Sprintf(msg, position, speed, angle, velocity)
}

// normalizeAngle normalizes the pole angle to the appropriate limits
func normalizeAngle(th float64, angleBounds r1.Interval) float64 {
	if angleBounds.Max != -angleBounds.Min {
		panic("angle bounds should be centered around 0")
	}

	if th > angleBounds.Max {
		divisor := int(th / angleBounds.Max)
		return -math.Pi + th - (angleBounds.Max * float64(divisor))
	} else if th < angleBounds.Min {
		divisor := int(th / angleBounds.Min)
		return math.Pi + th - (angleBounds.Min * float64(divisor))
	} else {
		return th
	}
}
