// Package cartpole implements the Cartpole classic control environment
package cartpole

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
	ts "sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/floatutils"
)

const (
	// Physical constants
	Gravity         float64 = 9.8
	CartMass        float64 = 1.0
	PoleMass        float64 = 0.1
	TotalMass       float64 = CartMass + PoleMass
	HalfPoleLength  float64 = 0.5  // half of pole length
	ForceMag        float64 = 10.0 // Magnification of force applied
	Dt              float64 = 0.02 // seconds between state updates
	ActionDims      int     = 1
	ObservationDims int     = 4

	// Bounds (+/-) on state variables
	PositionBounds        float64 = 4.8
	SpeedBounds           float64 = 2
	AngleBounds           float64 = math.Pi
	AngularVelocityBounds float64 = 2

	// Discrete Actions
	MinDiscreteAction int = 0
	MaxDiscreteAction int = 2

	// Continuous Actions
	MaxContinuousAction float64 = 1.0
	MinContinuousAction float64 = -MaxContinuousAction
)

// base implements the classic control environment Cartpole. In
// this environment, a pole is attached to a cart, which can move
// horizontally. Gravity pulls the pole downwards so that balancing
// it in an upright position is very difficult.
//
// The state features are continuous and consist of the cart's x
// position and speed, as well as the pole's angle from the positive
// y-axis and the pole's angular velocity. All state features are
// bounded by the constants defined in this package. For the position,
// speed, and angular velocity features, extreme values are clipped to
// within the legal ranges. For the pole's angle feature, extreme values
// are normalized so that all angles stay in the range (-π, π]. Upon
// reaching a position boundary, the velocity of the cart is set to 0.
//
// Actions determine the force to apply to the cart and in which
// direction to apply this force. Actions may be discrete or continuous.
// Environments that deal with discrete and continuous actions are
// the public cartpole.Discrete and cartpole.Continuous structs
// respectively. These are the only public CartPole environments.
//
// base does not implement the environment.Environment interface
type base struct {
	env.Task
	lastStep              ts.TimeStep
	discount              float64
	gravity               float64
	forceMagnification    float64
	poleMass              float64
	halfPoleLength        float64
	cartMass              float64
	dt                    float64
	positionBounds        r1.Interval
	speedBounds           r1.Interval
	angleBounds           r1.Interval
	angularVelocityBounds r1.Interval
}

// New constructs a new base Cartpole environment
func newBase(t env.Task, discount float64) (*base, ts.TimeStep) {
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

	cartpole := base{t, firstStep, discount, Gravity, ForceMag, PoleMass,
		HalfPoleLength, CartMass, Dt, positionBounds, speedBounds, angleBounds,
		angularVelocityBounds}

	return &cartpole, firstStep
}

// LastTimeStep returns the last TimeStep that occurred in the
// environment
func (b *base) LastTimeStep() timestep.TimeStep {
	return b.lastStep
}

// Reset resets the environment and returns a starting state drawn from
// the environment Starter
func (c *base) Reset() ts.TimeStep {
	state := c.Start()
	validateState(state, c.positionBounds, c.speedBounds, c.angleBounds,
		c.angularVelocityBounds)

	startStep := ts.New(ts.First, 0, c.discount, state, 0)
	c.lastStep = startStep

	return startStep

}

// ObservationSpec returns the observation specification of the
// environment
func (c *base) ObservationSpec() spec.Environment {
	shape := mat.NewVecDense(ObservationDims, nil)

	lower := []float64{c.positionBounds.Min, c.speedBounds.Min,
		c.angleBounds.Min, c.angularVelocityBounds.Min}
	lowerBound := mat.NewVecDense(ObservationDims, lower)

	upper := []float64{c.positionBounds.Max, c.speedBounds.Max,
		c.angleBounds.Max, c.angularVelocityBounds.Max}
	upperBound := mat.NewVecDense(ObservationDims, upper)

	return spec.NewEnvironment(shape, spec.Observation, lowerBound,
		upperBound, spec.Continuous)
}

// DiscountSpec returns the discounting specification of the environment
func (c *base) DiscountSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{c.discount})
	upperBound := mat.NewVecDense(1, []float64{c.discount})

	return spec.NewEnvironment(shape, spec.Discount, lowerBound,
		upperBound, spec.Continuous)
}

// nextState calculates the next state of the environment given a force
// to apply to the cart's base. Negative force will cause the cart to
// move left while positive force will move the cart right.
func (c *base) nextState(force float64) mat.Vector {
	// Magnify the force
	force *= c.forceMagnification

	// Get state variables
	state := c.lastStep.Observation
	x, xDot := state.AtVec(0), state.AtVec(1)
	th, thDot := state.AtVec(2), state.AtVec(3)

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
	// Update cart position
	x += (c.dt * xDot)
	x = floatutils.ClipInterval(x, c.positionBounds)

	// Update cart velocity
	xDot += (c.dt * xAcc)
	if x <= c.positionBounds.Min || x >= c.positionBounds.Max {
		// Cart hits the position boundaries, so velocity -> 0
		xDot = 0.0
	}
	xDot = floatutils.ClipInterval(xDot, c.speedBounds)

	// Update pole angle
	th += (c.dt * thDot)
	th = normalizeAngle(th, c.angleBounds)

	// Update pole angle velocity
	thDot += (c.dt * thAcc)
	thDot = floatutils.ClipInterval(thDot, c.angularVelocityBounds)

	// Create the next state
	return mat.NewVecDense(4, []float64{x, xDot, th, thDot})

}

// update calculates the next TimeStep in the environment given an
// action and the next state of the environment. This function then
// saves this TimeStep as the current step in the environment.
//
// This funciton is used so that the discrete and continuous action
// versions of Cartpole can be deal with uniformly. Each calculates
// the force to apply and calls this struct's nextState() function.
// The result of that function is then passed to this function as
// well as the action taken, which is needed to calculate the reward
// for the action.
func (c *base) update(a, newState mat.Vector) (ts.TimeStep, bool) {
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

// String returns the string representation of the environment
func (c *base) String() string {
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
