// Package pendulu, implements the pendulum classic control environment
package pendulum

import (
	"fmt"
	"math"
	"os"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/floatutils"
)

// default physical constants
const (
	AngleBound  float64 = math.Pi // +/- Angle bounds
	SpeedBound  float64 = 8.0     // +/- Speed bounds
	TorqueBound float64 = 2.0     // +/- Torque bounds

	MaxContinuousAction float64 = TorqueBound
	MinContinuousAction float64 = -MaxContinuousAction

	MaxDiscreteAction float64 = 4.0
	MinDiscreteAction float64 = 0.0

	dt              float64 = 0.05
	Gravity         float64 = 9.8
	Mass            float64 = 1.0
	Length          float64 = 1.0
	ActionDims      int     = 1
	ObservationDims int     = 2
)

// TODO: This documentation needs to be updated
// base implements the classic control environment base. In this
// environment, a pendulum is attached to a fixed base. An agent can
// swing the pendulum back and forth, but the swinging force /torque is
// underpowered. In order to be able to swing the pendulum straight up,
// it must first be rocked back and forth, using the momentum to
// gradually climb higher until the pendulum can point straight up or
// rotate fully around its fixed base.
//
// State features consist of the angle of the pendulum from the positive
// y-axis and the angular velocity of the pendulum. Both state features
// are bounded by the AngleBound and SpeedBound constants in this
// package. The sign of the angular velocity or speed indicates
// direction, with negative sign indicating counter clockwise rotation
// and positive sign indicating clockwise direction. The angular
// velocity is clipped betwee [-SpeedBound, SpeedBound]. Angles are
// normalized to stay within [-AngleBound, AngleBound] = [-π, π].
//
// Actions are continuous and 1-dimensional. Actions determine the
// torque to apply to the pendulum at its fixed base. Actions are
// bounded by [-2, 2] = [MinContinuousAction, MaxContinuousAction].
// Actions outside of this region are clipped to stay within these
// bounds.
//
// base implements the environment.Environment interface
type base struct {
	environment.Task
	dt           float64
	gravity      float64
	mass         float64
	length       float64
	angleBounds  r1.Interval
	speedBounds  r1.Interval
	torqueBounds r1.Interval
	lastStep     timestep.TimeStep
	discount     float64
}

// New creates and returns a new base environment
func newBase(t environment.Task, d float64) (*base, timestep.TimeStep) {
	angleBounds := r1.Interval{Min: -AngleBound, Max: AngleBound}
	speedBounds := r1.Interval{Min: -SpeedBound, Max: SpeedBound}
	torqueBounds := r1.Interval{Min: -TorqueBound, Max: TorqueBound}

	state := t.Start()
	validateState(state, angleBounds, speedBounds)

	firstStep := timestep.New(timestep.First, 0.0, d, state, 0)

	pendulum := base{t, dt, Gravity, Mass, Length, angleBounds,
		speedBounds, torqueBounds, firstStep, d}

	return &pendulum, firstStep
}

// LastTimeStep returns the last TimeStep that occurred in the
// environment
func (p *base) LastTimeStep() timestep.TimeStep {
	return p.lastStep
}

// Reset resets the environment and returns a starting state drawn from the
// Starter
func (p *base) Reset() timestep.TimeStep {
	state := p.Start()
	validateState(state, p.angleBounds, p.speedBounds)
	startStep := timestep.New(timestep.First, 0, p.discount, state, 0)
	p.lastStep = startStep

	return startStep
}

// nextState computes the next state of the environment given a timestep and
// an amount of torque to apply to the fixed base of the pendulum. The
// torque is first clipped to the appropriate torque bounds.
func (p *base) nextState(t timestep.TimeStep, torque float64) *mat.VecDense {
	obs := t.Observation
	th, thdot := obs.AtVec(0), obs.AtVec(1)

	// Clip the torque
	torque = floatutils.ClipInterval(torque, p.torqueBounds)

	newthdot := thdot + (-3*p.gravity/(2*p.length)*math.Sin(th+math.Pi)+
		3.0/(p.mass*math.Pow(p.length, 2))*torque)*p.dt

	newth := th + (newthdot * p.dt)

	// Clip the angular velocity
	newthdot = floats.Min([]float64{newthdot, p.speedBounds.Max})
	newthdot = floats.Max([]float64{newthdot, p.speedBounds.Min})

	// Normalize the angle
	newth = normalizeAngle(newth, p.angleBounds)

	newObs := mat.NewVecDense(2, []float64{newth, newthdot})
	return newObs
}

func (p *base) update(action, newState *mat.VecDense) (timestep.TimeStep,
	bool) {
	// Create the new timestep
	reward := p.GetReward(p.LastTimeStep().Observation, action, newState)
	nextStep := timestep.New(timestep.Mid, reward, p.discount, newState,
		p.LastTimeStep().Number+1)

	// Check if the step is the last in the episode and adjust step type
	// if necessary
	p.End(&nextStep)

	p.lastStep = nextStep
	return nextStep, nextStep.Last()
}

// DiscountSpec returns the discount specification of the environment
func (p *base) DiscountSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)

	lowerBound := mat.NewVecDense(1, []float64{p.discount})

	upperBound := mat.NewVecDense(1, []float64{p.discount})

	return spec.NewEnvironment(shape, spec.Discount, lowerBound, upperBound,
		spec.Continuous)

}

// ObservationSpec returns the observation specification of the environment
func (p *base) ObservationSpec() spec.Environment {
	shape := mat.NewVecDense(ObservationDims, nil)

	minObs := []float64{p.angleBounds.Min, p.speedBounds.Min}
	lowerBound := mat.NewVecDense(ObservationDims, minObs)

	maxObs := []float64{p.angleBounds.Max, p.speedBounds.Max}
	upperBound := mat.NewVecDense(ObservationDims, maxObs)

	return spec.NewEnvironment(shape, spec.Observation, lowerBound, upperBound,
		spec.Continuous)

}

// String converts the environment to a string representation
func (p *base) String() string {
	str := "base  |  theta: %v  |  theta dot: %v\n"
	theta := p.lastStep.Observation.AtVec(0)
	thetadot := p.lastStep.Observation.AtVec(1)

	return fmt.Sprintf(str, theta, thetadot)
}

// Render renders the current timestep to the terminal
func (p *base) Render() {
	angle := p.lastStep.Observation.AtVec(0)
	var frame string

	if angle > -math.Pi/8 && angle < math.Pi/8 {
		frame = "  | \n  ."
	} else if angle > -math.Pi/8 && angle < (3*math.Pi/8) {
		frame = "   / \n  ."
	} else if angle >= (3*math.Pi/8) && angle < (5*math.Pi/8) {
		frame = "  .--\n"
	} else if angle >= (5*math.Pi/8) && angle < (7*math.Pi/8) {
		frame = "  . \n   \\"
	} else if angle >= (7*math.Pi/8) && angle < (9*math.Pi/8) {
		frame = "  . \n  |"
	} else if angle > (-9*math.Pi/8) && angle <= (-7*math.Pi/8) {
		frame = "  . \n  |"
	} else if angle > (-7*math.Pi/8) && angle <= (-5*math.Pi/8) {
		frame = "  . \n/"
	} else if angle > (-5*math.Pi/8) && angle <= (-3*math.Pi/8) {
		frame = "--.\n"
	} else if angle > (-3*math.Pi/8) && angle <= (-math.Pi/8) {
		frame = "\\ \n  ."
	}
	os.Stdout.WriteString("\x1b[3;J\x1b[H\x1b[2J")
	fmt.Printf("\n\n%s\n\n", frame)

}

// normalizeAngle normalizes the pendulum angle to the appropriate limits
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

// validateState validates the state to ensure that the angle and angular
// velocity are within the environmental limits
func validateState(obs mat.Vector, angleBounds, speedBounds r1.Interval) {
	// Check if the angle is within bounds
	thWithinBounds := obs.AtVec(0) <= angleBounds.Max &&
		obs.AtVec(0) >= angleBounds.Min
	if !thWithinBounds {
		panic(fmt.Sprintf("theta is not within bounds %v", angleBounds))
	}

	// Check if the angular velocity is within bounds
	thdotWithinBounds := obs.AtVec(1) <= speedBounds.Max &&
		obs.AtVec(1) >= speedBounds.Min
	if !thdotWithinBounds {
		panic(fmt.Sprintf("theta dot is not within bounds %v", speedBounds))
	}
}
