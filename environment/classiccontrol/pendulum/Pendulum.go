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
	AngleBound          float64 = math.Pi // The angle bounds
	SpeedBound          float64 = 8.0     // The angular velocity/speed bounds
	MaxContinuousAction float64 = 2.0     // The torque bounds
	MinContinuousAction float64 = -MaxContinuousAction
	dt                  float64 = 0.05
	Gravity             float64 = 9.8
	Mass                float64 = 1.0
	Length              float64 = 1.0
	ActionDims          int     = 1
	ObservationDims     int     = 2
)

// Pendulum implements the classic control environment Pendulum. In this
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
// Pendulum implements the environment.Environment interface
type Pendulum struct {
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

// New creates and returns a new Pendulum environment
func New(t environment.Task, d float64) (*Pendulum, timestep.TimeStep) {
	angleBounds := r1.Interval{Min: -AngleBound, Max: AngleBound}
	speedBounds := r1.Interval{Min: -SpeedBound, Max: SpeedBound}
	torqueBounds := r1.Interval{Min: MinContinuousAction,
		Max: MaxContinuousAction}

	state := t.Start()
	validateState(state, angleBounds, speedBounds)

	firstStep := timestep.New(timestep.First, 0.0, d, state, 0)

	pendulum := Pendulum{t, dt, Gravity, Mass, Length, angleBounds,
		speedBounds, torqueBounds, firstStep, d}

	return &pendulum, firstStep
}

// Reset resets the environment and returns a starting state drawn from the
// Starter
func (p *Pendulum) Reset() timestep.TimeStep {
	state := p.Start()
	validateState(state, p.angleBounds, p.speedBounds)
	startStep := timestep.New(timestep.First, 0, p.discount, state, 0)
	p.lastStep = startStep

	return startStep
}

// NextState computes the next state of the environment given a timestep and
// an action a
func (p *Pendulum) NextState(t timestep.TimeStep, a mat.Vector) mat.Vector {
	if a.Len() != 1 {
		panic("only 1D actions are allowed")
	}

	obs := t.Observation
	th, thdot := obs.AtVec(0), obs.AtVec(1)

	// Clip the torque
	action := a.AtVec(0)
	action = floatutils.ClipInterval(action, p.torqueBounds)

	newthdot := thdot + (-3*p.gravity/(2*p.length)*math.Sin(th+math.Pi)+
		3.0/(p.mass*math.Pow(p.length, 2))*action)*p.dt

	newth := th + (newthdot * p.dt)

	// Clip the angular velocity
	newthdot = floats.Min([]float64{newthdot, p.speedBounds.Max})
	newthdot = floats.Max([]float64{newthdot, p.speedBounds.Min})

	// Normalize the angle
	newth = normalizeAngle(newth, p.angleBounds)

	newObs := mat.NewVecDense(2, []float64{newth, newthdot})

	return newObs
}

// Step takes one environmental step given action a and returns the next
// timestep as a timestep.TimeStep and a bool indicating whether or not
// the episode has ended. Actions are 1-dimensional and continuous, c
// onsisting of the horizontal force to apply to the cart. Actions
// outside the legal range of [-1, 1] are clipped to stay within this
// range.
func (p *Pendulum) Step(action mat.Vector) (timestep.TimeStep, bool) {
	// Ensure action is 1-dimensional
	if action.Len() > ActionDims {
		panic("Actions should be 1-dimensional")
	}

	nextState := p.NextState(p.lastStep, action)
	newth := nextState.AtVec(0)

	stepNum := p.lastStep.Number + 1
	stepType := timestep.Mid

	reward := math.Cos(newth)
	step := timestep.New(stepType, reward, p.discount, nextState, stepNum)

	p.End(&step)

	p.lastStep = step
	return step, step.Last()
}

// DiscountSpec returns the discount specification of the environment
func (p *Pendulum) DiscountSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)

	lowerBound := mat.NewVecDense(1, []float64{p.discount})

	upperBound := mat.NewVecDense(1, []float64{p.discount})

	return spec.NewEnvironment(shape, spec.Discount, lowerBound, upperBound,
		spec.Continuous)

}

// ActionSpec returns the action specification of the environment
func (p *Pendulum) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(ActionDims, nil)

	minAction, maxAction := p.torqueBounds.Min, p.torqueBounds.Max
	lowerBound := mat.NewVecDense(ActionDims, []float64{minAction})
	upperBound := mat.NewVecDense(ActionDims, []float64{maxAction})

	return spec.NewEnvironment(shape, spec.Action, lowerBound, upperBound,
		spec.Continuous)

}

// ObservationSpec returns the observation specification of the environment
func (p *Pendulum) ObservationSpec() spec.Environment {
	shape := mat.NewVecDense(ObservationDims, nil)

	minObs := []float64{p.angleBounds.Min, p.speedBounds.Min}
	lowerBound := mat.NewVecDense(ObservationDims, minObs)

	maxObs := []float64{p.angleBounds.Max, p.speedBounds.Max}
	upperBound := mat.NewVecDense(ObservationDims, maxObs)

	return spec.NewEnvironment(shape, spec.Observation, lowerBound, upperBound,
		spec.Continuous)

}

// String converts the environment to a string representation
func (p *Pendulum) String() string {
	str := "Pendulum  |  theta: %v  |  theta dot: %v\n"
	theta := p.lastStep.Observation.AtVec(0)
	thetadot := p.lastStep.Observation.AtVec(1)

	return fmt.Sprintf(str, theta, thetadot)
}

// Render renders the current timestep to the terminal
func (p *Pendulum) Render() {
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
