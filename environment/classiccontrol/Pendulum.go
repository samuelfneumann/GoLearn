// Package classic implements classic control environments
package classiccontrol

// TODO: Ensure Pendulum bounds should be taken from Starter.Bounds(), and
// TODO: it should be made sure the lower bound == - upper bound for all bounds

// TODO: when starting a new episode

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
)

// default physical constants
const (
	angle   float64 = math.Pi
	speed   float64 = 8.0
	torque  float64 = 2.0
	dt      float64 = 0.05
	gravity float64 = 9.8
	mass    float64 = 1.0
	length  float64 = 1.0
)

// Pendulum is an classic control environment where an agent must learn to
// swing a pendulum up and hold it in an upright position. Actions are continuous
// and 1D, consisting of the torque applied to the pendulum's fixed base.
// Angles, angular velocity, and torque are all bounded by the respective
// constants defined in this file.
type Pendulum struct {
	//environment.Starter
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

// NewPendulum creates and returns a new Pendulum environment
func NewPendulum(t environment.Task, d float64) (*Pendulum, timestep.TimeStep) {
	angleBounds := r1.Interval{Min: -angle, Max: angle}
	speedBounds := r1.Interval{Min: -speed, Max: speed}
	torqueBounds := r1.Interval{Min: -torque, Max: torque}

	state := t.Start()
	validateState(state, angleBounds, speedBounds)
	firstStep := timestep.New(timestep.First, 0.0, d, state, 0)

	pendulum := Pendulum{t, dt, gravity, mass, length, angleBounds,
		speedBounds, torqueBounds, firstStep, d}

	return &pendulum, firstStep
}

// ValidateState validates the state to ensure that the angle and angular
// velocity are within the environmental limits
func validateState(obs mat.Vector, angleBounds, speedBounds r1.Interval) {
	thWithinBounds := obs.AtVec(0) <= angleBounds.Max &&
		obs.AtVec(0) >= angleBounds.Min
	if !thWithinBounds {
		panic(fmt.Sprintf("theta is not within bounds %v", angleBounds))
	}

	thdotWithinBounds := obs.AtVec(1) <= speedBounds.Max &&
		obs.AtVec(1) >= speedBounds.Min
	if !thdotWithinBounds {
		panic(fmt.Sprintf("theta dot is not within bounds %v", speedBounds))
	}
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
	action = floats.Max([]float64{action, p.torqueBounds.Min})
	action = floats.Min([]float64{action, p.torqueBounds.Max})

	newthdot := thdot + (-3*p.gravity/(2*p.length)*math.Sin(th+math.Pi)+
		3.0/(p.mass*math.Pow(p.length, 2))*action)*p.dt

	newth := th + (newthdot * p.dt)

	// Clip the angular velocity
	newthdot = floats.Min([]float64{newthdot, p.speedBounds.Max})
	newthdot = floats.Max([]float64{newthdot, p.speedBounds.Min})

	// Normalize the angle
	newth = p.normalize(newth)

	newObs := mat.NewVecDense(2, []float64{newth, newthdot})

	return newObs
}

// Step takes one environmental step given an action and returns a TimeStep
// representing the new state and a bool indicating whether or not this is the
// last environmental transition
func (p *Pendulum) Step(action mat.Vector) (timestep.TimeStep, bool) {
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

// RewardSpec returns the reward specification of the environment
func (p *Pendulum) RewardSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)

	minReward := p.Min()
	lowerBound := mat.NewVecDense(2, []float64{minReward})

	maxReward := p.Max()
	upperBound := mat.NewVecDense(1, []float64{maxReward})

	return spec.NewEnvironment(shape, spec.Reward, lowerBound, upperBound,
		spec.Continuous)

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
	shape := mat.NewVecDense(1, nil)

	minAction, maxAction := p.torqueBounds.Min, p.torqueBounds.Max
	lowerBound := mat.NewVecDense(1, []float64{minAction})
	upperBound := mat.NewVecDense(1, []float64{maxAction})

	return spec.NewEnvironment(shape, spec.Action, lowerBound, upperBound,
		spec.Continuous)

}

// ObservationSpec returns the observation specification of the environment
func (p *Pendulum) ObservationSpec() spec.Environment {
	shape := mat.NewVecDense(2, nil)

	minObs := []float64{p.angleBounds.Min, p.speedBounds.Min}
	lowerBound := mat.NewVecDense(2, minObs)

	maxObs := []float64{p.angleBounds.Max, p.speedBounds.Max}
	upperBound := mat.NewVecDense(2, maxObs)

	return spec.NewEnvironment(shape, spec.Observation, lowerBound, upperBound,
		spec.Continuous)

}

// Normalize normalizes the pendulum angle to the appropriate limits
func (p *Pendulum) normalize(th float64) float64 {
	if p.angleBounds.Max != -p.angleBounds.Min {
		panic("angle bounds should be centered aroudn 0")
	}

	if th > p.angleBounds.Max {
		divisor := int(th / p.angleBounds.Max)
		return -math.Pi + th - (p.angleBounds.Max * float64(divisor))
	} else if th < p.angleBounds.Min {
		divisor := int(th / p.angleBounds.Min)
		return math.Pi + th - (p.angleBounds.Min * float64(divisor))
	} else {
		return th
	}
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

type PendulumSwingUp struct {
	environment.Starter
	maxSteps int
}

func NewPendulumSwingUp(s environment.Starter, maxSteps int) *PendulumSwingUp {
	return &PendulumSwingUp{s, maxSteps}
}

func (s *PendulumSwingUp) GetReward(t timestep.TimeStep, _ mat.Vector) float64 {
	th := t.Observation.AtVec(0)
	return math.Cos(th)
}

func (s *PendulumSwingUp) AtGoal(state mat.Matrix) bool {
	return state.At(0, 0) == 0
}

func (s *PendulumSwingUp) Min() float64 {
	return -1.0
}

func (s *PendulumSwingUp) Max() float64 {
	return 1.0
}

func (s *PendulumSwingUp) End(t *timestep.TimeStep) bool {
	if t.Number >= s.maxSteps {
		t.StepType = timestep.Last
		return true
	}
	return false
}
