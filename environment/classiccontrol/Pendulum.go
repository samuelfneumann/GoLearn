// Package classic implements classic control environments
package classiccontrol

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
)

// TODO: NormalizeWrapper() which normalizes states to some bounds

// TODO: Ensure Pendulum bounds should be taken from Starter.Bounds(), and
// TODO: it should be made sure the lower bound == - upper bound for all bounds

// TODO: instead of panicing just normalize and clip to between bounds
// TODO: when starting a new episode

type Pendulum struct {
	environment.Starter
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
	maxSteps     int
}

func NewPendulum(s environment.Starter, t environment.Task,
	d float64, maxSteps int) (*Pendulum, timestep.TimeStep) {
	maxAngle := math.Pi
	angleBounds := r1.Interval{Min: -maxAngle, Max: maxAngle}

	maxSpeed := 8.0
	speedBounds := r1.Interval{Min: -maxSpeed, Max: maxSpeed}

	maxTorque := 2.0
	torqueBounds := r1.Interval{Min: -maxTorque, Max: maxTorque}

	dt := 0.05
	gravity := 9.8
	mass := 1.0
	length := 1.0

	state := s.Start()
	validateState(state, angleBounds, speedBounds)
	firstStep := timestep.New(timestep.First, 0.0, d, state, 0)

	pendulum := Pendulum{s, t, dt, gravity, mass, length, angleBounds,
		speedBounds, torqueBounds, firstStep, d, maxSteps}

	return &pendulum, firstStep
}

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

func (p *Pendulum) Reset() timestep.TimeStep {
	state := p.Start()
	validateState(state, p.angleBounds, p.speedBounds)
	startStep := timestep.New(timestep.First, 0, p.discount, state, 0)
	p.lastStep = startStep

	return startStep
}

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

func (p *Pendulum) Step(a mat.Vector) (timestep.TimeStep, bool) {
	nextState := p.NextState(p.lastStep, a)
	newth := nextState.AtVec(0)

	stepNum := p.lastStep.Number + 1
	stepType := timestep.Mid
	if stepNum == p.maxSteps {
		stepType = timestep.Last
	}

	reward := math.Cos(newth)
	step := timestep.New(stepType, reward, p.discount, nextState, stepNum)

	p.lastStep = step
	return step, step.Last()
}

func (p *Pendulum) RewardSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)

	minReward := p.Min()
	lowerBound := mat.NewVecDense(2, []float64{minReward})

	maxReward := p.Max()
	upperBound := mat.NewVecDense(1, []float64{maxReward})

	return spec.NewEnvironment(shape, spec.Reward, lowerBound, upperBound)

}

func (p *Pendulum) DiscountSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)

	lowerBound := mat.NewVecDense(1, []float64{p.discount})

	upperBound := mat.NewVecDense(1, []float64{p.discount})

	return spec.NewEnvironment(shape, spec.Discount, lowerBound, upperBound)

}

func (p *Pendulum) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)

	minAction, maxAction := p.torqueBounds.Min, p.torqueBounds.Max
	lowerBound := mat.NewVecDense(1, []float64{minAction})
	upperBound := mat.NewVecDense(1, []float64{maxAction})

	return spec.NewEnvironment(shape, spec.Action, lowerBound, upperBound)

}

func (p *Pendulum) ObservationSpec() spec.Environment {
	shape := mat.NewVecDense(2, nil)

	minObs := []float64{p.angleBounds.Min, p.speedBounds.Min}
	lowerBound := mat.NewVecDense(2, minObs)

	maxObs := []float64{p.angleBounds.Max, p.speedBounds.Max}
	upperBound := mat.NewVecDense(2, maxObs)

	return spec.NewEnvironment(shape, spec.Observation, lowerBound, upperBound)

}

func (p *Pendulum) normalize(th float64) float64 {
	if p.angleBounds.Max != -p.angleBounds.Min {
		panic("angle bounds should be centered aroudn 0")
	}

	if th > p.angleBounds.Max {
		divisor := int(th / p.angleBounds.Max)
		return th - (p.angleBounds.Max * float64(divisor))
	} else if th < p.angleBounds.Min {
		divisor := int(th / p.angleBounds.Min)
		return th - (p.angleBounds.Min * float64(divisor))
	} else {
		return th
	}
}

func (p *Pendulum) String() string {
	str := "Pendulum  |  theta: %v  |  theta dot: %v\n"
	theta := p.lastStep.Observation.AtVec(0)
	thetadot := p.lastStep.Observation.AtVec(1)

	return fmt.Sprintf(str, theta, thetadot)
}

type PendulumSwingUp struct{}

func NewPendulumSwingUp() *PendulumSwingUp {
	return &PendulumSwingUp{}
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
