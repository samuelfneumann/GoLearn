package acrobot

import (
	"gonum.org/v1/gonum/mat"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/floatutils"
)

type Continuous struct {
	*Acrobot
}

func NewContinuous(t env.Task, discount float64) (env.Environment, ts.TimeStep) {
	acrobot, firstStep := newBase(t, discount)

	return &Continuous{acrobot}, firstStep
}

// ActionSpec returns the action specification of the environment
func (d *Continuous) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(ActionDims, nil)
	lowerBound := mat.NewVecDense(ActionDims, []float64{float64(MinContinuousAction)})
	upperBound := mat.NewVecDense(ActionDims, []float64{float64(MaxContinuousAction)})

	return spec.NewEnvironment(shape, spec.Action, lowerBound,
		upperBound, spec.Continuous)
}

// Step takes one environmental step given action a and returns the next
// timestep as a timestep.TimeStep and a bool indicating whether or not
// the episode has ended. Actions are continuous, consisting of the
// torque applied to the acrobot's fixed base. Action are bounded
// by [MinContinuousAction, MaxContinuousAction]. Actions outside
// this range will cause the environment to panic.
func (d *Continuous) Step(a *mat.VecDense) (ts.TimeStep, bool) {
	// Ensure action is 1-dimensional
	if a.Len() > ActionDims {
		panic("Actions should be 1-dimensional")
	}

	// Calculate the torque applied
	torque := floatutils.Clip(a.AtVec(0), MinContinuousAction,
		MaxContinuousAction)

	// Calculate the next state given the force/action
	newState := d.nextState(torque)

	// Update embedded base Acrobot environment
	return d.update(a, newState)
}
