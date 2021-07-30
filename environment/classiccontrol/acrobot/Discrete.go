package acrobot

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
)

type Discrete struct {
	*Acrobot
}

func NewDiscrete(t env.Task, discount float64) (env.Environment, ts.TimeStep) {
	acrobot, firstStep := newBase(t, discount)

	return &Discrete{acrobot}, firstStep
}

// ActionSpec returns the action specification of the environment
func (d *Discrete) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(ActionDims, nil)
	lowerBound := mat.NewVecDense(ActionDims, []float64{float64(MinDiscreteAction)})
	upperBound := mat.NewVecDense(ActionDims, []float64{float64(MaxDiscreteAction)})

	return spec.NewEnvironment(shape, spec.Action, lowerBound,
		upperBound, spec.Discrete)
}

// Step takes one environmental step given action a and returns the next
// timestep as a timestep.TimeStep and a bool indicating whether or not
// the episode has ended. Actions are discrete, consisting of the
// torque applied to the acrobot's base and are in the set
// {MinDiscreteAction, MinDiscreteAction+1, ..., MaxDiscreteAction}.
// Actions outside this range will cause the environment to panic.
func (d *Discrete) Step(a *mat.VecDense) (ts.TimeStep, bool) {
	// Ensure action is 1-dimensional
	if a.Len() > ActionDims {
		panic("Actions should be 1-dimensional")
	}

	// Discrete action in {0, 1, 2}
	action := a.AtVec(0)

	// Ensure a legal action was selected
	intAction := int(action)
	if intAction > MaxDiscreteAction || intAction < MinDiscreteAction {
		panic(fmt.Sprintf("illegal action %v \u2209 (0, 1, 2)", intAction))
	}

	// Calculate the torque applied
	torque := float64(intAction) - 1.0

	// Calculate the next state given the force/action
	newState := d.nextState(torque)

	// Update embedded base Acrobot environment
	return d.update(a, newState)
}
