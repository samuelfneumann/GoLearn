package acrobot

import (
	"fmt"

	"github.com/samuelfneumann/golearn/environment"
	env "github.com/samuelfneumann/golearn/environment"
	ts "github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/golearn/utils/floatutils"
	"gonum.org/v1/gonum/mat"
)

// Continuous implements the classic control environment Acrobot. In
// this environment, a double hindged and double linked pendulum is
// attached to a single actuated fixed base. Torque can be applied to
// the base to swing the double pendulum (acrobot) around.
//
// State feature vectors are 4-dimensional and consist of the angle
// of the first pendulum link measured from the negative y-axis,
// the angle of the second pendulum link measured from the negative
// y-axis, the angular velocity of the first link, and the angular
// velocity of the second link. That is, a feature vector has the
// form:
//
//		v ⃗	= [θ1, θ2, θ̇1, θ̇2], where:
//		θ1 = angle of the first link measured from the negative y-axis
//		θ2 = angle of the second link measured from the negative y-axis
//		θ̇1 = angular velocity of the first link
//		θ̇2 = angular velocity of the second link
//
//
// State features are bounded. Angles are bounded to be between [-π, π]
// and angular velocity is bounded between [MinVel1, MaxVel1] for the
// first pendulum link and [MinVel2, MaxVel2] for the second pendulum
// link. Angles outside of [-π, π] are wrapped around to stay within
// this range, and angular velocity is clipped to stay within the
// legal range.
//
// Actions are continuous in [MinContinuousAction, MaxContinuousAction].
// Actions outside of these bounds are clipped to stay within these
// bounds.
//
// Continuous implements the environment.Environment interface.

type Continuous struct {
	*base
}

// NewDiscrete returns a new Acrobot environment with continuous actions
func NewContinuous(t env.Task, discount float64) (env.Environment, ts.TimeStep,
	error) {
	acrobot, firstStep, err := newBase(t, discount)
	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("newContinuous: %v", err)
	}

	return &Continuous{acrobot}, firstStep, nil
}

// ActionSpec returns the action specification of the environment
func (d *Continuous) ActionSpec() environment.Spec {
	shape := mat.NewVecDense(ActionDims, nil)
	lowerBound := mat.NewVecDense(ActionDims, []float64{float64(MinContinuousAction)})
	upperBound := mat.NewVecDense(ActionDims, []float64{float64(MaxContinuousAction)})

	return environment.NewSpec(shape, environment.Action, lowerBound,
		upperBound, environment.Continuous)
}

// Step takes one environmental step given action a and returns the next
// timestep as a timestep.TimeStep and a bool indicating whether or not
// the episode has ended. Actions are continuous, consisting of the
// torque applied to the acrobot's fixed base. Action are bounded
// by [MinContinuousAction, MaxContinuousAction]. Actions outside
// this range will cause the environment to panic.
func (d *Continuous) Step(a *mat.VecDense) (ts.TimeStep, bool, error) {
	// Ensure action is 1-dimensional
	if a.Len() > ActionDims {
		return ts.TimeStep{}, true, fmt.Errorf("Actions should be " +
			"1-dimensional")
	}

	// Calculate the torque applied
	torque := floatutils.Clip(a.AtVec(0), MinContinuousAction,
		MaxContinuousAction)

	// Calculate the next state given the force/action
	newState, err := d.nextState(torque)
	if err != nil {
		return ts.TimeStep{}, true, fmt.Errorf("step: could not calculate "+
			"next state: %v", err)
	}

	// Update embedded base Acrobot environment
	nextStep, done := d.update(a, newState)
	return nextStep, done, nil
}
