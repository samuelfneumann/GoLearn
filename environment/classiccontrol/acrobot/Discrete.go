package acrobot

import (
	"fmt"

	"github.com/samuelfneumann/golearn/environment"
	env "github.com/samuelfneumann/golearn/environment"
	ts "github.com/samuelfneumann/golearn/timestep"
	"gonum.org/v1/gonum/mat"
)

// Discrete implements the classic control environment Acrobot. In
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
// Actions are discrete in the set {MinDiscreteAction,
// MinDiscreteAction+1, ..., MaxDiscreteAction}. Actions outside of
// these bounds result in the envirnoment panicing. Given that the
// constants MinDiscreteAction = 0 and MaxDiscreteAction = 2, actions
// have the following interpretations:
//
//		Action		 Torque
//		  0			MinTorque
//		  1			  0.0
//		  2			MaxTorque
//
// Discrete implements the environment.Environment interface.

type Discrete struct {
	*base
}

// NewDiscrete returns a new Acrobot environment with discrete actions
func NewDiscrete(t env.Task, discount float64) (env.Environment, ts.TimeStep,
	error) {
	acrobot, firstStep, err := newBase(t, discount)
	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("newDiscrete: %v", err)
	}

	return &Discrete{acrobot}, firstStep, nil
}

// ActionSpec returns the action specification of the environment
func (d *Discrete) ActionSpec() environment.Spec {
	shape := mat.NewVecDense(ActionDims, nil)
	lowerBound := mat.NewVecDense(ActionDims,
		[]float64{float64(MinDiscreteAction)})
	upperBound := mat.NewVecDense(ActionDims,
		[]float64{float64(MaxDiscreteAction)})

	return environment.NewSpec(shape, environment.Action, lowerBound,
		upperBound, environment.Discrete)
}

// Step takes one environmental step given action a and returns the next
// timestep as a timestep.TimeStep and a bool indicating whether or not
// the episode has ended. Actions are discrete, consisting of the
// torque applied to the acrobot's base and are in the set
// {MinDiscreteAction, MinDiscreteAction+1, ..., MaxDiscreteAction}.
// Actions outside this range will cause an error to be returned.
func (d *Discrete) Step(a *mat.VecDense) (ts.TimeStep, bool, error) {
	// Ensure action is 1-dimensional
	if a.Len() > ActionDims {
		return ts.TimeStep{}, true, fmt.Errorf("Actions should be " +
			"1-dimensional")
	}

	// Discrete action in {0, 1, 2}
	action := a.AtVec(0)

	// Ensure a legal action was selected
	intAction := int(action)
	if intAction > MaxDiscreteAction || intAction < MinDiscreteAction {
		return ts.TimeStep{}, true, fmt.Errorf("step: illegal action %v "+
			"\u2209 (0, 1, 2)", intAction)
	}

	// Calculate the torque applied
	var torque float64
	if intAction == MinDiscreteAction {
		torque = MinTorque
	} else if intAction == MaxDiscreteAction {
		torque = MaxTorque
	} else if intAction == 0 {
		torque = 0.0
	} else {
		return ts.TimeStep{}, true, fmt.Errorf("step: illegal action %v "+
			"\u2209 (0, 1, 2)", intAction)
	}

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
