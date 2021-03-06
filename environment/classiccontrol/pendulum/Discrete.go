// Package pendulu, implements the pendulum classic control environment
package pendulum

import (
	"fmt"

	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/timestep"
	"gonum.org/v1/gonum/mat"
)

// NOTE: This file has not been tested as of yet

// Discrete implements the classic control environment Pendulum. In this
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
// Actions are discrete and 1-dimensional. Actions determine the
// torque to apply to the pendulum at its fixed base:
//
//		Action		TorqueApplied
//		0			MinContinuousAction
//		1			MinContinuousAction / 2.0
//		2			MinContinuousAction / 4.0
//		3			0
//		4			MaxContinuousAction / 4.0
//		5			MaxContinuousAction / 2.0
//		6			MaxContinuousAction
//
// Discrete implements the environment.Environment interface
type Discrete struct {
	*base
}

// New creates and returns a new Discrete environment
func NewDiscrete(t environment.Task,
	discount float64) (environment.Environment, timestep.TimeStep,
	error) {
	baseEnv, firstStep, err := newBase(t, discount)
	if err != nil {
		return nil, timestep.TimeStep{}, fmt.Errorf("newDiscrete: %v",
			err)
	}

	pendulum := Discrete{baseEnv}

	return &pendulum, firstStep, nil
}

// Step takes one environmental step given action a and returns the next
// timestep as a timestep.TimeStep and a bool indicating whether or not
// the episode has ended. Actions are 1-dimensional and discrete in
// (0, 1, 2, 3, 4, 5, 6). Actions outside this set will cause an
// error to be returned
func (p *Discrete) Step(action *mat.VecDense) (timestep.TimeStep, bool,
	error) {
	// Ensure action is 1-dimensional
	if action.Len() > ActionDims {
		return timestep.TimeStep{}, true, fmt.Errorf("step: ctions should be " +
			"1-dimensional")
	}

	// Convert discrete action to torque applied to fixed base
	var torque float64
	if action.AtVec(0) <= 0.0 {
		torque = MinContinuousAction
	} else if action.AtVec(0) == 1.0 {
		torque = MinContinuousAction / 2.0
	} else if action.AtVec(0) == 2.0 {
		torque = MinContinuousAction / 4.0
	} else if action.AtVec(0) == 3.0 {
		torque = 0.0
	} else if action.AtVec(0) == 4.0 {
		torque = MaxContinuousAction / 4.0
	} else if action.AtVec(0) == 5.0 {
		torque = MaxContinuousAction / 2.0
	} else if action.AtVec(0) == 6.0 {
		torque = MaxContinuousAction
	} else {
		return timestep.TimeStep{}, true, fmt.Errorf("step: illegal action %v",
			action.AtVec(0))
	}

	// Calculate the next state given the torque/action
	nextState := p.nextState(p.lastStep, torque)

	// Update the embedded base environment
	nextStep, last := p.update(action, nextState)

	return nextStep, last, nil
}

// ActionSpec returns the action specification of the environment
func (p *Discrete) ActionSpec() environment.Spec {
	shape := mat.NewVecDense(ActionDims, nil)

	lowerBound := mat.NewVecDense(ActionDims, []float64{MinDiscreteAction})
	upperBound := mat.NewVecDense(ActionDims, []float64{MaxDiscreteAction})

	return environment.NewSpec(shape, environment.Action, lowerBound, upperBound,
		environment.Discrete)
}

// String converts the environment to a string representation
func (p *Discrete) String() string {
	str := "Discrete  |  theta: %v  |  theta dot: %v\n"
	theta := p.lastStep.Observation.AtVec(0)
	thetadot := p.lastStep.Observation.AtVec(1)

	return fmt.Sprintf(str, theta, thetadot)
}
