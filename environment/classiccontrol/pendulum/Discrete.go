// Package pendulu, implements the pendulum classic control environment
package pendulum

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
)

// TODO: This documentation needs to be updated
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
// Actions are continuous and 1-dimensional. Actions determine the
// torque to apply to the pendulum at its fixed base. Actions are
// bounded by [-2, 2] = [MinDiscreteAction, MaxDiscreteAction].
// Actions outside of this region are clipped to stay within these
// bounds.
//
// Discrete implements the environment.Environment interface
type Discrete struct {
	*base
}

// New creates and returns a new Discrete environment
func NewDiscrete(t environment.Task, discount float64) (*Discrete, timestep.TimeStep) {
	baseEnv, firstStep := newBase(t, discount)

	pendulum := Discrete{baseEnv}

	return &pendulum, firstStep
}

// Step takes one environmental step given action a and returns the next
// timestep as a timestep.TimeStep and a bool indicating whether or not
// the episode has ended. Actions are 1-dimensional and continuous, c
// onsisting of the horizontal force to apply to the cart. Actions
// outside the legal range of [-1, 1] are clipped to stay within this
// range.
func (p *Discrete) Step(action *mat.VecDense) (timestep.TimeStep, bool) {
	// Ensure action is 1-dimensional
	if action.Len() > ActionDims {
		panic("Actions should be 1-dimensional")
	}

	// Convert discrete action to torque applied to fixed base
	var torque float64
	if action.AtVec(0) <= 0.0 {
		torque = MinContinuousAction
	} else if action.AtVec(0) == 1.0 {
		torque = MinContinuousAction / 2.0
	} else if action.AtVec(0) == 2.0 {
		torque = 0.0
	} else if action.AtVec(0) == 3.0 {
		torque = MaxContinuousAction / 2.0
	} else if action.AtVec(0) == 4.0 {
		torque = MaxContinuousAction
	} else {
		panic(fmt.Sprintf("step: illegal action %v", action.AtVec(0)))
	}

	// Calculate the next state given the torque/action
	nextState := p.nextState(p.lastStep, torque)

	// Update the embedded base environment
	nextStep, last := p.update(action, nextState)

	return nextStep, last
}

// ActionSpec returns the action specification of the environment
func (p *Discrete) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(ActionDims, nil)

	lowerBound := mat.NewVecDense(ActionDims, []float64{MinDiscreteAction})
	upperBound := mat.NewVecDense(ActionDims, []float64{MaxDiscreteAction})

	return spec.NewEnvironment(shape, spec.Action, lowerBound, upperBound,
		spec.Discrete)
}

// String converts the environment to a string representation
func (p *Discrete) String() string {
	str := "Discrete  |  theta: %v  |  theta dot: %v\n"
	theta := p.lastStep.Observation.AtVec(0)
	thetadot := p.lastStep.Observation.AtVec(1)

	return fmt.Sprintf(str, theta, thetadot)
}
