// Package pendulu, implements the pendulum classic control environment
package pendulum

import (
	"fmt"

	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/golearn/utils/floatutils"
	"gonum.org/v1/gonum/mat"
)

// Continuous implements the classic control environment Pendulum. In this
// environment, a pendulum is attached to a fixed base. An agent can
// swing the pendulum back and forth, but the swinging force/torque is
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
// Continuous implements the environment.Environment interface
type Continuous struct {
	*base
}

// New creates and returns a new Continuous environment
func NewContinuous(t environment.Task, discount float64) (*Continuous, timestep.TimeStep) {
	baseEnv, firstStep := newBase(t, discount)

	pendulum := Continuous{baseEnv}

	return &pendulum, firstStep
}

// Step takes one environmental step given action a and returns the next
// timestep as a timestep.TimeStep and a bool indicating whether or not
// the episode has ended. Actions are 1-dimensional and continuous, c
// onsisting of the horizontal force to apply to the cart. Actions
// outside the legal range of [-1, 1] are clipped to stay within this
// range.
func (p *Continuous) Step(action *mat.VecDense) (timestep.TimeStep, bool) {
	// Ensure action is 1-dimensional
	if action.Len() > ActionDims {
		panic("Actions should be 1-dimensional")
	}

	// Clip action to ensure that it is in the legal range of continuous
	// actions
	torque := floatutils.Clip(action.AtVec(0), MinContinuousAction,
		MaxContinuousAction)

	// Calculate the next state given the torque/action
	nextState := p.nextState(p.lastStep, torque)

	// Update the embedded base environment
	nextStep, last := p.update(action, nextState)

	return nextStep, last
}

// ActionSpec returns the action specification of the environment
func (p *Continuous) ActionSpec() environment.Spec {
	shape := mat.NewVecDense(ActionDims, nil)

	minAction, maxAction := p.torqueBounds.Min, p.torqueBounds.Max
	lowerBound := mat.NewVecDense(ActionDims, []float64{minAction})
	upperBound := mat.NewVecDense(ActionDims, []float64{maxAction})

	return environment.NewSpec(shape, environment.Action, lowerBound, upperBound,
		environment.Continuous)

}

// String converts the environment to a string representation
func (p *Continuous) String() string {
	str := "Continuous  |  theta: %v  |  theta dot: %v\n"
	theta := p.lastStep.Observation.AtVec(0)
	thetadot := p.lastStep.Observation.AtVec(1)

	return fmt.Sprintf(str, theta, thetadot)
}
