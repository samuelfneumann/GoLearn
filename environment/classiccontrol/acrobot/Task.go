package acrobot

import (
	"fmt"
	"math"

	"github.com/samuelfneumann/golearn/environment"
	ts "github.com/samuelfneumann/golearn/timestep"
	"gonum.org/v1/gonum/mat"
)

const (
	// Goal position in the classic control problem is to swing the
	// tip above one link length above the fixed base. Here, we use
	// the length of the first link (which is also equal to the length
	// of the second link).
	GoalHeight float64 = LinkLength1

	// Max and min reward possible for the default classic control case.
	// maxReward is given at episode termination, and minReward is
	// given on all other timesteps.
	maxReward, minReward float64 = 0.0, -1.0
)

// SwingUp implements the classic control Acrobot task where the
// agent must swing the tip of the second link above some set
// height.
//
// The task is a cost-to-goal task:
// A reward of -1.0 is given on all timesteps except for the timestep
// which transitions the acrobot's second link above the goal line.
// On this tiemstep, a reward of 0.0 is given.
//
// Episodes are ended when the acrobot's second link swings above the
// goal height or a step limit is reached.
type SwingUp struct {
	environment.Starter
	stepLimitEnder environment.Ender // Ends when step limit reached

	f         func(*mat.VecDense) bool // Function for lineEnder
	lineEnder environment.Ender        // Ends tip is above a line
}

// NewSwingUp returns a new SwingUp task with start state distribution
// defined by s, episodic step limit stepLimit, and goal height
// goalHeight. For the default classic control case, the goal height
// should be set to the GoalHeight constant defined in this package.
func NewSwingUp(s environment.Starter, stepLimit int,
	goalHeight float64) environment.Task {
	stepLimitEnder := environment.NewStepLimit(stepLimit)

	// Function which determines if at goal
	endFunc := func(obs *mat.VecDense) bool {
		// Ensure state observations have at least two dimensions
		if obs.Len() < 2 {
			panic(fmt.Sprintf("end: state must consist of minimum "+
				"two features \n\twant(>2) \n\thave(%v)", obs.Len()))
		}

		return -math.Cos(obs.AtVec(0))-
			math.Cos(obs.AtVec(1)+obs.AtVec(0)) > goalHeight
	}

	lineEnder := environment.NewFunctionEnder(endFunc, ts.TerminalStateReached)

	return &SwingUp{s, stepLimitEnder, endFunc, lineEnder}
}

// AtGoal returns whether the argument state is a goal state
func (s *SwingUp) AtGoal(state mat.Matrix) bool {
	// Ensure that the argument state is a single state vector
	r, c := state.Dims()
	if c > 1 {
		panic("atGoal: state consist of a single observation")
	}

	// Convert the state to a vector
	stateVec, ok := state.(*mat.VecDense)
	if !ok {
		stateVec = mat.NewVecDense(r, nil)
		for i := 0; i < r; i++ {
			stateVec.SetVec(i, state.At(i, 0))
		}
	}
	return s.f(stateVec)
}

// End determines if a timestep is the last timestep in the episode.
// If so, it changes the TimeStep's StepType to timestep.Last and
// adjusts the TimeStep's EndType to the appropriate ending type. This
// function returns true if the argument TimeStep is the last timestep
// in the episode and false otherwise.
func (s *SwingUp) End(t *ts.TimeStep) bool {
	if ended := s.lineEnder.End(t); ended {
		return true
	}
	if ended := s.stepLimitEnder.End(t); ended {
		return true
	}
	return false
}

// GetReward returns the reward for a given state and action, resulting
// in a given next state. Since this is a cost-to-goal Task, rewards are
// -1.0 for all actions, except for an action which leads to the goal
// state, which results in a reward of 0.0.
func (s *SwingUp) GetReward(state, action, nextState mat.Vector) float64 {

	// Convert input nextState mat.Vector to *mat.VecDense to use with
	// ending function s.f
	var nextStateVecDense *mat.VecDense
	nextStateVecDense, ok := nextState.(*mat.VecDense)
	if !ok {
		nextStateVecDense := mat.NewVecDense(nextState.Len(), nil)
		for i := 0; i < nextState.Len(); i++ {
			nextStateVecDense.SetVec(i, nextState.AtVec(i))
		}
	}

	if terminalReached := s.f(nextStateVecDense); terminalReached {
		return maxReward
	} else {
		return minReward
	}
}

// Min returns the minimum attainable reward over all timesteps
func (s *SwingUp) Min() float64 {
	return minReward
}

// Max returns the maximum attainable reward over all timesteps
func (s *SwingUp) Max() float64 {
	return maxReward
}

// RewardSpec returns the reward environmentification for the environment
func (s *SwingUp) RewardSpec() environment.Spec {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{s.Min()})
	upperBound := mat.NewVecDense(1, []float64{s.Max()})

	return environment.NewSpec(shape, environment.Reward, lowerBound, upperBound,
		environment.Continuous)
}
