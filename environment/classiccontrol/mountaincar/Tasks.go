package mountaincar

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
)

const (
	// Commonly used goal position
	GoalPosition float64 = 0.45
)

// Goal implements the classic control task of reaching a goal on
// Mountain Car. In this task, the agent must learn to drive the car
// up the hill and reach the goal state. Since the car is underpowered,
// it must rock back and forth from hill to hill until it reaches the
// goal.
//
// Rewards are -1 on each timestep and 0 for the action which
// transitions the car to the goal.
//
// Episodes end after a step limit or when the car reaches the goal
// state.
type Goal struct {
	environment.Starter
	goalEnder *environment.IntervalLimit
	stepEnder *environment.StepLimit
	goalX     float64 // x position of goal
}

// NewGoal creates and returns a new Goal struct given a Starter, which
// determines the starting states; the maximum number of episode
// steps; and the goal x position.
func NewGoal(s environment.Starter, episodeSteps int, goalX float64) *Goal {
	stepEnder := environment.NewStepLimit(episodeSteps)

	interval := []r1.Interval{{Min: math.Inf(-1), Max: goalX}}
	positionIndex := []int{0}
	goalEnder := environment.NewIntervalLimit(interval, positionIndex,
		timestep.TerminalStateReached)
	return &Goal{s, goalEnder, stepEnder, goalX}
}

// AtGoal returns a boolean indicating whether or not the argument state
// is the goal state
func (g *Goal) AtGoal(state mat.Matrix) bool {
	return state.At(0, 0) >= g.goalX
}

// GetReward returns the reward for a given state and action, resulting
// in a given next state. Since this is a cost-to-goal Task, rewards are
// -1.0 for all actions, except for an action which leads to the goal
// state, which results in a reward of 0.0
func (g *Goal) GetReward(state mat.Vector, _ mat.Vector,
	nextState mat.Vector) float64 {
	xPosition := nextState.AtVec(0)

	if xPosition >= g.goalX {
		return 0.0
	}
	return -1.0
}

// Min returns the minimum attainable reward over all timesteps
func (g *Goal) Min() float64 { return -1.0 }

// Max returns the maximum attainable reward over all timesteps
func (g *Goal) Max() float64 { return 0.0 }

// RewardSpec returns the reward specification of the Task
func (g *Goal) RewardSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{g.Min()})
	upperBound := mat.NewVecDense(1, []float64{g.Max()})

	return spec.NewEnvironment(shape, spec.Reward, lowerBound, upperBound,
		spec.Discrete)
}

// End determines if a timestep is the last timestep in the episode.
// If so, it changes the TimeStep's StepType to timestep.Last. This
// function returns true if the argument timestep is the last timestep
// in the episode and false otherwise.
//
// This function is needed to ensure that the Goal struct implements the
// Ender interface, since a Goal Task has two Ender fields, but no
// embedded Ender struct. This function function wraps both fields,
// using each field's End() method to determine if the episode has
// ended or not.
func (g *Goal) End(t *timestep.TimeStep) bool {
	// Check if the goal was reached, modifying t.StepType if appropriate
	if end := g.goalEnder.End(t); end {
		return true
	}

	// Check if the max steps was reached, modifying t.StepType if appropriate
	if end := g.stepEnder.End(t); end {

		return true
	}
	return false
}
