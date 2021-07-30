package pendulum

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
)

// SwingUp implements the classic control Pendulum swingup task. In
// this task, the agent must learn to swing the pendulum up and hold it
// in a vertical position facing upwards. The force that the agent
// applies to the pendulum is underpowered, so the agent must learn to
// rock the pendulum from side to side until it can an upwards
// position.
//
// Rewards are the cosine of the angle of the pendulum measured from the
// positive y-axis. For facing straight up, a reward of +1 is given.
// For facing straight down a reward of -1 is given.
//
// Episodes end after a step limit.
type SwingUp struct {
	environment.Starter
	environment.Ender // Ends when step limit reached
}

// NewSwingUp creates and returns a new SwingUp task
func NewSwingUp(s environment.Starter, maxSteps int) *SwingUp {
	ender := environment.NewStepLimit(maxSteps)
	return &SwingUp{s, ender}
}

// GetReward gets the reward at the current timestep
func (s *SwingUp) GetReward(state mat.Vector, _ mat.Vector,
	nextState mat.Vector) float64 {
	th := nextState.AtVec(0)
	return math.Cos(th)
}

// AtGoal determines whether or not the current state is the goal state
func (s *SwingUp) AtGoal(state mat.Matrix) bool {
	return state.At(0, 0) == 0
}

// Min returns the minimum possible reward
func (s *SwingUp) Min() float64 {
	return -1.0
}

// Max returns the maximum possible reward
func (s *SwingUp) Max() float64 {
	return 1.0
}

// RewardSpec returns the reward specification of the Task
func (s *SwingUp) RewardSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)

	minReward := s.Min()
	lowerBound := mat.NewVecDense(2, []float64{minReward})

	maxReward := s.Max()
	upperBound := mat.NewVecDense(1, []float64{maxReward})

	return spec.NewEnvironment(shape, spec.Reward, lowerBound, upperBound,
		spec.Continuous)
}
