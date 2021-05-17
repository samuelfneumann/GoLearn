package pendulum

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
)

// SwingUp implements a task where the agent must swing the pendulum up
// and hold it in a vertical position. Rewards are the cosine of the
// pendulum angle measured from the positive y-axis. The goal state
// is the pendulum sticking straight up, at which point the agent gets
// a reward of 1.0 on each timestep
type SwingUp struct {
	environment.Starter
	environment.Ender
}

// NewSwingUp creates and returns a new SwingUp task
func NewSwingUp(s environment.Starter, maxSteps int) *SwingUp {
	ender := environment.NewStepLimit(maxSteps)
	return &SwingUp{s, ender}
}

// GetReward gets the reward at the current timestep
func (s *SwingUp) GetReward(t timestep.TimeStep, _ mat.Vector) float64 {
	th := t.Observation.AtVec(0)
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
