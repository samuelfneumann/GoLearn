package cartpole

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
)

const (
	FailAngle float64 = 12 * 2 * math.Pi / 360
)

// Balance implements the classic control Cartpole Balance task. In this
// Task, the goal of the agent is to balance the pole on the cart in
// an upright position for as long as possible. Goal positions consist
// of the pole above some set angle threshold θ.
//
// The rewards are +1 for every timestep and -1 when the pole has fallen
// below some set angle threshold θ.
//
// Episodes end after a step limit or after the pole has fallen below
// some angle threshold θ.
type Balance struct {
	env.Starter
	stepLimiter  *env.StepLimit
	angleLimiter *env.IntervalLimit
	failAngle    float64
}

// NewBalance creates and returns a new Balance task
func NewBalance(s env.Starter, episodeSteps int, failAngle float64) *Balance {
	stepLimiter := env.NewStepLimit(episodeSteps)

	// Create the Enders
	legalAngles := []r1.Interval{{Min: -failAngle, Max: failAngle}}
	angleFeatureIndex := []int{2}

	angleLimiter := env.NewIntervalLimit(legalAngles, angleFeatureIndex)

	return &Balance{s, stepLimiter, angleLimiter, failAngle}
}

// End checks if a TimeStep is the last in an episode. If so, it adjusts
// the TimeStep's StepType to timestep.Last and returns true. Otherwise,
// the function does not adjust the TimeStep and returns false.
func (b *Balance) End(t *ts.TimeStep) bool {
	if end := b.angleLimiter.End(t); end {
		return true
	}
	if end := b.stepLimiter.End(t); end {
		return true
	}
	return false
}

// GetReward returns the reward for an action taken in some state,
// resulting in a transition to the next state nextState.
func (b *Balance) GetReward(_ mat.Vector, _ mat.Vector,
	nextState mat.Vector) float64 {
	angle := math.Abs(nextState.AtVec(2))

	// Angle of 0 is pointing straight up, so we want angles to be
	// less than the failAngle
	if angle < b.failAngle {
		return 1.0
	}
	return -1.0
}

// AtGoal returns whether or not the goal position has been reached.
func (b *Balance) AtGoal(state mat.Matrix) bool {
	return math.Abs(state.At(0, 2)) > b.failAngle
}

// Min returns the minimum possible reward that can be received in the
// environment
func (b *Balance) Min() float64 {
	return -1.0
}

// Max returns the maximum possible reward that can be received in the
// environment
func (b *Balance) Max() float64 {
	return 1.0
}

// RewardSpec returns the reward specification for the environment
func (b *Balance) RewardSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{b.Min()})
	upperBound := mat.NewVecDense(1, []float64{b.Max()})

	return spec.NewEnvironment(shape, spec.Reward, lowerBound, upperBound,
		spec.Continuous)
}
