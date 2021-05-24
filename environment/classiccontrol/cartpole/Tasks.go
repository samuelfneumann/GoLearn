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

type Balance struct {
	env.Starter
	stepLimiter  *env.StepLimit
	angleLimiter *env.IntervalLimit
	failAngle    float64
}

func NewBalance(s env.Starter, episodeSteps int, failAngle float64) *Balance {
	stepLimiter := env.NewStepLimit(episodeSteps)

	// Create the Enders
	legalAngles := []r1.Interval{{Min: -failAngle, Max: failAngle}}
	angleFeatureIndex := []int{2}

	angleLimiter := env.NewIntervalLimit(legalAngles, angleFeatureIndex)

	return &Balance{s, stepLimiter, angleLimiter, failAngle}
}

func (b *Balance) End(t *ts.TimeStep) bool {
	if end := b.angleLimiter.End(t); end {
		return true
	}
	if end := b.stepLimiter.End(t); end {
		return true
	}
	return false
}

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

func (b *Balance) AtGoal(state mat.Matrix) bool {
	return math.Abs(state.At(0, 2)) > b.failAngle
}

func (b *Balance) Min() float64 {
	return -1.0
}

func (b *Balance) Max() float64 {
	return 1.0
}

func (b *Balance) RewardSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{b.Min()})
	upperBound := mat.NewVecDense(1, []float64{b.Max()})

	return spec.NewEnvironment(shape, spec.Reward, lowerBound, upperBound,
		spec.Continuous)
}
