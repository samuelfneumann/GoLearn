package lunarlander

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
)

type lunarLanderTask interface {
	environment.Task
	registerEnv(*lunarLander)
	reset()
}

type Land struct {
	environment.Starter
	stepLimit environment.Ender

	prevShaping *float64

	env *lunarLander
}

func NewLand(s environment.Starter, cutoff int) lunarLanderTask {
	stepLimit := environment.NewStepLimit(cutoff)

	return &Land{Starter: s, stepLimit: stepLimit, prevShaping: new(float64)}
}

func (l *Land) registerEnv(env *lunarLander) {
	l.env = env
}

func (l *Land) reset() {
	l.prevShaping = new(float64)
}

func (l *Land) AtGoal(state mat.Matrix) bool {
	leg1Contact, leg2Contact := l.env.GroundContact()
	return leg1Contact && leg2Contact
}

func (l *Land) GetReward(_, _, nextState mat.Vector) float64 {
	reward := 0.0
	shaping := (-100 * math.Sqrt(nextState.AtVec(0)*nextState.AtVec(0)+
		nextState.AtVec(1)*nextState.AtVec(1))) +
		(-100 * math.Sqrt(nextState.AtVec(2)*nextState.AtVec(2)+
			nextState.AtVec(3)*nextState.AtVec(3))) +
		(-100 * math.Abs(nextState.AtVec(4))) +
		(10 * nextState.AtVec(6)) +
		(10 * nextState.AtVec(7))

	if l.prevShaping != nil {
		reward = shaping - *l.prevShaping
	}
	*l.prevShaping = shaping

	// Less fuel spent is better
	reward -= (l.env.MPower() * 0.30)
	reward -= (l.env.SPower() * 0.03)

	if l.env.gameOver || math.Abs(nextState.AtVec(0)) >= 1.0 ||
		math.Abs(nextState.AtVec(1)) >= 1.0 {
		reward = -100
	} else if !l.env.lander.IsAwake() {
		reward = 100
	}
	return reward
}

func (l *Land) End(t *ts.TimeStep) bool {
	var done bool
	if l.env.IsGameOver() && false {
		done = true
	} else if !l.env.Lander().IsAwake() {
		done = true
	} else if math.Abs(t.Observation.AtVec(0)) >= 1.0 {
		// Due to the boundaries on the lunarLander environment, this
		// case should never run
		done = true
	}

	if done {
		t.StepType = ts.Last
		t.SetEnd(ts.TerminalStateReached)
	} else {
		l.stepLimit.End(t)
	}

	return t.Last()
}

func (l *Land) Max() float64 {
	return 100.0
}

func (l *Land) Min() float64 {
	// Technically, this reward can be achieved by this task, but it
	// most likely will never occur.
	return math.Inf(-1)
}

func (l *Land) RewardSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)

	lowerBound := mat.NewVecDense(1, []float64{l.Min()})
	upperBound := mat.NewVecDense(1, []float64{l.Max()})

	return spec.NewEnvironment(shape, spec.Reward, lowerBound, upperBound,
		spec.Continuous)
}
