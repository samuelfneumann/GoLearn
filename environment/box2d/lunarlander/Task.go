package lunarlander

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/environment"
)

type lunarLanderTask interface {
	environment.Task
	registerEnv(*lunarLander)
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

func (l *Land) resetShaping() {
	l.prevShaping = new(float64)
}

func (l *Land) AtGoal(state mat.Matrix) bool {
	leg1Contact, leg2Contact := l.env.GroundContact()
	return leg1Contact && leg2Contact
}

func (l *Land) GetReward(s, a, nextState mat.Vector) float64 {
	state := nextState.(*mat.VecDense).RawVector().Data

	reward := 0.0
	shaping := (-100 * math.Sqrt(state[0]*state[0]+state[1]*state[1])) +
		(-100 * math.Sqrt(state[2]*state[2]+state[3]*state[3])) +
		(-100 * math.Abs(state[4])) +
		(10 * state[6]) +
		(10 * state[7])

	if l.prevShaping != nil {
		reward = shaping - *l.prevShaping
	}
	*l.prevShaping = shaping

	// Less fuel spent is better
	reward -= (l.env.MPower() * 0.30)
	reward -= (l.env.SPower() * 0.03)

	if l.env.gameOver || math.Abs(nextState.AtVec(0)) >= 1.0 {
		reward = -100
	} else if !l.env.lander.IsAwake() {
		reward = 100
	}
	return reward
}
