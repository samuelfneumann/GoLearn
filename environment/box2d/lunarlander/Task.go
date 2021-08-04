package lunarlander

import (
	"math"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
)

// lunarLanderTask implements functionality for a Task to track internal
// lunarLander variables needed for reward computation, goal compuation,
// etc.
type lunarLanderTask interface {
	environment.Task
	registerEnv(*lunarLander)
	reset()
}

// Land implements the task where an agent must learn to land the
// Lunar Lander on the moon without the body of the ship touching
// any objects or any boundary. Reward for successfully landing on the
// moon is 100. In the Lunar Lander environment, a landing pad exists
// at (0, 0) always. If the ship moves away from this landing pad,
// a small negative reward is given. A negative reward is also given
// proportional to how much power each engine uses. For successfully
// landing each leg on the moon without crashing, the agent gets
// an additional 10 reward per leg. If the environment experiences
// a game over (the lander's body collides with anything), then a
// reward of -100 is given. For this task, only the legs should ever
// collide with any other objects.
type Land struct {
	environment.Starter
	stepLimit environment.Ender

	prevShaping *float64

	env *lunarLander
}

// NewLand returns a new Land task
func NewLand(s environment.Starter, cutoff int) lunarLanderTask {
	stepLimit := environment.NewStepLimit(cutoff)

	return &Land{Starter: s, stepLimit: stepLimit, prevShaping: new(float64)}
}

// registerEnv registers a lunarLander environment with the task so
// that internals about the environment can be used to construct
// rewards, state features, etc.
func (l *Land) registerEnv(env *lunarLander) {
	l.env = env
}

// reset performs cleaning and should be called at the end of an
// episode to clean up any variables that should be reset between
// episodes.
func (l *Land) reset() {
	l.prevShaping = new(float64)
}

// AtGoal returns whether the argument state is a goal state
func (l *Land) AtGoal(state mat.Matrix) bool {
	leg1Contact, leg2Contact := l.env.GroundContact()
	return leg1Contact && leg2Contact
}

// GetReward returns the reward for taking the argument action in the
// argument state, resulting in a transition to the argument nextState.
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

	if l.env.gameOver || math.Abs(nextState.AtVec(0)) >= 1.0 {
		// Second case will never be true due to boundaries on the
		// lunarLander environment.
		reward = -100
	} else if !l.env.lander.IsAwake() {
		reward = 100
	}
	return reward
}

// End checks if the argument timestep is the last in the episode
// and changes the step's type to reflect this.
func (l *Land) End(t *ts.TimeStep) bool {
	var done bool
	if l.env.IsGameOver() {
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

// Max returns the maximum attainable reward in the environment
func (l *Land) Max() float64 {
	return 100.0
}

// Min returns the minimum attainable reward in the environment
func (l *Land) Min() float64 {
	// Technically, this reward can be achieved by this task, but it
	// most likely will never occur.
	return math.Inf(-1)
}

// RewardSpec returns the reward specification for the environment
func (l *Land) RewardSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)

	lowerBound := mat.NewVecDense(1, []float64{l.Min()})
	upperBound := mat.NewVecDense(1, []float64{l.Max()})

	return spec.NewEnvironment(shape, spec.Reward, lowerBound, upperBound,
		spec.Continuous)
}
