package hopper

import (
	"fmt"
	"math"
	"os"

	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/environment/mujoco/internal/mujocoenv"
	ts "github.com/samuelfneumann/golearn/timestep"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	"gonum.org/v1/gonum/stat/distmv"
)

// Hop implements the "regular" Hop task for the Hopper environment.
// This is the same task as that in OpenAI Gym Hopper-v2. In this
// task, the agent is rewarded based on how far its current X position
// is from its X position at the previous timestep. Note that the reward
// depends on the X position, but that the observation space of Hopper
// does not include the X position.
//
// Episodes are ended when a timestep limit is reached or:
//		1. An illegal value exists in the underlying state, such as
//		   NaN or ±Inf
//		2. The height (z-position of the torso) goes below 0.7
//		3. The angle (y-angle of the torso) goes above 0.2
//		4. At least one element of the state observation vector (except
// 		   the first - z-position) is not within [-100, 100]
type Hop struct {
	env *Hopper // Registered Hopper environment

	// registered denotes whether or not a Hopper environment has been
	// registered with the task
	registered bool

	stepLimit environment.Ender // Step limit ender

	// Random number generation for starting states
	seed   uint64
	posRng *distmv.Uniform
	velRng *distmv.Uniform
}

// NewHop returns a new Hopper environment
func NewHop(seed uint64, cutoff int) environment.Task {
	stepLimit := environment.NewStepLimit(cutoff)

	return &Hop{
		seed:       seed,
		registered: false,
		stepLimit:  stepLimit,
	}
}

// AtGoal satisfies the environment.Task interface. Since Hopper does
// not have a goal state, this function simply prints an error message
// to standard error.
func (h *Hop) AtGoal(state mat.Matrix) bool {
	if !h.registered {
		panic("atGoal: no registered Hopper environment")
	}

	fmt.Fprintf(os.Stderr, "atGoal: no goal state for Hop task")
	return false
}

// End checks if a timestep should be the last in the episode and
// adjusts the timestep accordingly. End returns whether the argument
// timestep is the last in the episode.
func (h *Hop) End(t *ts.TimeStep) bool {
	if !h.registered {
		panic("end: no registered Hopper environment to end")
	}

	// Get the next state observation
	nextObs := h.env.QPos()
	height, angle := nextObs[1], nextObs[2]

	// Check if done
	s := mujocoenv.StateVector(h.env.Data, h.env.Nq, h.env.Nv)
	count := 0
	for i := 0; i < s.Len(); i++ {
		if math.IsNaN(s.AtVec(i)) || math.Abs(s.AtVec(i)) == math.Inf(1.0) {
			t.StepType = ts.Last
			t.SetEnd(ts.TerminalStateReached)
			return true
		}

		if i >= 2 && math.Abs(s.AtVec(i)) < 100 {
			count++
		}
	}
	if count != s.Len()-2 {
		t.StepType = ts.Last
		t.SetEnd(ts.TerminalStateReached)
		return true
	}
	if height < 0.7 || math.Abs(angle) > 0.2 {
		t.StepType = ts.Last
		t.SetEnd(ts.TerminalStateReached)
		return true
	}

	// Check if timelimit reached
	if done := h.stepLimit.End(t); done {
		return true
	}

	return false
}

// GetReward returns the reward for a state, action, next state
// transition.
func (h *Hop) GetReward(state, action, nextState mat.Vector) float64 {
	if !h.registered {
		panic("getReward: no registered Hopper environment to get reward of")
	}

	posBefore := state.AtVec(0)
	posAfter := nextState.AtVec(0)

	alive_bonus := 1.0
	reward := (posAfter - posBefore) / h.env.Dt()
	reward += alive_bonus
	reward -= 1e-3 * mat.Dot(action, action)

	return reward
}

// RewardSpec returns the reward specification for the environment
func (h *Hop) RewardSpec() environment.Spec {
	shape := mat.NewVecDense(1, nil)
	low := mat.NewVecDense(1, []float64{h.Min()})
	high := mat.NewVecDense(1, []float64{h.Max()})

	return environment.NewSpec(shape, environment.Reward, low, high,
		environment.Continuous)
}

// Max returns the maximum possible reward
func (h *Hop) Max() float64 {
	return math.Inf(1.0)
}

// Min returns the minimum possible reward
func (h *Hop) Min() float64 {
	return math.Inf(-1.0)
}

// Start returns a new starting state for the task
func (h *Hop) Start() *mat.VecDense {
	if !h.registered {
		panic("start: no registered Hopper environment to start")
	}

	// Get random starting position
	posRand := h.posRng.Rand(nil)
	posStart := mat.NewVecDense(len(posRand), posRand)
	posStart.AddVec(posStart, h.env.InitQPos)

	// Get random starting velocity
	velRand := h.velRng.Rand(nil)
	velStart := mat.NewVecDense(len(velRand), velRand)
	velStart.AddVec(velStart, h.env.InitQVel)

	backing := make([]float64, posStart.Len()+velStart.Len())
	copy(backing[:posStart.Len()], posStart.RawVector().Data)
	copy(backing[posStart.Len():], velStart.RawVector().Data)

	// Get the starting state, not the starting observation
	return mat.NewVecDense(h.env.Nq+h.env.Nv, backing)
}

// register registers the Hop task with a Hopper environment.
// This is required since the Hop task needs access to some of the
// Hopper methods to correctly compute starting and ending states.
func (h *Hop) register(env *Hopper) {
	// Track the Hopper environment
	h.env = env

	// Position starting RNG
	posSrc := rand.NewSource(h.seed)
	posBounds := make([]r1.Interval, h.env.Nq)
	for i := range posBounds {
		posBounds[i] = r1.Interval{Min: -0.005, Max: 0.005}
	}
	h.posRng = distmv.NewUniform(posBounds, posSrc)

	// Velocity starting RNG
	velSrc := rand.NewSource(h.seed)
	velBounds := make([]r1.Interval, h.env.Nv)
	for i := range velBounds {
		velBounds[i] = r1.Interval{Min: -0.005, Max: 0.005}
	}
	h.velRng = distmv.NewUniform(velBounds, velSrc)

	h.registered = true
}
