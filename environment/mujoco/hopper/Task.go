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

type Hop struct {
	env        *Hopper
	registered bool

	stepLimit environment.Ender

	seed   uint64
	posRng *distmv.Uniform
	velRng *distmv.Uniform
}

func NewHop(seed uint64, cutoff int) environment.Task {
	stepLimit := environment.NewStepLimit(cutoff)

	return &Hop{
		seed:       seed,
		registered: false,
		stepLimit:  stepLimit,
	}
}

func (h *Hop) AtGoal(state mat.Matrix) bool {
	fmt.Fprintf(os.Stderr, "atGoal: no goal state for Hop task")
	return false
}

func (h *Hop) End(t *ts.TimeStep) bool {
	// Get the next state observation
	nextObs := h.env.QPos()
	height, angle := nextObs[1], nextObs[2]

	s := mujocoenv.StateVector(h.env.Data, h.env.Nq, h.env.Nv)
	count := 0
	for i := 0; i < s.Len(); i++ {
		if math.IsNaN(s.AtVec(i)) || math.Abs(s.AtVec(i)) == math.Inf(1.0) {
			fmt.Println("IS NAN")
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

	if done := h.stepLimit.End(t); done {
		return true
	}

	return false
}

func (h *Hop) GetReward(state, action, nextState mat.Vector) float64 {
	posBefore := state.AtVec(0)
	posAfter := nextState.AtVec(0)

	alive_bonus := 1.0
	reward := (posAfter - posBefore) / h.env.Dt()
	reward += alive_bonus
	reward -= 1e-3 * mat.Dot(action, action)

	return reward
}

func (h *Hop) RewardSpec() environment.Spec {
	shape := mat.NewVecDense(1, nil)
	low := mat.NewVecDense(1, []float64{h.Min()})
	high := mat.NewVecDense(1, []float64{h.Max()})

	return environment.NewSpec(shape, environment.Reward, low, high,
		environment.Continuous)
}

func (h *Hop) Max() float64 {
	return math.Inf(1.0)
}

func (h *Hop) Min() float64 {
	return math.Inf(-1.0)
}

func (h *Hop) Start() *mat.VecDense {
	if !h.registered {
		panic("start: no registered Hopper environment to start")
	}
	posRand := h.posRng.Rand(nil)
	posStart := mat.NewVecDense(len(posRand), posRand)
	posStart.AddVec(posStart, h.env.InitQPos)

	velRand := h.velRng.Rand(nil)
	velStart := mat.NewVecDense(len(velRand), velRand)
	velStart.AddVec(velStart, h.env.InitQVel)

	// Gets the starting state, not the starting observation
	return mat.NewVecDense(
		h.env.Nq+h.env.Nv,
		append(posStart.RawVector().Data, velStart.RawVector().Data...),
	)
}

func (h *Hop) registerHopper(env *Hopper) {
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
