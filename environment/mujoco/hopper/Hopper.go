// Package hopper implements the Hopper environment. This environment
// is conceptually similar to the Hopper-v2 environment of OpenAI Gym,
// which can be found at https://gym.openai.com/envs/Hopper-v2/.
// Major differences between this implementation and the OpenAI Gym
// implementation can be found in the documentation comment for
// the Hopper struct.
package hopper

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"

	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/environment/mujoco/internal/mujocoenv"
	ts "github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/golearn/utils/floatutils"
)

// Hopper implements the Hopper environment. In this environment, an
// agent control a "hopper": a creature composed of a single leg. The
// agent can control three of its joints to hop or move around. The
// movable joints are the thigh joint, leg join, and foot joint.
//
// The state observation space is a vector of 11 components:
// [
// 		torso Z position
//		torso Y orientation
//		thigh -Y orientation
//		leg -Y orientation
//		foot -Y orientation
//		torso X linear velocity
//		torso Z linear velocity
//		torso Y anglular velocity
//		thigh Y angular velocity
//		leg Y angular velocity
//		foot Y angular velocity
// ]
// where +/-Y denotes rotation about the positive or negative Y axis
// respectively. All features of the state space are unbounded.
// Velocities are clipped within [-10, 10]
//
// Action are continuous and consist of the torque to apply at each of
// the three movable joints and are bounded between [-1, -1, -1] and
// [1, 1, 1] element-wise. Actions outside of this range are dealt with
// by the MuJoCo physics engine.
//
// Hopper satisfies the environment.Environment interface.
//
// One major difference between this implementation and the OpenAI Gym
// implementation is that this implementation allows the user to set
// the number of frames to skip, whereas OpenAI Gym sets this value to
// 4 always.
//
// See https://gym.openai.com/envs/Hopper-v2/
type Hopper struct {
	*mujocoenv.MujocoEnv
	environment.Task
	obsLen          int
	currentTimeStep ts.TimeStep
}

// New returns a new Hopper environment
func New(t environment.Task, frameSkip int, seed uint64,
	discount float64) (environment.Environment, ts.TimeStep, error) {
	if frameSkip < 0 {
		return nil, ts.TimeStep{},
			fmt.Errorf("newHopper: frameSkip should be positive")
	}
	m, err := mujocoenv.NewMujocoEnv("hopper.xml", frameSkip, seed,
		discount)
	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("newHopper: %v", err)
	}

	h := &Hopper{
		MujocoEnv: m,
		Task:      t,
		obsLen:    m.Nq - 1 + m.Nv,
	}

	// Register task with Hopper environment if appropriate
	_, ok := h.Task.(*Hop)
	if ok {
		h.Task.(*Hop).register(h)
	}

	firstStep := h.Reset()
	return h, firstStep, nil
}

// CurrentTimeStep returns the current time step
func (h *Hopper) CurrentTimeStep() ts.TimeStep {
	return h.currentTimeStep
}

// Step takes one environmental step given some action control
func (h *Hopper) Step(action *mat.VecDense) (ts.TimeStep, bool) {
	state := mujocoenv.StateVector(h.Data, h.Nq, h.Nv)

	// Set the action
	if action.Len() != h.Nu {
		panic(fmt.Sprintf("step: invalid number of action dimensions \n\t"+
			"have(%v) \n\twant(%v)", action.Len(), h.Nu))
	}

	h.DoSimulation(action, h.FrameSkip)

	nextState := mujocoenv.StateVector(h.Data, h.Nq, h.Nv)
	reward := h.GetReward(state, action, nextState)

	t := ts.New(ts.Mid, reward, h.Discount, h.getObs(),
		h.CurrentTimeStep().Number+1)
	last := h.End(&t)
	h.currentTimeStep = t

	return t, last
}

// getObs returns the current state observation of the environment
func (h *Hopper) getObs() *mat.VecDense {
	pos := h.QPos()
	vel := floatutils.ClipSlice(h.QVel(), -10.0, 10.0)

	return mat.NewVecDense(h.obsLen, append(pos[1:], vel...))
}

// Reset resets the environment to some starting state
func (h *Hopper) Reset() ts.TimeStep {
	// Reset the embedded base MujocoEnv
	h.MujocoEnv.Reset()

	// Get and set the starting state for the next episode
	startVec := h.Start()
	posStart := startVec.RawVector().Data[:h.Nq]
	velStart := startVec.RawVector().Data[h.Nq:]
	h.SetState(posStart, velStart)

	// Save the current timestep
	firstStep := ts.New(ts.First, 0, h.Discount, h.getObs(), 0)
	h.currentTimeStep = firstStep

	return firstStep
}

// ObservationSpec returns the observation specification for the
// Hopper environment
func (h *Hopper) ObservationSpec() environment.Spec {
	shape := mat.NewVecDense(h.obsLen, nil)
	low := make([]float64, h.obsLen)
	high := make([]float64, h.obsLen)
	for i := range high {
		high[i] = math.Inf(1.0)
		low[i] = math.Inf(-1.0)
	}

	lowVec := mat.NewVecDense(h.obsLen, low)
	highVec := mat.NewVecDense(h.obsLen, high)

	return environment.NewSpec(shape, environment.Observation, lowVec, highVec,
		environment.Continuous)
}
