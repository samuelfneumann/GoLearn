// Package reacher implements the reacher environment
package reacher

import (
	"fmt"
	"math"

	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/environment/mujoco/internal/mujocoenv"
	ts "github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/golearn/utils/floatutils"
	"gonum.org/v1/gonum/mat"
)

// * One thing that this implementation does differently from OpenAI Gym:
// * Gym computes the reward for some action based on the previous state.
// * For the tuples (S A R S'), R is calculated based on the distance of
// * S from the target. This codebase computes the reward based on the
// * distance between S' and the target, which is the correct implementation.
// *
// * Another thing: actions are clipped to stay within their legal bounds
// * before being sent to the simulator, but rewards are constructed based
// * on the unclipped actions. Otherwise, the simulator's internal state
// * could contain NaN values. This "glitch" is also present in OpenAI gym,
// * but they leave it up to the agent to perform the clipping instead.
type Reacher struct {
	*mujocoenv.MujocoEnv
	environment.Task
	obsLen          int
	currentTimeStep ts.TimeStep
}

func New(t environment.Task, frameSkip int, seed uint64,
	discount float64) (environment.Environment, ts.TimeStep, error) {
	if frameSkip < 0 {
		return nil, ts.TimeStep{},
			fmt.Errorf("newReacher: frameSkip should be positive")
	}
	m, err := mujocoenv.NewMujocoEnv("reacher.xml", frameSkip, seed,
		discount)
	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("newReacher: %v", err)
	}

	r := &Reacher{
		MujocoEnv: m,
		Task:      t,
		obsLen:    m.Nq + 7,
	}

	reach, ok := t.(*Reach)
	if ok {
		reach.register(r)
	}

	firstStep := r.Reset()
	return r, firstStep, nil
}

// ! Clipping action causes it to no longer produce warning...
func (r *Reacher) Step(action *mat.VecDense) (ts.TimeStep, bool) {
	// Get the state
	state, err := r.BodyXPos("fingertip")
	if err != nil {
		panic(fmt.Sprintf("step: could not get fingertip centre of mass "+
			"for state calculation: %v",
			err))
	}

	// Run simulation, then get the next state
	newAction := r.clipAction(action)
	r.DoSimulation(newAction, r.FrameSkip)
	nextState, err := r.BodyXPos("fingertip")
	if err != nil {
		panic(fmt.Sprintf("step: could not get fingertip centre of mass "+
			"for next state calculation: %v",
			err))
	}
	reward := r.GetReward(state, action, nextState)

	obs, err := r.getObs()
	if err != nil {
		panic(fmt.Sprintf("step: could not get next state observation: %v",
			err))
	}

	t := ts.New(ts.Mid, reward, r.Discount, obs, r.CurrentTimeStep().Number+1)
	r.currentTimeStep = t
	done := r.End(&t)

	return t, done

}

// clipAction returns a copy of the argument action which is clipped to
// be within the action bounds of the environment.
func (r *Reacher) clipAction(action *mat.VecDense) *mat.VecDense {
	spec := r.ActionSpec()
	min := spec.LowerBound
	max := spec.UpperBound

	clipped := mat.VecDenseCopyOf(action)

	for i := 0; i < clipped.Len(); i++ {
		clipped.SetVec(i, floatutils.Clip(clipped.AtVec(i), min.AtVec(i),
			max.AtVec(i)))
	}

	return clipped
}

// ! WARNING: Unknown warining type Time = X.XXXX is from resetting!
// Reset resets the environment to some starting state
func (r *Reacher) Reset() ts.TimeStep {
	// Reset the embedded base MujocoEnv
	r.MujocoEnv.Reset()

	// Get and set the starting state for the next episode
	startVec := r.Start()
	posStart := startVec.RawVector().Data[:r.Nq]
	velStart := startVec.RawVector().Data[r.Nq:]

	r.SetState(posStart, velStart)

	// Save the current timestep
	obs, err := r.getObs()
	if err != nil {
		panic(fmt.Sprintf("reset: could not get state observation: %v", err))
	}
	firstStep := ts.New(ts.First, 0, r.Discount, obs, 0)
	r.currentTimeStep = firstStep

	// fmt.Println("\n", posStart, velStart)
	// fmt.Println()

	return firstStep
}

// CurrentTimeStep returns the current time step
func (r *Reacher) CurrentTimeStep() ts.TimeStep {
	return r.currentTimeStep
}

func (r *Reacher) fingerToTargetVector() ([]float64, error) {
	centreOfMassFinger, err := r.BodyXPos("fingertip")
	if err != nil {
		return nil, fmt.Errorf("fingerToTargetDistance: could not get "+
			"fingertip centre of mass: %v", err)
	}
	centreOfMassTarget, err := r.BodyXPos("target")
	if err != nil {
		return nil, fmt.Errorf("fingerToTargetDistance: could not get "+
			"target centre of mass: %v", err)
	}
	distance := make([]float64, centreOfMassFinger.Len())
	for i := range distance {
		distance[i] = centreOfMassFinger.AtVec(i) - centreOfMassTarget.AtVec(i)
	}

	return distance, nil
}

func (r *Reacher) getObs() (*mat.VecDense, error) {
	pos := r.QPos()
	vel := r.QVel()

	distance, err := r.fingerToTargetVector()
	if err != nil {
		return nil, fmt.Errorf("getObs: %v", err)
	}

	theta := []float64{pos[0], pos[1]}
	cosTheta := floatutils.PreserveApply(theta, math.Cos)
	sinTheta := floatutils.Apply(theta, math.Sin)

	obs := make([]float64, r.obsLen)
	copy(obs[:2], cosTheta)
	copy(obs[2:4], sinTheta)
	copy(obs[4:r.Nq+2], pos[2:])
	copy(obs[r.Nq+2:r.Nq+4], vel[:2])
	copy(obs[len(obs)-3:], distance)

	return mat.NewVecDense(r.obsLen, obs), nil
}

func (r *Reacher) ObservationSpec() environment.Spec {
	shape := mat.NewVecDense(r.obsLen, nil)

	low := mat.NewVecDense(r.obsLen, nil)
	high := mat.NewVecDense(r.obsLen, nil)
	for i := 0; i < low.Len(); i++ {
		low.SetVec(i, math.Inf(-1))
		high.SetVec(i, math.Inf(1))
	}

	return environment.NewSpec(shape, environment.Observation, low, high,
		environment.Continuous)
}
