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
			fmt.Errorf("newHopper: frameSkip should be positive")
	}
	m, err := mujocoenv.NewMujocoEnv("hopper.xml", frameSkip, seed,
		discount)
	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("newHopper: %v", err)
	}

	r := &Reacher{
		MujocoEnv: m,
		Task:      t,
		obsLen:    m.Nq + 7,
	}

	firstStep := r.Reset()
	return r, firstStep, nil
}

// Argument action is modified!
func (r *Reacher) Step(action *mat.VecDense) (ts.TimeStep, bool) {
	// distance, err := r.fingerToTargetDistance()
	// if err != nil {
	// 	panic(fmt.Sprintf("step: could not get distance to target: %v", err))
	// }

	// Get the state
	state, err := r.GetBodyCentreOfMass("fingertip")
	if err != nil {
		panic(fmt.Sprintf("step: could not get fingertip centre of mass "+
			"for state calculation: %v",
			err))
	}

	// Run simulation, then get the next state
	r.DoSimulation(action, r.FrameSkip)
	nextState, err := r.GetBodyCentreOfMass("fingertip")
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

	t := ts.New(ts.Mid, reward, r.Discount, obs, r.currentTimeStep.Number+1)
	done := r.End(&t)

	return t, done

}

// Reset resets the environment to some starting state
func (r *Reacher) Reset() ts.TimeStep {
	// Reset the embedded base MujocoEnv
	r.MujocoEnv.Reset()

	// Get and set the starting state for the next episode
	startVec := r.Start()
	posStart := startVec.RawVector().Data[:r.Nq] // ! This should have last two elements being the goal state position
	velStart := startVec.RawVector().Data[r.Nq:] // ! This should have last two elements being the goal state velocity

	// Goal should not be moving
	velStart[len(velStart)-1] = 0.0
	velStart[len(velStart)-2] = 0.0

	r.SetState(posStart, velStart)

	// Save the current timestep
	obs, err := r.getObs()
	if err != nil {
		panic(fmt.Sprintf("reset: could not get state observation: %v", err))
	}
	firstStep := ts.New(ts.First, 0, r.Discount, obs, 0)
	r.currentTimeStep = firstStep

	return firstStep
}

// CurrentTimeStep returns the current time step
func (r *Reacher) CurrentTimeStep() ts.TimeStep {
	return r.currentTimeStep
}

func (r *Reacher) fingerToTargetDistance() ([]float64, error) {
	centreOfMassFinger, err := r.GetBodyCentreOfMass("fingertip")
	if err != nil {
		return nil, fmt.Errorf("fingerToTargetDistance: could not get "+
			"fingertip centre of mass: %v", err)
	}
	centreOfMassTarget, err := r.GetBodyCentreOfMass("target")
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

	distance, err := r.fingerToTargetDistance()
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
