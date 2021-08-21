package hopper

// TODO: Remove C and unsafe from public API

// * Leaving the cgo directives in so VSCode doesn't complain, even though
// * CGO_CFLAGS and CGO_LDFLAGS have been set.

// #cgo CFLAGS: -O2 -I/home/samuel/.mujoco/mujoco200_linux/include -mavx -pthread
// #cgo LDFLAGS: -L/home/samuel/.mujoco/mujoco200_linux/bin -lmujoco200nogl
// #include "mujoco.h"
// #include <stdio.h>
// #include <stdlib.h>
import "C"

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"

	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/environment/mujoco/internal/mujocoenv"
	ts "github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/golearn/utils/floatutils"
)

type Hopper struct {
	*mujocoenv.MujocoEnv
	environment.Task

	obsLen int

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

	h := &Hopper{
		MujocoEnv: m,
		Task:      t,
		obsLen:    m.Nq - 1 + m.Nv,
	}
	_, ok := h.Task.(*Hop)
	if ok {
		h.Task.(*Hop).registerHopper(h)
	}

	firstStep := h.Reset()

	return h, firstStep, nil
}

func (h *Hopper) CurrentTimeStep() ts.TimeStep {
	return h.currentTimeStep
}

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

func (h *Hopper) getObs() *mat.VecDense {
	pos := h.QPos()
	vel := floatutils.ClipSlice(h.QVel(), -10.0, 10.0)

	return mat.NewVecDense(h.obsLen, append(pos[1:], vel...))
}

func (h *Hopper) Reset() ts.TimeStep {
	h.MujocoEnv.Reset()
	startVec := h.Start()
	posStart := startVec.RawVector().Data[:h.Nq]
	velStart := startVec.RawVector().Data[h.Nq:]

	h.SetState(posStart, velStart)

	firstStep := ts.New(ts.First, 0, h.Discount, h.getObs(), 0)
	h.currentTimeStep = firstStep
	return firstStep
}

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
