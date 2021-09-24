// Package gae implements functionality for storing a generalized
// advantage estimate buffer
package gae

import (
	"fmt"

	"github.com/samuelfneumann/golearn/utils/matutils"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
)

// Interesting: This is a GAE(λ) buffer. What about n-Step GAE?

// Buffer implements a forward view generalized advantage estimate -
// GAE(λ) - buffer following https://arxiv.org/abs/1506.02438. This
// implementation is adapted from:
//
// https://github.com/openai/spinningup/tree/master/spinup/algos/tf1/vpg
type Buffer struct {
	obsSize    int // Size of state observations
	actionSize int // Number of action dimensions
	maxSize    int // Max buffer size

	currentPos   int // Current position in the buffer
	pathStartIdx int // Position in the buffere where current trajectory starts

	lambda float64 // λ for GAE(λ) calculation
	gamma  float64 // Discount factor ℽ; overwrites env discount factor

	// Buffers for storing data
	obsBuffer []float64
	actBuffer []float64
	advBuffer []float64
	rewBuffer []float64
	retBuffer []float64
	valBuffer []float64
}

// New creates and returns a new GAE(λ) buffer
func New(obsDim, actDim, size int, lambda, gamma float64) *Buffer {
	obsBuffer := make([]float64, size*obsDim)
	actBuffer := make([]float64, size*actDim)
	advBuffer := make([]float64, size)
	rewBuffer := make([]float64, size)
	retBuffer := make([]float64, size)
	valBuffer := make([]float64, size)

	return &Buffer{
		obsSize:      obsDim,
		actionSize:   actDim,
		maxSize:      size,
		currentPos:   0,
		pathStartIdx: 0,
		lambda:       lambda,
		gamma:        gamma,
		obsBuffer:    obsBuffer,
		actBuffer:    actBuffer,
		advBuffer:    advBuffer,
		rewBuffer:    rewBuffer,
		retBuffer:    retBuffer,
		valBuffer:    valBuffer,
	}
}

// Store stores a single timestep state, action, reward, and value to
// the Buffer.
func (v *Buffer) Store(obs, act []float64, rew, val float64) error {
	if v.currentPos >= v.maxSize {
		return fmt.Errorf("store: cannot add new transition, buffer at" +
			"maximum capacity")
	}
	if len(obs) != v.obsSize {
		return fmt.Errorf("store: illegal obs length \n\twant(%v)\n\thave(%v)",
			v.obsSize, len(obs))
	}
	if len(act) != v.actionSize {
		return fmt.Errorf("store: illegal act length \n\twant(%v)\n\thave(%v)",
			v.actionSize, len(act))
	}

	// Add observations
	start := v.currentPos * v.obsSize
	stop := start + v.obsSize
	copy(v.obsBuffer[start:stop], obs)

	// Add actions
	start = v.currentPos * v.actionSize
	stop = start + v.actionSize
	copy(v.actBuffer[start:stop], act)

	v.rewBuffer[v.currentPos] = rew
	v.valBuffer[v.currentPos] = val
	v.currentPos++
	return nil
}

// FinishPath computes advatange estimates using GAE(λ) and
// rewards-to-go estiamtes for each state for the current trajectory.
// This should be called at the end of a trajectory or when one gets
// cut off by an epoch ending.
//
// The lastVal argument should be 0 if the trajectory ended because
// the agent reached a terminal state, and otherwise it should be
// v(s), the value estimate of the current state. This allows for
// bootstrapping the rewards-to-go calculation to account for timesteps
// beyond the arbitrary episode horizon or epoch cutoff. The lastVal
// parameter is also used to compute the generalized advantage estimate
// for all states. See SpinningUp's Tensorflow implementation of
// vpgBuffer for more details (https://github.com/openai/spinningup).
func (v *Buffer) FinishPath(lastVal float64) {
	start := v.pathStartIdx
	stop := v.currentPos
	rews := append(v.rewBuffer[start:stop], lastVal)
	vals := append(v.valBuffer[start:stop], lastVal)

	// GAE-lambda advantage calculation
	stateVals := mat.NewVecDense(len(vals)-1, vals[:len(vals)-1])
	nextStateVals := mat.NewVecDense(len(vals)-1, vals[1:])
	rewards := mat.NewVecDense(len(rews)-1, rews[:len(rews)-1])

	deltas := mat.NewVecDense(stateVals.Len(), nil)
	deltas.AddScaledVec(rewards, v.gamma, nextStateVals)
	deltas.SubVec(deltas, stateVals)

	copy(v.advBuffer[start:stop],
		discountCumSum(deltas, v.gamma*v.lambda))

	// Rewards-to-go
	rewards = mat.NewVecDense(len(rews), rews)
	rewsToGo := discountCumSum(rewards, v.gamma)

	copy(v.retBuffer[start:stop], rewsToGo[:len(rewsToGo)-1])

	v.pathStartIdx = v.currentPos
}

// Get returns the observations, action, advantages, and returns stored
// in the buffer. Advantages are first standardized to mean 0 and
// standard deviation 1.
func (v *Buffer) Get() ([]float64, []float64, []float64, []float64, error) {
	if v.currentPos != v.maxSize {
		err := fmt.Errorf("get: buffer must be full before sampling")
		return nil, nil, nil, nil, err
	}

	v.currentPos = 0
	v.pathStartIdx = 0

	// Advantage normalization
	adv := mat.NewVecDense(len(v.advBuffer), v.advBuffer)
	ones := matutils.VecOnes(adv.Len())
	mean := stat.Mean(v.advBuffer, nil)
	std := stat.StdDev(v.advBuffer, nil) + 1e-8
	stdVec := mat.NewVecDense(adv.Len(), nil)
	stdVec.AddScaledVec(stdVec, std, ones)

	adv.AddScaledVec(adv, -mean, ones)
	adv.DivElemVec(adv, stdVec)

	return v.obsBuffer, v.actBuffer, adv.RawVector().Data, v.retBuffer, nil
}

// discountCumSum computes and returns the discounted cumulative sum
// of all elements of a vector. Given a vector v = [x0 x1 x2 ... xN]
// and discount ℽ, this function computes and returns:
//
// [
//	x0 + ℽ x1 + ℽ^2 x2 + ℽ^3 x3 + ... + ℽ^(N-1) x(N-1) + ℽ^N xN
//	x1 + ℽ^1 x2 + ℽ^2 x3 + ... + ℽ^(N-2) x(N-1) + ℽ^(N-1) xN
//	x2 + ℽ^1 x3 + ... + ℽ^(N-3) x(N-1) + ℽ^(N-2) xN
// ...
// xN
// ]
func discountCumSum(x *mat.VecDense, discount float64) []float64 {
	discounts := mat.NewVecDense(x.Len(), nil)
	cumSums := make([]float64, x.Len())
	nextScaledRews := mat.NewVecDense(x.Len(), nil)
	backing := nextScaledRews.RawVector().Data

	for i := 0; i < x.Len(); i++ {
		discounts.ScaleVec(discount, discounts)
		discounts.SetVec(x.Len()-i-1, 1)

		nextScaledRews.MulElemVec(discounts, x)
		cumSums[x.Len()-i-1] = floats.Sum(backing[x.Len()-i-1:])
	}

	return cumSums
}
