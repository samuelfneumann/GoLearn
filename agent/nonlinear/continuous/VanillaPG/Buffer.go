// Package vanillapg implements the Vanilla Policy Gradient or REINFORCE
// algorithm with generalized advantage estimation (GAE)
//
// Adapted from https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/vpg/vpg.py
package vanillapg

import (
	"fmt"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat"
	"sfneuman.com/golearn/utils/matutils"
)

// This could be implemented as an expreplay with a null remover
// and a fifo sampler. Min capacity = max capcacity = episode length
// it does seem a bit forced though.
//
type vpgBuffer struct {
	obsSize      int
	actionSize   int
	maxSize      int // Max buffer size
	currentPos   int
	pathStartIdx int
	lambda       float64
	gamma        float64

	obsBuffer []float64
	actBuffer []float64
	advBuffer []float64
	rewBuffer []float64
	retBuffer []float64
	valBuffer []float64

	// For estimating KL divergence between old and new policies
	// log_p_buffer []float64
}

func newVPGBuffer(obsDim, actDim, size int, lambda, gamma float64) *vpgBuffer {
	obsBuffer := make([]float64, size*obsDim)
	actBuffer := make([]float64, size*actDim)
	advBuffer := make([]float64, size)
	rewBuffer := make([]float64, size)
	retBuffer := make([]float64, size)
	valBuffer := make([]float64, size)

	return &vpgBuffer{
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

// store stores a single timestep state, action, reward, and value to
// the vpgBuffer.
func (v *vpgBuffer) store(obs, act []float64, rew, val, dis float64) error {
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

	start := v.currentPos * v.obsSize
	stop := start + v.obsSize
	copy(v.obsBuffer[start:stop], obs)

	start = v.currentPos * v.actionSize
	stop = start + v.actionSize
	copy(v.actBuffer[start:stop], act)

	v.rewBuffer[v.currentPos] = rew
	v.valBuffer[v.currentPos] = val
	v.currentPos++
	return nil
}

func (v *vpgBuffer) finishPath(lastVal float64) {
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
	deltas.AddScaledVec(deltas, -1.0, stateVals)

	copy(v.advBuffer[start:stop],
		discountCumSum(deltas, v.gamma*v.lambda))

	// Rewardsw-to-go
	rewsToGo := discountCumSum(rewards, v.gamma)
	copy(v.retBuffer[start:stop], rewsToGo[:len(rewsToGo)-1])

	v.pathStartIdx = v.currentPos
}

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

// returns state, next state, action, next action, reward, discount
func (v *vpgBuffer) get() ([]float64, []float64, []float64, []float64, error) {
	if v.currentPos != v.maxSize {
		err := fmt.Errorf("get: buffer must be full before sampling")
		return nil, nil, nil, nil, err
	}

	v.currentPos = 0
	v.pathStartIdx = 0

	// Advantage normalization
	adv := mat.NewVecDense(len(v.advBuffer), v.advBuffer)
	ones := matutils.VecOnes(adv.Len())
	mean, std := stat.MeanStdDev(v.advBuffer, nil)
	stdVec := mat.NewVecDense(adv.Len(), nil)
	stdVec.AddScaledVec(stdVec, std, ones)

	adv.AddScaledVec(adv, -mean, ones)
	adv.DivElemVec(adv, stdVec)

	return v.obsBuffer, v.actBuffer, adv.RawVector().Data, v.retBuffer, nil
}
