// Package vanillapg implements the Vanilla Policy Gradient or REINFORCE
// algorithm
package vanillapg

import "fmt"

// This could be implemented as an expreplay with a null remover
// and a fifo sampler. Min capacity = max capcacity = episode length
// it does seem a bit forced though.
type vpgBuffer struct {
	obsSize    int
	actionSize int
	maxSize    int // Max buffer size
	currentPos int

	obsBuffer   []float64
	actBuffer   []float64
	qBuffer     []float64
	rewBuffer   []float64
	retBuffer   []float64
	valBuffer   []float64
	gammaBuffer []float64

	// For estimating KL divergence between old and new policies
	// log_p_buffer []float64
}

func newVPGBuffer(obsDim, actDim, size int) *vpgBuffer {
	obsBuffer := make([]float64, size*obsDim)
	actBuffer := make([]float64, size*actDim)
	qBuffer := make([]float64, size)
	rewBuffer := make([]float64, size)
	retBuffer := make([]float64, size)
	valBuffer := make([]float64, size)
	gammaBuffer := make([]float64, size)

	return &vpgBuffer{
		obsSize:     obsDim,
		actionSize:  actDim,
		maxSize:     size,
		currentPos:  0,
		obsBuffer:   obsBuffer,
		actBuffer:   actBuffer,
		qBuffer:     qBuffer,
		rewBuffer:   rewBuffer,
		retBuffer:   retBuffer,
		valBuffer:   valBuffer,
		gammaBuffer: gammaBuffer,
	}
}

// store stores a single timestep state, action, reward, and value to
// the vpgBuffer.
func (v *vpgBuffer) store(obs, act []float64, rew, discount float64) error {
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
	v.gammaBuffer[v.currentPos] = discount
	v.currentPos++
	return nil
}

// returns state, next state, action, next action, reward, discount
func (v *vpgBuffer) get() ([]float64, []float64, []float64, []float64, error) {
	if v.currentPos != v.maxSize {
		err := fmt.Errorf("get: buffer must be full before sampling")
		return nil, nil, nil, nil, err
	}

	v.currentPos = 0
	return v.obsBuffer, v.actBuffer, v.rewBuffer, v.gammaBuffer, nil
}
