package expreplay

import (
	"fmt"

	"sfneuman.com/golearn/timestep"
)

// fifoRemove1Cache implements a concrete ExperienceReplayer where
// elements are removed from the buffer in a FiFo manner, and only a
// single element is removed from the cache at a time. This is the most
// common use of experience replay.
//
// The fifoRemove1Cache is implemented to increase the efficiency of the
// cache struct when a FiFo Remover is used that removes only a single
// element from the cache at a time. In such cases, we can reduce the
// used RAM and increase the computational speed since we can take
// advantage of knowing the concrete type of the Remover.
type fifoRemove1Cache struct {
	stateCache      []float64
	actionCache     []float64
	rewardCache     []float64
	discountCache   []float64
	nextStateCache  []float64
	nextActionCache []float64

	indices         []int
	currentInUsePos int
	isFull          bool

	// Outlines how data is removed and sampled
	sampler Selector

	minCapacity int
	maxCapacity int
	featureSize int
	actionSize  int
}

// newFifoRemove1Cache returns a new fifoRemove1Cache
func newFifoRemove1Cache(sampler Selector, minCapacity, maxCapacity, featureSize,
	actionSize int) *fifoRemove1Cache {
	stateCache := make([]float64, maxCapacity*featureSize)
	nextStateCache := make([]float64, maxCapacity*featureSize)

	actionCache := make([]float64, maxCapacity*actionSize)
	nextActionCache := make([]float64, maxCapacity*actionSize)

	rewardCache := make([]float64, maxCapacity)
	discountCache := make([]float64, maxCapacity)

	indices := make([]int, maxCapacity)
	for i := 0; i < maxCapacity; i++ {
		indices[i] = i
	}

	return &fifoRemove1Cache{
		stateCache:      stateCache,
		actionCache:     actionCache,
		rewardCache:     rewardCache,
		discountCache:   discountCache,
		nextStateCache:  nextStateCache,
		nextActionCache: nextActionCache,

		indices:         indices,
		currentInUsePos: 0,
		isFull:          false,

		sampler: sampler,

		minCapacity: minCapacity,
		maxCapacity: maxCapacity,
		featureSize: featureSize,
		actionSize:  actionSize,
	}
}

// String returns the string representation of the fifoRemove1Cache
func (c *fifoRemove1Cache) String() string {
	var emptyIndices []int
	var usedIndices []int
	if !c.isFull {
		emptyIndices = c.indices[c.currentInUsePos:]
		usedIndices = c.indices[:c.currentInUsePos]
	} else {
		emptyIndices = []int{}
		usedIndices = c.indices
	}

	baseStr := "Indices Available: %v \nIndices Used: %v \nStates: %v" +
		" \nActions: %v \nRewards: %v \nDiscounts: %v \nNext States: %v \n" +
		"Next Actions: %v"
	return fmt.Sprintf(baseStr, emptyIndices, usedIndices, c.stateCache,
		c.actionCache, c.rewardCache, c.discountCache, c.nextStateCache, c.nextActionCache)
}

// BatchSize returns the number of samples sampled using Sample() -
// a.k.a the batch size
func (c *fifoRemove1Cache) BatchSize() int {
	return c.sampler.BatchSize()
}

// insertOrder returns the insertion order of samples into the buffer
func (c *fifoRemove1Cache) insertOrder(n int) []int {
	if !c.isFull {
		return c.indices[:c.currentInUsePos]
	}

	currentIndices := make([]int, c.MaxCapacity())
	copy(currentIndices[c.currentInUsePos:], c.indices[c.currentInUsePos:])
	copy(currentIndices[:c.currentInUsePos], c.indices[:c.currentInUsePos])

	return currentIndices[:n]
}

// sampleFrom returns the slice of indices to sample from
func (c *fifoRemove1Cache) sampleFrom() []int {
	if !c.isFull {
		return c.indices[:c.currentInUsePos]
	}
	return c.indices
}

// Sample samples and returns a batch of transitions from the replay
// buffer
func (c *fifoRemove1Cache) Sample() ([]float64, []float64, []float64, []float64,
	[]float64, []float64, error) {
	if c.Capacity() == 0 {
		err := &ExpReplayError{
			Op:  "sample",
			Err: errEmptyCache,
		}
		return nil, nil, nil, nil, nil, nil, err
	}
	if c.Capacity() < c.MinCapacity() {
		err := &ExpReplayError{
			Op:  "sample",
			Err: errInsufficientSamples,
		}
		return nil, nil, nil, nil, nil, nil, err
	}

	indices := c.sampler.choose(c)

	stateBatch := make([]float64, c.BatchSize()*c.featureSize)
	nextStateBatch := make([]float64, c.BatchSize()*c.featureSize)
	for i, index := range indices {
		batchStartInd := i * c.featureSize
		expStartInd := index * c.featureSize
		copy(stateBatch[batchStartInd:batchStartInd+c.featureSize],
			c.stateCache[expStartInd:expStartInd+c.featureSize],
		)
		copy(nextStateBatch[batchStartInd:batchStartInd+c.featureSize],
			c.nextStateCache[expStartInd:expStartInd+c.featureSize],
		)
	}

	actionBatch := make([]float64, c.BatchSize()*c.actionSize)
	nextActionBatch := make([]float64, c.BatchSize()*c.actionSize)
	for i, index := range indices {
		batchStartInd := i * c.actionSize
		expStartInd := index * c.actionSize
		copy(actionBatch[batchStartInd:batchStartInd+c.actionSize],
			c.actionCache[expStartInd:expStartInd+c.actionSize],
		)
		copy(nextActionBatch[batchStartInd:batchStartInd+c.actionSize],
			c.nextActionCache[expStartInd:expStartInd+c.actionSize],
		)
	}

	rewardBatch := make([]float64, c.BatchSize())
	discountBatch := make([]float64, c.BatchSize())
	for i, index := range indices {
		discountBatch[i] = c.discountCache[index]
		rewardBatch[i] = c.rewardCache[index]
	}

	return stateBatch, actionBatch, rewardBatch, discountBatch, nextStateBatch,
		nextActionBatch, nil
}

// Capacity returns the current number of elements in the fifoRemove1Cache that
// are available for sampling
func (c *fifoRemove1Cache) Capacity() int {
	if c.isFull {
		return c.MaxCapacity()
	}
	return c.currentInUsePos
}

// MaxCapacity returns the maximum number of elements that are allowed
// in the fifoRemove1Cache
func (c *fifoRemove1Cache) MaxCapacity() int {
	return c.maxCapacity
}

// MinCapacity returns the minimum number of elements required in the
// fifoRemove1Cache before sampling is allowed
func (c *fifoRemove1Cache) MinCapacity() int {
	return c.minCapacity
}

// Add adds a transition to the fifoRemove1Cache
func (c *fifoRemove1Cache) Add(t timestep.Transition) error {
	index := c.currentInUsePos
	if !c.isFull && index+1 == c.MaxCapacity() {
		c.isFull = true
	}

	if t.State.Len() != c.featureSize || t.NextState.Len() != c.featureSize {
		return fmt.Errorf("add: invalid feature size \n\twant(%v)\n\thave(%v)",
			t.State.Len(), c.featureSize)
	}
	if t.Action.Len() != c.actionSize || t.NextAction.Len() != c.actionSize {
		return fmt.Errorf("add: invalid action size \n\twant(%v)\n\thave(%v)",
			t.Action.Len(), c.actionSize)
	}

	// Copy states
	stateInd := index * c.featureSize
	copy(c.stateCache[stateInd:stateInd+c.featureSize], t.State.RawVector().Data)
	copy(c.nextStateCache[stateInd:stateInd+c.featureSize],
		t.NextState.RawVector().Data)

	// Copy actions
	actionInd := index * c.actionSize
	copy(c.actionCache[actionInd:actionInd+c.actionSize],
		t.Action.RawVector().Data)
	copy(c.nextActionCache[actionInd:actionInd+c.actionSize],
		t.NextAction.RawVector().Data)

	// Copy reward R
	c.rewardCache[index] = t.Reward
	c.discountCache[index] = t.Discount

	c.currentInUsePos = (c.currentInUsePos + 1) % c.MaxCapacity()
	return nil
}
