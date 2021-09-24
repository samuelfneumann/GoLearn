package expreplay

import (
	"fmt"
	"sync"

	"github.com/samuelfneumann/golearn/timestep"
)

// defaultCache implements a concrete ExperienceReplayer where
// elements are removed from the buffer in a FiFo manner and only a
// single element is removed from the cache at a time. This is the most
// common use of experience replay.
//
// The defaultCache is implemented to increase the efficiency of the
// cache struct when a FiFo Remover is used that removes only a single
// element from the cache at a time. In such cases, we can reduce the
// used RAM and increase the computational speed since we can take
// advantage of knowing the concrete type of the Remover.
type defaultCache struct {
	// includeNextAction denotes whether the next action in the SARSA
	// tuple should be stored and returned
	includeNextAction bool

	wait            sync.WaitGroup // Guards the following caches
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

// newDefaultCache returns a new defaultCache. The sampler
// parameter is a Selectors which determines how data is sampled
// from the replay buffer. The featureSize and actionSize
// parameters define the size of the feature and action vectors.
// The minCapacity parameter determines the minimum number of samples
// that should be in the buffer before sampling is allowed.
// The maxCapacity parameter determines the maximum number of samples
// allowed in the buffer at any given time.
// The includeNextAction parameter determiines whether or not
// the next action in the SARSA tuple should also be stored.
func newDefaultCache(sampler Selector, minCapacity, maxCapacity,
	featureSize, actionSize int, includeNextAction bool) *defaultCache {
	stateCache := make([]float64, maxCapacity*featureSize)
	nextStateCache := make([]float64, maxCapacity*featureSize)

	actionCache := make([]float64, maxCapacity*actionSize)
	var nextActionCache []float64
	if includeNextAction {
		nextActionCache = make([]float64, maxCapacity*actionSize)
	}

	rewardCache := make([]float64, maxCapacity)
	discountCache := make([]float64, maxCapacity)

	indices := make([]int, maxCapacity)
	for i := 0; i < maxCapacity; i++ {
		indices[i] = i
	}

	return &defaultCache{
		includeNextAction: includeNextAction,

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

// String returns the string representation of the defaultCache
func (d *defaultCache) String() string {
	var emptyIndices []int
	var usedIndices []int
	if !d.isFull {
		emptyIndices = d.indices[d.currentInUsePos:]
		usedIndices = d.indices[:d.currentInUsePos]
	} else {
		emptyIndices = []int{}
		usedIndices = d.indices
	}

	baseStr := "Indices Available: %v \nIndices Used: %v \nStates: %v" +
		" \nActions: %v \nRewards: %v \nDiscounts: %v \nNext States: %v \n" +
		"Next Actions: %v"
	return fmt.Sprintf(baseStr, emptyIndices, usedIndices, d.stateCache,
		d.actionCache, d.rewardCache, d.discountCache, d.nextStateCache, d.nextActionCache)
}

// BatchSize returns the number of samples sampled using Sample() -
// a.k.a the batch size
func (d *defaultCache) BatchSize() int {
	return d.sampler.BatchSize()
}

// insertOrder returns the insertion order of samples into the buffer
func (d *defaultCache) insertOrder(n int) []int {
	d.wait.Wait()

	if !d.isFull {
		return d.indices[:d.currentInUsePos]
	}

	currentIndices := make([]int, d.MaxCapacity())
	copy(currentIndices[d.currentInUsePos:], d.indices[d.currentInUsePos:])
	copy(currentIndices[:d.currentInUsePos], d.indices[:d.currentInUsePos])

	return currentIndices[:n]
}

// sampleFrom returns the slice of indices to sample from
func (d *defaultCache) sampleFrom() []int {
	d.wait.Wait()

	if !d.isFull {
		return d.indices[:d.currentInUsePos]
	}
	return d.indices
}

// Sample samples and returns a batch of transitions from the replay
// buffer. The returned values are the state, action, reward, discount,
// next state, and next action.
func (d *defaultCache) Sample() ([]float64, []float64, []float64,
	[]float64, []float64, []float64, error) {
	d.wait.Wait()

	if d.Capacity() == 0 {
		err := &ExpReplayError{
			Op:  "sample",
			Err: errEmptyCache,
		}
		return nil, nil, nil, nil, nil, nil, err
	}
	if d.Capacity() < d.MinCapacity() {
		err := &ExpReplayError{
			Op:  "sample",
			Err: errInsufficientSamples,
		}
		return nil, nil, nil, nil, nil, nil, err
	}

	indices := d.sampler.choose(d)

	// Create the state batches
	stateBatch := make([]float64, d.BatchSize()*d.featureSize)
	nextStateBatch := make([]float64, d.BatchSize()*d.featureSize)

	// Fill the state batches
	d.wait.Add(2 * len(indices))
	for i, index := range indices {
		batchStartInd := i * d.featureSize
		expStartInd := index * d.featureSize

		go func() {
			copyInto(stateBatch, batchStartInd, batchStartInd+d.featureSize,
				d.stateCache[expStartInd:expStartInd+d.featureSize])
			d.wait.Done()
		}()

		go func() {
			copyInto(nextStateBatch, batchStartInd,
				batchStartInd+d.featureSize,
				d.nextStateCache[expStartInd:expStartInd+d.featureSize],
			)
			d.wait.Done()
		}()
	}

	// Create the action batches
	actionBatch := make([]float64, d.BatchSize()*d.actionSize)
	var nextActionBatch []float64
	if d.includeNextAction {
		nextActionBatch = make([]float64, d.BatchSize()*d.actionSize)
	}

	// Fill the action batches
	d.wait.Add(2 * len(indices))
	for i, index := range indices {
		batchStartInd := i * d.actionSize
		expStartInd := index * d.actionSize

		go func() {
			copyInto(actionBatch, batchStartInd, batchStartInd+d.actionSize,
				d.actionCache[expStartInd:expStartInd+d.actionSize],
			)
			d.wait.Done()
		}()

		go func() {
			if d.includeNextAction {
				copyInto(nextActionBatch, batchStartInd,
					batchStartInd+d.actionSize,
					d.nextActionCache[expStartInd:expStartInd+d.actionSize],
				)
			}
			d.wait.Done()
		}()
	}

	rewardBatch := make([]float64, d.BatchSize())
	discountBatch := make([]float64, d.BatchSize())
	for i, index := range indices {
		discountBatch[i] = d.discountCache[index]
		rewardBatch[i] = d.rewardCache[index]
	}

	d.wait.Wait()
	return stateBatch, actionBatch, rewardBatch, discountBatch, nextStateBatch,
		nextActionBatch, nil
}

// Capacity returns the current number of elements in the defaultCache that
// are available for sampling
func (d *defaultCache) Capacity() int {
	d.wait.Wait()

	if d.isFull {
		return d.MaxCapacity()
	}
	return d.currentInUsePos
}

// MaxCapacity returns the maximum number of elements that are allowed
// in the defaultCache
func (d *defaultCache) MaxCapacity() int {
	return d.maxCapacity
}

// MinCapacity returns the minimum number of elements required in the
// defaultCache before sampling is allowed
func (d *defaultCache) MinCapacity() int {
	return d.minCapacity
}

// Add adds a transition to the defaultCache
func (d *defaultCache) Add(t timestep.Transition) error {
	// Finish the last Add operation, then start
	d.wait.Wait()
	d.wait.Add(4)

	index := d.currentInUsePos
	if !d.isFull && index+1 == d.MaxCapacity() {
		d.isFull = true
	}

	if t.State.Len() != d.featureSize || t.NextState.Len() != d.featureSize {
		return fmt.Errorf("add: invalid feature size \n\twant(%v)\n\thave(%v)",
			t.State.Len(), d.featureSize)
	}
	if t.Action.Len() != d.actionSize || t.NextAction.Len() != d.actionSize {
		return fmt.Errorf("add: invalid action size \n\twant(%v)\n\thave(%v)",
			t.Action.Len(), d.actionSize)
	}

	// Copy states
	stateInd := index * d.featureSize
	go func() {
		copyInto(d.stateCache, stateInd, stateInd+d.featureSize,
			t.State.RawVector().Data)
		d.wait.Done()
	}()
	go func() {
		copyInto(d.nextStateCache, stateInd, stateInd+d.featureSize,
			t.NextState.RawVector().Data)
		d.wait.Done()
	}()

	// Copy actions
	actionInd := index * d.actionSize
	go func() {
		copyInto(d.actionCache, actionInd, actionInd+d.actionSize,
			t.Action.RawVector().Data)
		d.wait.Done()
	}()
	go func() {
		if d.includeNextAction {
			copyInto(d.nextActionCache, actionInd, actionInd+d.actionSize,
				t.NextAction.RawVector().Data)
		}
		d.wait.Done()
	}()

	// Copy reward R
	d.rewardCache[index] = t.Reward
	d.discountCache[index] = t.Discount

	d.currentInUsePos = (d.currentInUsePos + 1) % d.MaxCapacity()
	return nil
}
