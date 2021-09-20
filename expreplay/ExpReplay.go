package expreplay

import (
	"container/list"
	"fmt"
	"os"
	"sync"

	"github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/golearn/utils/intutils"
)

// orderedSampler implements an experience replay buffer that can return
// its underlying indices to sample from and insertion order of these
// indices
type orderedSampler interface {
	ExperienceReplayer
	sampleFrom() []int

	// insertOrder returns the first n indices that were added to the
	// buffer
	insertOrder(n int) []int
}

// Config implements a specific configuration of an ExperienceReplayer
type Config struct {
	RemoveMethod      SelectorType
	SampleMethod      SelectorType
	RemoveSize        int
	SampleSize        int
	MaxReplayCapacity int
	MinReplayCapacity int
}

// BatchSize returns the size of batches sampled from the experience
// replayer defined by the config
func (c Config) BatchSize() int {
	return c.SampleSize
}

// Create creates and returns the ExperienceReplayer with the specified
// Config. The includeNextAction parameter determiines whether or not
// the next action in the SARSA tuple should also be stored.
func (c Config) Create(featureSize, actionSize int,
	seed int64, includeNextAction bool) (ExperienceReplayer, error) {

	return Factory(c.RemoveMethod, c.SampleMethod, c.MinReplayCapacity,
		c.MaxReplayCapacity, featureSize, actionSize, c.RemoveSize,
		c.SampleSize, seed, includeNextAction)
}

// ExperienceReplayer implements an experience replay buffer
type ExperienceReplayer interface {
	// Add adds a transition to the buffer
	Add(t timestep.Transition) error

	// Sample samples a batch of experience from the buffer and returns
	// the batch of (state, action, reward, discount, next state,
	// next action) tuples as []float64
	Sample() ([]float64, []float64, []float64, []float64, []float64,
		[]float64, error)

	// Capacity returns the current number of samples in the buffer
	Capacity() int

	// MaxCapacity returns the maximum allowable samples in the buffer
	MaxCapacity() int

	// MinCapacity returns the number of samples required to be in
	// the buffer before the buffer can be sampled
	MinCapacity() int

	// BatchSize returns the number of samples returned by Sample()
	BatchSize() int
}

// cache implements a concrete ExperienceReplayer
type cache struct {
	// includeNextAction denotes whether the next action in the
	// SARSA tuple should be stored and returned
	includeNextAction bool

	wait            sync.WaitGroup // Guards the following caches
	stateCache      []float64
	actionCache     []float64
	rewardCache     []float64
	discountCache   []float64
	nextStateCache  []float64
	nextActionCache []float64

	// The indices of the cache that are empty and have no data
	emptyIndices []int

	// The indices of the cache that have data
	inUseIndices []int

	// orderOfInsert outlines the order the chronological order of
	// inserts. For i > j, the data at index orderOfInsert[i] was
	// inserted into the buffer after the data at index orderOfInsert[j]
	orderOfInsert *list.List

	// Outlines how data is removed and sampled
	remover Selector
	sampler Selector

	minCapacity int
	maxCapacity int
	featureSize int
	actionSize  int
}

// Factory is a factory method for creating an ExperienceReplayer
func Factory(removeMethod, sampleMethod SelectorType, minCapacity, maxCapacity,
	featureSize, actionSize, removeSize, sampleSize int,
	seed int64, includeNextAction bool) (ExperienceReplayer, error) {
	remover := CreateSelector(removeMethod, removeSize, seed)
	sampler := CreateSelector(sampleMethod, sampleSize, seed)

	return New(remover, sampler, minCapacity, maxCapacity, featureSize,
		actionSize, includeNextAction)
}

// New creates and returns a new ExperienceReplayer. The remover and
// sampler parameters are Selectors which determine how data is removed
// and sampled from the replay buffer. The featureSize and actionSize
// parameters define the size of the feature and action vectors.
// The includeNextAction parameter determiines whether or not
// the next action in the SARSA tuple should also be stored.
// The minCapacity parameter determines the minimum number of samples
// that should be in the buffer before sampling is allowed.
// The maxCapacity parameter determines the maximum number of samples
// allowed in the buffer at any given time.
//
// Pixel observations should be flattened before adding to the buffer.
func New(remover, sampler Selector, minCapacity, maxCapacity, featureSize,
	actionSize int, includeNextAction bool) (ExperienceReplayer, error) {
	if minCapacity <= 0 {
		return &cache{}, fmt.Errorf("new: minCapacity must be > 0")
	}
	if maxCapacity < 1 {
		return &cache{}, fmt.Errorf("new: maxCapacity must be >= 1")
	}
	if maxCapacity < sampler.BatchSize() {
		return &cache{}, fmt.Errorf("new: cannot have batch size(%v) > max "+
			"buffer capacity (%v)", sampler.BatchSize(), maxCapacity)
	}

	// If minCapacity == maxCapacity == 1, then the replay buffer
	// only stores the most recent online transition. In this case,
	// onlineCache makes a number of efficiency improvements
	if minCapacity == 1 && maxCapacity == 1 {
		if sampler.BatchSize() > 1 || remover.BatchSize() > 1 {
			msg := "new: using online sampler, ignoring batch size > 1"
			fmt.Fprintln(os.Stderr, msg)
		}
		return newOnline(includeNextAction), nil
	}

	if _, ok := remover.(*fifoSelector); ok && remover.BatchSize() == 1 {
		return newFifoRemove1Cache(sampler, minCapacity, maxCapacity, featureSize,
			actionSize, includeNextAction), nil
	}

	stateCache := make([]float64, maxCapacity*featureSize)
	nextStateCache := make([]float64, maxCapacity*featureSize)

	actionCache := make([]float64, maxCapacity*actionSize)
	var nextActionCache []float64
	if includeNextAction {
		nextActionCache = make([]float64, maxCapacity*actionSize)
	}

	rewardCache := make([]float64, maxCapacity)
	discountCache := make([]float64, maxCapacity)

	orderOfInsert := list.New()

	remover.registerAsRemover()

	emptyIndices := make([]int, maxCapacity)
	inUseIndices := make([]int, 0, maxCapacity)
	for i := 0; i < maxCapacity; i++ {
		emptyIndices[i] = i
	}

	return &cache{
		includeNextAction: includeNextAction,

		stateCache:      stateCache,
		actionCache:     actionCache,
		rewardCache:     rewardCache,
		discountCache:   discountCache,
		nextStateCache:  nextStateCache,
		nextActionCache: nextActionCache,

		emptyIndices:  emptyIndices,
		inUseIndices:  inUseIndices,
		orderOfInsert: orderOfInsert,

		remover: remover,
		sampler: sampler,

		minCapacity: minCapacity,
		maxCapacity: maxCapacity,
		featureSize: featureSize,
		actionSize:  actionSize,
	}, nil
}

// sampleFrom returns the indices to sample from
func (c *cache) sampleFrom() []int {
	c.wait.Wait()

	return c.inUseIndices
}

// insertOrder returns a slice of at most n indices which describes
// the order that the first n data were inserted into the buffer.
// The length of the returned slice is the minimum between n and the
// number of elements currently in the buffer
//
// For example, if this function returns []int{9, 15, 1}, this means
// that the first data was inserted into the buffer at position 9, the
// next at position 15, and the last at position 1
func (c *cache) insertOrder(n int) []int {
	c.wait.Wait()

	size := intutils.Min(n, c.Capacity())
	insertOrder := make([]int, size)
	element := c.orderOfInsert.Front()

	for i := 0; i < size; i++ {
		insertOrder[i] = element.Value.(int)
		element = element.Next()
		if element == nil {
			break
		}
	}
	return insertOrder
}

// String returns the string representation of the cache
func (c *cache) String() string {
	emptyIndices := c.emptyIndices
	usedIndices := c.inUseIndices

	baseStr := "Indices Available: %v \nIndices Used: %v \nStates: %v" +
		" \nActions: %v \nRewards: %v \nDiscounts: %v \nNext States: %v \n" +
		"Next Actions: %v"
	return fmt.Sprintf(baseStr, emptyIndices, usedIndices, c.stateCache,
		c.actionCache, c.rewardCache, c.discountCache, c.nextStateCache, c.nextActionCache)
}

// BatchSize returns the number of samples sampled using Sample() -
// a.k.a the batch size
func (c *cache) BatchSize() int {
	return c.sampler.BatchSize()
}

// remove removes elements from the cache using indices sampled from the
// cache's remover
func (c *cache) remove() error {
	c.wait.Wait()

	if c.Capacity() <= c.minCapacity {
		return fmt.Errorf("remove: cannot remove, cache at min capacity")
	}

	indices := c.remover.choose(c)
	for _, index := range indices {

		for i := range c.inUseIndices {
			if c.inUseIndices[i] == index {
				c.inUseIndices[i] = c.inUseIndices[len(c.inUseIndices)-1]
				c.inUseIndices = c.inUseIndices[:len(c.inUseIndices)-1]
				break
			}
		}

		c.emptyIndices = append(c.emptyIndices, indices...)
	}
	return nil
}

// removeFront removes the earliest tracked index that is at
// which data was inserted at.
//
// The cache keeps track of the order of indices at which data was
// inserted. This function will remove the earliest index from the front
// of this list.
func (c *cache) removeFront() {
	c.orderOfInsert.Remove(c.orderOfInsert.Front())
}

// Sample samples and returns a batch of transitions from the replay
// buffer. The returned values are the state, action, reward, discount,
// next state, and next action.
func (c *cache) Sample() ([]float64, []float64, []float64, []float64,
	[]float64, []float64, error) {
	c.wait.Wait()

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

	// Create the state batches
	stateBatch := make([]float64, c.BatchSize()*c.featureSize)
	nextStateBatch := make([]float64, c.BatchSize()*c.featureSize)

	// Fill the state batches
	c.wait.Add(2 * len(indices))
	for i, index := range indices {
		batchStartInd := i * c.featureSize
		expStartInd := index * c.featureSize
		go func() {
			copyInto(stateBatch, batchStartInd, batchStartInd+c.featureSize,
				c.stateCache[expStartInd:expStartInd+c.featureSize])
			c.wait.Done()
		}()

		go func() {
			copyInto(nextStateBatch, batchStartInd,
				batchStartInd+c.featureSize,
				c.nextStateCache[expStartInd:expStartInd+c.featureSize],
			)
			c.wait.Done()
		}()
	}

	// Create the action batches
	actionBatch := make([]float64, c.BatchSize()*c.actionSize)
	var nextActionBatch []float64
	if c.includeNextAction {
		nextActionBatch = make([]float64, c.BatchSize()*c.actionSize)
	}

	// Fill the action batches
	c.wait.Add(2 * len(indices))
	for i, index := range indices {
		batchStartInd := i * c.actionSize
		expStartInd := index * c.actionSize

		go func() {
			copyInto(actionBatch, batchStartInd, batchStartInd+c.actionSize,
				c.actionCache[expStartInd:expStartInd+c.actionSize],
			)
			c.wait.Done()
		}()

		go func() {
			if c.includeNextAction {
				copyInto(nextActionBatch, batchStartInd,
					batchStartInd+c.actionSize,
					c.nextActionCache[expStartInd:expStartInd+c.actionSize],
				)
			}
			c.wait.Done()
		}()
	}

	rewardBatch := make([]float64, c.BatchSize())
	discountBatch := make([]float64, c.BatchSize())
	for i, index := range indices {
		discountBatch[i] = c.discountCache[index]
		rewardBatch[i] = c.rewardCache[index]
	}

	c.wait.Wait()
	return stateBatch, actionBatch, rewardBatch, discountBatch, nextStateBatch,
		nextActionBatch, nil
}

// Capacity returns the current number of elements in the cache that
// are available for sampling
func (c *cache) Capacity() int {
	c.wait.Wait()

	return len(c.inUseIndices)
}

// MaxCapacity returns the maximum number of elements that are allowed
// in the cache
func (c *cache) MaxCapacity() int {
	return c.maxCapacity
}

// MinCapacity returns the minimum number of elements required in the
// cache before sampling is allowed
func (c *cache) MinCapacity() int {
	return c.minCapacity
}

// Add adds a transition to the cache
func (c *cache) Add(t timestep.Transition) error {
	// Wait for previous Add operation, then add again
	c.wait.Wait()
	c.wait.Add(4)

	if c.Capacity() >= c.maxCapacity {
		err := c.remove()
		if err != nil {
			return fmt.Errorf("add: cannot add to buffer: %v", err)
		}
	}

	emptyIndicesLength := len(c.emptyIndices)
	index := c.emptyIndices[emptyIndicesLength-1]
	c.emptyIndices = c.emptyIndices[:emptyIndicesLength-1]
	c.orderOfInsert.PushBack(index)
	c.inUseIndices = append(c.inUseIndices, index)

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
	go func() {
		copyInto(c.stateCache, stateInd, stateInd+c.featureSize,
			t.State.RawVector().Data)
		c.wait.Done()
	}()
	go func() {
		copyInto(c.nextStateCache, stateInd, stateInd+c.featureSize,
			t.NextState.RawVector().Data)
		c.wait.Done()
	}()

	// Copy actions
	actionInd := index * c.actionSize
	go func() {
		copyInto(c.actionCache, actionInd, actionInd+c.actionSize,
			t.Action.RawVector().Data)
		c.wait.Done()
	}()
	go func() {
		if c.includeNextAction {
			copyInto(c.nextActionCache, actionInd, actionInd+c.actionSize,
				t.NextAction.RawVector().Data)
		}
		c.wait.Done()
	}()

	// Copy reward R
	c.rewardCache[index] = t.Reward
	c.discountCache[index] = t.Discount

	c.wait.Wait()
	return nil
}

// copyInto copies src into dest[start:end]
func copyInto(dest []float64, start, end int, src []float64) int {
	return copy(dest[start:end], src)
}
