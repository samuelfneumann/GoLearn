package expreplay

import (
	"math/rand"

	"sfneuman.com/golearn/utils/intutils"
)

// Selector implements functionality for choosing how data should be
// sampled and/or removed from an experience replay buffer
type Selector interface {
	// choose selects the indices at which data should be sampled from
	// the experience replay buffer
	choose(c *cache) []int

	// BatchSize returns the number of elements that will be selected
	BatchSize() int

	// registerAsRemover registers a Selector as a remover
	//
	//Some Selectors require different behaviour if they are removers,
	// so they should be notified if they become a remover to add this
	// additional behaviour
	registerAsRemover()
}

// uniformSelector is a Selector which selects data from an experience
// replay buffer uniformly randomly
type uniformSelector struct {
	samples int
	rng     *rand.Rand
}

// NewUniformSelector returns a new Selector which selects data uniformly
// randomly from an experience replay buffer
func NewUniformSelector(samples int, seed int64) Selector {
	source := rand.NewSource(seed)
	rng := rand.New(source)

	return &uniformSelector{samples: samples, rng: rng}
}

// registerAsRemover implements Selector interface
func (u *uniformSelector) registerAsRemover() {}

// size gets the number of samples in a batch drawn from the buffer
func (u *uniformSelector) BatchSize() int {
	return u.samples
}

// choose selects a number of indices at which to draw data from the
// buffer
func (u *uniformSelector) choose(c *cache) []int {
	selected := make([]int, u.BatchSize())
	keys := keysWithValue(c.emptyIndices, false)

	for i := 0; i < u.BatchSize(); i++ {
		index := u.rng.Int() % c.Capacity()
		selected[i] = keys[index]
	}

	return selected
}

// fifoSelector is a Selector which selects data from an experience
// replay buffer as first-in-first-out.
type fifoSelector struct {
	samples int
	remover bool
}

// NewFifoSelector returns a new Selector which draws data from an
// experience replay buffer in as FiFo.
func NewFifoSelector(samples int) Selector {
	return &fifoSelector{samples: samples, remover: false}
}

// registerAsRemover implements Selector interface
func (f *fifoSelector) registerAsRemover() {
	f.remover = true
}

// size gets the number of samples in a batch drawn from the buffer
func (f *fifoSelector) BatchSize() int {
	return f.samples
}

// choose selects a number of indices at which to draw data from the
// buffer
func (f *fifoSelector) choose(c *cache) []int {
	selected := make([]int, intutils.Min(f.BatchSize(), c.Capacity()))
	insertOrder := c.insertOrder(f.BatchSize())

	for i := 0; i < f.BatchSize() && i < c.Capacity(); i++ {
		selected[i] = insertOrder[i]

		if f.remover {
			// In a Fifo remover, the indices at which data was first
			// added get freed first, so we can remove these from the
			// ordering of inserted indices
			c.removeFront()
		}
	}

	return selected
}
