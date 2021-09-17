package expreplay

import (
	"github.com/samuelfneumann/golearn/timestep"
)

// onlineCache implements an experience replay buffer for sampling
// completely online.
//
// When creating a new experience replay buffer, the user could
// choose to use a buffer with a maximum capacity of 1. In this case,
// experience replay reduces to online sampling. This struct increases
// computational efficiency when a new cache is created with
// online sampling.
type onlineCache struct {
	// Whether the next action in the SARSA tuple should be store and
	// returned.
	includeNextAction bool

	stateCache      []float64
	actionCache     []float64
	rewardCache     []float64
	discountCache   []float64
	nextStateCache  []float64
	nextActionCache []float64
}

// newOnline returns a new online replay buffer
func newOnline(includeNextAction bool) ExperienceReplayer {
	return &onlineCache{
		includeNextAction: includeNextAction,
	}
}

func (o *onlineCache) Add(t timestep.Transition) error {
	o.stateCache = t.State.RawVector().Data
	o.actionCache = t.Action.RawVector().Data
	o.rewardCache = []float64{t.Reward}
	o.discountCache = []float64{t.Discount}
	o.nextStateCache = t.NextState.RawVector().Data

	if o.includeNextAction {
		o.nextActionCache = t.NextAction.RawVector().Data
	}

	return nil
}

// Sample samples and returns a batch of transitions from the replay
// buffer
func (o *onlineCache) Sample() ([]float64, []float64, []float64, []float64,
	[]float64, []float64, error) {
	if len(o.stateCache) == 0 {
		err := &ExpReplayError{
			Op:  "sample",
			Err: errEmptyCache,
		}
		return nil, nil, nil, nil, nil, nil, err
	}

	var nextActionCache []float64
	if o.includeNextAction {
		nextActionCache = o.nextActionCache
	}

	return o.stateCache, o.actionCache, o.rewardCache, o.discountCache,
		o.nextStateCache, nextActionCache, nil
}

// Capacity returns the current number of elements in the cache that
// are available for sampling
func (o *onlineCache) Capacity() int {
	return 1
}

// MaxCapacity returns the maximum number of elements that are allowed
// in the cache
func (o *onlineCache) MaxCapacity() int {
	return 1
}

// MinCapacity returns the minimum number of elements required in the
// cache before sampling is allowed
func (o *onlineCache) MinCapacity() int {
	return 1
}

// BatchSize returns the number of samples sampled using Sample() -
// a.k.a the batch size
func (o *onlineCache) BatchSize() int {
	return 1
}
