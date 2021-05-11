package policy

// NewGreedy creates a new Greedy policy
func NewGreedy(seed uint64, features, actions int) *EGreedy {
	return NewEGreedy(0.0, seed, features, actions)
}
