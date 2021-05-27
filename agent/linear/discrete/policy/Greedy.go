package policy

import "sfneuman.com/golearn/environment"

// NewGreedy creates a new Greedy policy
func NewGreedy(seed uint64, env environment.Environment) *EGreedy {
	return NewEGreedy(0.0, seed, env)
}
