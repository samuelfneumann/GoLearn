package policy

import (
	"github.com/samuelfneumann/golearn/agent"
	"github.com/samuelfneumann/golearn/environment"
)

// NewGreedy creates a new Greedy policy
func NewGreedy(seed uint64, env environment.Environment) (agent.Policy, error) {
	return NewEGreedy(0.0, seed, env)
}
