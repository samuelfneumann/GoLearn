package agent

import (
	"sfneuman.com/golearn/environment"
)

// Config represents a configuration for creating an agent
type Config interface {
	// CreateAgent creates the agent that the config describes
	CreateAgent(env environment.Environment, seed uint64) (Agent, error)

	// ValidAgent returns whether the argument agent is valid for the
	// Config
	ValidAgent(Agent) bool

	// Validate returns an error describing whether or not the
	// configuration is valid or not.
	Validate() error
}

// PolicyType represents a type of distribution that a policy could be
type PolicyType string

const (
	Gaussian    PolicyType = "Gaussian"
	Categorical PolicyType = "Softmax"
	EGreedy     PolicyType = "EGreedy"
)
