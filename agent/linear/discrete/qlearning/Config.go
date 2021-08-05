package qlearning

import (
	"fmt"
	"reflect"

	"github.com/samuelfneumann/golearn/agent"
	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/utils/matutils/initializers/weights"
)

func init() {
	// Register ConfigList type so that it can be typed using
	// agent.TypedConfigList to help with serialization/deserialization.
	agent.Register(agent.EGreedyQLearningLinear, ConfigList{})
}

// ConfigList implements functionality for storing a number of Config's
// in a simple manner. Instead of storing a slice of Configs, the
// ConfigList stores each field's values and constructs the list by
// every combination of field values.
type ConfigList struct {
	Epsilon      []float64
	LearningRate []float64
}

// NewConfigList returns a new ConfigList as an agent.TypedConfigList
// so that it can easily be JSON serialized/deserialized without
// knowing the underlying concrete type.
func NewConfigList(ɛ, learningRate []float64) agent.TypedConfigList {
	config := ConfigList{Epsilon: ɛ, LearningRate: learningRate}
	return agent.NewTypedConfigList(config)
}

// Config returns an empty Config that is of the type stored by
// ConfigList
func (c ConfigList) Config() agent.Config {
	return Config{}
}

// Type returns the type of agent that can be constructed by Config's
// stored by the list
func (c ConfigList) Type() agent.Type {
	return c.Config().Type()
}

// NumFields returns the number of settable fields for the ConfigList
func (c ConfigList) NumFields() int {
	rValue := reflect.ValueOf(c)
	return rValue.NumField()
}

// Len returns the number of Configs stored by the list
func (c ConfigList) Len() int {
	return len(c.Epsilon) * len(c.LearningRate)
}

// Config represents a configuration for the QLearning agent
type Config struct {
	Epsilon      float64 // epislon for behaviour policy
	LearningRate float64
}

// CreateAgent creates the agent from the Config. Agent weights are
// always initialized to zero using this function. To initialize from
// some other distribution, use the agent's constructor manually.
func (c Config) CreateAgent(env environment.Environment,
	seed uint64) (agent.Agent, error) {

	// Create the zero weight initializer
	rand := weights.NewZeroUV() // Zero RNG
	init := weights.NewLinearUV(rand)

	return New(env, c, init, seed)
}

// ValidAgent returns whether the argument agent is a valid agent for
// construction with the Config
func (c Config) ValidAgent(a agent.Agent) bool {
	_, ok := a.(*QLearning)
	return ok
}

// Validate ensures that the Config is valid
func (c Config) Validate() error {
	if c.Epsilon < 0 {
		return fmt.Errorf("epislon cannot be lower than 0")
	}
	return nil
}

// Type returns the type of the agent constructed by the Config
func (c Config) Type() agent.Type {
	return agent.EGreedyQLearningLinear
}
