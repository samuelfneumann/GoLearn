package actorcritic

import (
	"reflect"

	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/utils/matutils/initializers/weights"
)

func init() {
	// Register ConfigList type so that it can be typed using
	// agent.TypedConfigList to help with serialization/deserialization.
	agent.Register(agent.GaussianActorCriticLinear, LinearGaussianConfigList{})
}

// LinearGaussianConfigList implements functionality for storing a
// number of Config's in a simple manner. Instead of storing a slice
// of Configs, the LinearGaussianConfigList stores each field's values
// and constructs the list by every combination of field values.
type LinearGaussianConfigList struct {
	ActorLearningRate  []float64
	CriticLearningRate []float64
	Decay              []float64

	// Whether to scale the actor learning rate by the predicted
	// variance as in
	// https://hal.inria.fr/hal-00764281/PDF/DegrisACC2012.pdf
	ScaleActorLR []bool
}

// NewConfigList returns a new ConfigList as an agent.TypedConfigList
// so that it can easily be JSON serialized/deserialized without
// knowing the underlying concrete type.
func NewConfigList(actorLR, criticLR, decay []float64) agent.TypedConfigList {
	config := LinearGaussianConfigList{
		ActorLearningRate:  actorLR,
		CriticLearningRate: criticLR,
		Decay:              decay,
	}
	return agent.NewTypedConfigList(config)
}

// Config returns an empty Config that is of the type stored by
// ConfigList
func (c LinearGaussianConfigList) Config() agent.Config {
	return LinearGaussianConfig{}
}

// Type returns the type of agent that can be constructed by Config's
// stored by the list
func (c LinearGaussianConfigList) Type() agent.Type {
	return c.Config().Type()
}

// NumFields returns the number of settable fields for the ConfigList
func (c LinearGaussianConfigList) NumFields() int {
	rValue := reflect.ValueOf(c)
	return rValue.NumField()
}

// Len returns the number of Configs stored by the list
func (c LinearGaussianConfigList) Len() int {
	return len(c.ActorLearningRate) * len(c.CriticLearningRate) * len(c.Decay)
}

// Config represents a configuration for an Actor Critic agent
type LinearGaussianConfig struct {
	ActorLearningRate  float64
	CriticLearningRate float64
	Decay              float64

	// Whether to scale the actor learning rate by the predicted
	// variance as in
	// https://hal.inria.fr/hal-00764281/PDF/DegrisACC2012.pdf
	ScaleActorLR bool
}

// CreateAgent creates the agent from the Config. Agent weights are
// always initialized to zero using this function. To initialize from
// some other distribution, use the agent's constructor manually.
func (c LinearGaussianConfig) CreateAgent(env environment.Environment,
	seed uint64) (agent.Agent, error) {

	// Create the zero weight initializer
	rand := weights.NewZeroUV() // Zero RNG
	init := weights.NewLinearUV(rand)

	return NewLinearGaussian(env, c, init, seed)
}

// ValidAgent returns whether the argument agent is a valid agent for
// construction with the Config
func (c LinearGaussianConfig) ValidAgent(a agent.Agent) bool {
	_, ok := a.(*LinearGaussian)
	return ok
}

// Validate ensures that the Config is valid
func (c LinearGaussianConfig) Validate() error {
	return nil
}

// Type returns the type of the agent constructed by the Config
func (c LinearGaussianConfig) Type() agent.Type {
	return agent.GaussianActorCriticLinear
}
