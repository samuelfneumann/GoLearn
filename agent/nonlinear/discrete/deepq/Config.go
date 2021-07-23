package deepq

import (
	"fmt"
	"reflect"

	G "gorgonia.org/gorgonia"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/agent/nonlinear/discrete/policy"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/expreplay"
	"sfneuman.com/golearn/initwfn"
	"sfneuman.com/golearn/network"
	"sfneuman.com/golearn/solver"
)

func init() {
	// Register ConfigList type so that it can be typed using
	// agent.TypedConfigList to help with serialization/deserialization.
	agent.Register(agent.EGreedyDeepQ, ConfigList{})
}

// ConfigList implements a list of Config's in a more efficient manner
// than simply using a slice of Config's.
type ConfigList struct {
	PolicyLayers [][]int                 // Layer sizes in neural net
	Biases       [][]bool                // Whether each layer should have a bias
	Activations  [][]*network.Activation // Activation of each layer
	Solver       []*solver.Solver        // Solver for learning weights

	// Initialization algorithm for weights
	InitWFn []*initwfn.InitWFn

	Epsilon []float64 // Behaviour policy epsilon

	// Experience replay parameters
	ExpReplay []expreplay.Config

	// Target net updates
	Tau                  []float64 // Polyak averaging constant
	TargetUpdateInterval []int     // Number of steps target network updates
}

// NewConfigList returns a new ConfigList as an agent.TypedConfigList.
// Because the returned value is a TypedList, it can safely be JSON
// serialized and deserialized without specifying what the type of
// the ConfigList is.
func NewConfigList(
	PolicyLayers [][]int,
	Biases [][]bool,
	Activations [][]*network.Activation,
	Solver []*solver.Solver,
	InitWFn []*initwfn.InitWFn,
	Epsilon []float64,
	ExpReplay []expreplay.Config,
	Tau []float64,
	TargetUpdateInterval []int,
) agent.TypedConfigList {
	configs := ConfigList{
		PolicyLayers:         PolicyLayers,
		Biases:               Biases,
		Activations:          Activations,
		Solver:               Solver,
		InitWFn:              InitWFn,
		Epsilon:              Epsilon,
		ExpReplay:            ExpReplay,
		Tau:                  Tau,
		TargetUpdateInterval: TargetUpdateInterval,
	}

	return agent.NewTypedConfigList(configs)
}

// Type returns the type of Config stored in the list
func (c ConfigList) Type() agent.Type {
	return c.Config().Type()
}

// NumFields returns the number of settable fields in a Config
func (c ConfigList) NumFields() int {
	rValue := reflect.ValueOf(c)
	return rValue.NumField()
}

// Config returns an empty Config of the same type as that stored
// by the ConfigList
func (c ConfigList) Config() agent.Config {
	return Config{}
}

// Len returns the number of Config's in the list
func (c ConfigList) Len() int {
	return len(c.PolicyLayers) * len(c.Biases) * len(c.Activations) *
		len(c.Solver) * len(c.InitWFn) * len(c.Epsilon) * len(c.ExpReplay) *
		len(c.Tau) * len(c.TargetUpdateInterval)
}

// Config implements a configuration for a DeepQ agent
type Config struct {
	PolicyLayers []int                 // Layer sizes in neural net
	Biases       []bool                // Whether each layer should have a bias
	Activations  []*network.Activation // Activation of each layer
	Solver       *solver.Solver        // Solver for learning weights

	// Initialization algorithm for weights
	InitWFn *initwfn.InitWFn

	// The behaviourPolicy selects actions at the current step. The
	// targetPolicy looks at the next action and selects that with the
	// highest value for the Q-learning update.
	behaviourPolicy agent.EGreedyNNPolicy // Action selection
	targetPolicy    agent.EGreedyNNPolicy // Greedy next-action selection

	Epsilon float64 // Behaviour policy epsilon

	// Experience replay parameters
	ExpReplay expreplay.Config

	// Target net updates
	Tau                  float64 // Polyak averaging constant
	TargetUpdateInterval int     // Number of steps target network updates
}

// BatchSize returns the batch size of the agent constructed using this
// Config
func (c Config) BatchSize() int {
	return c.ExpReplay.SampleSize
}

// Type returns the type of the configuration
func (c Config) Type() agent.Type {
	return agent.EGreedyDeepQ
}

// Validate checks a Config to ensure it is a valid configuration of a
// DeepQ agent.
func (c Config) Validate() error {
	// Error checking

	if len(c.PolicyLayers) != len(c.Biases) {
		msg := fmt.Sprintf("new: invalid number of biases\n\twant(%v)"+
			"\n\thave(%v)", len(c.PolicyLayers), len(c.Biases))
		return fmt.Errorf(msg)
	}

	if len(c.PolicyLayers) != len(c.Activations) {
		msg := fmt.Sprintf("new: invalid number of activations\n\twant(%v)"+
			"\n\thave(%v)", len(c.PolicyLayers), len(c.Activations))
		return fmt.Errorf(msg)
	}

	if c.TargetUpdateInterval < 1 {
		err := fmt.Errorf("new: target networks must be updated at positive "+
			"timestep intervals \n\twant(>0) \n\thave(%v)",
			c.TargetUpdateInterval)
		return err
	}

	return nil
}

// ValidAgent returns whether the agent is valid for the configuration.
// That is, whether Agent a can be constructed with Config c.
func (c Config) ValidAgent(a agent.Agent) bool {
	_, ok := a.(*DeepQ)
	return ok
}

// CreateAgent creates a new DeepQ agent based on the configuration
func (c Config) CreateAgent(e env.Environment, s uint64) (agent.Agent, error) {
	seed := int64(s)

	// Extract configuration variables
	hiddenSizes := c.PolicyLayers
	biases := c.Biases
	activations := c.Activations
	init := c.InitWFn.InitWFn()
	ε := c.Epsilon

	// Behaviour policy
	g := G.NewGraph()
	behaviourPolicy, err := policy.NewMultiHeadEGreedyMLP(
		ε,
		e,
		g,
		hiddenSizes,
		biases,
		init,
		activations,
		seed,
	)
	if err != nil {
		return &DeepQ{}, err
	}

	// Create the target policy for action selection
	targetPolicyClone, err := behaviourPolicy.Clone()
	if err != nil {
		return &DeepQ{}, fmt.Errorf("new: could not create target policy")
	}
	targetPolicy := targetPolicyClone.(agent.EGreedyNNPolicy)
	targetPolicy.SetEpsilon(0.0)

	c.behaviourPolicy = behaviourPolicy
	c.targetPolicy = targetPolicy

	return New(e, c, seed)
}
