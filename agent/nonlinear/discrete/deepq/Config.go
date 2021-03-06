package deepq

import (
	"fmt"
	"reflect"

	"github.com/samuelfneumann/golearn/agent"
	"github.com/samuelfneumann/golearn/agent/nonlinear/discrete/policy"
	"github.com/samuelfneumann/golearn/buffer/expreplay"
	env "github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/initwfn"
	"github.com/samuelfneumann/golearn/network"
	"github.com/samuelfneumann/golearn/solver"
	G "gorgonia.org/gorgonia"
)

func init() {
	// Register ConfigList type so that it can be typed using
	// agent.TypedConfigList to help with serialization/deserialization.
	agent.Register(agent.EGreedyDeepQMLP, ConfigList{})
}

// ConfigList implements a list of Config's in a more efficient manner
// than simply using a slice of Config's.
type ConfigList struct {
	Layers      [][]int                 // Layer sizes in neural net
	Biases      [][]bool                // Whether each layer should have a bias
	Activations [][]*network.Activation // Activation of each layer
	Solver      []*solver.Solver        // Solver for learning weights

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
	Layers [][]int,
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
		Layers:               Layers,
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
	return len(c.Layers) * len(c.Biases) * len(c.Activations) *
		len(c.Solver) * len(c.InitWFn) * len(c.Epsilon) * len(c.ExpReplay) *
		len(c.Tau) * len(c.TargetUpdateInterval)
}

// Config implements a configuration for a DeepQ agent
type Config struct {
	Layers      []int                 // Layer sizes in neural net
	Biases      []bool                // Whether each layer should have a bias
	Activations []*network.Activation // Activation of each layer
	Solver      *solver.Solver        // Solver for learning weights

	// Initialization algorithm for weights
	InitWFn *initwfn.InitWFn

	// The behaviourPolicy selects actions at the current step. The
	// targetPolicy looks at the next action and selects that with the
	// highest value for the Q-learning update.
	policy    agent.EGreedyNNPolicy // Action selection
	targetNet network.NeuralNet
	trainNet  network.NeuralNet

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
	return agent.EGreedyDeepQMLP
}

// Validate checks a Config to ensure it is a valid configuration of a
// DeepQ agent.
func (c Config) Validate() error {
	// Error checking
	if len(c.Layers) != len(c.Biases) {
		msg := fmt.Sprintf("new: invalid number of biases\n\twant(%v)"+
			"\n\thave(%v)", len(c.Layers), len(c.Biases))
		return fmt.Errorf(msg)
	}

	if len(c.Layers) != len(c.Activations) {
		msg := fmt.Sprintf("new: invalid number of activations\n\twant(%v)"+
			"\n\thave(%v)", len(c.Layers), len(c.Activations))
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
	hiddenSizes := c.Layers
	biases := c.Biases
	activations := c.Activations
	init := c.InitWFn.InitWFn()
	?? := c.Epsilon

	// Behaviour policy
	behaviourPolicy, err := policy.NewMultiHeadEGreedyMLP(
		??,
		1,
		e,
		G.NewGraph(),
		hiddenSizes,
		biases,
		init,
		activations,
		seed,
	)
	if err != nil {
		return &DeepQ{}, fmt.Errorf("createAgent: could not create "+
			"behaviour policy: %v", err)
	}

	// Create the target (greedy) policy
	targetPolicy, err := policy.NewMultiHeadEGreedyMLP(
		0.0,
		1,
		e,
		G.NewGraph(),
		hiddenSizes,
		biases,
		init,
		activations,
		seed,
	)
	if err != nil {
		return &DeepQ{}, fmt.Errorf("new: could not create target policy")
	}

	// Create the target network
	targetNetPolicy, err := policy.NewMultiHeadEGreedyMLP(
		0.0,
		c.BatchSize(),
		e,
		G.NewGraph(),
		hiddenSizes,
		biases,
		init,
		activations,
		seed,
	)
	if err != nil {
		return &DeepQ{}, fmt.Errorf("new: could not create target policy")
	}
	c.targetNet = targetNetPolicy.Network()

	// Create the target network
	trainNetPolicy, err := policy.NewMultiHeadEGreedyMLP(
		0.0,
		c.BatchSize(),
		e,
		G.NewGraph(),
		hiddenSizes,
		biases,
		init,
		activations,
		seed,
	)
	if err != nil {
		return &DeepQ{}, fmt.Errorf("new: could not create target policy")
	}
	c.trainNet = trainNetPolicy.Network()

	// Set the policies to have the same weights
	network.Set(behaviourPolicy.Network(), targetPolicy.Network())
	network.Set(c.targetNet, targetPolicy.Network())
	network.Set(c.trainNet, targetPolicy.Network())

	// Behaviour policy can be set to evaluation mode to get the target
	// policy since it is an EGreedy policy and DeepQ's target policy
	// is greedy with respect to action values.
	c.policy = behaviourPolicy.(agent.EGreedyNNPolicy)

	return New(e, c, seed)
}
