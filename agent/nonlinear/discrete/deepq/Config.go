package deepq

import (
	"fmt"

	G "gorgonia.org/gorgonia"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/agent/nonlinear/discrete/policy"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/expreplay"
	"sfneuman.com/golearn/initwfn"
	"sfneuman.com/golearn/network"
	"sfneuman.com/golearn/solver"
)

// Config implements a configuration for a DeepQ agent
type Config struct {
	Policy agent.PolicyType // Type of policy to create

	// Policy       agent.EGreedyNNPolicy
	PolicyLayers []int                 // Layer sizes in neural net
	Biases       []bool                // Whether each layer should have a bias
	Activations  []*network.Activation // Activation of each layer
	Solver       *solver.Solver        // Solver for learning weights

	// Initialization algorithm for weights
	InitWFn initwfn.InitWFn

	// The behaviourPolicy selects actions at the current step. The
	// targetPolicy looks at the next action and selects that with the
	// highest value for the Q-learning update.
	behaviourPolicy agent.EGreedyNNPolicy // Action selection
	targetPolicy    agent.EGreedyNNPolicy // Greedy next-action selection

	Epsilon float64 // Behaviour policy epsilon

	// Experience replay parameters
	Remover         expreplay.Selector
	Sampler         expreplay.Selector
	MaximumCapacity int
	MinimumCapacity int

	// Target net updates
	Tau                  float64 // Polyak averaging constant
	TargetUpdateInterval int     // Number of steps target network updates
}

// BatchSize returns the batch size of the agent constructed using this
// Config
func (c *Config) BatchSize() int {
	return c.Sampler.BatchSize()
}

// Validate checks a Config to ensure it is a valid configuration of a
// DeepQ agent.
func (c *Config) Validate() error {
	// Error checking
	if c.Policy != agent.EGreedy {
		return fmt.Errorf("cannot create %v policy for DeepQ "+
			"configuration, must be %v", c.Policy, agent.EGreedy)
	}

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
func (c *Config) ValidAgent(a agent.Agent) bool {
	_, ok := a.(*DeepQ)
	return ok
}

// CreateAgent creates a new DeepQ agent based on the configuration
func (c *Config) CreateAgent(e env.Environment, s uint64) (agent.Agent, error) {
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
