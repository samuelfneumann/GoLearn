package deepq

import (
	"fmt"

	G "gorgonia.org/gorgonia"
	"sfneuman.com/golearn/expreplay"
	"sfneuman.com/golearn/network"
)

// Config implements a configuration for a DeepQ agent
type Config struct {
	PolicyLayers []int                // Layer sizes in neural net
	Biases       []bool               // Whether each layer should have a bias
	Activations  []network.Activation // Activation of each layer
	InitWFn      G.InitWFn            // Initialization algorithm for weights
	Solver       G.Solver             // Solver for learning weights

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
func (c Config) BatchSize() int {
	return c.Sampler.BatchSize()
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
