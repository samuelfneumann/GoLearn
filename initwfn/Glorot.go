package initwfn

import G "gorgonia.org/gorgonia"

// GlorotUConfig implements a configuration of the Glorot Uniform
// initialization algorithm.
type GlorotUConfig struct {
	Gain float64
}

// NewGlorotU returns a new Glorot Uniform weight initializer
func NewGlorotU(gain float64) (*InitWFn, error) {
	config := GlorotUConfig{
		Gain: gain,
	}

	return newInitWFn(config)
}

// Type returns the type of initialization algorithm described by
// the configuration.
func (g GlorotUConfig) Type() Type {
	return GlorotU
}

// Create returns the weight initialization algorithm as a Gorgonia
// InitWFn
func (g GlorotUConfig) Create() G.InitWFn {
	return G.GlorotU(g.Gain)
}

// GlorotNConfig implements a configuration of the Glorot Normal
// initialization algorithm.
type GlorotNConfig struct {
	Gain float64
}

// NewGlorotN returns a new Glorot Normal weight initializer.
func NewGlorotN(gain float64) (*InitWFn, error) {
	config := GlorotNConfig{
		Gain: gain,
	}

	return newInitWFn(config)
}

// Type returns the type of initialization algorithm described by the
// configuration.
func (g GlorotNConfig) Type() Type {
	return GlorotN
}

// Create returns the weight initialization algorithm as a Gorgonia
// InitWFn
func (g GlorotNConfig) Create() G.InitWFn {
	return G.GlorotN(g.Gain)
}
