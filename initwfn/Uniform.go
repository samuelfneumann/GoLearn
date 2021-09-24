package initwfn

import G "gorgonia.org/gorgonia"

// Uniform implements a configuration of a weight initializer that
// draws weights from a uniform distribution
type UniformConfig struct {
	Low, High float64
}

// NewUniform returns a new uniform weight initializer
func NewUniform(low, high float64) (*InitWFn, error) {
	config := UniformConfig{
		Low:  low,
		High: high,
	}

	return newInitWFn(config)
}

// Type returns the type of initialization algorithm described by
// the configuration.
func (u UniformConfig) Type() Type {
	return Uniform
}

// Create returns the weight initialization algorithm as a Gorgonia
// InitWFn
func (u UniformConfig) Create() G.InitWFn {
	return G.Uniform(u.Low, u.High)
}
