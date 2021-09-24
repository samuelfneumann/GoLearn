package initwfn

import G "gorgonia.org/gorgonia"

// HeUConfig implements a configuration of the He uniform
// initialization algorithm.
type HeUConfig struct {
	Gain float64
}

// NewHeU returns a new He Uniform weight initializer
func NewHeU(gain float64) (*InitWFn, error) {
	config := HeUConfig{
		Gain: gain,
	}

	return newInitWFn(config)
}

// Type returns the type of initialization algorithm described by
// the configuration.
func (h HeUConfig) Type() Type {
	return HeU
}

// Create returns the weight initialization algorithm as a Gorgonia
// InitWFn
func (h HeUConfig) Create() G.InitWFn {
	return G.HeU(h.Gain)
}

// HeNConfig implements a configuration of the He normal
// initialization algorithm.
type HeNConfig struct {
	Gain float64
}

// NewHeN returns a new He Nniform weight initializer
func NewHeN(gain float64) (*InitWFn, error) {
	config := HeNConfig{
		Gain: gain,
	}

	return newInitWFn(config)
}

// Type returns the type of initialization algorithm described by
// the configuration.
func (h HeNConfig) Type() Type {
	return HeN
}

// Create returns the weight initialization algorithm as a Gorgonia
// InitWFn
func (h HeNConfig) Create() G.InitWFn {
	return G.HeN(h.Gain)
}
