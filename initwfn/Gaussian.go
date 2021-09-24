package initwfn

import G "gorgonia.org/gorgonia"

// Gaussian implements a configuration of a weight initializer that
// draws weights from a gaussian distribution
type GaussianConfig struct {
	Mean, StdDev float64
}

// NewGaussian returns a new gaussian weight initializer
func NewGaussian(mean, stddev float64) (*InitWFn, error) {
	config := GaussianConfig{
		Mean:   mean,
		StdDev: stddev,
	}

	return newInitWFn(config)
}

// Type returns the type of initialization algorithm described by
// the configuration.
func (u GaussianConfig) Type() Type {
	return Gaussian
}

// Create returns the weight initialization algorithm as a Gorgonia
// InitWFn
func (u GaussianConfig) Create() G.InitWFn {
	return G.Gaussian(u.Mean, u.StdDev)
}
