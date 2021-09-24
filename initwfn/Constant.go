package initwfn

import G "gorgonia.org/gorgonia"

// ZeroesConfig implements a configuration of a zero weight initializer
type ZeroesConfig struct{}

// Zeroes returns a new zeroes weight intializer
func NewZeroes() (*InitWFn, error) {
	config := ZeroesConfig{}

	return newInitWFn(config)
}

// Type returns the type of the weight initializer created using this
// config
func (z ZeroesConfig) Type() Type {
	return Zeroes
}

// Create creates the Gorgonia weight initializer from this
// initializer config
func (z ZeroesConfig) Create() G.InitWFn {
	return G.Zeroes()
}

// OnesConfig implements a configuration of a weight initializer that
// initializes all weights to 1.
type OnesConfig struct{}

// Ones returns a new zeroes weight intializer
func NewOnes() (*InitWFn, error) {
	config := OnesConfig{}

	return newInitWFn(config)
}

// Type returns the type of the weight initializer created using this
// config
func (o OnesConfig) Type() Type {
	return Ones
}

// Create creates the Gorgonia weight initializer from this
// initializer config
func (o OnesConfig) Create() G.InitWFn {
	return G.Ones()
}

// ConstantConfig implements a configuration of a weight initializer
// that initializes all weights to a constant value.
type ConstantConfig struct {
	Value float64
}

// Constant returns a new zeroes weight intializer
func NewConstant(value float64) (*InitWFn, error) {
	config := ConstantConfig{value}

	return newInitWFn(config)
}

// Type returns the type of the weight initializer created using this
// config
func (c ConstantConfig) Type() Type {
	return Constant
}

// Create creates the Gorgonia weight initializer from this
// initializer config
func (c ConstantConfig) Create() G.InitWFn {
	return G.ValuesOf(c.Value)
}
