package initwfn

import G "gorgonia.org/gorgonia"

type ZeroesConfig struct {
}

// NewAdam returns a new Adam Solver
func NewZeroes() (*InitWFn, error) {
	config := ZeroesConfig{}

	return newInitWFn(config)
}

func (z ZeroesConfig) Type() Type {
	return Zeroes
}

func (g ZeroesConfig) Create() G.InitWFn {
	return G.Zeroes()
}

func (g ZeroesConfig) ValidType(t Type) bool {
	return t == Zeroes
}
