package initwfn

import G "gorgonia.org/gorgonia"

type GlorotUConfig struct {
	Gain float64
}

// NewAdam returns a new Adam Solver
func NewGlorotU(gain float64) (*InitWFn, error) {
	config := GlorotUConfig{
		Gain: gain,
	}

	return newInitWFn(GlorotU, config)
}

func (g GlorotUConfig) Create() G.InitWFn {
	return G.GlorotU(g.Gain)
}

func (g GlorotUConfig) ValidType(t Type) bool {
	return t == GlorotU
}

type GlorotNConfig struct {
	Gain float64
}

func NewGlorotN(gain float64) (*InitWFn, error) {
	config := GlorotNConfig{
		Gain: gain,
	}

	return newInitWFn(GlorotN, config)
}

func (g GlorotNConfig) Create() G.InitWFn {
	return G.GlorotN(g.Gain)
}

func (g GlorotNConfig) ValidType(t Type) bool {
	return t == GlorotN
}
