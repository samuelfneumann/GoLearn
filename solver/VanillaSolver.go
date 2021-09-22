package solver

import G "gorgonia.org/gorgonia"

// VanillaConfig describes a configuration of the vanilla gradient
// descent solver.
type VanillaConfig struct {
	StepSize float64
	Batch    int
	Clip     float64 // <= 0 if no clipping
}

// NewVanilla returns a new Vanilla Solver
func NewVanilla(stepSize float64, batchSize int,
	clip float64) (*Solver, error) {
	vanilla := VanillaConfig{
		StepSize: stepSize,
		Batch:    int(batchSize),
		Clip:     clip,
	}

	return newSolver(Vanilla, vanilla)
}

// Create returns a Gorgonia Vanilla Solver as described by the
// VanillaConfig
func (v VanillaConfig) Create() G.Solver {
	var solver G.Solver

	if v.Clip <= 0 {
		solver = G.NewVanillaSolver(
			G.WithLearnRate(v.StepSize),
			G.WithBatchSize(float64(v.Batch)),
		)
	} else {
		solver = G.NewVanillaSolver(
			G.WithLearnRate(v.StepSize),
			G.WithBatchSize(float64(v.Batch)),
			G.WithClip(v.Clip),
		)
	}
	return solver
}

// ValidType returns if the given Solver type is a valid type to be
// created with this config.
func (v VanillaConfig) ValidType(t Type) bool {
	return t == Vanilla
}
