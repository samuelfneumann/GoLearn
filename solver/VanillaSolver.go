package solver

import G "gorgonia.org/gorgonia"

// VanillaConfig describes a configuration of the vanilla gradient
// descent solver.
type VanillaConfig struct {
	StepSize float64
	Batch    int
}

// NewVanilla returns a new Vanilla Solver
func NewVanilla(stepSize float64, batchSize int) (*Solver, error) {
	vanilla := VanillaConfig{
		StepSize: stepSize,
		Batch:    int(batchSize),
	}

	return newSolver(Vanilla, vanilla)
}

// Create returns a Gorgonia Vanilla Solver as described by the
// VanillaConfig
func (a VanillaConfig) Create() G.Solver {
	solver := G.NewVanillaSolver(
		G.WithLearnRate(a.StepSize),
		G.WithBatchSize(float64(a.Batch)),
	)
	return solver
}

// ValidType returns if the given Solver type is a valid type to be
// created with this config.
func (a VanillaConfig) ValidType(t Type) bool {
	return t == Vanilla
}
