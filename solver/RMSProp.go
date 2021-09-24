package solver

import (
	"fmt"

	G "gorgonia.org/gorgonia"
)

// RMSProprConfig implements a specific configuration of the RMSProp
// solver
type RMSPropConfig struct {
	StepSize float64
	Epsilon  float64
	Eta      float64 // Only default value of 0.001 supported by Gorgonia
	Rho      float64
	Batch    int
	Clip     float64 // <= 0 if no clipping
}

// NewDefaultRMSProp returns a new RMSProp Solver with default
// hyperparameters
func NewDefaultRMSProp(stepSize float64, batchSize int) (*Solver, error) {
	return NewRMSProp(stepSize, 1e-8, 0.001, 0.999, batchSize, -1.0)
}

// NewRMSProp returns a new RMSProp Solver
func NewRMSProp(stepSize, epsilon, eta, rho float64, batchSize int,
	clip float64) (*Solver, error) {
	if eta != 0.001 {
		return nil, fmt.Errorf("newRMSProp: only the default value of " +
			"η = 0.001 is currently supported")
	}

	rmsprop := RMSPropConfig{
		StepSize: stepSize,
		Epsilon:  epsilon,
		Eta:      eta,
		Rho:      rho,
		Batch:    int(batchSize),
		Clip:     clip,
	}

	return newSolver(RMSProp, rmsprop)
}

// Create returns a new Gorgonia RMSProp Solver as described by the
// RMSPropConfig
func (r RMSPropConfig) Create() G.Solver {
	var solver G.Solver

	if r.Clip <= 0 {
		solver = G.NewRMSPropSolver(
			G.WithLearnRate(r.StepSize),
			G.WithEps(r.Epsilon),
			// G.WithEta(r.Eta), // Unsupported by Gorgonia
			G.WithRho(r.Rho),
			G.WithBatchSize(float64(r.Batch)),
		)
	} else {
		solver = G.NewAdamSolver(
			G.WithLearnRate(r.StepSize),
			G.WithEps(r.Epsilon),
			// G.WithEta(r.Eta), // Unsupported by Gorgonia
			G.WithRho(r.Rho),
			G.WithBatchSize(float64(r.Batch)),
			G.WithClip(r.Clip),
		)
	}
	return solver
}

// ValidType returns if the given Solver type is a valid type to be
// created with this config.
func (r RMSPropConfig) ValidType(t Type) bool {
	return t == RMSProp
}
