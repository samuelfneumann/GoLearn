package solver

import G "gorgonia.org/gorgonia"

// AdamConfig describes a configuration of the Adam solver
type AdamConfig struct {
	StepSize float64
	Epsilon  float64 // Smoothing factor
	Beta1    float64
	Beta2    float64
	Batch    int
}

// NewDefaultAdam returns a new Adam Solver with default hyperparameters
func NewDefaultAdam(stepSize float64, batchSize int) (*Solver, error) {
	return NewAdam(stepSize, 1e-8, 0.9, 0.999, batchSize)
}

// NewAdam returns a new Adam Solver
func NewAdam(stepSize, epsilon, beta1, beta2 float64, batchSize int) (*Solver,
	error) {
	adam := AdamConfig{
		StepSize: stepSize,
		Epsilon:  epsilon,
		Beta1:    beta1,
		Beta2:    beta2,
		Batch:    int(batchSize),
	}

	return newSolver(Adam, adam)
}

// Create returns a new Gorgonia Adam Solver as described by the
// AdamConfig
func (a AdamConfig) Create() G.Solver {
	solver := G.NewAdamSolver(
		G.WithLearnRate(a.StepSize),
		G.WithEps(a.Epsilon),
		G.WithBeta1(a.Beta1),
		G.WithBeta2(a.Beta2),
		G.WithBatchSize(float64(a.Batch)),
	)
	return solver
}

// ValidType returns if the given Solver type is a valid type to be
// created with this config.
func (a AdamConfig) ValidType(t Type) bool {
	return t == Adam
}
