package vanillapg

import (
	"github.com/samuelfneumann/golearn/agent"
	"github.com/samuelfneumann/golearn/initwfn"
	"github.com/samuelfneumann/golearn/network"
	"github.com/samuelfneumann/golearn/solver"
)

// Config implements an interface for any VanillaPG configuration.
// This is needed so that the VanillaPG constructor can take in
// either a Gaussian or Categorical (or any other distribution)
// Config struct.
type config interface {
	agent.Config

	trainPolicy() agent.LogPdfOfer
	behaviourPolicy() agent.NNPolicy

	valueFn() network.NeuralNet
	trainValueFn() network.NeuralNet

	initWFn() *initwfn.InitWFn

	policySolver() *solver.Solver
	vSolver() *solver.Solver

	batchSize() int
	epochLength() int

	// Number of gradient steps to take for the value functions per
	// epoch
	valueGradSteps() int

	// FinishEpisodeOnEpochEnd denotes if the current episode should
	// be finished before starting a new epoch. If true, then the
	// agent is updated when the current epoch ends, then the current
	// episode is finished, then the next epoch starts. If false, the
	// agent is updated when the current epoch is finished, and the
	// next epoch starts at the next timestep, which may be in the
	// middle of an episode.
	finishEpisodeOnEpochEnd() bool

	// Generalized Advantage Estimation
	lambda() float64
	gamma() float64
}
