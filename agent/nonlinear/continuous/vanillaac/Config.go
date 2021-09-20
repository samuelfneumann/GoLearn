package vanillaac

import (
	"github.com/samuelfneumann/golearn/agent"
	"github.com/samuelfneumann/golearn/expreplay"
	"github.com/samuelfneumann/golearn/initwfn"
	"github.com/samuelfneumann/golearn/network"
	"github.com/samuelfneumann/golearn/solver"
)

// config implements an interface for any VanillaAC configuration.
// This is needed so that the VanillaAC constructor can take in
// either a Gaussian or Categorical (or any other distribution)
// configuration struct.
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

	// Number of gradient steps to take for the value functions per
	// epoch
	valueGradSteps() int

	// expReplay returns the experience replayer to use
	expReplay() expreplay.Config
}
