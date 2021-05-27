package actorcritic

import (
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/agent/linear/continuous/policy"
	"sfneuman.com/golearn/timestep"
)

type GaussianLearner struct {
	meanWeights  *mat.Dense
	stdWeights   *mat.Dense
	step         timestep.TimeStep
	action       mat.Vector
	nextStep     timestep.TimeStep
	learningRate float64
}

func NewGaussianLearner(weights map[string]*mat.Dense,
	learningRate float64) *GaussianLearner {
	step := timestep.TimeStep{}

	// Get the weights for the mean
	meanWeights, ok := weights[policy.MeanWeightsKey]
	if !ok {
		panic("no weights for predicing the mean")
	}

	// Get the weights for the variance
	stdWeights, ok := weights[policy.StdWeightsKey]
	if !ok {
		panic("no weights for predicing the standard deviation")
	}

	learner := GaussianLearner{meanWeights, stdWeights, step, nil, step,
		learningRate}

	return &learner
}

func (g *GaussianLearner) ObserveFirst(t timestep.TimeStep) {
	if !t.First() {
		fmt.Fprintf(os.Stderr, "Warning: ObserveFirst() should only be"+
			"called on the first timestep (current timestep = %d)", t.Number)
	}
	g.step = timestep.TimeStep{}
	g.nextStep = t
}

func (g *GaussianLearner) Observe(action mat.Vector,
	nextStep timestep.TimeStep) {
	g.step = g.nextStep
	g.action = action
	g.nextStep = nextStep
}
