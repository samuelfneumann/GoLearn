package actorcritic

import (
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/agent/linear/continuous/policy"
	"sfneuman.com/golearn/timestep"
)

type OnlineGaussianLearner struct {
	meanWeights        *mat.Dense
	stdWeights         *mat.Dense
	step               timestep.TimeStep
	action             mat.Vector
	nextStep           timestep.TimeStep
	actorLearningRate  float64
	criticLearningRate float64
}

func NewOnlineGaussianLearner(weights map[string]*mat.Dense,
	actorLearningRate, criticLearningRate float64) *OnlineGaussianLearner {
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

	learner := OnlineGaussianLearner{meanWeights, stdWeights,
		step, nil, step, actorLearningRate, criticLearningRate}

	return &learner
}

func (g *OnlineGaussianLearner) ObserveFirst(t timestep.TimeStep) {
	if !t.First() {
		fmt.Fprintf(os.Stderr, "Warning: ObserveFirst() should only be"+
			"called on the first timestep (current timestep = %d)", t.Number)
	}
	g.step = timestep.TimeStep{}
	g.nextStep = t
}

func (g *OnlineGaussianLearner) Observe(action mat.Vector,
	nextStep timestep.TimeStep) {
	g.step = g.nextStep
	g.action = action
	g.nextStep = nextStep
}

func (g *OnlineGaussianLearner) SetWeights(weights map[string]*mat.Dense) error {
	// Set the weights for the mean
	meanWeights, ok := weights[policy.MeanWeightsKey]
	if !ok {
		return fmt.Errorf("SetWeights: no weights named \"%v\"",
			policy.MeanWeightsKey)
	}
	g.meanWeights = meanWeights

	// Set the weights for the std deviation
	stdWeights, ok := weights[policy.StdWeightsKey]
	if !ok {
		return fmt.Errorf("SetWeights: no weights named \"%v\"",
			policy.StdWeightsKey)
	}
	g.stdWeights = stdWeights

	return nil
}

func (g *OnlineGaussianLearner) Weights() map[string]*mat.Dense {
	weights := make(map[string]*mat.Dense)
	weights[policy.MeanWeightsKey] = g.meanWeights
	weights[policy.StdWeightsKey] = g.stdWeights

	return weights
}
