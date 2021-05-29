package actorcritic

import (
	"fmt"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/agent/linear/continuous/policy"
	"sfneuman.com/golearn/timestep"
)

// OnlineGaussianLearner learns only using a single action
type OnlineGaussianLearner struct {
	meanWeights   *mat.VecDense
	stdWeights    *mat.VecDense
	criticWeights *mat.VecDense

	// // Eligibility traces
	// meanTrace   *mat.Dense
	// stdTrace    *mat.Dense
	// criticTrace *mat.Dense

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
	meanWeightsMat, ok := weights[policy.MeanWeightsKey]
	if !ok {
		panic("no weights for predicing the mean")
	}
	r, c := meanWeightsMat.Dims()
	if r != 1 {
		panic("can only have a single set of weights")
	}
	meanWeights := mat.NewVecDense(c, meanWeightsMat.RawMatrix().Data)

	// Get the weights for the variance
	stdWeightsMat, ok := weights[policy.StdWeightsKey]
	if !ok {
		panic("no weights for predicing the standard deviation")
	}
	r, c = stdWeightsMat.Dims()
	if r != 1 {
		panic("can only have a single set of weights")
	}
	stdWeights := mat.NewVecDense(c, stdWeightsMat.RawMatrix().Data)

	// Critic weights
	criticWeights := mat.NewVecDense(c, nil)

	learner := OnlineGaussianLearner{meanWeights, stdWeights, criticWeights,
		step, nil, step, actorLearningRate, criticLearningRate}

	return &learner
}

func (g *OnlineGaussianLearner) TdError(t timestep.Transition) float64 {
	stateValue := mat.Dot(g.criticWeights, t.State)
	nextStateValue := mat.Dot(g.criticWeights, t.NextState)

	return t.Reward + t.Discount*nextStateValue - stateValue
}

func (g *OnlineGaussianLearner) Step() {
	discount := g.nextStep.Discount
	state := g.step.Observation.(*mat.VecDense)
	nextState := g.nextStep.Observation
	reward := g.nextStep.Reward

	stateValue := mat.Dot(g.criticWeights, state)
	nextStateValue := mat.Dot(g.criticWeights, nextState)

	tdError := reward + discount*nextStateValue - stateValue

	// Update the critic
	g.criticWeights.AddScaledVec(g.criticWeights, g.criticLearningRate*tdError,
		state)

	// Actor gradients
	std := math.Exp(mat.Dot(g.stdWeights, state))
	mean := mat.Dot(g.meanWeights, state)

	// Actor gradient scales
	if g.action.Len() != 1 {
		panic("actions must be 1-dimensional for GaussianLearner")
	}

	action := g.action.AtVec(0)
	meanGradScale := (1 / (std * std)) * (action - mean)
	meanGradScale *= (g.actorLearningRate * tdError)
	stdGradScale := math.Pow(((action-mean)/std), 2) - 1.0
	stdGradScale *= (g.actorLearningRate * tdError)

	// Update actor
	g.meanWeights.AddScaledVec(g.meanWeights, meanGradScale, state)
	g.stdWeights.AddScaledVec(g.stdWeights, stdGradScale, state)

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
	meanWeightsMat, ok := weights[policy.MeanWeightsKey]
	if !ok {
		return fmt.Errorf("SetWeights: no weights named \"%v\"",
			policy.MeanWeightsKey)
	}
	r, c := meanWeightsMat.Dims()
	if r != 1 {
		return fmt.Errorf("SetWeights: too many rows for %v \n\twant(1)"+
			"\n\thave(%v)", policy.MeanWeightsKey, r)
	}
	g.meanWeights = mat.NewVecDense(c, meanWeightsMat.RawMatrix().Data)

	// Set the weights for the std deviation
	stdWeightsMat, ok := weights[policy.StdWeightsKey]
	if !ok {
		return fmt.Errorf("SetWeights: no weights named \"%v\"",
			policy.StdWeightsKey)
	}
	r, c = stdWeightsMat.Dims()
	if r != 1 {
		return fmt.Errorf("SetWeights: too many rows for %v \n\twant(1)"+
			"\n\thave(%v)", policy.StdWeightsKey, r)
	}
	g.stdWeights = mat.NewVecDense(c, stdWeightsMat.RawMatrix().Data)

	// Set the critic weights
	criticWeightsMat, ok := weights[policy.CriticWeightsKey]
	if !ok {
		return fmt.Errorf("SetWeights: no weights named \"%v\"",
			policy.CriticWeightsKey)
	}
	r, c = criticWeightsMat.Dims()
	if r != 1 {
		return fmt.Errorf("SetWeights: too many rows for %v \n\twant(1)"+
			"\n\thave(%v)", policy.CriticWeightsKey, r)
	}
	g.criticWeights = mat.NewVecDense(c, criticWeightsMat.RawMatrix().Data)

	return nil
}

func (g *OnlineGaussianLearner) Weights() map[string]*mat.Dense {
	weights := make(map[string]*mat.Dense)

	// Return the mean weights
	weights[policy.MeanWeightsKey] = mat.NewDense(1, g.meanWeights.Len(),
		g.meanWeights.RawVector().Data)

	// Return the std weights
	weights[policy.StdWeightsKey] = mat.NewDense(1, g.stdWeights.Len(),
		g.stdWeights.RawVector().Data)

	// Return the critic weights
	weights[policy.CriticWeightsKey] = mat.NewDense(1, g.criticWeights.Len(),
		g.criticWeights.RawVector().Data)

	return weights
}
