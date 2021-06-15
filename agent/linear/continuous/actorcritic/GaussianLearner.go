package actorcritic

import (
	"fmt"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/agent/linear/continuous/policy"
	"sfneuman.com/golearn/timestep"
)

// GaussianLearner does not use experience replay and 1d actions
type GaussianLearner struct {
	meanWeights   *mat.VecDense
	stdWeights    *mat.VecDense
	criticWeights *mat.VecDense

	// Eligibility traces
	decay       float64
	meanTrace   *mat.VecDense
	stdTrace    *mat.VecDense
	criticTrace *mat.VecDense

	step               timestep.TimeStep
	action             mat.Vector
	nextStep           timestep.TimeStep
	actorLearningRate  float64
	criticLearningRate float64
}

func NewGaussianLearner(gaussian *policy.Gaussian,
	actorLearningRate, criticLearningRate, decay float64) (*GaussianLearner, error) {
	step := timestep.TimeStep{}

	learner := GaussianLearner{nil, nil, nil, decay, nil, nil, nil, step, nil,
		step, actorLearningRate, criticLearningRate}

	weights := gaussian.Weights()

	// Policy has no concept of a critic, so init the critic weights
	// before setting
	r, c := weights[policy.MeanWeightsKey].Dims()
	criticWeights := mat.NewDense(r, c, nil)
	weights[policy.CriticWeightsKey] = criticWeights
	err := learner.SetWeights(weights)

	// Set up eligibility traces initialized to 0
	learner.meanTrace = mat.NewVecDense(learner.meanWeights.Len(), nil)
	learner.stdTrace = mat.NewVecDense(learner.stdWeights.Len(), nil)
	learner.criticTrace = mat.NewVecDense(learner.criticWeights.Len(), nil)

	return &learner, err
}

func (g *GaussianLearner) TdError(t timestep.Transition) float64 {
	stateValue := mat.Dot(g.criticWeights, t.State)
	nextStateValue := mat.Dot(g.criticWeights, t.NextState)

	return t.Reward + t.Discount*nextStateValue - stateValue
}

func (g *GaussianLearner) Step() {
	discount := g.nextStep.Discount
	state := g.step.Observation.(*mat.VecDense)
	nextState := g.nextStep.Observation
	reward := g.nextStep.Reward

	stateValue := mat.Dot(g.criticWeights, state)
	nextStateValue := mat.Dot(g.criticWeights, nextState)

	tdError := reward + discount*nextStateValue - stateValue

	// Update the critic
	g.criticTrace.AddScaledVec(state, discount*g.decay, g.criticTrace)
	g.criticWeights.AddScaledVec(g.criticWeights, g.criticLearningRate*tdError,
		g.criticTrace)

	// Actor gradient
	std := math.Exp(mat.Dot(g.stdWeights, state))
	mean := mat.Dot(g.meanWeights, state)

	// Actor gradient scales
	if g.action.Len() != 1 {
		panic("actions must be 1-dimensional for GaussianLearner")
	}

	action := g.action.AtVec(0)
	meanGradScale := (1 / (std * std)) * (action - mean)
	meanGradScale *= tdError
	stdGradScale := math.Pow(((action-mean)/std), 2) - 1.0
	stdGradScale *= tdError

	// Compute actor gradients
	meanGrad := mat.NewVecDense(state.Len(), state.RawVector().Data)
	meanGrad.ScaleVec(meanGradScale, meanGrad)
	stdGrad := mat.NewVecDense(state.Len(), state.RawVector().Data)
	stdGrad.ScaleVec(stdGradScale, stdGrad)

	// Update actor traces
	g.meanTrace.AddScaledVec(g.meanTrace, discount*g.decay, meanGrad)
	g.stdTrace.AddScaledVec(g.stdTrace, discount*g.decay, stdGrad)

	// Update actor
	g.meanWeights.AddScaledVec(g.meanWeights, g.actorLearningRate, g.meanTrace)
	g.stdWeights.AddScaledVec(g.stdWeights, g.actorLearningRate/math.Pow(std, 2), g.stdTrace)
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

func (g *GaussianLearner) SetWeights(weights map[string]*mat.Dense) error {
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

func (g *GaussianLearner) Weights() map[string]*mat.Dense {
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
