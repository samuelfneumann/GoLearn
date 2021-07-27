package actorcritic

import (
	"fmt"
	"math"
	"os"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/agent/linear/continuous/policy"
	"sfneuman.com/golearn/timestep"
)

// GaussianLearner implements the learner for the Linear Gaussian
// Actor-Critic algorithm:
//
// https://hal.inria.fr/hal-00764281/PDF/DegrisACC2012.pdf
//
// This learner uses eligibility traces and linear function
// approximation as outlined in the paper above. The learner also
// scales the learning rate for the standard deviation by the predicted
// variance at each gradient step which has been shown to stabilize
// training
type GaussianLearner struct {
	// Linear function approximation weights
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
	policy             *policy.Gaussian // The policy we are learning
}

// NewGaussianLearner creates and returns a new GaussianLearner
func NewGaussianLearner(gaussian *policy.Gaussian, actorLearningRate,
	criticLearningRate, decay float64) (*GaussianLearner, error) {
	step := timestep.TimeStep{}

	learner := GaussianLearner{nil, nil, nil, decay, nil, nil, nil, step, nil,
		step, actorLearningRate, criticLearningRate, gaussian}

	// Weights of the policy to learn
	weights := gaussian.Weights()

	// Policy has no concept of a critic, so initialize the critic weights
	// before setting
	r, c := weights[policy.MeanWeightsKey].Dims()
	criticWeights := mat.NewDense(r, c, nil)
	weights[policy.CriticWeightsKey] = criticWeights
	if err := learner.SetWeights(weights); err != nil {
		return &GaussianLearner{}, err
	}

	// Set up eligibility traces initialized to 0
	learner.meanTrace = mat.NewVecDense(learner.meanWeights.Len(), nil)
	learner.stdTrace = mat.NewVecDense(learner.stdWeights.Len(), nil)
	learner.criticTrace = mat.NewVecDense(learner.criticWeights.Len(), nil)

	return &learner, nil
}

// TdError calculates the TD error generated by the learner on some
// transition.
func (g *GaussianLearner) TdError(t timestep.Transition) float64 {
	stateValue := mat.Dot(g.criticWeights, t.State)
	nextStateValue := mat.Dot(g.criticWeights, t.NextState)

	return t.Reward + t.Discount*nextStateValue - stateValue
}

// Step takes one learning step
func (g *GaussianLearner) Step() {
	// fmt.Println("MEAN:", g.meanWeights)
	// fmt.Println("STD:", g.stdWeights)
	// fmt.Println("CRITIC:", g.criticWeights)

	// Get variables needed to compute state values
	discount := g.nextStep.Discount
	state := g.step.Observation
	nextState := g.nextStep.Observation
	reward := g.nextStep.Reward

	// Compute the state values and TD error to update the actor and critic
	stateValue := mat.Dot(g.criticWeights, state)
	nextStateValue := mat.Dot(g.criticWeights, nextState)
	tdError := reward + discount*nextStateValue - stateValue

	// Update the critic
	g.criticTrace.AddScaledVec(state, discount*g.decay, g.criticTrace)
	g.criticWeights.AddScaledVec(g.criticWeights, g.criticLearningRate*tdError,
		g.criticTrace)

	// Compute the std and mean of the policy
	stdVec := g.policy.Std(state)
	meanVec := g.policy.Mean(state)
	if stdVec.Len() != 1 || meanVec.Len() != 1 {
		panic("Step: actions should be 1-dimensional")
	}
	std := stdVec.AtVec(0)
	mean := meanVec.AtVec(0)

	// Actor gradient scales
	if g.action.Len() != 1 {
		panic("Step: actions must be 1-dimensional for GaussianLearner")
	}
	action := g.action.AtVec(0)

	// Compute actor gradients
	meanGradScale := (action - mean) / math.Pow(std, 2)
	meanGrad := mat.NewVecDense(state.Len(), nil)
	meanGrad.ScaleVec(meanGradScale, state)

	stdGradScale := math.Pow(((action-mean)/std), 2) - 1.0
	stdGrad := mat.NewVecDense(state.Len(), nil)
	stdGrad.ScaleVec(stdGradScale, state)

	// Update actor traces
	g.meanTrace.AddScaledVec(meanGrad, discount*g.decay, g.meanTrace)
	g.stdTrace.AddScaledVec(stdGrad, discount*g.decay, g.stdTrace)

	// Update actor weights
	g.meanWeights.AddScaledVec(g.meanWeights, g.actorLearningRate*tdError,
		g.meanTrace)
	g.stdWeights.AddScaledVec(g.stdWeights,
		tdError*(g.actorLearningRate/math.Pow(std, 2)), g.stdTrace)
}

// ObserveFirst observes and records the first episodic timestep
func (g *GaussianLearner) ObserveFirst(t timestep.TimeStep) {
	if !t.First() {
		fmt.Fprintf(os.Stderr, "Warning: ObserveFirst() should only be"+
			"called on the first timestep (current timestep = %d)", t.Number)
	}
	g.step = timestep.TimeStep{}
	g.nextStep = t
}

// Observe observes and records any timestep other than the first timestep
func (g *GaussianLearner) Observe(action mat.Vector,
	nextStep timestep.TimeStep) {
	g.step = g.nextStep
	g.action = action
	g.nextStep = nextStep
}

// SetWeights sets the weight pointers to point to a new set of weights.
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

// Cleanup at the end of an episode
func (g *GaussianLearner) EndEpisode() {
	g.meanTrace.Zero()
	g.stdTrace.Zero()
	g.criticTrace.Zero()
}
