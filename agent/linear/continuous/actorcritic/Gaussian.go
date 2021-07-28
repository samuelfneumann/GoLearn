package actorcritic

import (
	"fmt"
	"math"
	"os"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/agent/linear/continuous/policy"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	ts "sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/floatutils"
	"sfneuman.com/golearn/utils/matutils/initializers/weights"
)

// ! This will be first written for 1d actions
type LinearGaussian struct {
	*policy.Gaussian

	step     ts.TimeStep
	action   *mat.VecDense
	nextStep ts.TimeStep

	seed      uint64
	stdNormal *distmv.Normal
	eval      bool

	meanWeights   *mat.VecDense
	stdWeights    *mat.VecDense
	criticWeights *mat.VecDense

	meanTrace   *mat.VecDense
	stdTrace    *mat.VecDense
	criticTrace *mat.VecDense

	actorLR  float64
	criticLR float64
	decay    float64
}

func NewLinearGaussian(env environment.Environment, c agent.Config,
	init weights.Initializer, seed uint64) (agent.Agent, error) {
	// Ensure continuous action environment is used
	actionSpec := env.ActionSpec()
	if actionSpec.Cardinality != spec.Continuous {
		return nil, fmt.Errorf("newLinearGaussian: actions must be continuous")
	}
	if actionSpec.Shape.Len() != 1 {
		return nil, fmt.Errorf("newLinearGaussian: LinearGaussian does not " +
			"yet support multi-dimensional actions")
	}
	if !c.ValidAgent(&LinearGaussian{}) {
		return nil, fmt.Errorf("newLinearGaussian: invalid agent for "+
			"configuration type %T", c)
	}
	config, ok := c.(LinearGaussianConfig)
	if !ok {
		return nil, fmt.Errorf("newLinearGaussian: invalid config for agent " +
			"LinearGaussian")
	}
	err := config.Validate()
	if err != nil {
		return nil, fmt.Errorf("newLinearGaussian: %v", err)
	}

	pol := policy.NewGaussian(seed, env)
	weights := pol.Weights()
	if r, _ := weights[policy.MeanWeightsKey].Dims(); r != 1 {
		return nil, fmt.Errorf("newLinearGaussian: multi-dimensional " +
			"actions not yet supported")
	}
	if r, _ := weights[policy.StdWeightsKey].Dims(); r != 1 {
		return nil, fmt.Errorf("newLinearGaussian: multi-dimensional " +
			"actions not yet supported")
	}
	init.Initialize(weights[policy.MeanWeightsKey])
	init.Initialize(weights[policy.StdWeightsKey])

	// Initialize the weights for the agent. Each should share the
	// same backing data as the policy.
	features := env.ObservationSpec().Shape.Len()
	actionDims := env.ActionSpec().Shape.Len()
	meanWeights := mat.NewVecDense(
		features,
		weights[policy.MeanWeightsKey].RawMatrix().Data,
	)
	stdWeights := mat.NewVecDense(
		features,
		weights[policy.StdWeightsKey].RawMatrix().Data,
	)
	criticWeightsMat := mat.NewDense(1, features, nil)
	init.Initialize(criticWeightsMat)
	criticWeights := mat.NewVecDense(
		features,
		criticWeightsMat.RawMatrix().Data,
	)

	// Create the standard normal for action selection
	means := make([]float64, actionDims)
	std := mat.NewDiagDense(actionDims, floatutils.Ones(actionDims))
	src := rand.NewSource(seed)
	stdNormal, ok := distmv.NewNormal(means, std, src)
	if !ok {
		return nil, fmt.Errorf("newLinearGaussian: could not construct " +
			"standard normal for action selection")
	}

	agent := LinearGaussian{
		Gaussian:  pol,
		seed:      seed,
		stdNormal: stdNormal,
		eval:      false,

		meanWeights:   meanWeights,
		stdWeights:    stdWeights,
		criticWeights: criticWeights,

		meanTrace:   mat.NewVecDense(features, nil),
		stdTrace:    mat.NewVecDense(features, nil),
		criticTrace: mat.NewVecDense(features, nil),

		actorLR:  config.ActorLearningRate,
		criticLR: config.CriticLearningRate,
		decay:    config.Decay,
	}

	return &agent, nil
}

func (l *LinearGaussian) SelectAction(t ts.TimeStep) *mat.VecDense {
	if l.eval {
		return l.Gaussian.Mean(t.Observation)
	}
	action := l.Gaussian.SelectAction(t)
	return action
}

func (l *LinearGaussian) TdError(t ts.Transition) float64 {
	panic("tdError: not implemented")
}

func (l *LinearGaussian) Step() {

	state := l.step.Observation
	nextState := l.nextStep.Observation

	// Calculate TD error δ
	r := l.nextStep.Reward
	ℽ := l.nextStep.Discount
	stateValue := mat.Dot(l.criticWeights, state)
	nextStateValue := mat.Dot(l.criticWeights, nextState)
	δ := r + ℽ*nextStateValue - stateValue
	l.criticTrace.AddScaledVec(state, ℽ*l.decay, l.criticTrace)

	// Update critic weights
	l.criticWeights.AddScaledVec(l.criticWeights, l.criticLR*δ, l.criticTrace)

	// Calculate mean and std gradients
	mean := l.Gaussian.Mean(state).AtVec(0)
	std := l.Gaussian.Std(state).AtVec(0)
	// fmt.Println("MEAN", mean)
	// fmt.Println("STD", std)
	action := l.action.AtVec(0)
	if std <= 0 {
		panic("step: standard deviation <= 0")
	}
	meanGradScale := (action - mean) / math.Pow(std, 2)
	meanGrad := mat.NewVecDense(state.Len(), nil)
	meanGrad.ScaleVec(meanGradScale, state)

	stdGradScale := math.Pow((action-mean)/std, 2) - 1.0
	stdGrad := mat.NewVecDense(state.Len(), nil)
	stdGrad.ScaleVec(stdGradScale, state)

	// Calculate actor traces
	l.meanTrace.AddScaledVec(meanGrad, ℽ*l.decay, l.meanTrace)
	l.stdTrace.AddScaledVec(stdGrad, ℽ*l.decay, l.stdTrace)

	// Update actor weights
	l.meanWeights.AddScaledVec(l.meanWeights, l.actorLR*δ, l.meanTrace)
	l.stdWeights.AddScaledVec(l.stdWeights, (l.actorLR*δ)/math.Pow(std, 2), l.stdTrace)
}

func (l *LinearGaussian) Observe(a mat.Vector, nextStep ts.TimeStep) {
	l.step = l.nextStep
	l.action = a.(*mat.VecDense)
	l.nextStep = nextStep
}

func (l *LinearGaussian) ObserveFirst(t ts.TimeStep) {
	if !t.First() {
		fmt.Fprintf(os.Stderr, "warning: ObserveFirst() called on %v "+
			"timestep", t.StepType)
	}
	l.step = t
	l.nextStep = t
}

func (l *LinearGaussian) Train() {
	l.eval = false
}

func (l *LinearGaussian) Eval() {
	l.eval = true
}

func (l *LinearGaussian) EndEpisode() {
	l.criticTrace.Zero()
	l.stdTrace.Zero()
	l.meanTrace.Zero()
}
