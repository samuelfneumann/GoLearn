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

// LinearGaussian implements the Linear-Gaussian Actor-Critic algorithm:
//
// https://hal.inria.fr/hal-00764281/PDF/DegrisACC2012.pdf
//
// This algorithm uses linear function approximation to learn both
// a linear state value function critic and a Gaussian policy actor.
// The policy itself may select n-dimensional actions. The algorithm
// uses  eligibility traces for both actor and critc grdients.
//
// See the paper above for more details.
type LinearGaussian struct {
	*policy.Gaussian

	step     ts.TimeStep
	action   *mat.VecDense
	nextStep ts.TimeStep

	seed      uint64
	stdNormal *distmv.Normal
	eval      bool

	// Weights for linear function approximation
	meanWeights   *mat.Dense
	stdWeights    *mat.Dense
	criticWeights *mat.VecDense

	// Eligibility traces
	meanTrace   *mat.Dense
	stdTrace    *mat.Dense
	criticTrace *mat.VecDense

	actorLR      float64
	criticLR     float64
	decay        float64
	scaleActorLR bool
	features     int
	actionDims   int
}

// NewLinearGaussian returns a new LinearGaussian
func NewLinearGaussian(env environment.Environment, c agent.Config,
	init weights.Initializer, seed uint64) (agent.Agent, error) {
	// Error checking
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

	// Create the Gaussian policy
	pol := policy.NewGaussian(seed, env)
	gaussianPolicy := pol.(*policy.Gaussian)
	weights := gaussianPolicy.Weights()
	if r, _ := weights[policy.MeanWeightsKey].Dims(); r != 1 {
		return nil, fmt.Errorf("newLinearGaussian: multi-dimensional " +
			"actions not yet supported")
	}
	if r, _ := weights[policy.StdWeightsKey].Dims(); r != 1 {
		return nil, fmt.Errorf("newLinearGaussian: multi-dimensional " +
			"actions not yet supported")
	}

	// Store features and actions dimensions
	features := env.ObservationSpec().Shape.Len()
	actionDims := env.ActionSpec().Shape.Len()

	// Initialize the weights for the agent
	meanWeights := gaussianPolicy.Weights()[policy.MeanWeightsKey]
	stdWeights := gaussianPolicy.Weights()[policy.StdWeightsKey]
	criticWeightsMat := mat.NewDense(1, features, nil)
	init.Initialize(meanWeights)
	init.Initialize(stdWeights)
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

	rows, cols := meanWeights.Dims()
	agent := LinearGaussian{
		Gaussian:  gaussianPolicy,
		seed:      seed,
		stdNormal: stdNormal,
		eval:      false,

		meanWeights:   meanWeights,
		stdWeights:    stdWeights,
		criticWeights: criticWeights,

		meanTrace:   mat.NewDense(rows, cols, nil),
		stdTrace:    mat.NewDense(rows, cols, nil),
		criticTrace: mat.NewVecDense(features, nil),

		actorLR:      config.ActorLearningRate,
		criticLR:     config.CriticLearningRate,
		decay:        config.Decay,
		scaleActorLR: config.ScaleActorLR,
		features:     features,
		actionDims:   actionDims,
	}

	return &agent, nil
}

// TdError computes the TD error of the algorithm at a given transition
func (l *LinearGaussian) TdError(t ts.Transition) float64 {
	state := t.State
	nextState := t.NextState

	r := l.nextStep.Reward
	ℽ := l.nextStep.Discount
	stateValue := mat.Dot(l.criticWeights, state)
	nextStateValue := mat.Dot(l.criticWeights, nextState)

	return r + ℽ*nextStateValue - stateValue
}

// Step updates the algorithm's weights
func (l *LinearGaussian) Step() {
	// If in evaluation mode, do not step
	if l.IsEval() {
		return
	}

	state := l.step.Observation
	nextState := l.nextStep.Observation

	// Calculate TD error δ
	r := l.nextStep.Reward
	ℽ := l.nextStep.Discount
	stateValue := mat.Dot(l.criticWeights, state)
	nextStateValue := mat.Dot(l.criticWeights, nextState)
	δ := r + ℽ*nextStateValue - stateValue

	// Update the critic trace
	l.criticTrace.AddScaledVec(state, ℽ*l.decay, l.criticTrace)

	// Update critic weights
	l.criticWeights.AddScaledVec(l.criticWeights, l.criticLR*δ, l.criticTrace)

	// Variables needed for gradient computation
	mean := l.Gaussian.Mean(state)
	std := l.Gaussian.Std(state)
	action := l.action
	row, col := l.meanWeights.Dims()

	// Compute the gradient of the mean
	meanGradScale := mat.NewVecDense(l.actionDims, nil)
	meanGradScale.SubVec(action, mean)
	meanGradDiv := mat.NewVecDense(l.actionDims, nil)
	meanGradDiv.MulElemVec(std, std)
	meanGradScale.DivElemVec(meanGradScale, meanGradDiv)
	meanGrad := mat.NewDense(row, col, nil)
	meanGrad.Outer(1.0, meanGradScale, state)

	// Compute the gradient of the standard deviation
	stdGradScale := mat.NewVecDense(l.actionDims, nil)
	stdGradScale.SubVec(action, mean)
	stdGradScale.MulElemVec(stdGradScale, stdGradScale)
	stdGradDiv := mat.NewVecDense(l.actionDims, nil)
	stdGradDiv.MulElemVec(std, std)
	stdGradScale.DivElemVec(stdGradScale, stdGradDiv)
	ones := mat.NewVecDense(l.actionDims, floatutils.Ones(l.actionDims))
	stdGradScale.SubVec(stdGradScale, ones)
	stdGrad := mat.NewDense(row, col, nil)
	stdGrad.Outer(1.0, stdGradScale, state)

	// Calculate and update the actor traces
	addMeanTrace := mat.NewDense(row, col, nil)
	addMeanTrace.Scale(ℽ*l.decay, l.meanTrace)
	l.meanTrace.Add(meanGrad, addMeanTrace)

	addStdTrace := mat.NewDense(row, col, nil)
	addStdTrace.Scale(ℽ*l.decay, l.stdTrace)
	l.stdTrace.Add(stdGrad, addStdTrace)

	// Update actor weights
	actorLR := l.actorLR
	if l.scaleActorLR && std.Len() == 1 {
		actorLR *= math.Pow(std.AtVec(0), 2)
	}
	addMean := mat.NewDense(row, col, nil)
	addMean.Scale(actorLR*δ, l.meanTrace)
	l.meanWeights.Add(l.meanWeights, addMean)

	addStd := mat.NewDense(row, col, nil)
	addStd.Scale(actorLR*δ, l.stdTrace)
	l.stdWeights.Add(l.stdWeights, addStd)
}

// Observe records the previously selected action and the timestep
// that it led to
func (l *LinearGaussian) Observe(a mat.Vector, nextStep ts.TimeStep) {
	l.step = l.nextStep
	l.action = a.(*mat.VecDense)
	l.nextStep = nextStep
}

// ObserveFirst observes the first timestep in an episode
func (l *LinearGaussian) ObserveFirst(t ts.TimeStep) {
	if !t.First() {
		fmt.Fprintf(os.Stderr, "warning: ObserveFirst() called on %v "+
			"timestep", t.StepType)
	}
	l.step = t
	l.nextStep = t
}

// EndEpisode adjusts variables after an episode has completed
func (l *LinearGaussian) EndEpisode() {
	l.criticTrace.Zero()
	l.stdTrace.Zero()
	l.meanTrace.Zero()
}
