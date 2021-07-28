// Package policy implements linear continuous-action policies
package policy

import (
	"fmt"
	"math"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/floatutils"
)

// StdOffset is added to the standard deviation for numerical stability
const StdOffset float64 = 1e-3

const (
	// Keys for weights map: map[string]*mat.Dense
	MeanWeightsKey   string = "mean"
	StdWeightsKey    string = "standard deviation"
	CriticWeightsKey string = "critic"
)

// Guassian implements a multi-dimensional linear Gaussian policy.
// The policy uses linear function approximation to compute the mean
// and standard deviation of the policy.
type Gaussian struct {
	meanWeights *mat.Dense
	stdWeights  *mat.Dense
	actionDims  int

	eval bool

	stdNormal *distmv.Normal
}

// NewGaussian creates a new Gaussian policy
func NewGaussian(seed uint64, env environment.Environment) agent.Policy {
	// Calculate the dimension of actions
	actionDims := env.ActionSpec().Shape.Len()

	// Calculate the number of features
	features := env.ObservationSpec().Shape.Len()

	meanWeights := mat.NewDense(actionDims, features, nil)
	stdWeights := mat.NewDense(actionDims, features, nil)

	// Create the standard normal for action selection
	means := make([]float64, actionDims)
	std := mat.NewDiagDense(actionDims, floatutils.Ones(actionDims))
	src := rand.NewSource(seed)
	stdNormal, ok := distmv.NewNormal(means, std, src)
	if !ok {
		panic("newLinearGaussian: could not construct standard normal " +
			"for action selection")
	}

	return &Gaussian{meanWeights, stdWeights, actionDims, false, stdNormal}
}

// Std gets the standard deviation of the policy given some state
// observation obs
func (g *Gaussian) Std(obs mat.Vector) *mat.VecDense {
	stdVec := mat.NewVecDense(g.actionDims, nil)
	stdVec.MulVec(g.stdWeights, obs)
	for i := 0; i < stdVec.Len(); i++ {
		std := math.Exp(stdVec.AtVec(i))
		stdVec.SetVec(i, std+StdOffset)
	}
	return stdVec
}

// Mean gets the mean of the policy given some state observation obs
func (g *Gaussian) Mean(obs mat.Vector) *mat.VecDense {
	mean := mat.NewVecDense(g.actionDims, nil)
	mean.MulVec(g.meanWeights, obs)
	return mean
}

// Eval sets the policy to evaluation mode
func (g *Gaussian) Eval() {
	g.eval = true
}

// Train sets the policy to training mode
func (g *Gaussian) Train() {
	g.eval = false
}

// IsEval returns whether or not the policy is in evaluation mode
func (g *Gaussian) IsEval() bool {
	return g.eval
}

// SelectAction selects an action from the policy at a given timestep
func (g *Gaussian) SelectAction(t timestep.TimeStep) *mat.VecDense {
	obs := t.Observation

	mean := g.Mean(obs)

	// If in evaluation mode, return the mean action only
	if g.IsEval() {
		return mean
	}

	stdVec := g.Std(obs)
	eps := mat.NewVecDense(mean.Len(), g.stdNormal.Rand(nil))

	stdVec.MulElemVec(stdVec, eps)
	mean.AddVec(mean, stdVec)
	return mean
}

// Weights gets and returns the weights of the learner
func (g *Gaussian) Weights() map[string]*mat.Dense {
	weights := make(map[string]*mat.Dense)

	weights[MeanWeightsKey] = g.meanWeights
	weights[StdWeightsKey] = g.stdWeights

	return weights
}

// SetWeights sets the weight pointers to point to a new set of weights.
func (g *Gaussian) SetWeights(weights map[string]*mat.Dense) error {
	// Set the weights for the mean
	meanWeights, ok := weights[MeanWeightsKey]
	if !ok {
		return fmt.Errorf("SetWeights: no weights named \"%v\"",
			MeanWeightsKey)
	}

	g.meanWeights = meanWeights

	// Set the weights for the std deviation
	stdWeights, ok := weights[StdWeightsKey]
	if !ok {
		return fmt.Errorf("SetWeights: no weights named \"%v\"",
			StdWeightsKey)
	}

	g.stdWeights = stdWeights

	return nil
}
