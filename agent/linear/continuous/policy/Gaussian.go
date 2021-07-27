// Package policy implements linear continuous-action policies
package policy

import (
	"fmt"
	"math"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/matutils"
)

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
	source      rand.Source
}

// NewGaussian creates a new Gaussian policy
func NewGaussian(seed uint64, env environment.Environment) *Gaussian {
	// Calculate the dimension of actions
	actionDims := env.ActionSpec().Shape.Len()

	// Calculate the number of features
	features := env.ObservationSpec().Shape.Len()

	meanWeights := mat.NewDense(actionDims, features, nil)
	stdWeights := mat.NewDense(actionDims, features, nil)

	source := rand.NewSource(seed)

	return &Gaussian{meanWeights, stdWeights, actionDims, source}
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

// SelectAction selects an action from the policy for a given timestep
func (g *Gaussian) SelectAction(t timestep.TimeStep) *mat.VecDense {
	obs := t.Observation

	mean := g.Mean(obs)
	stdVec := g.Std(obs)

	// Generate the Gaussian policy and sampler
	std := mat.NewDiagDense(stdVec.Len(), stdVec.RawVector().Data)
	dist, ok := distmv.NewNormal(mean.RawVector().Data, std, g.source)
	if !ok {
		msg := fmt.Sprintf("*Normal has non-positive-definite covariance %v",
			matutils.Format(std))
		panic(msg)
	}

	// Sample an action
	action := mat.NewDense(1, g.actionDims, dist.Rand(nil))

	// Ensure only a single action was sampled
	underlyingMatrix := action.RawMatrix()
	if underlyingMatrix.Rows != 1 {
		panic("SelectAction: more than one action generated")
	}

	// Convert the action to a mat.Vector and return
	a := mat.NewVecDense(g.actionDims, underlyingMatrix.Data)
	return a
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
