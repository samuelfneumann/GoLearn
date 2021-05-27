// Package policy implements linear continuous-action policies
package policy

import (
	"fmt"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/gonum/stat/samplemv"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/timestep"
)

const (
	// Keys for weights map: map[string]*mat.Dense
	MeanWeights string = "mean"
	StdWeights  string = "standard deviation"
)

type Gaussian struct {
	meanWeights *mat.Dense
	stdWeights  *mat.Dense
	actionDims  int
	source      rand.Source
}

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

func (g *Gaussian) SelectAction(t timestep.TimeStep) mat.Vector {
	obs := t.Observation

	// Get the predicted mean the policy
	mean := mat.NewVecDense(g.actionDims, nil)
	mean.MulVec(g.meanWeights, obs)

	// Get the predicted variance of the policy
	stdVec := mat.NewVecDense(g.actionDims, nil)
	stdVec.MulVec(g.stdWeights, obs)
	std := mat.NewDiagDense(stdVec.Len(), stdVec.RawVector().Data)

	// Generate the Gaussian policy and sampler
	dist, _ := distmv.NewNormal(mean.RawVector().Data, std, g.source)
	sampler := samplemv.IID{Dist: dist}

	// Sample an action
	action := mat.NewDense(1, g.actionDims, nil)
	sampler.Sample(action)

	// Ensure only a single action was sampled
	underlyingMatrix := action.RawMatrix()
	if underlyingMatrix.Rows != 1 {
		panic("SelectAction: more than one action generated")
	}

	// Convert the action to a mat.Vector and return
	return mat.NewVecDense(g.actionDims, underlyingMatrix.Data)
}

func (g *Gaussian) Weights() map[string]*mat.Dense {
	weights := make(map[string]*mat.Dense)

	weights[MeanWeights] = g.meanWeights
	weights[StdWeights] = g.stdWeights

	return weights
}

func (g *Gaussian) SetWeights(weights map[string]*mat.Dense) error {
	// Set the weights for the mean
	meanWeights, ok := weights[MeanWeights]
	if !ok {
		return fmt.Errorf("SetWeights: no weights named \"%v\"",
			MeanWeights)
	}

	g.meanWeights = meanWeights

	// Set the weights for the std deviation
	stdWeights, ok := weights[StdWeights]
	if !ok {
		return fmt.Errorf("SetWeights: no weights named \"%v\"",
			StdWeights)
	}

	g.stdWeights = stdWeights

	return nil
}
