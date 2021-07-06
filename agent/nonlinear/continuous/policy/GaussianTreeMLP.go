package policy

import (
	"fmt"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/gonum/stat/samplemv"
	G "gorgonia.org/gorgonia"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/network"
	"sfneuman.com/golearn/spec"
)

type GaussianTreeMLP struct {
	network.NeuralNet

	mean, std []float64
	source    rand.Source
	seed      uint64
}

func NewGaussianTreeMLP(env environment.Environment, batch int,
	g *G.ExprGraph, rootHiddenSizes []int, rootBiases []bool,
	rootActivations []*network.Activation, leafHiddenSizes [][]int,
	leafBiases [][]bool, leafActivations [][]*network.Activation,
	init G.InitWFn, seed uint64) (agent.PolicyLogProber, error) {

	// Error checking
	if env.ActionSpec().Cardinality == spec.Discrete {
		err := fmt.Errorf("newGaussianTreeMLP: gaussian policy cannot be " +
			"used with discrete actions")
		return &GaussianTreeMLP{}, err
	}

	if len(leafHiddenSizes) != 2 {
		return &GaussianTreeMLP{}, fmt.Errorf("newGaussianTreeMLP: gaussian " +
			"policy requires 2 leaf networks alone")
	}

	features := env.ObservationSpec().Shape.Len()
	actionDims := env.ActionSpec().Shape.Len()

	net, err := network.NewTreeMLP(features, batch, actionDims, g,
		rootHiddenSizes, rootBiases, rootActivations, leafHiddenSizes,
		leafBiases, leafActivations, init)
	if err != nil {
		return &GaussianTreeMLP{}, fmt.Errorf("newGaussianTreeMLP: could "+
			"not create policy network: %v", err)
	}

	source := rand.NewSource(seed)

	policy := GaussianTreeMLP{
		NeuralNet: net,
		mean:      nil,
		std:       nil,
		source:    source,
		seed:      seed,
	}

	return &policy, nil
}

func (g *GaussianTreeMLP) Network() network.NeuralNet {
	return g.NeuralNet
}

func (g *GaussianTreeMLP) ClonePolicyWithBatch(batch int) (agent.NNPolicy,
	error) {
	net, err := g.Network().CloneWithBatch(batch)
	if err != nil {
		return &GaussianTreeMLP{}, fmt.Errorf("clonePolicyWithBatch: could "+
			"not clone policy neural net: %v", err)
	}

	source := rand.NewSource(g.seed)
	newPolicy := GaussianTreeMLP{
		NeuralNet: net,
		mean:      g.mean,
		std:       g.std,
		source:    source,
		seed:      g.seed,
	}

	return &newPolicy, nil
}

func (g *GaussianTreeMLP) ClonePolicy() (agent.NNPolicy, error) {
	return g.ClonePolicyWithBatch(g.BatchSize())
}

// VM should be run before runnint logprobability
func (g *GaussianTreeMLP) LogProbability(action []float64) (float64, error) {
	g.std = g.Output()[0].Data().([]float64)
	g.mean = g.Output()[1].Data().([]float64)

	std := mat.NewDiagDense(len(g.std), g.std)
	dist, ok := distmv.NewNormal(g.mean, std, g.source)
	if !ok {
		return -1.0, fmt.Errorf("logProbability: standard deviation of " +
			"normal is not positive definite")
	}

	return dist.LogProb(action), nil
}

func (g *GaussianTreeMLP) numActions() int {
	return g.Outputs()
}

// VM should be run before selecting action
func (g *GaussianTreeMLP) SelectAction() *mat.VecDense {
	g.std = g.Output()[0].Data().([]float64)
	g.mean = g.Output()[1].Data().([]float64)

	std := mat.NewDiagDense(len(g.std), g.std)
	dist, ok := distmv.NewNormal(g.mean, std, g.source)
	if !ok {
		panic("non-positive definite standard deviation")
	}

	sampler := samplemv.IID{Dist: dist}

	action := mat.NewDense(g.numActions(), 1, nil)
	sampler.Sample(action)

	return mat.NewVecDense(g.numActions(), action.RawMatrix().Data)
}
