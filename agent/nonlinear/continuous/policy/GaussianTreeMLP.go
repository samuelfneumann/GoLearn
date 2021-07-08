package policy

import (
	"fmt"
	"log"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	"gonum.org/v1/gonum/stat/samplemv"
	G "gorgonia.org/gorgonia"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/network"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
)

type GaussianTreeMLP struct {
	network.NeuralNet

	mean, std []float64
	source    rand.Source
	seed      uint64

	vm G.VM // VM for action selection
}

func NewGaussianTreeMLP(env environment.Environment,
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

	net, err := network.NewTreeMLP(features, 1, actionDims, g,
		rootHiddenSizes, rootBiases, rootActivations, leafHiddenSizes,
		leafBiases, leafActivations, init)
	if err != nil {
		return &GaussianTreeMLP{}, fmt.Errorf("newGaussianTreeMLP: could "+
			"not create policy network: %v", err)
	}

	// If the policy predicts actions from batches of data, then there
	// is no need for a VM to select actions at each timestep
	var vm G.VM
	vm = G.NewTapeMachine(net.Graph())

	source := rand.NewSource(seed)

	policy := GaussianTreeMLP{
		NeuralNet: net,
		mean:      nil,
		std:       nil,
		source:    source,
		seed:      seed,
		vm:        vm,
	}

	return &policy, nil
}

func (g *GaussianTreeMLP) Network() network.NeuralNet {
	return g.NeuralNet
}

func (g *GaussianTreeMLP) cloneWithBatch(batch int) (agent.NNPolicy,
	error) {
	net, err := g.Network().CloneWithBatch(batch)
	if err != nil {
		return &GaussianTreeMLP{}, fmt.Errorf("clonePolicyWithBatch: could "+
			"not clone policy neural net: %v", err)
	}

	vm := G.NewTapeMachine(net.Graph())

	source := rand.NewSource(g.seed)
	newPolicy := GaussianTreeMLP{
		NeuralNet: net,
		mean:      g.mean,
		std:       g.std,
		source:    source,
		seed:      g.seed,
		vm:        vm,
	}

	return &newPolicy, nil
}

func (g *GaussianTreeMLP) Clone() (agent.NNPolicy, error) {
	return g.cloneWithBatch(g.BatchSize())
}

// VM should be run before running logprobability
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
func (g *GaussianTreeMLP) SelectAction(t timestep.TimeStep) *mat.VecDense {
	if g.BatchSize() != 1 {
		log.Fatal("selectAction: cannot select an action from batch policy, " +
			"can only learn weights using a batch policy")
	}

	g.SetInput(t.Observation.RawVector().Data)
	g.vm.RunAll()

	g.std = g.Output()[0].Data().([]float64)
	g.mean = g.Output()[1].Data().([]float64)
	g.vm.Reset()

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
