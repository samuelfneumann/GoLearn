package policy

import (
	"fmt"
	"log"
	"math"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/network"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
)

type Slice struct {
	start, end, step int
}

func (s Slice) Start() int {
	return s.start
}

func (s Slice) End() int {
	return s.end
}

func (s Slice) Step() int {
	return s.step
}

func NewSlice(start, stop, step int) Slice {
	return Slice{start, stop, step}
}

type GaussianTreeMLP struct {
	network.NeuralNet

	mean, std G.Value

	logProb    *G.Node
	actions    *G.Node
	actionsVal G.Value

	source rand.Source
	seed   uint64

	vm G.VM // VM for action selection
}

// SAC e.g. will have two GaussianTreeMLPs, one with and one without
// batches. With batchs --> learning weights. Without batches --> selecting
// actions
func NewGaussianTreeMLP(env environment.Environment, batchForLogProb int,
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
			"policy requires 2 leaf networks only")
	}

	features := env.ObservationSpec().Shape.Len()
	actionDims := env.ActionSpec().Shape.Len()

	net, err := network.NewTreeMLP(features, batchForLogProb, actionDims, g,
		rootHiddenSizes, rootBiases, rootActivations, leafHiddenSizes,
		leafBiases, leafActivations, init)
	if err != nil {
		return &GaussianTreeMLP{}, fmt.Errorf("newGaussianTreeMLP: could "+
			"not create policy network: %v", err)
	}

	// Exponentiate the standard deviation
	logStd := net.Prediction()[0]
	stdNode := G.Must(G.Exp(logStd))
	meanNode := net.Prediction()[1]

	// Reparameterization trick
	actionPerturb := G.GaussianRandomNode(net.Graph(), tensor.Float64,
		0, 1, batchForLogProb, net.Outputs()[0])
	actionStd := G.Must(G.HadamardProd(stdNode, actionPerturb))
	actions := G.Must(G.Add(meanNode, actionStd))
	fmt.Println(meanNode.Shape(), stdNode.Shape(), actionPerturb.Shape(), actionStd.Shape(), actions.Shape())

	// Calculate log prob
	logProbNode, err := logProb(meanNode, stdNode, actions)
	if err != nil {
		return nil, fmt.Errorf("newGaussianTreeMLP: could not calculate "+
			"log probabiltiy: %v", err)
	}

	source := rand.NewSource(seed)

	policy := GaussianTreeMLP{
		NeuralNet: net,
		logProb:   logProbNode,
		actions:   actions,
		source:    source,
		seed:      seed,
	}
	G.Read(policy.actions, &policy.actionsVal)
	vm := G.NewTapeMachine(net.Graph())
	policy.vm = vm

	return &policy, nil
}

func logProb(mean, std, actions *G.Node) (*G.Node, error) {
	graph := mean.Graph()
	if graph != std.Graph() || graph != actions.Graph() {
		return nil, fmt.Errorf("logProb: mean, std, and actions should " +
			"all have the same graph")
	}

	dims := float64(mean.Shape()[0])
	multiplier := G.NewConstant(math.Pow(math.Pi*2, -dims/2))
	negativeHalf := G.NewConstant(-0.5)

	fmt.Println("std shape:", std.Shape())
	var det *G.Node
	if std.Shape()[0] != 1 {
		det = G.Must(G.Slice(std, nil, NewSlice(0, 1, 1)))
		for i := 1; i < std.Shape()[1]; i++ {
			s := G.Must(G.Slice(std, nil, NewSlice(i, i+1, 1)))
			det = G.Must(G.HadamardProd(det, s))
		}
	} else {
		det = std
	}
	invDet := G.Must(G.Inverse(det))

	det = G.Must(G.Pow(det, negativeHalf))
	det = G.Must(G.Mul(multiplier, det))

	diff := G.Must(G.Sub(actions, mean))
	exponent := G.Must(G.HadamardProd(diff, invDet))
	exponent = G.Must(G.HadamardProd(exponent, diff))
	exponent = G.Must(G.Sum(exponent, 1))
	exponent = G.Must(G.Mul(exponent, negativeHalf))

	prob := G.Must(G.Exp(exponent))
	prob = G.Must(G.HadamardProd(multiplier, prob))

	logProb := G.Must(G.Log(prob))

	fmt.Println("SHAPES:", det.Shape(), exponent.Shape(), logProb.Shape())

	return logProb, nil
}

// Mean returns the mean of the Gaussian policy
func (g *GaussianTreeMLP) Mean() []float64 {
	return g.mean.Data().([]float64)
}

// Std returns the standard deviation of the Gaussian policy
func (g *GaussianTreeMLP) Std() []float64 {
	return g.std.Data().([]float64)
}

// Network returns the NeuralNet used by the policy for function
// approximation
func (g *GaussianTreeMLP) Network() network.NeuralNet {
	return g.NeuralNet
}

// cloneWithBatch clones the policy with a new input batch size
func (g *GaussianTreeMLP) CloneWithBatch(batch int) (agent.NNPolicy,
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

// Clone clones the policy
func (g *GaussianTreeMLP) Clone() (agent.NNPolicy, error) {
	return g.CloneWithBatch(g.BatchSize())
}

func (g *GaussianTreeMLP) LogProb() *G.Node {
	return g.logProb
}

// SelectAction selects and returns a new action given a TimeStep
func (g *GaussianTreeMLP) SelectAction(t timestep.TimeStep) *mat.VecDense {
	if g.BatchSize() != 1 {
		log.Fatal("selectAction: cannot select an action from batch policy " +
			"- can only learn weights using a batch policy")
	}

	g.Network().SetInput(t.Observation.RawVector().Data)
	g.vm.RunAll()
	defer g.vm.Reset()

	fmt.Println("DATA:", g.Network().Prediction()[0].Value(), g.std)
	fmt.Println(g.actions.Value(), g.actionsVal)

	// ! This only works if batchsize == 1
	return mat.NewVecDense(1, g.actionsVal.Data().([]float64))
}
