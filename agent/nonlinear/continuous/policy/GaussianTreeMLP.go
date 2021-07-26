package policy

import (
	"fmt"
	"math"
	"os"

	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/network"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/floatutils"
	"sfneuman.com/golearn/utils/tensorutils"
)

// For stability, the standard deviation of the Gaussian distribution
// should be offset from 0.
const stdOffset float64 = 1e-3

// GaussianTreeMLP implements a Gaussian policy parameterized by a
// tree MLP. The MLP has a single root network. The root network breaks
// off into two leaf networks. One predicts the mean, and the other
// the log standard deviation. See the network.TreeMLP struct for
// more details.
//
// Given a nework prediction of the mean μ and standard deviation σ of
// the Gaussian policy, actions are selected by sampling from the
// standard normal ɛ ~ N(0, 1) and computing action := μ + σ * ɛ
// similar to the reparameterization trick.
//
// Given a number of continuous actions in a number of states, the
// GaussianTreeMLP can calculate the log probability of selecting
// each of these actions in each corresponding state. This is useful
// for constructing policy gradients in a similar way to Vanilla
// Policy Gradient or REINFORCE. The GaussianTreeMLp cannot yet produce
// the log probability of the actions actually selected by the policy
// due to limitations of Gorgonia.
type GaussianTreeMLP struct {
	vm  G.VM
	net network.NeuralNet

	actions    *G.Node
	logPdfNode *G.Node
	logPdfVal  G.Value

	normal          distmv.Rander
	actionDims      int
	batchForLogProb int

	meanVal   G.Value
	stddevVal G.Value
}

// NewGaussianTreeMLP returns a new GaussianTreeMLP policy. The
// Gaussian policy will select actions from the argument environment.
// The neural network parameterization of the Gaussian policy is
// defined by rootHiddenSizes, rootBiases, rootActivations,
// leafHiddenSizes, leafBiases, and leafActivations. See the
// network.TreeMLP struct for details on what each of these parameters
// defines.
//
// The policy can be a batch policy when batchForLobProb > 1. In such
// a case, the log probability of actions can be computed for a batch
// of actions, but actions cannot be selected on each timestep with
// SelectAction(). Only when batchForLogProb = 1 can actions be
// selected at each timestep. When a policy is created as a batch
// policy (batchForLogProb > 1), it is assumed that the weights of
// the policy will be learned instead of using the policy for action
// selection.
//
// The init parameter determines the weight initialization scheme for
// the neural net and the seed parameter determines the seed of the
// policy's action sampler.
func NewGaussianTreeMLP(env environment.Environment, batchForLogProb int,
	g *G.ExprGraph, rootHiddenSizes []int, rootBiases []bool,
	rootActivations []*network.Activation, leafHiddenSizes [][]int,
	leafBiases [][]bool, leafActivations [][]*network.Activation,
	init G.InitWFn, seed uint64) (agent.LogPdfOfer, error) {

	if env.ActionSpec().Cardinality != spec.Continuous {
		panic("newGaussianTreeMLP: actions should be continuous")
	}
	if len(leafHiddenSizes) != 2 {
		panic("newGaussianTreeMLP: gaussian policy requires 2 leaf " +
			"networks only")
	}

	features := env.ObservationSpec().Shape.Len()
	actionDims := env.ActionSpec().Shape.Len()

	net, err := network.NewTreeMLP(
		features,
		batchForLogProb,
		actionDims,
		G.NewGraph(),
		rootHiddenSizes,
		rootBiases,
		rootActivations,
		leafHiddenSizes,
		leafBiases,
		leafActivations,
		init,
	)
	if err != nil {
		panic(err)
	}

	// Calculate the standard deviation and offset it for numerical
	// stability
	mean := net.Prediction()[0]
	offset := G.NewConstant(stdOffset)
	logStd := net.Prediction()[1]
	std := G.Must(G.Exp(logStd))
	std = G.Must(G.Add(offset, std))

	// Calculate log probability of input actions
	actions := G.NewMatrix(
		net.Graph(),
		tensor.Float64,
		G.WithName("InputActions"),
		G.WithShape(batchForLogProb, actionDims),
		G.WithInit(G.Zeroes()),
	)
	logPdfNode := logPdf(mean, std, actions)

	// Create standard normal for action selection
	means := make([]float64, actionDims)
	stds := mat.NewDiagDense(actionDims, floatutils.Ones(actionDims))
	source := rand.NewSource(seed)
	normal, ok := distmv.NewNormal(means, stds, source)
	if !ok {
		panic("newGaussianTreeMLP: could not create standard normal for " +
			"action selection")
	}

	pol := &GaussianTreeMLP{
		net: net,

		actions:    actions,
		logPdfNode: logPdfNode,

		normal:          normal,
		actionDims:      actionDims,
		batchForLogProb: batchForLogProb,
	}

	// Record values of Gorgonia nodes
	G.Read(pol.logPdfNode, &pol.logPdfVal)
	G.Read(mean, &pol.meanVal)
	G.Read(std, &pol.stddevVal)

	// Policy can select actions at each timestep only if using a batch
	// size of 1.
	if net.BatchSize() == 1 {
		pol.vm = G.NewTapeMachine(net.Graph())
	}

	return pol, nil
}

// logPdf adds nodes to the computaitonal graph of mean/std/actions for
// computing the log probability of actions given nodes mean and std
// which hold the mean and standard deviation of the policy
// respectively.
func logPdf(mean, std, actions *G.Node) *G.Node {
	graph := mean.Graph()
	if graph != std.Graph() || graph != actions.Graph() {
		panic("logPdf: all nodes must share the same graph")
	}

	negativeHalf := G.NewConstant(-0.5)

	if std.Shape()[1] != 1 {
		fmt.Fprintf(os.Stderr, "logProb: warning - not tested for "+
			"multi-dimensional actions")
		// Multi-dimensional actions
		// Calculate det(σ). Since σ is a diagonal matrix stored as a vector,
		// the determinant == prod(diagonal of σ) = prod(σ)
		dims := float64(mean.Shape()[1])
		multiplier := G.NewConstant(math.Pow(math.Pi*2, -dims/2), G.WithName("multiplier"))
		det := G.Must(G.Slice(std, nil, tensorutils.NewSlice(0, 1, 1)))
		for i := 1; i < std.Shape()[1]; i++ {
			s := G.Must(G.Slice(std, nil, tensorutils.NewSlice(i, i+1, 1)))
			det = G.Must(G.HadamardProd(det, s))
		}
		invDet := G.Must(G.Inverse(det))

		// Calculate (2*π)^(-k/2) * det(σ)
		det = G.Must(G.Pow(det, negativeHalf))
		multiplier = G.Must(G.Mul(multiplier, det))

		// Calculate (-1/2) * (A - μ)^T σ^(-1) (A - μ)
		// Since everything is stored as a vector, this boils down to a
		// bunch of Hadamard products, sums, and differences.
		diff := G.Must(G.Sub(actions, mean))
		exponent := G.Must(G.HadamardProd(diff, invDet))
		exponent = G.Must(G.HadamardProd(exponent, diff))
		exponent = G.Must(G.Sum(exponent, 1))
		exponent = G.Must(G.Mul(exponent, negativeHalf))

		// Calculate the probability
		prob := G.Must(G.Exp(exponent))
		prob = G.Must(G.HadamardProd(multiplier, prob))

		logProb := G.Must(G.Log(prob))

		return logProb
	} else {
		two := G.NewConstant(2.0)
		exponent := G.Must(G.Sub(actions, mean))
		exponent = G.Must(G.HadamardDiv(exponent, std))
		exponent = G.Must(G.Pow(exponent, two))
		exponent = G.Must(G.HadamardProd(negativeHalf, exponent))

		term2 := G.Must(G.Log(std))
		// term2 := G.Must(G.HadamardProd(two, logStd))
		term3 := G.NewConstant(math.Log(math.Pow(2*math.Pi, 0.5)))

		terms := G.Must(G.Add(term2, term3))
		logProb := G.Must(G.Sub(exponent, terms))
		logProb = G.Must(G.Ravel(logProb))

		return logProb
	}
}

// LogPdfOf sets the state and action inputs of the policy's
// computational graph to the argument state and actions (s and a
// respectively) so that when a VM of the policy is run, the log
// probabliity of actions a taken in states s will be computed and
// stored in the policy's associate log PDF node, which is returned.
//
// The reason this function does not return the log PDF of actions is
// because this would require running the policy's VM, which does
// not contain any loss function. The log PDF of actions is generally
// needed in loss functions, and a separate, external VM will be needed
// to calculate the loss of the policy using the log PDF and update
// the weights accordingly.
func (g *GaussianTreeMLP) LogPdfOf(s, a []float64) (*G.Node, error) {
	fmt.Println(s[len(s)-10:])
	fmt.Println(a[len(a)-10:])
	if err := g.Network().SetInput(s); err != nil {
		panic(err)
	}

	actionsTensor := tensor.NewDense(tensor.Float64,
		[]int{g.batchForLogProb, g.actionDims},
		tensor.WithBacking(a),
	)
	err := G.Let(g.actions, actionsTensor)
	if err != nil {
		return nil, fmt.Errorf("logPdfOf: could not set actions: %v", err)
	}

	return g.LogPdfNode(), nil
}

// SelectAction selects and returns an action at the argument timestep
// t.
func (c *GaussianTreeMLP) SelectAction(t timestep.TimeStep) *mat.VecDense {
	if size := c.Network().BatchSize(); size != 1 {
		panic(fmt.Sprintf("selectAction: action selection can only be done "+
			"with a policy with batch size 1 \n\twant(1) \n\thave(%v)", size))
	}

	obs := t.Observation.RawVector().Data
	if err := c.Network().SetInput(obs); err != nil {
		panic(fmt.Sprintf("selectAction: cannot set input: %v", err))
	}

	if err := c.vm.RunAll(); err != nil {
		panic(fmt.Sprintf("selectAction: could not run policy VM: %v", err))
	}
	defer c.vm.Reset()

	mean := mat.NewVecDense(c.actionDims, c.meanVal.Data().([]float64))
	stddev := mat.NewVecDense(c.actionDims, c.stddevVal.Data().([]float64))
	eps := mat.NewVecDense(c.actionDims, c.normal.Rand(nil))

	stddev.MulElemVec(stddev, eps)
	mean.AddVec(mean, stddev)

	return mean
}

// LogPdfNode returns the node that will hold the log probability
// of actions when the comptuational graph is run.
func (c *GaussianTreeMLP) LogPdfNode() *G.Node {
	return c.logPdfNode
}

// LogPdfVal returns the value of the node returned by LogPdfNode()
func (c *GaussianTreeMLP) LogPdfVal() G.Value {
	return c.logPdfVal
}

// Clone clones a GaussianTreeMLP
func (c *GaussianTreeMLP) Clone() (agent.NNPolicy, error) {
	panic("not implemented")
}

// CloneWithBatch clones a GaussianTreeMLP with a new batch size
func (c *GaussianTreeMLP) CloneWithBatch(batch int) (agent.NNPolicy, error) {
	panic("not implemented")
}

// Network returns the network of the GaussianTreeMLP
func (g *GaussianTreeMLP) Network() network.NeuralNet {
	return g.net
}
