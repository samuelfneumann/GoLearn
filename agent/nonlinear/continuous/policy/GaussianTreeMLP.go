package policy

import (
	"fmt"

	"github.com/samuelfneumann/golearn/agent"
	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/network"
	"github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/golearn/utils/floatutils"
	"github.com/samuelfneumann/golearn/utils/op"
	"golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// For stability, the standard deviation of the Gaussian distribution
// should be offset from 0.
const stdOffset float64 = 1e-3

// GaussianTreeMLP implements a Gaussian policy parameterized by a
// tree MLP. The MLP has a single root network. The root network breaks
// off into two leaf networks. One predicts the mean, and the other
// the log standard deviation. See the network.TreeMLP struct for
// more details. The network's prediction of the mean is passed through
// a hyperbolic tangent layer and then multiplied by the upper bound
// of actions in the environment to ensure that the mean of the
// Gaussian policy is always within the action bounds.
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

	eval bool
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

	if env.ActionSpec().Cardinality != environment.Continuous {
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

	// Scale the mean to be within the action bounds
	mean := net.Prediction()[0]

	mean = G.Must(G.Tanh(mean))
	actionLen := env.ActionSpec().UpperBound.Len()
	shape := []int{
		batchForLogProb,
		env.ActionSpec().UpperBound.Len(),
	}
	upperBound := make([]float64, actionLen*batchForLogProb)
	for i := 0; i < len(upperBound); i += actionLen {
		copy(
			upperBound[i*actionLen:(i+1)*actionLen],
			env.ActionSpec().UpperBound.(*mat.VecDense).RawVector().Data,
		)
	}

	actionScaleTensor := tensor.NewDense(
		tensor.Float64,
		shape,
		tensor.WithBacking(upperBound),
	)
	actionScale := G.NewConstant(
		actionScaleTensor,
		G.WithName("ActionScale"),
	)
	mean = G.Must(G.HadamardProd(mean, actionScale))

	// Calculate the standard deviation and offset it for numerical
	// stability
	offset := G.NewConstant(stdOffset)
	logStd := net.Prediction()[1]
	std := G.Must(G.Exp(logStd))
	std = G.Must(G.Add(offset, std))

	// Calculate log probability of input actions
	var actions *G.Node
	var logPdfNode *G.Node
	if batchForLogProb > 1 {
		actions = G.NewMatrix(
			net.Graph(),
			tensor.Float64,
			G.WithName("InputActions"),
			G.WithShape(batchForLogProb, actionDims),
			G.WithInit(G.Zeroes()),
		)
		logPdfNode = op.GaussianLogPdf(mean, std, actions)
	}

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
		eval:            false,
	}

	// Record values of Gorgonia nodes
	if batchForLogProb > 1 {
		G.Read(pol.logPdfNode, &pol.logPdfVal)
	}
	G.Read(mean, &pol.meanVal)
	G.Read(std, &pol.stddevVal)

	// Policy can select actions at each timestep only if using a batch
	// size of 1.
	if net.BatchSize() == 1 {
		pol.vm = G.NewTapeMachine(net.Graph())
	}

	return pol, nil
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
func (g *GaussianTreeMLP) SelectAction(t timestep.TimeStep) *mat.VecDense {
	if size := g.Network().BatchSize(); size != 1 {
		panic(fmt.Sprintf("selectAction: action selection can only be done "+
			"with a policy with batch size 1 \n\twant(1) \n\thave(%v)", size))
	}

	obs := t.Observation.RawVector().Data
	if err := g.Network().SetInput(obs); err != nil {
		panic(fmt.Sprintf("selectAction: cannot set input: %v", err))
	}

	if err := g.vm.RunAll(); err != nil {
		panic(fmt.Sprintf("selectAction: could not run policy VM: %v", err))
	}
	defer g.vm.Reset()

	// Make a copy of the mean value so that the meanVal field is
	// not modified
	meanVal := make([]float64, len(g.meanVal.Data().([]float64)))
	copy(meanVal, g.meanVal.Data().([]float64))
	mean := mat.NewVecDense(g.actionDims, meanVal)

	// If in evaluation mode, return the mean action only
	if g.IsEval() {
		return mean
	}

	// Make a copy of the stddev value so that the stddevVal field is
	// not modified
	stddevVal := make([]float64, len(g.stddevVal.Data().([]float64)))
	copy(stddevVal, g.stddevVal.Data().([]float64))
	stddev := mat.NewVecDense(g.actionDims, stddevVal)

	eps := mat.NewVecDense(g.actionDims, g.normal.Rand(nil))

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
	panic("clone: not implemented")
}

// CloneWithBatch clones a GaussianTreeMLP with a new batch size
func (c *GaussianTreeMLP) CloneWithBatch(batch int) (agent.NNPolicy, error) {
	panic("cloneWithBatch: not implemented")
}

// Network returns the network of the GaussianTreeMLP
func (g *GaussianTreeMLP) Network() network.NeuralNet {
	return g.net
}

// Train sets the policy to training mode
func (g *GaussianTreeMLP) Train() {
	g.eval = false
}

// Eval sets the policy to evaluation mode
func (g *GaussianTreeMLP) Eval() {
	g.eval = true
}

// IsEval returns whether or not the policy is in evaluation mode
func (g *GaussianTreeMLP) IsEval() bool {
	return g.eval
}

func (g *GaussianTreeMLP) Mean() G.Value {
	return g.meanVal
}

func (g *GaussianTreeMLP) StdDev() G.Value {
	return g.stddevVal
}
