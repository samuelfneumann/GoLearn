package policy

import (
	"fmt"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/network"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
)

// CategoricalMLP implements a categorical policy using an MLP to
// predict action logits in each state. Given an environment with N
// actions in each state, the probabilities of selecting any action
// are:
//
//		π(a|s) := softmax(MLP(s))
//
// Given a number of discrete actions in a number of states, the
// CategoricalMLP can calculate the log probability of selecting
// each of these actions in each corresponding state. This is useful
// for constructing policy gradients in a similar way to Vanilla
// Policy Gradient or REINFORCE. The CategoricalMLP cannot yet produce
// the log probability of the actions actually selected by the policy
// due to limitations of Gorgonia. Additionally, how could one compute
// the gradient of sampling from a categorical distribution?
//
// https://towardsdatascience.com/what-is-gumbel-softmax-7f6d9cdcb90e
type CategoricalMLP struct {
	net network.NeuralNet
	vm  G.VM

	// Node that holds the logits of the actions in each state.
	logits     *G.Node
	logitsVals G.Value

	// Node that holds the unnormalized probability of selecting each
	// action in a given state. These are used for action selection
	// only.
	probs    *G.Node
	probsVal G.Value

	// Log probability of actions that were input to the policy to
	// compute the log probability of. These actions may or may not
	// have been ever selected by the policy and are provided by
	// an external source.
	logProbInputActions    *G.Node
	logProbInputActionsVal G.Value

	// Matrix of one-hot rows, where each row specifies which action
	// to calculate the log prob of. These actions are input to the
	// policy, they may or may not have been selected by the policy
	// previously and are provided by an external source using the
	// LogProbOf() method.
	actionIndices *G.Node

	batchForLogProb int         // Number of actions to comput log prob of
	numActions      int         // Number of avalable actions in each state
	source          rand.Source // Source for action selection RNG
	seed            uint64      // Seed for source
}

// NewCategoricalMLP creates a new CategoricalMLP. The CategoricalMLP
// policy selects actions from a given environment env. The MLP is
// populated on a given Gorgonia ExprGraph g.
//
// The parameters hiddenSizes, biases, and activations determine the
// architecture of the MLP. For index i, hiddenSize[i] determines the
// number of hidden units in layer i, biases[i] determines if a bias
// unit is used in layer i, and activations[i] determines the activation
// function used in layer i.
//
// The batchForLogProb parameter controls how many (state, action) pairs
// will be used when predicting the log probability of input actions.
func NewCategoricalMLP(env environment.Environment, batchForLogProb int,
	g *G.ExprGraph, hiddenSizes []int, biases []bool,
	activations []*network.Activation, init G.InitWFn,
	seed uint64) (agent.LogPdfOfer, error) {
	// Categorical policies can only be used with discrete actions
	if env.ActionSpec().Cardinality == spec.Continuous {
		err := fmt.Errorf("newCategoricalMLP: softmax policy cannot be " +
			"used with continuous actions")
		return &CategoricalMLP{}, err
	}

	features := env.ObservationSpec().Shape.Len()
	numActions := int(env.ActionSpec().UpperBound.AtVec(0)) + 1

	// Create the MLP for predicting the action logits in each state
	net, err := network.NewMultiHeadMLP(features, batchForLogProb, numActions,
		g, hiddenSizes, biases, init, activations)
	if err != nil {
		return &CategoricalMLP{}, fmt.Errorf("newCategoricalMLP: could "+
			"not create policy network: %v", err)
	}

	// Logits and probabilities of action selection for the current
	// policy in the state(s) inputted to the policy's neural net.
	logits := net.Prediction()[0]
	probs := G.Must(G.Exp(logits))

	// Compute the log probability of actions that are input by an
	// external source using the LogProbOf() method.
	actionIndices := G.NewMatrix(
		net.Graph(),
		tensor.Float64,
		G.WithShape(logits.Shape()...),
		G.WithInit(G.Zeroes()),
		G.WithName("Action Indices"),
	)
	logitsInputActions := G.Must(G.HadamardProd(actionIndices, logits))
	logitsInputActions = G.Must(G.Sum(logitsInputActions, 1))
	inputsLogSumExp := LogSumExp(logits, 1)
	logProbInputActions := G.Must(G.Sub(logitsInputActions, inputsLogSumExp))

	// Create the rng for breaking action ties
	source := rand.NewSource(seed)

	pol := &CategoricalMLP{
		net:    net,
		logits: logits,
		probs:  probs,

		actionIndices: actionIndices,

		logProbInputActions: logProbInputActions,

		batchForLogProb: batchForLogProb,
		numActions:      numActions,

		source: source,
		seed:   seed,
	}

	// Keep track of some node's values
	G.Read(pol.logits, &pol.logitsVals)
	G.Read(pol.logProbInputActions, &pol.logProbInputActionsVal)
	G.Read(probs, &pol.probsVal)

	// If using a batch to compute the log probabilities of action
	// selection, then the policy cannot be used for action selection
	// at each timestep (which has only 1 action => batch size 1).
	// In this case, it's assumed that an external VM will be used for
	// learning the policy's weights.
	if batchForLogProb == 1 {
		vm := G.NewTapeMachine(net.Graph())
		pol.vm = vm
	}

	return pol, nil
}

// LogSumExp calculates the log of the summation of exponentials of
// logits along the given axis.
func LogSumExp(logits *G.Node, along int) *G.Node {
	max := G.Must(G.Max(logits, along))

	exponent := G.Must(G.BroadcastSub(logits, max, nil, []byte{1}))
	exponent = G.Must(G.Exp(exponent))

	sum := G.Must(G.Sum(exponent, along))
	log := G.Must(G.Log(sum))

	return G.Must(G.Add(max, log))
}

// Logits returns the logits predicted by the last run of the policy's
// computational graph.
func (c *CategoricalMLP) Logits() G.Value {
	return c.logitsVals
}

// LogPdfOf sets the computational graph of the policy with states s
// and actions a so that running a VM of the policy's computational
// graph will populate the log probability node with the log probability
// of selecting actions a in states s. The argument a should be a
// slice of discrete actions (0, 1, ..., N) and *not* a one-hot encoded
// version of these actions.
func (c *CategoricalMLP) LogPdfOf(s, a []float64) (*G.Node, error) {
	if err := c.Network().SetInput(s); err != nil {
		panic(err)
	}

	actionIndices := make([]float64, 0, c.numActions*c.batchForLogProb)
	for i := range a {
		row := make([]float64, c.numActions)
		row[int(a[i])] = 1.0
		actionIndices = append(actionIndices, row...)
	}
	actionIndicesTensor := tensor.NewDense(tensor.Float64,
		[]int{c.batchForLogProb, c.numActions},
		tensor.WithBacking(actionIndices),
	)
	G.Let(c.actionIndices, actionIndicesTensor)

	return c.LogPdfNode(), nil
}

// SelectAction selects and returns an action at the argument timestep
// t.
func (c *CategoricalMLP) SelectAction(t timestep.TimeStep) *mat.VecDense {
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
	logProbActions := c.probsVal.Data().([]float64)
	c.vm.Reset()

	dist := distuv.NewCategorical(logProbActions, c.source)
	action := mat.NewVecDense(1, []float64{dist.Rand()})

	return action
}

// LogPdfNode returns the node that will hold the log probability
// of actions when the comptuational graph is run.
func (c *CategoricalMLP) LogPdfNode() *G.Node {
	return c.logProbInputActions
}

// LogPdfVal returns the value of the node returned by LogPdfNode()
func (c *CategoricalMLP) LogPdfVal() G.Value {
	return c.logProbInputActionsVal
}

// Clone clones a CategoricalMLP
func (c *CategoricalMLP) Clone() (agent.NNPolicy, error) {
	return nil, fmt.Errorf("clone: not implemented")
}

// CloneWithBatch clones a CategoricalMLP with a new batch size
func (c *CategoricalMLP) CloneWithBatch(batch int) (agent.NNPolicy, error) {
	return nil, fmt.Errorf("cloneWithBatch: not implemented")
}

// Network returns the network of the CategoricalMLP
func (c *CategoricalMLP) Network() network.NeuralNet {
	return c.net
}
