package policy

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/network"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/floatutils"
)

type DeterministicCategoricalMLP struct {
	net network.NeuralNet
	vm  G.VM

	LogSumExp G.Value

	logits     *G.Node
	logitsVals G.Value

	// logProbSelectedActions *G.Node

	logProbInputActions    *G.Node
	LogProbInputActionsVal G.Value
	actionIndices          *G.Node

	batchForLogProb int
	numActions      int

	rng *rand.Rand // RNG for breaking action ties
}

// NewDeterministicCategoricalMLP creates a new CategoricalMLP. The CategoricalMLP
// policy selects actions from a given environment env. The MLP is
// populated on a given Gorgonia ExprGraph.
//
// The parameters hiddenSizes, biases, and activations determine the
// architecture of the MLP.
func NewDeterministicCategoricalMLP(env environment.Environment, batchForLogProb int,
	g *G.ExprGraph, hiddenSizes []int, biases []bool,
	activations []*network.Activation, init G.InitWFn,
	seed int64) (agent.PolicyLogProber, error) {
	// Error checking
	if env.ActionSpec().Cardinality == spec.Continuous {
		err := fmt.Errorf("newDeterministicCategoricalMLP: softmax policy cannot be " +
			"used with continuous actions")
		return &DeterministicCategoricalMLP{}, err
	}

	features := env.ObservationSpec().Shape.Len()
	numActions := int(env.ActionSpec().UpperBound.AtVec(0)) + 1

	// Create the tree MLP, which may predict batches of means and
	// log standard deviations -> one for each state in the batch
	net, err := network.NewMultiHeadMLP(features, batchForLogProb, numActions, g,
		hiddenSizes, biases, init, activations)
	if err != nil {
		return &DeterministicCategoricalMLP{}, fmt.Errorf("newCategoricalMLP: could "+
			"not create policy network: %v", err)
	}

	logits := net.Prediction()[0]

	// // Calculate the log prob of selected action
	// // Selected actions will have highest logits
	// max := G.Must(G.Max(logits, 1))
	// logSumExp := LogSumExp(logits, 1)
	// logProbSelectedActions := G.Must(G.Sub(max, logSumExp))

	// Log probability of actions inputted by user with LogProbOf()
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
	rng := rand.New(source)

	pol := &DeterministicCategoricalMLP{
		net:    net,
		logits: logits,

		actionIndices: actionIndices,
		// logProbSelectedActions: logProbSelectedActions,

		logProbInputActions: logProbInputActions,

		batchForLogProb: batchForLogProb,
		numActions:      numActions,

		rng: rng,
	}
	G.Read(pol.logits, &pol.logitsVals)
	G.Read(pol.logProbInputActions, &pol.LogProbInputActionsVal)
	G.Read(inputsLogSumExp, &pol.LogSumExp)

	if batchForLogProb == 1 {
		vm := G.NewTapeMachine(net.Graph())
		pol.vm = vm
	}

	return pol, nil
}

// func (c *DeterministicCategoricalMLP) LogProbSelectedActions() *G.Node {
// 	return c.logProbSelectedActions
// }

func (c *DeterministicCategoricalMLP) Logits() G.Value {
	return c.logitsVals
}

func (c *DeterministicCategoricalMLP) LogProbOf(s, a []float64) (*G.Node, error) {
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

	return c.LogProbNode(), nil
}

func (c *DeterministicCategoricalMLP) SelectAction(t timestep.TimeStep) *mat.VecDense {

	obs := t.Observation.RawVector().Data

	if err := c.Network().SetInput(obs); err != nil {
		panic(err)
	}

	if err := c.vm.RunAll(); err != nil {
		panic(err)
	}
	logProbActions := c.logitsVals.Data().([]float64)
	c.vm.Reset()

	actions := floatutils.ArgMax(logProbActions...)
	index := c.rng.Int() % len(actions)
	action := float64(actions[index])

	return mat.NewVecDense(1, []float64{action})

}

func (c *DeterministicCategoricalMLP) LogProbNode() *G.Node {
	return c.logProbInputActions
}

func (c *DeterministicCategoricalMLP) Clone() (agent.NNPolicy, error) {
	return nil, fmt.Errorf("clone: not implemented")
}

func (c *DeterministicCategoricalMLP) CloneWithBatch(int) (agent.NNPolicy, error) {
	return nil, fmt.Errorf("cloneWithBatch: not implemented")
}

func (c *DeterministicCategoricalMLP) Network() network.NeuralNet {
	return c.net
}
