package policy

import (
	"fmt"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"gonum.org/v1/gonum/stat/sampleuv"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/network"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
)

type CategoricalMLP struct {
	net network.NeuralNet
	vm  G.VM

	LogSumExp G.Value

	logits     *G.Node
	logitsVals G.Value
	probs      *G.Node
	probsVal   G.Value

	// logProbSelectedActions *G.Node

	logProbInputActions    *G.Node
	LogProbInputActionsVal G.Value
	actionIndices          *G.Node

	batchForLogProb int
	numActions      int

	source rand.Source
}

// NewCategoricalMLP creates a new CategoricalMLP. The CategoricalMLP
// policy selects actions from a given environment env. The MLP is
// populated on a given Gorgonia ExprGraph.
//
// The parameters hiddenSizes, biases, and activations determine the
// architecture of the MLP.
func NewCategoricalMLP(env environment.Environment, batchForLogProb int,
	g *G.ExprGraph, hiddenSizes []int, biases []bool,
	activations []*network.Activation, init G.InitWFn,
	seed uint64) (agent.PolicyLogProber, error) {
	// Error checking
	if env.ActionSpec().Cardinality == spec.Continuous {
		err := fmt.Errorf("newCategoricalMLP: softmax policy cannot be " +
			"used with continuous actions")
		return &CategoricalMLP{}, err
	}

	features := env.ObservationSpec().Shape.Len()
	numActions := int(env.ActionSpec().UpperBound.AtVec(0)) + 1

	// Create the tree MLP, which may predict batches of means and
	// log standard deviations -> one for each state in the batch
	net, err := network.NewMultiHeadMLP(features, batchForLogProb, numActions, g,
		hiddenSizes, biases, init, activations)
	if err != nil {
		return &CategoricalMLP{}, fmt.Errorf("newCategoricalMLP: could "+
			"not create policy network: %v", err)
	}

	logits := net.Prediction()[0]
	probs := G.Must(G.Exp(logits))

	// // Calculate the log prob of selected action
	// // Selected actions will have highest logits
	// ! This needs to be reworkd -> selected action != max always!
	// ! We need another one-hot row matrix to choose selected actions
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

	pol := &CategoricalMLP{
		net:    net,
		logits: logits,
		probs:  probs,

		actionIndices: actionIndices,
		// logProbSelectedActions: logProbSelectedActions,

		logProbInputActions: logProbInputActions,

		batchForLogProb: batchForLogProb,
		numActions:      numActions,

		source: source,
	}
	G.Read(pol.logits, &pol.logitsVals)
	G.Read(pol.logProbInputActions, &pol.LogProbInputActionsVal)
	G.Read(inputsLogSumExp, &pol.LogSumExp)
	G.Read(probs, &pol.probsVal)

	if batchForLogProb == 1 {
		vm := G.NewTapeMachine(net.Graph())
		pol.vm = vm
	}

	return pol, nil
}

func LogSumExp(logits *G.Node, along int) *G.Node {
	// Calculate the max logit per row
	max := G.Must(G.Max(logits, along))

	exponent := G.Must(G.BroadcastSub(logits, max, nil, []byte{1}))
	exponent = G.Must(G.Exp(exponent))

	// Sum along rows
	sum := G.Must(G.Sum(exponent, along))

	log := G.Must(G.Log(sum))

	return G.Must(G.Add(max, log))
}

// func (c *CategoricalMLP) LogProbSelectedActions() *G.Node {
// 	return c.logProbSelectedActions
// }

func (c *CategoricalMLP) Logits() G.Value {
	return c.logitsVals
}

func (c *CategoricalMLP) LogProbOf(s, a []float64) (*G.Node, error) {
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
	fmt.Println(actionIndicesTensor)
	G.Let(c.actionIndices, actionIndicesTensor)

	return c.LogProbNode(), nil
}

func (c *CategoricalMLP) SelectAction(t timestep.TimeStep) *mat.VecDense {

	obs := t.Observation.RawVector().Data

	if err := c.Network().SetInput(obs); err != nil {
		panic(err)
	}

	if err := c.vm.RunAll(); err != nil {
		panic(err)
	}
	logProbActions := c.probsVal.Data().([]float64)
	c.vm.Reset()

	dist := distuv.NewCategorical(logProbActions, c.source)
	sampler := sampleuv.IIDer{Dist: dist}

	action := mat.NewVecDense(1, nil) // ? This needs to be N-dim action
	sampler.Sample(action.RawVector().Data)

	return action
}

func (c *CategoricalMLP) LogProbNode() *G.Node {
	return c.logProbInputActions
}

func (c *CategoricalMLP) Clone() (agent.NNPolicy, error) {
	return nil, fmt.Errorf("clone: not implemented")
}

func (c *CategoricalMLP) CloneWithBatch(int) (agent.NNPolicy, error) {
	return nil, fmt.Errorf("cloneWithBatch: not implemented")
}

func (c *CategoricalMLP) Network() network.NeuralNet {
	return c.net
}