package network

import (
	"fmt"
	"log"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type MultiHeadMLP struct {
	g          *G.ExprGraph
	layers     []FCLayer
	input      *G.Node
	numOutputs int
	numInputs  int
	batchSize  int

	prediction *G.Node
	predVal    G.Value
}

func NewMultiHeadMLP(features, batch, outputs int, g *G.ExprGraph,
	hiddenSizes []int, biases []bool, init G.InitWFn,
	activations []Activation) (NeuralNet, error) {

	// Ensure we have one activation per layer
	if len(hiddenSizes) != len(activations) {
		msg := "newmultiheadegreedymlp: invalid number of activations" +
			"\n\twant(%d)\n\thave(%d)"
		return nil, fmt.Errorf(msg, len(hiddenSizes), len(activations))
	}

	// Ensure one bias bool per layer
	if len(hiddenSizes) != len(biases) {
		msg := "newmultiheadegreedymlp: invalid number of biases\n\twant(%d)" +
			"\n\thave(%d)"
		return nil, fmt.Errorf(msg, len(hiddenSizes), len(biases))
	}

	// Set up the input node
	input := G.NewMatrix(g, tensor.Float64, G.WithShape(batch, features),
		G.WithName("input"), G.WithInit(G.Zeroes()))

	// If no given hidden layers, then use a single linear layer so that
	// the output has output heads
	hiddenSizes = append(hiddenSizes, outputs)
	biases = append(biases, true)
	activations = append(activations, nil)

	// Create the fully connected layers
	layers := make([]FCLayer, 0, len(hiddenSizes))
	for i := range hiddenSizes {
		// Create the weights for the layer
		var Weights *G.Node
		if i == 0 {
			// First layer
			weightName := fmt.Sprintf("L%dW", i)
			Weights = G.NewMatrix(
				g,
				tensor.Float64,
				G.WithShape(features, hiddenSizes[i]),
				G.WithName(weightName),
				G.WithInit(init),
			)
		} else {
			// Layers other than the first
			weightName := fmt.Sprintf("L%dW", i)
			Weights = G.NewMatrix(
				g,
				tensor.Float64,
				G.WithShape(hiddenSizes[i-1], hiddenSizes[i]),
				G.WithName(weightName),
				G.WithInit(init),
			)

		}

		// Create the bias unit for the layer
		var Bias *G.Node
		if biases[i] {
			biasName := fmt.Sprintf("L%dB", i)
			Bias = G.NewVector(
				g,
				tensor.Float64,
				G.WithShape(hiddenSizes[i]),
				G.WithName(biasName),
				G.WithInit(init),
			)
		}

		// Create the fully connected layer
		layer := FCLayer{
			Weights: Weights,
			Bias:    Bias,
			Act:     activations[i],
		}
		layers = append(layers, layer)
	}

	// Create the network and run the forward pass on the input node
	network := MultiHeadMLP{
		g:          g,
		layers:     layers,
		input:      input,
		numOutputs: outputs,
		numInputs:  features,
		batchSize:  batch,
	}
	_, err := network.Fwd(input)
	if err != nil {
		log.Fatal(err)
	}

	return &network, nil
}

// Graph returns the computational graph of the MultiHeadMLP.
func (e *MultiHeadMLP) Graph() *G.ExprGraph {
	return e.g
}

// Clone clones a MultiHeadMLP
func (e *MultiHeadMLP) Clone() (NeuralNet, error) {
	batchSize := e.input.Shape()[0]
	return e.CloneWithBatch(batchSize)
}

// CloneWithBatch clones a MultiHeadMLP with a new input batch
// size.
func (e *MultiHeadMLP) CloneWithBatch(
	batchSize int) (NeuralNet, error) {
	graph := G.NewGraph()

	// Copy fully connected layers
	l := make([]FCLayer, len(e.layers))
	for i := range e.layers {
		l[i] = e.layers[i].CloneTo(graph)
	}

	// Create the input node
	inputShape := e.input.Shape()
	var input *G.Node
	if e.input.IsMatrix() {
		batchShape := append([]int{batchSize}, inputShape[1:]...)
		input = G.NewMatrix(
			graph,
			tensor.Float64,
			G.WithShape(batchShape...),
			G.WithName("input"),
			G.WithInit(G.Zeroes()),
		)
	} else {
		panic("clone: invalid input type")
	}

	// Create the network and run the forward pass on the input node
	network := MultiHeadMLP{
		g:          graph,
		layers:     l,
		input:      input,
		numOutputs: e.numOutputs,
		numInputs:  e.numInputs,
		batchSize:  batchSize,
	}
	_, err := network.Fwd(input)
	if err != nil {
		msg := fmt.Sprintf("clonewithbatch: could not clone: %v", err)
		panic(msg)
	}

	return &network, nil
}

// BatchSize returns the batch size of inputs to the policy
func (e *MultiHeadMLP) BatchSize() int {
	return e.batchSize
}

// Features returns the number of features in a single observation
// vector that the policy takes as input.
func (e *MultiHeadMLP) Features() int {
	return e.numInputs
}

// SetInput sets the value of the input node before running the forward
// pass.
func (e *MultiHeadMLP) SetInput(input []float64) error {
	if len(input) != e.numInputs*e.batchSize {
		msg := fmt.Sprintf("invalid number of inputs\n\twant(%v)"+
			"\n\thave(%v)", e.numInputs*e.batchSize, len(input))
		panic(msg)
	}
	inputTensor := tensor.New(
		tensor.WithBacking(input),
		tensor.WithShape(e.input.Shape()...),
	)
	return G.Let(e.input, inputTensor)
}

// Set sets the weights of a MultiHeadMLP to be equal to the
// weights of another MultiHeadMLP
func (dest *MultiHeadMLP) Set(source NeuralNet) error {
	sourceNodes := source.Learnables()
	nodes := dest.Learnables()
	for i, destLearnable := range nodes {
		sourceLearnable := sourceNodes[i].Clone()
		err := G.Let(destLearnable, sourceLearnable.(*G.Node).Value())
		if err != nil {
			return err
		}
	}
	return nil
}

// Polyak sets the weights of a MultiHeadMLP to be a polyak
// average between its existing weights and the weights of another
// MultiHeadMLP
func (dest *MultiHeadMLP) Polyak(source NeuralNet, tau float64) error {
	sourceNodes := source.Learnables()
	nodes := dest.Learnables()
	for i := range nodes {
		weights := nodes[i].Value().(*tensor.Dense)
		sourceWeights := sourceNodes[i].Value().(*tensor.Dense)

		weights, err := weights.MulScalar(1-tau, true)
		if err != nil {
			return err
		}

		sourceWeights, err = sourceWeights.MulScalar(tau, true)
		if err != nil {
			return err
		}

		var newWeights *tensor.Dense
		newWeights, err = weights.Add(sourceWeights)
		if err != nil {
			return err
		}

		G.Let(nodes[i], newWeights)
	}
	return nil
}

// Learnables returns the learnable nodes in a MultiHeadMLP
func (e *MultiHeadMLP) Learnables() G.Nodes {
	learnables := make([]*G.Node, 0, 2*len(e.layers))

	for i := range e.layers {
		learnables = append(learnables, e.layers[i].Weights)
		if bias := e.layers[i].Bias; bias != nil {
			learnables = append(learnables, bias)
		}
	}
	return G.Nodes(learnables)
}

// Model returns the learnables nodes with their gradients.
func (e *MultiHeadMLP) Model() []G.ValueGrad {
	var model []G.ValueGrad = make([]G.ValueGrad, 0, 2*len(e.layers))

	for i := range e.layers {
		model = append(model, e.layers[i].Weights)
		if bias := e.layers[i].Bias; bias != nil {
			model = append(model, bias)
		}
	}
	return model
}

// Fwd performs the forward pass of the MultiHeadMLP on the input
// node
func (e *MultiHeadMLP) Fwd(input *G.Node) (*G.Node, error) {
	inputShape := input.Shape()[len(input.Shape())-1]
	if inputShape%e.numInputs != 0 {
		return nil, fmt.Errorf("invalid shape for input to neural net:"+
			" \n\twant(%v) \n\thave(%v)", e.numInputs, inputShape)
	}

	pred := input
	var err error
	for _, l := range e.layers {
		if pred, err = l.Fwd(pred); err != nil {
			log.Fatal(err)
		}
	}
	e.prediction = pred
	G.Read(e.prediction, &e.predVal)

	return pred, nil
}

// Output returns the output of the MultiHeadMLP. The output is
// a vector of N dimensions, where each dimension corresponds to an
// environmental action.
func (e *MultiHeadMLP) Output() G.Value {
	return e.predVal
}

// Prediction returns the node of the computational graph the stores
// the output of the MultiHeadMLP
func (e *MultiHeadMLP) Prediction() *G.Node {
	return e.prediction
}
