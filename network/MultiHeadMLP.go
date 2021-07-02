package network

import (
	"bytes"
	"encoding/gob"
	"fmt"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// multiHeadMLP implements a multi-layered perceptron with multiple
// output nodes, one for each value that should be predicted.
type multiHeadMLP struct {
	g          *G.ExprGraph
	layers     []Layer
	input      *G.Node
	numOutputs int
	numInputs  int
	batchSize  int

	// Data needed for gobbing
	hiddenSizes []int
	biases      []bool
	activations []*Activation

	prediction *G.Node
	predVal    G.Value
}

// NewmultiHeadMLP creates and returns a new multi-layered perceptron
// that has multiple output nodes, The number of outputs nodes is equal
// to outputs. The graph parameter g is populated with the MLP.
//
// The MLP has number of layers equal to len(hiddenSizes) + 1. A final
// layer is always added such that given any input, the output will
// be outputs. The final layer also contains a bias unit, and bias units
// for each additional hidden layer is specified by biases. The final
// layer will contain no activations, and the activations of additional
// hidden layers is specified by activations. The parameter init
// determines the weight initialization scheme.
//
// The function works such that for index i, hiddenSizes[i] is the
// number of nodes in hidden layer i; biases[i] is true if the
// hidden layer will contain a bias unit and false otherwise; and
// activations[i] is the activation function for hidden layer i.
func NewMultiHeadMLP(features, batch, outputs int, g *G.ExprGraph,
	hiddenSizes []int, biases []bool, init G.InitWFn,
	activations []*Activation) (NeuralNet, error) {

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
	activations = append(activations, Identity())

	// Create the fully connected layers
	layers := make([]Layer, 0, len(hiddenSizes))
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
		layer := &fcLayer{
			weights: Weights,
			bias:    Bias,
			act:     activations[i],
		}
		layers = append(layers, layer)
	}

	// Create the network and run the forward pass on the input node
	network := multiHeadMLP{
		g:           g,
		layers:      layers,
		input:       input,
		numOutputs:  outputs,
		numInputs:   features,
		batchSize:   batch,
		hiddenSizes: hiddenSizes,
		biases:      biases,
		activations: activations,
	}
	_, err := network.fwd(input)
	if err != nil {
		msg := "newmultiheadmlp: could not compute forward pass: %v"
		return &multiHeadMLP{}, fmt.Errorf(msg, err)
	}

	return &network, nil
}

// Graph returns the computational graph of the multiHeadMLP.
func (e *multiHeadMLP) Graph() *G.ExprGraph {
	return e.g
}

// Clone clones a multiHeadMLP
func (e *multiHeadMLP) Clone() (NeuralNet, error) {
	batchSize := e.input.Shape()[0]
	return e.CloneWithBatch(batchSize)
}

// CloneWithBatch clones a multiHeadMLP with a new input batch
// size.
func (e *multiHeadMLP) CloneWithBatch(
	batchSize int) (NeuralNet, error) {
	graph := G.NewGraph()

	// Copy fully connected layers
	l := make([]Layer, len(e.layers))
	for i := range e.layers {
		l[i] = e.layers[i].CloneTo(graph).(*fcLayer)
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
	network := multiHeadMLP{
		g:           graph,
		layers:      l,
		input:       input,
		numOutputs:  e.numOutputs,
		numInputs:   e.numInputs,
		batchSize:   batchSize,
		hiddenSizes: e.hiddenSizes,
		biases:      e.biases,
		activations: e.activations,
	}
	_, err := network.fwd(input)
	if err != nil {
		msg := fmt.Sprintf("clonewithbatch: could not clone: %v", err)
		panic(msg)
	}

	return &network, nil
}

// BatchSize returns the batch size of inputs to the policy
func (e *multiHeadMLP) BatchSize() int {
	return e.batchSize
}

// Features returns the number of features in a single observation
// vector that the policy takes as input.
func (e *multiHeadMLP) Features() int {
	return e.numInputs
}

// Outputs returns the number of outputs from the network
func (e *multiHeadMLP) Outputs() int {
	return e.numOutputs
}

// SetInput sets the value of the input node before running the forward
// pass.
func (e *multiHeadMLP) SetInput(input []float64) error {
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

// Set sets the weights of a multiHeadMLP to be equal to the
// weights of another multiHeadMLP
func (dest *multiHeadMLP) Set(source NeuralNet) error {
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

// Polyak sets the weights of a multiHeadMLP to be a polyak
// average between its existing weights and the weights of another
// multiHeadMLP
func (dest *multiHeadMLP) Polyak(source NeuralNet, tau float64) error {
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

// Learnables returns the learnable nodes in a multiHeadMLP
func (e *multiHeadMLP) Learnables() G.Nodes {
	learnables := make([]*G.Node, 0, 2*len(e.layers))

	for i := range e.layers {
		learnables = append(learnables, e.layers[i].Weights())
		if bias := e.layers[i].Bias(); bias != nil {
			learnables = append(learnables, bias)
		}
	}
	return G.Nodes(learnables)
}

// Model returns the learnables nodes with their gradients.
func (e *multiHeadMLP) Model() []G.ValueGrad {
	var model []G.ValueGrad = make([]G.ValueGrad, 0, 2*len(e.layers))

	for i := range e.layers {
		model = append(model, e.layers[i].Weights())
		if bias := e.layers[i].Bias(); bias != nil {
			model = append(model, bias)
		}
	}
	return model
}

// fwd performs the forward pass of the multiHeadMLP on the input
// node
func (e *multiHeadMLP) fwd(input *G.Node) (*G.Node, error) {
	inputShape := input.Shape()[len(input.Shape())-1]
	if inputShape%e.numInputs != 0 {
		return nil, fmt.Errorf("invalid shape for input to neural net:"+
			" \n\twant(%v) \n\thave(%v)", e.numInputs, inputShape)
	}

	pred := input
	var err error
	for _, l := range e.layers {
		if pred, err = l.fwd(pred); err != nil {
			msg := "fwd: could not compute forward pass of layer %v"
			return nil, fmt.Errorf(msg, err)
		}
	}
	e.prediction = pred
	G.Read(e.prediction, &e.predVal)

	return pred, nil
}

// Output returns the output of the multiHeadMLP. The output is
// a vector of N dimensions, where each dimension corresponds to an
// environmental action.
func (e *multiHeadMLP) Output() G.Value {
	return e.predVal
}

// Prediction returns the node of the computational graph the stores
// the output of the multiHeadMLP
func (e *multiHeadMLP) Prediction() *G.Node {
	return e.prediction
}

// GobEncode implements the gob.GobEncoder interface
func (e *multiHeadMLP) GobEncode() ([]byte, error) {
	gob.Register(multiHeadMLP{})
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	err := enc.Encode(e.numOutputs)
	if err != nil {
		return nil, fmt.Errorf("gobencode: could not encode number of outputs")
	}

	err = enc.Encode(e.numInputs)
	if err != nil {
		return nil, fmt.Errorf("gobencode: could not encode number of inputs")
	}

	err = enc.Encode(e.BatchSize())
	if err != nil {
		return nil, fmt.Errorf("gobencode: could not encode batch size")
	}

	err = enc.Encode(e.hiddenSizes)
	if err != nil {
		return nil, fmt.Errorf("gobencode: could not encode hidden sizes")
	}

	err = enc.Encode(e.biases)
	if err != nil {
		return nil, fmt.Errorf("gobencode: could not encode biases")
	}

	err = enc.Encode(e.activations)
	if err != nil {
		return nil, fmt.Errorf("gobencode: could not encode activations")
	}

	// Store the fcLayers
	gob.Register(fcLayer{})
	for i, layer := range e.layers {
		// Encode layer
		err := enc.Encode(layer)
		if err != nil {
			msg := "gobencode: could not encode layer %v: %v"
			return nil, fmt.Errorf(msg, i, err)
		}
	}

	return buf.Bytes(), nil
}

// GobDecode implements the gob.GobDecoder interface
func (e *multiHeadMLP) GobDecode(in []byte) error {
	gob.Register(multiHeadMLP{})
	buf := bytes.NewReader(in)
	dec := gob.NewDecoder(buf)

	var numOutputs int
	err := dec.Decode(&numOutputs)
	if err != nil {
		return fmt.Errorf("gobdecode: could not decode number of outputs")
	}

	var numInputs int
	err = dec.Decode(&numInputs)
	if err != nil {
		return fmt.Errorf("gobdecode: could not decode number of inputs")
	}

	var batchSize int
	err = dec.Decode(&batchSize)
	if err != nil {
		return fmt.Errorf("gobdecode: could not decode batch size")
	}

	var hiddenSizes []int
	err = dec.Decode(&hiddenSizes)
	if err != nil {
		return fmt.Errorf("gobdecode: could not decode hidden sizes")
	}
	hiddenSizes = hiddenSizes[:len(hiddenSizes)-1]

	var biases []bool
	err = dec.Decode(&biases)
	if err != nil {
		return fmt.Errorf("gobdecode: could not decode biases")
	}
	biases = biases[:len(biases)-1]

	var activations []*Activation
	err = dec.Decode(&activations)
	if err != nil {
		return fmt.Errorf("gobdecode: could not decode activations")
	}
	activations = activations[:len(activations)-1]

	// Create a new MLP
	g := G.NewGraph()
	newNet, err := NewMultiHeadMLP(numInputs, batchSize, numOutputs, g,
		hiddenSizes, biases, G.Zeroes(), activations)
	if err != nil {
		return fmt.Errorf("gobdecode: could not construct new MLP")
	}
	newMLP, ok := newNet.(*multiHeadMLP)
	if !ok {
		panic("NewmultiHeadMLP() returned type != multiHeadMLP")
	}

	// Fill new MLP's layers with fcLayer weights, equivalent to:
	// for i in 0, 1, 2, ... N:
	//     newMLP.layer[i].Weights().Value <- fcLayer[i].Weights.Value
	gob.Register(fcLayer{})
	numLayers := len(newMLP.layers)
	layers := newMLP.layers
	for i := 0; i < numLayers; i++ {
		err = dec.Decode(layers[i])
		if err != nil {
			return fmt.Errorf("gobdecode: could not decode layer %v: %v", i,
				err)
		}
	}

	*e = *newMLP
	return nil
}