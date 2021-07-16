package network

import (
	"fmt"
	"log"

	"gonum.org/v1/gonum/floats"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TreeMLP implements a multi-layered perceptron with a base observation
// netowrk and multiple leaf networks that use the output of the root
// observation network as their own inputs. A diagram of a tree MLP:
//
// 					  ╭─→ Leaf Network 1 	   ─→ Output
//					  ├─→ Leaf Network 2	   ─→ Output
//					  ├─→ ...				   ─→  ...
// Input ─→ Root Net ─┼─→ ...				   ─→  ...
//					  ├─→ ...				   ─→  ...
//					  ├─→ Leaf Network (N - 1) ─→ Output
//					  ╰─→ Leaf Network N	   ─→ Output
//
// To create outputs that are of the form [leaf1 output, leaf2 output,
// ..., leaf(N-1) output, leaf(N) output], once can simply call
// gorgonia.Concat() on the result of calling Prediction() on a TreeMLP.
type TreeMLP struct {
	g            *G.ExprGraph
	rootNetwork  NeuralNet   // Observation network
	leafNetworks []NeuralNet // Leaf networks
	input        *G.Node     // Input to observation network

	numOutputs []int // Number of outputs per leaf layer
	numInputs  int   // Features input for observation network
	batchSize  int

	// Store learnables and model so that they don't need to be computed
	// each time a gradient step is taken
	learnables G.Nodes
	model      []G.ValueGrad

	// Configuration data needed for gobbing
	rootHiddenSizes []int
	rootBiases      []bool
	rootActivations []*Activation
	leafHiddenSizes [][]int
	leafBiases      [][]bool
	leafActivations [][]*Activation

	predVal    []G.Value // Values predicted by each leaf node
	prediction []*G.Node // Nodes holding the predictions
}

// validateTreeMLP validates the arguments of newTreeMLP() to ensure
// they are legal.
func validateTreeMLP(numOutputs int, rootHiddenSizes []int, rootBiases []bool,
	rootActivations []*Activation, leafHiddenSizes [][]int,
	leafBiases [][]bool, leafActivations [][]*Activation) error {
	// Validate observation/root network
	if len(rootHiddenSizes) == 0 {
		return fmt.Errorf("root network must have at least one hidden layer")
	}

	if len(rootHiddenSizes) != len(rootActivations) {
		msg := "invalid number of root activations" +
			"\n\twant(%d)\n\thave(%d)"
		return fmt.Errorf(msg, len(rootHiddenSizes),
			len(rootActivations))
	}

	if len(rootHiddenSizes) != len(rootBiases) {
		msg := "invalid number of root biases" +
			"\n\twant(%d)\n\thave(%d)"
		return fmt.Errorf(msg, len(rootHiddenSizes), len(rootBiases))
	}

	// Validate number of leaf networks
	if len(leafHiddenSizes) <= 0 || len(leafBiases) <= 0 ||
		len(leafActivations) <= 0 {
		msg := "there must be at least one leaf network specified"
		return fmt.Errorf(msg)
	}

	if numOutputs <= 0 {
		return fmt.Errorf("there must be more than 0 outputs per leaf network")
	}

	if len(leafHiddenSizes) != len(leafActivations) {
		msg := "invalid number of leaf network activations " +
			"\n\twant(%v) \n\thave(%v)"
		return fmt.Errorf(msg, len(leafHiddenSizes), len(leafActivations))
	}

	if len(leafHiddenSizes) != len(leafBiases) {
		msg := "invalid number of leaf network biases " +
			"\n\twant(%v) \n\thave(%v)"
		return fmt.Errorf(msg, len(leafHiddenSizes), len(leafBiases))
	}

	// Validate architecture of leaf networks
	for i := 0; i < len(leafHiddenSizes); i++ {
		if len(leafHiddenSizes[i]) != len(leafActivations[i]) {
			msg := "invalid number of activations for leaf " +
				"network %v \n\twant(%v) \n\thave(%v)"
			return fmt.Errorf(msg, i, len(leafHiddenSizes[i]),
				len(leafActivations[i]))
		}

		if len(leafHiddenSizes[i]) != len(leafBiases[i]) {
			msg := "invalid number of biases for leaf " +
				"network %v \n\twant(%v) \n\thave(%v)"
			return fmt.Errorf(msg, i, len(leafHiddenSizes[i]),
				len(leafBiases[i]))
		}
	}

	return nil
}

// NewTreeMLP returns a new NeuralNet with a tree MLP architecture.
//
// The observation network has number of layers equal to
// len(rootHiddenSizes). For index i, rootHiddenSizes[i] determines the
// number of hidden units in that layer, rootBiases[i] determines if a
// bias unit is added to the hidden layer, and rootActivations[i]
// determines the activation function to apply to that hidden layer.
//
// The number of leaf networks is defined by len(leafHiddenSizes).
// For indices i and j, leafHiddenSizes[i][j], leafBiases[i][j], and
// leafActivations[i][j] determine the number of hidden units of layer
// j in leaf network i, whether a bias is added to layer j of leaf
// network i, and the activation of layer j of leaf network i
// respectively. The length of leafHiddenSizes[i] determines the number
// of hidden layers in leaf network i.
// For example, leafHiddenSizes = [][]int{{5, 3, 2}, {10, 90}} will
// cause this function to create two leaf networks. The first has
// layers of size 5, 3, and then 2. The second has two layers of size
// 10 and 90 respectively. For all leaf networks, a final linear layer
// with a bias and no activations is added to ensure the output of each
// leaf network has the shape output.
//
// To create a network with only a single linear layer per leaf network,
// set leafHiddenSize = [][]int{{}, {}, ..., {}} (similarly for
// leafBiases and leafActivations). The root observation can be left
// with nonlinearities to ensure all leaf networks use the same
// state representation but make (possibly different) predictions.
func NewTreeMLP(features, batch, outputs int, g *G.ExprGraph,
	rootHiddenSizes []int, rootBiases []bool, rootActivations []*Activation,
	leafHiddenSizes [][]int, leafBiases [][]bool,
	leafActivations [][]*Activation, init G.InitWFn) (NeuralNet, error) {

	err := validateTreeMLP(outputs, rootHiddenSizes, rootBiases, rootActivations,
		leafHiddenSizes, leafBiases, leafActivations)
	if err != nil {
		return nil, fmt.Errorf("newtreemlp: %v", err)
	}

	// Set up the input node
	input := G.NewMatrix(g, tensor.Float64, G.WithShape(batch, features),
		G.WithName("input"), G.WithInit(G.Zeroes()))

	// Create root/observation network and run its forward pass
	observationOutputs := rootHiddenSizes[len(rootHiddenSizes)-1]
	rootNetwork, err := newMultiHeadMLPFromInput([]*G.Node{input},
		observationOutputs, g, rootHiddenSizes, rootBiases, init,
		rootActivations, "Root", "", false)
	if err != nil {
		return nil, fmt.Errorf("newtreemlp: could not construct root "+
			"network: %v", err)
	}

	// Create leaf networks and run each of their forward passes
	rootOutput := rootNetwork.Prediction()
	// if len(rootOutput) != 1 {
	// 	return nil, fmt.Errorf("clonewithinputo: cannot use root network " +
	// 		" with multiple outputs as a single input to leaf networks")
	// }

	numOutputs := make([]int, len(leafHiddenSizes))
	leafNetworks := make([]NeuralNet, len(leafHiddenSizes))
	for i := 0; i < len(leafHiddenSizes); i++ {
		prefix := fmt.Sprintf("Leaf%d", i)

		leafNetworks[i], err = newMultiHeadMLPFromInput(rootOutput, outputs, g,
			leafHiddenSizes[i], leafBiases[i], init, leafActivations[i],
			prefix, "", true)

		if err != nil {
			return nil, fmt.Errorf("newtreemlp: could not construct leaf "+
				"network %v: %v", i, err)
		}
		numOutputs[i] = outputs
	}

	net := &TreeMLP{
		g:               g,
		rootNetwork:     rootNetwork,
		leafNetworks:    leafNetworks,
		input:           input,
		numOutputs:      numOutputs,
		numInputs:       features,
		batchSize:       batch,
		rootHiddenSizes: rootHiddenSizes,
		rootBiases:      rootBiases,
		rootActivations: rootActivations,
		leafHiddenSizes: leafHiddenSizes,
		leafBiases:      leafBiases,
		leafActivations: leafActivations,
		learnables:      nil,
		model:           nil,
	}

	// Compute the forward pass
	_, err = net.fwd([]*G.Node{input})
	if err != nil {
		msg := "newmtreemlp: could not compute forward pass: %v"
		return &TreeMLP{}, fmt.Errorf(msg, err)
	}

	// fmt.Println("TreeMLP", len(net.Graph().AllNodes()))
	return net, nil
}

func (t *TreeMLP) Layers() []Layer {
	return t.rootNetwork.(*MultiHeadMLP).Layers()
}

// SetInput sets the value of the input node before running the forward
// pass.
func (t *TreeMLP) SetInput(input []float64) error {
	if len(input) != t.numInputs*t.batchSize {
		msg := fmt.Sprintf("invalid number of inputs\n\twant(%v)"+
			"\n\thave(%v)", t.numInputs*t.batchSize, len(input))
		panic(msg)
	}
	inputTensor := tensor.New(
		tensor.WithBacking(input),
		tensor.WithShape(t.input.Shape()...),
	)

	w := t.rootNetwork.(*MultiHeadMLP).layers[0].Weights().Value().Data().([]float64)
	if floats.HasNaN(w) {
		log.Fatal("w has NaN")
	}

	return G.Let(t.input, inputTensor)
}

// Outputs returns the number of outputs per leaf network
func (t *TreeMLP) Outputs() []int {
	if len(t.numOutputs) != len(t.Prediction()) {
		panic("outputs: number of output layers incosistent with " +
			"computational graph as found by Prediction()")
	}
	return t.numOutputs
}

// OutputLayers returns the number of output layers in the network.
// There is one output layer per leaf network.
func (t *TreeMLP) OutputLayers() int {
	return len(t.Prediction())
}

// Graph returns the computational graph of the network
func (t *TreeMLP) Graph() *G.ExprGraph {
	return t.g
}

// Features returns the number of input features
func (t *TreeMLP) Features() []int {
	return []int{t.numInputs}
}

// Clone returns a clone of the TreeMLP.
func (t *TreeMLP) Clone() (NeuralNet, error) {
	return t.CloneWithBatch(t.batchSize)
}

// CloneWithBatch returns a clone of the TreeMLP with a new input
// batch size.
func (t *TreeMLP) CloneWithBatch(batchSize int) (NeuralNet, error) {
	graph := G.NewGraph()

	// Create the input node
	inputShape := t.input.Shape()
	var input *G.Node
	if t.input.IsMatrix() {
		batchShape := append([]int{batchSize}, inputShape[1:]...)
		input = G.NewMatrix(
			graph,
			tensor.Float64,
			G.WithShape(batchShape...),
			G.WithName("input"),
			G.WithInit(G.Zeroes()),
		)
	} else {
		return nil, fmt.Errorf("clonewithbatch: invalid input type")
	}

	retVal, err := t.cloneWithInputTo(-1, []*G.Node{input}, graph)

	return retVal, err
}

// cloneWithInputTo clones the TreeMLP to a new graph with a given
// input node. If multiple input nodes are given, then
// they are first concatenated along the specified axis. If the root
// network outputs multiple outputs, these outputs are also concatenated
// along the specified axis before being input to the leaf networks.
func (t *TreeMLP) cloneWithInputTo(axis int, inputs []*G.Node,
	graph *G.ExprGraph) (NeuralNet, error) {
	// Ensure all inputs share the same graph
	for _, input := range inputs {
		if input.Graph() != graph {
			return nil, fmt.Errorf("clonewithinputto: not all inputs " +
				"have the same graph")
		}
	}

	// Concatenate inputs if necessary
	var input []*G.Node
	if len(inputs) > 1 {
		input = []*G.Node{G.Must(G.Concat(axis, inputs...))}
	} else {
		input = []*G.Node{inputs[0]}
	}

	// Ensure the input is a matrix
	if !input[0].IsMatrix() {
		return nil, fmt.Errorf("cloneWithInputTo: input must be a matrix node")
	}
	batchSize := input[0].Shape()[0]
	features := input[0].Shape()[1]

	rootClone, err := t.rootNetwork.cloneWithInputTo(-1, input, graph)
	if err != nil {
		return nil, fmt.Errorf("clonewithbatch: could not clone root "+
			"network: %v", err)
	}

	rootOutput := rootClone.Prediction()

	leafClones := make([]NeuralNet, len(t.leafNetworks))
	for i := 0; i < len(leafClones); i++ {
		// Concatenate the root output along the specified axis if
		// there are multiple outputs from the root network
		leafClones[i], err = t.leafNetworks[i].cloneWithInputTo(axis, rootOutput, graph)
		if err != nil {
			msg := "cloneWithInputTo: could not clone leaf network %v: %v"
			return nil, fmt.Errorf(msg, i, err)
		}
	}

	net := &TreeMLP{
		g:            graph,
		rootNetwork:  rootClone,
		leafNetworks: leafClones,
		// rootLayers:      rootLayers,
		// leafLayers:      leafLayers,
		input:           input[0],
		numOutputs:      t.numOutputs,
		numInputs:       features,
		batchSize:       batchSize,
		rootHiddenSizes: t.rootHiddenSizes,
		rootBiases:      t.rootBiases,
		rootActivations: t.rootActivations,
		leafHiddenSizes: t.leafHiddenSizes,
		leafBiases:      t.leafBiases,
		leafActivations: t.leafActivations,
		learnables:      nil,
		model:           nil,
	}
	_, err = net.fwd(inputs)
	if err != nil {
		return nil, fmt.Errorf("cloneWithInputTo: could not compute "+
			"forward pass: %v", err)
	}

	return net, nil
}

// BatchSize returns the batch size for inputs to the network
func (t *TreeMLP) BatchSize() int {
	return t.rootNetwork.BatchSize()
}

// fwd computes the remaining steps of the forward pass of the TreeMLP
// that its root and leaf networks did not compute.
func (t *TreeMLP) fwd(inputs []*G.Node) (*G.Node, error) {
	// Because of the way TreeMLPs are constructed, there is nothing
	// to do with the input to the network, it's already been sent
	// through each sub-net's forward pass.
	if len(inputs) != 1 {
		return nil, fmt.Errorf("fwd: TreeMLP only supports a single "+
			"input \n\twant(1) \n\thave(%v)", len(inputs))
	}

	// Store all leaf predictions in a slice
	leafPredictions := make([]*G.Node, 0, len(t.leafNetworks))
	for _, leafNet := range t.leafNetworks {
		leafPredictions = append(leafPredictions, leafNet.Prediction()...)
	}
	t.prediction = leafPredictions

	t.predVal = make([]G.Value, len(t.prediction))
	for i, pred := range t.prediction {
		G.Read(pred, &t.predVal[i])
	}

	return nil, nil
}

// Output returns the output of the TreeMLP. The output is
// a matrix of NxM dimensions, where N corresponds to the number of
// outputs per leaf network and M the number of leaf networks.
func (t *TreeMLP) Output() []G.Value {
	return t.predVal
}

// Prediction returns the node of the computational graph the stores
// the output of the TreeMLP
func (t *TreeMLP) Prediction() []*G.Node {
	return t.prediction
}

// Model returns the learnable nodes with their gradients.
func (t *TreeMLP) Model() []G.ValueGrad {
	// Lazy instantiation of model
	if t.model == nil {
		t.model = t.computeModel()
	}
	return t.model
}

// computeModel gets and returns all learnables of the network with
// their gradients
func (t *TreeMLP) computeModel() []G.ValueGrad {
	var model []G.ValueGrad
	for _, learnable := range t.Learnables() {
		model = append(model, learnable)
	}
	return model
}

// Learnables returns the learnable nodes in a multiHeadMLP
func (t *TreeMLP) Learnables() G.Nodes {
	// Lazy instantiation of learnables
	if t.learnables == nil {
		t.learnables = t.computeLearnables()
	}
	return t.learnables
}

// computeLearnables gets and returns all learnables of the network
func (t *TreeMLP) computeLearnables() G.Nodes {
	// Allocate array of learnables
	numLearnables := 2 * len(t.rootHiddenSizes)
	for _, layer := range t.leafHiddenSizes {
		numLearnables += (2 * len(layer))
	}
	learnables := make([]*G.Node, 0, numLearnables)

	// Add learnables to array
	learnables = append(learnables, t.rootNetwork.Learnables()...)
	for _, leafNet := range t.leafNetworks {
		learnables = append(learnables, leafNet.Learnables()...)

	}

	return G.Nodes(learnables)
}
