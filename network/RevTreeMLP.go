package network

import (
	"fmt"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"sfneuman.com/golearn/utils/intutils"
)

// revTreeMLP implements a reverse tree MLP. A reverse tree MLP has
// multiple root networks, each with its own input. Each of these root
// networks predicts somve value. All the predictions of all the root
// networks are concatenated into a single feature vector and sent
// as input to a final, single leaf network. The overall architecture:
//
// Input 1     ─→ Root Network 1     ─╮
// Input 2     ─→ Root Network 2     ─┤
//   ... 	   ─→ 	   ...           ─┤
//   ... 	   ─→ 	   ...           ─┼─→ Leaf Network ─→ Output
//   ... 	   ─→ 	   ...           ─┤
// Input (N-1) ─→ Root Network (N-1) ─┤
// Input N     ─→ Root Network N     ─╯
type revTreeMLP struct {
	g            *G.ExprGraph
	rootNetworks []NeuralNet // Observation network
	leafNetwork  NeuralNet   // Leaf networks
	inputs       []*G.Node   // Input to observation network

	numOutputs int
	numInputs  []int // input features per root network
	batchSize  int

	// Store learnables and model so that they don't need to be computed
	// each time a gradient step is taken
	learnables G.Nodes
	model      []G.ValueGrad

	// Configuration data needed for gobbing
	rootHiddenSizes [][]int
	rootBiases      [][]bool
	rootActivations [][]*Activation
	leafHiddenSizes []int
	leafBiases      []bool
	leafActivations []*Activation

	predVal    []G.Value // Values predicted by each leaf node
	prediction []*G.Node // Nodes holding the predictions
}

// validateRevTreeMLP validates the arguments of NewRevTreeMLP() to ensure
// they are legal.
func validateRevTreeMLP(features []int, rootHiddenSizes [][]int, rootBiases [][]bool,
	rootActivations [][]*Activation, leafHiddenSizes []int,
	leafBiases []bool, leafActivations []*Activation) error {
	if len(leafHiddenSizes) != len(leafActivations) {
		msg := "invalid number of leaf activations" +
			"\n\twant(%d)\n\thave(%d)"
		return fmt.Errorf(msg, len(leafHiddenSizes),
			len(leafActivations))
	}

	if len(leafHiddenSizes) != len(leafBiases) {
		msg := "invalid number of leaf biases" +
			"\n\twant(%d)\n\thave(%d)"
		return fmt.Errorf(msg, len(leafHiddenSizes), len(leafBiases))
	}

	if len(rootHiddenSizes) <= 0 || len(rootBiases) <= 0 ||
		len(rootActivations) <= 0 {
		msg := "there must be at least one root network specified"
		return fmt.Errorf(msg)
	}

	if len(rootHiddenSizes) != len(rootActivations) {
		msg := "invalid number of root network activations " +
			"\n\twant(%v) \n\thave(%v)"
		return fmt.Errorf(msg, len(rootHiddenSizes), len(rootActivations))
	}

	if len(rootHiddenSizes) != len(rootBiases) {
		msg := "invalid number of root network biases " +
			"\n\twant(%v) \n\thave(%v)"
		return fmt.Errorf(msg, len(rootHiddenSizes), len(rootBiases))
	}

	if len(features) != len(rootHiddenSizes) {
		msg := "must specify length of features for each root network"
		return fmt.Errorf(msg)
	}

	// Validate architecture of leaf networks
	for i := 0; i < len(rootHiddenSizes); i++ {
		if len(rootHiddenSizes[i]) != len(rootActivations[i]) {
			msg := "invalid number of activations for root " +
				"network %v \n\twant(%v) \n\thave(%v)"
			return fmt.Errorf(msg, i, len(rootHiddenSizes[i]),
				len(rootActivations[i]))
		}

		if len(rootHiddenSizes[i]) != len(rootBiases[i]) {
			msg := "invalid number of biases for root " +
				"network %v \n\twant(%v) \n\thave(%v)"
			return fmt.Errorf(msg, i, len(rootHiddenSizes[i]),
				len(rootBiases[i]))
		}
	}

	return nil
}

// NewRevTreeMLP returns a new NeuralNet with a reverse tree MLP
// structure.
//
// The number of observation/root networks is equal to
// len(rootHiddenSizes). For index i, rootHiddenSizes[i], rootBiases[i],
// and rootActivations[i] determine the architecture of the ith
// root network. The ith root network has number of layers equal to
// len(rootHiddenSizes[i]). Given index j, rootHiddenSizes[i][j]
// determines the number of hidden units in layer j of root network i;
// rootBiases[i][j] determines whether a bias unit is added to layer j
// of root network i; rootActivations[i][j] determines the activation
// function to use for layer j of root network i.
//
// The leaf network's architecture is defined by leafHiddenSizes,
// leafBiases, and leafActivations in a similar manner. The number of
// layers in the leaf network is defined by len(leafHiddenSizes). Given
// index i, leafHiddenSizes[i] determines the number of hidden units
// in layer i; leafBiases[i] determines whether or not a bias unit is
// added to layer i; leafActivations[i] determines the activation
// function for layer i. A final linear layer with a bias unit and no
// activations is added to the leaf network to ensure the output of the
// network has the shape outputs.
//
// To create a network that has a single linear layer as the leaf
// network, simply use leafHiddenSizes = []int{}, leafBiases = []bool{},
// and leafActivations = []*network.Activations{}. The final linear
// layer will be added automatically.
func NewRevTreeMLP(features []int, batch, outputs int, g *G.ExprGraph,
	rootHiddenSizes [][]int, rootBiases [][]bool,
	rootActivations [][]*Activation, leafHiddenSizes []int, leafBiases []bool,
	leafActivations []*Activation, init G.InitWFn) (NeuralNet, error) {
	// Ensure the input is valid
	err := validateRevTreeMLP(features, rootHiddenSizes, rootBiases,
		rootActivations, leafHiddenSizes, leafBiases, leafActivations)
	if err != nil {
		return nil, fmt.Errorf("newtreemlp: %v", err)
	}

	// Construct root networks
	rootNetworks := make([]NeuralNet, len(rootHiddenSizes))
	rootPredictions := make([]*G.Node, 0, len(rootHiddenSizes))
	inputs := make([]*G.Node, len(rootHiddenSizes))
	for i := range rootNetworks {
		// Set up the input node
		inputs[i] = G.NewMatrix(g, tensor.Float64, G.WithShape(batch, features[i]),
			G.WithName(fmt.Sprintf("Root%dInput", i)), G.WithInit(G.Zeroes()))

		// Create individual root networks and run each's forward pass
		rootOutputs := rootHiddenSizes[i][len(rootHiddenSizes)-1]
		prefix := fmt.Sprintf("Root%d", i)
		rootInput := []*G.Node{inputs[i]}
		rootNetwork, err := newMultiHeadMLPFromInput(rootInput, rootOutputs, g,
			rootHiddenSizes[i], rootBiases[i], init, rootActivations[i],
			prefix, "", false)
		if err != nil {
			return nil, fmt.Errorf("newrevtreemlp: could not construct root "+
				"network %v: %v", i, err)
		}
		rootNetworks[i] = rootNetwork
		rootPredictions = append(rootPredictions, rootNetwork.Prediction()...)
	}

	// Concatenate outputs of root networks
	rootOutput := []*G.Node{G.Must(G.Concat(1, rootPredictions...))}

	// Create leaf networks and run its forward pass
	leafNetwork, err := newMultiHeadMLPFromInput(rootOutput, outputs, g,
		leafHiddenSizes, leafBiases, init, leafActivations,
		"Leaf", "", true)
	if err != nil {
		return nil, fmt.Errorf("newtreemlp: could not construct leaf "+
			"network: %v", err)
	}

	net := &revTreeMLP{
		g:               g,
		rootNetworks:    rootNetworks,
		leafNetwork:     leafNetwork,
		inputs:          inputs,
		numOutputs:      outputs * len(leafHiddenSizes),
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

	return net, nil
}

// SetInput sets the value of the input node before running the forward
// pass. The input should be constructed as follows in row major order:
//
// Given a batch size of N with M root networks:
// input =
// [all features for all observations in the batch for root network 1,
//  all features for all observations in the batch for root network 2,
//  ...,
//  all features for all observations in the batch for root network M-1,
//  all features for all observations in the batch for root network M,
// ]
//
// So, the first N * numFeatures for root network 1 will be all the
// features for each sample in the batch and will be input to root net
// 1; the next N * numFeatures for root network 2 will be all the
// features for each sample in the batch and will be input to root net
// 2; and so on.
//
// For example, given 2 root networks, with the first taking in 10
// features and the second taking in 7 features with a batch size of 5,
// the first 10 * 5 = 50 floats in input will be used as the input for
// the first root network, with the first 10 features forming the first
// sample in the batch, the next 10 features forming the second sample
// in the batch, etc. The next 7 * 5 = 35 float in input will be
// used as input for the second root network, with the first 7 features
// forming the first sample in the batch, the next 7 features forming
// the second sample in the batch, etc.
func (t *revTreeMLP) SetInput(input []float64) error {
	if len(input) != intutils.Prod(t.Features()...)*t.batchSize {
		msg := fmt.Sprintf("invalid number of inputs\n\twant(%v)"+
			"\n\thave(%v)", intutils.Prod(t.Features()...)*t.batchSize,
			len(input))
		panic(msg)
	}

	start := 0
	stop := 0
	for i, rootInput := range t.inputs {
		start = stop
		stop = t.numInputs[i] * t.BatchSize()
		inputTensor := tensor.New(
			tensor.WithBacking(input[start:stop]),
			tensor.WithShape(rootInput.Shape()...),
		)
		return G.Let(rootInput, inputTensor)
	}

	return nil
}

// Outputs returns the number of outputs per leaf network
func (t *revTreeMLP) Outputs() []int {
	return []int{t.numOutputs}
}

// OutputLayers returns the number of output layers in the network.
// There is one output layer per leaf network.
func (t *revTreeMLP) OutputLayers() int {
	return len(t.Prediction())
}

// Graph returns the computational graph of the network
func (t *revTreeMLP) Graph() *G.ExprGraph {
	return t.g
}

// Features returns the number of input features
func (t *revTreeMLP) Features() []int {
	return t.numInputs
}

// Clone returns a clone of the revTreeMLP.
func (t *revTreeMLP) Clone() (NeuralNet, error) {
	return t.CloneWithBatch(t.batchSize)
}

// CloneWithBatch returns a clone of the revTreeMLP with a new input
// batch size.
func (t *revTreeMLP) CloneWithBatch(batchSize int) (NeuralNet, error) {
	graph := G.NewGraph()

	// Create the input nodes
	inputs := make([]*G.Node, len(t.inputs))

	for i, input := range t.inputs {
		if input.IsMatrix() {
			inputs[i] = input.CloneTo(graph)
		} else {
			return nil, fmt.Errorf("clonewithbatch: invalid input type")
		}

	}

	return t.cloneWithInputTo(-1, inputs, graph)
}

// cloneWithInputTo clones the revTreeMLP to a new graph with a given
// input node. There should be one input node for each root network.
// If the leaf network only takes in a single input node, then the
// outputs of the root networks are first concatenated.
func (t *revTreeMLP) cloneWithInputTo(axis int, inputs []*G.Node,
	graph *G.ExprGraph) (NeuralNet, error) {
	// Ensure one input is given for each root network
	if len(inputs) != len(t.inputs) {
		return nil, fmt.Errorf("clonewithinputto: must specify a single "+
			"input for each network root node \n\twant(%v) \n\thave(%v)",
			len(t.inputs), len(inputs))
	}

	rootClones := make([]NeuralNet, len(t.rootNetworks))
	rootOutputs := make([]*G.Node, 0, len(t.rootNetworks))
	features := make([]int, len(t.rootNetworks))
	// Ensure all inputs share the same graph
	for i, input := range inputs {
		if input.Graph() != graph {
			return nil, fmt.Errorf("clonewithinputto: not all inputs " +
				"have the same graph")
		}

		if !input.IsMatrix() {
			return nil, fmt.Errorf("cloneWithInputTo: input must be a " +
				"matrix node")
		}
		features[i] = input.Shape()[1]

		rootClone, err := t.rootNetworks[i].cloneWithInputTo(-1,
			[]*G.Node{input}, graph)
		if err != nil {
			return nil, fmt.Errorf("clonewithbatch: could not clone root "+
				"network %v: %v", i, err)
		}
		rootClones[i] = rootClone
		rootOutputs = append(rootOutputs, rootClone.Prediction()...)
	}
	batchSize := inputs[0].Shape()[0]

	leafClone, err := t.leafNetwork.cloneWithInputTo(axis, rootOutputs, graph)
	if err != nil {
		msg := "cloneWithInputTo: could not clone leaf network: %v"
		return nil, fmt.Errorf(msg, err)
	}

	net := &revTreeMLP{
		g:            graph,
		rootNetworks: rootClones,
		leafNetwork:  leafClone,
		// rootLayers:      rootLayers,
		// leafLayers:      leafLayers,
		inputs:          inputs,
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

	return net, nil
}

// BatchSize returns the batch size for inputs to the network
func (t *revTreeMLP) BatchSize() int {
	return t.rootNetworks[0].BatchSize()
}

// fwd computes the remaining steps of the forward pass of the revTreeMLP
// that its root and leaf networks did not compute.
func (t *revTreeMLP) fwd(input *G.Node) (*G.Node, error) {
	return nil, nil
}

// Output returns the output of the revTreeMLP. The output is
// a matrix of NxM dimensions, where N corresponds to the number of
// outputs per leaf network and M the number of leaf networks.
func (t *revTreeMLP) Output() []G.Value {
	return t.predVal
}

// Prediction returns the node of the computational graph the stores
// the output of the revTreeMLP
func (t *revTreeMLP) Prediction() []*G.Node {
	return t.prediction
}

// Model returns the learnable nodes with their gradients.
func (t *revTreeMLP) Model() []G.ValueGrad {
	// Lazy instantiation of model
	if t.model == nil {
		t.model = t.computeModel()
	}
	return t.model
}

// computeModel gets and returns all learnables of the network with
// their gradients
func (t *revTreeMLP) computeModel() []G.ValueGrad {
	var model []G.ValueGrad
	for _, learnable := range t.Learnables() {
		model = append(model, learnable)
	}
	return model
}

// Learnables returns the learnable nodes in a multiHeadMLP
func (t *revTreeMLP) Learnables() G.Nodes {
	// Lazy instantiation of learnables
	if t.learnables == nil {
		t.learnables = t.computeLearnables()
	}
	return t.learnables
}

// computeLearnables gets and returns all learnables of the network
func (t *revTreeMLP) computeLearnables() G.Nodes {
	// Allocate array of learnables
	numLearnables := 2 * len(t.leafHiddenSizes)
	for _, layer := range t.rootHiddenSizes {
		numLearnables += (2 * len(layer))
	}
	learnables := make([]*G.Node, 0, numLearnables)

	// Add learnables to array
	learnables = append(learnables, t.leafNetwork.Learnables()...)
	for _, leafNet := range t.rootNetworks {
		learnables = append(learnables, leafNet.Learnables()...)

	}

	return G.Nodes(learnables)
}
