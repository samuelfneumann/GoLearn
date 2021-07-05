package network

import (
	"fmt"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// treeMLP implements a multi-layered perceptron with a base observation
// netowrk and multiple leaf networks that use the output of the root
// observation network as their own inputs. A diagram of a tree MLP:
//
// 					 ╭─ Leaf Network 1 			-> Output
//					 ├─ Leaf Network 2			-> Output
//					 ├─ ...						...
// Input -> Root Net ┼─ ...						...
//					 ├─ ...						...
//					 ├─ Leaf Network (N - 1)	-> Output
//					 ╰─ Leaf Network N			-> Output
//
type treeMLP struct {
	g            *G.ExprGraph
	rootNetwork  NeuralNet   // Observation network
	leafNetworks []NeuralNet // Leaf networks
	input        *G.Node     // Input to observation network

	// numOutputs records the number of outputs *per leaf layer*.
	// The total number of outputs is numLeafLayers * numOutputs
	numOutputs int
	numInputs  int // Features input for observation network
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
// For index i, leafHiddenSizes[i], leafBiases[i], and
// leafActivations[i] determine the architecture of leaf network i.
// The length of any one of these slices determines the number of
// layers in the leaf network.
//
// Given additional index j, leafHiddenSizes[i][j], leafBiases[i][j],
// and leafActivations[i][j] determine the number of hidden units,
// whether or not a bias is added ot the hidden layer, and the
// activaiton function of the hidden layer i respectively.
// For example, leafHiddenSizes = [][]int{{5, 3, 2}, {10, 90}} will
// cause this function to create two leaf networks. The first has
// layers of size 5, 3, and then 2. The second has two layers of size
// 10 and 90 respectively
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
	rootNetwork, err := newMultiHeadMLPFromInput(input, observationOutputs, g,
		rootHiddenSizes, rootBiases, init, rootActivations, "Root", "", false)
	if err != nil {
		return nil, fmt.Errorf("newtreemlp: could not construct root "+
			"network: %v", err)
	}

	// Create leaf networks and run each of their forward passes
	rootOutput := rootNetwork.Prediction()
	if len(rootOutput) != 1 {
		return nil, fmt.Errorf("clonewithinputo: cannot use root network " +
			" with multiple outputs as a single input to leaf networks")
	}

	leafNetworks := make([]NeuralNet, len(leafHiddenSizes))
	for i := 0; i < len(leafHiddenSizes); i++ {
		prefix := fmt.Sprintf("Leaf%d", i)

		leafNetworks[i], err = newMultiHeadMLPFromInput(rootOutput[0], outputs, g,
			leafHiddenSizes[i], leafBiases[i], init, leafActivations[i],
			prefix, "", true)

		if err != nil {
			return nil, fmt.Errorf("newtreemlp: could not construct leaf "+
				"network %v: %v", i, err)
		}
	}

	net := &treeMLP{
		g:               g,
		rootNetwork:     rootNetwork,
		leafNetworks:    leafNetworks,
		input:           input,
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

	// Compute the forward pass
	_, err = net.fwd(input)
	if err != nil {
		msg := "newmtreemlp: could not compute forward pass: %v"
		return &treeMLP{}, fmt.Errorf(msg, err)
	}

	return net, nil
}

// SetInput sets the value of the input node before running the forward
// pass.
func (t *treeMLP) SetInput(input []float64) error {
	if len(input) != t.numInputs*t.batchSize {
		msg := fmt.Sprintf("invalid number of inputs\n\twant(%v)"+
			"\n\thave(%v)", t.numInputs*t.batchSize, len(input))
		panic(msg)
	}
	inputTensor := tensor.New(
		tensor.WithBacking(input),
		tensor.WithShape(t.input.Shape()...),
	)
	return G.Let(t.input, inputTensor)
}

// Set sets the weights of a treeMLP to be equal to the
// weights of another treeMLP
func (dest *treeMLP) Set(source NeuralNet) error {
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

// Polyak compute the polyak average of weights of dest with the weights
// of source and stores these averaged weights as the new weights of
// dest.
func (dest *treeMLP) Polyak(source NeuralNet, tau float64) error {
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

// Outputs returns the number of outputs per leaf network
func (t *treeMLP) Outputs() int {
	return t.numOutputs
}

// OutputLayers returns the number of output layers in the network.
// There is one output layer per leaf network.
func (t *treeMLP) OutputLayers() int {
	return len(t.Prediction())
}

// Graph returns the computational graph of the network
func (t *treeMLP) Graph() *G.ExprGraph {
	return t.g
}

// Features returns the number of input features
func (t *treeMLP) Features() int {
	return t.numInputs
}

// Clone returns a clone of the treeMLP.
func (t *treeMLP) Clone() (NeuralNet, error) {
	return t.CloneWithBatch(t.batchSize)
}

// CloneWithBatch returns a clone of the treeMLP with a new input
// batch size.
func (t *treeMLP) CloneWithBatch(batchSize int) (NeuralNet, error) {
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

	return t.cloneWithInputTo(input, graph)
}

// cloneWithInputTo clones the treeMLP to a new graph with a given
// input node
func (t *treeMLP) cloneWithInputTo(input *G.Node,
	graph *G.ExprGraph) (NeuralNet, error) {
	if graph != input.Graph() {
		return nil, fmt.Errorf("clonewithinputo: input graph and graph " +
			"are not the same computaitonal graph")
	}
	if !input.IsMatrix() {
		return nil, fmt.Errorf("cloneWithInputTo: input must be a matrix node")
	}
	batchSize := input.Shape()[0]
	features := input.Shape()[1]

	rootClone, err := t.rootNetwork.cloneWithInputTo(input, graph)
	if err != nil {
		return nil, fmt.Errorf("clonewithbatch: could not clone root "+
			"network: %v", err)
	}
	rootOutput := rootClone.Prediction()
	if len(rootOutput) != 1 {
		return nil, fmt.Errorf("clonewithinputo: cannot use root network " +
			" with multiple outputs as a single input to leaf networks")
	}

	leafClones := make([]NeuralNet, len(t.leafNetworks))
	for i := 0; i < len(leafClones); i++ {
		leafClones[i], err = t.cloneWithInputTo(rootOutput[0], graph)
		if err != nil {
			msg := "cloneWithInputTo: could not clone leaf network %v: %v"
			return nil, fmt.Errorf(msg, i, err)
		}
	}

	net := &treeMLP{
		g:            graph,
		rootNetwork:  rootClone,
		leafNetworks: leafClones,
		// rootLayers:      rootLayers,
		// leafLayers:      leafLayers,
		input:           input,
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
func (t *treeMLP) BatchSize() int {
	return t.rootNetwork.BatchSize()
}

// fwd computes the remaining steps of the forward pass of the treeMLP
// that its root and leaf networks did not compute.
func (t *treeMLP) fwd(input *G.Node) (*G.Node, error) {
	leafPredictions := make([]*G.Node, 0, len(t.leafNetworks))

	for _, leafNet := range t.leafNetworks {
		leafPredictions = append(leafPredictions, leafNet.Prediction()...)
	}
	t.prediction = leafPredictions

	t.predVal = make([]G.Value, len(t.prediction))
	for i, pred := range t.prediction {
		G.Read(pred, &t.predVal[i])
	}

	return nil, nil //concatPred, nil
}

// Output returns the output of the treeMLP. The output is
// a matrix of NxM dimensions, where N corresponds to the number of
// outputs per leaf network and M the number of leaf networks.
func (t *treeMLP) Output() []G.Value {
	return t.predVal
}

// Prediction returns the node of the computational graph the stores
// the output of the treeMLP
func (t *treeMLP) Prediction() []*G.Node {
	return t.prediction
}

// Model returns the learnable nodes with their gradients.
func (t *treeMLP) Model() []G.ValueGrad {
	// Laxy instantiation of model
	if t.model == nil {
		t.model = t.computeModel()
	}
	return t.model
}

// computeModel gets and returns all learnables of the network with
// their gradients
func (t *treeMLP) computeModel() []G.ValueGrad {
	var model []G.ValueGrad
	for _, learnable := range t.Learnables() {
		model = append(model, learnable)
	}
	return model
}

// Learnables returns the learnable nodes in a multiHeadMLP
func (t *treeMLP) Learnables() G.Nodes {
	// Lazy instantiation of learnables
	if t.learnables == nil {
		t.learnables = t.computeLearnables()
	}
	return t.learnables
}

// computeLearnables gets and returns all learnables of the network
func (t *treeMLP) computeLearnables() G.Nodes {
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
