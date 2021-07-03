package network

import (
	"fmt"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// TODO: TreeMLP should be un-exported

// TreeMLP
// 					 ╭─ Leaf Network 1
//					 ├─ Leaf Network 2
//					 ├─ ...
// Input -> Root Net ┼─ ...
//					 ├─ ...
//					 ├─ Leaf Network (N - 1)
//					 ╰─ Leaf Network N
//
type TreeMLP struct {
	g            *G.ExprGraph
	rootNetwork  NeuralNet
	leafNetworks []NeuralNet
	input        *G.Node

	numOutputs int // Total outputs = numLeafLayers * numOutputs
	numInputs  int
	batchSize  int

	// Store learnables and model so that they don't need to be computed
	// each time a gradient step is taken
	learnables G.Nodes
	model      []G.ValueGrad

	// Data needed for gobbing
	rootHiddenSizes []int
	rootBiases      []bool
	rootActivations []*Activation
	leafHiddenSizes [][]int
	leafBiases      [][]bool
	leafActivations [][]*Activation

	prediction *G.Node
	predVal    G.Value
}

func validateTreeMLP(rootHiddenSizes []int, rootBiases []bool,
	rootActivations []*Activation, leafHiddenSizes [][]int,
	leafBiases [][]bool, leafActivations [][]*Activation) error {
	// Validate observation/root network
	if len(rootHiddenSizes) != len(rootActivations) {
		msg := "newtreemlp: invalid number of activations" +
			"\n\twant(%d)\n\thave(%d)"
		return fmt.Errorf(msg, len(rootHiddenSizes),
			len(rootActivations))
	}

	if len(rootHiddenSizes) != len(rootBiases) {
		msg := "newtreemlp: invalid number of biases" +
			"\n\twant(%d)\n\thave(%d)"
		return fmt.Errorf(msg, len(rootHiddenSizes), len(rootBiases))
	}

	// Validate number of leaf networks
	if len(leafHiddenSizes) <= 0 {
		msg := "newtreemlp: there must be at least one leaf network specified"
		return fmt.Errorf(msg)
	}

	if len(leafHiddenSizes) != len(leafActivations) {
		msg := "newtreemlp: invalid number of leaf network activations " +
			"\n\twant(%v) \n\thave(%v)"
		return fmt.Errorf(msg, len(leafHiddenSizes), len(leafActivations))
	}

	if len(leafHiddenSizes) != len(leafBiases) {
		msg := "newtreemlp: invalid number of leaf network biases " +
			"\n\twant(%v) \n\thave(%v)"
		return fmt.Errorf(msg, len(leafHiddenSizes), len(leafBiases))
	}

	// Validate architecture of leaf networks
	for i := 0; i < len(leafHiddenSizes); i++ {
		if len(leafHiddenSizes[i]) != len(leafActivations[i]) {
			msg := "newtreemlp: invalid number of activations for leaf " +
				"network %v \n\twant(%v) \n\thave(%v)"
			return fmt.Errorf(msg, i, len(leafHiddenSizes[i]),
				len(leafActivations[i]))
		}

		if len(leafHiddenSizes[i]) != len(leafBiases[i]) {
			msg := "newtreemlp: invalid number of biases for leaf " +
				"network %v \n\twant(%v) \n\thave(%v)"
			return fmt.Errorf(msg, i, len(leafHiddenSizes[i]),
				len(leafBiases[i]))
		}
	}

	return nil
}

// TODO: Doesn't work with batch size 1 and 1 output node!
func TestTreeMLP() {
	t, err := NewTreeMLP(
		3,
		1,
		1,
		G.NewGraph(),
		[]int{5, 5},
		[]bool{true, true},
		[]*Activation{ReLU(), ReLU()},
		[][]int{{3, 2}, {3, 2}},
		[][]bool{{true, true}, {true, true}},
		[][]*Activation{{ReLU(), ReLU()}, {ReLU(), ReLU()}},
		G.GlorotU(1.0),
	)

	if err != nil {
		panic(err)
	}
	fmt.Println("\nRoot Net", t.(*TreeMLP).rootNetwork)

	vm := G.NewTapeMachine(t.Graph())
	t.SetInput([]float64{1, 2, 3})
	vm.RunAll()
	fmt.Println(t.Output())
	vm.Reset()
}

func NewTreeMLP(features, batch, outputs int, g *G.ExprGraph,
	rootHiddenSizes []int, rootBiases []bool, rootActivations []*Activation,
	leafHiddenSizes [][]int, leafBiases [][]bool,
	leafActivations [][]*Activation, init G.InitWFn) (NeuralNet, error) {

	err := validateTreeMLP(rootHiddenSizes, rootBiases, rootActivations,
		leafHiddenSizes, leafBiases, leafActivations)
	if err != nil {
		return nil, err
	}

	// Set up the input node
	input := G.NewMatrix(g, tensor.Float64, G.WithShape(batch, features),
		G.WithName("input"), G.WithInit(G.Zeroes()))

	// Root/observation network
	observationOutputs := rootHiddenSizes[len(rootHiddenSizes)-1]
	rootNetwork, err := newMultiHeadMLPFromInput(input, observationOutputs, g,
		rootHiddenSizes, rootBiases, init, rootActivations, "Root", "", false)
	if err != nil {
		return nil, fmt.Errorf("newtreemlp: could not construct root "+
			"network: %v", err)
	}

	// Leaf networks
	rootOutput := rootNetwork.Prediction()
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
	}

	net := &TreeMLP{
		g:            g,
		rootNetwork:  rootNetwork,
		leafNetworks: leafNetworks,
		// rootLayers:      rootLayers,
		// leafLayers:      leafLayers,
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

	_, err = net.fwd(input)
	if err != nil {
		msg := "newmtreemlp: could not compute forward pass: %v"
		return &TreeMLP{}, fmt.Errorf(msg, err)
	}

	return net, nil
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
	return G.Let(t.input, inputTensor)
}

// Set sets the weights of a multiHeadMLP to be equal to the
// weights of another multiHeadMLP
func (dest *TreeMLP) Set(source NeuralNet) error {
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

func (dest *TreeMLP) Polyak(source NeuralNet, tau float64) error {
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

func (t *TreeMLP) Outputs() int {
	return t.numOutputs
}

func (t *TreeMLP) Graph() *G.ExprGraph {
	return t.g
}

func (t *TreeMLP) Features() int {
	return t.numInputs
}

func (t *TreeMLP) Clone() (NeuralNet, error) {
	return t.CloneWithBatch(t.batchSize)
}

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

	return t.cloneWithInputTo(input, graph)
}

func (t *TreeMLP) cloneWithInputTo(input *G.Node, graph *G.ExprGraph) (NeuralNet, error) {
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

	leafClones := make([]NeuralNet, len(t.leafNetworks))
	for i := 0; i < len(leafClones); i++ {
		leafClones[i], err = t.cloneWithInputTo(rootOutput, graph)
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

func (t *TreeMLP) BatchSize() int {
	return t.rootNetwork.BatchSize()
}

func (t *TreeMLP) fwd(input *G.Node) (*G.Node, error) {
	leafPredictions := make([]*G.Node, len(t.leafNetworks))
	// Compute forward pass on each leaf layer
	for i, leafNet := range t.leafNetworks {
		leafPredictions[i] = leafNet.Prediction()
		fmt.Println(leafPredictions[i].Shape())
	}

	for _, l := range t.leafNetworks {
		fmt.Println(l)
	}

	// Concatenate all leaf network outputs
	concatPred, err := G.Concat(0, leafPredictions...)
	if err != nil {
		return nil, fmt.Errorf("fwd: could not concatenate leaf node "+
			"predictions: %v", err)
	}

	t.prediction = concatPred
	G.Read(t.prediction, &t.predVal)

	return concatPred, nil
}

// Output returns the output of the TreeMLP. The output is
// a matrix of NxM dimensions, where N corresponds to the number of
// outputs per leaf network and M the number of leaf networks.
func (t *TreeMLP) Output() G.Value {
	return t.predVal
}

// Prediction returns the node of the computational graph the stores
// the output of the multiHeadMLP
func (t *TreeMLP) Prediction() *G.Node {
	return t.prediction
}

// Model returns the learnables nodes with their gradients.
func (t *TreeMLP) Model() []G.ValueGrad {
	// Laxy instantiation of model
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
