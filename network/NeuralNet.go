package network

import (
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// NeuralNet implements a neural network which can be used in policy
// parameterizations or value functions/critics.
type NeuralNet interface {
	// Clone clones the NeuralNet to a new graph
	Clone() (NeuralNet, error)

	// CloneWithBatch clones the NeuralNet with a new input batch size
	// to a new graph.
	CloneWithBatch(int) (NeuralNet, error)

	// Getter methods
	Graph() *G.ExprGraph
	BatchSize() int
	Features() []int
	Outputs() []int        // Number of outputs per output layer
	OutputLayers() int     // Layers that will output Outputs() values
	Output() []G.Value     // Returns the predictions of the network
	Prediction() []*G.Node // Returns the nodes that hold the predictions

	// Polyak computes the polyak average of the receiver's weights
	// with another networks weights and saves this average as the
	// new weights of the reciever.
	// Polyak(NeuralNet, float64) error
	// Set(NeuralNet) error          // Sets the weights to those of another network

	// Learnables returns the nodes of the network that can be learned
	Learnables() G.Nodes

	// Model returns the nodes of the network that can be learned and
	// their gradients
	Model() []G.ValueGrad

	SetInput([]float64) error     // Sets the input to the network
	fwd(*G.Node) (*G.Node, error) // Performs the forward pass)

	// cloneWithInputTo clones a NeuralNet, setting its input node as
	// input and cloning the network to a given computational graph g.
	cloneWithInputTo(axis int, input []*G.Node,
		graph *G.ExprGraph) (NeuralNet, error)
}

// Layer implements a single layer of a NeuralNet. This could be a
// fully connected layer, a convolutional layer, etc.
type Layer interface {
	fwd(*G.Node) (*G.Node, error)
	CloneTo(g *G.ExprGraph) Layer

	Weights() *G.Node
	Bias() *G.Node
	Activation() *Activation
}

// Set sets the weights of a dest to be equal to the weights of source
func Set(dest, source NeuralNet) error {
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
func Polyak(dest, source NeuralNet, tau float64) error {
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
