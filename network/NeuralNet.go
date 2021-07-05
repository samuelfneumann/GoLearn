package network

import (
	G "gorgonia.org/gorgonia"
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
	Features() int
	Outputs() int          // Number of outputs per output layer
	OutputLayers() int     // Layers that will output Outputs() values
	Output() []G.Value     // Returns the predictions of the network
	Prediction() []*G.Node // Returns the nodes that hold the predictions

	// Polyak computes the polyak average of the receiver's weights
	// with another networks weights and saves this average as the
	// new weights of the reciever.
	Polyak(NeuralNet, float64) error

	// Learnables returns the nodes of the network that can be learned
	Learnables() G.Nodes

	// Model returns the nodes of the network that can be learned and
	// their gradients
	Model() []G.ValueGrad

	SetInput([]float64) error     // Sets the input to the network
	Set(NeuralNet) error          // Sets the weights to those of another network
	fwd(*G.Node) (*G.Node, error) // Performs the forward pass)

	// cloneWithInputTo clones a NeuralNet, setting its input node as
	// input and cloning the network to a given computational graph g
	cloneWithInputTo(input *G.Node, graph *G.ExprGraph) (NeuralNet, error)
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
