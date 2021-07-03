package network

import (
	G "gorgonia.org/gorgonia"
)

// NeuralNet implements a neural network which can be used in policy
// parameterizations or value functions/critics.
type NeuralNet interface {
	Graph() *G.ExprGraph
	Clone() (NeuralNet, error)
	CloneWithBatch(int) (NeuralNet, error)
	BatchSize() int
	Features() int
	Outputs() int
	SetInput([]float64) error
	Set(NeuralNet) error
	Polyak(NeuralNet, float64) error
	Learnables() G.Nodes
	Model() []G.ValueGrad

	fwd(*G.Node) (*G.Node, error)
	cloneWithInputTo(input *G.Node, graph *G.ExprGraph) (NeuralNet, error)

	// To keep networks consistent, there should be a single output value
	// and prediction node. If there are multiple outputs that are not
	// part of the same gorgonia.Node (e.g. the treeMLP), then these
	// outputs should be concatenated into a single output value and
	// prediction node.
	//
	// This pattern keeps all neural network outputs consistent. Using
	// slices of outputs would cause algorithms to have to deal with
	// differing number of output networks differently.
	Output() G.Value
	Prediction() *G.Node
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
