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
	Activation() Activation
}
