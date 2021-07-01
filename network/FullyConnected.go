package network

import (
	G "gorgonia.org/gorgonia"
)

// Activation represents an activation function type
type Activation func(x *G.Node) (*G.Node, error)

// fcLayer implements a fully connected layer of a feed forward neural
// network
type fcLayer struct {
	weights *G.Node
	bias    *G.Node
	act     Activation
}

// Fwd adds the forward pass of the fcLayer to the computational graph
func (f *fcLayer) fwd(x *G.Node) (*G.Node, error) {
	if f.Weights() != nil {
		x = G.Must(G.Mul(x, f.Weights()))
	}
	if f.Bias() != nil {
		// Broadcast the bias weights to all samples along the batch
		// dimension
		x = G.Must(G.BroadcastAdd(x, f.Bias(), nil, []byte{0}))
	}
	if f.Activation() == nil {
		return x, nil
	}
	return f.Activation()(x)
}

// CloneTo clones an fcLayer to a new computational graph
func (f *fcLayer) CloneTo(g *G.ExprGraph) Layer {
	var newWeights, newBias *G.Node

	if f.Weights() != nil {
		newWeights = f.Weights().CloneTo(g)
	}
	if f.Bias() != nil {
		newBias = f.Bias().CloneTo(g)
	}

	return &fcLayer{
		weights: newWeights,
		bias:    newBias,
		act:     f.act,
	}
}

func (f *fcLayer) Activation() Activation {
	return f.act
}

func (f *fcLayer) Bias() *G.Node {
	return f.bias
}

func (f *fcLayer) Weights() *G.Node {
	return f.weights
}
