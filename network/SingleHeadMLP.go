package network

import G "gorgonia.org/gorgonia"

// NewSingleHeadMLP returns an MLP with a single output node. This
// function is a convenience function for calling NewMultiHeadMLP with
// an output size of 1.
//
// See NewMultiHeadMLP for more details.
func NewSingleHeadMLP(features, batch int, g *G.ExprGraph, hiddenSizes []int,
	biases []bool, init G.InitWFn, activations []*Activation) (NeuralNet, error) {
	return NewMultiHeadMLP(features, batch, 1, g, hiddenSizes,
		biases, init, activations)
}
