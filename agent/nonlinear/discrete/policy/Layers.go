package policy

import (
	G "gorgonia.org/gorgonia"
)

// Activation represents an activation function type
type Activation func(x *G.Node) (*G.Node, error)

// FCLayer implements a fully connected layer of a feed forward neural
// network
type FCLayer struct {
	Weights *G.Node
	Bias    *G.Node
	Act     Activation
}

// Fwd adds the forward pass of the FCLayer to the computational graph
func (f *FCLayer) Fwd(x *G.Node) (*G.Node, error) {
	if f.Weights != nil {
		x = G.Must(G.Mul(x, f.Weights))
	}
	if f.Bias != nil {
		// Broadcast the bias weights to all samples along the batch
		// dimension
		x = G.Must(G.BroadcastAdd(x, f.Bias, nil, []byte{0}))
	}
	if f.Act == nil {
		return x, nil
	}
	return f.Act(x)
}

// CloneTo clones an FCLayer to a new computational graph
func (f *FCLayer) CloneTo(g *G.ExprGraph) FCLayer {
	var newWeights, newBias *G.Node

	if f.Weights != nil {
		newWeights = f.Weights.CloneTo(g)
	}
	if f.Bias != nil {
		newBias = f.Bias.CloneTo(g)
	}

	return FCLayer{
		Weights: newWeights,
		Bias:    newBias,
		Act:     f.Act,
	}
}
