package network

import (
	"bytes"
	"encoding/gob"
	"fmt"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// fcLayer implements a fully connected layer of a feed forward neural
// network
type fcLayer struct {
	weights *G.Node
	bias    *G.Node
	act     *Activation
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
	if act := f.Activation(); act.IsIdentity() || act.IsNil() {
		return x, nil
	}
	return f.Activation().fwd(x)
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

// Activation returns the activation of the layer
func (f *fcLayer) Activation() *Activation {
	return f.act
}

// Bias returns the bias to the layer
func (f *fcLayer) Bias() *G.Node {
	return f.bias
}

// Weights returns the weights of the layer
func (f *fcLayer) Weights() *G.Node {
	return f.weights
}

// GobEncode implements the gob.GobEncoder interface.
//
// Since fcLayer uses Gorgonia Nodes, only the Node's values can be
// saved. Therefore, gobbing an fcLayer is like gobbing the Values of
// its weights and biases.
func (f *fcLayer) GobEncode() ([]byte, error) {
	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	err := enc.Encode(f.Weights().Value())
	if err != nil {
		return nil, fmt.Errorf("gobencode: could not encode weights: %v", err)
	}

	err = enc.Encode(f.Bias().Value())
	if err != nil {
		return nil, fmt.Errorf("gobencode: could not encode bias: %v", err)
	}

	err = enc.Encode(f.Activation())
	if err != nil {
		return nil, fmt.Errorf("gobencode: could not encode activation: %v",
			err)
	}

	return buf.Bytes(), nil
}

// GobDecode implements the gob.GobDecoder interface.
//
// This function will fill the weights and bias of an initialized
// fcLayer with the values saved in a binary file of a previously
// encoded fcLayer. Note that the shape of the weights and bias are also
// required to be the same between the new and old fcLayers. The old
// fcLayer's activation is copied to the new fcLayer's activation.
//
// This function is somewhat equivalent to:
//		newLayer := new fcLayer
//		oldLayer := read old fcLayer from file
//		gorgonia.Let(newLayer.Weights, oldLayer.Weights)
//		gorgonia.Let(newLayer.Bias, oldLayer.Bias)
//		newLayer.activation = oldLayer.activation
//
// Note that it is important that the fcLayer has already been
// initialized and its weights and biases have the same shape as the
// those of the serialized fcLayer. Otherwise, if the fcLayer has not
// been initialized, a null pointer error may be encountered due to
// it weights and bias *Node having no backing Node.
func (f *fcLayer) GobDecode(in []byte) error {
	if f.Weights() == nil || f.Bias() == nil {
		return fmt.Errorf("gobdecode: fcLayer must have all node pointers " +
			"initialized and registered with a graph before decoding")
	}

	buf := bytes.NewReader(in)
	dec := gob.NewDecoder(buf)

	var weights *tensor.Dense
	err := dec.Decode(&weights)
	if err != nil {
		return fmt.Errorf("gobdecode: could not decode weights: %v", err)
	}
	err = G.Let(f.Weights(), weights)
	if err != nil {
		return fmt.Errorf("gobdecode: could not set weights %v", err)
	}

	var bias *tensor.Dense
	err = dec.Decode(&bias)
	if err != nil {
		return fmt.Errorf("gobdecode: could not decode bias: %v", err)
	}
	err = G.Let(f.Bias(), bias)
	if err != nil {
		return fmt.Errorf("gobdecode: could not set bias: %v", err)
	}

	err = dec.Decode(f.Activation())
	if err != nil {
		return fmt.Errorf("gobdecode: could not decode weights: %v", err)
	}

	return nil
}

// addfcLayers adds fcLayers to a computational graph and returns
// a slice of the fcLayers which were added ot the graph. For integer
// i, hiddenSizes[i] determines the number of hidden units in hidden
// layer i, biases[i] determines if a bias unit is added to layer i,
// and activations[i] determines the activation function to use on
// layer i. For the returned slice, the ith index will hold the fcLayer
// generated by hiddenSizes[i], biases[i], and activations[i] and is
// the ith layer in the feedforward network.
//
// The parameter features determines the number of input
// features to use for the first hidden layer added. That is, the
// first set of weights added to the graph will have shape
// (features, hiddenSizes[0]) while consecutive layers will have weights
// with shape (hiddenSizes[i-1], hiddenSizes[i]).
//
// The parameters prefix and suffix refer to the prefix and suffix to
// add to the names of the weights and biases of the fcLayer.
//
//
// Note that this function only adds fcLayers to the graph g. It does
// not perform the forward pass or outline any relationships between
// nodes. To generate the forward pass given the the returned slice R,
// simply call:
//		input := new matrix input
//		prediction, _ := R[0].fwd(input)
//		for layer, _ := range R[1:] {
// 			prediction = layer.fwd(prediction)
// 		}
//		return prediction
func addfcLayers(g *G.ExprGraph, hiddenSizes []int, biases []bool,
	activations []*Activation, init G.InitWFn, features int,
	prefix, suffix string) []Layer {
	// Create the fully connected layers
	layers := make([]Layer, 0, len(hiddenSizes))
	for i := range hiddenSizes {
		// Create the weights for the layer
		var Weights *G.Node
		if i == 0 {
			// First layer
			weightName := fmt.Sprintf("%vL%dW%v", prefix, i, suffix)
			Weights = G.NewMatrix(
				g,
				tensor.Float64,
				G.WithShape(features, hiddenSizes[i]),
				G.WithName(weightName),
				G.WithInit(init),
			)
		} else {
			// Layers other than the first
			weightName := fmt.Sprintf("%vL%dW%v", prefix, i, suffix)
			Weights = G.NewMatrix(
				g,
				tensor.Float64,
				G.WithShape(hiddenSizes[i-1], hiddenSizes[i]),
				G.WithName(weightName),
				G.WithInit(init),
			)

		}

		// Create the bias unit for the layer
		var Bias *G.Node
		if biases[i] {
			biasName := fmt.Sprintf("%vL%dB%v", prefix, i, suffix)
			Bias = G.NewVector(
				g,
				tensor.Float64,
				G.WithShape(hiddenSizes[i]),
				G.WithName(biasName),
				G.WithInit(init),
			)
		}

		// Create the fully connected layer
		layer := &fcLayer{
			weights: Weights,
			bias:    Bias,
			act:     activations[i],
		}
		layers = append(layers, layer)
	}
	return layers
}
