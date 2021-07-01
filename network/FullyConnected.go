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

// GobEncode implements the GobEncoder interface
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

// GobDecode implements the GobDecoder interface
func (f *fcLayer) GobDecode(in []byte) error {
	buf := bytes.NewBuffer(in)
	dec := gob.NewDecoder(buf)

	var weights *tensor.Dense
	err := dec.Decode(&weights)
	if err != nil {
		return fmt.Errorf("gobdecode: could not decode weights: %v", err)
	}
	err = G.Let(f.Weights(), weights)
	if err != nil {
		return fmt.Errorf("gobdecode: could not set weights: %v", err)
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
