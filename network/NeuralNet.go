package network

import (
	G "gorgonia.org/gorgonia"
)

type NeuralNet interface {
	Graph() *G.ExprGraph
	Clone() (NeuralNet, error)
	CloneWithBatch(int) (NeuralNet, error)
	BatchSize() int
	Features() int
	SetInput([]float64) error
	Set(NeuralNet) error
	Polyak(NeuralNet, float64) error
	Learnables() G.Nodes
	Model() []G.ValueGrad
	Fwd(*G.Node) (*G.Node, error)
	Output() G.Value
	Prediction() *G.Node
}
