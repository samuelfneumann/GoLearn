// Package agent defines an agent interface
package agent

import (
	"gonum.org/v1/gonum/mat"
	"gorgonia.org/gorgonia"
	"sfneuman.com/golearn/timestep"
)

// Agent determines the implementation details of an agent or algorithm
//
// An Agent is composed of a Learner, which learns weights, and a Policy
// which chooses actions in each state. The Policy chooses which actions
// are taken, and the Learner uses these actions to update the Policy.
type Agent interface {
	Learner
	Policy
	Eval()  // Set agent to evaluation mode
	Train() // Set agent to training mode
}

// Learner implements a learning algorithm that defines how weights are
// updated.
type Learner interface {
	Step() // Performs an update
	Observe(action mat.Vector, nextObs timestep.TimeStep)
	ObserveFirst(timestep.TimeStep)
	TdError(t timestep.Transition) float64
	EndEpisode() // Cleanup at the end of episode
}

// Policy represents a policy that an agent can have.
//
// Policies determine how agents select actions. Agents usually have a
// target and behaviour policy. For a given agent, the Policy and Learner
// should have pointers to the same weights so that any changes the learner
// makes to the weights are reflected in the actions the Policy chooses
type Policy interface {
	SelectAction(t timestep.TimeStep) *mat.VecDense
}

// type LinearPolicy interface {
// 	Policy
// 	Weights() map[string]*mat.Dense
// 	SetWeights(map[string]*mat.Dense) error
// }

type NNPolicy interface {
	SelectAction() (*mat.VecDense, float64)
	Prediction() *gorgonia.Node
	SetInput([]float64) error
	BatchSize() int
	Features() int
	Graph() *gorgonia.ExprGraph
	Clone() (NNPolicy, error)
	CloneWithBatch(int) (NNPolicy, error)
	Learnables() gorgonia.Nodes
	Model() []gorgonia.ValueGrad
	Set(NNPolicy) error
	Polyak(NNPolicy, float64) error
	Output() gorgonia.Value
}

// EGreedyNNPolicy implements an epsilon greedy policy using neural
// network function approximation. Any neural network can be used to
// approximate the policy (CNN, RNN, MLP, etc.) as long as the epsilon
// value for the epsilon greeyd policy can be set and retrieved.
type EGreedyNNPolicy interface {
	NNPolicy
	SetEpsilon(float64)
	Epsilon() float64
}

// type ContinuousPolicy interface {
// 	NNPolicy
// 	Dims() []int
// }
