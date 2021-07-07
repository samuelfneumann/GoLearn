// Package agent defines an agent interface
package agent

import (
	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/network"
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

// NNPolicy represents a policy that uses neural network function
// approximation.
//
// Policies implemented by neural networks satsify a different interface
// from Policy. This is because a VM is needed to run the policy, but
// the same VM is needed for both the policy and the Learner so that
// the weights are updated for each.
type NNPolicy interface {
	Policy
	// network.NeuralNet
	// SelectAction() *mat.VecDense
	ClonePolicy() (NNPolicy, error)
	// ClonePolicyWithBatch(int) (NNPolicy, error)
	Network() network.NeuralNet
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

type PolicyLogProber interface {
	NNPolicy
	LogProbability(action []float64) (float64, error)
}
