// Package agent defines an agent interface
package agent

import (
	"github.com/samuelfneumann/golearn/network"
	"github.com/samuelfneumann/golearn/timestep"
	"gonum.org/v1/gonum/mat"
	G "gorgonia.org/gorgonia"
)

// Agent determines the implementation details of an agent or algorithm
//
// An Agent is composed of a Learner, which learns weights, and a Policy
// which chooses actions in each state. The Policy chooses which actions
// are taken, and the Learner uses these actions to update the Policy.
type Agent interface {
	Learner
	Policy
}

// A Closer is an agent that must be closed after it is done learning
type Closer interface {
	Agent
	Close() error
}

// Learner implements a learning algorithm that defines how weights are
// updated.
type Learner interface {
	// Step performs a single update to the learner
	Step() error

	// Observe records that an action lead to some timestep
	Observe(action mat.Vector, nextObs timestep.TimeStep) error

	// ObserveFirst records the first timestep in an episode
	ObserveFirst(timestep.TimeStep) error

	// EndEpisode performs cleanup at the end of an episode
	EndEpisode()
}

// TdErrorer is a Learner that can return the TdError of some transition
type TdErrorer interface {
	Learner

	// TdError returns the TD error on a transition
	TdError(t timestep.Transition) float64
}

// Policy represents a policy that an agent can have.
//
// Policies determine how agents select actions. Agents usually have a
// target and behaviour policy. For a given agent, the Policy and Learner
// should have pointers to the same weights so that any changes the learner
// makes to the weights are reflected in the actions the Policy chooses
type Policy interface {
	SelectAction(t timestep.TimeStep) *mat.VecDense
	Eval()        // Set policy to evaluation mode
	Train()       // Set policy to training mode
	IsEval() bool // Indicates if in evaluation mode
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
	Clone() (NNPolicy, error)
	CloneWithBatch(int) (NNPolicy, error)
	Network() network.NeuralNet
	Close() error
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

// LogPdfOfer implements a policy type that can calculate the log
// of the probability density function of the policy for taking some
// (externally inputted) action in some (externally inputted) state.
// Because of this, the gradient will not be computed through the
// action selection process.
type LogPdfOfer interface {
	NNPolicy

	// LogPdfNode returns the node that calculates the log probability
	// of the policy's selected actions
	LogPdfNode() *G.Node

	// LogPdfVal returns the value of the node returned by
	// LogProbNode()
	LogPdfVal() G.Value

	// LogPdfOf returns the log probability of taking the argument
	// actions in the argument states. Inputs should be constructed in
	// row major order.
	LogPdfOf(states, actions []float64) (*G.Node, error)
}
