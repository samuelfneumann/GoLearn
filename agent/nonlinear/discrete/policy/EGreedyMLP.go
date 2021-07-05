// Package policy implements policies using function approximation using
// Gorgonia. Many of these policies use nonlinear function
// aprpoximation.
package policy

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"log"
	"math/rand"

	"gonum.org/v1/gonum/mat"
	G "gorgonia.org/gorgonia"

	"sfneuman.com/golearn/agent"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/experiment/checkpointer"
	"sfneuman.com/golearn/network"
	"sfneuman.com/golearn/utils/floatutils"
)

// MultiHeadEGreedyMLP implements an epsilon greedy policy using a
// feedforward neural network/MLP. Given an environment with N actions,
// the neural network will produce N outputs, each predicting the
// value of a distinct action.
//
// MultiHeadEGreedyMLP simply populates a gorgonia.ExprGraph with
// the neural network function approximator and selects actions
// based on the output of this neural network. The struct does not
// have a vm of its own. An external VM should be used to run the
// computational graph of the policy externally. The VM should always
// be run before selecting an action with the policy.
//
// For example, given an observation vector obs, we should first call
// the SetInput() function to set the input to the policy as this
// observation. Then, we can run the VM to get a prediction from the
// policy. The policy will predict N action values given N actions.
// At this point, the SelectAction() function can be called which
// will look through these action values and select one based on the
// policy. The way to get an action from the policy is summarized as:
//
//		Set up VM with policy's graph:	vm = NewVM(policy.Graph())
//		Get state observation vector:	obs
//		Set input to policy's network:	policy.SetInput(obs)
//		Predict the action values:		vm.RunAll()
//		Select an action:				action = policy.SelectAction()
type MultiHeadEGreedyMLP struct {
	network.NeuralNet
	epsilon float64

	rng  *rand.Rand
	seed int64
}

// GobDecode implements the gob.GobDecoder interface
func (m *MultiHeadEGreedyMLP) GobDecode(in []byte) error {
	buf := bytes.NewReader(in)
	dec := gob.NewDecoder(buf)

	err := dec.Decode(&m.NeuralNet)
	if err != nil {
		return fmt.Errorf("gobdecode: could not decode network: %v", err)
	}

	err = dec.Decode(&m.epsilon)
	if err != nil {
		return fmt.Errorf("gobdecode: could not decode epsilon: %v", err)
	}

	err = dec.Decode(&m.rng)
	if err != nil {
		return fmt.Errorf("gobdecode: could not decode rng: %v", err)
	}

	err = dec.Decode(&m.seed)
	if err != nil {
		return fmt.Errorf("gobdecode: could not decode seed: %v", err)
	}

	return nil
}

// GobEncode implements the gob.GobEncoder interface
func (m *MultiHeadEGreedyMLP) GobEncode() ([]byte, error) {
	// ! might have to use reflection here to register neural net type
	// ! although we could just register the type in the neural net GobEncode()

	var buf bytes.Buffer
	enc := gob.NewEncoder(&buf)

	serializableNet, ok := m.NeuralNet.(checkpointer.Serializable)
	if !ok {
		return nil, fmt.Errorf("gobencode: neural network not serializable")
	}

	err := enc.Encode(serializableNet)
	if err != nil {
		return nil, fmt.Errorf("gobencode: could not encode network: %v", err)
	}

	err = enc.Encode(m.epsilon)
	if err != nil {
		return nil, fmt.Errorf("gobencode: could not encode epsilon: %v", err)
	}

	err = enc.Encode(m.rng)
	if err != nil {
		return nil, fmt.Errorf("gobencode: could not encode rng: %v", err)
	}

	err = enc.Encode(m.seed)
	if err != nil {
		return nil, fmt.Errorf("gobencode: could not encode seed: %v", err)
	}

	return buf.Bytes(), nil
}

// NewMultiHeadEGreedyMLP creates and returns a new MultiHeadEGreedyMLP
// The hiddenSizes parameter defines the number of nodes in each hidden
// layer. The biases parameter outlines which layers should include
// bias units. The activations parameter determines the activation
// function for each layer. The batch parameter determines the number
// of inputs in a batch.
//
// Note that this constructor will always add an additional hidden
// layer (with a bias unit and no activation) such that the number of
// network outputs equals the number of actions in the environment.
// That is, regardless of the constructor arguments, an additional,
// final linear layer is added so that the output of the network
// equals the number of environmental actions.
//
//
// Because of this, it is easy to create a linear EGreedy policy by
// setting hiddenSizes to []int{}, biases to []bool{}, and activations
// to []Activation{}.
func NewMultiHeadEGreedyMLP(epsilon float64, env env.Environment,
	batch int, g *G.ExprGraph, hiddenSizes []int, biases []bool,
	init G.InitWFn, activations []*network.Activation,
	seed int64) (agent.EGreedyNNPolicy, error) {
	// Calculate the number of actions and state features
	numActions := int(env.ActionSpec().UpperBound.AtVec(0)) + 1
	features := env.ObservationSpec().Shape.Len()

	net, err := network.NewMultiHeadMLP(features, batch, numActions, g,
		hiddenSizes, biases, init, activations)
	if err != nil {
		return &MultiHeadEGreedyMLP{},
			fmt.Errorf("new: could not create policy: %v", err)
	}
	if predictions := len(net.Prediction()); predictions != 1 {
		msg := "new: egreedy policy expects function approximator to output " +
			"a single prediction node\n\twant(1)\n\thave(%v)"
		return &MultiHeadEGreedyMLP{}, fmt.Errorf(msg, predictions)
	}

	// Create RNG for sampling actions
	source := rand.NewSource(seed)
	rng := rand.New(source)

	// Create the policy
	nn := MultiHeadEGreedyMLP{
		epsilon:   epsilon,
		rng:       rng,
		seed:      seed,
		NeuralNet: net,
	}

	return &nn, nil
}

// Network returns the neural network function approximator that the
// policy uses.
func (e *MultiHeadEGreedyMLP) Network() network.NeuralNet {
	return e.NeuralNet
}

// ClonePolicy clones a MultiHeadEGreedyMLP
func (e *MultiHeadEGreedyMLP) ClonePolicy() (agent.NNPolicy, error) {
	batchSize := e.BatchSize()
	return e.ClonePolicyWithBatch(batchSize)
}

// ClonePolicyWithBatch clones a MultiHeadEGreedyMLP with a new input
// batch size.
func (e *MultiHeadEGreedyMLP) ClonePolicyWithBatch(
	batchSize int) (agent.NNPolicy, error) {
	net, err := e.Network().Clone()
	if err != nil {
		msg := "clonepolicywithbatch: could not clone policy: %v"
		return &MultiHeadEGreedyMLP{}, fmt.Errorf(msg, err)
	}

	// Create RNG for sampling actions
	source := rand.NewSource(e.seed)
	rng := rand.New(source)

	// Create the network and run the forward pass on the input node
	nn := MultiHeadEGreedyMLP{
		epsilon:   e.epsilon,
		rng:       rng,
		seed:      e.seed,
		NeuralNet: net,
	}

	return &nn, nil
}

// SetEpsilon sets the value for epsilon in the epsilon greedy policy.
func (e *MultiHeadEGreedyMLP) SetEpsilon(ε float64) {
	e.epsilon = ε
}

// Epsilon gets the value of epsilon for the policy.
func (e *MultiHeadEGreedyMLP) Epsilon() float64 {
	return e.epsilon
}

// SelectAction selects an action according to the action values
// generated from the last run of the computational graph. This
// funtion returns the action selected as well as the approximated value
// of that action.
func (e *MultiHeadEGreedyMLP) SelectAction() (*mat.VecDense, float64) {
	if e.Output() == nil {
		log.Fatal("vm must be run before selecting an action")
	}

	// Get the action values from the last run of the computational graph
	actionValues := e.Output()[0].Data().([]float64)

	// With probability epsilon return a random action
	if probability := rand.Float64(); probability < e.epsilon {
		action := rand.Int() % e.numActions()
		return mat.NewVecDense(1, []float64{float64(action)}),
			actionValues[action]
	}

	// Get the actions of maximum value
	_, maxIndices := floatutils.MaxSlice(actionValues)

	// If multiple actions have max value, return a random max-valued action
	action := maxIndices[e.rng.Int()%len(maxIndices)]
	return mat.NewVecDense(1, []float64{float64(action)}),
		actionValues[action]
}

// numActions returns the number of actions that the policy chooses
// between.
func (e *MultiHeadEGreedyMLP) numActions() int {
	return e.Outputs()
}
