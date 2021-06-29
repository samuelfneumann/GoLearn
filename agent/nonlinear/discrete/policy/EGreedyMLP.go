// Package policy implements policies using function approximation using
// Gorgonia. Many of these policies use nonlinear function
// aprpoximation.
package policy

import (
	"fmt"
	"log"
	"math/rand"

	"gonum.org/v1/gonum/mat"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	"sfneuman.com/golearn/agent"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/utils/floatutils"
)

// MultiHeadEGreedyMLP implements an epsilon greedy policy using a
// feedforward neural network/MLP. Given an environment with N actions,
// the neural network will produce N outputs, ach predicting the
// value of a distinct action.
//
// MultiHeadEGreedyMLP simply populates a gorgonia.ExprGraph with
// the neural network function approximator and selects actions
// based on the output of this neural network. The struct does not
// have a vm of its own. An external G.VM should be used to run the
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
	g          *G.ExprGraph
	layers     []FCLayer
	input      *G.Node
	epsilon    float64
	numActions int
	numInputs  int
	batchSize  int

	prediction *G.Node
	predVal    G.Value

	rng  *rand.Rand
	seed int64
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
	init G.InitWFn, activations []Activation,
	seed int64) (agent.EGreedyNNPolicy, error) {
	// Ensure we have one activation per layer
	if len(hiddenSizes) != len(activations) {
		msg := "newmultiheadegreedymlp: invalid number of activations" +
			"\n\twant(%d)\n\thave(%d)"
		return nil, fmt.Errorf(msg, len(hiddenSizes), len(activations))
	}

	// Ensure one bias bool per layer
	if len(hiddenSizes) != len(biases) {
		msg := "newmultiheadegreedymlp: invalid number of biases\n\twant(%d)" +
			"\n\thave(%d)"
		return nil, fmt.Errorf(msg, len(hiddenSizes), len(biases))
	}

	// Calculate the number of actions and state features
	numActions := int(env.ActionSpec().UpperBound.AtVec(0)) + 1
	features := env.ObservationSpec().Shape.Len()

	// Set up the input node
	input := G.NewMatrix(g, tensor.Float64, G.WithShape(batch, features),
		G.WithName("input"), G.WithInit(G.Zeroes()))

	// If no given hidden layers, then use a single linear layer so that
	// the output has numActions heads
	hiddenSizes = append(hiddenSizes, numActions)
	biases = append(biases, true)
	activations = append(activations, nil)

	// Create the fully connected layers
	layers := make([]FCLayer, 0, len(hiddenSizes))
	for i := range hiddenSizes {
		// Create the weights for the layer
		var Weights *G.Node
		if i == 0 {
			// First layer
			weightName := fmt.Sprintf("L%dW", i)
			Weights = G.NewMatrix(
				g,
				tensor.Float64,
				G.WithShape(features, hiddenSizes[i]),
				G.WithName(weightName),
				G.WithInit(init),
			)
		} else {
			// Layers other than the first
			weightName := fmt.Sprintf("L%dW", i)
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
			biasName := fmt.Sprintf("L%dB", i)
			Bias = G.NewVector(
				g,
				tensor.Float64,
				G.WithShape(hiddenSizes[i]),
				G.WithName(biasName),
				G.WithInit(init),
			)
		}

		// Create the fully connected layer
		layer := FCLayer{
			Weights: Weights,
			Bias:    Bias,
			Act:     activations[i],
		}
		layers = append(layers, layer)
	}

	// Create RNG for sampling actions
	source := rand.NewSource(seed)
	rng := rand.New(source)

	// Create the network and run the forward pass on the input node
	network := MultiHeadEGreedyMLP{
		g:          g,
		layers:     layers,
		input:      input,
		epsilon:    epsilon,
		numActions: numActions,
		numInputs:  features,
		batchSize:  batch,
		rng:        rng,
		seed:       seed,
	}
	_, err := network.fwd(input)
	if err != nil {
		log.Fatal(err)
	}

	return &network, nil
}

// Graph returns the computational graph of the MultiHeadEGreedyMLP.
func (e *MultiHeadEGreedyMLP) Graph() *G.ExprGraph {
	return e.g
}

// Clone clones a MultiHeadEGreedyMLP
func (e *MultiHeadEGreedyMLP) Clone() (agent.NNPolicy, error) {
	batchSize := e.input.Shape()[0]
	return e.CloneWithBatch(batchSize)
}

// CloneWithBatch clones a MultiHeadEGreedyMLP with a new input batch
// size.
func (e *MultiHeadEGreedyMLP) CloneWithBatch(
	batchSize int) (agent.NNPolicy, error) {
	graph := G.NewGraph()

	// Copy fully connected layers
	l := make([]FCLayer, len(e.layers))
	for i := range e.layers {
		l[i] = e.layers[i].CloneTo(graph)
	}

	// Create the input node
	inputShape := e.input.Shape()
	var input *G.Node
	if e.input.IsMatrix() {
		batchShape := append([]int{batchSize}, inputShape[1:]...)
		input = G.NewMatrix(
			graph,
			tensor.Float64,
			G.WithShape(batchShape...),
			G.WithName("input"),
			G.WithInit(G.Zeroes()),
		)
	} else {
		panic("clone: invalid input type")
	}

	// Create RNG for sampling actions
	source := rand.NewSource(e.seed)
	rng := rand.New(source)

	// Create the network and run the forward pass on the input node
	network := MultiHeadEGreedyMLP{
		g:          graph,
		layers:     l,
		input:      input,
		epsilon:    e.epsilon,
		numActions: e.numActions,
		numInputs:  e.numInputs,
		batchSize:  batchSize,
		rng:        rng,
		seed:       e.seed,
	}
	_, err := network.fwd(input)
	if err != nil {
		log.Fatal(err)
	}

	return &network, nil
}

// SetEpsilon sets the value for epsilon in the epsilon greedy policy.
func (e *MultiHeadEGreedyMLP) SetEpsilon(ε float64) {
	e.epsilon = ε
}

// Epsilon gets the value of epsilon for the policy.
func (e *MultiHeadEGreedyMLP) Epsilon() float64 {
	return e.epsilon
}

// BatchSize returns the batch size of inputs to the policy
func (e *MultiHeadEGreedyMLP) BatchSize() int {
	return e.batchSize
}

// Features returns the number of features in a single observation
// vector that the policy takes as input.
func (e *MultiHeadEGreedyMLP) Features() int {
	return e.numInputs
}

// SetInput sets the value of the input node before running the forward
// pass.
func (e *MultiHeadEGreedyMLP) SetInput(input []float64) error {
	if len(input) != e.numInputs*e.batchSize {
		msg := fmt.Sprintf("invalid number of inputs\n\twant(%v)"+
			"\n\thave(%v)", e.numInputs*e.batchSize, len(input))
		panic(msg)
	}
	inputTensor := tensor.New(
		tensor.WithBacking(input),
		tensor.WithShape(e.input.Shape()...),
	)
	return G.Let(e.input, inputTensor)
}

// SelectAction selects an action according to the action values
// generated from the last run of the computational graph. This
// funtion returns the action selected as well as the approximated value
// of that action.
func (e *MultiHeadEGreedyMLP) SelectAction() (*mat.VecDense, float64) {
	if e.predVal == nil {
		log.Fatal("vm must be run before selecting an action")
	}

	// Get the action values from the last run of the computational graph
	actionValues := e.predVal.Data().([]float64)

	// With probability epsilon return a random action
	if probability := rand.Float64(); probability < e.epsilon {
		action := rand.Int() % e.numActions
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

// Set sets the weights of a MultiHeadEGreedyMLP to be equal to the
// weights of another MultiHeadEGreedyMLP
func (dest *MultiHeadEGreedyMLP) Set(source agent.NNPolicy) error {
	sourceNodes := source.Learnables()
	nodes := dest.Learnables()
	for i, destLearnable := range nodes {
		sourceLearnable := sourceNodes[i].Clone()
		err := G.Let(destLearnable, sourceLearnable.(*G.Node).Value())
		if err != nil {
			return err
		}
	}
	return nil
}

// Polyak sets the weights of a MultiHeadEGreedyMLP to be a polyak
// average between its existing weights and the weights of another
// MultiHeadEGreedyMLP
func (dest *MultiHeadEGreedyMLP) Polyak(source agent.NNPolicy,
	tau float64) error {
	sourceNodes := source.Learnables()
	nodes := dest.Learnables()
	for i := range nodes {
		weights := nodes[i].Value().(*tensor.Dense)
		sourceWeights := sourceNodes[i].Value().(*tensor.Dense)

		weights, err := weights.MulScalar(1-tau, true)
		if err != nil {
			return err
		}

		sourceWeights, err = sourceWeights.MulScalar(tau, true)
		if err != nil {
			return err
		}

		var newWeights *tensor.Dense
		newWeights, err = weights.Add(sourceWeights)
		if err != nil {
			return err
		}

		G.Let(nodes[i], newWeights)
	}
	return nil
}

// Learnables returns the learnable nodes in a MultiHeadEGreedyMLP
func (e *MultiHeadEGreedyMLP) Learnables() G.Nodes {
	learnables := make([]*G.Node, 0, 2*len(e.layers))

	for i := range e.layers {
		learnables = append(learnables, e.layers[i].Weights)
		if bias := e.layers[i].Bias; bias != nil {
			learnables = append(learnables, bias)
		}
	}
	return G.Nodes(learnables)
}

// Model returns the learnables nodes with their gradients.
func (e *MultiHeadEGreedyMLP) Model() []G.ValueGrad {
	var model []G.ValueGrad = make([]G.ValueGrad, 0, 2*len(e.layers))

	for i := range e.layers {
		model = append(model, e.layers[i].Weights)
		if bias := e.layers[i].Bias; bias != nil {
			model = append(model, bias)
		}
	}
	return model
}

// Fwd performs the forward pass of the MultiHeadEGreedyMLP on the input
// node
func (e *MultiHeadEGreedyMLP) fwd(input *G.Node) (*G.Node, error) {
	inputShape := input.Shape()[len(input.Shape())-1]
	if inputShape%e.numInputs != 0 {
		return nil, fmt.Errorf("invalid shape for input to neural net:"+
			" \n\twant(%v) \n\thave(%v)", e.numInputs, inputShape)
	}

	pred := input
	var err error
	for _, l := range e.layers {
		if pred, err = l.Fwd(pred); err != nil {
			log.Fatal(err)
		}
	}
	e.prediction = pred
	G.Read(e.prediction, &e.predVal)

	return pred, nil
}

// Output returns the output of the MultiHeadEGreedyMLP. The output is
// a vector of N dimensions, where each dimension corresponds to an
// environmental action.
func (e *MultiHeadEGreedyMLP) Output() G.Value {
	return e.predVal
}

// Prediction returns the node of the computational graph the stores
// the output of the MultiHeadEGreedyMLP
func (e *MultiHeadEGreedyMLP) Prediction() *G.Node {
	return e.prediction
}
