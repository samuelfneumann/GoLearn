package policy

import (
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"

	"gonum.org/v1/gonum/mat"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	env "sfneuman.com/golearn/environment"
)

type MultiHeadEGreedyMLP struct {
	g          *G.ExprGraph
	l          []FCLayer
	input      *G.Node
	epsilon    float64
	numActions int
	numInputs  int

	Prediction *G.Node
	predVal    G.Value

	rng  *rand.Rand
	seed int64
}

// NewEGreedyMLP creates and returns a new EGreedyMLP populated in the graph g
func NewMultiHeadEGreedyMLP(epsilon float64, env env.Environment,
	batch int, g *G.ExprGraph, hiddenSizes []int, biases []bool,
	init G.InitWFn, activations []Activation,
	seed int64) (*MultiHeadEGreedyMLP, error) {
	// Ensure we have one activation per layer
	if len(hiddenSizes) != len(activations) {
		msg := "newegreedymlp: invalid number of activations\n\twant(%d)" +
			"\n\thave(%d)"
		return nil, fmt.Errorf(msg, len(hiddenSizes), len(activations))
	}

	// Calculate the number of actions and state features
	numActions := int(env.ActionSpec().UpperBound.AtVec(0)) + 1
	features := env.ObservationSpec().Shape.Len()

	// Set up the input node
	input := G.NewMatrix(g, tensor.Float64, G.WithShape(batch, features),
		G.WithName("input"), G.WithInit(G.Zeroes()))

	// If no given hidden layers, then use a single linear layer so that
	// the output has numActions heads
	if len(hiddenSizes) == 0 {
		hiddenSizes = []int{numActions}
		biases = []bool{true}
		activations = []Activation{nil}
	} else {
		// Append the number of actions to hiddenSizes so that the MLP has
		// numActions output heads
		hiddenSizes = append(hiddenSizes, numActions)
		biases = append(biases, false)
		activations = append(activations, nil)
	}

	layers := make([]FCLayer, 0, len(hiddenSizes))
	for i := range hiddenSizes {
		var Weights *G.Node
		if i == 0 {
			// Create the weights
			weightName := fmt.Sprintf("L%dW", i)
			Weights = G.NewMatrix(
				g,
				tensor.Float64,
				G.WithShape(features, hiddenSizes[i]),
				G.WithName(weightName),
				G.WithInit(init),
			)
		} else {
			weightName := fmt.Sprintf("L%dW", i)
			Weights = G.NewMatrix(
				g,
				tensor.Float64,
				G.WithShape(hiddenSizes[i-1], hiddenSizes[i]),
				G.WithName(weightName),
				G.WithInit(init),
			)

		}

		// Create the bias unit if applicable
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

		// Create the layer
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

	network := MultiHeadEGreedyMLP{
		g:          g,
		l:          layers,
		input:      input,
		epsilon:    epsilon,
		numActions: numActions,
		numInputs:  features,
		rng:        rng,
		seed:       seed,
	}

	_, err := network.fwd(input)
	if err != nil {
		log.Fatal(err)
	}

	return &network, nil
}

func (e *MultiHeadEGreedyMLP) Graph() *G.ExprGraph {
	return e.g
}

func (e *MultiHeadEGreedyMLP) Clone() (*MultiHeadEGreedyMLP, error) {
	graph := G.NewGraph()
	// Copy fully connected layers
	l := make([]FCLayer, len(e.l))
	for i := range e.l {
		l[i] = e.l[i].CloneTo(graph)
	}

	inputShape := e.input.Shape()
	var input *G.Node
	if e.input.IsVec() {

		input = G.NewVector(graph, tensor.Float64, G.WithShape(inputShape...), G.WithName("input"), G.WithInit(G.Zeroes()))
	} else if e.input.IsMatrix() {
		input = G.NewMatrix(graph, tensor.Float64, G.WithShape(inputShape...), G.WithName("input"), G.WithInit(G.Zeroes()))
	} else {
		panic("clone: invalid input type")
	}

	source := rand.NewSource(e.seed)
	rng := rand.New(source)

	network := MultiHeadEGreedyMLP{
		g:          graph,
		l:          l,
		input:      input,
		epsilon:    e.epsilon,
		numActions: e.numActions,
		numInputs:  e.numInputs,
		rng:        rng,
		seed:       e.seed,
	}

	_, err := network.fwd(input)
	if err != nil {
		log.Fatal(err)
	}
	ioutil.WriteFile("clone2.dot", []byte(graph.ToDot()), 0644)

	return &network, nil
}

func (e *MultiHeadEGreedyMLP) SetEpsilon(ε float64) {
	e.epsilon = ε
}

func (e *MultiHeadEGreedyMLP) Epsilon() float64 {
	return e.epsilon
}

func (e *MultiHeadEGreedyMLP) SetInput(input []float64) error {
	if len(input)%e.numInputs != 0 {
		msg := fmt.Sprintf("invalid number of inputs\n\twant(%v)\n\thave(%v)",
			e.numInputs, len(input))
		panic(msg)
	}
	inputTensor := tensor.New(tensor.WithBacking(input), tensor.WithShape(e.input.Shape()...))
	return G.Let(e.input, inputTensor)
}

// Assumes that the vm containing the policy has already been run
func (e *MultiHeadEGreedyMLP) SelectAction() mat.Vector {
	// fmt.Println(e.l[1].Weights.Value())
	if e.predVal == nil {
		log.Fatal("vm must be run before selecting an action")
	}
	actionValues := e.predVal.Data().([]float64)
	probability := rand.Float64()

	// With probability epsilon return a random action
	if probability < e.epsilon {
		action := float64(rand.Int() % e.numActions)
		return mat.NewVecDense(1, []float64{action})
	}

	// Return the max value action
	maxValue, maxInd := actionValues[0], []int{0}
	for i, val := range actionValues {
		if val > maxValue {
			maxValue = val
			maxInd = []int{i}
		} else if val == maxValue {
			maxInd = append(maxInd, i)
		}
	}

	// If multiple actions have max value, return a random max-valued action
	if len(maxInd) > 1 {
		swap := func(i, j int) { maxInd[i], maxInd[j] = maxInd[j], maxInd[i] }
		rand.Shuffle(len(maxInd), swap)
	}
	return mat.NewVecDense(1, []float64{float64(maxInd[0])})
}

func (e *MultiHeadEGreedyMLP) Set(source *MultiHeadEGreedyMLP) error {
	sourceNodes := source.Learnables()
	nodes := e.Learnables()
	for i, destLearnable := range nodes {
		sourceLearnable := sourceNodes[i].Clone()
		err := G.Let(destLearnable, sourceLearnable.(*G.Node).Value())
		if err != nil {
			return err
		}
	}
	return nil
}

func (e *MultiHeadEGreedyMLP) Polyak(source *MultiHeadEGreedyMLP, tau float64) error {
	sourceNodes := source.Learnables()
	nodes := e.Learnables()
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

func (e *MultiHeadEGreedyMLP) Learnables() G.Nodes {
	learnables := make([]*G.Node, 0, 2*len(e.l))
	for i := range e.l {
		learnables = append(learnables, e.l[i].Weights)
		if bias := e.l[i].Bias; bias != nil {
			learnables = append(learnables, bias)
		}
	}
	return G.Nodes(learnables)
}

func (e *MultiHeadEGreedyMLP) Model() []G.ValueGrad {
	var model []G.ValueGrad = make([]G.ValueGrad, 0, 2*len(e.l))
	for i := range e.l {
		model = append(model, e.l[i].Weights)
		if bias := e.l[i].Bias; bias != nil {
			model = append(model, bias)
		}
	}
	return model
}

// Fwd performs the forward pass of the neural net on the input node
func (e *MultiHeadEGreedyMLP) fwd(input *G.Node) (*G.Node, error) {
	inputShape := input.Shape()[len(input.Shape())-1]
	if inputShape%e.numInputs != 0 {
		return nil, fmt.Errorf("invalid shape for input to neural net:"+
			" \n\twant(%v) \n\thave(%v)", e.numInputs, inputShape)
	}

	pred := input
	var err error
	for _, l := range e.l {
		if pred, err = l.Fwd(pred); err != nil {
			log.Fatal(err)
		}
	}
	e.Prediction = pred
	G.Read(e.Prediction, &e.predVal)

	return pred, nil
}

func (e *MultiHeadEGreedyMLP) Output() G.Value {
	return e.predVal
}
