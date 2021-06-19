package main

import (
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"

	"gonum.org/v1/gonum/mat"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type Activation func(x *G.Node) (*G.Node, error)

type FCLayer struct {
	Weights *G.Node
	Bias    *G.Node
	Act     Activation
}

func (f *FCLayer) fwd(x *G.Node) (*G.Node, error) {
	xw := G.Must(G.Mul(f.Weights, x))
	if f.Bias != nil {
		xw = G.Must(G.Add(f.Bias, xw))
	}
	if f.Act == nil {
		return xw, nil
	}
	return f.Act(xw)
}

type EGreedyMLP struct {
	g          *G.ExprGraph
	l          []FCLayer
	input      *G.Node
	epsilon    float64
	numActions int
	numInputs  int

	pred    *G.Node
	predVal G.Value
}

func NewEGreedyMLP(epsilon float64, numActions int, input *G.Node, g *G.ExprGraph) *EGreedyMLP {
	if !input.IsVector() {
		panic("not implemented")
	}
	features := input.Shape()[len(input.Shape())-1]
	l := []FCLayer{
		{Weights: G.NewMatrix(g, tensor.Float64, G.WithShape(features, numActions),
			G.WithName("L0W"), G.WithInit(G.GlorotU(1.0))),
		// Bias: G.NewVector(g, tensor.Float64, G.WithShape(numActions), G.WithName("L0B"), G.WithInit(G.GlorotU(1.0))),
		// Act:  G.Tanh,
		},
	}

	fmt.Println("l:", l)

	network := EGreedyMLP{
		g:          g,
		l:          l,
		input:      input,
		epsilon:    epsilon,
		numActions: numActions,
		numInputs:  features,
	}
	_, err := network.fwd(input)
	if err != nil {
		log.Fatal(err)
	}

	return &network
}

// SelectAction assume that VM that contains the policy has been run
func (e *EGreedyMLP) SelectAction(obs mat.VecDense) mat.Vector {
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

func (e *EGreedyMLP) Set(source *EGreedyMLP) error {
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

func (e *EGreedyMLP) Polyak(source *EGreedyMLP, tau float64) error {
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

func (e *EGreedyMLP) Learnables() G.Nodes {
	learnables := make([]*G.Node, 0, 2*len(e.l))
	for i := range e.l {
		learnables = append(learnables, e.l[i].Weights)
		if bias := e.l[i].Bias; bias != nil {
			learnables = append(learnables, bias)
		}
	}
	return G.Nodes(learnables)
}

func (e *EGreedyMLP) Model() []G.ValueGrad {
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
func (e *EGreedyMLP) fwd(input *G.Node) (*G.Node, error) {
	inputShape := input.Shape()[len(input.Shape())-1]
	if inputShape != e.numInputs {
		return nil, fmt.Errorf("invalid shape for input to neural net:"+
			" \n\twant(%v) \n\thave(%v)", e.numInputs, inputShape)
	}

	pred := input
	var err error
	for _, l := range e.l {
		if pred, err = l.fwd(pred); err != nil {
			log.Fatal(err)
		}
	}
	e.pred = pred
	G.Read(e.pred, &e.predVal)

	return pred, nil
}

func (e *EGreedyMLP) Output() G.Value {
	return e.predVal
}

type Learner struct {
	policy       *EGreedyMLP
	targetPolicy *EGreedyMLP
	updateTarget *G.Node
	vm           G.VM
	vmNext       G.VM
	solver       G.Solver

	x, xNext *G.Node // state/nextState inputs

	prevState  *mat.VecDense
	prevAction int
	nextState  *mat.VecDense
	nextAction int

	selectedAction *G.Node
}

func NewLearner() *Learner {
	g := G.NewGraph()
	x := G.NewVector(g, tensor.Float64, G.WithShape(4), G.WithName("X"), G.WithInit(G.Zeroes()))
	policy := NewEGreedyMLP(0.1, 4, x, g)
	updateTarget := G.NewScalar(g, tensor.Float64, G.WithName("updateTarget"))
	selectedAction := G.NewVector(g, tensor.Float64, G.WithName("selectedActions"), G.WithShape(policy.numActions))

	gTarget := G.NewGraph()
	xNext := G.NewVector(gTarget, tensor.Float64, G.WithShape(4), G.WithName("xNext"), G.WithInit(G.Zeroes()))
	targetPolicy := NewEGreedyMLP(0.0, 4, xNext, gTarget)

	prevAction := 1
	prevState := mat.NewVecDense(4, []float64{1, 3, 12, 111})
	nextState := mat.NewVecDense(4, []float64{1, 1, 2, 1})
	nextAction := 2

	selectedActionValue := G.Must(G.Mul(policy.pred, selectedAction))
	losses := G.Must(G.Sub(updateTarget, selectedActionValue))
	cost := G.Must(G.Mean(losses))
	G.Grad(cost, policy.Learnables()...)

	vm := G.NewTapeMachine(g, G.BindDualValues(policy.Learnables()...))
	vmNext := G.NewTapeMachine(gTarget)
	solver := G.NewVanillaSolver()
	return &Learner{policy, targetPolicy, updateTarget, vm, vmNext, solver, x, xNext, prevState, prevAction, nextState,
		nextAction, selectedAction}
}

func (l *Learner) Step() {
	data := tensor.New(tensor.WithBacking(l.prevState.RawVector().Data))
	err := G.Let(l.x, data)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Value:", l.x.Value())

	nextData := tensor.New(tensor.WithBacking(l.nextState.RawVector().Data))
	err = G.Let(l.xNext, nextData)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Value:", l.xNext.Value())
	fmt.Println("Weights before polyak:", l.targetPolicy.Learnables()[0].Value())

	l.vmNext.RunAll()

	updateTarget := l.targetPolicy.Output()
	fmt.Println("Update target:", updateTarget.Data().([]float64)[l.nextAction])
	G.Let(l.updateTarget, updateTarget.Data().([]float64)[l.nextAction])
	fmt.Println("Update target:", l.updateTarget.Value())

	action := make([]float64, l.policy.numActions)
	action[l.prevAction] = 1.0
	selectedAction := tensor.New(tensor.WithBacking(action))
	G.Let(l.selectedAction, selectedAction)

	l.vmNext.Reset()

	l.vm.RunAll()
	fmt.Println(l.policy.Model())
	l.solver.Step(l.policy.Model())
	l.vm.Reset()

	l.targetPolicy.Set(l.policy)
}

func main() {
	l := NewLearner()
	l.Step()

	ioutil.WriteFile("simple_graphLearner.dot", []byte(l.policy.g.ToDot()), 0644)

	// =========================================

	// solver := G.NewVanillaSolver(G.WithLearnRate(0.001))

	// g := G.NewGraph()
	// xNext := G.NewVector(g, tensor.Float64, G.WithShape(4), G.WithName("xNext"), G.WithInit(G.Zeroes()))
	// x := G.NewVector(g, tensor.Float64, G.WithShape(4), G.WithName("X"), G.WithInit(G.Zeroes()))
	// W := G.NewVector(g, tensor.Float64, G.WithShape(4),
	// 	G.WithName("L0W"), G.WithInit(G.GlorotU(1.0)))
	// val, _ := G.Mul(x, W)
	// grads, _ := G.Grad(val, W)
	// nextVal, _ := G.Mul(xNext, W)
	// fmt.Println(nextVal)
	// s := G.Must(G.Sub(val, nextVal))
	// for _, grad := range grads {
	// 	G.Must(G.HadamardProd(grad, s))
	// }

	// model := []G.ValueGrad{W}
	// vm := G.NewTapeMachine(g, G.BindDualValues(W))
	// vm.RunAll()
	// solver.Step(model)

	// ioutil.WriteFile("simple_graph.dot", []byte(g.ToDot()), 0644)
}

