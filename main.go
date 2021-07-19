package main

import (
	"fmt"
	"time"

	"gonum.org/v1/gonum/spatial/r1"
	"sfneuman.com/golearn/agent/nonlinear/continuous/policy"
	vanillapg "sfneuman.com/golearn/agent/nonlinear/continuous/vanillapg"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/environment/classiccontrol/mountaincar"
	"sfneuman.com/golearn/experiment"
	"sfneuman.com/golearn/experiment/tracker"
	"sfneuman.com/golearn/network"

	G "gorgonia.org/gorgonia"
)

func main() {

	// vanillapg.TestBuffer2()
	var useed uint64 = 192382
	// var seed int64 = 192382

	// Create the environment
	// Use an artificially easier problem for testing
	goalPosition := mountaincar.GoalPosition - 0.45
	position := r1.Interval{Min: -0.6, Max: -0.4}
	velocity := r1.Interval{Min: 0.0, Max: 0.0}

	s := environment.NewUniformStarter([]r1.Interval{position, velocity}, useed)
	task := mountaincar.NewGoal(s, 500, goalPosition) // ! Make the env easier and see if it learns
	m, step := mountaincar.NewDiscrete(task, 0.99)
	fmt.Println(step)

	c, err := policy.NewCategoricalMLP(m, 3, G.NewGraph(), []int{10}, []bool{true},
		[]*network.Activation{network.ReLU()}, G.GlorotU(1.0), 10)
	if err != nil {
		panic(err)
	}

	// fmt.Println(c.SelectAction(step))

	vm := G.NewTapeMachine(c.Network().Graph())
	c.LogPdfOf([]float64{0.1, 0.2, 0.1, 0.3, 1.0, 9.1}, []float64{1.0, 2.0, 0.0})
	vm.RunAll()
	vm.Reset()
	fmt.Println("\nLogProb of input actions\n", c.(*policy.CategoricalMLP).LogPdfNode().Value())

	fmt.Println("\nLogits of each obs")
	fmt.Println(c.(*policy.CategoricalMLP).Logits())

	// fmt.Println("LogProb of selected actions")
	// fmt.Println(c.(*policy.CategoricalMLP).LogProbSelectedActions().Value())

	// ===========================================================================
	// ===========================================================================
	// ===========================================================================
	// ===========================================================================
	// ===========================================================================

	// ioutil.WriteFile("net.dot", []byte(c.Network().Graph().ToDot()), 0644)

	// // g, err := policy.NewGaussianTreeMLP(m, 10, gorgonia.NewGraph(), []int{64}, []bool{false}, []*network.Activation{network.ReLU()},
	// // 	[][]int{{}, {}}, [][]bool{{}, {}}, [][]*network.Activation{{}, {}}, gorgonia.GlorotU(1.0), useed)
	// // if err != nil {
	// // 	panic(err)
	// // }
	// // ioutil.WriteFile("net.dot", []byte(g.Network().Graph().ToDot()), 0644)

	// args := vanillapg.TreePolicyConfig{
	// 	Policy:            vanillapg.Gaussian,
	// 	PolicyLayers:      []int{64, 64},
	// 	PolicyBiases:      []bool{true, true},
	// 	PolicyActivations: []*network.Activation{network.ReLU(), network.ReLU()},
	// 	LeafLayers:        [][]int{{64}, {64}},
	// 	LeafBiases:        [][]bool{{true}, {true}},
	// 	LeafActivations:   [][]*network.Activation{{network.ReLU()}, {network.ReLU()}},

	// 	ValueFnLayers:      []int{64, 64},
	// 	ValueFnBiases:      []bool{true, true},
	// 	ValueFnActivations: []*network.Activation{network.ReLU(), network.ReLU()},

	// 	InitWFn:      gorgonia.GlorotN(0.5),
	// 	PolicySolver: gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.0005)),
	// 	VSolver:      gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.0001)),

	// 	ValueGradSteps: 1,
	// 	EpochLength:    1000,
	// 	Lambda:         0.95,
	// 	Gamma:          0.99,
	// }

	// // args := vanillapg.TreePolicyConfig{
	// // 	Policy:            vanillapg.Gaussian,
	// // 	PolicyLayers:      []int{64},
	// // 	PolicyBiases:      []bool{false},
	// // 	PolicyActivations: []*network.Activation{network.ReLU()},
	// // 	LeafLayers:        [][]int{{}, {}},
	// // 	LeafBiases:        [][]bool{{}, {}},
	// // 	LeafActivations:   [][]*network.Activation{{}, {}},

	// // 	ValueFnLayers:      []int{64},
	// // 	ValueFnBiases:      []bool{true},
	// // 	ValueFnActivations: []*network.Activation{network.ReLU()},

	// // 	InitWFn:      gorgonia.GlorotU(1.0),
	// // 	PolicySolver: gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.005)),
	// // 	VSolver:      gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.005)),

	// // 	ValueGradSteps: 100,
	// // 	EpochLength:    1000,
	// // 	Lambda:         0.95,
	// // 	Gamma:          0.99,
	// // }

	args := vanillapg.CategoricalMLPConfig{
		Policy:            vanillapg.Categorical,
		PolicyLayers:      []int{100, 50, 25},
		PolicyBiases:      []bool{true, true, true},
		PolicyActivations: []*network.Activation{network.ReLU(), network.ReLU(), network.ReLU()},

		ValueFnLayers:      []int{100, 50, 25},
		ValueFnBiases:      []bool{true, true, true},
		ValueFnActivations: []*network.Activation{network.ReLU(), network.ReLU(), network.ReLU()},

		InitWFn:      G.GlorotN(1.0),
		PolicySolver: G.NewAdamSolver(G.WithLearnRate(5e-4), G.WithBatchSize(1000)),
		VSolver:      G.NewAdamSolver(G.WithLearnRate(5e-4), G.WithBatchSize(1000)),

		ValueGradSteps: 1,
		EpochLength:    10000,
		Lambda:         0.97,
		Gamma:          0.99,
	}

	env := m

	agent, err := args.CreateAgent(env, useed)
	if err != nil {
		panic(err)
	}

	start := time.Now()
	var saver tracker.Tracker = tracker.NewReturn("./data.bin")
	e := experiment.NewOnline(env, agent, 1000*7000, []tracker.Tracker{saver}, nil)
	e.Run()
	fmt.Println("Elapsed:", time.Since(start))
	e.Save()

	data := tracker.LoadData("./data.bin")
	fmt.Println(data)

	// ===================================
	// mlp, err := network.NewTreeMLP(1, 4, 2, gorgonia.NewGraph(), []int{64, 64}, []bool{true, true}, []*network.Activation{network.ReLU(), network.ReLU()},
	// 	[][]int{{64}}, [][]bool{{true}}, [][]*network.Activation{{network.ReLU()}}, gorgonia.GlorotU(1.0))
	// if err != nil {
	// 	panic(err)
	// }

	// mlp, err := network.NewMultiHeadMLP(1, 4, 2, gorgonia.NewGraph(),
	// 	[]int{64, 64}, []bool{true, true}, gorgonia.GlorotU(1.0), []*network.Activation{network.ReLU(), network.ReLU()})
	// if err != nil {
	// 	panic(err)
	// }

	// targets := gorgonia.NewMatrix(mlp.Graph(), tensor.Float64, gorgonia.WithShape(mlp.Prediction()[0].Shape()...))

	// loss := gorgonia.Must(gorgonia.Sub(targets, mlp.Prediction()[0]))
	// loss = gorgonia.Must(gorgonia.Square(loss))
	// loss = gorgonia.Must(gorgonia.Mean(loss))
	// // loss2 := gorgonia.Must(gorgonia.Sub(mlp.Prediction()[1], targets))
	// // loss2 = gorgonia.Must(gorgonia.Square(loss2))
	// // loss2 = gorgonia.Must(gorgonia.Mean(loss2))
	// // loss := gorgonia.Must(gorgonia.Add(loss1, loss2))

	// _, err = gorgonia.Grad(loss, mlp.Learnables()...)
	// if err != nil {
	// 	panic(err)
	// }

	// solver := gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.01))
	// vm := gorgonia.NewTapeMachine(mlp.Graph(), gorgonia.BindDualValues(mlp.Learnables()...))

	// targetsTensor := tensor.NewDense(tensor.Float64, tensor.Shape([]int{2, 4}), tensor.WithBacking([]float64{1, 2, 3, 4, 5, 6, 7, 8}))
	// for i := 0; i < 1000; i++ {
	// 	gorgonia.Let(targets, targetsTensor)
	// 	mlp.SetInput([]float64{0, 2, 4, 6})
	// 	vm.RunAll()

	// 	err := solver.Step(mlp.Model())
	// 	if err != nil {
	// 		panic(err)
	// 	}
	// 	fmt.Println(mlp.Output()[0])
	// 	vm.Reset()
	// }

	// fmt.Println()
	// mlp.SetInput([]float64{4, 5, 6, 7})
	// vm.RunAll()
	// fmt.Println(mlp.Output())
	// vm.Reset()

	// vanillapg.TestBuffer()

	// ======================================
	// ======================================
	// ======================================
	// ======================================
	// ======================================
	// ======================================
	// g := gorgonia.NewGraph()
	// p, err := policy.NewGaussianTreeMLP(
	// 	m,
	// 	2,
	// 	g,
	// 	[]int{100, 50},
	// 	[]bool{true, true},
	// 	[]*network.Activation{network.ReLU(), network.ReLU()},
	// 	[][]int{{50, 25}, {50, 25}},
	// 	[][]bool{{true, true}, {true, true}},
	// 	[][]*network.Activation{{network.ReLU(), network.ReLU()}, {network.ReLU(), network.ReLU()}},
	// 	gorgonia.GlorotU(1.0),
	// 	useed,
	// )
	// if err != nil {
	// 	log.Fatal(err)
	// }

	// // fmt.Println(p)
	// // fmt.Println(p.SelectAction(step))

	// vm := gorgonia.NewTapeMachine(p.Network().Graph())
	// _, _ = p.LogProbOf([]float64{1.0, 0.1, -0.1, -0.9}, []float64{0.5, 0.5})
	// vm.RunAll()
	// fmt.Println(policy.LogProb)
	// fmt.Printf("norm = s.norm(loc=%v, scale=%v); norm.logpdf(0.5)\n", p.(*policy.GaussianTreeMLP).Mean()[0], p.(*policy.GaussianTreeMLP).Std()[0])
	// fmt.Printf("norm = s.norm(loc=%v, scale=%v); norm.logpdf(0.5)\n", p.(*policy.GaussianTreeMLP).Mean()[1], p.(*policy.GaussianTreeMLP).Std()[1])
	// vm.Reset()

	// ==============================
	// ==============================
	// ==============================
	// ==============================
	// ==============================
	// ==============================

	// var useed uint64 = 192382
	// var seed int64 = 192382

	// // Create the environment
	// bounds := r1.Interval{Min: -0.01, Max: 0.01}

	// s := environment.NewUniformStarter([]r1.Interval{bounds, bounds}, useed)
	// task := mountaincar.NewGoal(s, 250, mountaincar.GoalPosition)
	// m, _ := mountaincar.NewDiscrete(task, 1.0)

	// // Create the learning algorithm
	// args := deepq.Config{
	// 	PolicyLayers:         []int{100, 50, 25},
	// 	Biases:               []bool{true, true, true},
	// 	Activations:          []*network.Activation{network.ReLU(), network.ReLU(), network.ReLU()},
	// 	InitWFn:              gorgonia.GlorotU(1.0),
	// 	Epsilon:              0.1,
	// 	Remover:              expreplay.NewFifoSelector(1),
	// 	Sampler:              expreplay.NewUniformSelector(1, seed),
	// 	MaximumCapacity:      1,
	// 	MinimumCapacity:      1,
	// 	Tau:                  1.0,
	// 	TargetUpdateInterval: 1,
	// 	Solver:               gorgonia.NewAdamSolver(gorgonia.WithLearnRate(0.00001)),
	// }
	// q, err := deepq.New(m, args, seed)
	// if err != nil {
	// 	panic(err)
	// }

	// // Experiment
	// start := time.Now()
	// var saver tracker.Tracker = tracker.NewReturn("./data.bin")
	// e := experiment.NewOnline(m, agent, 1000*120, []tracker.Tracker{saver}, nil)
	// e.Run()
	// fmt.Println("Elapsed:", time.Since(start))
	// e.Save()

	// data := tracker.LoadData("./data.bin")
	// fmt.Println(data)

	// ===========

	// p, err := network.NewMultiHeadMLP(10, 32, 5, gorgonia.NewGraph(), []int{10}, []bool{true}, gorgonia.GlorotU(1.0), []*network.Activation{network.ReLU()})
	// if err != nil {
	// 	panic(err)
	// }
	// f, err := os.Create("net.bin")
	// if err != nil {
	// 	panic(err)
	// }
	// enc := gob.NewEncoder(f)
	// err = enc.Encode(p)
	// if err != nil {
	// 	panic(err)
	// }
	// f.Close()

	// f2, err := os.Open("net.bin")
	// if err != nil {
	// 	panic(err)
	// }
	// dec := gob.NewDecoder(f2)
	// p2, _ := network.NewMultiHeadMLP(11, 33, 6, gorgonia.NewGraph(), []int{11}, []bool{true}, gorgonia.GlorotU(1.0), []*network.Activation{network.ReLU()})
	// // var p2 network.NeuralNet
	// // p2 = p2.(*network.MultiHeadMLP)
	// err = dec.Decode(p2)
	// if err != nil {
	// 	panic(err)
	// }
	// fmt.Println(p2)
	// fmt.Printf("%T\n", p2)
	// f2.Close()

	// network.TestGobFCLayer()

	// ============================================

	// remover := expreplay.NewFifoSelector(10)
	// sampler := expreplay.NewUniformSelector(2, 1243)
	// exp, _ := expreplay.New(remover, sampler, 1, 2, 3, 1)

	// s, a, r, g, ns, na, err := exp.Sample()
	// fmt.Println("Sample on empty error:", s, a, r, g, ns, na, err)

	// // Add first element to exp replay
	// ts := timestep.New(timestep.First, 1, 1, mat.NewVecDense(3, []float64{1, 2, 3}), 1)
	// action := mat.NewVecDense(1, []float64{1})
	// nextTs := timestep.New(timestep.First, 1, 1, mat.NewVecDense(3, []float64{4, 5, 6}), 1)
	// nextAction := mat.NewVecDense(1, []float64{2})
	// t := timestep.NewTransition(ts, action, nextTs, nextAction)

	// fmt.Println("Capacity:", exp.Capacity())
	// fmt.Println()

	// err = exp.Add(t)
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// fmt.Println(exp)
	// fmt.Println()

	// s, a, r, g, ns, na, _ = exp.Sample()
	// fmt.Println("Sampling:", s, a, r, g, ns, na)
	// fmt.Println()

	// // Add second element to exp replay
	// ts2 := timestep.New(timestep.Mid, 1, 1, mat.NewVecDense(3, []float64{9, 9, 9}), 1)
	// action2 := mat.NewVecDense(1, []float64{15})
	// nextTs2 := timestep.New(timestep.Mid, 1, 1, mat.NewVecDense(3, []float64{10, 10, 10}), 1)
	// nextAction2 := mat.NewVecDense(1, []float64{21})
	// t2 := timestep.NewTransition(ts2, action2, nextTs2, nextAction2)

	// err = exp.Add(t2)
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// fmt.Println("Capacity:", exp.Capacity())
	// fmt.Println()
	// fmt.Println(exp)
	// fmt.Println()

	// s, a, r, g, ns, na, _ = exp.Sample()
	// fmt.Println("Sampling:", s, a, r, g, ns, na)
	// fmt.Println()

	// // Add a new element and see how the cache removes the oldest
	// ts3 := timestep.New(timestep.Mid, 1, 1, mat.NewVecDense(3, []float64{14, 14, 14}), 1)
	// action3 := mat.NewVecDense(1, []float64{212})
	// nextTs3 := timestep.New(timestep.Mid, 1, 1, mat.NewVecDense(3, []float64{33, 33, 33}), 1)
	// nextAction3 := mat.NewVecDense(1, []float64{33})
	// t3 := timestep.NewTransition(ts3, action3, nextTs3, nextAction3)

	// err = exp.Add(t3)
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// fmt.Println("Capacity:", exp.Capacity())
	// fmt.Println()
	// fmt.Println(exp)
	// fmt.Println()

	// s, a, r, g, ns, na, err = exp.Sample()
	// fmt.Println("Sampling:", s, a, r, g, ns, na, err)
	// fmt.Println()

	// // err = exp.Add(t3)
	// // if err != nil {
	// // 	log.Fatal(err)
	// // }
	// // fmt.Println("Capacity:", exp.Capacity())
	// // fmt.Println()
	// // fmt.Println(exp)
	// // fmt.Println()

	// // s, a, r, g, ns, na, err = exp.Sample()
	// // fmt.Println("Sampling:", s, a, r, g, ns, na, err)
	// // fmt.Println()
}
