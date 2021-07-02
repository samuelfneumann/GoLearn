package main

import (
	"encoding/gob"
	"fmt"
	"os"

	"gorgonia.org/gorgonia"
	"sfneuman.com/golearn/network"
)

func main() {
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
	// e := experiment.NewOnline(m, q, 20_000, []tracker.Tracker{saver}, nil)
	// e.Run()
	// fmt.Println("Elapsed:", time.Since(start))
	// e.Save()

	// data := tracker.LoadData("./data.bin")
	// fmt.Println(data)

	p, err := network.NewMultiHeadMLP(10, 32, 5, gorgonia.NewGraph(), []int{10}, []bool{true}, gorgonia.GlorotU(1.0), []*network.Activation{network.ReLU()})
	if err != nil {
		panic(err)
	}
	f, err := os.Create("net.bin")
	if err != nil {
		panic(err)
	}
	enc := gob.NewEncoder(f)
	err = enc.Encode(p)
	if err != nil {
		panic(err)
	}
	f.Close()

	f2, err := os.Open("net.bin")
	if err != nil {
		panic(err)
	}
	dec := gob.NewDecoder(f2)
	p2, _ := network.NewMultiHeadMLP(11, 33, 6, gorgonia.NewGraph(), []int{11}, []bool{true}, gorgonia.GlorotU(1.0), []*network.Activation{network.ReLU()})
	// var p2 network.NeuralNet
	p2 = p2.(*network.MultiHeadMLP)
	err = dec.Decode(p2)
	if err != nil {
		panic(err)
	}
	fmt.Println(p2)
	f2.Close()

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
