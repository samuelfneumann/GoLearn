package main

import (
	"fmt"
	"log"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/expreplay"
	"sfneuman.com/golearn/timestep"
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
	// 	PolicyLayers: []int{100, 50},
	// 	Biases:       []bool{true, true},
	// 	Activations:  []policy.Activation{gorgonia.Rectify, gorgonia.Rectify},
	// 	InitWFn:      gorgonia.GlorotU(1.0),
	// 	LearningRate: 0.0001,
	// 	Epsilon:      0.1,
	// }
	// q, err := deepq.New(m, args, seed)
	// if err != nil {
	// 	panic(err)
	// }

	// // Experiment
	// start := time.Now()
	// var tracker trackers.Tracker = trackers.NewReturn("./data.bin")
	// tracker = trackers.Register(tracker, m)
	// e := experiment.NewOnline(m, q, 20_000, tracker)
	// e.Run()
	// fmt.Println("Elapsed:", time.Since(start))
	// e.Save()

	// data := trackers.LoadData("./data.bin")
	// fmt.Println(data)

	remover := expreplay.NewFifoSelector(10)
	sampler := expreplay.NewUniformSelector(2, 1243)
	exp, _ := expreplay.New(remover, sampler, 1, 2, 3, 1)

	s, a, r, g, ns, na, err := exp.Sample()
	fmt.Println("Sample on empty error:", s, a, r, g, ns, na, err)

	// Add first element to exp replay
	ts := timestep.New(timestep.First, 1, 1, mat.NewVecDense(3, []float64{1, 2, 3}), 1)
	action := mat.NewVecDense(1, []float64{1})
	nextTs := timestep.New(timestep.First, 1, 1, mat.NewVecDense(3, []float64{4, 5, 6}), 1)
	nextAction := mat.NewVecDense(1, []float64{2})
	t := timestep.NewTransition(ts, action, nextTs, nextAction)

	fmt.Println("Capacity:", exp.Capacity())
	fmt.Println()

	err = exp.Add(t)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println(exp)
	fmt.Println()

	s, a, r, g, ns, na, _ = exp.Sample()
	fmt.Println("Sampling:", s, a, r, g, ns, na)
	fmt.Println()

	// Add second element to exp replay
	ts2 := timestep.New(timestep.Mid, 1, 1, mat.NewVecDense(3, []float64{9, 9, 9}), 1)
	action2 := mat.NewVecDense(1, []float64{15})
	nextTs2 := timestep.New(timestep.Mid, 1, 1, mat.NewVecDense(3, []float64{10, 10, 10}), 1)
	nextAction2 := mat.NewVecDense(1, []float64{21})
	t2 := timestep.NewTransition(ts2, action2, nextTs2, nextAction2)

	err = exp.Add(t2)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Capacity:", exp.Capacity())
	fmt.Println()
	fmt.Println(exp)
	fmt.Println()

	s, a, r, g, ns, na, _ = exp.Sample()
	fmt.Println("Sampling:", s, a, r, g, ns, na)
	fmt.Println()

	// Add a new element and see how the cache removes the oldest
	ts3 := timestep.New(timestep.Mid, 1, 1, mat.NewVecDense(3, []float64{14, 14, 14}), 1)
	action3 := mat.NewVecDense(1, []float64{212})
	nextTs3 := timestep.New(timestep.Mid, 1, 1, mat.NewVecDense(3, []float64{33, 33, 33}), 1)
	nextAction3 := mat.NewVecDense(1, []float64{33})
	t3 := timestep.NewTransition(ts3, action3, nextTs3, nextAction3)

	err = exp.Add(t3)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Println("Capacity:", exp.Capacity())
	fmt.Println()
	fmt.Println(exp)
	fmt.Println()

	s, a, r, g, ns, na, err = exp.Sample()
	fmt.Println("Sampling:", s, a, r, g, ns, na, err)
	fmt.Println()

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
}
