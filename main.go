package main

import (
	"fmt"
	"time"

	"gonum.org/v1/gonum/spatial/r1"
	"gorgonia.org/gorgonia"
	"sfneuman.com/golearn/agent/nonlinear/discrete/deepq"
	"sfneuman.com/golearn/agent/nonlinear/discrete/policy"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/environment/classiccontrol/mountaincar"
	"sfneuman.com/golearn/experiment"
	"sfneuman.com/golearn/experiment/trackers"
)

func main() {
	var useed uint64 = 192382
	var seed int64 = 192382

	// Create the environment
	bounds := r1.Interval{Min: -0.01, Max: 0.01}

	s := environment.NewUniformStarter([]r1.Interval{bounds, bounds}, useed)
	task := mountaincar.NewGoal(s, 250, mountaincar.GoalPosition)
	m, _ := mountaincar.NewDiscrete(task, 1.0)

	// Create the learning algorithm
	args := deepq.Config{
		PolicyLayers: []int{100, 50},
		Biases:       []bool{true, true},
		Activations:  []policy.Activation{gorgonia.Rectify, gorgonia.Rectify},
		InitWFn:      gorgonia.GlorotU(1.0),
		LearningRate: 0.0001,
		Epsilon:      0.1,
	}
	q, err := deepq.New(m, args, seed)
	if err != nil {
		panic(err)
	}

	// Experiment
	start := time.Now()
	var tracker trackers.Tracker = trackers.NewReturn("./data.bin")
	tracker = trackers.Register(tracker, m)
	e := experiment.NewOnline(m, q, 20_000, tracker)
	e.Run()
	fmt.Println("Elapsed:", time.Since(start))
	e.Save()

	data := trackers.LoadData("./data.bin")
	fmt.Println(data)
}
