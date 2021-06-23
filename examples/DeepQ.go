package examples

import (
	"fmt"
	"time"

	"gonum.org/v1/gonum/spatial/r1"
	G "gorgonia.org/gorgonia"
	"sfneuman.com/golearn/agent/nonlinear/discrete/deepq"
	"sfneuman.com/golearn/agent/nonlinear/discrete/policy"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/environment/classiccontrol/mountaincar"
	"sfneuman.com/golearn/environment/wrappers"
	"sfneuman.com/golearn/experiment"
	"sfneuman.com/golearn/experiment/trackers"
)

func DeepQ() {
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
		Activations:  []policy.Activation{G.Rectify, G.Rectify},
		InitWFn:      G.GlorotU(1.0),
		LearningRate: 0.0001,
	}
	q, err := deepq.New(m, args, seed)
	if err != nil {
		panic(err)
	}

	// Experiment
	start := time.Now()
	var tracker trackers.Tracker = trackers.NewReturn("./data.bin")
	tracker = trackers.Register(tracker, m)
	e := experiment.NewOnline(m, q, 10_000, tracker)
	e.Run()
	fmt.Println("Elapsed:", time.Since(start))
	e.Save()

	data := trackers.LoadData("./data.bin")
	fmt.Println(data)
}

func QLearning() {
	var useed uint64 = 192382
	var seed int64 = 192382

	// Create the environment
	bounds := r1.Interval{Min: -0.01, Max: 0.01}

	s := environment.NewUniformStarter([]r1.Interval{bounds, bounds}, useed)
	task := mountaincar.NewGoal(s, 1000, mountaincar.GoalPosition)
	m, _ := mountaincar.NewDiscrete(task, 1.0)

	numTilings := 10
	tilings := make([][]int, numTilings)
	for i := 0; i < len(tilings)/2; i++ {
		tilings[i] = []int{5, 5}
	}
	for i := len(tilings) / 2; i < len(tilings); i++ {
		tilings[i] = []int{3, 3}
	}
	tm, _ := wrappers.NewTileCoding(m, tilings, useed)

	// Create the learning algorithm
	args := deepq.Config{
		PolicyLayers: []int{},
		Biases:       []bool{},
		Activations:  []policy.Activation{},
		InitWFn:      G.GlorotU(1.0),
		LearningRate: 0.01,
	}
	q, err := deepq.New(tm, args, seed)
	if err != nil {
		panic(err)
	}

	// Experiment
	start := time.Now()
	var tracker trackers.Tracker = trackers.NewReturn("./data.bin")
	tracker = trackers.Register(tracker, m)
	e := experiment.NewOnline(tm, q, 100_000, tracker)
	e.Run()
	fmt.Println("Elapsed:", time.Since(start))
	e.Save()

	data := trackers.LoadData("./data.bin")
	fmt.Println(data[len(data)-10:])
}
