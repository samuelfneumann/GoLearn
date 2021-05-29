package main

import (
	"fmt"

	"gonum.org/v1/gonum/spatial/r1"
	"sfneuman.com/golearn/agent/linear/discrete/qlearning"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/environment/classiccontrol/mountaincar"
	"sfneuman.com/golearn/environment/wrappers"
	"sfneuman.com/golearn/experiment"
	"sfneuman.com/golearn/experiment/trackers"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/utils/matutils/initializers/weights"
)

func main() {
	var seed uint64 = 192382

	// Create the environment
	bounds := r1.Interval{Min: -0.01, Max: 0.01}

	s := environment.NewUniformStarter([]r1.Interval{bounds, bounds}, seed)
	task := mountaincar.NewGoal(s, 250, mountaincar.GoalPosition)
	m, _ := mountaincar.NewDiscrete(task, 1.0)

	// Create the TileCoding env wrapper
	numTilings := 10
	tilings := make([][]int, numTilings)

	for i := 0; i < len(tilings); i++ {
		tilings[i] = []int{10, 10}
	}
	tm, _ := wrappers.NewTileCoding(m, tilings, seed)

	ttm, _ := wrappers.NewAverageReward(tm, 0.0, 0.01, true)

	// Zero RNG
	weightSize := make([]float64, tm.ObservationSpec().Shape.Len())
	rand := weights.NewZero(weightSize)

	// Create the weight initializer with the RNG
	init := weights.NewLinearMV(rand)

	// Create the learning algorithm
	args := spec.QLearning{E: 0.1, LearningRate: 0.05}
	q, _ := qlearning.New(tm, args, init, seed)

	// Register learner with average reward
	ttm.Register(q)

	// Experiment
	var tracker trackers.Tracker = trackers.NewReturn("./data.bin")
	tracker = trackers.Register(tracker, m)
	e := experiment.NewOnline(ttm, q, 100_000, tracker)
	e.Run()
	e.Save()

	data := trackers.LoadData("./data.bin")
	fmt.Println(data[len(data)-10:])
}
