package examples

import (
	"fmt"
	"time"

	"gonum.org/v1/gonum/spatial/r1"
	"sfneuman.com/golearn/agent/nonlinear/continuous/vanillapg"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/environment/classiccontrol/mountaincar"
	"sfneuman.com/golearn/experiment"
	"sfneuman.com/golearn/experiment/tracker"
	"sfneuman.com/golearn/network"

	G "gorgonia.org/gorgonia"
)

// VanillaPG provides an example on how to use the vanillapg package.
func VanillaPG() {

	var useed uint64 = 192382

	// Create the environment
	// Use an artificially easier problem for testing
	goalPosition := mountaincar.GoalPosition - 0.45
	position := r1.Interval{Min: -0.6, Max: -0.4}
	velocity := r1.Interval{Min: 0.0, Max: 0.0}

	s := environment.NewUniformStarter([]r1.Interval{position, velocity}, useed)
	task := mountaincar.NewGoal(s, 500, goalPosition)
	env, step := mountaincar.NewDiscrete(task, 0.99)
	fmt.Println(step)

	args := vanillapg.CategoricalMLPConfig{
		Policy:            vanillapg.Categorical,
		PolicyLayers:      []int{100, 50, 25},
		PolicyBiases:      []bool{true, true, true},
		PolicyActivations: []*network.Activation{network.ReLU(), network.ReLU(), network.ReLU()},

		ValueFnLayers:      []int{100, 50, 25},
		ValueFnBiases:      []bool{true, true, true},
		ValueFnActivations: []*network.Activation{network.ReLU(), network.ReLU(), network.ReLU()},

		InitWFn:      G.GlorotN(1.0),
		PolicySolver: G.NewAdamSolver(G.WithLearnRate(5e-4), G.WithBatchSize(50000)),
		VSolver:      G.NewAdamSolver(G.WithLearnRate(5e-4), G.WithBatchSize(50000)),

		ValueGradSteps: 1,
		EpochLength:    50000,
		Lambda:         0.97,
		Gamma:          0.99,
	}

	agent, err := args.CreateAgent(env, useed)
	if err != nil {
		panic(err)
	}

	start := time.Now()
	var saver tracker.Tracker = tracker.NewReturn("./data.bin")
	e := experiment.NewOnline(env, agent, 50000*1000, []tracker.Tracker{saver}, nil)
	e.Run()
	fmt.Println("Elapsed:", time.Since(start))
	e.Save()

	data := tracker.LoadData("./data.bin")
	fmt.Println(data)
}
