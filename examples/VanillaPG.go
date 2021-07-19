package examples

import (
	"fmt"
	"math"
	"time"

	"gonum.org/v1/gonum/spatial/r1"
	"sfneuman.com/golearn/agent/nonlinear/continuous/vanillapg"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/environment/classiccontrol/mountaincar"
	"sfneuman.com/golearn/environment/gridworld"
	"sfneuman.com/golearn/experiment"
	"sfneuman.com/golearn/experiment/tracker"
	"sfneuman.com/golearn/network"

	G "gorgonia.org/gorgonia"
)

// VanillaPG provides an example on how to use the vanillapg package.
func VanillaPG() {
	var useed uint64 = 1923821

	// Create an artificially easier problem for example
	goalPosition := mountaincar.GoalPosition - 0.45 - 0.25 // Goal == -0.25
	position := r1.Interval{Min: -0.6, Max: -0.4}
	velocity := r1.Interval{Min: 0.0, Max: 0.0}

	s := environment.NewUniformStarter([]r1.Interval{position, velocity}, useed)
	task := mountaincar.NewGoal(s, 500, goalPosition)
	m, step := mountaincar.NewDiscrete(task, 0.99)
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
		PolicySolver: G.NewAdamSolver(G.WithLearnRate(1e-2), G.WithBatchSize(50000)),
		VSolver:      G.NewAdamSolver(G.WithLearnRate(1e-2), G.WithBatchSize(50000)),

		ValueGradSteps: 25,
		EpochLength:    50000,
		Lambda:         1.0,
		Gamma:          0.99,
	}

	env := m

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

// VanillaPgGridWorld gives an example of VanillaPG running on a
// gridworld. Note the embarassingly sad performance compared to
// linear function approximation methods.
func VanillaPgGridWorld() {
	var useed uint64 = 1923812121431427

	r, c := 5, 5

	// Create the start-state distribution
	starter, err := gridworld.NewSingleStart(0, 0, r, c)
	if err != nil {
		fmt.Println("Could not create starter")
		return
	}

	// Create the gridworld task of reaching a goal state. The goals
	// are specified as a []int, representing (x, y) coordinates
	goalX, goalY := []int{r - 1}, []int{c - 1}
	timestepReward, goalReward := -0.1, 1.0
	goal, err := gridworld.NewGoal(starter, goalX, goalY, r, c,
		timestepReward, goalReward)

	if err != nil {
		fmt.Println("Could not create goal")
		return
	}

	// Create the gridworld
	discount := 0.99
	env, t := gridworld.New(r, c, goal, discount)
	fmt.Println(t)

	nonlinearity := network.ReLU()
	args := vanillapg.CategoricalMLPConfig{
		Policy:            vanillapg.Categorical,
		PolicyLayers:      []int{100, 50, 25},
		PolicyBiases:      []bool{true, true, true},
		PolicyActivations: []*network.Activation{nonlinearity, nonlinearity, nonlinearity},

		ValueFnLayers:      []int{100, 50, 25},
		ValueFnBiases:      []bool{true, true, true},
		ValueFnActivations: []*network.Activation{nonlinearity, nonlinearity, nonlinearity},

		InitWFn:      G.GlorotN(math.Sqrt(2)),
		PolicySolver: G.NewAdamSolver(G.WithLearnRate(1e-2)),
		VSolver:      G.NewAdamSolver(G.WithLearnRate(1e-2)),

		ValueGradSteps: 25,
		EpochLength:    50000,
		Lambda:         1.0,
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
