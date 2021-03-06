package examples

import (
	"fmt"
	"math"
	"time"

	"github.com/samuelfneumann/golearn/agent/nonlinear/continuous/vanillapg"
	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/environment/classiccontrol/cartpole"
	"github.com/samuelfneumann/golearn/environment/envconfig"
	"github.com/samuelfneumann/golearn/environment/gridworld"
	"github.com/samuelfneumann/golearn/experiment"
	"github.com/samuelfneumann/golearn/experiment/tracker"
	"github.com/samuelfneumann/golearn/initwfn"
	"github.com/samuelfneumann/golearn/network"
	"github.com/samuelfneumann/golearn/solver"
	"gonum.org/v1/gonum/spatial/r1"
)

// VanillaPG provides an example on how to use the vanillapg package.
func VanillaPG() {
	var useed uint64 = 1923812121431427

	bounds := r1.Interval{Min: -0.05, Max: 0.05}
	starter := environment.NewUniformStarter([]r1.Interval{
		bounds,
		bounds,
		bounds,
		bounds,
	}, useed)

	task := cartpole.NewBalance(starter, 500, cartpole.FailAngle)
	env, _, err := cartpole.NewDiscrete(task, 0.99)
	if err != nil {
		panic(err)
	}

	policySolver, err := solver.NewDefaultAdam(5e-3, 1)
	if err != nil {
		panic(err)
	}
	valueSolver, err := solver.NewDefaultAdam(5e-3, 1)
	if err != nil {
		panic(err)
	}
	initWFn, err := initwfn.NewGlorotN(math.Sqrt(2.0))
	if err != nil {
		panic(err)
	}
	nonlinearity := network.ReLU()
	config := vanillapg.CategoricalMLPConfig{
		PolicyLayers:      []int{100, 50, 25},
		PolicyBiases:      []bool{true, true, true},
		PolicyActivations: []*network.Activation{nonlinearity, nonlinearity, nonlinearity},

		ValueFnLayers:      []int{100, 50, 25},
		ValueFnBiases:      []bool{true, true, true},
		ValueFnActivations: []*network.Activation{nonlinearity, nonlinearity, nonlinearity},

		InitWFn:      initWFn,
		PolicySolver: policySolver,
		VSolver:      valueSolver,

		ValueGradSteps: 25,
		EpochLength:    50000,
		Lambda:         1.0,
		Gamma:          0.99,
	}

	agent, err := config.CreateAgent(env, useed)
	if err != nil {
		panic(err)
	}

	start := time.Now()
	var saver tracker.Tracker = tracker.NewReturn("./data.bin")
	e := experiment.NewOnline(env, agent, 50000*100, []tracker.Tracker{saver}, nil)
	e.Run()
	fmt.Println("Elapsed:", time.Since(start))
	e.Save()

	data := tracker.LoadFData("./data.bin")
	fmt.Println(data)
}

// VanillaPgGridWorld gives an example of VanillaPG running on a
// gridworld. Note the embarassingly sad performance compared to
// linear function approximation methods. Note that the gridworld here
// does not contain a time limit, so VanillaPG may fail to ever finish
// a single episode.
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
		timestepReward, goalReward, 50_000)

	if err != nil {
		fmt.Println("Could not create goal")
		return
	}

	// Create the gridworld
	discount := 0.99
	env, t, err := gridworld.New(r, c, goal, discount)
	if err != nil {
		panic(err)
	}
	fmt.Println(t)

	nonlinearity := network.ReLU()
	policySolver, err := solver.NewDefaultAdam(1e-2, 1)
	if err != nil {
		panic(err)
	}
	valueSolver, err := solver.NewDefaultAdam(1e-2, 1)
	if err != nil {
		panic(err)
	}
	initWFn, err := initwfn.NewGlorotN(math.Sqrt(2.0))
	if err != nil {
		panic(err)
	}
	args := vanillapg.CategoricalMLPConfig{
		PolicyLayers:      []int{100, 50, 25},
		PolicyBiases:      []bool{true, true, true},
		PolicyActivations: []*network.Activation{nonlinearity, nonlinearity, nonlinearity},

		ValueFnLayers:      []int{100, 50, 25},
		ValueFnBiases:      []bool{true, true, true},
		ValueFnActivations: []*network.Activation{nonlinearity, nonlinearity, nonlinearity},

		InitWFn:      initWFn,
		PolicySolver: policySolver,
		VSolver:      valueSolver,

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

	data := tracker.LoadFData("./data.bin")
	fmt.Println(data)
}

// COntinuousVanilaPGCartpole shows how to use the VanillaPG algorithm
// for continuous environment, in particular, Cartpole.
func ContinuousVanillaPGCartpole() {
	var useed uint64 = 1923812121431427

	envConf := envconfig.NewConfig(envconfig.Cartpole, envconfig.Balance,
		true, 500, 0.99, false)
	env, _, err := envConf.CreateEnv(useed)
	if err != nil {
		panic(err)
	}

	policySolver, _ := solver.NewDefaultAdam(5e-4, 1)
	valueSolver, _ := solver.NewDefaultAdam(5e-3, 1)
	Wfn, _ := initwfn.NewGlorotN(math.Sqrt(2))
	nonlinearity := network.ReLU()
	args := vanillapg.GaussianTreeMLPConfig{
		RootLayers: []int{64, 64},
		RootBiases: []bool{true, true},
		RootActivations: []*network.Activation{
			nonlinearity,
			nonlinearity,
		},

		LeafLayers: [][]int{{64, 64}, {64, 64}},
		LeafBiases: [][]bool{{true, true}, {true, true}},
		LeafActivations: [][]*network.Activation{
			{
				nonlinearity,
				nonlinearity,
			},
			{
				nonlinearity,
				nonlinearity,
			},
		},

		ValueFnLayers: []int{100, 50, 25},
		ValueFnBiases: []bool{true, true, true},
		ValueFnActivations: []*network.Activation{
			nonlinearity,
			nonlinearity,
			nonlinearity,
		},

		InitWFn:      Wfn,
		PolicySolver: policySolver,
		VSolver:      valueSolver,

		ValueGradSteps:          25,
		EpochLength:             500,
		FinishEpisodeOnEpochEnd: true,
		Lambda:                  1.0,
		Gamma:                   0.99,
	}
	agent, err := args.CreateAgent(env, useed)
	if err != nil {
		panic(err)
	}

	start := time.Now()
	var saver tracker.Tracker = tracker.NewReturn("./data.bin")
	e := experiment.NewOnline(env, agent, 50000*100, []tracker.Tracker{saver}, nil)
	e.Run()
	fmt.Println("Elapsed:", time.Since(start))
	e.Save()

	data := tracker.LoadFData("./data.bin")
	fmt.Println(data)
}
