package examples

import (
	"fmt"
	"time"

	"github.com/samuelfneumann/golearn/agent/linear/discrete/qlearning"
	"github.com/samuelfneumann/golearn/agent/nonlinear/discrete/deepq"
	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/environment/classiccontrol/mountaincar"
	"github.com/samuelfneumann/golearn/environment/envconfig"
	"github.com/samuelfneumann/golearn/environment/wrappers"
	"github.com/samuelfneumann/golearn/experiment"
	"github.com/samuelfneumann/golearn/experiment/tracker"
	"github.com/samuelfneumann/golearn/expreplay"
	"github.com/samuelfneumann/golearn/initwfn"
	"github.com/samuelfneumann/golearn/network"
	"github.com/samuelfneumann/golearn/solver"
	"gonum.org/v1/gonum/spatial/r1"
)

// DeepQCartpole gives an example of Deep Q on Cartpole
func DeepQCartpole() {
	var useed uint64 = 192382

	// Create the environment config with default parameters
	envConf := envconfig.NewConfig(envconfig.Cartpole, envconfig.Balance,
		false, 500, 0.99, false)

	// Create the solver
	sol, err := solver.NewDefaultAdam(0.00001, 1)
	if err != nil {
		panic(err)
	}
	initWFn, err := initwfn.NewGlorotU(1.0)
	if err != nil {
		panic(err)
	}

	// Create the learning algorithm
	agentConf := deepq.NewConfigList(
		[][]int{{64, 64}},
		[][]bool{{true, true}},
		[][]*network.Activation{
			{
				network.ReLU(),
				network.ReLU(),
			},
		},
		[]*solver.Solver{sol},
		[]*initwfn.InitWFn{initWFn},
		[]float64{0.1},
		[]expreplay.Config{
			{
				RemoveMethod:      expreplay.Fifo,
				SampleMethod:      expreplay.Uniform,
				RemoveSize:        1,
				SampleSize:        32,
				MaxReplayCapacity: 100000,
				MinReplayCapacity: 100,
			},
		},
		[]float64{1.0},
		[]int{1},
	)

	// Create the experiment configuration
	exp := experiment.Config{
		Type:        experiment.OnlineExp,
		MaxSteps:    25_000,
		EnvConfig:   envConf,
		AgentConfig: agentConf,
	}

	// Experiment
	var saver tracker.Tracker = tracker.NewReturn("./data.bin")
	e := exp.CreateExp(0, useed, []tracker.Tracker{saver}, nil)

	start := time.Now()
	e.Run()
	fmt.Println("Elapsed:", time.Since(start))
	e.Save()

	data := tracker.LoadFData("./data.bin")
	fmt.Println(data)
}

// DeepQMountainCar gives an example of Deep Q on Mountain Car
func DeepQMountainCar() {
	var useed uint64 = 192382

	// Create the environment config with default parameters
	envConf := envconfig.NewConfig(envconfig.MountainCar, envconfig.Goal,
		false, 500, 0.99, false)

	// Create the solver
	sol, err := solver.NewDefaultAdam(0.00001, 1)
	if err != nil {
		panic(err)
	}
	initWFn, err := initwfn.NewGlorotU(1.0)
	if err != nil {
		panic(err)
	}

	// Create the learning algorithm
	agentConf := deepq.NewConfigList(
		[][]int{{64, 64}},
		[][]bool{{true, true}},
		[][]*network.Activation{
			{
				network.ReLU(),
				network.ReLU(),
			},
		},
		[]*solver.Solver{sol},
		[]*initwfn.InitWFn{initWFn},
		[]float64{0.1},
		[]expreplay.Config{
			{
				RemoveMethod:      expreplay.Fifo,
				SampleMethod:      expreplay.Uniform,
				RemoveSize:        1,
				SampleSize:        32,
				MaxReplayCapacity: 100000,
				MinReplayCapacity: 100,
			},
		},
		[]float64{1.0},
		[]int{1},
	)

	// Create the experiment configuration
	exp := experiment.Config{
		Type:        experiment.OnlineExp,
		MaxSteps:    25_000,
		EnvConfig:   envConf,
		AgentConfig: agentConf,
	}

	// Experiment
	var saver tracker.Tracker = tracker.NewReturn("./data.bin")
	e := exp.CreateExp(0, useed, []tracker.Tracker{saver}, nil)

	start := time.Now()
	e.Run()
	fmt.Println("Elapsed:", time.Since(start))
	e.Save()

	data := tracker.LoadFData("./data.bin")
	fmt.Println(data)
}

// QLearningMountainCarWithConfigs gives an example of how to use
// Config structs for agents, environments, and experiments to create
// a Qlearning experiment on Mountain Car using the Gorgonia library
// for Qlearning implementation.
func QLearningMountainCarWithConfigs() {
	var useed uint64 = 192382

	// Create the environment config with default parameters
	envConf := envconfig.NewConfig(envconfig.MountainCar, envconfig.Goal,
		false, 500, 0.99, false)

	// Create the solver
	sol, err := solver.NewDefaultAdam(0.00001, 1)
	if err != nil {
		panic(err)
	}
	initWFn, err := initwfn.NewGlorotU(1.0)
	if err != nil {
		panic(err)
	}

	// Create the learning algorithm
	agentConf := deepq.NewConfigList(
		[][]int{{}},
		[][]bool{{}},
		[][]*network.Activation{{}},
		[]*solver.Solver{sol},
		[]*initwfn.InitWFn{initWFn},
		[]float64{0.1},
		[]expreplay.Config{
			{
				RemoveMethod:      expreplay.Fifo,
				SampleMethod:      expreplay.Uniform,
				RemoveSize:        1,
				SampleSize:        1,
				MaxReplayCapacity: 1,
				MinReplayCapacity: 1,
			},
		},
		[]float64{1.0},
		[]int{1},
	)

	// Create the experiment configuration
	exp := experiment.Config{
		Type:        experiment.OnlineExp,
		MaxSteps:    25_000,
		EnvConfig:   envConf,
		AgentConfig: agentConf,
	}

	// Experiment
	var saver tracker.Tracker = tracker.NewReturn("./data.bin")
	e := exp.CreateExp(0, useed, []tracker.Tracker{saver}, nil)

	start := time.Now()
	e.Run()
	fmt.Println("Elapsed:", time.Since(start))
	e.Save()

	data := tracker.LoadFData("./data.bin")
	fmt.Println(data)
}

// QLearningMountainCarFromScratch gives an example of how to create
// an experiment for running the Qlearning algorithm (with
// a Gorgonia library implementation) on th eMountain Car environment
// from complete scratch. That is, the environment and agent are
// create from scratch from their relevant constructors, and not
// the Config structs.
func QLearningMountainCarFromScratch() {
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
	args := qlearning.Config{
		LearningRate: 0.01,
		Epsilon:      0.1,
	}
	init, err := initwfn.NewZeroes()
	if err != nil {
		panic(err)
	}
	q, err := deepq.NewQlearning(tm, args, seed, init)
	if err != nil {
		panic(err)
	}

	// Experiment
	start := time.Now()
	var saver tracker.Tracker = tracker.NewReturn("./data.bin")
	saver = tracker.Register(saver, m)
	e := experiment.NewOnline(tm, q, 100_000, []tracker.Tracker{saver}, nil)
	e.Run()
	fmt.Println("Elapsed:", time.Since(start))
	e.Save()

	data := tracker.LoadFData("./data.bin")
	fmt.Println(data[len(data)-10:])
}
