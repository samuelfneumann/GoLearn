package examples

import (
	"fmt"

	"github.com/samuelfneumann/golearn/agent/linear/discrete/esarsa"
	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/environment/classiccontrol/mountaincar"
	"github.com/samuelfneumann/golearn/environment/wrappers"
	"github.com/samuelfneumann/golearn/experiment"
	"github.com/samuelfneumann/golearn/experiment/tracker"
	"github.com/samuelfneumann/golearn/utils/matutils/initializers/weights"
	"gonum.org/v1/gonum/spatial/r1"
)

func DifferentialESarsa() {
	var seed uint64 = 192382

	// Create the environment
	bounds := r1.Interval{Min: -0.01, Max: 0.01}

	s := environment.NewUniformStarter([]r1.Interval{bounds, bounds}, seed)
	task := mountaincar.NewGoal(s, 250, mountaincar.GoalPosition)
	m, _, err := mountaincar.NewDiscrete(task, 1.0)
	if err != nil {
		panic(err)
	}

	// Create the TileCoding env wrapper
	numTilings := 10
	tilings := make([][]int, numTilings)

	for i := 0; i < len(tilings); i++ {
		tilings[i] = []int{10, 10}
	}
	tm, _, err := wrappers.NewTileCoding(m, tilings, seed)
	if err != nil {
		panic(err)
	}

	// Create the average reward environment wrapper
	ttm, _, err := wrappers.NewAverageReward(tm, 0.0, 0.01, true)
	if err != nil {
		panic(err)
	}

	// Zero RNG
	weightSize := make([]float64, tm.ObservationSpec().Shape.Len())
	rand := weights.NewZeroMV(weightSize)

	// Create the weight initializer with the RNG
	init := weights.NewLinearMV(rand)

	// Create the learning algorithm
	args := esarsa.Config{BehaviourE: 0.1, TargetE: 0.05, LearningRate: 0.05}
	e, err := esarsa.New(tm, args, init, seed)
	if err != nil {
		panic(err)
	}

	// Register learner with average reward environment so that the
	// TDError of the learner can be used to update the average reward
	ttm.Register(e)

	// Experiment
	saver := tracker.NewReturn("./data.bin")
	exp := experiment.NewOnline(ttm, e, 100_000, []tracker.Tracker{saver}, nil)
	exp.Run()
	exp.Save()

	data := tracker.LoadFData("./data.bin")
	fmt.Println(data[len(data)-10:])
}
