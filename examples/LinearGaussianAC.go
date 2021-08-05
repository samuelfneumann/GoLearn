package examples

import (
	"fmt"

	"gonum.org/v1/gonum/spatial/r1"
	"github.com/samuelfneumann/golearn/agent/linear/continuous/actorcritic"
	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/environment/classiccontrol/mountaincar"
	"github.com/samuelfneumann/golearn/environment/wrappers"
	"github.com/samuelfneumann/golearn/experiment"
	"github.com/samuelfneumann/golearn/experiment/tracker"
	"github.com/samuelfneumann/golearn/utils/matutils/initializers/weights"
)

func LinearGaussianActorCritic() {
	var seed uint64 = 192382

	// Create the environment
	bounds := r1.Interval{Min: -0.01, Max: 0.01}

	s := environment.NewUniformStarter([]r1.Interval{bounds, bounds}, seed)
	task := mountaincar.NewGoal(s, 1000, mountaincar.GoalPosition)
	m, _ := mountaincar.NewContinuous(task, 1.0)

	// Create the TileCoding env wrapper
	numTilings := 10
	tilings := make([][]int, numTilings)

	for i := 0; i < len(tilings); i++ {
		tilings[i] = []int{10, 10}
	}
	tm, _ := wrappers.NewTileCoding(m, tilings, seed)

	ttm, _ := wrappers.NewAverageReward(tm, 0.0, 0.01, true)

	// Zero RNG
	rand := weights.NewZeroUV()

	// Create the weight initializer with the RNG
	init := weights.NewLinearUV(rand)

	// Create the learning algorithm
	args := actorcritic.LinearGaussianConfig{
		ActorLearningRate:  0.001,
		CriticLearningRate: 0.01,
		Decay:              0.5,
	}
	q, err := actorcritic.NewLinearGaussian(tm, args, init, seed)
	if err != nil {
		panic(err)
	}

	// Register learner with average reward
	ttm.Register(q)

	// Experiment
	var saver tracker.Tracker = tracker.NewReturn("./data.bin")
	saver = tracker.Register(saver, m)
	e := experiment.NewOnline(ttm, q, 1_000_000, []tracker.Tracker{saver}, nil)
	e.Run()
	e.Save()

	data := tracker.LoadFData("./data.bin")
	fmt.Println(data[len(data)-10:])
}
