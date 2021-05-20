package examples

import (
	"fmt"

	"gonum.org/v1/gonum/spatial/r1"
	"sfneuman.com/golearn/agent/linear/discrete/qlearning"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/environment/classiccontrol/mountaincar"
	"sfneuman.com/golearn/environment/wrappers"
	"sfneuman.com/golearn/experiment"
	"sfneuman.com/golearn/experiment/savers"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/utils/matutils/initializers/weights"
)

func QlearningMountainCar() {
	var seed uint64 = 1923812

	// Create the QLearning config
	args := spec.QLearning{E: 0.25, LearningRate: 0.01}

	// Generate the starting state distribution
	positionBounds := r1.Interval{Min: -0.2, Max: 0.2}
	speedBounds := r1.Interval{Min: -0.005, Max: 0.005}
	s := environment.NewUniformStarter([]r1.Interval{positionBounds,
		speedBounds}, seed)

	// Generate the task to complete on the environment
	maxEpisodeSteps := 1000
	goalPosition := 0.45
	task := mountaincar.NewGoal(s, maxEpisodeSteps, goalPosition)

	// Create the Mountain Car environment
	discount := 1.0
	env, _ := mountaincar.New(task, discount)
	fmt.Println(env)

	// To use Linear Q-learning, we need to tile code the environment
	// states. Tile coding requires that each state observation
	// dimension be bounded. The wrappers.TileCoding struct will take
	// care of bounding the state dimensions for us, we just have to
	// choose the number of tilings and the tiles for each dimension
	// per tiling. Here, we create 5 tilings of size 5x5 tiles and
	// 5 tilings of size 3x3 tiles for our TileCoder.
	numTilings := 10
	tilings := make([][]int, numTilings)
	for i := 0; i < len(tilings)/2; i++ {
		tilings[i] = []int{5, 5}
	}
	for i := len(tilings) / 2; i < len(tilings); i++ {
		tilings[i] = []int{3, 3}
	}
	tm, _ := wrappers.NewTileCoding(env, tilings, seed)

	// Create the Q-learning algorithm. First, we defined an initialization
	// method for the Linear Q-learning algorithm's weights.
	// First, we need to create an RNG that will sample weights for us.
	weightSize := make([]float64, tm.ObservationSpec().Shape.Len())
	rand := weights.NewZero(weightSize) // Zero RNG

	// Create the weight initializer with the RNG
	init := weights.NewLinearMV(rand)

	// Create the Q-learning algorithm
	q := qlearning.New(tm, args, init, seed)

	// Now, we will create the experiment. First, generate savers to
	// determine what data from the experiment we want to save
	saver := savers.NewReturn("./data.bin")

	// Create a new Online experiment. Online experiments will only
	// run the agent online, and no offline evaluation will occur
	e := experiment.NewOnline(tm, q, 1_00_000, saver)

	// Run the experiment
	e.Run()

	// Save the data generated by the experiment
	e.Save()

	// Load the data from the experiment and pring it
	data := savers.LoadData("./data.bin")
	fmt.Println(data)
}