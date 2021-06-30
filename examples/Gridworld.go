package examples

import (
	"fmt"

	"sfneuman.com/golearn/agent/linear/discrete/qlearning"
	"sfneuman.com/golearn/environment/gridworld"
	"sfneuman.com/golearn/experiment"
	"sfneuman.com/golearn/experiment/trackers"
	"sfneuman.com/golearn/utils/matutils/initializers/weights"
)

func Gridworld() {
	var seed uint64 = 1923812 // 1923
	r, c := 5, 5

	// Create the start-state distribution
	starter, err := gridworld.NewSingleStart(0, 0, r, c)
	if err != nil {
		fmt.Println("Could not create starter")
		return
	}

	// Create the gridworld task of reaching a goal state. The goals
	// are specified as a []int, representing (x, y) coordinates
	goalX, goalY := []int{4}, []int{4}
	timestepReward, goalReward := -0.1, 1.0
	goal, err := gridworld.NewGoal(starter, goalX, goalY, r, c,
		timestepReward, goalReward)

	if err != nil {
		fmt.Println("Could not create goal")
		return
	}

	// Create the gridworld
	discount := 0.99
	g, t := gridworld.New(r, c, goal, discount)
	fmt.Println(t)
	fmt.Println(g)

	// Create the Q-learning struct which will learn on this gridworld
	// Create the QLearning configuration
	args := qlearning.Config{Epsilon: 0.25, LearningRate: 0.01}

	// To create the Q-learning struct, we defined an initialization
	// method for the Linear Q-learning algorithm's weights.
	// First, we need to create an RNG that will sample weights for us.
	weightSize := make([]float64, g.ObservationSpec().Shape.Len())
	rand := weights.NewZeroMV(weightSize)

	// Create the weight initializer previously crated RNG
	init := weights.NewLinearMV(rand)

	// Create the learning algorithm
	q, err := qlearning.New(g, args, init, seed)
	if err != nil {
		panic(err)
	}

	// Experiment
	tracker := trackers.NewReturn("./data.bin")
	e := experiment.NewOnline(g, q, 100_000, []trackers.Tracker{tracker})
	e.Run()
	e.Save()

	data := trackers.LoadData("./data.bin")
	fmt.Println(data)
}
