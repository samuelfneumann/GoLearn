package main

import (
	"fmt"

	"sfneuman.com/golearn/agent/linear/discrete/esarsa"
	"sfneuman.com/golearn/environment/gridworld"
	"sfneuman.com/golearn/spec"
)

func main() {
	// Create the gridworld task
	x := []int{1}
	y := []int{0}
	goal, err := gridworld.NewGoal(x, y, 5, 5, -0.1, 1.0)
	if err != nil {
		fmt.Println("Could not create goal")
		return
	}

	// Create the start-state distribution
	starter, err := gridworld.NewSingleStart(0, 0, 5, 5)
	if err != nil {
		fmt.Println("Could not create starter")
		return
	}

	// Create the gridworld
	g, t := gridworld.New(5, 5, goal, 0.99, starter)
	fmt.Println(t)
	fmt.Println(g)

	// Create the QLearning spec
	args := spec.QLearning{E: 0.1, LearningRate: 0.1}
	// args := spec.ESarsa{TargetE: 0.0, BehaviourE: 0.1, LearningRate: 0.1}

	// Create the QLearning algorithm
	var seed uint64 = 192312
	// q := qlearning.New(g, args, seed)
	q := esarsa.New(g, args, seed)
	q.ObserveFirst(t)

	// Track the return
	total := 0.0
	episodicReward := make([]float64, 100)
	episodeReward := 0.0

	for i := 0; i < 1000000; i++ {
		// Take an action and send to env
		action := q.SelectAction(t)
		t, _ = g.Step(action)

		// Track return
		total += t.Reward
		episodeReward += t.Reward

		// Observe environmental change
		q.Observe(action, t)
		q.Step()

		// Reset env if end of episode
		if t.Last() {
			episodicReward = append(episodicReward, episodeReward)
			episodeReward = 0.0

			// Reset the environment and observe the first episode transition
			t = g.Reset()
			q.ObserveFirst(t)
		}
	}

	fmt.Println(q.Learner.Weights()["weights"])
	length := len(episodicReward)
	fmt.Println(episodicReward[length-5 : length])
}
