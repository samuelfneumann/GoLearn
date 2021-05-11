package main

import (
	"fmt"

	// "golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	// "gonum.org/v1/gonum/stat/distmv"

	"sfneuman.com/golearn/agent/linear/discrete/qlearning"
	"sfneuman.com/golearn/environment/gridworld"
	"sfneuman.com/golearn/spec"
	// "sfneuman.com/golearn/utils/matutils"
	"sfneuman.com/golearn/utils/matutils/initializers/weights"
)

func main() {
	r, c := 5, 5

	// Create the gridworld task
	x := []int{1}
	y := []int{0}
	goal, err := gridworld.NewGoal(x, y, r, c, -0.1, 1.0)
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
	args := spec.QLearning{E: 0.1, LearningRate: 0.5}
	// args := spec.ESarsa{TargetE: 0.0, BehaviourE: 0.1, LearningRate: 0.1}
	var seed uint64 = 192312

	// Zero RNG
	weightSize := make([]float64, r*c)
	rand := weights.NewZero(weightSize)

	// // Normal RNG
	// source := rand.NewSource(seed)
	// var mean []float64
	// var std []float64
	// for i := 0; i < c*r; i++ {
	// 	mean = append(mean, 0.2)
	// 	for j := 0; j < c*r; j++ {
	// 		if i != j {
	// 			std = append(std, 0.0)
	// 		} else {
	// 			std = append(std, 0.0001)
	// 		}
	// 	}
	// }
	// stddev := mat.NewSymDense(c*r, std)
	// rand, ok := distmv.NewNormal(mean, stddev, source)
	// if !ok {
	// 	panic("could not create distribution")
	// }

	// Create the weight initializer with the RNG
	init := weights.NewLinear(rand)

	// Create the learning algorithm
	q := qlearning.New(g, args, init, seed)
	// q := esarsa.New(g, args, init, seed)
	q.ObserveFirst(t)

	// Track the return
	total := 0.0
	episodicReward := make([]float64, 100)
	episodeReward := 0.0

	for i := 0; i < 2500000; i++ {
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

	fmt.Println()
	actions := []string{"left", "right", "up", "down"}
	w := q.Learner.Weights()["weights"]
	r, _ = w.Dims()
	for i := 0; i < r; i++ {
		fmt.Println("=== Action:", actions[i], "===")
		fmt.Println(w.RowView(i))
		fmt.Println()
	}
	w1 := q.Policy.Weights()["weights"]
	w2 := q.Target.Weights()["weights"]

	fmt.Println(mat.Equal(w1, w))
	fmt.Println(mat.Equal(w, w2))
	fmt.Println(mat.Equal(w1, w2))

	length := len(episodicReward)
	fmt.Println(episodicReward[length-5 : length])
}
