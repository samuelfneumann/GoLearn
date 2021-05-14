package main

import (
	"fmt"
	"math"
	"time"

	// "golang.org/x/exp/rand"
	"gonum.org/v1/gonum/mat"
	// "gonum.org/v1/gonum/stat/distmv"

	"gonum.org/v1/gonum/spatial/r1"
	"sfneuman.com/golearn/agent/linear/discrete/qlearning"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/environment/classiccontrol"
	"sfneuman.com/golearn/environment/gridworld"
	"sfneuman.com/golearn/spec"

	// "sfneuman.com/golearn/utils/matutils"
	"sfneuman.com/golearn/utils/matutils/initializers/weights"
)

func main() {

	r, c := 5, 5

	// Create the start-state distribution
	starter, err := gridworld.NewSingleStart(0, 0, r, c)
	if err != nil {
		fmt.Println("Could not create starter")
		return
	}

	// Create the gridworld task
	x := []int{1}
	y := []int{0}
	goal, err := gridworld.NewGoal(starter, x, y, r, c, -0.1, 1.0)
	if err != nil {
		fmt.Println("Could not create goal")
		return
	}

	// Create the gridworld
	g, t := gridworld.New(5, 5, goal, 0.99)
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

	for i := 0; i < 100; i++ {
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

	// Pendulum
	fmt.Println("=== === === Pendulum === === ===")
	maxAngle := math.Pi
	angleBounds := r1.Interval{Min: -maxAngle, Max: maxAngle}

	maxSpeed := 8.0
	speedBounds := r1.Interval{Min: -maxSpeed, Max: maxSpeed}

	s := environment.NewUniformStarter([]r1.Interval{angleBounds, speedBounds}, seed)

	task := classiccontrol.NewPendulumSwingUp(s, 1000)
	p, t := classiccontrol.NewPendulum(task, 0.99)
	for i := 0; i < 110; i++ {
		action := 5.0
		if t.Observation.AtVec(1) < 0 {
			action = -5.0
		}

		t, _ = p.Step(mat.NewVecDense(1, []float64{action}))
		p.Render()
		time.Sleep(50000000)
		fmt.Println(t.Observation.AtVec(0))
	}
	fmt.Println(p, t)
}
