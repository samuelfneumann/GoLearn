package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/environment/gridworld"
)

func main() {
	x := []int{2}
	y := []int{6}

	goal, err := gridworld.NewGoal(x, y, 10, 10, -0.1, 1.0)
	if err != nil {
		fmt.Println("Could not create goal")
		return
	}

	starter, err := gridworld.NewSingleStart(0, 0, 10, 10)
	if err != nil {
		fmt.Println("Could not create starter")
		return
	}

	g, t := gridworld.New(0, 0, 10, 10, goal, 0.99, starter)
	fmt.Println(t)
	fmt.Println(g)
	fmt.Println()

	actions := []float64{1, 2, 2, 1, 2, 2, 2, 2, 1, 1, 1, 1, 1, 3, 3, 2, 2}

	for _, action := range actions {
		t, _ = g.Step(mat.NewVecDense(1, []float64{action}))
		fmt.Println(t)
		fmt.Println(g)
		fmt.Println()
		if t.Last() {
			fmt.Println("========== At goal, resetting ==========")
			t = g.Reset()
			fmt.Println(t)
			fmt.Println(g)
			fmt.Println()
		}
	}

}
