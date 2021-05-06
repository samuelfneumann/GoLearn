package main

import (
	"fmt"

	"sfneuman.com/golearn/environment/gridworld"
)

func main() {
	x := []int{2}
	y := []int{6}

	goal, err := gridworld.NewGoal(x, y, 10, 10)
	if err != nil {
		fmt.Println("Could not create gridworld")
		return
	}

	g := gridworld.New(0, 0, 10, 10, goal, 0.99)
	fmt.Println(g)

}
