package qlearning

import (
	"testing"
	"time"

	"gonum.org/v1/gonum/spatial/r1"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/environment/box2d/lunarlander"
	"sfneuman.com/golearn/environment/wrappers"
	"sfneuman.com/golearn/utils/matutils/initializers/weights"
)

func BenchmarkIndexTileCoderLunarLanderAgentStep(b *testing.B) {
	// Set up the lunar lander environment
	seed := uint64(time.Now().UnixNano())
	s := environment.NewUniformStarter([]r1.Interval{
		{Min: lunarlander.InitialX, Max: lunarlander.InitialX},
		{Min: lunarlander.InitialY, Max: lunarlander.InitialY},
		{Min: lunarlander.InitialRandom, Max: lunarlander.InitialRandom},
	}, seed)
	task := lunarlander.NewLand(s, 250)
	discount := 1.0
	env, _ := lunarlander.NewDiscrete(task, discount, seed)

	// Set up the index tile coding environment
	numTilings := 5
	tilings := make([][]int, numTilings)
	for i := 0; i < len(tilings); i++ {
		tilings[i] = []int{4, 4, 4, 4, 4, 4, 4, 4}
	}
	tm, step := wrappers.NewIndexTileCoding(env, tilings, seed)

	// Create the QLearning agent
	args := Config{Epsilon: 0.25, LearningRate: 0.01}
	rand := weights.NewZeroUV() // Zero RNG
	init := weights.NewLinearUV(rand)
	q, err := New(tm, args, init, seed)
	if err != nil {
		panic(err)
	}

	// Observe the first environment transition
	q.ObserveFirst(step)
	a := q.SelectAction(step)
	step, _ = tm.Step(a)
	q.Observe(a, step)

	// Evaluate the stepping time of the agent
	for i := 0; i < b.N; i++ {
		q.Step()
	}
}
