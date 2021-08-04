package qlearning

import (
	"testing"
	"time"

	"gonum.org/v1/gonum/spatial/r1"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/environment/box2d/lunarlander"
	"sfneuman.com/golearn/environment/wrappers"
	"sfneuman.com/golearn/experiment"
	"sfneuman.com/golearn/utils/matutils/initializers/weights"
)

func BenchmarkLunarLander(b *testing.B) {
	seed := uint64(time.Now().UnixNano())
	s := environment.NewUniformStarter([]r1.Interval{
		{Min: lunarlander.InitialX, Max: lunarlander.InitialX},
		{Min: lunarlander.InitialY, Max: lunarlander.InitialY},
		{Min: lunarlander.InitialRandom, Max: lunarlander.InitialRandom},
	}, seed)
	task := lunarlander.NewLand(s, 500)

	// Create the QLearning config
	args := Config{Epsilon: 0.25, LearningRate: 0.01}
	// Create the Mountain Car environment
	discount := 1.0
	env, _ := lunarlander.NewDiscrete(task, discount, seed)

	numTilings := 1
	tilings := make([][]int, numTilings)
	for i := 0; i < len(tilings); i++ {
		tilings[i] = []int{8, 8, 8, 8, 8, 8, 8, 8}
	}
	tm, _ := wrappers.NewTileCoding(env, tilings, seed)

	rand := weights.NewZeroUV() // Zero RNG
	init := weights.NewLinearUV(rand)
	q, err := New(tm, args, init, seed)
	if err != nil {
		panic(err)
	}

	exp := experiment.NewOnline(tm, q, 20, nil, nil)
	exp.Run()

	// q.ObserveFirst(step)

	// for i := 0; i < b.N; i++ {
	// 	a := q.SelectAction(step)
	// 	var last bool
	// 	step, last = tm.Step(a)
	// 	q.Observe(a, step)
	// 	q.Step()

	// 	if last {
	// 		step = tm.Reset()
	// 		q.ObserveFirst(step)
	// 	}
	// }

}
