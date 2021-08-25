package qlearning

import (
	"testing"
	"time"

	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/environment/box2d/lunarlander"
	"github.com/samuelfneumann/golearn/environment/wrappers"
	"github.com/samuelfneumann/golearn/utils/matutils/initializers/weights"
	"gonum.org/v1/gonum/spatial/r1"
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
	env, _, err := lunarlander.NewDiscrete(task, discount, seed)
	if err != nil {
		b.Error(err)
	}

	// Set up the index tile coding environment
	numTilings := 5
	tilings := make([][]int, numTilings)
	for i := 0; i < len(tilings); i++ {
		tilings[i] = []int{4, 4, 4, 4, 4, 4, 4, 4}
	}
	tm, step, err := wrappers.NewIndexTileCoding(env, tilings, seed)
	if err != nil {
		b.Error(err)
	}

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
	step, _, err = tm.Step(a)
	if err != nil {
		b.Error(err)
	}
	q.Observe(a, step)

	// Evaluate the stepping time of the agent
	for i := 0; i < b.N; i++ {
		q.Step()
	}
}
