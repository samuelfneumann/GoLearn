package gym_test

import (
	"testing"

	"github.com/samuelfneumann/gogym"
	"github.com/samuelfneumann/golearn/environment/gym"
	ts "github.com/samuelfneumann/golearn/timestep"
	"gonum.org/v1/gonum/mat"
)

func TestNew(t *testing.T) {
	envs := []string{
		// Classic Control
		"MountainCarContinuous-v0",
		"MountainCar-v0",
		"Pendulum-v0",
		"CartPole-v0",
		"Acrobot-v1",

		// MuJoCo
		"Ant-v2",
		"Hopper-v2",
		"Humanoid-v2",
		"HumanoidStandup-v2",
		"Walker2d-v2",
		"HalfCheetah-v2",
		"InvertedDoublePendulum-v2",
		"InvertedPendulum-v2",
		"Reacher-v2",
		"Swimmer-v2",

		// Box2D
		"LunarLander-v2",
		"LunarLanderContinuous-v2",
		"BipedalWalker-v3",
		"BipedalWalkerHardcore-v3",
	}

	for _, envName := range envs {
		// Create GymEnv
		env, step, err := gym.New(envName, 0.99, 123)
		if err != nil {
			t.Errorf("env %v: %v", envName, err)
		} else if (env == nil || step == ts.TimeStep{}) {
			t.Error("new: env or step should not be nil if err is nil")
		}

		// Take a bunch of steps in the environment to ensure it works
		size := env.ActionSpec().LowerBound.Len()
		for i := 0; i < 15; i++ {
			next, done, err := env.Step(mat.NewVecDense(size, nil))
			if err != nil {
				t.Errorf("env %v: %v", envName, err)
			} else if (next == ts.TimeStep{}) {
				t.Errorf("step: timestep %v should be non-nil", i)
			}

			if done {
				next, err := env.Reset()
				if err != nil {
					t.Errorf("env %v: %v", envName, err)
				} else if (next == ts.TimeStep{}) {
					t.Error("reset: start timestep should be non-nil")
				}
			}
		}

		// Reset the environment
		step, err = env.Reset()
		if err != nil {
			t.Errorf("env %v: %v", envName, err)
		} else if (step == ts.TimeStep{}) {
			t.Error("reset: start timestep should be non-nil")
		}

		// Check that the spec functions work
		env.ObservationSpec()
		env.ActionSpec()
		env.DiscountSpec()

		// Close the environment
		env.(*gym.GymEnv).Close()
	}
	// Close the package
	gogym.Close()
}
