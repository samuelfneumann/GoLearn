package wrappers

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
)

type AverageReward struct {
	environment.Environment
	avgReward      float64
	learningRate   float64
	useTDError     bool
	learner        agent.Learner
	lastStep       timestep.TimeStep
	secondLastStep timestep.TimeStep
	lastAction     mat.Vector
}

func NewAverageReward(env environment.Environment, init, learningRate float64,
	useTDError bool) (*AverageReward, timestep.TimeStep) {

	step := env.Reset()
	step.Discount = 1.0
	secondLastStep := timestep.TimeStep{}

	averageR := &AverageReward{env, init, learningRate, useTDError, nil, step,
		secondLastStep, nil}

	return averageR, step
}

func (a *AverageReward) Reset() timestep.TimeStep {
	step := a.Environment.Reset()
	step.Discount = 1.0

	a.lastStep = step
	a.secondLastStep = timestep.TimeStep{}
	a.lastAction = nil

	return step

}

func (a *AverageReward) Register(l agent.Learner) {
	a.learner = l
}

// Step function should update the average reward and return the
// differential reward. It should also modify the timestep so that
// the discount == 1.0
func (a *AverageReward) Step(action mat.Vector) (timestep.TimeStep, bool) {
	if a.learner == nil && a.useTDError {
		panic("when using the TD error to update the average reward, " +
			"a learner must first be registered using the Register() method.")
	}

	// Step in the environment to find TimeStep with S_{t+1}, R_{t} for action
	// A_{t}
	step, _ := a.Environment.Step(action)

	// Update avgReward_{t-1} -> avgReward_{t}
	if a.useTDError && !a.lastStep.First() {
		// Calculate differential TD error for S_{t-1}, A_{t-1}, R_{t-1},
		// S_{t}, A_{t}
		transition := timestep.NewTransition(a.secondLastStep, a.lastAction,
			a.lastStep, action)
		tdError := a.learner.TdError(transition)

		// Update average reward estimate using differential TD error
		a.avgReward += a.learningRate * tdError
	} else {
		// Update without using TD error
		a.avgReward += a.learningRate * (a.lastStep.Reward)
	}

	// Step contains the reward R_{t} for aciton A_{t}. Augment the reward
	// to become the average reard: R_{t} <- R_{t} - avgReward_{t}
	step.Reward -= a.avgReward
	step.Discount = 1.0

	a.secondLastStep = a.lastStep
	a.lastAction = action
	a.lastStep = step

	// Return the next state S_{t+1} and Reward R_{t} for taking Action A_{t}
	// in state S_{t}
	return step, step.Last()
}

func (a *AverageReward) RewardSpec() spec.Environment {
	rewardSpec := a.Environment.RewardSpec()
	rewardSpec.Type = spec.AverageReward
	return rewardSpec
}

func (a *AverageReward) DiscountSpec() spec.Environment {
	discountSpec := a.Environment.DiscountSpec()

	// Replace each discount value with a 1.0 since no discounting is used.
	bounds := make([]float64, discountSpec.Shape.Len())
	for i := 0; i < discountSpec.Shape.Len(); i++ {
		bounds[i] = 1.0
	}

	// Update the discount lower and upper bounds
	vecBounds := mat.NewVecDense(len(bounds), bounds)
	discountSpec.LowerBound = vecBounds
	discountSpec.UpperBound = vecBounds

	return discountSpec
}

func (a *AverageReward) String() string {
	return fmt.Sprintf("Average Reward: %v", a.Environment)
}
