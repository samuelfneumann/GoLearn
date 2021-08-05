package wrappers

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"github.com/samuelfneumann/golearn/agent"
	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/spec"
	"github.com/samuelfneumann/golearn/timestep"
)

// AverageReward wraps an environment and alters rewards so that the
// differential reward is returned for each action. By wrapping an
// environment in an AverageReward, an algorithm is easily changed
// to become its differential counterpart. For example, training
// a Qlearning Agent on an AverageReward environment will cause
// the Qlearning Agent to use the differential Q-learning algorithm
// instead of the discounted Q-learning algorithm.
//
// AverageReward itself implements the environment.Environment
// interface, and is therefore itself an Environment.
//
// The average reward of a policy can be calculated as an exponential
// moving average in one of two ways. Either method uses the following
// equation:
//
//		avgReward <- avgReward + learningRate * (target - avgReward)
//
// In the first method, the average reward of a policy is estimated
// the most recent reward from the environment as the target of the
// update. In thes econd method, the differential TD error of the most
// recent online environmental transition is used as the target of
// the update. The second method is generally lower variance, but may
// be biased. Either method may be used, and is controlled by the
// useTDError constructor parameter. If using the differential TD error
// as the update target, then an agent.Learner must first be registered
// with the environment using the Register() method. This learner will
// generate the differential TD error to use in updates. Usually, the
// agent.Learner of the agent.Agent that is acting in the environment
// should be used. If not, the differential return may diverge,
// resulting in algorithms that do no learn.
type AverageReward struct {
	environment.Environment
	avgReward    float64
	learningRate float64
	useTDError   bool

	// learner calculates the TD error for the average reward update target
	learner agent.Learner

	// The last timestep is needed to calculate the TD error, since it stores
	// the reward R_{t} for the last action A_{t} taken in the last state S_{t}
	// and the next state S_{t+1}
	lastStep timestep.TimeStep

	// The second last timestep is needed to calculate the TD error, since
	// it stores the last state S_{t} that the last action A_{t}/lastAction
	// was taken in
	secondLastStep timestep.TimeStep
	lastAction     *mat.VecDense
}

// NewAverageReward creates and returns a new AverageReward Environment
// wrapper. The init parameter is the initial value for the average
// reward, usually set to 0. The useTDError parameter controls whether
// the average reward estimate is updated using the TD error of a
// registered learner as the update target or not. If false, then the
// environmental reward is used as the average reward update target.
func NewAverageReward(env environment.Environment, init, learningRate float64,
	useTDError bool) (*AverageReward, timestep.TimeStep) {
	// Get the first step from the embedded environment
	step := env.Reset()

	// AverageReward does not use discounting
	step.Discount = 1.0

	// Track the second last step so that the TD error can be properly
	// calculated using the SARSA tuple (S_{t-1}, A_{t-1}, R_{t-1}, S_{t},
	// A_{t}). The secondLastStep TimeStep stores S_{t-1}
	secondLastStep := timestep.TimeStep{}

	averageR := &AverageReward{env, init, learningRate, useTDError, nil, step,
		secondLastStep, nil}

	return averageR, step
}

// Reset resets the environment and returns a starting state drawn from
// the environment Starter
func (a *AverageReward) Reset() timestep.TimeStep {
	step := a.Environment.Reset()
	step.Discount = 1.0

	a.lastStep = step
	a.secondLastStep = timestep.TimeStep{}
	a.lastAction = nil

	return step

}

// Register registers an agent.Learner with the environment so that
// the TD error can be calculated and used to update the average reward
// estimate.
//
// If the AverageReward environment was constructed with the useTDError
// parameter set to true, then this method is required to be called
// before calling the Step() method. Failure to do so will result in a
// panic. If the useTDError parameter was set to false, then this
// method will have no effect.
func (a *AverageReward) Register(l agent.Learner) {
	a.learner = l
}

// Step takes one environmental step given action a and returns the next
// timestep as a timestep.TimeStep and a bool indicating whether or not
// the episode has ended.
func (a *AverageReward) Step(action *mat.VecDense) (timestep.TimeStep, bool) {
	// If using the TD error to update the average reward estimate, then
	// Register() must have been called first.
	if a.learner == nil && a.useTDError {
		panic("when using the TD error to update the average reward, " +
			"a learner must first be registered using the Register() method.")
	}

	// Take a step in the embedded environment
	// step will be the TimeStep with S_{t+1} and R_{t} for action A_{t}
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
	// to become the differential reward: R_{t} <- R_{t} - avgReward_{t}
	step.Reward -= a.avgReward

	// Average reward setting does not have discounting
	step.Discount = 1.0

	// Update the stored states and actions so that the average reward
	// estimate can be updated for the next timestep {t+1} later
	a.secondLastStep = a.lastStep
	a.lastAction = action
	a.lastStep = step

	// Return the next state S_{t+1} and Reward R_{t} for taking Action A_{t}
	// in state S_{t}
	return step, step.Last()
}

// RewardSpec returns the reward specification for the environment
func (a *AverageReward) RewardSpec() spec.Environment {
	rewardSpec := a.Environment.RewardSpec()

	// Bounds depend on environment and policy which is constantly
	// changing, so the bounds cannot be calculated
	rewardSpec.LowerBound = nil
	rewardSpec.UpperBound = nil

	rewardSpec.Type = spec.AverageReward
	return rewardSpec
}

// DiscountSpec returns the discount specification for the environment
// Average reward setting does not use discounting, so the discount
// value is always set to 1.0.
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

// String returns a string representation of the AverageReward
//environment
func (a *AverageReward) String() string {
	return fmt.Sprintf("Average Reward: %v", a.Environment)
}
