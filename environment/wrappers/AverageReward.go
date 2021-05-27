package wrappers

import (
	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
)

type AverageReward struct {
	environment.Environment
	avgReward    float64
	learningRate float64
}

func NewAverageReward(env environment.Environment, init,
	learningRate float64) *AverageReward {
	return &AverageReward{env, init, learningRate}
}

// Step function should update the average reward and return the
// differential reward. It should also modify the timestep so that
// the discount == 1.0
func (a *AverageReward) Step(action mat.Vector) {

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
