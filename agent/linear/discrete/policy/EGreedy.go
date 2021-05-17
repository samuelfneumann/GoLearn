// Package policy implements policies using linear function
// approximation
package policy

import (
	"fmt"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/matutils"
)

// EGreedy implements an ε-greedy policy using linear function
// approximation
type EGreedy struct {
	weights *mat.Dense
	epsilon float64
	seed    rand.Source // Seed for random number generation
}

// NewEGreedy constructs a new EGreedy policy, where e=epislon is the
// probability with which a random action is selected; features is the
// number of features in a given feature vector for the environment;
// actions are the number of actions in the environment
func NewEGreedy(e float64, seed uint64, features, actions int) *EGreedy {
	source := rand.NewSource(seed)

	weights := mat.NewDense(actions, features, nil)

	return &EGreedy{weights, e, source}

}

// Weights gets and returns the weights of the EGreedy policy as a
// string description -> weights
func (p *EGreedy) Weights() map[string]*mat.Dense {
	weights := make(map[string]*mat.Dense)
	weights["weights"] = p.weights

	return weights
}

// SelectAction selects and action from an ε-greedy policy
func (p *EGreedy) SelectAction(t timestep.TimeStep) mat.Vector {
	obs := t.Observation

	// Calculate all action values
	numActions, _ := p.weights.Dims()
	actionValues := mat.NewVecDense(numActions, nil)
	actionValues.MulVec(p.weights, obs)

	// Find the greedy action
	greedyAction := matutils.MaxVec(actionValues)

	// Calculate the ε probability of choosing any action at random
	prob := (p.epsilon) / float64(numActions)
	actionProbabilites := make([]float64, numActions)
	for i := 0; i < numActions; i++ {
		actionProbabilites[i] = prob
	}

	// Adjust the probability of choosing the greedy action
	actionProbabilites[greedyAction] += (1.0 - p.epsilon)

	// Construct a categorical distribution over actions using action
	// probabilities
	dist := distuv.NewCategorical(actionProbabilites, p.seed)

	// Sample an action given the action probabilites and return
	action := mat.NewVecDense(1, []float64{dist.Rand()})
	return action
}

// SetWeights sets the weight pointers to point to a new set of weights.
// The SetWeights function can take the output of a call to Weights()
// on another EGreedy Policy directly
func (p *EGreedy) SetWeights(weights map[string]*mat.Dense) error {
	newWeights, ok := weights["weights"]
	if !ok {
		return fmt.Errorf("SetWeights: no weights named \"weights\"")
	}

	p.weights = newWeights
	return nil
}
