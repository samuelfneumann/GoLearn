// Package policy implements policies using linear function
// approximation
package policy

import (
	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distuv"
	"sfneuman.com/golearn/timestep"
)

// EGreedy implements an ε-greedy policy using linear function
// approximation
type EGreedy struct {
	weights      *mat.Dense
	GreedyPolicy *Greedy
	epsilon      float64
	seed         rand.Source // Seed for random number generation
}

// NewEGreedy constructs a new EGreedy policy, where e=epislon is the
// probability with which a random action is selected
func NewEGreedy(e float64, seed uint64, features, actions int) *EGreedy {
	source := rand.NewSource(seed)

	weights := mat.NewDense(actions, features, nil)
	greedyPolicy := &Greedy{weights} // Share weights between both Policies

	return &EGreedy{weights, greedyPolicy, e, source}

}

// Weights gets and returns the weights of the EGreedy policy as a
// string description -> weights
func (p *EGreedy) Weights() map[string]*mat.Dense {
	weights := make(map[string]*mat.Dense)
	weights["weights"] = p.weights

	return weights
}

// SelectAction selects and action from an ε-greedy policy
func (e *EGreedy) SelectAction(t timestep.TimeStep) mat.Vector {
	// Get the greedy action
	greedyAction := int(e.GreedyPolicy.SelectAction(t).AtVec(0))

	// Calculate the ε probability of choosing any action at random
	numActions, _ := e.weights.Dims()
	prob := (e.epsilon) / float64(numActions)
	actionProbabilites := make([]float64, numActions)
	for i := 0; i < numActions; i++ {
		actionProbabilites[i] = prob
	}

	// Adjust the probability of choosing the greedy action
	actionProbabilites[greedyAction] += (1.0 - e.epsilon)

	// Construct a categorical distribution over actions using action
	// probabilities
	dist := distuv.NewCategorical(actionProbabilites, e.seed)

	// Sample an action given the action probabilites and return
	action := mat.NewVecDense(1, []float64{dist.Rand()})
	return action
}
