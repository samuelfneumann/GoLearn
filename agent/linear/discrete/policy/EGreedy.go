// Package policy implements policies using linear function
// approximation
package policy

import (
	"fmt"

	"golang.org/x/exp/rand"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/floatutils"
	"sfneuman.com/golearn/utils/matutils"
)

const (
	// Keys for weights map: map[string]*mat.Dense
	WeightsKey string = "weights"
)

// EGreedy implements an ε-greedy policy using linear function
// approximation
type EGreedy struct {
	weights *mat.Dense
	epsilon float64
	rng     *rand.Rand // Seed for random number generation
	eval    bool
}

// NewEGreedy constructs a new EGreedy policy, where e=epislon is the
// probability with which a random action is selected; features is the
// number of features in a given feature vector for the environment;
// actions are the number of actions in the environment
func NewEGreedy(e float64, seed uint64,
	env environment.Environment) (agent.Policy, error) {
	source := rand.NewSource(seed)
	rng := rand.New(source)

	// Ensure actions are 1-dimensional
	if env.ActionSpec().Shape.Len() != 1 {
		return &EGreedy{}, fmt.Errorf("egreedy: can only use " +
			"1-dimensional actions")
	}

	// Ensure actions are discrete
	if env.ActionSpec().Cardinality != spec.Discrete {
		return &EGreedy{}, fmt.Errorf("egreedy: can only use " +
			"discrete actions")
	}

	// Calculate the number of actions
	actions := int(env.ActionSpec().UpperBound.AtVec(0)) + 1

	// Calculate the number of features
	features := env.ObservationSpec().Shape.Len()

	// Create the weight matrix: rows = actions, cols = features
	weights := mat.NewDense(actions, features, nil)

	return &EGreedy{weights, e, rng, false}, nil

}

// Weights gets and returns the weights of the EGreedy policy as a
// string description -> weights
func (p *EGreedy) Weights() map[string]*mat.Dense {
	weights := make(map[string]*mat.Dense)
	weights[WeightsKey] = p.weights

	return weights
}

// SetWeights sets the weight pointers to point to a new set of weights.
// The SetWeights function can take the output of a call to Weights()
// on another EGreedy Policy directly
func (p *EGreedy) SetWeights(weights map[string]*mat.Dense) error {
	newWeights, ok := weights[WeightsKey]
	if !ok {
		return fmt.Errorf("SetWeights: no weights named \"weights\"")
	}

	p.weights = newWeights
	return nil
}

// actionValues calculates the values of each action in a state
func (e *EGreedy) actionValues(obs mat.Vector) *mat.VecDense {
	// Calculate all action values
	numActions, _ := e.weights.Dims()
	actionValues := mat.NewVecDense(numActions, nil)
	actionValues.MulVec(e.weights, obs)

	return actionValues
}

// Eval sets the policy to evaluation mode
func (p *EGreedy) Eval() { p.eval = true }

// IsEval returns whether the policy is in evaulation mode or not
func (p *EGreedy) IsEval() bool { return p.eval }

// Train sets the policy to training mode
func (p *EGreedy) Train() { p.eval = false }

// SelectAction selects an action from an ε-greedy policy
func (p *EGreedy) SelectAction(t timestep.TimeStep) *mat.VecDense {
	obs := t.Observation

	// Calculate all action values
	actionValues := p.actionValues(obs).RawVector().Data
	numActions := len(actionValues)

	var maxIndices []int
	if p.IsEval() {
		maxIndices = floatutils.ArgMax(actionValues...)
	} else {
		// With probability epsilon return a random action
		if probability := rand.Float64(); probability < p.epsilon {
			action := rand.Int() % numActions
			return mat.NewVecDense(1, []float64{float64(action)})
		}

		// Get the actions of maximum value
		_, maxIndices = floatutils.MaxSlice(actionValues)
	}

	// If multiple actions have max value, return a random max-valued action
	action := maxIndices[p.rng.Int()%len(maxIndices)]
	return mat.NewVecDense(1, []float64{float64(action)})
}

// ActionProbabilites returns the probability of taking each action in
// a given state
func (e *EGreedy) ActionProbabilities(obs mat.Vector) mat.Vector {
	actionValues := e.actionValues(obs)

	prob := make([]float64, 0, actionValues.Len())
	numActions, _ := e.weights.Dims()
	epsProb := e.epsilon / float64(numActions)

	// Calculate the ε probability of taking each action
	for i := 0; i < actionValues.Len(); i++ {
		prob = append(prob, epsProb)
	}
	maxAction := matutils.MaxVec(actionValues)
	prob[maxAction] += (1.0 - e.epsilon)

	return mat.NewVecDense(len(prob), prob)
}
