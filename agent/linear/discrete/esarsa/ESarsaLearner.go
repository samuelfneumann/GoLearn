package esarsa

import (
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/agent/linear/discrete/policy"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/matutils"
)

// ESarsaLearner implements the update functionality for the
// online Expected Sarsa algorithm.
type ESarsaLearner struct {
	weights *mat.Dense

	// Store the latest transition
	step     timestep.TimeStep
	action   int
	nextStep timestep.TimeStep

	learningRate float64

	// ϵ of ϵ-greedy target policy
	targetE float64
}

// NewESarsaLearner creates a new ESarsaLearner struct
//
// egreedy is the policy.EGreedy to learn
func NewESarsaLearner(egreedy *policy.EGreedy, learningRate,
	targetE float64) (*ESarsaLearner, error) {
	step := timestep.TimeStep{}
	nextStep := timestep.TimeStep{}

	learner := &ESarsaLearner{nil, step, 0, nextStep, learningRate, targetE}
	weights := egreedy.Weights()
	err := learner.SetWeights(weights)
	return learner, err
}

// ObserveFirst observes and records the first episodic timestep
func (e *ESarsaLearner) ObserveFirst(t timestep.TimeStep) {
	if !t.First() {
		fmt.Fprintf(os.Stderr, "Warning: ObserveFirst() should only be"+
			"called on the first timestep (current timestep = %d)", t.Number)
	}
	e.step = timestep.TimeStep{}
	e.nextStep = t
}

// Observe observes and records any timestep other than the first timestep
func (e *ESarsaLearner) Observe(action mat.Vector, nextStep timestep.TimeStep) {
	if action.Len() != 1 {
		fmt.Fprintf(os.Stderr, "Warning: value-based methods should not "+
			"have multi-dimensional actions (action dim = %d)", action.Len())
	}
	e.step = e.nextStep
	e.action = int(action.AtVec(0))
	e.nextStep = nextStep
}

// targetProbabilites returns the probability of taking each action under
// the target policy
func (e *ESarsaLearner) targetProbabilities(actionValues mat.Vector) mat.Vector {
	prob := make([]float64, 0, actionValues.Len())
	numActions, _ := e.weights.Dims()
	epsProb := e.targetE / float64(numActions)

	// Calculate the ε probability of taking each action
	for i := 0; i < actionValues.Len(); i++ {
		prob = append(prob, epsProb)
	}
	maxAction := matutils.MaxVec(actionValues)
	prob[maxAction] += (1.0 - e.targetE)

	return mat.NewVecDense(len(prob), prob)
}

// TdError calculates the TD error generated by the learner on some
// transition.
func (e *ESarsaLearner) TdError(t timestep.Transition) float64 {
	if t.Action.Len() > 1 || t.NextAction.Len() > 1 {
		panic("actions should be 1-dimensional")
	}
	action := int(t.Action.AtVec(0))
	actionVal := mat.Dot(e.weights.RowView(action), t.State)

	// Find the next action values
	numActions, _ := e.weights.Dims()
	nextActionValues := mat.NewVecDense(numActions, nil)
	nextActionValues.MulVec(e.weights, t.NextState)

	// Calculate the probability of taking each action under the target
	// policy
	probabilities := e.targetProbabilities(nextActionValues)

	// Calculate the expected Q value under the target policy
	expectedQ := mat.Dot(probabilities, nextActionValues)

	// Calculate the TD error
	tdError := t.Reward + t.Discount*expectedQ - actionVal

	return tdError
}

// Step updates the weights of the Agent's Learner and Policy
func (e *ESarsaLearner) Step() {
	numActions, _ := e.weights.Dims()

	// Calculate the action values in the next state
	actionValues := mat.NewVecDense(numActions, nil)
	nextState := e.nextStep.Observation
	actionValues.MulVec(e.weights, nextState)

	// Find the target policy's probability of each action
	targetProbs := e.targetProbabilities(actionValues)

	// Create the update target
	discount := e.nextStep.Discount
	expectedQ := mat.Dot(targetProbs, actionValues)
	target := e.nextStep.Reward + discount*expectedQ

	// Find the current estimate of the taken action
	weights := e.weights.RowView(e.action)
	state := e.step.Observation
	currentEstimate := mat.Dot(weights, state)

	// Construct the scaling factor of the gradient
	scale := e.learningRate * (target - currentEstimate)

	// Perform gradient descent: ∇weights = scale * state
	newWeights := mat.NewVecDense(weights.Len(), nil)
	newWeights.AddScaledVec(weights, scale, state)
	e.weights.SetRow(e.action, mat.Col(nil, 0, newWeights))

}

// Weights gets and returns the weights of the learner
func (e *ESarsaLearner) Weights() map[string]*mat.Dense {
	weights := make(map[string]*mat.Dense)
	weights[policy.WeightsKey] = e.weights

	return weights
}

// SetWeights sets the weight pointers to point to a new set of weights.
// The SetWeights function can take the output of a call to Weights()
// on another Learner or Linear Policy that has a key "weights"
func (e *ESarsaLearner) SetWeights(weights map[string]*mat.Dense) error {
	newWeights, ok := weights[policy.WeightsKey]
	if !ok {
		return fmt.Errorf("SetWeights: no weights named \"%v\"",
			policy.WeightsKey)
	}

	e.weights = newWeights
	return nil
}
