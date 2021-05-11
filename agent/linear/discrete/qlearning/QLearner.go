package qlearning

import (
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/timestep"
)

// QLearner implements the update functionality for the Q-Learning
// algorithm.
type QLearner struct {
	weights      *mat.Dense
	step         timestep.TimeStep
	action       int
	nextStep     timestep.TimeStep
	learningRate float64
}

// NewQLearner creates a new QLearner struct
//
// weights are the weights of the policy to learn
func NewQLearner(weights *mat.Dense, learningRate float64) *QLearner {
	step := timestep.TimeStep{}
	nextStep := timestep.TimeStep{}

	return &QLearner{weights, step, -0, nextStep, learningRate}
}

// ObserveFirst observes and records the first episodic timestep
func (q *QLearner) ObserveFirst(t timestep.TimeStep) {
	if !t.First() {
		fmt.Fprintf(os.Stderr, "Warning: ObserveFirst() should only be"+
			"called on the first timestep (current timestep = %d)", t.Number)
	}
	q.step = timestep.TimeStep{}
	q.nextStep = t
}

// Observe observes and records any timestep other than the first timestep
func (q *QLearner) Observe(action mat.Vector, nextStep timestep.TimeStep) {
	if action.Len() != 1 {
		fmt.Fprintf(os.Stderr, "Warning: value-based methods should not "+
			"have multi-dimensional actions (action dim = %d)", action.Len())
	}
	q.step = q.nextStep
	q.action = int(action.AtVec(0))
	q.nextStep = nextStep
}

// Step updates the weights of the Agent's Learner and Policy
func (q *QLearner) Step() {
	numActions, _ := q.weights.Dims()

	// Calculate the action values in the next state
	actionValues := mat.NewVecDense(numActions, nil)
	nextState := q.nextStep.Observation
	actionValues.MulVec(q.weights, nextState)

	// Find the maximum action value in the next state
	maxVal := mat.Max(actionValues)

	// Create the update target
	discount := q.nextStep.Discount
	target := q.nextStep.Reward + discount*maxVal

	// Find the current estimate of the taken action
	weights := q.weights.RowView(q.action)
	state := q.step.Observation
	currentEstimate := mat.Dot(weights, state)

	// Construct the scaling factor of the gradient
	scale := q.learningRate * (target - currentEstimate)

	// Perform gradient descent: ∇weights = scale * state
	newWeights := mat.NewVecDense(weights.Len(), nil)
	newWeights.AddScaledVec(weights, scale, state)
	q.weights.SetRow(q.action, mat.Col(nil, 0, newWeights))

}

// Weights gets and returns the weights of the learner
func (q *QLearner) Weights() map[string]*mat.Dense {
	weights := make(map[string]*mat.Dense)
	weights["weights"] = q.weights

	return weights
}
