package esarsa

import (
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/matutils"
)

// ESarsaLearner implements the update functionality for the Q-Learning
// algorithm.
type ESarsaLearner struct {
	weights      *mat.Dense
	step         timestep.TimeStep
	action       int
	nextStep     timestep.TimeStep
	learningRate float64
	targetE      float64
}

// NewESarsaLearner creates a new ESarsaLearner struct
//
// weights are the weights of the policy to learn
func NewESarsaLearner(weights *mat.Dense, learningRate,
	targetE float64) *ESarsaLearner {
	step := timestep.TimeStep{}
	nextStep := timestep.TimeStep{}

	return &ESarsaLearner{weights, step, -0, nextStep, learningRate, targetE}
}

// ObserveFirst observes and records the first episodic timestep
func (q *ESarsaLearner) ObserveFirst(t timestep.TimeStep) {
	if !t.First() {
		fmt.Fprintf(os.Stderr, "Warning: ObserveFirst() should only be"+
			"called on the first timestep (current timestep = %d)", t.Number)
	}
	q.step = timestep.TimeStep{}
	q.nextStep = t
}

// Observe observes and records any timestep other than the first timestep
func (q *ESarsaLearner) Observe(action mat.Vector, nextStep timestep.TimeStep) {
	if action.Len() != 1 {
		fmt.Fprintf(os.Stderr, "Warning: value-based methods should not "+
			"have multi-dimensional actions (action dim = %d)", action.Len())
	}
	q.step = q.nextStep
	q.action = int(action.AtVec(0))
	q.nextStep = nextStep
}

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

// Step updates the weights of the Agent's Learner and Policy
func (q *ESarsaLearner) Step() {
	numActions, _ := q.weights.Dims()

	// Calculate the action values in the next state
	actionValues := mat.NewVecDense(numActions, nil)
	nextState := q.nextStep.Observation
	actionValues.MulVec(q.weights, nextState)

	// Find the target policy's probability of each action
	targetProbs := q.targetProbabilities(actionValues)

	// Create the update target
	discount := q.nextStep.Discount
	expectedQ := mat.Dot(targetProbs, actionValues)
	target := q.nextStep.Reward + discount*expectedQ

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
func (q *ESarsaLearner) Weights() map[string]*mat.Dense {
	weights := make(map[string]*mat.Dense)
	weights["weights"] = q.weights

	return weights
}
