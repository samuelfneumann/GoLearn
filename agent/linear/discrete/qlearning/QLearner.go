package qlearning

import (
	"fmt"

	"github.com/samuelfneumann/golearn/agent/linear/discrete/policy"
	"github.com/samuelfneumann/golearn/timestep"
	"gonum.org/v1/gonum/mat"
)

// QLearner implements the update functionality for the Q-Learning
// algorithm.
type QLearner struct {
	// Store policy weights instead of policy to increase computational
	// efficiency. If policy was stored, then Weights() would need to
	// be called often
	weights *mat.Dense

	// Store the latest transition as multiple TimeSteps instead of
	// timestep.Transition for efficiency. This way, a Transition
	// doesn't need to be constructed on each Step(), but instead
	// pointers can be reassigned quickly
	step     timestep.TimeStep
	action   int
	nextStep timestep.TimeStep

	learningRate float64

	// indexTileCoding represents whether the environment is using
	// tile coding and returning the non-zero indices as features
	indexTileCoding bool
}

// NewQLearner creates a new QLearner struct
//
// egreedy is the policy.EGreedy to learn
func NewQLearner(egreedy *policy.EGreedy,
	learningRate float64, indexTileCoding bool) (*QLearner, error) {
	step := timestep.TimeStep{}
	nextStep := timestep.TimeStep{}

	learner := &QLearner{nil, step, 0, nextStep, learningRate, indexTileCoding}
	weights := egreedy.Weights()

	err := learner.SetWeights(weights)

	return learner, err
}

// ObserveFirst observes and records the first episodic timestep
func (q *QLearner) ObserveFirst(t timestep.TimeStep) error {
	if !t.First() {
		return fmt.Errorf("observeFirst: timestep "+
			"called on the first timestep (current timestep = %d)", t.Number)
	}
	q.step = timestep.TimeStep{}
	q.nextStep = t

	return nil
}

// Observe observes and records any timestep other than the first timestep
func (q *QLearner) Observe(action mat.Vector,
	nextStep timestep.TimeStep) error {
	if action.Len() != 1 {
		return fmt.Errorf("observe: cannot observe multi-dimensional "+
			"actions (action dim = %d)", action.Len())
	}
	q.step = q.nextStep
	q.action = int(action.AtVec(0))
	q.nextStep = nextStep

	return nil
}

// TdError calculates the TD error generated by the learner on some
// transition.
func (q *QLearner) TdError(t timestep.Transition) float64 {
	if t.Action.Len() > 1 || t.NextAction.Len() > 1 {
		panic("actions should be 1-dimensional")
	}

	action := int(t.Action.AtVec(0))
	actionVal := mat.Dot(q.weights.RowView(action), t.State)

	// Find the max next action value
	numActions, _ := q.weights.Dims()
	nextActionValues := mat.NewVecDense(numActions, nil)
	if q.indexTileCoding {
		for _, i := range t.NextState.RawVector().Data {
			nextActionValues.AddVec(
				nextActionValues,
				q.weights.ColView(int(i)),
			)
		}
	} else {
		nextActionValues.MulVec(q.weights, t.NextState)
	}
	nextActionVal := mat.Max(nextActionValues)

	tdError := t.Reward + t.Discount*nextActionVal - actionVal

	return tdError
}

// stepIndex updates the weights of the Agent's Learner and Policy
// assuming that the last seen feature vector was of the form returned
// by environment/wrappers.IndexTileCoding, that is, the feature vector
// records the indices of non-zero components of a tile-coded state
// observation vector.
func (q *QLearner) stepIndex() {
	numActions, _ := q.weights.Dims()
	actionValues := mat.NewVecDense(numActions, nil)

	for _, i := range q.nextStep.Observation.RawVector().Data {
		actionValues.AddVec(actionValues, q.weights.ColView(int(i)))
	}
	maxVal := mat.Max(actionValues)

	// Create the update target
	discount := q.nextStep.Discount
	target := q.nextStep.Reward + discount*maxVal

	// Find current estimate of the taken action
	currentEstimate := 0.0
	for _, i := range q.step.Observation.RawVector().Data {
		currentEstimate += q.weights.At(q.action, int(i))
	}

	scale := q.learningRate * (target - currentEstimate)

	// Upate weights
	for _, i := range q.step.Observation.RawVector().Data {
		w := q.weights.At(q.action, int(i))
		newW := w + scale
		q.weights.Set(q.action, int(i), newW)
	}
}

// Step updates the weights of the Agent's Learner and Policy
func (q *QLearner) Step() error {
	if q.indexTileCoding {
		q.stepIndex()
	} else {
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
		weights.(*mat.VecDense).AddScaledVec(weights, scale, state)
	}
	return nil
}

// SetWeights sets the weight pointers to point to a new set of weights.
// The SetWeights function can take the output of a call to Weights()
// on another Learner or Linear Policy that has a key "weights"
func (e *QLearner) SetWeights(weights map[string]*mat.Dense) error {
	newWeights, ok := weights[policy.WeightsKey]
	if !ok {
		return fmt.Errorf("SetWeights: no weights named \"%v\"",
			policy.WeightsKey)
	}

	e.weights = newWeights
	return nil
}

// Cleanup at the end of an episode
func (q *QLearner) EndEpisode() {}
