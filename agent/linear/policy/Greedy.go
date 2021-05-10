package policy

import (
	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/matutils"
)

// Greedy implements a greedy policy using linear function approximation
type Greedy struct {
	weights *mat.Dense
}

// NewGreedy creates a new Greedy policy
func NewGreedy(features, actions int) *Greedy {
	weights := mat.NewDense(actions, features, nil)
	return &Greedy{weights}
}

// Weights gets and returns the weights of the EGreedy policy as a
// string description -> weights
func (p *Greedy) Weights() map[string]*mat.Dense {
	weights := make(map[string]*mat.Dense)
	weights["weights"] = p.weights

	return weights
}

// SelectAction selects and action from the greedy policy
func (p *Greedy) SelectAction(t timestep.TimeStep) mat.Vector {
	obs := t.Observation

	// Calculate all action values
	numActions, _ := p.weights.Dims()
	actionValues := mat.NewVecDense(numActions, nil)
	actionValues.MulVec(p.weights, obs)

	// Find and return the greedy action
	action := float64(matutils.MaxVec(actionValues))
	return mat.NewVecDense(1, []float64{action})
}
