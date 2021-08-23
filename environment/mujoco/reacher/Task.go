package reacher

import (
	"fmt"

	"github.com/samuelfneumann/golearn/environment"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/stat/distmv"
)

type Reach struct {
	reacher    *Reacher
	registered bool

	environment.StepLimit

	// Random number generation for starting states
	seed   uint64
	posRng *distmv.Uniform
	velRng *distmv.Uniform
}

func NewReach(seed uint64, cutoff int) environment.Task {
	stepLimit := environment.NewStepLimit(cutoff)

	return &Reach{
		seed:       seed,
		registered: false,
		StepLimit:  stepLimit,
	}
}

// AtGoal returns whether (x, y, z) position determined by the argument
// state is at the goal state.
func (r *Reach) AtGoal(state mat.Matrix) bool {
	rows, c := state.Dims()
	if c != 1 || rows != 3 {
		panic(fmt.Sprintf("atGoal: argument state should be (x, y, z) " +
			"coordinates"))
	}
	goalPos, err := r.reacher.GetBodyCentreOfMass("target")
	if err != nil {
		panic(fmt.Sprintf("atGoal: could not find target: %v", err))
	}

	goalRadius, err := r.reacher.GetBoundingSphereRadius("target")
	if err != nil {
		panic(fmt.Sprintf("atGoal: could not get goal radius: %v", err))
	}

	currentPos := mat.NewVecDense(3, []float64{
		state.At(0, 0),
		state.At(1, 0),
		state.At(2, 0),
	})

	currentPos.SubVec(goalPos, currentPos)

	return mat.Norm(currentPos, 2) < goalRadius
}

// state and nextState should be a 3-dimensional vector of x, y, z position
// of the Reacher's fingertip's position.
func (r *Reach) GetReward(state, action, nextState mat.Vector) float64 {
	if !r.registered {
		panic("getReward: must register with Reacher environment first")
	}
	if state.Len() != 3 || nextState.Len() != 3 {
		panic(fmt.Sprintf("getReward: state and nextState should provide " +
			"(x, y, z) coordinates of Reacher fingertip"))
	}

	goalPos, err := r.reacher.GetBodyCentreOfMass("target")
	if err != nil {
		panic(fmt.Sprintf("getReward: could not get "+
			"target centre of mass: %v", err))
	}

	distVec := mat.NewVecDense(state.Len(), nil)
	distVec.SubVec(goalPos, nextState)
	rewardDist := mat.Norm(distVec, 2.0)

	a := mat.NewVecDense(action.Len(), nil)
	a.MulElemVec(action, action)
	rewardCtrl := mat.Sum(a)

	return rewardDist + rewardCtrl
}

func (r *Reacher) register(reacher *Reacher) {}
