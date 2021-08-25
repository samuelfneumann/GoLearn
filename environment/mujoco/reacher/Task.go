package reacher

import (
	"fmt"

	"golang.org/x/exp/rand"

	"github.com/samuelfneumann/golearn/environment"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	"gonum.org/v1/gonum/stat/distmv"
)

// Reach implements the Reach task. This is the same task used
// in the OpenAI Gym implementation of Reacher-v2. In this task, a
// Reacher must place its fingertip within a target location. The
// reward is a function of the distance between the Reacher's fingertip
// and the target location as well as a function of the action taken.
// This encourages the Reacher to find the target fast, while penalizing
// very large actions. Episodes are ended after a time limit.
//
// The Reach Task must be registered with a Reacher before it can
// be used.
type Reach struct {
	env        *Reacher
	registered bool
	*environment.StepLimit

	// Random number generation for starting states
	seed    uint64
	posRng  *distmv.Uniform
	velRng  *distmv.Uniform
	goalRng *distmv.Uniform
}

// NewReach returns a new Reach Task.
func NewReach(seed uint64, cutoff int) environment.Task {
	stepLimit := environment.NewStepLimit(cutoff).(*environment.StepLimit)

	return &Reach{
		seed:       seed,
		registered: false,
		StepLimit:  stepLimit,
	}
}

// AtGoal returns whether the (x, y, z) position determined by the
// argument state is near the goal state.
func (r *Reach) AtGoal(state mat.Matrix) bool {
	rows, c := state.Dims()
	if c != 1 || rows != 3 {
		panic(fmt.Sprintf("atGoal: argument state should be (x, y, z) " +
			"coordinates"))
	}
	goalPos, err := r.env.BodyXPos("target")
	if err != nil {
		panic(fmt.Sprintf("atGoal: could not find target: %v", err))
	}

	goalRadius, err := r.env.GeomBoundingSphereRadius("target")
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

// GetReward returns the reward for some transition
func (r *Reach) GetReward(state, action, nextState mat.Vector) float64 {
	if !r.registered {
		panic("getReward: must register with Reacher environment first")
	}
	if state.Len() != 3 || nextState.Len() != 3 {
		panic(fmt.Sprintf("getReward: state and nextState should provide " +
			"(x, y, z) coordinates of Reacher fingertip"))
	}

	goalPos, err := r.env.BodyXPos("target")
	if err != nil {
		panic(fmt.Sprintf("getReward: could not get "+
			"target centre of mass: %v", err))
	}

	distVec := mat.NewVecDense(state.Len(), nil)
	distVec.SubVec(nextState, goalPos)
	rewardDist := mat.Norm(distVec, 2.0)

	rewardCtrl := mat.Dot(action, action)

	return -(rewardDist + rewardCtrl)
}

// Start returns a starting state for a new episode. The start state
// is a vector of [p⃗^T, v⃗^T], where p⃗ is the position vector of joints,
// v⃗ is the velocity vector, and ^T indicates a transpose.
func (r *Reach) Start() *mat.VecDense {
	// Starting position
	qposRand := r.posRng.Rand(nil)
	qposStart := mat.NewVecDense(r.env.Nq, qposRand)
	qposStart.AddVec(qposStart, r.env.InitQPos)

	// Adjust target position
	var goal *mat.VecDense
	for {
		goal = mat.NewVecDense(2, r.goalRng.Rand(nil))
		if mat.Norm(goal, 2.0) < 0.2 {
			break
		}
	}
	qposStart.SetVec(qposStart.Len()-2, goal.AtVec(0))
	qposStart.SetVec(qposStart.Len()-1, goal.AtVec(1))

	// Starting velocity
	qvelRand := r.velRng.Rand(nil)
	qvelStart := mat.NewVecDense(r.env.Nq, qvelRand)
	qvelStart.AddVec(qvelStart, r.env.InitQVel)

	// Adjust target velocity
	qvelStart.SetVec(qvelStart.Len()-2, 0.0)
	qvelStart.SetVec(qvelStart.Len()-1, 0.0)

	// Create the start state vector
	backing := make([]float64, qposStart.Len()+qvelStart.Len())
	copy(backing[:qposStart.Len()], qposStart.RawVector().Data)
	copy(backing[qposStart.Len():], qvelStart.RawVector().Data)

	// Get the starting state, not the starting observation
	return mat.NewVecDense(r.env.Nq+r.env.Nv, backing)
}

// register registers a Reacher environment with the Reach Task. This
// is required because a Reach Task must search through object in the
// MuJoCo environment to calculate rewards.
func (r *Reach) register(env *Reacher) {
	// Track the Reacher environment
	r.env = env

	// Position starting RNG
	posSrc := rand.NewSource(r.seed)
	posBounds := make([]r1.Interval, r.env.Nq)
	for i := range posBounds {
		posBounds[i] = r1.Interval{Min: -0.1, Max: 0.1}
	}
	r.posRng = distmv.NewUniform(posBounds, posSrc)

	// Velocity starting RNG
	velSrc := rand.NewSource(r.seed)
	velBounds := make([]r1.Interval, r.env.Nv)
	for i := range velBounds {
		velBounds[i] = r1.Interval{Min: -0.005, Max: 0.005}
	}
	r.velRng = distmv.NewUniform(velBounds, velSrc)

	// Goal RNG
	goalSrc := rand.NewSource(r.seed)
	goalBounds := make([]r1.Interval, 2)
	goalBounds[0] = r1.Interval{Min: -0.2, Max: 0.2}
	goalBounds[1] = r1.Interval{Min: -0.2, Max: 0.2}
	r.goalRng = distmv.NewUniform(goalBounds, goalSrc)

	r.registered = true
}
