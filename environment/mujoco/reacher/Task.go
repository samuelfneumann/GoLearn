package reacher

import (
	"fmt"
	"math"

	"golang.org/x/exp/rand"

	"github.com/samuelfneumann/golearn/environment"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	"gonum.org/v1/gonum/stat/distmv"
)

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

func NewReach(seed uint64, cutoff int) environment.Task {
	stepLimit := environment.NewStepLimit(cutoff).(*environment.StepLimit)

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

// ! ERROR: goalPos moves, state and nextState are always [0 0 0.1]
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

// Max returns the maximum possible reward
func (r *Reach) Max() float64 {
	return math.Inf(1.0)
}

// Min returns the minimum possible reward
func (r *Reach) Min() float64 {
	return math.Inf(-1.0)
}

// RewardSpec returns the reward specification for the environment
func (r *Reach) RewardSpec() environment.Spec {
	shape := mat.NewVecDense(1, nil)
	low := mat.NewVecDense(1, []float64{r.Min()})
	high := mat.NewVecDense(1, []float64{r.Max()})

	return environment.NewSpec(shape, environment.Reward, low, high,
		environment.Continuous)
}

func (r *Reach) Start() *mat.VecDense {
	qposRand := r.posRng.Rand(nil)
	qposStart := mat.NewVecDense(r.env.Nq, qposRand)
	qposStart.AddVec(qposStart, r.env.InitQPos)

	// Set target position
	var goal *mat.VecDense
	for {
		goal = mat.NewVecDense(2, r.goalRng.Rand(nil))
		if mat.Norm(goal, 2.0) < 0.2 {
			break
		}
	}
	qposStart.SetVec(qposStart.Len()-2, goal.AtVec(0))
	qposStart.SetVec(qposStart.Len()-1, goal.AtVec(1))

	qvelRand := r.velRng.Rand(nil)
	qvelStart := mat.NewVecDense(r.env.Nq, qvelRand)
	qvelStart.AddVec(qvelStart, r.env.InitQVel)

	// Target has no velocity
	qvelStart.SetVec(qvelStart.Len()-2, 0.0)
	qvelStart.SetVec(qvelStart.Len()-1, 0.0)

	backing := make([]float64, qposStart.Len()+qvelStart.Len())
	copy(backing[:qposStart.Len()], qposStart.RawVector().Data)
	copy(backing[qposStart.Len():], qvelStart.RawVector().Data)

	// Get the starting state, not the starting observation
	return mat.NewVecDense(r.env.Nq+r.env.Nv, backing)
}

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
