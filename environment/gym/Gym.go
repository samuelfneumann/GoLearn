// Package gym provides access to use OpenAI's Gym environments in
// GoLearn.
//
// All environments in the Classic Control and MuJoCo suites can be
// used. All environments only work with their default tasks and episode
// cutoffs. Once GoGym implements functionality for chaning the
// episode cutoffs, this package will also be changed.
//
// This is made possible through the Go bindings for OpenAI Gym,
// found at https://github.com/samuelfneumann/GoGym.
package gym

import (
	"fmt"

	"github.com/samuelfneumann/gogym"
	env "github.com/samuelfneumann/golearn/environment"
	ts "github.com/samuelfneumann/golearn/timestep"
	"gonum.org/v1/gonum/mat"
)

// GymEnv implements access to an OpenAI Gym environment using GoGym
type GymEnv struct {
	gogym.Environment

	currentStep ts.TimeStep
	discount    float64
}

// New returns a new GymEnv with the given name, which must be a legal
// name from the OpenAI Gym suite.
func New(name string, discount float64, seed uint64) (env.Environment,
	ts.TimeStep, error) {
	goGymEnv, err := gogym.Make(name)
	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("new: could not create "+
			"environment: %v", err)
	}

	goGymEnv.Seed(int(seed))
	obs, err := goGymEnv.Reset()
	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("new: could not reset "+
			"environment: %v", err)
	}

	gymEnv := &GymEnv{
		Environment: goGymEnv,
		discount:    discount,
	}

	t := ts.New(ts.First, 0, discount, obs, 0)
	gymEnv.currentStep = t

	return gymEnv, t, nil
}

// Step takes a single environmental step
func (g *GymEnv) Step(a *mat.VecDense) (ts.TimeStep, bool, error) {
	obs, reward, done, err := g.Environment.Step(a)
	if err != nil {
		return ts.TimeStep{}, true, fmt.Errorf("step: could not step "+
			"GoGym environment: %v", err)
	}

	t := ts.New(ts.Mid, reward, g.discount, obs, g.CurrentTimeStep().Number+1)
	if done {
		t.StepType = ts.Last
	}
	g.currentStep = t

	return t, done, nil
}

// Reset resets the environment to some starting state
func (g *GymEnv) Reset() (ts.TimeStep, error) {
	obs, err := g.Environment.Reset()
	if err != nil {
		return ts.TimeStep{}, fmt.Errorf("new: could not reset "+
			"environment: %v", err)
	}

	t := ts.New(ts.First, 0, g.discount, obs, 0)
	g.currentStep = t

	return t, nil
}

// CurrentTimeStep returns the current timestep in the environment
func (g *GymEnv) CurrentTimeStep() ts.TimeStep {
	return g.currentStep
}

// ObservationSpec returns the observation spec of the environment
func (g *GymEnv) ObservationSpec() env.Spec {
	space := g.ObservationSpace()

	var low, high, shape *mat.VecDense
	switch space.(type) {
	case *gogym.BoxSpace, *gogym.DiscreteSpace:
		low = space.Low()[0]
		high = space.High()[0]
		shape = mat.NewVecDense(low.Len(), nil)
	default:
		panic("observationSpec: invalid space type, package gym supports " +
			"only GoGym's BoxSpace or DiscreteSpace")
	}

	return env.NewSpec(shape, env.Observation, low, high, env.Continuous)
}

// ActionSpec returns the action specification of the environment
func (g *GymEnv) ActionSpec() env.Spec {
	space := g.ActionSpace()

	var low, high, shape *mat.VecDense
	switch space.(type) {
	case *gogym.BoxSpace, *gogym.DiscreteSpace:
		low = space.Low()[0]
		high = space.High()[0]
		shape = mat.NewVecDense(low.Len(), nil)
	default:
		panic("observationSpec: invalid space type, package gym supports " +
			"only GoGym's BoxSpace or DiscreteSpace")
	}

	return env.NewSpec(shape, env.Action, low, high, env.Continuous)
}

// DiscountSpec returns the discount specification of the environment
func (g *GymEnv) DiscountSpec() env.Spec {
	shape := mat.NewVecDense(1, nil)
	low := mat.NewVecDense(1, []float64{g.discount})

	return env.NewSpec(shape, env.Discount, low, low, env.Continuous)
}

// Start implements the environment.Environment interface. This function
// panics.
func (g *GymEnv) Start() *mat.VecDense {
	panic("start: cannot caluclate starting state for GymEnv")
}

// GetReward implements the environment.Environment interface. This
// function panics.
func (g *GymEnv) GetReward(_, _, _ mat.Vector) float64 {
	panic("getReward: cannot calculate reward for transition in GymEnv")
}

// End implements the environment.Environment interface. This
// function panics.
func (g *GymEnv) End(*ts.TimeStep) bool {
	panic("end: cannot calulate ending for GymEnv")
}

// AtGoal implements the environment.Environment interface. This
// function panics.
func (g *GymEnv) AtGoal(mat.Matrix) bool {
	panic("atGoal: cannot calculate at goal for GymEnv")
}

// Close performs resource cleanup after the environment is no longer
// needed
func (g *GymEnv) Close() error {
	g.Environment.Close()
	return nil
}
