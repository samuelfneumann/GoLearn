// Package envconfig provides configuration structs for configuring
// environments with default physical parameters and tasks. Environment
// configurations in this package are JSON serializable.
package envconfig

import (
	"fmt"

	"gonum.org/v1/gonum/spatial/r1"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/environment/classiccontrol/cartpole"
	"sfneuman.com/golearn/environment/classiccontrol/mountaincar"
	"sfneuman.com/golearn/environment/classiccontrol/pendulum"
	ts "sfneuman.com/golearn/timestep"
)

// EnvName stores the name of environments that can be configured with
// this package
type EnvName string

// Environments available for configuration
const (
	MountainCar EnvName = "MountainCar"
	Pendulum    EnvName = "Pendulum"
	Cartpole    EnvName = "Cartpole"
)

// TaskName stores the tasks that can be configured with this package.
// Note that not all tasks can be used with all environments. The tasks
// that can be used with each environment are as follows:
//
//	Environment			Task
//	MountainCar			Goal
//	Cartpole			Balance
//						SwingUp (soon to come)
//	Pendulum			SwingUp
type TaskName string

// Tasks available for configuration
const (
	Goal    TaskName = "Goal"
	SwingUp TaskName = "SwingUp"
	Balance TaskName = "Balance"
)

// Config implements a specific configuration of a specific environment
// and specific task. Not all environments can have all tasks.
type Config struct {
	Environment       EnvName
	Task              TaskName
	ContinuousActions bool
	EpisodeCutoff     uint
	Discount          float64
	Gym               bool
}

// NewConfig returns a new environment Config
func NewConfig(envName EnvName, taskName TaskName, continuousActions bool,
	episodeCutoff uint, discount float64, gym bool) Config {
	if gym {
		panic("newConfig: no gym environments implemented yet")
	}

	return Config{
		Environment:       envName,
		Task:              taskName,
		ContinuousActions: continuousActions,
		EpisodeCutoff:     episodeCutoff,
		Discount:          discount,
		Gym:               gym,
	}
}

// Create returns the environment described by the Config as well as
// the first timestep of the environment.
func (c Config) Create(seed uint64) (env.Environment, ts.TimeStep) {
	switch c.Environment {
	case MountainCar:
		return CreateMountainCar(c.ContinuousActions, c.Task,
			int(c.EpisodeCutoff), seed, c.Discount)

	case Cartpole:
		return CreateCartpole(c.ContinuousActions, c.Task,
			int(c.EpisodeCutoff), seed, c.Discount)

	case Pendulum:
		return CreatePendulum(c.ContinuousActions, c.Task,
			int(c.EpisodeCutoff), seed, c.Discount)
	}

	panic(fmt.Sprintf("create: cannot create environment %v, no such "+
		"environment", c.Environment))
}

// CreateMountainCar is a factory for creating the MountainCar
// environment with default physical parameters and default task
// parameters.
func CreateMountainCar(continuousActions bool, taskName TaskName, cutoff int,
	seed uint64, discount float64) (env.Environment, ts.TimeStep) {
	position := r1.Interval{Min: -0.6, Max: -0.4}
	velocity := r1.Interval{Min: 0.0, Max: 0.0}

	s := env.NewUniformStarter([]r1.Interval{position, velocity}, seed)

	var task env.Task
	switch taskName {
	case Goal:
		task = mountaincar.NewGoal(s, cutoff, mountaincar.GoalPosition)

	default:
		panic(fmt.Sprintf("createMountainCar: MountainCar environment has "+
			"no task %v", taskName))
	}

	if continuousActions {
		return mountaincar.NewContinuous(task, discount)
	}
	return mountaincar.NewDiscrete(task, discount)
}

// CreateCartpole is a factory for creating the Cartpole environment
// with default physical parameters and default task parameters.
func CreateCartpole(continuousActions bool, taskName TaskName, cutoff int,
	seed uint64, discount float64) (env.Environment, ts.TimeStep) {
	bounds := r1.Interval{Min: -0.05, Max: 0.05}
	s := env.NewUniformStarter([]r1.Interval{
		bounds,
		bounds,
		bounds,
		bounds,
	}, seed)

	var task env.Task
	switch taskName {
	case Goal:
		task = cartpole.NewBalance(s, cutoff, cartpole.FailAngle)

	default:
		panic(fmt.Sprintf("createCartpole: Cartpole environment has "+
			"no task %v", taskName))
	}

	if continuousActions {
		return cartpole.NewContinuous(task, discount)
	}
	return cartpole.NewDiscrete(task, discount)

}

// CreatePendulum is a factory for creating the Pendulum environment
// with default physical parameters and default task parameters.
func CreatePendulum(continuousActions bool, taskName TaskName,
	cutoff int, seed uint64, discount float64) (env.Environment, ts.TimeStep) {
	angle := r1.Interval{Min: -pendulum.AngleBound, Max: pendulum.AngleBound}
	speed := r1.Interval{Min: -1.0, Max: 1.0}

	s := env.NewUniformStarter([]r1.Interval{angle, speed}, seed)

	var task env.Task
	switch taskName {
	case SwingUp:
		task = pendulum.NewSwingUp(s, cutoff)

	default:
		panic(fmt.Sprintf("createPendulum: Pendulum environment has "+
			"no task %v", taskName))
	}

	if continuousActions {
		return pendulum.NewContinuous(task, discount)
	}
	return pendulum.NewDiscrete(task, discount)
}
