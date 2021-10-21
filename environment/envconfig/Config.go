// Package envconfig provides configuration structs for configuring
// environments with default physical parameters and tasks. Environment
// configurations in this package are JSON serializable.
package envconfig

import (
	"fmt"

	env "github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/environment/box2d/lunarlander"
	"github.com/samuelfneumann/golearn/environment/classiccontrol/acrobot"
	"github.com/samuelfneumann/golearn/environment/classiccontrol/cartpole"
	"github.com/samuelfneumann/golearn/environment/classiccontrol/mountaincar"
	"github.com/samuelfneumann/golearn/environment/classiccontrol/pendulum"
	"github.com/samuelfneumann/golearn/environment/gridworld"
	"github.com/samuelfneumann/golearn/environment/maze"
	"github.com/samuelfneumann/golearn/environment/mujoco/hopper"
	"github.com/samuelfneumann/golearn/environment/mujoco/reacher"
	"github.com/samuelfneumann/golearn/environment/wrappers"
	ts "github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/gomaze"
	"gonum.org/v1/gonum/spatial/r1"
)

// EnvName stores the name of environments that can be configured with
// this package
type EnvName string

// Environments available for configuration
const (
	MountainCar EnvName = "MountainCar"
	Pendulum    EnvName = "Pendulum"
	Cartpole    EnvName = "Cartpole"
	Acrobot     EnvName = "Acrobot"
	Gridworld   EnvName = "Gridworld"
	LunarLander EnvName = "LunarLander"
	Hopper      EnvName = "Hopper"
	Reacher     EnvName = "Reacher"
	Maze        EnvName = "Maze"
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
// 	Acrobot				SwingUp
//						Balance (soon to come)
type TaskName string

// Tasks available for configuration
const (
	Goal    TaskName = "Goal"
	SwingUp TaskName = "SwingUp"
	Balance TaskName = "Balance"
	Land    TaskName = "Land"
	Hop     TaskName = "Hop"
	Reach   TaskName = "Reach"
)

// Config implements a specific configuration of a specific environment
// and specific task. Not all environments can have all tasks.
type Config struct {
	Environment       EnvName
	Task              TaskName
	ContinuousActions bool
	EpisodeCutoff     uint
	Discount          float64

	// Whether to use the OpenAI Gym or GoLearn environment implementation
	Gym bool

	// TileCoding indicates if tile coding should be used and if so,
	// what bins should be used
	TileCoding tileCodingConfig
}

// NewConfig returns a new environment Config describing an environment with
// default task and physical paramters
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

// CreateEnv returns the environment described by the Config as well as
// the first timestep of the environment.
func (c Config) CreateEnv(seed uint64) (env.Environment, ts.TimeStep,
	error) {
	var e env.Environment
	var step ts.TimeStep
	var err error
	switch c.Environment {
	case MountainCar:
		e, step, err = CreateMountainCar(c.ContinuousActions, c.Task,
			int(c.EpisodeCutoff), seed, c.Discount)

	case Cartpole:
		e, step, err = CreateCartpole(c.ContinuousActions, c.Task,
			int(c.EpisodeCutoff), seed, c.Discount)

	case Pendulum:
		e, step, err = CreatePendulum(c.ContinuousActions, c.Task,
			int(c.EpisodeCutoff), seed, c.Discount)

	case Acrobot:
		e, step, err = CreateAcrobot(c.ContinuousActions, c.Task,
			int(c.EpisodeCutoff), seed, c.Discount)

	case Gridworld:
		e, step, err = CreateGridworld(c.ContinuousActions, c.Task,
			int(c.EpisodeCutoff), seed, c.Discount)

	case Maze:
		e, step, err = CreateMaze(c.ContinuousActions, c.Task,
			int(c.EpisodeCutoff), seed, c.Discount)

	case LunarLander:
		e, step, err = CreateLunarLander(c.ContinuousActions, c.Task,
			int(c.EpisodeCutoff), seed, c.Discount)

	case Hopper:
		e, step, err = CreateHopper(c.ContinuousActions, c.Task,
			int(c.EpisodeCutoff), seed, c.Discount)

	case Reacher:
		e, step, err = CreateReacher(c.ContinuousActions, c.Task,
			int(c.EpisodeCutoff), seed, c.Discount)

	default:
		return nil, ts.TimeStep{}, fmt.Errorf("createEnv: cannot create "+
			"environment %v, no such environment", c.Environment)
	}

	if c.TileCoding.UseTileCoding {
		if c.TileCoding.UseIndices {
			e, step, err = wrappers.NewIndexTileCoding(e, c.TileCoding.Bins, seed)
		} else {
			e, step, err = wrappers.NewTileCoding(e, c.TileCoding.Bins, seed)
		}
	}

	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("createEnv: %v", err)
	}
	return e, step, nil

}

// CreateMountainCar is a factory for creating the MountainCar
// environment with default physical parameters and default task
// parameters.
func CreateMountainCar(continuousActions bool, taskName TaskName, cutoff int,
	seed uint64, discount float64) (env.Environment, ts.TimeStep, error) {
	position := r1.Interval{Min: -0.6, Max: -0.4}
	velocity := r1.Interval{Min: 0.0, Max: 0.0}

	s := env.NewUniformStarter([]r1.Interval{position, velocity}, seed)

	var task env.Task
	switch taskName {
	case Goal:
		task = mountaincar.NewGoal(s, cutoff, mountaincar.GoalPosition)

	default:
		return nil, ts.TimeStep{}, fmt.Errorf("createMountainCar: "+
			"MountainCar environment has no task %v", taskName)
	}

	if continuousActions {
		return mountaincar.NewContinuous(task, discount)
	}
	return mountaincar.NewDiscrete(task, discount)
}

// CreateCartpole is a factory for creating the Cartpole environment
// with default physical parameters and default task parameters.
func CreateCartpole(continuousActions bool, taskName TaskName, cutoff int,
	seed uint64, discount float64) (env.Environment, ts.TimeStep, error) {
	bounds := r1.Interval{Min: -0.05, Max: 0.05}
	s := env.NewUniformStarter([]r1.Interval{
		bounds,
		bounds,
		bounds,
		bounds,
	}, seed)

	var task env.Task
	switch taskName {
	case Balance:
		task = cartpole.NewBalance(s, cutoff, cartpole.FailAngle)

	default:
		return nil, ts.TimeStep{}, fmt.Errorf("createCartpole: Cartpole "+
			"environment has no task %v", taskName)
	}

	if continuousActions {
		return cartpole.NewContinuous(task, discount)
	}
	return cartpole.NewDiscrete(task, discount)

}

// CreatePendulum is a factory for creating the Pendulum environment
// with default physical parameters and default task parameters.
func CreatePendulum(continuousActions bool, taskName TaskName,
	cutoff int, seed uint64, discount float64) (env.Environment, ts.TimeStep,
	error) {
	angle := r1.Interval{Min: -pendulum.AngleBound, Max: pendulum.AngleBound}
	speed := r1.Interval{Min: -1.0, Max: 1.0}

	s := env.NewUniformStarter([]r1.Interval{angle, speed}, seed)

	var task env.Task
	switch taskName {
	case SwingUp:
		task = pendulum.NewSwingUp(s, cutoff)

	default:
		return nil, ts.TimeStep{}, fmt.Errorf("createPendulum: Pendulum "+
			"environment has no task %v", taskName)
	}

	if continuousActions {
		return pendulum.NewContinuous(task, discount)
	}
	return pendulum.NewDiscrete(task, discount)
}

// CreateAcrobot is a factory for creating the Acrobot environment
// with default physical parameters and default task parameters.
func CreateAcrobot(continuousActions bool, taskName TaskName,
	cutoff int, seed uint64, discount float64) (env.Environment, ts.TimeStep,
	error) {
	angle := r1.Interval{Min: -0.1, Max: 0.1}
	speed := r1.Interval{Min: -0.1, Max: 0.1}

	s := env.NewUniformStarter([]r1.Interval{angle, angle, speed, speed}, seed)

	var task env.Task
	switch taskName {
	case SwingUp:
		task = acrobot.NewSwingUp(s, cutoff, acrobot.GoalHeight)

	default:
		return nil, ts.TimeStep{}, fmt.Errorf("createAcrobot: Acrobot "+
			"environment has no task %v", taskName)
	}

	if continuousActions {
		return acrobot.NewContinuous(task, discount)
	}
	return acrobot.NewDiscrete(task, discount)
}

// CreateGridworld is a factory for creating a Gridworld environment
// with default grid size (5 x 5) and default goal task parameters
func CreateGridworld(continuousActions bool, taskName TaskName, cutoff int,
	seed uint64, discount float64) (env.Environment, ts.TimeStep, error) {
	if continuousActions {
		return nil, ts.TimeStep{}, fmt.Errorf("createGridworld: gridworlds " +
			"only support discrete actions")
	}

	// Environment parameters
	r, c := 5, 5
	goalX, goalY := 4, 4
	timestepReward := -0.1
	goalReward := 1.0

	// Create the start-state distribution - always at (0, 0)
	starter, err := gridworld.NewSingleStart(0, 0, r, c)
	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("createGridworld: could " +
			"not create starter")
	}

	var task env.Task
	switch taskName {
	case Goal:
		// Create the gridworld task of reaching a goal state. The goals
		// are specified as a []int, representing (x, y) coordinates
		goalX, goalY := []int{goalX}, []int{goalY}
		task, err = gridworld.NewGoal(starter, goalX, goalY, r, c,
			timestepReward, goalReward, cutoff)
		if err != nil {
			return nil, ts.TimeStep{}, fmt.Errorf("createGridworld: could " +
				"not create goal")
		}
	default:
		return nil, ts.TimeStep{}, fmt.Errorf("createGridworld: Gridworld "+
			"environment has no task %v", taskName)
	}

	// Create the gridworld
	return gridworld.New(r, c, task, discount)
}

// CreateMaze is a factory for creating a Maze environment
// with default grid size (10 x 15) and default goal task parameters
// of reaching the bottom right cell, with initialization in the top
// left cell. The maze is constructed with Wilson initialization.
func CreateMaze(continuousActions bool, taskName TaskName, cutoff int,
	seed uint64, discount float64) (env.Environment, ts.TimeStep, error) {
	if continuousActions {
		return nil, ts.TimeStep{}, fmt.Errorf("createGridworld: gridworlds " +
			"only support discrete actions")
	}

	// Environment parameters
	r, c := 5, 5
	goalCol, goalRow := []int{c - 1}, []int{r - 1}

	// Create the start-state distribution - always at (0, 0)
	starter := env.NewCategoricalStarter([][]int{{0}, {0}}, int64(seed))

	var task env.Task
	var err error
	switch taskName {
	case Goal:
		task, err = maze.NewGoal(starter, goalCol, goalRow, r, c, cutoff)
		if err != nil {
			return nil, ts.TimeStep{}, fmt.Errorf("createMaze: could " +
				"not create goal")
		}

	default:
		return nil, ts.TimeStep{}, fmt.Errorf("createMaze: Maze "+
			"environment has no task %v", taskName)
	}

	// Create the maze
	init := gomaze.NewWilson(int64(seed))
	return maze.New(task, r, c, init, discount)
}

// CreateLunarLander is a factory for creating the Lunar Lander
// environment with default physical parameters and default task
// parameters.
func CreateLunarLander(continuousActions bool, taskName TaskName,
	cutoff int, seed uint64, discount float64) (env.Environment, ts.TimeStep,
	error) {
	xPosition := r1.Interval{
		Min: lunarlander.InitialX,
		Max: lunarlander.InitialX,
	}
	yPosition := r1.Interval{
		Min: lunarlander.InitialY,
		Max: lunarlander.InitialY,
	}
	initialRandom := r1.Interval{
		Min: lunarlander.InitialRandom,
		Max: lunarlander.InitialRandom,
	}

	s := env.NewUniformStarter([]r1.Interval{xPosition, yPosition,
		initialRandom}, seed)

	var task env.Task
	switch taskName {
	case Land:
		task = lunarlander.NewLand(s, cutoff)

	default:
		return nil, ts.TimeStep{}, fmt.Errorf("createLunarLander: "+
			"LunarLander environment has no task %v", taskName)
	}

	if continuousActions {
		return lunarlander.NewContinuous(task, discount, seed)
	}
	return lunarlander.NewDiscrete(task, discount, seed)
}

// CreateHopper is a factory for creating the Hopper
// environment with default physical parameters and default task
// parameters.
func CreateHopper(continuousActions bool, taskName TaskName,
	cutoff int, seed uint64, discount float64) (env.Environment, ts.TimeStep,
	error) {
	if !continuousActions {
		return nil, ts.TimeStep{}, fmt.Errorf("createHopper: hopper must " +
			"have continuous actions")
	}

	var task env.Task
	switch taskName {
	case Hop:
		task = hopper.NewHop(seed, cutoff)

	default:
		return nil, ts.TimeStep{}, fmt.Errorf("createHopper: Hopper "+
			"environment has no task %v", taskName)
	}

	env, firstStep, err := hopper.New(task, 1, seed, discount)
	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("createHopper: could not "+
			"create environment: %v", err)
	}

	return env, firstStep, nil
}

// CreateReacher is a factory for creating the Reacher
// environment with default physical parameters and default task
// parameters.
func CreateReacher(continuousActions bool, taskName TaskName,
	cutoff int, seed uint64, discount float64) (env.Environment, ts.TimeStep,
	error) {
	if !continuousActions {
		return nil, ts.TimeStep{}, fmt.Errorf("createReacher: reacher must " +
			"have continuous actions")
	}
	var task env.Task
	switch taskName {
	case Reach:
		task = reacher.NewReach(seed, cutoff)

	default:
		return nil, ts.TimeStep{}, fmt.Errorf("createReacher: Reacher "+
			"environment has no task %v", taskName)
	}

	env, firstStep, err := reacher.New(task, 2, seed, discount)
	if err != nil {
		return nil, ts.TimeStep{}, fmt.Errorf("createReacher: could not "+
			"create environment: %v", err)
	}

	return env, firstStep, nil
}

// tileCodingConfig implements configuration settings for tile coding
// of environments. A separate struct for the environment config is
// used to make the JSON file look prettier.
type tileCodingConfig struct {
	UseTileCoding bool

	// UseIndices determines whether the tile-coded feature vector
	// should be represented as a one-hot vector or as the non-zero
	// indices of the one-hot vector.
	UseIndices bool

	// Bins determines the number of tilings and tiles per tiling.
	// if len(Bins) == m, then m tilings are used.
	// If Bins[m] = []int{m, n, p}, then there are m bins along the
	// first dimension, n along the second, and p along the third.
	Bins [][]int
}
