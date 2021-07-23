// Package experiment implements functionality for running an experiment
package experiment

import (
	"fmt"

	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/environment/envconfig"
	"sfneuman.com/golearn/experiment/checkpointer"
	"sfneuman.com/golearn/experiment/tracker"
	ts "sfneuman.com/golearn/timestep"
)

// Interface Experiment outlines structs that can run experiments.
// Experiments will track environment TimeSteps, caching each TimeStep
// in RAM to be later saved to disk. The Save() function
// will then take all cached data and save it to disk. This is usually
// performed after an experiment has been run. The Run() method will
// run all episodes util the maximum timestep limit is reached, or some
// other ending condition is reached. The RunEpisode() function will
// run a single episode.
//
// In order to save data, Experiments use Savers. Savers determine which
// data generated during the experiment is saved. Experiments will
// send each TimeStep to Savers using the Saver's Track() method. The
// Saver then determines which data from the TimeStep it caches and
// saves. New Savers can be registered with an Experiment through the
// consturctor or through an Experiment's Register() function.
type Experiment interface {
	Run()
	RunEpisode() bool // Returns whether or not the current episode finished

	// Tracks current timestep by sending it to Savers
	track(ts.TimeStep)

	// Save all tracked data to disk
	Save()

	// Adds a new tracker.Tracker to the (possibly already running) experiment.
	// Useful if you want to track data only after a specified event.
	Register(t tracker.Tracker)

	// Saves the current state of all agents
	checkpoint(ts.TimeStep)
}

type Type string

const (
	OnlineExp Type = "OnlineExperiment"
)

// Config represents a configuration of an experiment.
type Config struct {
	Type
	MaxSteps  uint
	EnvConf   envconfig.Config
	AgentConf agent.TypedConfigList
}

func (c Config) CreateExp(i int, seed uint64, t []tracker.Tracker,
	check []checkpointer.Checkpointer) Experiment {
	env, _ := c.EnvConf.CreateEnv(seed)
	agent, err := c.AgentConf.At(i).CreateAgent(env, seed)
	if err != nil {
		panic(fmt.Sprintf("createExp: could not create agent: %v", err))
	}

	switch c.Type {
	case OnlineExp:
		return NewOnline(env, agent, c.MaxSteps, t, check)
	}

	panic(fmt.Sprintf("createExp: no such experiment type %v", c.Type))
}
