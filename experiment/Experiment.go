// Package experiment implements functionality for running an experiment
package experiment

import (
	"fmt"

	"github.com/samuelfneumann/golearn/agent"
	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/environment/envconfig"
	"github.com/samuelfneumann/golearn/experiment/checkpointer"
	"github.com/samuelfneumann/golearn/experiment/tracker"
	ts "github.com/samuelfneumann/golearn/timestep"
)

type Closer interface {
	Close()
}

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
	Run() error

	// RunEpisode runs a single episode and returns in the step limit was
	// reached as well as if any errors occurred during the run
	RunEpisode() (bool, error)

	// Tracks current timestep by sending it to Savers
	track(ts.TimeStep)

	// Save all tracked data to disk
	Save()

	// Adds a new tracker.Tracker to the (possibly already running) experiment.
	// Useful if you want to track data only after a specified event.
	Register(t tracker.Tracker)

	// Saves the current state of all agents
	checkpoint(ts.TimeStep)

	// Getters
	Environment() environment.Environment
	Agent() agent.Agent
}

// Type describes a specific experiment type. It is used in Experiment
// configurations to create a specific type of experiment.
type Type string

const (
	OnlineExp Type = "OnlineExperiment"
)

// Config represents a configuration of an experiment.
type Config struct {
	Type
	MaxSteps    uint
	EnvConfig   envconfig.Config
	AgentConfig agent.TypedConfigList
}

// CreateExp creates the experiment determined by the Config
func (c Config) CreateExp(i int, seed uint64, t []tracker.Tracker,
	check []checkpointer.Checkpointer) (Experiment, error) {
	env, _, err := c.EnvConfig.CreateEnv(seed)
	if err != nil {
		return nil, fmt.Errorf("createExpL could not create environment: %v",
			err)
	}
	agent, err := c.AgentConfig.At(i).CreateAgent(env, seed)
	if err != nil {
		return nil, fmt.Errorf("createExp: could not create agent: %v", err)
	}

	switch c.Type {
	case OnlineExp:
		return NewOnline(env, agent, c.MaxSteps, t, check), nil
	}

	return nil, fmt.Errorf("createExp: no such experiment type %v", c.Type)
}
