package experiment

import (
	"sfneuman.com/golearn/agent"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/experiment/trackers"
	ts "sfneuman.com/golearn/timestep"
)

// Online is an Experiment that runs an agent online only. No offline
// evaluation is performed.
type Online struct {
	env.Environment
	agent.Agent
	maxSteps     uint
	currentSteps uint
	savers       []trackers.Tracker
}

// NewOnline creates and returns a new online experiment on a given
// environment with a given agent. The steps parameter determines how
// many timesteps the experiment is run for, and the s parameter
// is a slice of savers.Saver which determine what data is saved.
func NewOnline(e env.Environment, a agent.Agent, steps uint,
	t []trackers.Tracker) *Online {
	return &Online{e, a, steps, 0, t}
}

// Register registers a saver.Saver with an Experiment so that data
// generated during the experiment can be tracked and saved
func (o *Online) Register(t trackers.Tracker) {
	o.savers = append(o.savers, t)
}

// RunEpisode runs a single episode of the experiment
func (o *Online) RunEpisode() bool {
	step := o.Environment.Reset()
	o.Agent.ObserveFirst(step)
	o.track(step)

	// Run the next timestep
	for !step.Last() && o.currentSteps < o.maxSteps {
		o.currentSteps++

		// Select action, step in environment
		action := o.Agent.SelectAction(step)
		step, _ = o.Environment.Step(action)

		// Cache the environment step in each Saver
		o.track(step)

		// Observe the timestep and step the agent
		o.Agent.Observe(action, step)
		o.Agent.Step()
	}

	// Return whether or not the max timestep limit has been reached
	return o.currentSteps >= o.maxSteps
}

// Run runs the entire experiment for all timesteps
func (o *Online) Run() {
	ended := false
	o.Agent.Train()

	for !ended {
		ended = o.RunEpisode()
		o.Agent.EndEpisode()
	}
}

// Save saves all the data cached by the Savers to disk
func (o *Online) Save() {
	for _, saver := range o.savers {
		saver.Save()
	}
}

// track tracks the current timestep by caching its data in each saver
func (o *Online) track(t ts.TimeStep) {
	for _, saver := range o.savers {
		saver.Track(t)
	}
}
