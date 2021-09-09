package experiment

import (
	"fmt"
	"time"

	ag "github.com/samuelfneumann/golearn/agent"
	env "github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/experiment/checkpointer"
	"github.com/samuelfneumann/golearn/experiment/tracker"
	ts "github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/progressbar"
	"gonum.org/v1/gonum/mat"
)

// Online is an Experiment that runs an agent online only. No offline
// evaluation is performed.
type Online struct {
	environment   env.Environment
	agent         ag.Agent
	maxSteps      uint
	currentSteps  uint
	savers        []tracker.Tracker
	checkpointers []checkpointer.Checkpointer
	progBar       *progressbar.ProgressBar
}

// NewOnline creates and returns a new online experiment on a given
// environment with a given agent. The steps parameter determines how
// many timesteps the experiment is run for, and the s parameter
// is a slice of savers.Saver which determine what data is saved.
func NewOnline(e env.Environment, a ag.Agent, steps uint,
	t []tracker.Tracker, c []checkpointer.Checkpointer) *Online {

	// Deal with null c inputs
	var checkpointers []checkpointer.Checkpointer
	if c == nil {
		checkpointers = []checkpointer.Checkpointer{}
	} else {
		checkpointers = c
	}

	// Deal with null t inputs
	var trackers []tracker.Tracker
	if t == nil {
		trackers = []tracker.Tracker{}
	} else {
		trackers = t
	}

	// Create a progress bar for watching experiment progress
	progBar := progressbar.New(50, int(steps), time.Second, true)
	progBar.Display()

	return &Online{e, a, steps, 0, trackers, checkpointers, progBar}
}

// Register registers a saver.Saver with an Experiment so that data
// generated during the experiment can be tracked and saved
func (o *Online) Register(t tracker.Tracker) {
	o.savers = append(o.savers, t)
}

// RunEpisode runs a single episode of the experiment and returns whether
// the step limit has been reached as well as any errors that occurred
// during the episode
func (o *Online) RunEpisode() (bool, error) {
	step, err := o.environment.Reset()
	if err != nil {
		return o.currentSteps >= o.maxSteps, fmt.Errorf("runEpisode: could "+
			"not reset environment: %v", err)
	}
	o.agent.ObserveFirst(step)
	o.track(step)

	// Run the next timestep
	for !step.Last() && o.currentSteps < o.maxSteps {
		o.progBar.Increment()
		o.currentSteps++

		// Select action
		action := o.agent.SelectAction(step)

		// Step in the environment with a *copy* of the action. A copy is
		// necessary because some Agents store their actions for updates and
		// many Environments clip actions. Failing to copy would cause the
		// Agent's stored value to also be clipped.
		step, _, err = o.environment.Step(mat.VecDenseCopyOf(action))
		if err != nil {
			return o.currentSteps >= o.maxSteps, fmt.Errorf("runEpisode: "+
				"could not step environment: %v", err)
		}

		// Cache the environment step in each Saver
		o.track(step)

		// Checkpoint the experiment
		o.checkpoint(step)

		// Observe the timestep and step the agent
		o.agent.Observe(action, step)
		o.agent.Step()
	}

	o.progBar.AddMessage(fmt.Sprintf("Episode Length: %v", step.Number))

	// Return whether or not the max timestep limit has been reached
	return o.currentSteps >= o.maxSteps, nil
}

// Run runs the entire experiment for all timesteps
func (o *Online) Run() error {
	ended := false
	var err error
	o.agent.Train()

	for !ended {
		ended, err = o.RunEpisode()
		if err != nil {
			return fmt.Errorf("run: %v", err)
		}

		o.agent.EndEpisode()
	}

	// Close the environment if needed
	if env, ok := o.environment.(Closer); ok {
		env.Close()
	}
	o.progBar.Close()
	return nil
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

// checkpoint checkpoints the current state of the environment
func (o *Online) checkpoint(t ts.TimeStep) {
	for _, c := range o.checkpointers {
		c.Checkpoint(t)
	}
}

// Environment returns the environment that the experiment is run on
func (o *Online) Environment() env.Environment {
	return o.environment
}

// Agent returns the agent that the experiment is run with
func (o *Online) Agent() ag.Agent {
	return o.agent
}
