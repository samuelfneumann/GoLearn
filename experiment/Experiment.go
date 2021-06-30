// Package experiment implements functionality for running an experiment
package experiment

import (
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

	track(ts.TimeStep) // Tracks current timestep by sending it to Savers
	Save()             // Save all tracked data
	Register(t tracker.Tracker)

	// Saves the current state of all agents
	checkpoint(ts.TimeStep)
}
