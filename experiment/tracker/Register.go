package tracker

import (
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/timestep"
)

// registeredTracker registers an Environment with some Tracker so
// that the Tracker tracks data from the registered Environment only.
// registeredTracker itself is a Tracker.
//
// The Track() and Save() methods of a register call those of the
// embedded Tracker. The only difference is that registeredTracker calls
// the Track() method of the embedded Tracker using the most recent
// TimeStep of the registered Environment, and the argument to
// registeredTracker.Track() is ignored. The logic of the embedded
// Tracker's Track() and Save() methods remain unmodified.
//
// This may be useful if an experiment is run using an Environment
// wrapper as the Environment but the data from the wrapped Environment
// is needed to be tracker. For example, if an experiment is run on
// an AverageReward Environment, this Tracker allows the wrapped
// Environment to be registered with another Tracker so that the return
// is tracked instead of the differential return of episodes.
type registeredTracker struct {
	Tracker
	env environment.Environment
}

// Register registers a new Tracker with an Environment, to track data
// from the registered Environment only. Register returns a copy of the
// argument Tracker that is registered with the argument Environment.
//
// Note: the underlying concrete type of the registered Tracker is
// lost when registering an Environment with a Tracker.
func Register(t Tracker, env environment.Environment) Tracker {
	return &registeredTracker{t, env}
}

// Track calls Track() on the embedded Tracker using the most recent
// TimeStep from the registered Environment.
//
// The TimeStep argument to this function is completely ignored,
// and is only there to ensure Register follows the Tracker interface
// to track and save data during an experiment.
func (r *registeredTracker) Track(timestep.TimeStep) {
	step := r.env.LastTimeStep()
	r.Tracker.Track(step)
}
