package trackers

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"

	ts "sfneuman.com/golearn/timestep"
)

// Return tracks and saves the episodic return in an experiment. When
// an environment returns a TimeStep, this Tracker will extract the
// reward and accumulate the return for each episode in the experiment.
//
// Note: If an environment is wrapped by some environment wrapper
// which modifies rewards, then this Tracker tracks the modified rewards
// returned by the wrapped environment. For example, if an experiment
// is run on an experiments/wrappers.AverageReward environment
// wrapping Mountain Car, then this Tracker will track the cumulative
// episodic differential rewards, and not the episodic return from the
// Mountain Car Environment.
//
// Note: An episode must finish for this Tracker to save its data.
// If the last episode in an experiment does not finish, that episode's
// return will not be saved.
type Return struct {
	lastTimeStep   int
	currentReturn  float64
	episodeReturns []float64
	filename       string
}

// NewReturn creates and returns a new *Return Tracker
func NewReturn(filename string) Tracker {
	var saver Return
	saver.lastTimeStep = -1
	saver.filename = filename
	return &saver
}

// Track tracks the rewards seen on a timestep. By calling this method
// on every timestep, the Tracker will store all rewards seen in the
// episode, and save the cumulative reward for that episode as the
// episodic return. When a new episode starts, this method will
// automatically detect this and start accumulating the rewards for this
// new episode separately from the rewards seen on previous episodes.
//
// Track panics if it is called for non-sequential timesteps
func (r *Return) Track(step ts.TimeStep) {
	// Ensure that Track is called on sequential timesteps
	if r.lastTimeStep+1 != step.Number {
		msg := fmt.Sprintf("warning: last two timesteps tracked are not"+
			"sequential: timestep %v --> timestep %v were tracked",
			r.lastTimeStep, step.Number)
		panic(msg)
	}

	// Check if the timestep is the last in the episode, if so, cache
	// the episodic return and start recording return for the next
	// episode
	if !step.Last() {
		// Track return for same episode
		r.currentReturn += step.Reward
		r.lastTimeStep = step.Number
	} else {
		// Episode has ended, save the return and begin tracking the
		// return for a new episode
		r.currentReturn += step.Reward
		r.episodeReturns = append(r.episodeReturns, r.currentReturn)

		// Reset tracking variables
		r.currentReturn = 0.0
		r.lastTimeStep = -1
	}
}

// Save saves the data tracked by the Return Tracker to disk.
func (r *Return) Save() {
	// Open the file to save to
	file, err := os.Create(r.filename)
	if err != nil {
		log.Fatalf("could not open save file: %v", err)
	}
	defer file.Close()

	// Encode and save the file
	en := gob.NewEncoder(file)
	if err = en.Encode(r.episodeReturns); err != nil {
		log.Fatalf("Could not encode online return data: %v", err)
	}
}
