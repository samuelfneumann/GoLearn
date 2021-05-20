package savers

import (
	"encoding/gob"
	"fmt"
	"log"
	"os"

	ts "sfneuman.com/golearn/timestep"
)

// Interface Saver keeps track of the experiment data and saves the data
// after the experiment has finished
type Saver interface {
	Track(t ts.TimeStep)
	Save()
}

// Return tracks and saves the episodic return in an experiment
type Return struct {
	lastTimeStep   int
	currentReturn  float64
	episodeReturns []float64
	filename       string
}

// NewReturn creates and returns a new *Return Saver
func NewReturn(filename string) *Return {
	var saver Return
	saver.lastTimeStep = -1
	saver.filename = filename
	return &saver
}

// Track tracks the rewards seen on a timestep. By calling this method
// on every timestep, the Saver will store all rewards seen in the
// episode, and save the cumulative reward for that episode as the
// episodic return. When a new episode starts, this method will
// automatically detect this and start accumulating the rewards for this
// new episode separately from the rewards seen on previous episodes.
//
// Track panics if it is called for non-sequential timesteps
func (o *Return) Track(step ts.TimeStep) {
	// Ensure that Track is called on sequential timesteps
	if o.lastTimeStep+1 != step.Number {
		msg := fmt.Sprintf("warning: last two timesteps tracked are not"+
			"sequential: timestep %v --> timestep %v were tracked",
			o.lastTimeStep, step.Number)
		panic(msg)
	}

	// Check if the timestep is the last in the episode, if so, cache
	// the episodic return and start recording return for the next
	// episode
	if !step.Last() {
		// Track return for same episode
		o.currentReturn += step.Reward
		o.lastTimeStep = step.Number
	} else {
		// Episode has ended, save the return and begin tracking the
		// return for a new episode
		o.currentReturn += step.Reward
		o.episodeReturns = append(o.episodeReturns, o.currentReturn)

		// Reset tracking variables
		o.currentReturn = 0.0
		o.lastTimeStep = -1
	}
}

// Save saves the data tracked by the Online Saver to disk.
func (o *Return) Save() {
	// Open the file to save to
	file, err := os.Create(o.filename)
	if err != nil {
		log.Fatalf("could not open save file: %v", err)
	}
	defer file.Close()

	// Encode and save the file
	en := gob.NewEncoder(file)
	if err = en.Encode(o.episodeReturns); err != nil {
		log.Fatalf("Could not encode online return data: %v", err)
	}
}

// LoadData loads and returns the data saved by a Saver
func LoadData(filename string) []float64 {
	// Open file
	file, err := os.Open(filename)
	if err != nil {
		log.Fatalf("could not open data file: %v", err)
	}
	defer file.Close()

	// Create the decoder and the variable to store the data in
	dec := gob.NewDecoder(file)
	var data []float64

	// Decode the data
	err = dec.Decode(&data)
	if err != nil {
		log.Fatalf("could not decode data: %v", err)
	}

	return data
}
