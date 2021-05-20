// Package savers implements Savers, which track and save data in an
// experiment
package savers

import (
	"encoding/gob"
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
