// Package savers implements Savers, which track and save data in an
// experiment
package tracker

import (
	"encoding/gob"
	"log"
	"os"

	ts "github.com/samuelfneumann/golearn/timestep"
)

// Interface Tracker keeps track of experiment data and saves the data
// after the experiment has finished
type Tracker interface {
	Track(t ts.TimeStep)
	Save()
}

// LoadFDataloads and returns the data saved by a Tracker as a []float64
func LoadFData(filename string) []float64 {
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

// LoadIData loads and returns the data saved by a Tracker as a []int
func LoadIData(filename string) []int {
	// Open file
	file, err := os.Open(filename)
	if err != nil {
		log.Fatalf("could not open data file: %v", err)
	}
	defer file.Close()

	// Create the decoder and the variable to store the data in
	dec := gob.NewDecoder(file)
	var data []int

	// Decode the data
	err = dec.Decode(&data)
	if err != nil {
		log.Fatalf("could not decode data: %v", err)
	}

	return data
}
