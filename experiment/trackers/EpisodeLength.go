package trackers

import (
	"encoding/gob"
	"log"
	"os"

	"sfneuman.com/golearn/timestep"
)

// EpisodeLength tracks and saves the lengths of episodes in an
// experiment.
// Note that an episode must finish for this Tracker to save its data.
// If the last episode in an experiment does not finish, that episode's
// length will not be saved.
type EpisodeLength struct {
	episodeLengths []int
	filename       string
}

// NewEpisodeLength returns a new EpisodeLength saver which will save
// its data at the specified location filename
func NewEpisodeLength(filename string) Tracker {
	var saver EpisodeLength
	saver.filename = filename
	return &saver
}

// Track tracks the episode lengths in an experiment. When this function
// is called, it caches the episode length if the timestep passed to it
// is the last timestep in the episode. Otherwise, it waits to receive
// the last timestep in an episode before caching and storing the
// episode lengths, for saving later.
func (e *EpisodeLength) Track(t timestep.TimeStep) {
	if t.Last() {
		e.episodeLengths = append(e.episodeLengths, t.Number)
	}
}

// Save saves the data tracked by the EpisodeLength Tracker to disk.
func (e *EpisodeLength) Save() {
	// Open the file to save to
	file, err := os.Create(e.filename)
	if err != nil {
		log.Fatalf("could not open save file: %v", err)
	}
	defer file.Close()

	// Encode and save the file
	en := gob.NewEncoder(file)
	if err = en.Encode(e.episodeLengths); err != nil {
		log.Fatalf("Could not encode online return data: %v", err)
	}

}
