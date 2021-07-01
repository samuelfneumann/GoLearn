package checkpointer

import (
	"encoding/gob"
	"fmt"
	"os"

	ts "sfneuman.com/golearn/timestep"
)

// nStep implements checkpointing every N steps by saving a Serializable
// to a file every n environmental steps in an experiment.
type nStep struct {
	interval int
	object   Serializable // Object to save

	// filename returns the string filename of the file to save the object
	// in.
	//
	// If each serialized object should be saved in a separate file with
	// each file having an incremented number as a suffix (e.g.
	// file1.bin, file2.bin, ..., fileK.bin), then simply use the
	// static function FilenameEnumerator, which will return a function
	// that will enumerate filenames.
	//
	// Otherwise, if each serialized object should be saved in a
	// separate file, but the filename does not matter, use the
	// static function FileTimer to generate the required naming
	// function. For example:
	//
	// n := NewNStep(10, object, FileTimer("filename", "bin"))
	filename func() string
}

// NewNStep returns a checkpointer that saves a Serializable to a file
// every n environmental steps of an experiment. If the file already
// exists, it is overwritten.
func NewNStep(n int, object Serializable,
	filename func() string) Checkpointer {
	return &nStep{
		interval: n,
		object:   object,
		filename: filename,
	}
}

// Checkpoint checkpoints the nStep's tracked Serializable object by
// calling its Save() method on a file. If the file already exists, it
// is overwritten.
func (n *nStep) Checkpoint(t ts.TimeStep) error {
	if t.Number%n.interval == 0 {
		out, err := os.Create(n.filename())
		if err != nil {
			return fmt.Errorf("checkpoint: cannot create file: %v", err)
		}

		enc := gob.NewEncoder(out)
		err = enc.Encode(n.object)
		if err != nil {
			return fmt.Errorf("checkpoint: could not checkpoint: %v", err)
		}
		return nil
	}
	return nil
}
