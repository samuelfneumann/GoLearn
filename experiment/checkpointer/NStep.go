package checkpointer

import ts "sfneuman.com/golearn/timestep"

// nStep implements checkpointing every N steps
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

// NewNStep returns a checkpointer that checkpoints every n steps.
func NewNStep(n int, object Serializable,
	filename func() string) Checkpointer {
	return &nStep{
		interval: n,
		object:   object,
		filename: filename,
	}
}

// Checkpoint checkpoints the Checkpointer's tracked object by calling
// its Save() method
func (n *nStep) Checkpoint(t ts.TimeStep) error {
	if t.Number%n.interval == 0 {
		return n.object.Save(n.filename())
	}
	return nil
}
