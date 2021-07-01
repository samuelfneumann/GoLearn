package checkpointer

import (
	"encoding/gob"

	ts "sfneuman.com/golearn/timestep"
)

// Serializable is an object that can be saved/serialized
type Serializable interface {
	gob.GobEncoder
	gob.GobDecoder
}

// Checkpointer checkpoints/saves serializable objects based on
// timestep.TimeSteps
type Checkpointer interface {
	Checkpoint(ts.TimeStep) error
}
