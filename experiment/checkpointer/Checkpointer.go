package checkpointer

import (
	ts "sfneuman.com/golearn/timestep"
)

// Serializable is an object that can be saved/serialized
type Serializable interface {
	Save(string) error
	Load(string) error
}

// Checkpointer checkpoints/saves serializable objects based on
// timestep.TimeSteps
type Checkpointer interface {
	Checkpoint(ts.TimeStep) error
}
