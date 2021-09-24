package expreplay

import "errors"

// ExpReplayError implements errors unique to an experience replay
// buffer.
type ExpReplayError struct {
	Op  string
	Err error
}

// Error satisifes the error interface
func (e *ExpReplayError) Error() string {
	return e.Op + ": " + e.Err.Error()
}

var errEmptyCache error = errors.New("cache empty")

var errInsufficientSamples = errors.New("minimum capacity not yet reached")

// IsInsufficientSamples returns whether or not an error reports that
// there are insufficient samples in the buffer to sample from the
// buffer.
//
// A buffer has too few samples to sample if its current capacity is
// less than its minimum capacity.
func IsInsufficientSamples(err error) bool {
	if replayErr, ok := err.(*ExpReplayError); ok {
		err = replayErr.Err
	}
	return err == errInsufficientSamples
}

// IsEmptyBuffer returns whether or not an error reports that a
// replay buffer is empty.
func IsEmptyBuffer(err error) bool {
	if replayErr, ok := err.(*ExpReplayError); ok {
		err = replayErr.Err
	}
	return err == errEmptyCache
}
