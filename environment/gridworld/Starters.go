package gridworld

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/environment"
)

// SingleStart represents a single starting position in a GridWorld
type SingleStart struct {
	state mat.Vector
	r, c  int
}

// NewSingleStart creates and returns a new SingleStart
func NewSingleStart(x, y, r, c int) (environment.Starter, error) {
	if x > c {
		return &SingleStart{}, fmt.Errorf("x = %d > cols = %d", x, c)
	} else if y > r {
		return &SingleStart{}, fmt.Errorf("y = %d > cols = %d", y, c)
	}

	start := cToV(x, y, r, c)
	return &SingleStart{start, r, c}, nil
}

// Start returns the starting state for a SingleStart
func (s *SingleStart) Start() mat.Vector {
	return s.state
}
