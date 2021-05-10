// Package agent defines an agent interface
package agent

import (
	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/timestep"
)

type Agent interface {
	Learner
	Policy
}

type Learner interface {
	Step() // Performs an update
	Observe(action mat.Vector, nextObs timestep.TimeStep)
	ObserveFirst(timestep.TimeStep)
	Weights() map[string]*mat.Dense
}

type Policy interface {
	SelectAction(t timestep.TimeStep) mat.Vector
	Weights() map[string]*mat.Dense
}
