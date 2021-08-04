package lunarlander

import (
	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
)

type Continuous struct {
	*lunarLander
}

func NewContinuous(task environment.Task, discount float64, seed uint64) (environment.Environment, timestep.TimeStep) {
	l, step := newLunarLander(task, discount, seed)
	return &Continuous{l}, step
}

func (c *Continuous) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(2, nil)
	lowerBound := mat.NewVecDense(2, []float64{-1., -1.})
	upperBound := mat.NewVecDense(2, []float64{1., 1.})

	return spec.NewEnvironment(shape, spec.Action, lowerBound, upperBound,
		spec.Continuous)
}
