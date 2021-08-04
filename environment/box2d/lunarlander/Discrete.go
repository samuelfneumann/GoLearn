package lunarlander

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
)

type Discrete struct {
	*lunarLander
}

func NewDiscrete(task environment.Task, discount float64, seed uint64) (environment.Environment, timestep.TimeStep) {
	l, step := newLunarLander(task, discount, seed)
	return &Continuous{l}, step
}

func (c *Discrete) ActionSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{float64(MinDiscreteAction)})
	upperBound := mat.NewVecDense(1, []float64{float64(MaxDiscreteAction)})

	return spec.NewEnvironment(shape, spec.Action, lowerBound, upperBound,
		spec.Discrete)
}

func (c *Discrete) Step(action *mat.VecDense) (timestep.TimeStep, bool) {
	a := int(action.AtVec(0))

	if a == 0 {
		// No operation
		return c.lunarLander.Step(mat.NewVecDense(2, []float64{0., 0.}))
	} else if a == 1 {
		// Fire left engine
		return c.lunarLander.Step(mat.NewVecDense(2, []float64{0.0, -1.0}))
	} else if a == 2 {
		// Fire main engine
		return c.lunarLander.Step(mat.NewVecDense(2, []float64{1.0, 0.0}))
	} else if a == 3 {
		// Fire right engine
		return c.lunarLander.Step(mat.NewVecDense(2, []float64{0.0, 1.0}))
	}
	panic(fmt.Sprintf("step: illegal action selection, expected action ϵ "+
		"[0, 1, 2, 3], received action = %v", a))
}
