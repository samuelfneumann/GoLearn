// Package acrobot implements the classic control problem base
package acrobot

import (
	"fmt"
	"math"

	"github.com/samuelfneumann/golearn/environment"
	env "github.com/samuelfneumann/golearn/environment"
	ts "github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/golearn/utils/floatutils"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
)

// dynamicsType determines whether the dynamics of the environment
// follows those defined in the NeurIPS paper or the RL book.
type dynamicsType bool

const (
	// Dynamics of environment is consistent with RL book
	book dynamicsType = true

	// Dynamics of environment is consistent with NeurIPS paper
	nips dynamicsType = false
)

const (
	dt float64 = 0.2

	// Physical constants
	LinkLength1 float64 = 1.0 // Metres, length of link 1
	LinkLength2 float64 = 1.0 // Metres, length of link 2
	LinkMass1   float64 = 1.0 // Kg, mass of link 1
	LinkMass2   float64 = 1.0 // Kg, mass of link 2
	LinkCOMPos1 float64 = 0.5 // Metres, centre of mass link 1
	LinkCOMPos2 float64 = 0.5 // Metres, cetnre of mass link 2
	LinkMOI     float64 = 1.0 // Moments of inertia for both links
	MaxVel1     float64 = 4 * math.Pi
	MinVel1     float64 = -MaxVel1
	MaxVel2     float64 = 9 * math.Pi
	MinVel2     float64 = -MaxVel2
	Gravity     float64 = 9.8
	MaxAngle    float64 = math.Pi
	MinAngle    float64 = -MaxAngle
	MinTorque   float64 = -1.0
	MaxTorque   float64 = 1.0

	// Environment constants
	ObservationDims     int     = 4
	ActionDims          int     = 1
	MinContinuousAction float64 = MinTorque
	MaxContinuousAction float64 = MaxTorque
	MinDiscreteAction   int     = 0 // Applies MinTorque
	MaxDiscreteAction   int     = 2 // Applies MaxTorque

	BookOrNips dynamicsType = book
)

// base implements the classic control environment Acrobot. In this
// environment, a double hindged and double linked pendulum is attached
// to a single actuated fixed base. Torque can be applied to the base
// to swing the double pendulum (acrobot) around.
//
// State feature vectors are 4-dimensional and consist of the angle
// of the first pendulum link measured from the negative y-axis,
// the angle of the second pendulum link measured from the negative
// y-axis, the angular velocity of the first link, and the angular
// velocity of the second link. That is, a feature vector has the
// form:
//
//		v ⃗	= [θ1, θ2, θ̇1, θ̇2], where:
//		θ1 = angle of the first link measured from the negative y-axis
//		θ2 = angle of the second link measured from the negative y-axis
//		θ̇1 = angular velocity of the first link
//		θ̇2 = angular velocity of the second link
//
//
// State features are bounded. Angles are bounded to be between [-π, π]
// and angular velocity is bounded between [MinVel1, MaxVel1] for the
// first pendulum link and [MinVel2, MaxVel2] for the second pendulum
// link. Angles outside of [-π, π] are wrapped around to stay within
// this range, and angular velocity is clipped to stay within the
// legal range.
//
// base does not implement the environment.Environment interface, but
// is embedded in Discrete and Continuous which do implement this
// interface. This struct is used to share code between discrete action
// and continuous action versions of the acrobot environment.
type base struct {
	env.Task
	lastStep        ts.TimeStep
	discount        float64
	angleBounds     r1.Interval
	velocity1Bounds r1.Interval
	velocity2Bounds r1.Interval
}

// validateState checks if the state is valid and returns an error
// denoting whether the state is a valid state or not.
func validateState(state *mat.VecDense, angleBounds, vel1Bounds,
	vel2Bounds r1.Interval) error {
	if l := state.Len(); l != 4 {
		return fmt.Errorf("illegal state length \n\twant(4) \n\thave(%v)", l)
	}
	if angleBounds.Min < state.AtVec(0) || angleBounds.Max > state.AtVec(0) {
		return fmt.Errorf("angle 1 out of bounds")
	}
	if angleBounds.Min < state.AtVec(1) || angleBounds.Max > state.AtVec(1) {
		return fmt.Errorf("angle 2 out of bounds")
	}
	if vel1Bounds.Min < state.AtVec(2) || vel1Bounds.Max > state.AtVec(2) {
		return fmt.Errorf("angular velocity 1 out of bounds")
	}
	if vel2Bounds.Min < state.AtVec(3) || vel2Bounds.Max > state.AtVec(3) {
		return fmt.Errorf("angular velocity 2 out of bounds")
	}
	return nil
}

// newBase returns a new base acrobot environment
func newBase(t env.Task, discount float64) (*base, ts.TimeStep) {
	state := t.Start()

	firstStep := ts.New(ts.First, 0.0, discount, state, 0)

	acrobot := base{
		Task:            t,
		lastStep:        firstStep,
		discount:        discount,
		angleBounds:     r1.Interval{Min: MinAngle, Max: MaxAngle},
		velocity1Bounds: r1.Interval{Min: MinVel1, Max: MaxVel1},
		velocity2Bounds: r1.Interval{Min: MinVel2, Max: MaxVel2},
	}

	// Ensure start state is valid
	err := validateState(state, acrobot.angleBounds, acrobot.velocity1Bounds,
		acrobot.velocity2Bounds)
	if err != nil {
		panic(fmt.Sprintf("new: %v", err))
	}

	return &acrobot, firstStep

}

// nextState returns the next state of the environment given the
// torque to apply to the fixed base of the acrobot.
func (a *base) nextState(torque float64) *mat.VecDense {
	s := a.CurrentTimeStep().Observation

	// Continuous action between [MinTorque, MaxTorque]
	torque = floatutils.Clip(torque, MinTorque, MaxTorque)

	sAugmented := mat.NewVecDense(s.Len()+1, nil)
	num := sAugmented.CopyVec(s)
	if num != s.Len() {
		panic("step: wrong number of state elements copied")
	}
	sAugmented.SetVec(sAugmented.Len()-1, torque)

	integrated := rk4(dsDt, sAugmented, []float64{0.0, dt})
	r, c := integrated.Dims()
	if c != 5 {
		panic("step: integration returned more than 5 components")
	}
	ns := integrated.RowView(r-1).(*mat.VecDense).SliceVec(0,
		c-1).(*mat.VecDense)

	// Ensure state stays in an acceptable range
	ns.SetVec(0, floatutils.WrapInterval(ns.AtVec(0), a.angleBounds))
	ns.SetVec(1, floatutils.WrapInterval(ns.AtVec(1), a.angleBounds))
	ns.SetVec(2, floatutils.ClipInterval(ns.AtVec(2), a.velocity1Bounds))
	ns.SetVec(3, floatutils.ClipInterval(ns.AtVec(3), a.velocity2Bounds))

	return ns
}

// update updates the base environment by constructing a new current
// TimeStep for the environment from the argument action and next
// state newState.
//
// This funciton is used so that the discrete and continuous action
// versions of Acrobot can be deal with uniformly. Each calculates
// the force to apply and calls this struct's nextState() function.
// The result of that function is then passed to this function as
// well as the action taken, which is needed to calculate the reward.
func (a *base) update(action, newState *mat.VecDense) (ts.TimeStep, bool) {
	// Create the new timestep
	reward := a.GetReward(a.CurrentTimeStep().Observation, action, newState)
	nextStep := ts.New(ts.Mid, reward, a.discount, newState,
		a.CurrentTimeStep().Number+1)

	// Check if the step is the last in the episode and adjust step type
	// if necessary
	a.End(&nextStep)

	a.lastStep = nextStep
	return nextStep, nextStep.Last()
}

// CurrentTimeStep returns the current timestep of the environment
func (a *base) CurrentTimeStep() ts.TimeStep {
	return a.lastStep
}

// Reset resets the environment, begins a new episode, and returns
// the first timestep of the new episode
func (a *base) Reset() ts.TimeStep {
	state := a.Start()
	err := validateState(state, a.angleBounds, a.velocity1Bounds,
		a.velocity2Bounds)
	if err != nil {
		panic(fmt.Sprintf("reset: %v", err))
	}

	startStep := ts.New(ts.First, 0, a.discount, state, 0)
	a.lastStep = startStep

	return startStep
}

// ObservationSpec returns the observation specification of the
// environment
func (a *base) ObservationSpec() environment.Spec {
	shape := mat.NewVecDense(ObservationDims, nil)
	lowerBound := mat.NewVecDense(ObservationDims, []float64{MinAngle,
		MinAngle, -MaxVel1, -MaxVel2})
	upperBound := mat.NewVecDense(ObservationDims, []float64{MaxAngle,
		MaxAngle, MaxVel1, MaxVel2})

	return environment.NewSpec(shape, environment.Observation, lowerBound,
		upperBound, environment.Continuous)
}

// DiscountSpec returns the discounting specification of the environment
func (a *base) DiscountSpec() environment.Spec {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{a.discount})
	upperBound := mat.NewVecDense(1, []float64{a.discount})

	return environment.NewSpec(shape, environment.Discount, lowerBound,
		upperBound, environment.Continuous)
}

// String implements the fmt.Stringer interface
func (a *base) String() string {
	state := a.CurrentTimeStep().Observation

	return fmt.Sprintf("base  |  θ1: %v  |  θ2: %v  |  θ̇1: %v  |  θ̇2: %v",
		state.AtVec(0), state.AtVec(1), state.AtVec(2), state.AtVec(3))
}

// dsDt calculate ds/dt for the environment, where s = the current
// environment state
func dsDt(sAugmented *mat.VecDense, t float64) []float64 {
	m1 := LinkMass1
	m2 := LinkMass2
	l1 := LinkLength1
	lc1 := LinkCOMPos1
	lc2 := LinkCOMPos2
	i1 := LinkMOI
	i2 := LinkMOI
	g := Gravity

	s := sAugmented.SliceVec(0, sAugmented.Len()-1)
	a := sAugmented.AtVec(sAugmented.Len() - 1)

	theta1 := s.AtVec(0)
	theta2 := s.AtVec(1)
	dtheta1 := s.AtVec(2)
	dtheta2 := s.AtVec(3)

	d1 := (m1*math.Pow(lc1, 2) +
		m2*(math.Pow(l1, 2)+math.Pow(lc2, 2)+2*l1*lc2*math.Cos(theta2)) +
		i1 + i2)

	d2 := m2*(math.Pow(lc2, 2)+l1*lc2*math.Cos(theta2)) + i2

	phi2 := m2 * lc2 * g * math.Cos(theta1+theta2-(math.Pi/2.0))
	phi1 := (-m2*l1*lc2*math.Pow(dtheta2, 2)*math.Sin(theta2) -
		2*m2*l1*lc2*dtheta2*dtheta1*math.Sin(theta2) +
		(m1*lc1+m2*l1)*g*math.Cos(theta1-(math.Pi/2.0)) +
		phi2)

	var ddtheta2 float64
	if BookOrNips == nips {
		ddtheta2 = (a + d2/d1*phi1 - phi2) / (m2*math.Pow(lc2, 2) + i2 -
			math.Pow(d2, 2)/d1)
	} else {
		ddtheta2 = (a + d2/d1*phi1 - m2*l1*lc2*math.Pow(dtheta1, 2)*
			math.Sin(theta2) - phi2) /
			(m2*math.Pow(lc2, 2) + i2 - math.Pow(d2, 2)/d1)
	}
	ddtheta1 := -(d2*ddtheta2 + phi1) / d1

	// Last component is da/dt == 0.0
	return []float64{dtheta1, dtheta2, ddtheta1, ddtheta2, 0.0}
}

// rk4 integrates an n-dimensional system of ODEs using 4-th order
// RUnge-Kutta.
//
// Adapted from OpenAI Gym Acrobot:
// https://github.com/openai/gym/blob/7c9ae6d14087fe50714d59bc36b1797560
// 961710/gym/envs/classic_control/acrobot.py
func rk4(derivs func(*mat.VecDense, float64) []float64, y0 *mat.VecDense, t []float64) *mat.Dense {
	Ny := y0.Len()

	var yout *mat.Dense
	if Ny == 1 {
		yout = mat.NewDense(len(t), 1, nil)
	} else {
		yout = mat.NewDense(len(t), Ny, nil)
	}

	yout.SetRow(0, y0.RawVector().Data)

	for i := 0; i < len(t)-1; i++ {
		thist := t[i]
		dt := t[i+1] - thist // shadowing package constant
		dt2 := dt / 2.0

		y0 := yout.RowView(i).(*mat.VecDense) // shadowing input y0

		dsdt := derivs(y0, thist)
		k1 := mat.NewVecDense(len(dsdt), dsdt)

		input := mat.NewVecDense(len(dsdt), nil)
		input.AddScaledVec(y0, dt2, k1)
		dsdt = derivs(input, thist+dt2)
		k2 := mat.NewVecDense(len(dsdt), dsdt)

		input.AddScaledVec(y0, dt2, k2)
		dsdt = derivs(input, thist+dt2)
		k3 := mat.NewVecDense(len(dsdt), dsdt)

		input.AddScaledVec(y0, dt, k3)
		dsdt = derivs(input, thist+dt)
		k4 := mat.NewVecDense(len(dsdt), dsdt)

		row := mat.NewVecDense(k1.Len(), nil)
		row.CopyVec(k1)
		row.AddScaledVec(row, 2.0, k2)
		row.AddScaledVec(row, 2.0, k3)
		row.AddVec(row, k4)
		row.AddScaledVec(y0, dt/6.0, row)

		yout.SetRow(i+1, row.RawVector().Data)
	}
	return yout
}
