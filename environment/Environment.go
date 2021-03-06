// Package environment outlines the interfaces and sturcts needed to implement
// concrete environments
package environment

import (
	"image"

	"github.com/fogleman/gg"
	"github.com/samuelfneumann/golearn/timestep"
	"gonum.org/v1/gonum/mat"
)

// Starter implements a distribution of starting states and samples starting
// states for environments
type Starter interface {
	Start() *mat.VecDense
}

// Ender determines when the agent-environment interaction ends an episode
type Ender interface {
	// End takes the next timestep and checks if it is the last in the episode
	// If it is the last timestep, End adjusts the StepType.
	// Returns whether or not the episode has ended
	End(*timestep.TimeStep) bool
}

// Task implements the reward scheme for taking actions in some environment
type Task interface {
	Starter
	Ender
	GetReward(state mat.Vector, a mat.Vector, nextState mat.Vector) float64
	AtGoal(state mat.Matrix) bool
}

// Environment implements a simualted environment, which includes a Task to
// complete. When using an environment constructor, the constructor should
// return both the environment and the first timestep, ready to train on.
type Environment interface {
	Task
	Reset() (timestep.TimeStep, error) // Resets between episodes
	Step(action *mat.VecDense) (timestep.TimeStep, bool, error)
	DiscountSpec() Spec
	ObservationSpec() Spec
	ActionSpec() Spec
	CurrentTimeStep() timestep.TimeStep
}

// PixelEnvironment describes an environment that can represent its
// current state as an image
type PixelEnvironment interface {
	Environment
	Pixels(scale float64, dc gg.Context, save bool) image.Image
}

// Closer is an environment which can be closed
type Closer interface {
	Environment
	Close() error
}

// RowColer is an environment that can return the rows and columns of
// its underlying state space. In effect, this is an interface
// which all tabular environments will satisfy.
type RowColer interface {
	Environment
	Rows() int
	Cols() int
}
