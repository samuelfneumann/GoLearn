// Package spec implements specifications/configurations for agents and
// environments
package spec

import "gonum.org/v1/gonum/mat"

// SpecType determines what kind of specification a Spec is. A Spec can
// specify the layout of an acion, an observation, a discount, or a reward
type SpecType int

const (
	Action SpecType = iota
	Observation
	Discount
	Reward
)

// Environment implements a specification, which tells the type, shape,
// and bounds of an action, observation, discount, or reward in an
// environment
type Environment struct {
	Shape      mat.Vector
	Type       SpecType
	LowerBound mat.Vector
	UpperBound mat.Vector
}

// NewEnvironment constructs a new environment specification
func NewEnvironment(shape mat.Vector, t SpecType, lowerBound,
	upperBound mat.Vector) Environment {
	return Environment{shape, t, lowerBound, upperBound}
}

type Agent interface {
	Spec() map[string]float64 // Configuration for an agent
}
