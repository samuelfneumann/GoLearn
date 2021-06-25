// Package spec implements specifications/configurations for agents and
// environments
package spec

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

// SpecType determines what kind of specification a Spec is. A Spec can
// specify the layout of an acion, an observation, a discount, or a reward
type SpecType int

const (
	Action SpecType = iota
	Observation
	Discount
	Reward
	AverageReward
)

// Cardinality determines the cardinality of a number (discrete or continuous)
type Cardinality string

const (
	Continuous Cardinality = "Continuous"
	Discrete   Cardinality = "Discrete"
)

// Environment implements a specification, which tells the type, shape,
// and bounds of an action, observation, discount, or reward in an
// environment
type Environment struct {
	Shape      mat.Vector
	Type       SpecType
	LowerBound mat.Vector
	UpperBound mat.Vector
	Cardinality
}

// NewEnvironment constructs a new environment specification
// The shape argument outlines the shape of the data described by the
// specification. The argument t outlines what the specification is
// describing (e.g. actions, observations, etc.). The cardinality
// arguments describes whether the values that the spec describes are
// continuous or discrete.
func NewEnvironment(shape mat.Vector, t SpecType, lowerBound,
	upperBound mat.Vector, cardinality Cardinality) Environment {
	if shape.Len() != lowerBound.Len() {
		panic(fmt.Sprintf("shape length %v must match lower bounds length %v",
			shape.Len(), lowerBound.Len()))
	}
	if shape.Len() != upperBound.Len() {
		panic(fmt.Sprintf("shape length %v must match uuper bounds length %v",
			shape.Len(), upperBound.Len()))
	}
	return Environment{shape, t, lowerBound, upperBound, cardinality}
}
