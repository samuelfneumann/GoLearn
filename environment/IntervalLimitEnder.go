package environment

import (
	"gonum.org/v1/gonum/spatial/r1"
	"sfneuman.com/golearn/timestep"
)

// IntervalLimit implements the Ender interface to end episodes
// whenever a single feature in a feature vector leaves some interval
type IntervalLimit struct {
	intervals []r1.Interval
	indices   []int
	endType   timestep.EndType
}

// NewIntervalLimit creates and returns a new inteval limit. The endType
// argument determines what the episode end should be considered as.
func NewIntervalLimit(limits []r1.Interval, obsIndices []int,
	endType timestep.EndType) Ender {
	if len(limits) != len(obsIndices) {
		panic("limits should have same length as observation indices")
	}

	return &IntervalLimit{limits, obsIndices, endType}
}

// End determines whether or not the current episode should be ended,
// returning a boolean to indicate episode temrination. If the episode
// should be ended End() will modify the timestep so that its StepType
// field is timestep.Last and its EndType is the appropriate ending
// type.
func (i *IntervalLimit) End(t *timestep.TimeStep) bool {
	for index := range i.indices {

		featureIndex := i.indices[index]
		interval := i.intervals[index]

		if t.Observation.AtVec(featureIndex) > interval.Max ||
			t.Observation.AtVec(featureIndex) < interval.Min {
			t.StepType = timestep.Last
			t.SetEnd(i.endType)
			return true
		}
	}
	return false
}
