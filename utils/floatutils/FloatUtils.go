// Package floatutils provides utilities for working with floats
package floatutils

import (
	"math"

	"gonum.org/v1/gonum/spatial/r1"
)

// Clip clips a floating point to within a minimum and maximum value.
// If the floating point exceeds max, then the function returns the max
// If min exceeds the floating point, then the function returns the min
func Clip(value, min, max float64) float64 {
	clipped := math.Min(value, max)
	return math.Max(clipped, min)
}

// ClipInterval is a wrapper to use Clip with an r1.Interval instead of
// a separate max and min value
func ClipInterval(value float64, interval r1.Interval) float64 {
	return Clip(value, interval.Min, interval.Max)
}

// Max gets the maximum value and indices of the maximum values in
// a slice of float64.
func Max(values []float64) (max float64, indices []int) {
	max, indices = values[0], []int{0}

	for i, value := range values {
		if value > max {
			max = value
			indices = []int{i}
		} else if value == max {
			indices = append(indices, i)
		}
	}
	return
}
