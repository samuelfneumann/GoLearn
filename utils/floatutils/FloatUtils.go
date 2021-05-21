// Package floatutils provides utilities for working with floats
package floatutils

import "math"

func Clip(value, min, max float64) float64 {
	clipped := math.Min(value, max)
	return math.Max(clipped, min)
}
