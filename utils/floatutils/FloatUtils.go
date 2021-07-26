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

// ClipSlice clips all elements of a slice in-place.
func ClipSlice(slice []float64, min, max float64) []float64 {
	for i := range slice {
		slice[i] = Clip(slice[i], min, max)
	}
	return slice
}

// Max gets the maximum value and indices of the maximum values in
// a slice of float64.
func MaxSlice(values []float64) (max float64, indices []int) {
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

// Min calculates and returns the minimum float64 in a list
func Min(floats ...float64) float64 {
	min := floats[0]
	for _, val := range floats {
		if val < min {
			min = val
		}
	}
	return min
}

// Max calculates and returns the maximum float64 in a list
func Max(floats ...float64) float64 {
	max := floats[0]
	for _, val := range floats {
		if val > max {
			max = val
		}
	}
	return max
}

// CountNonZero returns the number of nonzero elements in a slice
func CountNonZero(slice []float64) int {
	count := 0
	for i := range slice {
		if slice[i] != 0 {
			count++
		}
	}
	return count
}

// NonZero returns true if slice contains nonzero elements and false
// otherwise,
func NonZero(slice []float64) bool {
	for i := range slice {
		if slice[i] != 0.0 {
			return true
		}
	}
	return false
}

// Contains returns true if slice contains value and false otherwise.
func Contains(slice []float64, value float64) bool {
	for i := range slice {
		if slice[i] == value {
			return true
		}
	}
	return false
}

// ArgMax returns the indices of the maximum values in a list of floats
func ArgMax(floats ...float64) []int {
	ind := []int{0}
	max := floats[0]

	for i := 0; i < len(floats); i++ {
		if floats[i] > max {
			ind = []int{i}
			max = floats[i]
		} else if floats[i] == max {
			ind = append(ind, i)
		}
	}

	return ind
}

// Unique returns the unique values in a float slice
func Unique(floats ...float64) []float64 {
	set := make(map[float64]bool)
	count := 0
	for _, float := range floats {
		if !set[float] {
			count++
		}
		set[float] = true
	}

	uniq := make([]float64, 0, count)
	for key := range set {
		uniq = append(uniq, key)
	}

	return uniq
}

// Sum returns the sum of a list of floats
func Sum(floats ...float64) float64 {
	sum := 0.0
	for _, float := range floats {
		sum += float
	}
	return sum
}

// Equal returns whether two slices are equal
func Equal(s1, s2 []float64) bool {
	if len(s1) != len(s2) {
		return false
	}

	for i := range s1 {
		if s1[i] != s2[i] {
			return false
		}
	}
	return true
}

// Ones returns a slice of ones
func Ones(size int) []float64 {
	slice := make([]float64, size)
	for i := range slice {
		slice[i] = 1.0
	}
	return slice
}
