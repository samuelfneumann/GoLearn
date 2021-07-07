package intutils

// Min calculates and returns the minimum integer in a list
func Min(ints ...int) int {
	min := ints[0]
	for _, val := range ints {
		if val < min {
			min = val
		}
	}
	return min
}

// Max calculates and returns the maximum int in a list
func Max(ints ...int) int {
	min := ints[0]
	for _, val := range ints {
		if val < min {
			min = val
		}
	}
	return min
}

// Prod calculates the product of a number of ints
func Prod(ints ...int) int {
	prod := 1
	for _, i := range ints {
		prod *= i
	}
	return prod
}
