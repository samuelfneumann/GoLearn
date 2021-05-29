package spec

// QLearning represents a configuration for the QLearning agent
type QLearning struct {
	E            float64 // epislon for behaviour policy
	LearningRate float64
}

// Gets the configuration for QLearning. The spec.Qlearning specification
// can be used with the esarsa.ESarsa or qlearning.QLearning algorithms
func (q QLearning) Spec() map[Key]float64 {
	spec := make(map[Key]float64)

	// Ensure compatability with esarsa.ESarsa
	spec[TargetE] = 0.0

	spec[BehaviourE] = q.E
	spec[LearningRate] = q.LearningRate
	return spec
}
