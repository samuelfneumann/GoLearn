package spec

// QLearning represents a configuration for the QLearning agent
type QLearning struct {
	E            float64 // epislon for behaviour policy
	LearningRate float64
}

// Gets the configuration for QLearning. The spec.Qlearning specification
// can be used with the esarsa.ESarsa or qlearning.QLearning algorithms
func (q QLearning) Spec() map[string]float64 {
	spec := make(map[string]float64)
	spec["target epsilon"] = 0.0
	spec["behaviour epsilon"] = q.E

	spec["learning rate"] = q.LearningRate
	return spec
}
