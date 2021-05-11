package spec

// QLearning represents a configuration for the QLearning agent
type ESarsa struct {
	BehaviourE   float64 // epislon for behaviour policy
	TargetE      float64 // epsilon for target policy
	LearningRate float64
}

// Gets the configuration for QLearning
func (e ESarsa) Spec() map[string]float64 {
	spec := make(map[string]float64)
	spec["target epsilon"] = e.TargetE
	spec["behaviour epsilon"] = e.BehaviourE
	spec["learning rate"] = e.LearningRate
	return spec
}
