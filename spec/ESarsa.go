package spec

// ESarsa represents a configuration for the ESarsa agent
type ESarsa struct {
	BehaviourE   float64 // epislon for behaviour policy
	TargetE      float64 // epsilon for target policy
	LearningRate float64
}

// Gets the configuration for ESarsa spec
func (e ESarsa) Spec() map[Key]float64 {
	spec := make(map[Key]float64)
	spec[TargetE] = e.TargetE
	spec[BehaviourE] = e.BehaviourE
	spec[LearningRate] = e.LearningRate
	return spec
}
