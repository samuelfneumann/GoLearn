package spec

// ActorCritic represents a configuration for an Actor Critic agent
type LinearGaussianActorCritic struct {
	ActorLearningRate  float64
	CriticLearningRate float64
	Decay              float64
}

// Gets the configuration for ActorCritic spec
func (l LinearGaussianActorCritic) Spec() map[Key]float64 {
	spec := make(map[Key]float64)
	spec[ActorLearningRate] = l.ActorLearningRate
	spec[CriticLearningRate] = l.CriticLearningRate
	spec[Decay] = l.Decay
	return spec
}
