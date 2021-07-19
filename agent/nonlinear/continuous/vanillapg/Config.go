package vanillapg

type PolicyType string

const (
	Gaussian    PolicyType = "Gaussian"
	Categorical PolicyType = "Softmax"
)
