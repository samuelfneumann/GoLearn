package vanillapg

import (
	"fmt"
	"reflect"

	G "gorgonia.org/gorgonia"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/agent/nonlinear/continuous/policy"
	env "sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/initwfn"
	"sfneuman.com/golearn/network"
	"sfneuman.com/golearn/solver"
)

func init() {
	// Register ConfigList type so that it can be typed using
	// agent.TypedConfigList to help with serialization/deserialization.
	agent.Register(agent.CategoricalVanillaPGMLP, CategoricalMLPConfigList{})
}

// CategoricalMLPConfigList implements functionality for storing a list
// of CategoricalMLPConfig's in a simple way. Instead of storing
// a slice of Configs, the ConfigList stores each field's values and
// constructs the list by every combination of field values.
type CategoricalMLPConfigList struct {
	PolicyLayers      [][]int
	PolicyBiases      [][]bool
	PolicyActivations [][]*network.Activation

	// State value function neural net
	ValueFnLayers      [][]int
	ValueFnBiases      [][]bool
	ValueFnActivations [][]*network.Activation

	// Weight init function for all neural nets
	InitWFn []*initwfn.InitWFn

	PolicySolver []*solver.Solver
	VSolver      []*solver.Solver

	// Number of gradient steps to take for the value functions per
	// epoch
	ValueGradSteps          []int
	EpochLength             []int
	FinishEpisodeOnEpochEnd []bool

	// Generalized Advantage Estimation
	Lambda []float64
	Gamma  []float64
}

// NewCategoricalMLPConfigList returns a new CategoricalMLPConfigList
// as an agent.TypedConfigList. Because the returned value is a
// TypedList, it can safely be JSON serialized and deserialized without
// specifying what the type of the ConfigList is.
func NewCategoricalMLPConfigList(
	PolicyLayers [][]int,
	PolicyBiases [][]bool,
	PolicyActivations [][]*network.Activation,
	ValueFnLayers [][]int,
	ValueFnBiases [][]bool,
	ValueFnActivations [][]*network.Activation,
	InitWFn []*initwfn.InitWFn,
	PolicySolver []*solver.Solver,
	VSolver []*solver.Solver,
	ValueGradSteps []int,
	EpochLength []int,
	FinishEpisodeOnEpochEnd []bool,
	Lambda []float64,
	Gamma []float64,
) agent.TypedConfigList {
	config := CategoricalMLPConfigList{
		PolicyLayers:      PolicyLayers,
		PolicyBiases:      PolicyBiases,
		PolicyActivations: PolicyActivations,

		ValueFnLayers:      ValueFnLayers,
		ValueFnBiases:      ValueFnBiases,
		ValueFnActivations: ValueFnActivations,

		InitWFn: InitWFn,

		PolicySolver: PolicySolver,
		VSolver:      VSolver,

		ValueGradSteps:          ValueGradSteps,
		EpochLength:             EpochLength,
		FinishEpisodeOnEpochEnd: FinishEpisodeOnEpochEnd,

		Lambda: Lambda,
		Gamma:  Gamma,
	}

	return agent.NewTypedConfigList(config)
}

// Type returns the type of Config stored in the list
func (c CategoricalMLPConfigList) Type() agent.Type {
	return c.Config().Type()
}

// NumFields gets the total number of settable fields/hyperparameters
// for the agent configuration
func (c CategoricalMLPConfigList) NumFields() int {
	rValue := reflect.ValueOf(c)
	return rValue.NumField()
}

// Config returns an empty Config that is of the type stored by
// CategoricalMLPConfigList
func (c CategoricalMLPConfigList) Config() agent.Config {
	return CategoricalMLPConfig{}
}

// Len returns the number of configurations stored in the list
func (c CategoricalMLPConfigList) Len() int {
	return len(c.Lambda) * len(c.Gamma) * len(c.ValueGradSteps) *
		len(c.EpochLength) * len(c.InitWFn) * len(c.ValueFnActivations) *
		len(c.ValueFnBiases) * len(c.ValueFnLayers) * len(c.PolicySolver) *
		len(c.VSolver) * len(c.PolicyActivations) * len(c.PolicyBiases) *
		len(c.PolicyLayers)
}

// CategoricalMLPConfig implements a configuration for a categorical
// policy vanilla policy gradient agent. The categorical distribution
// is parameterized by a neural network with N outputs, one for each
// action in the environment. The network outputs the logit of each
// action, and action probabilities are comptued through the softmax
// function.
type CategoricalMLPConfig struct {
	// Policy neural net
	policy            agent.LogPdfOfer // VPG.trainPolicy
	behaviour         agent.NNPolicy   // VPG.behaviour
	PolicyLayers      []int
	PolicyBiases      []bool
	PolicyActivations []*network.Activation

	// State value function neural net
	vValueFn           network.NeuralNet
	vTrainValueFn      network.NeuralNet
	ValueFnLayers      []int
	ValueFnBiases      []bool
	ValueFnActivations []*network.Activation

	// Weight init function for all neural nets
	InitWFn *initwfn.InitWFn

	PolicySolver *solver.Solver
	VSolver      *solver.Solver

	// Number of gradient steps to take for the value functions per
	// epoch
	ValueGradSteps int
	EpochLength    int

	// FinishEpisodeOnEpochEnd denotes if the current episode should
	// be finished before starting a new epoch. If true, then the
	// agent is updated when the current epoch ends, then the current
	// episode is finished, then the next epoch starts. If false, the
	// agent is updated when the current epoch is finished, and the
	// next epoch starts at the next timestep, which may be in the
	// middle of an episode.
	FinishEpisodeOnEpochEnd bool

	// Generalized Advantage Estimation
	Lambda float64
	Gamma  float64
}

// BatchSize gets the batch size for the policy generated by this config
func (c CategoricalMLPConfig) BatchSize() int {
	return c.EpochLength
}

// Validate checks a Config to ensure it is a valid configuration
func (c CategoricalMLPConfig) Validate() error {
	if c.EpochLength <= 0 {
		return fmt.Errorf("cannot have epoch length < 1")
	}

	return nil
}

// Type returns the type of the configuration
func (c CategoricalMLPConfig) Type() agent.Type {
	return agent.CategoricalVanillaPGMLP
}

// ValidAgent returns whether the input agent is valid for this config
func (c CategoricalMLPConfig) ValidAgent(a agent.Agent) bool {
	switch a.(type) {
	case *VPG:
		return true
	}
	return false
}

// CreateAgent creates and returns the agent determine by the
// configuration
func (c CategoricalMLPConfig) CreateAgent(e env.Environment,
	seed uint64) (agent.Agent, error) {

	behaviour, err := policy.NewCategoricalMLP(
		e,
		1,
		G.NewGraph(),
		c.PolicyLayers,
		c.PolicyBiases,
		c.PolicyActivations,
		c.InitWFn.InitWFn(),
		seed,
	)
	if err != nil {
		return nil, fmt.Errorf("createAgent: could not create "+
			"behaviour policy: %v", err)
	}

	p, err := policy.NewCategoricalMLP(
		e,
		c.EpochLength,
		G.NewGraph(),
		c.PolicyLayers,
		c.PolicyBiases,
		c.PolicyActivations,
		c.InitWFn.InitWFn(),
		seed,
	)
	if err != nil {
		return nil, fmt.Errorf("createAgent: could not create policy: %v", err)
	}

	features := e.ObservationSpec().Shape.Len()

	valueFn, err := network.NewSingleHeadMLP(
		features,
		1,
		G.NewGraph(),
		c.ValueFnLayers,
		c.ValueFnBiases,
		c.InitWFn.InitWFn(),
		c.ValueFnActivations,
	)
	if err != nil {
		return nil, fmt.Errorf("createAgent: could not create value "+
			"function: %v", err)
	}

	trainValueFn, err := network.NewSingleHeadMLP(
		features,
		c.EpochLength,
		G.NewGraph(),
		c.ValueFnLayers,
		c.ValueFnBiases,
		c.InitWFn.InitWFn(),
		c.ValueFnActivations,
	)
	if err != nil {
		return nil, fmt.Errorf("createAgent: could not create "+
			"train value function: %v", err)
	}

	network.Set(behaviour.Network(), p.Network())
	network.Set(valueFn, trainValueFn)
	c.policy = p
	c.behaviour = behaviour
	c.vValueFn = valueFn
	c.vTrainValueFn = trainValueFn

	return New(e, c, int64(seed))
}

// Below implemented to satisfy the vanillapg.config interface
// See the Config.go file in the vanillapg package for more details.

// policy returns the constructed policy to train from the config
func (g CategoricalMLPConfig) trainPolicy() agent.LogPdfOfer {
	return g.policy
}

// behaviour returns the constructed behaviour policy from the config
func (g CategoricalMLPConfig) behaviourPolicy() agent.NNPolicy {
	return g.behaviour
}

// valueFn returns the constructed value function from the config
func (g CategoricalMLPConfig) valueFn() network.NeuralNet {
	return g.vValueFn
}

// trainValueFn returns the constructed value function to train from
// the config
func (g CategoricalMLPConfig) trainValueFn() network.NeuralNet {
	return g.vTrainValueFn
}

// initWFn returns the initWFn from the config
func (g CategoricalMLPConfig) initWFn() *initwfn.InitWFn {
	return g.InitWFn
}

// policySolver returns the constructed policy solver from the config
func (g CategoricalMLPConfig) policySolver() *solver.Solver {
	return g.PolicySolver
}

// vSolver reutrns the constructed value function solver from the
// config
func (g CategoricalMLPConfig) vSolver() *solver.Solver {
	return g.VSolver
}

// batchSize returns the batch size for the config
func (g CategoricalMLPConfig) batchSize() int {
	return g.BatchSize()
}

// valueGradSteps returns the number of gradient steps per environment
// step to take for the value function
func (g CategoricalMLPConfig) valueGradSteps() int {
	return g.ValueGradSteps
}

// epochLength returns the epoch length of the config
func (g CategoricalMLPConfig) epochLength() int {
	return g.EpochLength
}

// finishEpisodeOnEpochEnd returns whether or not the current episode
// should be finished before starting a new epoch, once the current
// epoch has ended
func (g CategoricalMLPConfig) finishEpisodeOnEpochEnd() bool {
	return g.FinishEpisodeOnEpochEnd
}

// lambda returns the λ from the config for GAE
func (g CategoricalMLPConfig) lambda() float64 {
	return g.Lambda
}

// gamma returns the ℽ from the config for GAE
func (g CategoricalMLPConfig) gamma() float64 {
	return g.Gamma
}
