package vanillaac

import (
	"fmt"
	"reflect"

	G "gorgonia.org/gorgonia"

	"github.com/samuelfneumann/golearn/agent"
	"github.com/samuelfneumann/golearn/agent/nonlinear/continuous/policy"
	env "github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/expreplay"
	"github.com/samuelfneumann/golearn/initwfn"
	"github.com/samuelfneumann/golearn/network"
	"github.com/samuelfneumann/golearn/solver"
)

func init() {
	// Register ConfigList type so that it can be typed using
	// agent.TypedConfigList to help with serialization/deserialization.
	agent.Register(agent.CategoricalVanillaACMLP, CategoricalMLPConfigList{})
}

type CategoricalMLPConfigList struct {
	// Policy neural net
	Layers      [][]int
	Biases      [][]bool
	Activations [][]*network.Activation

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
	ValueGradSteps []int

	ExpReplay []expreplay.Config

	Tau                  []float64
	TargetUpdateInterval []int
}

func NewCategoricalMLPConfigList(
	Layers [][]int,
	Biases [][]bool,
	Activations [][]*network.Activation,
	ValueFnLayers [][]int,
	ValueFnBiases [][]bool,
	ValueFnActivations [][]*network.Activation,
	InitWFn []*initwfn.InitWFn,
	PolicySolver []*solver.Solver,
	VSolver []*solver.Solver,
	ValueGradSteps []int,
	ExpReplay []expreplay.Config,
	Tau []float64,
	TargetUpdateInterval []int,
) agent.TypedConfigList {
	config := CategoricalMLPConfigList{
		Layers:      Layers,
		Biases:      Biases,
		Activations: Activations,

		ValueFnLayers:      ValueFnLayers,
		ValueFnBiases:      ValueFnBiases,
		ValueFnActivations: ValueFnActivations,

		InitWFn: InitWFn,

		PolicySolver: PolicySolver,
		VSolver:      VSolver,

		ValueGradSteps: ValueGradSteps,

		ExpReplay: ExpReplay,

		Tau:                  Tau,
		TargetUpdateInterval: TargetUpdateInterval,
	}

	return agent.NewTypedConfigList(config)
}

// Config returns an empty Config that is of the type stored by
// CategoricalMLPConfigList
func (c CategoricalMLPConfigList) Config() agent.Config {
	return CategoricalMLPConfig{}
}

// Type returns the type of Config stored in the list
func (c CategoricalMLPConfigList) Type() agent.Type {
	return c.Config().Type()
}

// Len returns the number of configurations stored in the list
func (c CategoricalMLPConfigList) Len() int {
	return len(c.Layers) * len(c.Biases) * len(c.Activations) *
		len(c.ValueFnLayers) * len(c.ValueFnBiases) *
		len(c.ValueFnActivations) * len(c.InitWFn) * len(c.PolicySolver) *
		len(c.VSolver) * len(c.ValueGradSteps) *
		len(c.ExpReplay) * len(c.Tau) * len(c.TargetUpdateInterval)
}

// NumFields gets the total number of settable fields/hyperparameters
// for the agent configuration
func (c CategoricalMLPConfigList) NumFields() int {
	rValue := reflect.ValueOf(c)
	return rValue.NumField()
}

// CategoricalMLPConfig implements a configuration for a Categorical
// policy vanilla actor critic agent. The Categorical policy is
// parameterized by a neural network which has a single input and
// a single root network. The root network then splits off into two
// leaf networks - one for the mean and one for the log standard
// deviation of the policy. See the policy.CategoricalMLP struct for
// more details. The action dimensions may be n-dimensional.
type CategoricalMLPConfig struct {
	// Policy neural net
	policy      agent.LogPdfOfer // VPG.trainPolicy
	behaviour   agent.NNPolicy   // VPG.behaviour
	Layers      []int
	Biases      []bool
	Activations []*network.Activation

	// State value function neural net
	vValueFn           network.NeuralNet
	vTrainValueFn      network.NeuralNet
	vTargetValueFn     network.NeuralNet
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

	// Experience replay parameters
	ExpReplay expreplay.Config

	Tau                  float64
	TargetUpdateInterval int
}

// BatchSize gets the batch size for the policy generated by this config
func (g CategoricalMLPConfig) BatchSize() int {
	return g.ExpReplay.BatchSize()
}

// Validate checks a Config to ensure it is a valid configuration
func (g CategoricalMLPConfig) Validate() error {
	if g.BatchSize() <= 0 {
		return fmt.Errorf("cannot have batch size %v < 1", g.BatchSize())
	}

	return nil
}

// ValidAgent returns true if the argument agent can be constructed
// from the Config and false otherwise.
func (g CategoricalMLPConfig) ValidAgent(a agent.Agent) bool {
	_, ok := a.(*VAC)
	return ok
}

// Type returns the type of agent constructed by the Config
func (c CategoricalMLPConfig) Type() agent.Type {
	return agent.CategoricalVanillaACMLP
}

// CreateAgent creates and returns the agent determine by the
// configuration
func (g CategoricalMLPConfig) CreateAgent(e env.Environment,
	seed uint64) (agent.Agent, error) {
	behaviour, err := policy.NewCategoricalMLP(
		e,
		1,
		G.NewGraph(),
		g.Layers,
		g.Biases,
		g.Activations,
		g.InitWFn.InitWFn(),
		seed,
	)
	if err != nil {
		return nil, fmt.Errorf("createAgent: could not create "+
			"behaviour policy: %v", err)
	}

	p, err := policy.NewCategoricalMLP(
		e,
		g.BatchSize(),
		G.NewGraph(),
		g.Layers,
		g.Biases,
		g.Activations,
		g.InitWFn.InitWFn(),
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
		g.ValueFnLayers,
		g.ValueFnBiases,
		g.InitWFn.InitWFn(),
		g.ValueFnActivations,
	)
	if err != nil {
		return nil, fmt.Errorf("createAgent: could not create "+
			"value function: %v", err)
	}

	trainValueFn, err := network.NewSingleHeadMLP(
		features,
		g.BatchSize(),
		G.NewGraph(),
		g.ValueFnLayers,
		g.ValueFnBiases,
		g.InitWFn.InitWFn(),
		g.ValueFnActivations,
	)
	if err != nil {
		return nil, fmt.Errorf("createAgent: could not create train "+
			"value function: %v", err)
	}

	targetValueFn, err := network.NewSingleHeadMLP(
		features,
		g.BatchSize(),
		G.NewGraph(),
		g.ValueFnLayers,
		g.ValueFnBiases,
		g.InitWFn.InitWFn(),
		g.ValueFnActivations,
	)
	if err != nil {
		return nil, fmt.Errorf("createAgent: could not create train "+
			"value function: %v", err)
	}

	network.Set(behaviour.Network(), p.Network())
	network.Set(valueFn, trainValueFn)
	network.Set(targetValueFn, trainValueFn)
	g.policy = p
	g.behaviour = behaviour
	g.vValueFn = valueFn
	g.vTrainValueFn = trainValueFn
	g.vTargetValueFn = targetValueFn

	return New(e, g, int64(seed))
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

// expReplay returns the experience replayer configuration for the
// agent
func (g CategoricalMLPConfig) expReplay() expreplay.Config {
	return g.ExpReplay
}

func (c CategoricalMLPConfig) targetValueFn() network.NeuralNet {
	return c.vTargetValueFn
}

func (c CategoricalMLPConfig) tau() float64 {
	return c.Tau
}

func (c CategoricalMLPConfig) targetUpdateInterval() int {
	return c.TargetUpdateInterval
}