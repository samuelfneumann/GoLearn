package vanillaac

import (
	"fmt"
	"reflect"

	G "gorgonia.org/gorgonia"

	"github.com/samuelfneumann/golearn/agent"
	"github.com/samuelfneumann/golearn/agent/nonlinear/continuous/policy"
	"github.com/samuelfneumann/golearn/buffer/expreplay"
	env "github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/initwfn"
	"github.com/samuelfneumann/golearn/network"
	"github.com/samuelfneumann/golearn/solver"
)

func init() {
	// Register ConfigList type so that it can be typed using
	// agent.TypedConfigList to help with serialization/deserialization.
	agent.Register(agent.GaussianVanillaACTreeMLP, GaussianTreeMLPConfigList{})
}

// GaussianTreeMLPConfigList implements functionality for storing a
// list of GaussianTreeMLPConfig's in a simple way. Instead of storing
// a slice of Configs, the ConfigList stores each field's values and
// constructs the list by every combination of field values.
type GaussianTreeMLPConfigList struct {
	// Policy neural net
	RootLayers      [][]int
	RootBiases      [][]bool
	RootActivations [][]*network.Activation

	LeafLayers      [][][]int
	LeafBiases      [][][]bool
	LeafActivations [][][]*network.Activation

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

// NewGaussianTreeMLPConfigList returns a new GaussianTreeMLPConfigList
// as an agent.TypedConfigList. Because the returned value is a
// TypedList, it can safely be JSON serialized and deserialized without
// specifying what the type of the ConfigList is.
func NewGaussianTreeMLPConfigList(
	RootLayers [][]int,
	RootBiases [][]bool,
	RootActivations [][]*network.Activation,
	LeafLayers [][][]int,
	LeafBiases [][][]bool,
	LeafActivations [][][]*network.Activation,
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
	config := GaussianTreeMLPConfigList{
		RootLayers:      RootLayers,
		RootBiases:      RootBiases,
		RootActivations: RootActivations,

		LeafLayers:      LeafLayers,
		LeafBiases:      LeafBiases,
		LeafActivations: LeafActivations,

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
// GaussianTreeMLPConfigList
func (g GaussianTreeMLPConfigList) Config() agent.Config {
	return GaussianTreeMLPConfig{}
}

// Type returns the type of Config stored in the list
func (g GaussianTreeMLPConfigList) Type() agent.Type {
	return g.Config().Type()
}

// Len returns the number of configurations stored in the list
func (g GaussianTreeMLPConfigList) Len() int {
	return len(g.RootLayers) * len(g.RootBiases) * len(g.RootActivations) *
		len(g.LeafLayers) * len(g.LeafBiases) * len(g.LeafActivations) *
		len(g.ValueFnLayers) * len(g.ValueFnBiases) *
		len(g.ValueFnActivations) * len(g.InitWFn) * len(g.PolicySolver) *
		len(g.VSolver) * len(g.ValueGradSteps) *
		len(g.ExpReplay) * len(g.Tau) * len(g.TargetUpdateInterval)
}

// NumFields gets the total number of settable fields/hyperparameters
// for the agent configuration
func (g GaussianTreeMLPConfigList) NumFields() int {
	rValue := reflect.ValueOf(g)
	return rValue.NumField()
}

// GaussianTreeMLPConfig implements a configuration for a Gaussian
// policy vanilla actor critic agent. The Gaussian policy is
// parameterized by a neural network which has a single input and
// a single root network. The root network then splits off into two
// leaf networks - one for the mean and one for the log standard
// deviation of the policy. See the policy.GaussianTreeMLP struct for
// more details. The action dimensions may be n-dimensional.
type GaussianTreeMLPConfig struct {
	// Policy neural net
	policy          agent.LogPdfOfer // VPG.trainPolicy
	behaviour       agent.NNPolicy   // VPG.behaviour
	RootLayers      []int
	RootBiases      []bool
	RootActivations []*network.Activation

	LeafLayers      [][]int
	LeafBiases      [][]bool
	LeafActivations [][]*network.Activation

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
func (g GaussianTreeMLPConfig) BatchSize() int {
	return g.ExpReplay.BatchSize()
}

// Validate checks a Config to ensure it is a valid configuration
func (g GaussianTreeMLPConfig) Validate() error {
	if g.BatchSize() <= 0 {
		return fmt.Errorf("cannot have batch size %v < 1", g.BatchSize())
	}

	return nil
}

// ValidAgent returns true if the argument agent can be constructed
// from the Config and false otherwise.
func (g GaussianTreeMLPConfig) ValidAgent(a agent.Agent) bool {
	_, ok := a.(*VAC)
	return ok
}

// Type returns the type of agent constructed by the Config
func (c GaussianTreeMLPConfig) Type() agent.Type {
	return agent.GaussianVanillaACTreeMLP
}

// CreateAgent creates and returns the agent determine by the
// configuration
func (g GaussianTreeMLPConfig) CreateAgent(e env.Environment,
	seed uint64) (agent.Agent, error) {
	behaviour, err := policy.NewGaussianTreeMLP(
		e,
		1,
		G.NewGraph(),
		g.RootLayers,
		g.RootBiases,
		g.RootActivations,
		g.LeafLayers,
		g.LeafBiases,
		g.LeafActivations,
		g.InitWFn.InitWFn(),
		seed,
	)
	if err != nil {
		return nil, fmt.Errorf("createAgent: could not create "+
			"behaviour policy: %v", err)
	}

	p, err := policy.NewGaussianTreeMLP(
		e,
		g.BatchSize(),
		G.NewGraph(),
		g.RootLayers,
		g.RootBiases,
		g.RootActivations,
		g.LeafLayers,
		g.LeafBiases,
		g.LeafActivations,
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
		return nil, fmt.Errorf("createAgent: could not create target "+
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
func (g GaussianTreeMLPConfig) trainPolicy() agent.LogPdfOfer {
	return g.policy
}

// behaviour returns the constructed behaviour policy from the config
func (g GaussianTreeMLPConfig) behaviourPolicy() agent.NNPolicy {
	return g.behaviour
}

// valueFn returns the constructed value function from the config
func (g GaussianTreeMLPConfig) valueFn() network.NeuralNet {
	return g.vValueFn
}

// trainValueFn returns the constructed value function to train from
// the config
func (g GaussianTreeMLPConfig) trainValueFn() network.NeuralNet {
	return g.vTrainValueFn
}

// targetValueFn returns the target value function
func (g GaussianTreeMLPConfig) targetValueFn() network.NeuralNet {
	return g.vTargetValueFn
}

// initWFn returns the initWFn from the config
func (g GaussianTreeMLPConfig) initWFn() *initwfn.InitWFn {
	return g.InitWFn
}

// policySolver returns the constructed policy solver from the config
func (g GaussianTreeMLPConfig) policySolver() *solver.Solver {
	return g.PolicySolver
}

// vSolver reutrns the constructed value function solver from the
// config
func (g GaussianTreeMLPConfig) vSolver() *solver.Solver {
	return g.VSolver
}

// batchSize returns the batch size for the config
func (g GaussianTreeMLPConfig) batchSize() int {
	return g.BatchSize()
}

// valueGradSteps returns the number of gradient steps per environment
// step to take for the value function
func (g GaussianTreeMLPConfig) valueGradSteps() int {
	return g.ValueGradSteps
}

// expReplay returns the experience replayer configuration for the
// agent
func (g GaussianTreeMLPConfig) expReplay() expreplay.Config {
	return g.ExpReplay
}

func (g GaussianTreeMLPConfig) tau() float64 {
	return g.Tau
}

func (g GaussianTreeMLPConfig) targetUpdateInterval() int {
	return g.TargetUpdateInterval
}
