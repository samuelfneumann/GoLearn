// Package vanillaac implements a vanilla actor-critic algorithm
//
// This algorithm uses a state value function critic and baseline
// with a target network for the state value function. The value
// function and policy are each learned from data sampled from an
// experience replay buffer.
package vanillaac

import (
	"fmt"

	"github.com/samuelfneumann/golearn/agent"
	env "github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/expreplay"
	"github.com/samuelfneumann/golearn/network"
	ts "github.com/samuelfneumann/golearn/timestep"
	"github.com/samuelfneumann/golearn/utils/floatutils"
	"gonum.org/v1/gonum/mat"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var PLoss G.Value
var VLoss G.Value
var LogPDF G.Value
var Adv G.Value
var PNextStateValue G.Value
var PReward G.Value
var PDiscount G.Value
var PStateValue G.Value

// VAV implements the vanilla actor-critic algorithm
type VAC struct {
	// Policy
	behaviour         agent.NNPolicy   // Has its own VM
	trainPolicy       agent.LogPdfOfer // Policy struct that is learned
	trainPolicySolver G.Solver
	trainPolicyVM     G.VM
	pStateValue       *G.Node // For computing the advantage
	pNextStateValue   *G.Node // For computing the advantage
	pDiscount         *G.Node // For computing the advantage
	pReward           *G.Node // For computing the advantage
	logProb           *G.Node // Log PDF of actions sampled from ER
	advantage         *G.Node

	replay expreplay.ExperienceReplayer

	prevStep   ts.TimeStep
	actionDims int

	// State value critic
	vValueFn        network.NeuralNet
	vVM             G.VM
	vTrainValueFn   network.NeuralNet
	vTrainValueFnVM G.VM
	vSolver         G.Solver
	valueGradSteps  int
	vNextStateValue *G.Node
	vDiscount       *G.Node
	vReward         *G.Node

	// Target value function
	vTargetValueFn       network.NeuralNet
	vTargetValueFnVM     G.VM
	tau                  float64
	targetUpdateInterval int
	stepsSinceUpdate     int
}

// New returns a new VAC as described by the configuration c with
// actions selected for the environment e
func New(e env.Environment, c agent.Config, seed int64) (agent.Agent, error) {
	if !c.ValidAgent(&VAC{}) {
		return nil, fmt.Errorf("new: invalid configuration type: %T", c)
	}

	// Ensure we have a VAC config, as described in this package
	config, ok := c.(config)
	if !ok {
		return nil, fmt.Errorf("new: invalid configuration type: %T", c)
	}

	// Validate and adjust policy/critics as needed
	err := config.Validate()
	if err != nil {
		return nil, fmt.Errorf("new %v", err)
	}

	// Create the experience replay buffer
	featureSize := e.ObservationSpec().Shape.Len()
	actionSize := e.ActionSpec().Shape.Len()
	replay, err := config.expReplay().Create(featureSize, actionSize, seed,
		false)
	if err != nil {
		return nil, fmt.Errorf("new: could not construct experience "+
			"replay buffer: %v", err)
	}

	// Create the online prediction value function
	valueFn := config.valueFn()
	vVM := G.NewTapeMachine(valueFn.Graph())

	// Create the target value function
	targetValueFn := config.targetValueFn()
	vTargetVM := G.NewTapeMachine(targetValueFn.Graph())

	// Create the training value function, whose weights are learned
	trainValueFn := config.trainValueFn()

	// Create the next state value, reward, and discount placeholders
	// for creating the update target for the critic
	vNextStateValue := G.NewVector(
		trainValueFn.Graph(),
		tensor.Float64,
		G.WithName("NextStateValue_CriticLoss"),
		G.WithShape(config.batchSize()),
	)
	vReward := G.NewVector(
		trainValueFn.Graph(),
		tensor.Float64,
		G.WithName("Reward_CriticLoss"),
		G.WithShape(config.batchSize()),
	)
	vDiscount := G.NewVector(
		trainValueFn.Graph(),
		tensor.Float64,
		G.WithName("Discount_CriticLoss"),
		G.WithShape(config.batchSize()),
	)

	// Compute the critic's update target
	trainValueFnTargets := G.Must(G.HadamardProd(vDiscount, vNextStateValue))
	trainValueFnTargets = G.Must(G.Add(vReward, trainValueFnTargets))

	// Critic MSE loss
	prediction := trainValueFn.Prediction()[0]
	valueFnLoss := G.Must(G.Sub(prediction, trainValueFnTargets))
	valueFnLoss = G.Must(G.Square(valueFnLoss))
	valueFnLoss = G.Must(G.Mean(valueFnLoss))
	G.Read(valueFnLoss, &VLoss)

	// Calculate the value function gradient
	_, err = G.Grad(valueFnLoss, trainValueFn.Learnables()...)
	if err != nil {
		return nil, fmt.Errorf("new: could not compute value function "+
			"gradient: %v", err)
	}

	trainValueFnVM := G.NewTapeMachine(trainValueFn.Graph(),
		G.BindDualValues(trainValueFn.Learnables()...))

	// Create the prediction policy
	behaviour := config.behaviourPolicy()

	// Create the training policy
	trainPolicy := config.trainPolicy()

	// Create the log pdf node and the state value, next state value,
	// reward, and discount placeholders for computing the advantage,
	// 𝔸, to use in the policy gradient update:
	// 𝔸 = r + ℽ * v(s') - v(s)
	logProb := trainPolicy.LogPdfNode()
	G.Read(logProb, &LogPDF)
	pStateValue := G.NewVector(
		trainPolicy.Network().Graph(),
		tensor.Float64,
		G.WithName("StateValue_PolicyLoss"),
		G.WithShape(config.batchSize()),
	)
	pNextStateValue := G.NewVector(
		trainPolicy.Network().Graph(),
		tensor.Float64,
		G.WithName("NextStateValue_PolicyLoss"),
		G.WithShape(config.batchSize()),
	)
	pDiscount := G.NewVector(
		trainPolicy.Network().Graph(),
		tensor.Float64,
		G.WithName("Discount_PolicyLoss"),
		G.WithShape(config.batchSize()),
	)
	pReward := G.NewVector(
		trainPolicy.Network().Graph(),
		tensor.Float64,
		G.WithName("Reward_PolicyLoss"),
		G.WithShape(config.batchSize()),
	)

	// Compute the advantage 𝔸
	advantage := G.Must(G.HadamardProd(pDiscount, pNextStateValue))
	advantage = G.Must(G.Add(pReward, advantage))
	advantage = G.Must(G.Sub(advantage, pStateValue))
	G.Read(advantage, &Adv)
	G.Read(pNextStateValue, &PNextStateValue)
	G.Read(pStateValue, &PStateValue)
	G.Read(pReward, &PReward)
	G.Read(pDiscount, &PDiscount)

	// Construct the policy loss: -𝔼[ln(π) * 𝔸]
	// Where the negation ensures gradient ascent
	policyLoss := G.Must(G.HadamardProd(logProb, advantage))
	policyLoss = G.Must(G.Mean(policyLoss))
	policyLoss = G.Must(G.Neg(policyLoss))
	G.Read(policyLoss, &PLoss)

	// Calculate the policy gradient
	_, err = G.Grad(policyLoss, trainPolicy.Network().Learnables()...)
	if err != nil {
		return nil, fmt.Errorf("new: could not compute the policy "+
			"gradient: %v", err)
	}

	trainPolicyVM := G.NewTapeMachine(trainPolicy.Network().Graph(),
		G.BindDualValues(trainPolicy.Network().Learnables()...))

	// Create the agent
	return &VAC{
		behaviour:         behaviour,
		trainPolicy:       trainPolicy,
		trainPolicyVM:     trainPolicyVM,
		trainPolicySolver: config.policySolver(),

		pStateValue:     pStateValue,
		pNextStateValue: pNextStateValue,
		pReward:         pReward,
		pDiscount:       pDiscount,
		logProb:         logProb,
		advantage:       advantage,

		vValueFn: valueFn,
		vVM:      vVM,

		vTrainValueFn:   trainValueFn,
		vTrainValueFnVM: trainValueFnVM,
		vSolver:         config.vSolver(),
		valueGradSteps:  config.valueGradSteps(),

		vNextStateValue: vNextStateValue,
		vReward:         vReward,
		vDiscount:       vDiscount,

		vTargetValueFn:       targetValueFn,
		vTargetValueFnVM:     vTargetVM,
		tau:                  config.tau(),
		targetUpdateInterval: config.targetUpdateInterval(),
		stepsSinceUpdate:     0,

		replay:     replay,
		actionDims: e.ActionSpec().Shape.Len(),
	}, nil
}

// SelectAction returns an action for the timestep t
func (v *VAC) SelectAction(t ts.TimeStep) *mat.VecDense {
	a := v.behaviour.SelectAction(t)
	fmt.Println()
	fmt.Println(a)
	return a
}

// EndEpisode performs cleanup at the end of an episode
func (v *VAC) EndEpisode() {}

// Eval sets the agent into evaluation mode
func (v *VAC) Eval() { v.behaviour.Eval() }

// Train sets the agent into training mode
func (v *VAC) Train() { v.behaviour.Train() }

// IsEval returns whether the agent is in evaluation mode or not
func (v *VAC) IsEval() bool { return v.behaviour.IsEval() }

// ObserveFirst stores the first timestep in the episode
func (v *VAC) ObserveFirst(t ts.TimeStep) error {
	if !t.First() {
		return fmt.Errorf("observeFirst: timestep "+
			"called on the first timestep (current timestep = %d)", t.Number)
	}

	v.prevStep = t
	return nil
}

// Observe stores an action taken in the environment and the next
// time step as a result of taking that action
func (v *VAC) Observe(action mat.Vector, nextStep ts.TimeStep) error {
	fmt.Println(action)
	if !nextStep.First() {
		nextAction := mat.NewVecDense(v.actionDims, nil)
		transition := ts.NewTransition(v.prevStep, action.(*mat.VecDense),
			nextStep, nextAction)
		err := v.replay.Add(transition)
		if err != nil {
			return fmt.Errorf("observe: could not add to replay buffer: %v",
				err)
		}
	}

	v.prevStep = nextStep
	return nil
}

// Step performs the update of the agent, updating both the policy and
// value function
func (v *VAC) Step() error {
	// If in evaluation mode, don't update
	if v.IsEval() {
		return nil
	}

	// Sample transitions from the replay buffer
	S, A, rewards, discounts, NextS, _, err := v.replay.Sample()
	if expreplay.IsEmptyBuffer(err) || expreplay.IsInsufficientSamples(err) {
		return nil
	}

	// === === Get Values Needed To Compute Losses === ===
	// Predict the state value for the policy update
	err = v.vTargetValueFn.SetInput(S)
	if err != nil {
		return fmt.Errorf("step: could not set target network input state: %v",
			err)
	}
	err = v.vTargetValueFnVM.RunAll()
	if err != nil {
		return fmt.Errorf("step: could not run target network vm to compute "+
			"state value: %v", err)
	}

	// Set the state value tensor placeholder
	pStateValueTensor := tensor.NewDense(
		tensor.Float64,
		v.pStateValue.Shape(),
		tensor.WithBacking(floatutils.Duplicate(
			v.vTargetValueFn.Output()[0].Data().([]float64),
		)),
	)
	err = G.Let(v.pStateValue, pStateValueTensor)
	if err != nil {
		return fmt.Errorf("step: could not set state value for policy "+
			"target: %v", err)
	}
	v.vTargetValueFnVM.Reset()

	// Predict the next state value for the policy and critic updates
	err = v.vTargetValueFn.SetInput(NextS)
	if err != nil {
		return fmt.Errorf("step: could not set target network input "+
			"next state: %v", err)
	}
	err = v.vTargetValueFnVM.RunAll()
	if err != nil {
		return fmt.Errorf("step: could not run target network vm to compute "+
			"next state value: %v", err)
	}

	// Set the next state value tensor placeholders
	pNextStateValueTensor := tensor.NewDense(
		tensor.Float64,
		v.pNextStateValue.Shape(),
		tensor.WithBacking(floatutils.Duplicate(
			v.vTargetValueFn.Output()[0].Data().([]float64),
		)),
	)
	err = G.Let(v.pNextStateValue, pNextStateValueTensor)
	if err != nil {
		return fmt.Errorf("step: could not set next state value for "+
			"policy target: %v", err)
	}
	vNextStateValueTensor := tensor.NewDense(
		tensor.Float64,
		v.vNextStateValue.Shape(),
		tensor.WithBacking(floatutils.Duplicate(
			v.vTargetValueFn.Output()[0].Data().([]float64),
		)),
	)
	err = G.Let(v.vNextStateValue, vNextStateValueTensor)
	if err != nil {
		return fmt.Errorf("step: could not set next state value for "+
			"critic target: %v", err)
	}
	v.vTargetValueFnVM.Reset()

	// Set the reward tensor placeholders
	pRewardTensor := tensor.NewDense(
		tensor.Float64,
		v.pReward.Shape(),
		tensor.WithBacking(floatutils.Duplicate(rewards)),
	)
	err = G.Let(v.pReward, pRewardTensor)
	if err != nil {
		return fmt.Errorf("step: could not set reward for policy target: %v",
			err)
	}
	vRewardTensor := tensor.NewDense(
		tensor.Float64,
		v.vReward.Shape(),
		tensor.WithBacking(floatutils.Duplicate(rewards)),
	)
	err = G.Let(v.vReward, vRewardTensor)
	if err != nil {
		return fmt.Errorf("step: could not set reward for critic target: %v",
			err)
	}

	// Set the discount tensor placeholders
	pDiscountTensor := tensor.NewDense(
		tensor.Float64,
		v.pDiscount.Shape(),
		tensor.WithBacking(floatutils.Duplicate(discounts)),
	)
	err = G.Let(v.pDiscount, pDiscountTensor)
	if err != nil {
		return fmt.Errorf("step: could not set discount for policy target: %v",
			err)
	}
	vDiscountTensor := tensor.NewDense(
		tensor.Float64,
		v.vDiscount.Shape(),
		tensor.WithBacking(floatutils.Duplicate(discounts)),
	)
	err = G.Let(v.vDiscount, vDiscountTensor)
	if err != nil {
		return fmt.Errorf("step: could not set discount for critic target: %v",
			err)
	}

	// === === Policy Step === ===
	// Set the log probability of actions
	_, err = v.trainPolicy.LogPdfOf(S, A)
	if err != nil {
		return fmt.Errorf("step: could not set state and action input to "+
			"compute log PDF: %v", err)
	}

	// Update the policy weights
	err = v.trainPolicyVM.RunAll()
	if err != nil {
		return fmt.Errorf("step: could not run policy vm: %v", err)
	}
	err = v.trainPolicySolver.Step(v.trainPolicy.Network().Model())
	if err != nil {
		return fmt.Errorf("step: could not step policy solver: %v", err)
	}

	// Update behaviour policy
	err = network.Set(v.behaviour.Network(), v.trainPolicy.Network())
	if err != nil {
		return fmt.Errorf("step: could not copy training policy weights "+
			"to behvaiour policy: %v", err)
	}
	v.trainPolicyVM.Reset()

	// === === Value Function Train === ===
	for i := 0; i < v.valueGradSteps; i++ {
		err = v.vTrainValueFn.SetInput(S)
		if err != nil {
			return fmt.Errorf("step: could not set critic input state on "+
				"training iteration %d: %v", i, err)
		}
		err = v.vTrainValueFnVM.RunAll()
		if err != nil {
			return fmt.Errorf("step: could not run critic vm on training "+
				"iteration %: %v", i, err)
		}
		err = v.vSolver.Step(v.vTrainValueFn.Model())
		if err != nil {
			return fmt.Errorf("step: could not run step critic solver on "+
				"training iteration %: %v", i, err)
		}
		v.vTrainValueFnVM.Reset()
	}

	// Update the online value function
	err = network.Set(v.vValueFn, v.vTrainValueFn)
	if err != nil {
		return fmt.Errorf("step: could not copy training critic weights "+
			"to online critic: %v", err)
	}

	// Update the target network
	v.stepsSinceUpdate++
	if v.stepsSinceUpdate%v.targetUpdateInterval == 0 {
		if v.tau == 1.0 {
			err = network.Set(v.vTargetValueFn, v.vTrainValueFn)
		} else {
			err = network.Polyak(v.vTargetValueFn, v.vTrainValueFn, v.tau)
		}
		if err != nil {
			return fmt.Errorf("step: could not update target critic: %v", err)
		}
	}

	return nil
}

// TdError computes the TD error of a single transition
func (v *VAC) TdError(t ts.Transition) float64 {
	state := t.State.RawVector().Data
	nextState := t.NextState.RawVector().Data
	r := t.Reward
	ℽ := t.Discount

	// Get state value
	if err := v.vValueFn.SetInput(state); err != nil {
		panic(fmt.Sprintf("tdError: could not set network input: %v", err))
	}
	v.vVM.RunAll()
	stateValue := v.vValueFn.Output()[0].Data().([]float64)
	v.vVM.Reset()
	if len(stateValue) != 1 {
		panic("tdError: more than one state value predicted")
	}

	// Get next state value
	if err := v.vValueFn.SetInput(nextState); err != nil {
		panic(fmt.Sprintf("tdError: could not set network input: %v", err))
	}
	v.vVM.RunAll()
	nextStateValue := v.vValueFn.Output()[0].Data().([]float64)
	v.vVM.Reset()
	if len(nextStateValue) != 1 {
		panic("tdError: more than one next state value predicted")
	}

	return r + ℽ*nextStateValue[0] - stateValue[0]
}
