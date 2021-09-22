// Package vanillaac implements a vanilla actor-critic algorithm
package vanillaac

import (
	"fmt"
	"os"

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

// With Adam, this explodes on Gridworld
// With vanilla, it does learn on Gridworl, but explodes on Cartpole...

var PLoss G.Value
var VLoss G.Value
var LogPDF G.Value
var Adv G.Value

type VAC struct {
	// Policy
	behaviour         agent.NNPolicy   // Has its own VM
	trainPolicy       agent.LogPdfOfer // Policy struct that is learned
	trainPolicySolver G.Solver
	trainPolicyVM     G.VM
	pStateValue       *G.Node
	pNextStateValue   *G.Node
	pDiscount         *G.Node
	pReward           *G.Node
	logProb           *G.Node

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

	vTargetValueFn       network.NeuralNet
	vTargetValueFnVM     G.VM
	tau                  float64
	targetUpdateInterval int
	stepsSinceUpdate     int
}

func New(e env.Environment, c agent.Config, seed int64) (agent.Agent, error) {
	if !c.ValidAgent(&VAC{}) {
		return nil, fmt.Errorf("new: invalid configuration type: %T", c)
	}

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

	// Create the prediction value function
	valueFn := config.valueFn()
	vVM := G.NewTapeMachine(valueFn.Graph())

	// Create the target value function
	targetValueFn := config.targetValueFn()
	vTargetVM := G.NewTapeMachine(targetValueFn.Graph())

	// Create the training value function
	trainValueFn := config.trainValueFn()
	graph := trainValueFn.Graph()

	// Create the training value function target and MSE loss
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
	trainValueFnTargets := G.Must(G.HadamardProd(vDiscount, vNextStateValue))
	trainValueFnTargets = G.Must(G.Add(vReward, trainValueFnTargets))

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

	trainValueFnVM := G.NewTapeMachine(graph,
		G.BindDualValues(trainValueFn.Learnables()...))

	// Create the prediction policy
	behaviour := config.behaviourPolicy()

	// Create the training policy
	trainPolicy := config.trainPolicy()

	// Create the training policy loss
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
	advantage := G.Must(G.HadamardProd(pDiscount, pNextStateValue))
	advantage = G.Must(G.Add(pReward, advantage))
	advantage = G.Must(G.Sub(advantage, pStateValue))
	G.Read(advantage, &Adv)

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
	vac := &VAC{
		behaviour:         behaviour,
		trainPolicy:       trainPolicy,
		trainPolicyVM:     trainPolicyVM,
		trainPolicySolver: config.policySolver(),

		pStateValue:     pStateValue,
		pNextStateValue: pNextStateValue,
		pReward:         pReward,
		pDiscount:       pDiscount,
		logProb:         logProb,

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
	}

	return vac, nil
}

func (v *VAC) SelectAction(t ts.TimeStep) *mat.VecDense {
	return v.behaviour.SelectAction(t)
}

func (v *VAC) EndEpisode() {}

func (v *VAC) Eval() { v.behaviour.Eval() }

func (v *VAC) Train() { v.behaviour.Train() }

func (v *VAC) IsEval() bool { return v.behaviour.IsEval() }

func (v *VAC) ObserveFirst(t ts.TimeStep) {
	if !t.First() {
		fmt.Fprintf(os.Stderr, "Warning: ObserveFirst() should only be "+
			"called on the first timestep (current timestep = %d", t.Number)
	}

	v.prevStep = t
}

func (v *VAC) Observe(action mat.Vector, nextStep ts.TimeStep) {
	if !nextStep.First() {
		nextAction := mat.NewVecDense(v.actionDims, nil)
		transition := ts.NewTransition(v.prevStep, action.(*mat.VecDense),
			nextStep, nextAction)
		err := v.replay.Add(transition)
		if err != nil {
			panic(fmt.Sprintf("observe: could not add to replay buffer: %v",
				err))
		}
	}

	v.prevStep = nextStep
}

func (v *VAC) Step() {
	if v.IsEval() {
		return
	}

	S, A, rewards, discounts, NextS, _, err := v.replay.Sample()
	if expreplay.IsEmptyBuffer(err) || expreplay.IsInsufficientSamples(err) {
		return
	}

	v.stepsSinceUpdate++

	// == === Get Values Needed To Compute Targets === ===
	// Predict the state value
	err = v.vTargetValueFn.SetInput(S)
	if err != nil {
		panic(err)
	}
	err = v.vTargetValueFnVM.RunAll()
	if err != nil {
		panic(err)
	}

	// Set the state value tensor
	pStateValueTensor := tensor.NewDense(
		tensor.Float64,
		v.pStateValue.Shape(),
		tensor.WithBacking(floatutils.Duplicate(
			v.vTargetValueFn.Output()[0].Data().([]float64),
		)),
	)
	err = G.Let(v.pStateValue, pStateValueTensor)
	if err != nil {
		panic(err)
	}
	v.vTargetValueFnVM.Reset()

	// Next state value
	err = v.vTargetValueFn.SetInput(NextS)
	if err != nil {
		panic(err)
	}
	err = v.vTargetValueFnVM.RunAll()
	if err != nil {
		panic(err)
	}

	// Set the next state value tensor
	pNextStateValueTensor := tensor.NewDense(
		tensor.Float64,
		v.pNextStateValue.Shape(),
		tensor.WithBacking(floatutils.Duplicate(
			v.vTargetValueFn.Output()[0].Data().([]float64),
		)),
	)
	err = G.Let(v.pNextStateValue, pNextStateValueTensor)
	if err != nil {
		panic(err)
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
		panic(err)
	}
	v.vTargetValueFnVM.Reset()

	// Set the reward tensor
	pRewardTensor := tensor.NewDense(
		tensor.Float64,
		v.pReward.Shape(),
		tensor.WithBacking(floatutils.Duplicate(rewards)),
	)
	err = G.Let(v.pReward, pRewardTensor)
	if err != nil {
		panic(err)
	}
	vRewardTensor := tensor.NewDense(
		tensor.Float64,
		v.vReward.Shape(),
		tensor.WithBacking(floatutils.Duplicate(rewards)),
	)
	err = G.Let(v.vReward, vRewardTensor)
	if err != nil {
		panic(err)
	}

	// Set the discount tensor
	pDiscountTensor := tensor.NewDense(
		tensor.Float64,
		v.pDiscount.Shape(),
		tensor.WithBacking(floatutils.Duplicate(discounts)),
	)
	err = G.Let(v.pDiscount, pDiscountTensor)
	if err != nil {
		panic(err)
	}
	vDiscountTensor := tensor.NewDense(
		tensor.Float64,
		v.vDiscount.Shape(),
		tensor.WithBacking(floatutils.Duplicate(discounts)),
	)
	err = G.Let(v.vDiscount, vDiscountTensor)
	if err != nil {
		panic(err)
	}

	if v.valueGradSteps != 1 {
		panic("valueGradSteps > 1 not implemented yet")
	}

	// === === Policy Train === ===
	// Set the log probability of actions
	_, err = v.trainPolicy.LogPdfOf(S, A)
	if err != nil {
		panic(err)
	}

	// Update the weights
	err = v.trainPolicyVM.RunAll()
	if err != nil {
		panic(err)
	}
	err = v.trainPolicySolver.Step(v.trainPolicy.Network().Model())
	if err != nil {
		panic(err)
	}

	// Update behaviour policy
	err = network.Set(v.behaviour.Network(), v.trainPolicy.Network())
	if err != nil {
		panic(err)
	}
	v.trainPolicyVM.Reset()

	// === === Value Function Train === ===
	err = v.vTrainValueFn.SetInput(S)
	if err != nil {
		panic(err)
	}
	err = v.vTrainValueFnVM.RunAll()
	if err != nil {
		panic(err)
	}
	err = v.vSolver.Step(v.vTrainValueFn.Model())
	if err != nil {
		panic(err)
	}

	// Update other value functions
	err = network.Set(v.vValueFn, v.vTrainValueFn)
	if err != nil {
		panic(err)
	}
	if v.stepsSinceUpdate%v.targetUpdateInterval == 0 {
		if v.tau == 1.0 {
			err = network.Set(v.vTargetValueFn, v.vTrainValueFn)
			if err != nil {
				panic(err)
			}
		} else {
			err = network.Polyak(v.vTargetValueFn, v.vTrainValueFn, v.tau)
			if err != nil {
				panic(err)
			}
		}
	}
	v.vTrainValueFnVM.Reset()
}

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
