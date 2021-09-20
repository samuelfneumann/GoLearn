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
	"gonum.org/v1/gonum/mat"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

var PLoss G.Value
var LogPDF G.Value

type VAC struct {
	// Policy
	behaviour         agent.NNPolicy   // Has its own VM
	trainPolicy       agent.LogPdfOfer // Policy struct that is learned
	trainPolicySolver G.Solver
	trainPolicyVM     G.VM
	advantages        *G.Node
	logProb           *G.Node

	replay expreplay.ExperienceReplayer

	prevStep   ts.TimeStep
	actionDims int

	// State value critic
	vValueFn             network.NeuralNet
	vVM                  G.VM
	vTrainValueFn        network.NeuralNet
	vTrainValueFnVM      G.VM
	vTrainValueFnTargets *G.Node
	vSolver              G.Solver
	valueGradSteps       int
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

	// Create the training value function
	trainValueFn := config.trainValueFn()

	// Create the training value function target and MSE loss
	trainValueFnTargets := G.NewMatrix(
		trainValueFn.Graph(),
		tensor.Float64,
		G.WithShape(trainValueFn.Prediction()[0].Shape()...),
		G.WithName("ValueFunctionUpdateTarget"),
	)
	valueFnLoss := G.Must(G.Sub(trainValueFn.Prediction()[0],
		trainValueFnTargets))
	valueFnLoss = G.Must(G.Square(valueFnLoss))
	valueFnLoss = G.Must(G.Mean(valueFnLoss))

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

	// Create the training policy loss
	logProb := trainPolicy.LogPdfNode()
	G.Read(logProb, &LogPDF)
	advantages := G.NewVector( // Really r + ℽv(S') - v(s) = TD error
		trainPolicy.Network().Graph(),
		tensor.Float64,
		G.WithName("Advantages"),
		G.WithShape(config.batchSize()),
	)
	policyLoss := G.Must(G.HadamardProd(logProb, advantages))
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

		advantages: advantages,
		logProb:    logProb,

		vValueFn: valueFn,
		vVM:      vVM,

		vTrainValueFn:        trainValueFn,
		vTrainValueFnTargets: trainValueFnTargets,
		vTrainValueFnVM:      trainValueFnVM,
		vSolver:              config.vSolver(),
		valueGradSteps:       config.valueGradSteps(),

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
	state, action, reward, discount, nextState, _, err := v.replay.Sample()
	if expreplay.IsEmptyBuffer(err) || expreplay.IsInsufficientSamples(err) {
		return
	} else if err != nil {
		panic(err)
	}

	// Needed to calculate the state value function
	target := tensor.NewDense(
		tensor.Float64,
		v.vTrainValueFnTargets.Shape(),
		tensor.WithBacking(make([]float64, v.vTrainValueFnTargets.Shape()[0])),
	)

	// Calculate state value
	err = v.vTrainValueFn.SetInput(state)
	if err != nil {
		panic(err)
	}
	err = G.Let(v.vTrainValueFnTargets, target)
	if err != nil {
		panic(err)
	}
	err = v.vTrainValueFnVM.RunAll()
	if err != nil {
		panic(err)
	}
	stateValue := v.vTrainValueFn.Output()[0].Data().([]float64)
	v.vTrainValueFnVM.Reset()

	// Calculate next state value
	err = v.vTrainValueFn.SetInput(nextState)
	if err != nil {
		panic(err)
	}
	err = G.Let(v.vTrainValueFnTargets, target)
	if err != nil {
		panic(err)
	}
	err = v.vTrainValueFnVM.RunAll()
	if err != nil {
		panic(err)
	}
	nextStateValue := v.vTrainValueFn.Output()[0].Data().([]float64)
	v.vTrainValueFnVM.Reset()

	// Calculate TD error
	sValue := mat.NewVecDense(len(stateValue), stateValue)
	nextSValue := mat.NewVecDense(len(nextStateValue), nextStateValue)
	r := mat.NewVecDense(len(reward), reward)
	d := mat.NewVecDense(len(discount), discount)

	advantage := mat.NewVecDense(sValue.Len(), nil)
	advantage.MulElemVec(d, nextSValue)
	advantage.AddVec(r, advantage)
	advantage.SubVec(advantage, sValue)
	advantageTensor := tensor.NewDense(
		tensor.Float64,
		v.advantages.Shape(),
		tensor.WithBacking(advantage.RawVector().Data),
	)

	// Update the policy
	err = G.Let(v.advantages, advantageTensor)
	if err != nil {
		panic(err)
	}
	v.trainPolicy.LogPdfOf(state, action)
	if err := v.trainPolicyVM.RunAll(); err != nil {
		panic(err)
	}
	err = v.trainPolicySolver.Step(v.trainPolicy.Network().Model())
	if err != nil {
		panic(err)
	}
	v.trainPolicyVM.Reset()

	// Update the value function
	for i := 0; i < v.valueGradSteps; i++ {
		target := tensor.NewDense(
			tensor.Float64,
			v.vTrainValueFnTargets.Shape(),
			tensor.WithBacking(nextSValue.RawVector().Data),
		)
		err = G.Let(v.vTrainValueFnTargets, target)
		if err != nil {
			panic(err)
		}

		if err := v.vTrainValueFnVM.RunAll(); err != nil {
			panic(err)
		}
		if err := v.vSolver.Step(v.vTrainValueFn.Model()); err != nil {
			panic(err)
		}
		v.vTrainValueFnVM.Reset()
	}

	// Update behaviour policy and prediction value function
	network.Set(v.behaviour.Network(), v.trainPolicy.Network())
	network.Set(v.vValueFn, v.vTrainValueFn)
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
