package vanillapg

import (
	"fmt"
	"strings"

	"github.com/samuelfneumann/golearn/agent"
	"github.com/samuelfneumann/golearn/buffer/gae"
	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/network"
	ts "github.com/samuelfneumann/golearn/timestep"
	"gonum.org/v1/gonum/mat"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// Note: Step() is called on each timestep. When the epoch is finished
// the current episode may not be finised, but Step() will be called,
// updating the current policy. In this case, we will finish the
// episode with an updated policy, but none of this data will be
// recorded or used to update the policy. Instead, we finish the episode
// and start the next epoch from the beginning of the next episode.
//
// Since the data collected at the end of the last episode will be
// collected with the updated policy, we could
// actually keep this data and begin adding it to the new buffer. Then
// we would be updating using the new buffer, which contains data from
// the middle of an episode. But, since this data is collected with the
// current policy, all the data used to update will be from the same
// policy, and everything would work fine if we updated with this
// data. Since many current implementations do not do this but rather
// throw out the data remaining in the episode, we also follow this
// practice, but note that in our implementation the other method is
// not incorrect.
//
// One caveat with the current implementation is that if given a
// timestep budget to learn, the current implementation will not
// update its policy at the end of training if the last timestep of
// training is not the last timestep needed in the epoch. With the
// altered implementation, we could say have 25,000 timesteps of
// learning, then easily say we want 5 epochs of 5,000 timesteps, and
// the final update would be done at the end of training. In the current
// implementation, we don't know how many timesteps will be disregraded
// due to epoch ends (but episodes not ending), so there is no way
// to ensure that the policy will update again at the end of training.

// VPG implements the Vanilla Policy Gradient algorithm with generalized
// advantage estimation. This implementation is adapted from:
//
// https://spinningup.openai.com/en/latest/algorithms/vpg.html
// https://github.com/openai/spinningup/blob/master/spinup/algos/tf1/vpg/vpg.py
type VPG struct {
	// Policy
	behaviour         agent.NNPolicy   // Has its own VM
	trainPolicy       agent.LogPdfOfer // Policy struct that is learned
	trainPolicySolver G.Solver
	trainPolicyVM     G.VM
	advantages        *G.Node // For gradient construction
	logProb           *G.Node // For gradient construction

	buffer           *gae.Buffer
	epochLength      int
	currentEpochStep int
	completedEpochs  int

	// finishEpoch becomes true when the number of steps recorded
	// is equal to the total number of steps allowed in the epoch.
	// In this case, the agent continues to act in the environment,
	// but we can no longer store any data in the buffer. Hence, the
	// rest of the episode is finished, but its data discarded.
	//
	// See note above.
	finishingEpisode bool

	// finishEpisodeOnEpochEnd denotes if the current episode should
	// be finished before starting a new epoch. If true, then the
	// agent is updated when the current epoch ends, then the current
	// episode is finished, then the next epoch starts. If false, the
	// agent is updated when the current epoch is finished, and the
	// next epoch starts at the next timestep, which may be in the
	// middle of an episode.
	finishEpisodeOnEpochEnd bool

	prevStep ts.TimeStep

	// State value critic
	vValueFn             network.NeuralNet
	vVM                  G.VM
	vTrainValueFn        network.NeuralNet
	vTrainValueFnVM      G.VM
	vTrainValueFnTargets *G.Node
	vSolver              G.Solver
	valueGradSteps       int
}

// New creates and returns a new VanillaPG.
func New(env environment.Environment, c agent.Config,
	seed int64) (agent.Agent, error) {
	if !c.ValidAgent(&VPG{}) {
		return nil, fmt.Errorf("new: invalid configuration type: %T", c)
	}

	config, ok := c.(config)
	if !ok {
		return nil, fmt.Errorf("new: invalid configuration type: %T", c)

	}

	// Validate and adjust policy/critics as needed
	err := config.Validate()
	if err != nil {
		return nil, fmt.Errorf("new: %v", err)
	}

	// Create the VPG buffer
	features := env.ObservationSpec().Shape.Len()
	actionDims := env.ActionSpec().Shape.Len()
	buffer := gae.New(features, actionDims, config.batchSize(),
		config.lambda(), config.gamma())

	// Create the prediction value function
	valueFn := config.valueFn()
	vVM := G.NewTapeMachine(valueFn.Graph())

	// Create the training value function
	trainValueFn := config.trainValueFn()

	trainValueFnTargets := G.NewMatrix(
		trainValueFn.Graph(),
		tensor.Float64,
		G.WithShape(trainValueFn.Prediction()[0].Shape()...),
		G.WithName("Value Function Update Target"),
	)

	valueFnLoss := G.Must(G.Sub(trainValueFn.Prediction()[0], trainValueFnTargets))
	valueFnLoss = G.Must(G.Square(valueFnLoss))
	valueFnLoss = G.Must(G.Mean(valueFnLoss))

	_, err = G.Grad(valueFnLoss, trainValueFn.Learnables()...)
	if err != nil {
		return nil, fmt.Errorf("new: could not compute value function "+
			"gradient: %v", err)
	}
	trainValueFnVM := G.NewTapeMachine(trainValueFn.Graph(), G.BindDualValues(trainValueFn.Learnables()...))

	// Create the prediction policy
	behaviour := config.behaviourPolicy()

	// Create the training policy
	trainPolicy := config.trainPolicy()
	logProb := trainPolicy.LogPdfNode()
	advantages := G.NewVector(
		trainPolicy.Network().Graph(),
		tensor.Float64,
		G.WithName("Advantages"),
		G.WithShape(config.epochLength()),
	)

	policyLoss := G.Must(G.HadamardProd(logProb, advantages))
	policyLoss = G.Must(G.Mean(policyLoss))
	policyLoss = G.Must(G.Neg(policyLoss))

	_, err = G.Grad(policyLoss, trainPolicy.Network().Learnables()...)
	if err != nil {
		return nil, fmt.Errorf("new: could not compute the policy "+
			"gradient: %v", err)
	}
	trainPolicyVM := G.NewTapeMachine(trainPolicy.Network().Graph(), G.BindDualValues(trainPolicy.Network().Learnables()...))

	vpg := &VPG{
		behaviour:         behaviour,
		trainPolicy:       trainPolicy,
		trainPolicyVM:     trainPolicyVM,
		trainPolicySolver: config.policySolver(),
		advantages:        advantages,
		logProb:           logProb,

		vValueFn: valueFn,
		vVM:      vVM,

		vTrainValueFn:        trainValueFn,
		vTrainValueFnTargets: trainValueFnTargets,
		vTrainValueFnVM:      trainValueFnVM,
		vSolver:              config.vSolver(),
		valueGradSteps:       config.valueGradSteps(),

		buffer:                  buffer,
		epochLength:             config.epochLength(),
		currentEpochStep:        0,
		completedEpochs:         0,
		finishingEpisode:        false,
		finishEpisodeOnEpochEnd: config.finishEpisodeOnEpochEnd(),
	}

	return vpg, nil
}

// SelectAction returns an action at the given timestep.
func (v *VPG) SelectAction(t ts.TimeStep) *mat.VecDense {
	return v.behaviour.SelectAction(t)
}

// EndEpisode performs cleanup at the end of an episode.
func (v *VPG) EndEpisode() {
	// If the previous epoch finished before the episode finished, the
	// ending of the previous episode would have been thrown out. Since
	// a new episode is starting now, we can begin storing data for
	// the current epoch.
	v.finishingEpisode = false
}

// Eval sets the algorithm into evaluation mode
func (v *VPG) Eval() { v.behaviour.Eval() }

// Train sets the algorithm into training mode
func (v *VPG) Train() { v.behaviour.Train() }

// IsEval returns whether the agent is in evaluation mode
func (v *VPG) IsEval() bool { return v.behaviour.IsEval() }

// ObserveFirst observes and records information about the first
// timestep in an episode.
func (v *VPG) ObserveFirst(t ts.TimeStep) error {
	if !t.First() {
		return fmt.Errorf("observeFirst: timestep is not first "+
			"(current timestep = %d)", t.Number)
	}
	v.prevStep = t

	return nil
}

// Observe observes and records any timestep other than the first timestep
func (v *VPG) Observe(action mat.Vector, nextStep ts.TimeStep) error {
	// Finish current episode to end epoch
	if v.finishingEpisode {
		v.prevStep = nextStep
		return nil
	}

	// Calculate value of previous step
	o := v.prevStep.Observation.RawVector().Data
	err := v.vValueFn.SetInput(o)
	if err != nil {
		return fmt.Errorf("observe: could not set value function input: %v", err)
	}
	err = v.vVM.RunAll()
	if err != nil {
		return fmt.Errorf("observe: could not run value function vm: %v", err)
	}
	vT := v.vValueFn.Output()[0].Data().([]float64)
	v.vVM.Reset()
	if len(vT) != 1 {
		// This should never happen if using Config structs
		panic("observe: multiple values predicted for state value")
	}
	r := nextStep.Reward
	a := action.(*mat.VecDense).RawVector().Data
	v.buffer.Store(o, a, r, vT[0])

	// Update obs (critical!)
	v.prevStep = nextStep
	o = nextStep.Observation.RawVector().Data

	v.currentEpochStep++
	terminal := nextStep.Last() || v.currentEpochStep == v.epochLength
	if terminal {
		if nextStep.TerminalEnd() {
			v.buffer.FinishPath(0.0)
		} else {
			err := v.vValueFn.SetInput(o)
			if err != nil {
				return fmt.Errorf("observe: could not set value function "+
					"input for terminal step: %v", err)
			}
			err = v.vVM.RunAll()
			if err != nil {
				return fmt.Errorf("observe: could not run value function "+
					"vm for terminal step: %v", err)
			}
			lastVal := v.vValueFn.Output()[0].Data().([]float64)
			v.vVM.Reset()
			if len(lastVal) != 1 {
				// This should never happen if using Config structs
				panic("observe: multiple values predicted for next " +
					"state value")
			}
			v.buffer.FinishPath(lastVal[0])
			v.finishingEpisode = (v.currentEpochStep == v.epochLength) &&
				v.finishEpisodeOnEpochEnd
		}
	}
	return nil
}

// Step updates the agent. If the agent is in evaluation mode, then
// this function simply returns.
func (v *VPG) Step() error {
	if v.currentEpochStep < v.epochLength || v.IsEval() {
		return nil
	}

	obs, act, adv, ret, err := v.buffer.Get()
	if err != nil {
		return fmt.Errorf("step: could not sample from buffer: %v", err)
	}

	// Policy gradient step
	advantagesTensor := tensor.NewDense(
		tensor.Float64,
		v.advantages.Shape(),
		tensor.WithBacking(adv),
	)
	err = G.Let(v.advantages, advantagesTensor)
	if err != nil {
		return fmt.Errorf("step: could not set advantages tensor: %v", err)
	}
	v.trainPolicy.LogPdfOf(obs, act)
	if err := v.trainPolicyVM.RunAll(); err != nil {
		return fmt.Errorf("step: could not set state and action for log PDF "+
			"calculation: %v", err)
	}
	if err := v.trainPolicySolver.Step(v.trainPolicy.Network().Model()); err != nil {
		return fmt.Errorf("step: could not step policy solver: %v", err)
	}
	v.trainPolicyVM.Reset()

	// Set value function input
	if err := v.vTrainValueFn.SetInput(obs); err != nil {
		return fmt.Errorf("step: could not set value function input: %v", err)
	}

	// Set value function target
	trainValueFnTargetsTensor := tensor.NewDense(
		tensor.Float64,
		v.vTrainValueFnTargets.Shape(),
		tensor.WithBacking(ret),
	)
	err = G.Let(v.vTrainValueFnTargets, trainValueFnTargetsTensor)
	if err != nil {
		return fmt.Errorf("step: could not set value function target: %v", err)
	}

	// Update value function
	for i := 0; i < v.valueGradSteps; i++ {
		if err := v.vTrainValueFnVM.RunAll(); err != nil {
			return fmt.Errorf("step: could not run value function vm "+
				"at training iteration %d: %v", i, err)
		}
		if err := v.vSolver.Step(v.vTrainValueFn.Model()); err != nil {
			return fmt.Errorf("step: could not step value function solver "+
				"at training iteration %d: %v", i, err)
		}
		v.vTrainValueFnVM.Reset()
	}

	// Update behaviour policy and prediction value funcion
	network.Set(v.behaviour.Network(), v.trainPolicy.Network())
	network.Set(v.vValueFn, v.vTrainValueFn)

	v.completedEpochs++
	v.currentEpochStep = 0

	return nil
}

// TdError returns the TD error of the agent's value function for a
// given transition.
func (v *VPG) TdError(t ts.Transition) float64 {
	state := t.State
	nextState := t.NextState
	r := t.Reward
	ℽ := t.Discount

	// Get state value
	if err := v.vValueFn.SetInput(state.RawVector().Data); err != nil {
		panic(fmt.Sprintf("tdError: could not set network input: %v", err))
	}
	v.vVM.RunAll()
	stateValue := v.vValueFn.Output()[0].Data().([]float64)
	v.vVM.Reset()
	if len(stateValue) != 1 {
		panic("tdError: more than one state value predicted")
	}

	// Get next state value
	if err := v.vValueFn.SetInput(nextState.RawVector().Data); err != nil {
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

// Close cleans up any used resources
func (v *VPG) Close() error {
	behaviourVMErr := v.behaviour.Close()
	trainPolicyVMErr := v.trainPolicy.Close()
	valueFnVMErr := v.vTrainValueFnVM.Close()
	trainValueFnVMErr := v.vTrainValueFnVM.Close()

	flag := false
	var errBuilder strings.Builder
	errBuilder.WriteString("close: could not close")

	if behaviourVMErr != nil {
		flag = true
		errBuilder.WriteString(" behaviour policy")
	}

	if trainPolicyVMErr != nil {
		if flag {
			errBuilder.WriteString(", train policy")
		} else {
			flag = true
			errBuilder.WriteString(" train policy")
		}
	}

	if valueFnVMErr != nil {
		if flag {
			errBuilder.WriteString(", value function")
		} else {
			flag = true
			errBuilder.WriteString(" value function")
		}
	}

	if trainValueFnVMErr != nil {
		if flag {
			errBuilder.WriteString(", train value function")
		} else {
			flag = true
			errBuilder.WriteString(" train value function")
		}
	}

	if flag {
		return fmt.Errorf(errBuilder.String())
	}
	return nil
}
