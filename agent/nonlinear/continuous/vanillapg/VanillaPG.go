package vanillapg

import (
	"fmt"
	"os"

	"gonum.org/v1/gonum/mat"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/network"
	ts "sfneuman.com/golearn/timestep"
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

	buffer           *gaeBuffer
	epochLength      int
	currentEpochStep int
	completedEpochs  int
	eval             bool

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
func New(env environment.Environment, c agent.Config, seed int64) (*VPG, error) {
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
	buffer := newGAEBuffer(features, actionDims, config.batchSize(),
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
		panic(err)
	}
	trainValueFnVM := G.NewTapeMachine(trainValueFn.Graph(), G.BindDualValues(trainValueFn.Learnables()...))

	// Create the prediction policy
	behaviour := config.behaviourPolicy()

	// Create the training policy
	trainPolicy := config.trainPolicy()
	logProb := trainPolicy.(agent.LogPdfOfer).LogPdfNode()
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
		panic(err)
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
		eval:                    false,
		finishingEpisode:        false,
		finishEpisodeOnEpochEnd: config.finishEpisodeOnEpochEnd(),
	}

	return vpg, nil
}

// SelectAction returns an action at the given timestep.
func (v *VPG) SelectAction(t ts.TimeStep) *mat.VecDense {
	if t != v.prevStep {
		panic("selectAction: timestep is different from that previously " +
			"recorded")
	}
	if !v.eval {
		return v.behaviour.SelectAction(t)
	} else {
		panic("selectAction: offline action selection not implemented")
	}
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
func (v *VPG) Eval() { v.eval = true }

// Train sets the algorithm into training mode
func (v *VPG) Train() { v.eval = false }

// ObserveFirst observes and records information about the first
// timestep in an episode.
func (v *VPG) ObserveFirst(t ts.TimeStep) {
	if !t.First() {
		fmt.Fprintf(os.Stderr, "Warning: ObserveFirst() should only be"+
			"called on the first timestep (current timestep = %d)", t.Number)
	}
	v.prevStep = t
}

// Observe observes and records any timestep other than the first timestep
func (v *VPG) Observe(action mat.Vector, nextStep ts.TimeStep) {
	// Finish current episode to end epoch
	if v.finishingEpisode {
		v.prevStep = nextStep
		return
	}

	// Calculate value of previous step
	o := v.prevStep.Observation.RawVector().Data
	err := v.vValueFn.SetInput(o)
	if err != nil {
		panic(err)
	}
	err = v.vVM.RunAll()
	if err != nil {
		panic(err)
	}
	vT := v.vValueFn.Output()[0].Data().([]float64)
	v.vVM.Reset()
	if len(vT) != 1 {
		panic("observe: multiple values predicted for state value")
	}
	r := nextStep.Reward
	a := action.(*mat.VecDense).RawVector().Data
	v.buffer.store(o, a, r, vT[0])

	// Update obs (critical!)
	v.prevStep = nextStep
	o = nextStep.Observation.RawVector().Data

	v.currentEpochStep++
	terminal := nextStep.Last() || v.currentEpochStep == v.epochLength
	if terminal {
		if nextStep.TerminalEnd() {
			v.buffer.finishPath(0.0)
		} else {
			err := v.vValueFn.SetInput(o)
			if err != nil {
				panic(err)
			}
			err = v.vVM.RunAll()
			if err != nil {
				panic(err)
			}
			lastVal := v.vValueFn.Output()[0].Data().([]float64)
			v.vVM.Reset()
			if len(lastVal) != 1 {
				panic("observe: multiple values predicted for next state value")
			}
			v.buffer.finishPath(lastVal[0])
			v.finishingEpisode = (v.currentEpochStep == v.epochLength) &&
				v.finishEpisodeOnEpochEnd
		}
	}
}

// Step updates the agent. If the agent is in evaluation mode, then
// this function simply returns.
func (v *VPG) Step() {
	if v.currentEpochStep < v.epochLength || v.eval {
		return
	}

	obs, act, adv, ret, err := v.buffer.get()
	if err != nil {
		panic(err)
	}

	// Policy gradient step
	advantagesTensor := tensor.NewDense( // * technically this needs to be called only once
		tensor.Float64,
		v.advantages.Shape(),
		tensor.WithBacking(adv),
	)
	err = G.Let(v.advantages, advantagesTensor)
	if err != nil {
		panic(err)
	}
	v.trainPolicy.LogPdfOf(obs, act)
	if err := v.trainPolicyVM.RunAll(); err != nil {
		panic(err)
	}
	if err := v.trainPolicySolver.Step(v.trainPolicy.Network().Model()); err != nil {
		panic(err)
	}
	v.trainPolicyVM.Reset()

	// Value function update
	for i := 0; i < v.valueGradSteps; i++ {
		trainValueFnTargetsTensor := tensor.NewDense(
			tensor.Float64,
			v.vTrainValueFnTargets.Shape(),
			tensor.WithBacking(ret),
		)
		err = G.Let(v.vTrainValueFnTargets, trainValueFnTargetsTensor)
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

	// Update behaviour policy and prediction value funcion
	network.Set(v.behaviour.Network(), v.trainPolicy.Network())
	network.Set(v.vValueFn, v.vTrainValueFn)
	v.completedEpochs++
	v.currentEpochStep = 0

}

// TdError implements the Agent interface; it always panics.
func (v *VPG) TdError(ts.Transition) float64 {
	panic("tderror: not implemented")
}
