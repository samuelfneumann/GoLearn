package policy

// ! SAC e.g. will have two GaussianTreeMLPs, one with and one without
// ! batches. With batchs --> learning weights. Without batches --> selecting
// ! actions. If the batch size is 1, then only one policy is required,
// ! and we can use the LobProb() of this policy without re-running it
// ! at each timestep to get the log probability. For batches > 1,
// ! we would first have to sample from replay -> run train policy to
// ! get log prob. For batches of size 1, the LogProb() will already
// ! hold the log prob of the last selected action -> no need to run the
// ! policy again.

import (
	"fmt"
	"log"
	"math"

	"gonum.org/v1/gonum/mat"
	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"sfneuman.com/golearn/agent"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/network"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/tensorutils"
)

// GaussianTreeMLP implements a Gaussian policy that uses a tree MLP to
// predict the mean and log standard deviation of its policy. Given an
// environment with N-dimensional actions, this policy will produce
// an N-dimensional action, sampled from a Gaussian distribution, in
// each input state.
//
// GaussianTreeMLP simply populates a gorgonia.ExprGraph with
// the neural network function approximator and selects actions
// based on the output of this neural network, which predicts the
// standard deviation and mean of a Gaussian policy, conditioned on
// a given input state. That is:
//
//		π(A|S) ~ N(μ, σ)
//
// where the mean μ and standard deviation σ are predicted by the neural
// net, and π denotes the policy.
//
// ! So far, the Gaussian policy is not seedable since we use
// ! gorgonia's GaussianRandomNode().
//
// ! This version of a Gaussian policy will take in some state S and
// ! compute the log probabilities for the actions the policy would
// ! sample in that state, not the actual actions taken. For example,
// ! if given a tuple (S, A, R, S'), this the logProb() will compute
// ! the log probability of A' taken in S' as well as compute each A'.
// ! This is because this implementation assumes a "SAC-like" update.
// ! Given (S, A, R, S') from a replay buffer, we use the policy to
// ! predict A' and logProb(A' | S'). This A' is then used in the
// ! in the critic to get a value of Q(A', S') to use in the gradient.
//
// ! if we want to compute log prob of action A in state S, we will
// ! need another function which takes batches of states/actions as
// ! inputs, sets the states as input to the NN, then adds the actions
// ! to an input node -> calculates the log prob of those actions.
type GaussianTreeMLP struct {
	network.NeuralNet

	mean, std G.Value

	logProb    *G.Node
	logProbVal G.Value
	actions    *G.Node // Node of action(s) to take in input state(s)
	actionsVal G.Value // Value of action(s) to take in input state(s)
	actionDims int

	// External actions refer to actions that are given to the policy
	// with which to calculate the log probability of.
	externActions           *G.Node
	externActionsLogProb    *G.Node
	externActionsLogProbVal G.Value

	seed uint64

	vm G.VM // VM for action selection
}

// NewGaussianTreeMLP creates and returns a new Gaussian policy, with
// mean and log standard deviation predicted by a tree MLP.
//
// The rootHiddenSizes, rootBiases, and rootActivations parameters
// determine the architecture of the root MLP of the tree MLP. For index
// i, rootHiddenSizes[i] determines the number of hidden units in the
// ith layer; rootBiases[i] determines whether or not a bias unit is
// added to the ith layer; rootActivations[i] dteremines the activation
// function of the ith layer. The number of layer in the root network is
// determined by len(rootHiddenSizes).
//
// The number of leaf networks is defined by len(leafHiddenSizes) and
// must be equal to 2, one to predict μ and the other to predict log(σ).
// For indices i and j, leafHiddenSizes[i][j], leafBiases[i][j], and
// leafActivations[i][j] determine the number of hidden units of layer
// j in leaf network i, whether a bias is added to layer j of leaf
// network i, and the activation of layer j of leaf network i
// respectively. The length of leafHiddenSizes[i] determines the number
// of hidden layers in leaf network i.
//
// To each leaf network, a final linear layer is added (with a bias
// unit and no activations) to ensure that the output shape matches
// the action dimensionality of the environment. Additionally, the
// predicted log(σ) is first exponentiated before sampling an action.
// For more details on the neural network function approximator, see
// the network package.
//
// The batch size is determined by batchForLogProb. If this value is
// set above 1, then it is assumed that the policy will not be used
// to select actions at each timestep. Instead, it is assumed that
// the policy will be used to produce the log(π(A|S)) required for
// learning the policy, and that this will be produced for all states
// in a batch. In this case, the method LogProb() will return the node
// that will hold the log probabilities when the computational graph
// is run. If the batch size is set to 1, then actions can be selected
// at each timestep, and the node returned by LogProb() will hold the
// log probability of selecting this action or any other action
// selected by the policy when its neural networks is given input and
// the computational graph run.
func NewGaussianTreeMLP(env environment.Environment, batchForLogProb int,
	g *G.ExprGraph, rootHiddenSizes []int, rootBiases []bool,
	rootActivations []*network.Activation, leafHiddenSizes [][]int,
	leafBiases [][]bool, leafActivations [][]*network.Activation,
	init G.InitWFn, seed uint64) (agent.PolicyLogProber, error) {

	// Error checking
	if env.ActionSpec().Cardinality == spec.Discrete {
		err := fmt.Errorf("newGaussianTreeMLP: gaussian policy cannot be " +
			"used with discrete actions")
		return &GaussianTreeMLP{}, err
	}

	if len(leafHiddenSizes) != 2 {
		return &GaussianTreeMLP{}, fmt.Errorf("newGaussianTreeMLP: gaussian " +
			"policy requires 2 leaf networks only")
	}

	features := env.ObservationSpec().Shape.Len()
	actionDims := env.ActionSpec().Shape.Len()

	// Create the tree MLP, which may predict batches of means and
	// log standard deviations -> one for each state in the batch
	net, err := network.NewTreeMLP(features, batchForLogProb, actionDims, g,
		rootHiddenSizes, rootBiases, rootActivations, leafHiddenSizes,
		leafBiases, leafActivations, init)
	if err != nil {
		return &GaussianTreeMLP{}, fmt.Errorf("newGaussianTreeMLP: could "+
			"not create policy network: %v", err)
	}

	// Exponentiate the log standard deviation
	logStd := net.Prediction()[0]
	stdNode := G.Must(G.Exp(logStd))
	meanNode := net.Prediction()[1]

	// Reparameterization trick A = μ + σ*ε, where ε ~ N(0, 1)
	actionPerturb := G.GaussianRandomNode(net.Graph(), tensor.Float64,
		0, 1, batchForLogProb, net.Outputs()[0])
	actionStd := G.Must(G.HadamardProd(stdNode, actionPerturb))
	actions := G.Must(G.Add(meanNode, actionStd))

	// Calculate log probability
	logProbNode, err := logProb(meanNode, stdNode, actions)
	if err != nil {
		return nil, fmt.Errorf("newGaussianTreeMLP: could not calculate "+
			"log probabiltiy: %v", err)
	}

	// Create external actions
	externActions := G.NewMatrix(
		net.Graph(),
		tensor.Float64,
		G.WithName("externActions"),
		G.WithShape(net.BatchSize(), actionDims),
	)
	logProbExternalActions, err := logProb(meanNode, stdNode, externActions)
	if err != nil {
		return nil, fmt.Errorf("newGaussianTreMLP: could not calculate "+
			"log probability of external input actions: %v", err)
	}

	policy := GaussianTreeMLP{
		NeuralNet:            net,
		logProb:              logProbNode,
		actions:              actions,
		seed:                 seed,
		externActions:        externActions,
		externActionsLogProb: logProbExternalActions,
		actionDims:           actionDims,
	}

	// Store the values of the actions selected for the batch, standard
	// deviations and means for the policy in each state in the batch,
	// and the log probability of selecting each action in the batch.
	G.Read(policy.actions, &policy.actionsVal)
	G.Read(stdNode, &policy.std)
	G.Read(meanNode, &policy.mean)
	G.Read(logProbNode, &policy.logProbVal)
	G.Read(logProbExternalActions, &policy.externActionsLogProbVal)

	// Action selection VM is used only for policies with batches of size 1.
	// If batch size > 1, it's assumed that the policy weights are being
	// learned, and so an external VM will be used after an external loss
	// has been added to the policy's graph.
	vm := G.NewTapeMachine(net.Graph())
	policy.vm = vm

	// fmt.Println("Gaussian", len(policy.Graph().AllNodes()))

	return &policy, nil
}

// Mean gets the mean of the policy when last run
func (g *GaussianTreeMLP) Mean() []float64 {
	return g.mean.Data().([]float64)
}

func (g *GaussianTreeMLP) Std() []float64 {
	return g.std.Data().([]float64)
}

// LogProbOf returns a node that computes the log probability of taking
// the argument actions in the argument states when a VM of the policy
// is run. No VM is run.
//
// This function simply sets the inputs to the neural net so that the
// returned node will compute the log probabilities of actions a
// in states s. To actually get these values, an external VM must be run.
func (g *GaussianTreeMLP) LogProbOf(s, a []float64) (*G.Node, error) {
	if expect := (g.Network().BatchSize()) * g.actionDims; len(a) != expect {
		return nil, fmt.Errorf("logProbOf: invalid action size\n\t"+
			"want(%v) \n\thave(%v)", expect, len(a))
	}

	g.Network().SetInput(s)
	actions := tensor.NewDense(tensor.Float64, g.externActions.Shape())
	err := G.Let(g.externActions, actions)
	if err != nil {
		return nil, fmt.Errorf("logProbOf: could not set action input: %v",
			err)
	}

	return g.externActionsLogProb, nil
}

// Actions returns the actions selected by the previous run of the
// policy. If SetInput() is called on the policy's NerualNet, this
// function returns the actions selected in the states that were
// inputted to the neural net. If SelectAction() was last called,
// this function returns the action selected at the last timestep.
//
// Given M actions, this node will be a vector of size M.
func (g *GaussianTreeMLP) Actions() *G.Node {
	return g.actions
}

// LogProb returns the node of the computational graph that computes the
// log probabilities of actions selected in the states inputted to the
// policy's neural network. If SetInput() was called on the policy's
// NeuralNet, this function returns the log probabilities of actions
// selected in the states that were inputted to the neural net. If
// SelectAction() was last called, this function returns the action
// selected at the last timestep.
//
// Given M actions, this node will be a vector of size M.
//
// This function makes use of the reprarameterization trick
// (https://spinningup.openai.com/en/latest/algorithms/sac.html)
// and should be used when taking an expectation - over actions selected
// from the policy - over the log probability of selecting actions.
func (g *GaussianTreeMLP) LogProb() *G.Node {
	return g.logProb
}

// logProb calculates the log probability of each action selected in a
// state
func logProb(mean, std, actions *G.Node) (*G.Node, error) {
	// Error checking
	graph := mean.Graph()
	if graph != std.Graph() || graph != actions.Graph() {
		return nil, fmt.Errorf("logProb: mean, std, and actions should " +
			"all have the same graph")
	}

	// Calculate (2*π)^(-k/2)
	negativeHalf := G.NewConstant(-0.5)
	dims := float64(mean.Shape()[1])
	multiplier := G.NewConstant(math.Pow(math.Pi*2, -dims/2))

	if std.Shape()[1] != 1 {
		// Multi-dimensional actions
		// Calculate det(σ). Since σ is a diagonal matrix stored as a vector,
		// the determinant == prod(diagonal of σ) = prod(σ)
		det := G.Must(G.Slice(std, nil, tensorutils.NewSlice(0, 1, 1)))
		for i := 1; i < std.Shape()[1]; i++ {
			s := G.Must(G.Slice(std, nil, tensorutils.NewSlice(i, i+1, 1)))
			det = G.Must(G.HadamardProd(det, s))
		}
		invDet := G.Must(G.Inverse(det))

		// Calculate (2*π)^(-k/2) * det(σ)
		det = G.Must(G.Pow(det, negativeHalf))
		multiplier = G.Must(G.Mul(multiplier, det))

		// Calculate (-1/2) * (A - μ)^T σ^(-1) (A - μ)
		// Since everything is stored as a vector, this boils down to a
		// bunch of Hadamard products, sums, and differences.
		diff := G.Must(G.Sub(actions, mean))
		exponent := G.Must(G.HadamardProd(diff, invDet))
		exponent = G.Must(G.HadamardProd(exponent, diff))
		exponent = G.Must(G.Sum(exponent, 1))
		exponent = G.Must(G.Mul(exponent, negativeHalf))

		// Calculate the probability
		prob := G.Must(G.Exp(exponent))
		prob = G.Must(G.HadamardProd(multiplier, prob))

		logProb := G.Must(G.Log(prob))

		return logProb, nil
	} else {
		// Single dimensional actions
		// Calculate (2π)^(-k/2) * σ ^(-1) == 1 / (σ √(2π))
		invStd := G.Must(G.Inverse(std))
		multiplier = G.Must(G.Mul(multiplier, invStd))

		// Calculate (-1/2) * ((A - μ) / σ) ^ 2
		exponent := G.Must(G.Sub(actions, mean))
		exponent = G.Must(G.HadamardProd(exponent, invStd))
		exponent = G.Must(G.Pow(exponent, G.NewConstant(2.0)))
		exponent = G.Must(G.HadamardProd(exponent, negativeHalf))

		// Calcualte probability
		prob := G.Must(G.Exp(exponent))
		prob = G.Must(G.HadamardProd(multiplier, prob))

		logProb := G.Must(G.Log(prob))
		logProb = G.Must(G.Ravel(logProb))

		return logProb, nil
	}
}

// Network returns the NeuralNet used by the policy for function
// approximation
func (g *GaussianTreeMLP) Network() network.NeuralNet {
	return g.NeuralNet
}

// CloneWithBatch clones the policy with a new input batch size
func (g *GaussianTreeMLP) CloneWithBatch(batch int) (agent.NNPolicy, error) {
	// Clone the network
	net, err := g.Network().CloneWithBatch(batch)
	if err != nil {
		return &GaussianTreeMLP{}, fmt.Errorf("clonePolicyWithBatch: could "+
			"not clone policy neural net: %v", err)
	}
	// fmt.Println("BEFORE CLONE", len(net.Graph().AllNodes()))

	// Exponentiate the log standard deviation
	logStd := net.Prediction()[0]
	stdNode := G.Must(G.Exp(logStd))
	meanNode := net.Prediction()[1]

	// Reparameterization trick A = μ + σ*ε, where ε ~ N(0, 1)
	actionPerturb := G.GaussianRandomNode(net.Graph(), tensor.Float64,
		0, 1, batch, net.Outputs()[0])
	actionStd := G.Must(G.HadamardProd(stdNode, actionPerturb))
	actions := G.Must(G.Add(meanNode, actionStd))

	// Calculate log probability
	logProbNode, err := logProb(meanNode, stdNode, actions)
	if err != nil {
		return nil, fmt.Errorf("newGaussianTreeMLP: could not calculate "+
			"log probabiltiy: %v", err)
	}

	// Create external actions
	externActions := G.NewMatrix(
		net.Graph(),
		tensor.Float64,
		G.WithName("externActions"),
		G.WithShape(net.BatchSize(), g.actionDims),
	)
	logProbExternalActions, err := logProb(meanNode, stdNode, externActions)
	if err != nil {
		return nil, fmt.Errorf("newGaussianTreMLP: could not calculate "+
			"log probability of external input actions: %v", err)
	}

	policy := GaussianTreeMLP{
		NeuralNet:            net,
		logProb:              logProbNode,
		actions:              actions,
		seed:                 g.seed,
		externActions:        externActions,
		externActionsLogProb: logProbExternalActions,
		actionDims:           g.actionDims,
	}

	// Store the values of the actions selected for the batch, standard
	// deviations and means for the policy in each state in the batch,
	// and the log probability of selecting each action in the batch.
	G.Read(policy.actions, &policy.actionsVal)
	G.Read(stdNode, &policy.std)
	G.Read(meanNode, &policy.mean)
	G.Read(logProbNode, &policy.logProbVal)
	G.Read(logProbExternalActions, &policy.externActionsLogProbVal)

	// Action selection VM is used only for policies with batches of size 1.
	// If batch size > 1, it's assumed that the policy weights are being
	// learned, and so an external VM will be used after an external loss
	// has been added to the policy's graph.
	var vm G.VM
	if batch == 1 {
		vm = G.NewTapeMachine(policy.Graph())
	}
	policy.vm = vm

	// fmt.Println("AFTER CLONE", len(net.Graph().AllNodes()))

	return &policy, nil
}

// Clone clones the policy
func (g *GaussianTreeMLP) Clone() (agent.NNPolicy, error) {
	return g.CloneWithBatch(g.BatchSize())
}

// SelectAction selects and returns a new action given a TimeStep
func (g *GaussianTreeMLP) SelectAction(t timestep.TimeStep) *mat.VecDense {
	if g.BatchSize() != 1 {
		log.Fatal("selectAction: cannot select an action from batch policy")
	}

	g.Network().SetInput(t.Observation.RawVector().Data)
	g.vm.RunAll()

	// fmt.Println("POL", g.Network().Output())
	// fmt.Println("\nACTIONS", g.actionsVal, g.actions.Value())

	// ! THIS DOESN'T WORK SOMETIMES...
	// action := g.actionsVal.Data().([]float64)
	action := g.actions.Value().Data().([]float64)

	fmt.Println("\nAction:", g.actionsVal)
	fmt.Println("STUFF:", g.mean, g.std, g.logProbVal)

	g.vm.Reset()
	return mat.NewVecDense(len(action), action)
}

func (g *GaussianTreeMLP) PrintVals() {
	fmt.Println("\nActions val print for non-cloned...", g.actionsVal)
	fmt.Println()
}
