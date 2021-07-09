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
type GaussianTreeMLP struct {
	network.NeuralNet

	mean, std G.Value

	logProb    *G.Node
	logProbVal G.Value
	actions    *G.Node
	actionsVal G.Value

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

	policy := GaussianTreeMLP{
		NeuralNet: net,
		logProb:   logProbNode,
		actions:   actions,
		seed:      seed,
	}

	// Store the values of the actions selected for the batch, standard
	// deviations and means for the policy in each state in the batch,
	// and the log probability of selecting each action in the batch.
	G.Read(policy.actions, &policy.actionsVal)
	G.Read(stdNode, &policy.std)
	G.Read(meanNode, &policy.mean)
	G.Read(logProbNode, &policy.logProbVal)

	// Action selection VM is used only for policies with batches of size 1.
	// If batch size > 1, it's assumed that the policy weights are being
	// learned, and so an external VM will be used after an external loss
	// has been added to the policy's graph.
	var vm G.VM
	if batchForLogProb == 1 {
		vm = G.NewTapeMachine(net.Graph())
	}
	policy.vm = vm

	return &policy, nil
}

// logProb calculates the log probability of each action
func logProb(mean, std, actions *G.Node) (*G.Node, error) {
	// Error checking
	graph := mean.Graph()
	if graph != std.Graph() || graph != actions.Graph() {
		return nil, fmt.Errorf("logProb: mean, std, and actions should " +
			"all have the same graph")
	}

	// Calculate (2*π)^(-k/2)
	negativeHalf := G.NewConstant(-0.5)
	dims := float64(mean.Shape()[0])
	multiplier := G.NewConstant(math.Pow(math.Pi*2, -dims/2))

	if std.Shape()[0] != 1 {
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
		exponent = G.Must(G.Mul(exponent, invStd))
		exponent = G.Must(G.Pow(exponent, G.NewConstant(2.0)))
		exponent = G.Must(G.Mul(exponent, negativeHalf))

		// Calcualte probability
		prob := G.Must(G.Exp(exponent))
		prob = G.Must(G.Mul(multiplier, prob))

		logProb := G.Must(G.Log(prob))

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
	// CLone the network
	net, err := g.Network().CloneWithBatch(batch)
	if err != nil {
		return &GaussianTreeMLP{}, fmt.Errorf("clonePolicyWithBatch: could "+
			"not clone policy neural net: %v", err)
	}

	// Only if batch size == 1 do we need a VM for action selection
	var vm G.VM
	if batch == 1 {
		vm = G.NewTapeMachine(net.Graph())
	}

	newPolicy := GaussianTreeMLP{
		NeuralNet: net,
		mean:      g.mean,
		std:       g.std,
		seed:      g.seed,
		vm:        vm,
	}

	return &newPolicy, nil
}

// Clone clones the policy
func (g *GaussianTreeMLP) Clone() (agent.NNPolicy, error) {
	return g.CloneWithBatch(g.BatchSize())
}

// LogProb returns the node of the computational graph that computes the
// log probabilities of actions. Given M actions, this node will be a
// vector of size M.
func (g *GaussianTreeMLP) LogProb() *G.Node {
	return g.logProb
}

// SelectAction selects and returns a new action given a TimeStep
func (g *GaussianTreeMLP) SelectAction(t timestep.TimeStep) *mat.VecDense {
	if g.BatchSize() != 1 {
		log.Fatal("selectAction: cannot select an action from batch policy")
	}

	g.Network().SetInput(t.Observation.RawVector().Data)
	g.vm.RunAll()
	defer g.vm.Reset()

	action := g.actionsVal.Data().([]float64)
	return mat.NewVecDense(len(action), action)
}
