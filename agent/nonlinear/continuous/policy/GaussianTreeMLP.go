package policy

// // ! SAC e.g. will have two GaussianTreeMLPs, one with and one without
// // ! batches. With batchs --> learning weights. Without batches --> selecting
// // ! actions. If the batch size is 1, then only one policy is required,
// // ! and we can use the LobProb() of this policy without re-running it
// // ! at each timestep to get the log probability. For batches > 1,
// // ! we would first have to sample from replay -> run train policy to
// // ! get log prob. For batches of size 1, the LogProb() will already
// // ! hold the log prob of the last selected action -> no need to run the
// // ! policy again.

// import (
// 	"fmt"
// 	"log"
// 	"math"

// 	"golang.org/x/exp/rand"

// 	"gonum.org/v1/gonum/mat"
// 	"gonum.org/v1/gonum/stat/distmv"
// 	"gonum.org/v1/gonum/stat/samplemv"
// 	G "gorgonia.org/gorgonia"
// 	"gorgonia.org/tensor"
// 	"sfneuman.com/golearn/agent"
// 	"sfneuman.com/golearn/environment"
// 	"sfneuman.com/golearn/network"
// 	"sfneuman.com/golearn/spec"
// 	"sfneuman.com/golearn/timestep"
// 	"sfneuman.com/golearn/utils/floatutils"
// 	"sfneuman.com/golearn/utils/tensorutils"
// )

// const stdOffset = 1e-8

// // var LogProb G.Value
// // var Mean, Std G.Value

// //
// //		π(A|S) ~ N(μ, σ)
// //
// // where the mean μ and standard deviation σ are predicted by the neural
// // net, and π denotes the policy.
// //
// // * Note that this version cannot produce gradients w.r.t. an action as is
// // * required for some methods like SAC.
// //
// // ! So far, the Gaussian policy is not seedable since we use
// // ! gorgonia's GaussianRandomNode().
// //
// // ! This version of a Gaussian policy will take in some state S and
// // ! compute the log probabilities for the actions the policy would
// // ! sample in that state, not the actual actions taken. For example,
// // ! if given a tuple (S, A, R, S'), this the logProb() will compute
// // ! the log probability of A' taken in S' as well as compute each A'.
// // ! This is because this implementation assumes a "SAC-like" update.
// // ! Given (S, A, R, S') from a replay buffer, we use the policy to
// // ! predict A' and logProb(A' | S'). This A' is then used in the
// // ! in the critic to get a value of Q(A', S') to use in the gradient.
// //
// // ! if we want to compute log prob of action A in state S, we will
// // ! need another function which takes batches of states/actions as
// // ! inputs, sets the states as input to the NN, then adds the actions
// // ! to an input node -> calculates the log prob of those actions.
// type GaussianTreeMLP struct {
// 	network.NeuralNet

// 	Mean, Std         G.Value
// 	MeanNode, StdNode *G.Node

// 	// logProbNode *G.Node
// 	// logProbVal  G.Value
// 	// actions     *G.Node // Node of action(s) to take in input state(s)
// 	// actionsVal  G.Value // Value of action(s) to take in input state(s)
// 	actionDims int

// 	// External actions refer to actions that are given to the policy
// 	// with which to calculate the log probability of.
// 	ExternActions           *G.Node
// 	ExternActionsVal        G.Value
// 	ExternActionsLogProb    *G.Node
// 	ExternActionsLogProbVal G.Value

// 	seed   uint64
// 	source rand.Source

// 	vm G.VM // VM for action selection
// }

// // NewGaussianTreeMLP creates and returns a new Gaussian policy, with
// // mean and log standard deviation predicted by a tree MLP.
// //
// // The rootHiddenSizes, rootBiases, and rootActivations parameters
// // determine the architecture of the root MLP of the tree MLP. For index
// // i, rootHiddenSizes[i] determines the number of hidden units in the
// // ith layer; rootBiases[i] determines whether or not a bias unit is
// // added to the ith layer; rootActivations[i] dteremines the activation
// // function of the ith layer. The number of layer in the root network is
// // determined by len(rootHiddenSizes).
// //
// // The number of leaf networks is defined by len(leafHiddenSizes) and
// // must be equal to 2, one to predict μ and the other to predict log(σ).
// // For indices i and j, leafHiddenSizes[i][j], leafBiases[i][j], and
// // leafActivations[i][j] determine the number of hidden units of layer
// // j in leaf network i, whether a bias is added to layer j of leaf
// // network i, and the activation of layer j of leaf network i
// // respectively. The length of leafHiddenSizes[i] determines the number
// // of hidden layers in leaf network i.
// //
// // To each leaf network, a final linear layer is added (with a bias
// // unit and no activations) to ensure that the output shape matches
// // the action dimensionality of the environment. Additionally, the
// // predicted log(σ) is first exponentiated before sampling an action.
// // For more details on the neural network function approximator, see
// // the network package.
// //
// // The batch size is determined by batchForLogProb. If this value is
// // set above 1, then it is assumed that the policy will not be used
// // to select actions at each timestep. Instead, it is assumed that
// // the policy will be used to produce the log(π(A|S)) required for
// // learning the policy, and that this will be produced for all states
// // in a batch. In this case, the method LogProb() will return the node
// // that will hold the log probabilities when the computational graph
// // is run. If the batch size is set to 1, then actions can be selected
// // at each timestep, and the node returned by LogProb() will hold the
// // log probability of selecting this action or any other action
// // selected by the policy when its neural networks is given input and
// // the computational graph run.
// func NewGaussianTreeMLP(env environment.Environment, batchForLogProb int,
// 	g *G.ExprGraph, rootHiddenSizes []int, rootBiases []bool,
// 	rootActivations []*network.Activation, leafHiddenSizes [][]int,
// 	leafBiases [][]bool, leafActivations [][]*network.Activation,
// 	init G.InitWFn, seed uint64) (agent.LogPDFer, error) {

// 	// Error checking
// 	if env.ActionSpec().Cardinality == spec.Discrete {
// 		err := fmt.Errorf("newGaussianTreeMLP: gaussian policy cannot be " +
// 			"used with discrete actions")
// 		return &GaussianTreeMLP{}, err
// 	}

// 	if len(leafHiddenSizes) != 2 {
// 		return &GaussianTreeMLP{}, fmt.Errorf("newGaussianTreeMLP: gaussian " +
// 			"policy requires 2 leaf networks only")
// 	}

// 	features := env.ObservationSpec().Shape.Len()
// 	actionDims := env.ActionSpec().Shape.Len()

// 	// Create the tree MLP, which may predict batches of means and
// 	// log standard deviations -> one for each state in the batch
// 	net, err := network.NewTreeMLP(features, batchForLogProb, actionDims, g,
// 		rootHiddenSizes, rootBiases, rootActivations, leafHiddenSizes,
// 		leafBiases, leafActivations, init)
// 	if err != nil {
// 		return &GaussianTreeMLP{}, fmt.Errorf("newGaussianTreeMLP: could "+
// 			"not create policy network: %v", err)
// 	}

// 	// Exponentiate the log standard deviation
// 	logStd := net.Prediction()[0]

// 	offset := G.NewConstant(stdOffset)
// 	stdNode := G.Must(G.Exp(logStd))
// 	stdNode = G.Must(G.Add(stdNode, offset))

// 	meanNode := net.Prediction()[1]

// 	source := rand.NewSource(seed)

// 	p := GaussianTreeMLP{
// 		NeuralNet: net,
// 		// logProbNode: logProbNode,
// 		// actions:    actions,
// 		seed:       seed,
// 		source:     source,
// 		actionDims: actionDims,
// 		MeanNode:   meanNode,
// 		StdNode:    stdNode,
// 	}

// 	// Store the values of the actions selected for the batch, standard
// 	// deviations and means for the policy in each state in the batch,
// 	// and the log probability of selecting each action in the batch.
// 	// G.Read(p.actions, &p.actionsVal)
// 	G.Read(meanNode, &p.Mean)
// 	G.Read(stdNode, &p.Std)

// 	// G.Read(logProbNode, &p.logProbVal)

// 	if net.BatchSize() > 1 {
// 		// ? Create external actions (not sure why but the
// 		// ? G.Read(p.actions, &p.actionsVal)) line won't work if this is
// 		// ? included, so when making train/behaviour policies, make sure
// 		// ? not to clone but to create brand new ones
// 		ExternActions := G.NewMatrix(
// 			net.Graph(),
// 			tensor.Float64,
// 			G.WithName("ExternActions"),
// 			G.WithShape(net.BatchSize(), actionDims),
// 		)
// 		logProbExternalActions, err := logProb(meanNode, stdNode, ExternActions)
// 		if err != nil {
// 			return nil, fmt.Errorf("newGaussianTreMLP: could not calculate "+
// 				"log probability of external input actions: %v", err)
// 		}
// 		p.ExternActions = ExternActions
// 		p.ExternActionsLogProb = logProbExternalActions
// 		G.Read(logProbExternalActions, &p.ExternActionsLogProbVal)
// 		G.Read(ExternActions, &p.ExternActionsVal)
// 	}

// 	// Action selection VM is used only for policies with batches of size 1.
// 	// If batch size > 1, it's assumed that the policy weights are being
// 	// learned, and so an external VM will be used after an external loss
// 	// has been added to the policy's graph.
// 	var vm G.VM
// 	if net.BatchSize() == 1 {
// 		vm = G.NewTapeMachine(net.Graph())
// 	}
// 	p.vm = vm

// 	return &p, nil
// }

// // // Mean gets the mean of the policy when last run
// // func (g *GaussianTreeMLP) Mean() []float64 {
// // 	return g.mean.Data().([]float64)
// // }

// // func (g *GaussianTreeMLP) Std() []float64 {
// // 	return g.std.Data().([]float64)
// // }

// // LogProbOf returns a node that computes the log probability of taking
// // the argument actions in the argument states when a VM of the policy
// // is run. No VM is run.
// //
// // This function simply sets the inputs to the neural net so that the
// // returned node will compute the log probabilities of actions a
// // in states s. To actually get these values, an external VM must be run.
// func (g *GaussianTreeMLP) LogProbOf(s, a []float64) (*G.Node, error) {
// 	if expect := (g.Network().BatchSize()) * g.actionDims; len(a) != expect {
// 		return nil, fmt.Errorf("logProbOf: invalid action size\n\t"+
// 			"want(%v) \n\thave(%v)", expect, len(a))
// 	}

// 	g.Network().SetInput(s)
// 	actions := tensor.NewDense(
// 		tensor.Float64,
// 		g.ExternActions.Shape(),
// 		tensor.WithBacking(a),
// 	)
// 	err := G.Let(g.ExternActions, actions)
// 	if err != nil {
// 		return nil, fmt.Errorf("logProbOf: could not set action input: %v",
// 			err)
// 	}

// 	return g.ExternActionsLogProb, nil
// }

// func (g *GaussianTreeMLP) LogProbNode() *G.Node {
// 	return g.ExternActionsLogProb
// }

// // // Actions returns the actions selected by the previous run of the
// // // policy. If SetInput() is called on the policy's NerualNet, this
// // // function returns the actions selected in the states that were
// // // inputted to the neural net. If SelectAction() was last called,
// // // this function returns the action selected at the last timestep.
// // //
// // // Given M actions, this node will be a vector of size M.
// // func (g *GaussianTreeMLP) Actions() *G.Node {
// // 	return g.actions
// // }

// // // LogProb returns the node of the computational graph that computes the
// // // log probabilities of actions selected in the states inputted to the
// // // policy's neural network. If SetInput() was called on the policy's
// // // NeuralNet, this function returns the log probabilities of actions
// // // selected in the states that were inputted to the neural net. If
// // // SelectAction() was last called, this function returns the action
// // // selected at the last timestep.
// // //
// // // Given M actions, this node will be a vector of size M.
// // //
// // // This function makes use of the reprarameterization trick
// // // (https://spinningup.openai.com/en/latest/algorithms/sac.html)
// // // and should be used when taking an expectation - over actions selected
// // // from the policy - over the log probability of selecting actions.
// // func (g *GaussianTreeMLP) LogProb() *G.Node {
// // 	return g.logProbNode
// // }

// // logProb calculates the log probability of each action selected in a
// // state
// //
// // ! this can be heavily optimized
// func logProb(mean, std, actions *G.Node) (*G.Node, error) {
// 	// Error checking
// 	graph := mean.Graph()
// 	if graph != std.Graph() || graph != actions.Graph() {
// 		return nil, fmt.Errorf("logProb: mean, std, and actions should " +
// 			"all have the same graph")
// 	}

// 	// Calculate (2*π)^(-k/2)
// 	negativeHalf := G.NewConstant(-0.5)
// 	dims := float64(mean.Shape()[1])
// 	multiplier := G.NewConstant(math.Pow(math.Pi*2, -dims/2), G.WithName("multiplier"))

// 	if std.Shape()[1] != 1 {
// 		// Multi-dimensional actions
// 		// Calculate det(σ). Since σ is a diagonal matrix stored as a vector,
// 		// the determinant == prod(diagonal of σ) = prod(σ)
// 		det := G.Must(G.Slice(std, nil, tensorutils.NewSlice(0, 1, 1)))
// 		for i := 1; i < std.Shape()[1]; i++ {
// 			s := G.Must(G.Slice(std, nil, tensorutils.NewSlice(i, i+1, 1)))
// 			det = G.Must(G.HadamardProd(det, s))
// 		}
// 		invDet := G.Must(G.Inverse(det))

// 		// Calculate (2*π)^(-k/2) * det(σ)
// 		det = G.Must(G.Pow(det, negativeHalf))
// 		multiplier = G.Must(G.Mul(multiplier, det))

// 		// Calculate (-1/2) * (A - μ)^T σ^(-1) (A - μ)
// 		// Since everything is stored as a vector, this boils down to a
// 		// bunch of Hadamard products, sums, and differences.
// 		diff := G.Must(G.Sub(actions, mean))
// 		exponent := G.Must(G.HadamardProd(diff, invDet))
// 		exponent = G.Must(G.HadamardProd(exponent, diff))
// 		exponent = G.Must(G.Sum(exponent, 1))
// 		exponent = G.Must(G.Mul(exponent, negativeHalf))

// 		// Calculate the probability
// 		prob := G.Must(G.Exp(exponent))
// 		prob = G.Must(G.HadamardProd(multiplier, prob))

// 		logProb := G.Must(G.Log(prob))

// 		return logProb, nil
// 	} else {
// 		two := G.NewConstant(2.0)
// 		exponent := G.Must(G.Sub(actions, mean))
// 		exponent = G.Must(G.HadamardDiv(exponent, std))
// 		exponent = G.Must(G.Pow(exponent, two))
// 		exponent = G.Must(G.HadamardProd(negativeHalf, exponent))

// 		term2 := G.Must(G.Log(std))
// 		// term2 := G.Must(G.HadamardProd(two, logStd))
// 		term3 := G.NewConstant(math.Log(math.Pow(2*math.Pi, 0.5)))

// 		terms := G.Must(G.Add(term2, term3))
// 		logProb := G.Must(G.Sub(exponent, terms))
// 		logProb = G.Must(G.Ravel(logProb))

// 		return logProb, nil
// 	}
// }

// // Network returns the NeuralNet used by the policy for function
// // approximation
// func (g *GaussianTreeMLP) Network() network.NeuralNet {
// 	return g.NeuralNet
// }

// // CloneWithBatch clones the policy with a new input batch size
// func (g *GaussianTreeMLP) CloneWithBatch(batch int) (agent.NNPolicy, error) {
// 	if batch == 1 && g.BatchSize() != 1 || batch != 1 && g.BatchSize() == 1 {
// 		return nil, fmt.Errorf("cloneWithBatch: due to Gorgonia bugs " +
// 			"GaussianTreeMLPs cannot be cloned from non-batch to batch")
// 	}

// 	// Clone the network
// 	net, err := g.Network().CloneWithBatch(batch)
// 	if err != nil {
// 		return &GaussianTreeMLP{}, fmt.Errorf("clonePolicyWithBatch: could "+
// 			"not clone policy neural net: %v", err)
// 	}
// 	// fmt.Println("BEFORE CLONE", len(net.Graph().AllNodes()))

// 	// Exponentiate the log standard deviation
// 	logStd := net.Prediction()[0]
// 	stdNode := G.Must(G.Exp(logStd))
// 	meanNode := net.Prediction()[1]

// 	// // Reparameterization trick A = μ + σ*ε, where ε ~ N(0, 1)
// 	// actionPerturb := G.GaussianRandomNode(net.Graph(), tensor.Float64,
// 	// 	0, 1, batch, net.Outputs()[0])
// 	// actionStd := G.Must(G.HadamardProd(stdNode, actionPerturb))
// 	// actions := G.Must(G.Add(meanNode, actionStd))

// 	// // Calculate log probability
// 	// logProbNode, err := logProb(meanNode, stdNode, actions)
// 	// if err != nil {
// 	// 	return nil, fmt.Errorf("newGaussianTreeMLP: could not calculate "+
// 	// 		"log probabiltiy: %v", err)
// 	// }

// 	// Create external actions
// 	ExternActions := G.NewMatrix(
// 		net.Graph(),
// 		tensor.Float64,
// 		G.WithName("ExternActions"),
// 		G.WithShape(net.BatchSize(), g.actionDims),
// 	)
// 	logProbExternalActions, err := logProb(meanNode, stdNode, ExternActions)
// 	if err != nil {
// 		return nil, fmt.Errorf("newGaussianTreMLP: could not calculate "+
// 			"log probability of external input actions: %v", err)
// 	}

// 	policy := GaussianTreeMLP{
// 		NeuralNet: net,
// 		// logProbNode:          logProbNode,
// 		// actions:              actions,
// 		seed:                 g.seed,
// 		ExternActions:        ExternActions,
// 		ExternActionsLogProb: logProbExternalActions,
// 		actionDims:           g.actionDims,
// 	}

// 	// Store the values of the actions selected for the batch, standard
// 	// deviations and means for the policy in each state in the batch,
// 	// and the log probability of selecting each action in the batch.
// 	// G.Read(policy.actions, &policy.actionsVal)
// 	G.Read(stdNode, &policy.Std)
// 	G.Read(meanNode, &policy.Mean)
// 	// G.Read(logProbNode, &policy.logProbVal)
// 	G.Read(logProbExternalActions, &policy.ExternActionsLogProbVal)

// 	// Action selection VM is used only for policies with batches of size 1.
// 	// If batch size > 1, it's assumed that the policy weights are being
// 	// learned, and so an external VM will be used after an external loss
// 	// has been added to the policy's graph.
// 	var vm G.VM
// 	if batch == 1 {
// 		vm = G.NewTapeMachine(policy.Graph())
// 	}
// 	policy.vm = vm

// 	// fmt.Println("AFTER CLONE", len(net.Graph().AllNodes()))

// 	return &policy, nil
// }

// // Clone clones the policy
// func (g *GaussianTreeMLP) Clone() (agent.NNPolicy, error) {
// 	return g.CloneWithBatch(g.BatchSize())
// }

// // SelectAction selects and returns a new action given a TimeStep
// func (g *GaussianTreeMLP) SelectAction(t timestep.TimeStep) *mat.VecDense {
// 	if g.BatchSize() != 1 {
// 		log.Fatal("selectAction: cannot select an action from batch policy")
// 	}

// 	obs := t.Observation.RawVector().Data
// 	g.Network().SetInput(obs)
// 	g.vm.RunAll()
// 	mean := g.Mean.Data().([]float64)
// 	mean = floatutils.ClipSlice(mean, -100, 100)
// 	std := g.Std.Data().([]float64)
// 	std = floatutils.ClipSlice(std, 0.0001, 1000.0)
// 	g.vm.Reset()

// 	stdDev := mat.NewDiagDense(len(std), std)
// 	pol, ok := distmv.NewNormal(mean, stdDev, g.source)
// 	if !ok {
// 		panic("selectAction: stdDev not P.S.D.")
// 	}

// 	sampler := samplemv.IID{Dist: pol}
// 	actions := (mat.NewDense(len(mean), 1, nil))
// 	sampler.Sample(actions)

// 	// fmt.Println(actions.At(0, 0))

// 	// fmt.Println("POL", g.Network().Output())
// 	// fmt.Println("\nACTIONS", g.actionsVal, g.actions.Value())

// 	// fmt.Println("\nAction:", actions)
// 	// fmt.Println("STUFF:", g.mean, g.std, g.ExternActionsLogProb)
// 	// fmt.Println("=== === === === Mean +/- StdDev:", g.Mean(), g.Std())

// 	return mat.NewVecDense(len(mean), actions.RawMatrix().Data)
// }

// func (g *GaussianTreeMLP) PrintVals() {
// 	// fmt.Println("Actions has NaN:", floats.HasNaN(g.ExternActionsVal.Data().([]float64)))
// 	// fmt.Println("Log prob NaN:", floats.HasNaN(g.ExternActionsLogProbVal.Data().([]float64)))
// 	// fmt.Println("Sum", floats.Sum(g.ExternActionsLogProbVal.Data().([]float64)))
// 	// fmt.Println("\tStd max:", floats.Max(g.Std()))
// 	// fmt.Println("\tStd min:", floats.Min(g.Std()))
// 	// fmt.Println("\tMean max:", floats.Max(g.Mean()))
// 	// fmt.Println("\tMean min:", floats.Min(g.Mean()))
// 	// fmt.Println("Lob prob max:", floats.Max(g.ExternActionsLogProbVal.Data().([]float64)))
// 	// fmt.Println("Lob prob min:", floats.Min(g.ExternActionsLogProbVal.Data().([]float64)))
// 	// fmt.Println(LogProb)
// 	// fmt.Println(floatutils.Unique(LogProb.Data().([]float64)...))
// 	// fmt.Println(floatutils.Unique(g.Mean()...))
// 	// fmt.Println(floatutils.Unique(g.Std()...))
// 	// // fmt.Println(g.ExternActionsVal.Data().([]float64))
// 	// fmt.Println(g.ExternActionsVal)
// 	// fmt.Println(g.Std())
// 	// fmt.Println(g.Mean())
// 	// log.Fatal()
// 	// ioutil.WriteFile("net.dot", []byte(g.Network().Graph().ToDot()), 0644)
// 	// log.Fatal()
// }
