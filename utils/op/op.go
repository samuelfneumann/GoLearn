// PackaG. op provides extended GorG.nia G.aph operations.
//
// Adapted from aunum/G.ld on GitHub
package op

import (
	"fmt"
	"math"
	"os"

	G "gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"sfneuman.com/golearn/utils/tensorutils"
)

// Clip clips the value of a node
func Clip(value *G.Node, min, max float64) (retVal *G.Node, err error) {
	// Construct clipping nodes
	var minNode, maxNode *G.Node
	switch value.Dtype() {
	case G.Float32:
		minNode = G.NewScalar(
			value.Graph(),
			G.Float32,
			G.WithValue(float32(min)),
			G.WithName("clip_min"),
		)
		maxNode = G.NewScalar(
			value.Graph(),
			G.Float32,
			G.WithValue(float32(max)),
			G.WithName("clip_max"),
		)
	case G.Float64:
		minNode = G.NewScalar(
			value.Graph(),
			G.Float64,
			G.WithValue(min),
			G.WithName("clip_min"),
		)
		maxNode = G.NewScalar(
			value.Graph(),
			G.Float64,
			G.WithValue(max),
			G.WithName("clip_max"),
		)
	}

	// Check if its the min value
	minMask, err := G.Lt(value, minNode, true)
	if err != nil {
		return nil, err
	}
	minVal, err := G.HadamardProd(minNode, minMask)
	if err != nil {
		return nil, err
	}

	// Check if its the given value
	isMaskGt, err := G.Gt(value, minNode, true)
	if err != nil {
		return nil, err
	}
	isMaskLt, err := G.Lt(value, maxNode, true)
	if err != nil {
		return nil, err
	}
	isMask, err := G.HadamardProd(isMaskGt, isMaskLt)
	if err != nil {
		return nil, err
	}
	isVal, err := G.HadamardProd(value, isMask)
	if err != nil {
		return nil, err
	}

	// Check if its the max value
	maxMask, err := G.Gt(value, maxNode, true)
	if err != nil {
		return nil, err
	}
	maxVal, err := G.HadamardProd(maxNode, maxMask)
	if err != nil {
		return nil, err
	}
	return G.ReduceAdd(G.Nodes{minVal, isVal, maxVal})
}

// Min returns the min value between the nodes. If values are equal
// the first value is returned
func Min(a *G.Node, b *G.Node) (retVal *G.Node, err error) {
	aMask, err := G.Lte(a, b, true)
	if err != nil {
		return nil, err
	}
	aVal, err := G.HadamardProd(a, aMask)
	if err != nil {
		return nil, err
	}

	bMask, err := G.Lt(b, a, true)
	if err != nil {
		return nil, err
	}
	bVal, err := G.HadamardProd(b, bMask)
	if err != nil {
		return nil, err
	}
	return G.Add(aVal, bVal)
}

// Max value between the nodes. If values are equal the first value is returned.
func Max(a *G.Node, b *G.Node) (retVal *G.Node, err error) {
	aMask, err := G.Gte(a, b, true)
	if err != nil {
		return nil, err
	}
	aVal, err := G.HadamardProd(a, aMask)
	if err != nil {
		return nil, err
	}

	bMask, err := G.Gt(b, a, true)
	if err != nil {
		return nil, err
	}
	bVal, err := G.HadamardProd(b, bMask)
	if err != nil {
		return nil, err
	}
	return G.Add(aVal, bVal)
}

// AddFauxF32 adds a the faux zero value 1e-6.
func AddFauxF32(n *G.Node) (retVal *G.Node, err error) {
	faux := G.NewScalar(n.Graph(), G.Float32, G.WithValue(float32(1e-6)))
	return G.BroadcastAdd(faux, n, []byte{}, []byte{})
}

// LogSumExp calculates the log of the summation of exponentials of
// all logits along the given axis.
func LogSumExp(logits *G.Node, along int) *G.Node {
	max := G.Must(G.Max(logits, along))

	exponent := G.Must(G.BroadcastSub(logits, max, nil, []byte{1}))
	exponent = G.Must(G.Exp(exponent))

	sum := G.Must(G.Sum(exponent, along))
	log := G.Must(G.Log(sum))

	return G.Must(G.Add(max, log))
}

// Prod calculates the product of a Node along an axis
func Prod(input *G.Node, along int) *G.Node {
	shape := input.Shape()

	// Calculate the first columns along the axis along
	dims := make([]tensor.Slice, len(shape))
	for i := 0; i < len(shape); i++ {
		if i == along {
			dims[i] = tensorutils.NewSlice(0, 1, 1)
		}
	}
	prod := G.Must(G.Slice(input, dims...))

	for i := 1; i < input.Shape()[along]; i++ {

		// Calculate the column that should be multiplied next
		for j := 0; j < len(shape); j++ {
			if j == along {
				dims[j] = tensorutils.NewSlice(i, i+1, 1)
			}
		}

		s := G.Must(G.Slice(input, dims...))
		prod = G.Must(G.HadamardProd(prod, s))
	}
	return prod
}

// GaussianLogPdf calculate the log of the probability density function
// of actions drawn from a Gaussian distribution with mean mean and
// standard deviation std.
func GaussianLogPdf(mean, std, actions *G.Node) *G.Node {
	graph := mean.Graph()
	if graph != std.Graph() || graph != actions.Graph() {
		panic("logPdf: all nodes must share the same graph")
	}

	negativeHalf := G.NewConstant(-0.5)

	if std.Shape()[1] != 1 {
		fmt.Fprintf(os.Stderr, "GuassianLogProb: warning - not tested for "+
			"multi-dimensional actions")
		// Multi-dimensional actions
		// Calculate det(σ). Since σ is a diagonal matrix stored as a vector,
		// the determinant == prod(diagonal of σ) = prod(σ)
		dims := float64(mean.Shape()[1])
		multiplier := G.NewConstant(math.Pow(math.Pi*2, -dims/2), G.WithName("multiplier"))
		// det := G.Must(G.Slice(std, nil, tensorutils.NewSlice(0, 1, 1)))
		// for i := 1; i < std.Shape()[1]; i++ {
		// 	s := G.Must(G.Slice(std, nil, tensorutils.NewSlice(i, i+1, 1)))
		// 	det = G.Must(G.HadamardProd(det, s))
		// }
		det := Prod(std, 1)
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

		return logProb
	} else {
		// Single-dimensional actions
		two := G.NewConstant(2.0)
		exponent := G.Must(G.Sub(actions, mean))
		exponent = G.Must(G.HadamardDiv(exponent, std))
		exponent = G.Must(G.Pow(exponent, two))
		exponent = G.Must(G.HadamardProd(negativeHalf, exponent))

		term2 := G.Must(G.Log(std))
		// term2 := G.Must(G.HadamardProd(two, logStd))
		term3 := G.NewConstant(math.Log(math.Pow(2*math.Pi, 0.5)))

		terms := G.Must(G.Add(term2, term3))
		logProb := G.Must(G.Sub(exponent, terms))
		logProb = G.Must(G.Ravel(logProb))

		return logProb
	}
}