package network

import (
	"encoding/json"
	"fmt"
	"strings"

	G "gorgonia.org/gorgonia"
)

type activationType string

const (
	relu     activationType = "relu"
	softplus activationType = "softplu"
	identity activationType = "identity"
	tanh     activationType = "tanh"
	log1p    activationType = "log1p"
	sigmoid  activationType = "sigmoid"
	sin      activationType = "sin"
	cos      activationType = "cos"
	sqrt     activationType = "sqrt"
	mish     activationType = "mish"
	nil_     activationType = "nil"
)

// Activation represents an activation function type
type Activation struct {
	activationType
	f func(x *G.Node) (*G.Node, error)
}

// Fwd performs the forward pass of an Activation
func (a *Activation) fwd(x *G.Node) (*G.Node, error) {
	return a.f(x)
}

// String implements the Stringer interface
func (a *Activation) String() string {
	return string(a.activationType)
}

// IsIdentity returns whether or not the Activation is the identity
// function.
func (a *Activation) IsIdentity() bool {
	return a.activationType == identity
}

// IsNil returns whether an activation is nil
func (a *Activation) IsNil() bool {
	return a.activationType == nil_
}

// MarshalJSON implements the json.Marshaler interface
func (a *Activation) MarshalJSON() ([]byte, error) {
	return json.Marshal(a.activationType)
}

// UnmarshalJSON implements the json.Unmarshaler interface
func (a *Activation) UnmarshalJSON(data []byte) error {
	stringData := strings.Trim(string(data), "\"")
	decoded := activationType(stringData)
	switch decoded {
	case relu:
		*a = *ReLU()

	case identity:
		*a = *Identity()

	case tanh:
		*a = *TanH()

	case sigmoid:
		*a = *Sigmoid()

	case sin:
		*a = *Sin()

	case cos:
		*a = *Cos()

	case sqrt:
		*a = *Sqrt()

	case mish:
		*a = *Mish()

	case log1p:
		*a = *Log1p()

	case softplus:
		*a = *Softplus()

	default:
		return fmt.Errorf("unmarshalJSON: illegal Activation type")
	}
	return nil
}

// GobEncode implements the gob.GobEncoder interface
func (a *Activation) GobEncode() ([]byte, error) {
	return []byte(a.activationType), nil
}

// GobDecode implements the gob.GobDecoder interface
func (a *Activation) GobDecode(encoded []byte) error {
	decoded := activationType(encoded)
	switch decoded {
	case relu:
		*a = *ReLU()

	case identity:
		*a = *Identity()

	case tanh:
		*a = *TanH()

	case sigmoid:
		*a = *Sigmoid()

	case sin:
		*a = *Sin()

	case cos:
		*a = *Cos()

	case sqrt:
		*a = *Sqrt()

	case mish:
		*a = *Mish()

	case log1p:
		*a = *Log1p()

	case softplus:
		*a = *Softplus()

	default:
		return fmt.Errorf("unmarshalJSON: illegal Activation type")
	}
	return nil
}

// Nil returns a nil activation
func Nil() *Activation {
	return &Activation{
		activationType: nil_,
		f:              nil,
	}
}

// Identity returns an identity activation
func Identity() *Activation {
	return &Activation{
		activationType: identity,
		f: func(x *G.Node) (*G.Node, error) {
			return x, nil
		},
	}
}

// ReLU returns a rectified linear unit activation
func ReLU() *Activation {
	return &Activation{
		activationType: relu,
		f:              G.Rectify,
	}
}

// TanH returns a hyperbolic tanget activation
func TanH() *Activation {
	return &Activation{
		activationType: tanh,
		f:              G.Tanh,
	}
}

// Sigmoid returns a sigmoid activation
func Sigmoid() *Activation {
	return &Activation{
		activationType: sigmoid,
		f:              G.Sigmoid,
	}
}

// Sin returns a sine activation
func Sin() *Activation {
	return &Activation{
		activationType: sin,
		f:              G.Sin,
	}
}

// Cos returns a cosine activation
func Cos() *Activation {
	return &Activation{
		activationType: cos,
		f:              G.Cos,
	}
}

// Sqrt returns a sqare root activation. Nodes are first passed through
// an absolute value activation before taking the square root.
func Sqrt() *Activation {
	sqrtFn := func(n *G.Node) (*G.Node, error) {
		var err error

		if n, err = G.Abs(n); err != nil {
			return nil, fmt.Errorf("sqrt: could not compute absolute "+
				"value %v", err)
		}
		return G.Sqrt(n)
	}

	return &Activation{
		activationType: sqrt,
		f:              sqrtFn,
	}
}

// Mish returns a mish activation
func Mish() *Activation {
	return &Activation{
		activationType: mish,
		f:              G.Mish,
	}
}

// Log1p returns a log(|x| + 1) activation. The input node is first
// passed through an absolute value activation, then a 1 is added to
// each element of the input node. Finally the log of the result is
// taken.
func Log1p() *Activation {
	logFn := func(n *G.Node) (*G.Node, error) {
		var err error

		if n, err = G.Abs(n); err != nil {
			return nil, fmt.Errorf("log1p: could not compute absolute "+
				"value: %v", err)
		}
		return G.Log1p(n)
	}

	return &Activation{
		activationType: log1p,
		f:              logFn,
	}
}

// Softplus returns a softplus activation
func Softplus() *Activation {
	return &Activation{
		activationType: softplus,
		f:              G.Softplus,
	}
}
