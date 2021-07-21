package network

import (
	"encoding/json"
	"fmt"

	G "gorgonia.org/gorgonia"
)

type activationType string

const (
	relu     activationType = "relu"
	identity activationType = "identity"
	tanh     activationType = "tanh"
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
	decoded := activationType(data)
	switch decoded {
	case relu:
		*a = *ReLU()
	case identity:
		*a = *Identity()
	case tanh:
		*a = *TanH()
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
	default:
		return fmt.Errorf("gobdecode: illegal Activation type")
	}
	return nil
}

// Nil returns a nil *Activation
func Nil() *Activation {
	return &Activation{
		activationType: nil_,
		f:              nil,
	}
}

// Identity returns an identity *Activation
func Identity() *Activation {
	return &Activation{
		activationType: identity,
		f: func(x *G.Node) (*G.Node, error) {
			return x, nil
		},
	}
}

// ReLU returns a ReLU *Activation
func ReLU() *Activation {
	return &Activation{
		activationType: relu,
		f:              G.Rectify,
	}
}

// TanH returns a tanh *Activation
func TanH() *Activation {
	return &Activation{
		activationType: tanh,
		f:              G.Tanh,
	}
}
