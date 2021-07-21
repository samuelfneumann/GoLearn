// Package solver implements functionality to wrap Gorgonia Solvers
// so that they can be JSON serialized into configuraiton files.
package solver

import (
	"encoding/json"
	"fmt"
	"reflect"

	G "gorgonia.org/gorgonia"
)

// Type describes different types of solvers that are available
type Type string

// Available solver types
const (
	Adam    Type = "Adam"
	Vanilla Type = "Vanilla"
)

// Solver wraps Gorgonia Solvers so that they can be JSON marshalled and
// unmarshalled.
type Solver struct {
	G.Solver `json:"-"`
	Type
	Config
}

// newSolver returns a new solver with the given type and configuration.
func newSolver(t Type, c Config) (*Solver, error) {
	if !c.ValidType(t) {
		return nil, fmt.Errorf("newSolver: invalid solver type %v for "+
			"configuration %T", t, c)
	}
	solver := Solver{Type: t, Config: c}
	solver.Solver = solver.Config.Create()

	return &solver, nil
}

// UnmarshalJSON implements the json.Unmarshaller interface
func (s *Solver) UnmarshalJSON(data []byte) error {
	config, typeName, err := unmarshalConfig(
		data,
		"Type",
		"Config",
		map[string]reflect.Type{
			string(Vanilla): reflect.TypeOf(VanillaConfig{}),
			string(Adam):    reflect.TypeOf(AdamConfig{}),
		})
	if err != nil {
		return err
	}

	s.Type = typeName
	s.Config = config
	s.Solver = s.Config.Create()

	return nil
}

// unmarshalConfig uses reflection to unmarshall a Config into its
// concrete type. Both the Config and its Type are returned.
func unmarshalConfig(data []byte, typeJsonField, valueJsonField string,
	customTypes map[string]reflect.Type) (Config, Type, error) {
	m := map[string]interface{}{}
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, "", err
	}

	typeName := m[typeJsonField].(string)
	var value Config
	if ty, found := customTypes[typeName]; found {
		value = reflect.New(ty).Interface().(Config)
	}

	valueBytes, err := json.Marshal(m[valueJsonField])
	if err != nil {
		return nil, "", err
	}

	if err = json.Unmarshal(valueBytes, &value); err != nil {
		return nil, "", err
	}

	return value, Type(typeName), nil
}

// Config implements a Gorgonia Solver configuration and can be used to
// create Gorgonia Solvers they describe.
type Config interface {
	Create() G.Solver

	// ValidType returns whether a specific Solver type can be created
	// with the Config
	ValidType(Type) bool
}
