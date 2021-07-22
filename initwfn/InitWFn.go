// Package initwfn implements functionality to wrap Gorgonia InitWFn
// so that they can be JSON serialized into configuraiton files.
package initwfn

import (
	"encoding/json"
	"fmt"
	"reflect"

	G "gorgonia.org/gorgonia"
)

// Type describes different types of InitWFn that are available
type Type string

// Available solver types
const (
	GlorotU Type = "GlorotU"
	GlorotN Type = "GlorotN"
	Zeroes  Type = "Zeroes"
)

// InitWFn wraps Gorgonia InitWFn so that they can be JSON marshalled and
// unmarshalled.
type InitWFn struct {
	initWFn G.InitWFn
	Type
	Config
}

func (w *InitWFn) InitWFn() G.InitWFn {
	return w.initWFn
}

func newInitWFn(t Type, c Config) (*InitWFn, error) {
	if !c.ValidType(t) {
		return nil, fmt.Errorf("newSolver: invalid InitWFn type %v for "+
			"configuration %T", t, c)
	}
	init := InitWFn{Type: t, Config: c}
	init.initWFn = init.Config.Create()

	return &init, nil
}

// UnmarshalJSON implements the json.Unmarshaller interface
func (i *InitWFn) UnmarshalJSON(data []byte) error {
	config, typeName, err := unmarshalConfig(
		data,
		"Type",
		"Config",
		map[string]reflect.Type{
			string(GlorotU): reflect.TypeOf(GlorotUConfig{}),
			string(GlorotN): reflect.TypeOf(GlorotNConfig{}),
		})
	if err != nil {
		return err
	}

	i.Type = typeName
	i.Config = config
	i.initWFn = i.Config.Create()

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
	Create() G.InitWFn

	// ValidType returns whether a specific Solver type can be created
	// with the Config
	ValidType(Type) bool
}
