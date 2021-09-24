// Package initwfn implements functionality to wrap Gorgonia InitWFn
// so that they can be JSON serialized into configuraiton files.
package initwfn

import (
	"encoding/json"
	"fmt"
	"reflect"

	G "gorgonia.org/gorgonia"
)

// Type describes different types of InitWFn that are available.
// Type is used to implement a basic type system of InitWFn's.
type Type string

// Available InitWFn types
const (
	GlorotU Type = "GlorotU"
	GlorotN Type = "GlorotN"
	HeU     Type = "HeU"
	HeN     Type = "HeN"
	Zeroes  Type = "Zeroes"
	Ones    Type = "Ones"
)

// InitWFn wraps Gorgonia InitWFn so that they can be JSON marshalled and
// unmarshalled.
type InitWFn struct {
	initWFn G.InitWFn
	Type
	Config
}

// newInitWFn returns a new InitWFn
func newInitWFn(c Config) (*InitWFn, error) {
	init := InitWFn{Type: c.Type(), Config: c}
	init.initWFn = init.Config.Create()

	return &init, nil
}

// InitWFn returns the wrapped Gorgonia InitWFn
func (w *InitWFn) InitWFn() G.InitWFn {
	return w.initWFn
}

// String implements the fmt.Stringer interface
func (i *InitWFn) String() string {
	return fmt.Sprintf("{%v InitWFn: %v}", i.Type, i.Config)
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
			string(HeU):     reflect.TypeOf(HeUConfig{}),
			string(HeN):     reflect.TypeOf(HeNConfig{}),
			string(Zeroes):  reflect.TypeOf(ZeroesConfig{}),
			string(Ones):    reflect.TypeOf(OnesConfig{}),
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
	concreteValue := reflect.ValueOf(value).Elem().Interface().(Config)

	return concreteValue, Type(typeName), nil
}

// Config implements a Gorgonia InitWFn configuration and can be used to
// create the described Gorgonia InitWFn's.
type Config interface {
	// Create returns the Gorgonia InitWFn that the Config describes
	Create() G.InitWFn

	// Type returns the type of Gorgonia InitWFn that is returned
	Type() Type
}
