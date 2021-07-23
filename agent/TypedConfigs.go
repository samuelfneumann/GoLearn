package agent

import (
	"encoding/json"
	"reflect"
)

// Type represents a type of an agent. For example VanillaPG, SAV, DQN
type Type string

// Registered types with the package. Once a Type has been registered
// with this map, a ConfigList with that type can be created.
var registeredTypes map[Type]reflect.Type

func init() {
	registeredTypes = make(map[Type]reflect.Type)
}

// JSONConfigList wraps a ConfigList to enable a ConfigList type to be
// JSON marshaled and unmarshaled into its underlying concrete type
type TypedConfigList struct {
	Type
	ConfigList
}

func NewTypedConfigList(c ConfigList) TypedConfigList {
	return TypedConfigList{Type: c.Type(), ConfigList: c}
}

func Register(agent Type, configs ConfigList) {
	registeredTypes[agent] = reflect.TypeOf(configs)
}

// UnmarshalJSON implements the json.Unmarshaller interface
func (j *TypedConfigList) UnmarshalJSON(data []byte) error {
	configs, typeName, err := unmarshalConfigList(
		data,
		"Type",
		"ConfigList")
	if err != nil {
		return err
	}

	j.Type = typeName
	j.ConfigList = configs

	return nil
}

// unmarshalConfig uses reflection to unmarshall a Config into its
// concrete type. Both the Config and its Type are returned.
func unmarshalConfigList(data []byte, typeJsonField, valueJsonField string) (ConfigList, Type, error) {
	m := map[string]interface{}{}
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, "", err
	}

	typeName := Type(m[typeJsonField].(string))
	var value ConfigList
	if ty, found := registeredTypes[typeName]; found {
		value = reflect.New(ty).Interface().(ConfigList)
	}

	valueBytes, err := json.Marshal(m[valueJsonField])
	if err != nil {
		return nil, "", err
	}

	if err = json.Unmarshal(valueBytes, &value); err != nil {
		return nil, "", err
	}
	concreteValue := reflect.ValueOf(value).Elem().Interface().(ConfigList)

	return concreteValue, typeName, nil
}

func (t *TypedConfigList) At(i int) Config {
	return ConfigAt(i, t.ConfigList)
}
