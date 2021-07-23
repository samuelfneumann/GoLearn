package agent

import (
	"encoding/json"
	"reflect"
)

// TypedConfigList implements functionality for typing a ConfigList.
// In this way, a ConfigList can explicitly have its type stored so
// that when deserializing the ConfigList, we can deserialize it into
// its concrete type without knowing beforehand or declaring beforehand
// a variable of its concrete type.
type TypedConfigList struct {
	Type
	ConfigList
}

// NewTypedConfigList types the argument ConfigList and returns it
// as a TypedConfigList which explicitly holds its Type.
func NewTypedConfigList(c ConfigList) TypedConfigList {
	return TypedConfigList{Type: c.Type(), ConfigList: c}
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

// At returns the Config at index i in the TypedConfigList
func (t *TypedConfigList) At(i int) Config {
	return ConfigAt(i, t.ConfigList)
}
