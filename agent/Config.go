package agent

import (
	"reflect"

	"github.com/samuelfneumann/golearn/environment"
)

// ConfigList represents a list of Config structs
type ConfigList interface {
	// NumFields returns the number of settable fields per Config
	NumFields() int

	// Len returns the number of Config's in the list
	Len() int

	// Config returns an empty Config of the same type as that stored
	// by the list
	Config() Config

	// Type returns the type of agent which can be constructed from
	// the Config's stored in the ConfigList. This is the same Type
	// returned by the Type() method of the Config's stored in the list.
	Type() Type
}

// ConfigAt returns the Config at index i % configs.Len() in the
// ConfigList.
func ConfigAt(i int, configs ConfigList) Config {
	return configAt(i, configs)
}

// configAt returns the configuration at index i %configs.Len() in the
// ConfigList.
//
// This funciton is private so that reflection is not exposed in the
// public API.
func configAt(i int, configs ConfigList) Config {
	config := configs.Config()
	reflectConfigList := reflect.ValueOf(configs)
	reflectConfig := reflect.New(reflect.Indirect(
		reflect.ValueOf(config)).Type()).Elem() // Settable config

	accum := 1
	for field := 0; field < configs.NumFields(); field++ {
		fieldName := reflectConfigList.Type().Field(field).Name
		settings := reflectConfigList.FieldByName(fieldName)

		switch settings.Kind() {
		case reflect.String:
			reflectConfig.FieldByName(fieldName).Set(settings)

		case reflect.Float64, reflect.Float32, reflect.Int, reflect.Int8,
			reflect.Int16, reflect.Int32, reflect.Int64, reflect.Uint,
			reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64,
			reflect.Complex64, reflect.Complex128:
			reflectConfig.FieldByName(fieldName).Set(settings)

		case reflect.Slice:
			numSettings := settings.Len()
			reflectConfig.FieldByName(fieldName).Set(settings.Index(((i /
				accum) % numSettings)))
			accum *= numSettings

		default:
			panic("configAt: illegal field type")
		}
	}

	return reflectConfig.Interface().(Config)
}

// Config represents a configuration for creating an agent
type Config interface {
	// CreateAgent creates the agent that the config describes
	CreateAgent(env environment.Environment, seed uint64) (Agent, error)

	// ValidAgent returns whether the argument Agent is valid for the
	// Config. This differs from the Type() method in that an actual
	// Agent struct is used here, whereas the Type() method returns
	// a Type (string) describing the Agent's type. For example a
	// Config's Type may be "CategoricalVanillaPG-MLP" or "Gaussian
	// VanillaPG-TreeMLP", which are different Config types, but each of
	// which have *vanillapg.VanillaPG as a valid Agent since that
	// is the Agent the Configs describe.
	ValidAgent(Agent) bool

	// Validate returns an error describing whether or not the
	// configuration is valid.
	Validate() error

	// Type returns the type of agent which can be constructed from
	// the Config.
	Type() Type
}
