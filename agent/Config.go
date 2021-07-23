package agent

import (
	"reflect"

	"sfneuman.com/golearn/environment"
)

// ConfigList represents a list of Config structs
type ConfigList interface {
	NumFields() int // Number of settable fields
	Len() int       // Number of settings
	Config() Config // Returns empty config of type in list
	Type() Type     // Returns the type of the configuration
}

func ConfigAt(i int, configs ConfigList) Config {
	return configAt(i, configs)
}

// configAt returns the configuration at index i in the ConfigList.
// This funciton is private so that reflection is not exposed in the
// public API.
func configAt(i int, configs ConfigList) Config {
	config := configs.Config()
	reflectConfigList := reflect.ValueOf(configs)
	reflectConfig := reflect.New(reflect.Indirect(reflect.ValueOf(config)).Type()).Elem() // reflect.ValueOf(config)

	accum := 1
	for field := 0; field < configs.NumFields(); field++ {
		fieldName := reflectConfigList.Type().Field(field).Name
		settings := reflectConfigList.FieldByName(fieldName)

		switch settings.Kind() {
		case reflect.String:
			reflectConfig.FieldByName(fieldName).Set(settings)
		case reflect.Slice:
			numSettings := settings.Len()

			reflectConfig.FieldByName(fieldName).Set(settings.Index(((i / accum) % numSettings)))
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

	// ValidAgent returns whether the argument agent is valid for the
	// Config
	ValidAgent(Agent) bool

	// Validate returns an error describing whether or not the
	// configuration is valid or not.
	Validate() error

	Type() Type // Returns the type of the configuration
}
