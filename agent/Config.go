package agent

import (
	"reflect"

	"sfneuman.com/golearn/environment"
)

// Configs represents a list of Config structs
type Configs interface {
	NumFields() int // Number of settable fields
	Len() int       // Number of settings
	Config() Config // Returns empty config of type in list
}

func ConfigAt(i int, configs Configs) Config {
	config := configs.Config()
	reflectConfigs := reflect.ValueOf(configs)
	reflectConfig := reflect.New(reflect.Indirect(reflect.ValueOf(config)).Type()).Elem() // reflect.ValueOf(config)

	accum := 1
	for field := 0; field < configs.NumFields(); field++ {
		fieldName := reflectConfigs.Type().Field(field).Name
		settings := reflectConfigs.FieldByName(fieldName)

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
}

// PolicyType represents a type of distribution that a policy could be
type PolicyType string

const (
	Gaussian    PolicyType = "Gaussian"
	Categorical PolicyType = "Softmax"
	EGreedy     PolicyType = "EGreedy"
)
