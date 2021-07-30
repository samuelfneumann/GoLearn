package agent

import (
	"fmt"
	"reflect"
)

// Type represents a specific type of an agent Config.
// Config's with this type can create Agents of the corresponding type.
//
// For example, if a Config has Type CategoricalVanillaPG, then the
// Config is used to construct Vanilla Policy Gradient agents using
// categorical policies.
type Type string

const (
	// Linear methods
	EGreedyQLearningLinear    Type = "EGreedyQLearning-Linear"
	EGreedyESarsaLinear       Type = "EGreedyESarsa-Linear"
	GaussianActorCriticLinear Type = "GaussianActorCritic-Linear"

	// Deep methods
	CategoricalVanillaPGMLP  Type = "CategoricalVanillaPG-MLP"
	GaussianVanillaPGTreeMLP Type = "GaussianVanillaPG-TreeMLP"

	EGreedyDeepQMLP Type = "EGreedyDeepQ-MLP"
)

// Registered types with the package. Once a Type has been registered
// with this map, a Config or ConfigList with that type can be created.
//
// No Type's are registered wtih this package upon initialization.
// Each separate package is in charge of registering its Type with
// the package separately to avoid circular imports.
var registeredTypes map[Type]reflect.Type

func init() {
	registeredTypes = make(map[Type]reflect.Type)
}

// Register registers an agent's Type with a concrete ConfigList type
// so that upon deserialization of a TypedConfigList, ConfigLists of
// type agentType are deserialized into the concrete type of configs.
//
// Note that each package is required to register its own Config's
// with an agentType separately. This package3 registers no agentTypes
// with any Config's. This is to avoid circular imports.
func Register(agentType Type, configs ConfigList) {
	fmt.Println("Registering:", agentType)
	registeredTypes[agentType] = reflect.TypeOf(configs)
}
