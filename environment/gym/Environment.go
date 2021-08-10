package gym

import (
	"fmt"

	"github.com/samuelfneumann/golearn/environment"
	"gonum.org/v1/gonum/mat"
)

type EnvironmentID string

const (
	MountainCarV0           EnvironmentID = "MountainCar-v0"
	MountainCarContinuousV0 EnvironmentID = "MountainCarContinuous=v0"
	AcrobotV1               EnvironmentID = "Acrobot-v1"
	CartPoleV1              EnvironmentID = "CartPole-v1"
	PendulumV0              EnvironmentID = "Pendulum-v0"
)

type GymEnv struct {
	client     *Client
	instanceId InstanceID
	envId      EnvironmentID
	discount   float64
}

func New(envId EnvironmentID, baseURL string,
	discount float64) environment.Environment {
	client, err := NewClient(baseURL)
	if err != nil {
		panic(fmt.Sprintf("new: cannot create client: %v", err))
	}

	instanceId, err := client.Create(string(envId))
	if err != nil {
		panic(fmt.Sprintf("new: cannot create gym environment: %v", err))
	}

	return &GymEnv{
		client:     client,
		instanceId: instanceId,
		envId:      envId,
		discount:   discount,
	}
}

func (g *GymEnv) ActionSpec() environment.Spec {
	space, err := g.client.ActionSpace(g.instanceId)
	if err != nil {
		panic(fmt.Sprintf("actionSpec: could not create action spec: %v", err))
	}

	if space.Name == "Discrete" {
		shape := mat.NewVecDense(1, nil)
		lowerBound := mat.NewVecDense(1, nil)
		upperBound := mat.NewVecDense(1, []float64{float64(space.N - 1)})

		return environment.NewSpec(shape, environment.Action, lowerBound,
			upperBound, environment.Discrete)
	} else if space.Name == "Box" {
		lowerBound := mat.NewVecDense(len(space.Low), space.Low)
		upperBound := mat.NewVecDense(len(space.High), space.High)
		shape := mat.NewVecDense(lowerBound.Len(), nil)

		return environment.NewSpec(shape, environment.Action, lowerBound,
			upperBound, environment.Continuous)
	} else {
		panic(fmt.Sprintf("actionSpec: no such action type: %v",
			space.Name))
	}
}

func (g *GymEnv) ObservationSpec() environment.Spec {
	space, err := g.client.ObservationSpace(g.id)
	if err != nil {
		panic(fmt.Sprintf("actionSpec: could not create action spec: %v", err))
	}

	if space.Name == "Box" {
		lowerBound := mat.NewVecDense(len(space.Low), space.Low)
		upperBound := mat.NewVecDense(len(space.High), space.High)
		shape := mat.NewVecDense(lowerBound.Len(), nil)

		return environment.NewSpec(shape, environment.Observation, lowerBound,
			upperBound, environment.Continuous)
	} else {
		panic(fmt.Sprintf("observationSpec: no such observation type: %v",
			space.Name))
	}
}

func (g *GymEnv) DiscountSpec() environment.Spec {
	shape := mat.NewVecDense(1, nil)
	bound := mat.NewVecDense(1, []float64{g.discount})

	return environment.NewSpec(shape, environment.Discount, bound, bound,
		environment.Continuous)
}

func (g *GymEnv) RewardSpec() environment.Spec {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{g.Min()})
	upperBound := mat.NewVecDense(1, []float64{g.Max()})

	return environment.NewSpec(shape, environment.Reward, lowerBound,
		upperBound, environment.Continuous)
}

func (g *GymEnv) Min() float64 {
	switch g.envId {
	case MountainCarV0:
		return -1.0

	case MountainCarContinuousV0:
		return -0.144

	case PendulumV0:
		return -16.2736044

	case AcrobotV1:
		return -1.0

	case CartPoleV1:
		return 0.0
	}

	panic(fmt.Sprintf("min: no such environment %v", g.envId))
}

func (g *GymEnv) Max() float64 {
	switch g.envId {
	case MountainCarV0:
		return 1.0

	case MountainCarContinuousV0:
		return 100.0

	case PendulumV0:
		return 0.0

	case AcrobotV1:
		return 0.0

	case CartPoleV1:
		return 1.0
	}

	panic(fmt.Sprintf("max: no such environment %v", g.envId))
}

func (g *GymEnv) AtGoal(state mat.Matrix) bool {
	panic("atGoal: goal checking is not enabled for Gym environments")
}
