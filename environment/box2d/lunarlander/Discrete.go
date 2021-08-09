package lunarlander

import (
	"fmt"

	"github.com/samuelfneumann/golearn/environment"
	"github.com/samuelfneumann/golearn/timestep"
	"gonum.org/v1/gonum/mat"
)

// Discrete implements the lunar lander environment. In this
// environment, an agent can fly a ship within a set bounding box
// viewport. At the bottom of the viewport is the moon, and the agent
// can land the ship on the moon. There is a landing pad on the moon,
// which is a completely horizontal portion of the moon and it is
// always located at the point (0, 0).
//
// State observations are vectors consisting of the following features
// in the following order:
//
//	1. The x distance from the lander to the center of the viewport
//	   Bounds: [-1, 1]
//	2. The y distance from the lander to the landing pad
//	   Bounds: [0, 1]
//	   Technically, the upper bound is ((ViewportH - (Lander.Top -
//	   Lander.Centre) - LegDown)/Scale - l.helipadY) /
//	   (ViewportH/Scale - l.helipadY) due to how the state observation
//	   is constructed and due to the fact that the bottom of the
//	   lander's legs cannot reach the boundary (since the lander cannot
//	   flip onto its back and fly upward), but an approximation of 1.0
//	   is sufficient. The true upper bound is approximately 0.88.
//	3. The x velocity of the lander
//	   Bounds: the bounds depend on the physical constants of the
// 	   Box2D universe. With the defaults in this file, the bounds are
//	   [-20, 20]
//	4. The y velocity of the lander
//	   Bounds: the bounds depend on the physical constants of the
// 	   Box2D universe. With the defaults in this file, the bounds are
//	   [-20, 20]
//	5. The angle of the lander
//	   Bounds: normalized between [-π, π]
//	6. The angular velocity of the lander
//	   Bounds: [-40, 40]
//	7. Whether the left leg has contact with the ground
//	   Bounds: feature in the set {0, 1}
//	8. Whether the right leg has contact with the ground
//	   Bounds: feature in the set {0, 1}
//
// More information on the Lunar Lander environment can be found at:
// https://gym.openai.com/envs/LunarLander-v2/
// https://gym.openai.com/envs/LunarLanderContinuous-v2/
//
// This implementation of LunarLander has a few differences from the
// OpenAI Gym implementation:
//
//	1. In this implementation, a boundary is placed around the viewport
//	   so that the lander cannot exit the viewport. This allows the
//	   x and y position features to be bounded. In the OpenAI gym
//	   implementation, there is no bounding box, and the agent is
//	   free to fly the lander as high as it wants, but episodes are
//	   terminated if the lander leaves the viewport along the x axis.
//	   Due to the boundary in this implementation, episodes are not
//	   cutoff when the lander leaves the viewport along the x axis,
//	   as this is not possible. The benefit of having a bounded x and
//	   y position is that tile-coding can easily be used.
//
//	2. State features are constructed slightly differently. In this
//	   implementation, the lander angle is normalized between [-π, π].
//	   So, if the lander rotates by an angle of 2π, the state
//	   observation results in an angle of 0. In the OpenAI Gym
//	   implementation, no normalization of the angle is done. Angle
//	   normalization allows for tile-coding to be easily used.
//
//	   This implementation normalized the y position using the distance
//	   from the landing pad to the top of the viewport bounding box.
//	   The OpenAI gym implementation normalizes by the halved height
//	   of the viewport. That is, the OpenAI Gym implementation's
//	   y position feature measures how far the ship's leg is from the
//	   landing pad in units of (viewport height / 2). This
//	   implementation's y position feature measures how far the ship's
//	   leg is from the landing pad in units of maximum possible
//	   distance from the ship to the landing pad.
//
//	3. The observation space in this implementation explicitly takes the
//	   limitations of Box2D into account. That is, the maximum velocity
//	   allowed in Box2D is considered and returned when the user
//	   asks for the observation space. In the OpenAI Gym implementation,
//	   the limitation of Box2D are not considered, and the observation
//	   space is considered to be (-∞, +∞) for all state features.
//
//	4. Due to (1), (2), and (3), bounds on the observation space are
//	   different between this implementation and the implementation of
//	   OpenAI Gym, and state observation features may be slightly
//	   different between this implementation and that of OpenAI Gym
//	   for the same underlying state.
//
// Any Task used in this struct must have a environmentific range of values
// for its Starter. The Starter should return a vector of 3 elements
// in the following order:
//
//	1. The x position to start at in the Box2D world. The environmentific
//	   values that this element can take on must be in the interval
//	   [0.05 * (ViewportW / Scale), 0.95 * (ViewportW / Scale)].
//	   The default value to use in the Starter is InitialX for the
//	   lower and upper bounds.
//	2. The y position to start at in the Box2D world. The environmentific
//	   values that this element can take on must be in the interval
//	   [ViewportH / Scale / 2, InitialY].
//	   The default value to use in the Starter is InitialY for the
//	   lower and upper bounds.
//	3. The initial random force to apply to the lander. This can be any
//	   value, but the default is InitialRandom for the lower and
//	   upper bounds for the Starter.
//
// Actions are discrete in the set {0, 1, 2, 3}. Actions outside of
// this range result in the environment to panic. The actions have
// the following meaning:
//
//	Action		Meaning
//	  0			No operation
//	  1			Fire left engine at 100%
//	  2			Fire main engine at 100%
//	  3			Fire right engine at 100%
//
// Discrete implements the environment.Environment interface.
type Discrete struct {
	*lunarLander
}

// NewDiscrete returns a new lunar lander environment with discrete
// actions
func NewDiscrete(task environment.Task, discount float64,
	seed uint64) (environment.Environment, timestep.TimeStep) {
	l, step := newLunarLander(task, discount, seed)
	return &Discrete{l}, step
}

// ActionSpec returns the action environmentification of the environment
func (c *Discrete) ActionSpec() environment.Spec {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{float64(MinDiscreteAction)})
	upperBound := mat.NewVecDense(1, []float64{float64(MaxDiscreteAction)})

	return environment.NewSpec(shape, environment.Action, lowerBound, upperBound,
		environment.Discrete)
}

// Step takes one environmental step and returns the next timestep
// and whether that timestep is the last in the episode
func (c *Discrete) Step(action *mat.VecDense) (timestep.TimeStep, bool) {
	a := int(action.AtVec(0))

	if a == 0 {
		// No operation
		return c.lunarLander.Step(mat.NewVecDense(2, []float64{0., 0.}))
	} else if a == 1 {
		// Fire left engine
		return c.lunarLander.Step(mat.NewVecDense(2, []float64{0.0, -1.0}))
	} else if a == 2 {
		// Fire main engine
		return c.lunarLander.Step(mat.NewVecDense(2, []float64{1.0, 0.0}))
	} else if a == 3 {
		// Fire right engine
		return c.lunarLander.Step(mat.NewVecDense(2, []float64{0.0, 1.0}))
	}
	panic(fmt.Sprintf("step: illegal action selection, expected action ϵ "+
		"[0, 1, 2, 3], received action = %v", a))
}
