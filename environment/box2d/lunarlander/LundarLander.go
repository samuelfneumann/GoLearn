// Package lunarlander provides an implementation of the Lunar Lander
// environment.
package lunarlander

import (
	"fmt"
	"image/color"
	"math"
	"time"

	"golang.org/x/exp/rand"

	"github.com/ByteArena/box2d"
	"github.com/fogleman/gg"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/gonum/spatial/r1"
	"gonum.org/v1/gonum/stat/distuv"
	"sfneuman.com/golearn/environment"
	"sfneuman.com/golearn/spec"
	"sfneuman.com/golearn/timestep"
	"sfneuman.com/golearn/utils/floatutils"
)

const (
	FPS float64 = 50

	// speed of game, adjusts forces as well
	Scale float64 = 30.0

	XGravity float64 = 0.0
	YGravity float64 = -10.0

	MainEnginePower float64 = 13.0
	SideEnginePower float64 = 0.6

	LegAway         float64 = 20.0
	LegDown         float64 = 18.0
	LegW            float64 = 2.0
	LegH            float64 = 8.0
	LegSpringTorque float64 = 40.0

	SideEngineHeight float64 = 14.0
	SideEngineAway   float64 = 12.0

	Chunks int = 11.0

	ViewportW float64 = 600
	ViewportH float64 = 400

	// Action
	MaxContinuousAction float64 = 1.0
	MinContinuousAction float64 = -MaxContinuousAction
	MinDiscreteAction   int     = 0
	MaxDiscreteAction   int     = 3

	// State observations
	StateObservations int     = 8
	MinAngle          float64 = -math.Pi
	MaxAngle          float64 = math.Pi
	// Box2D limits on velocity: 2.0 units per timestep
	MaxVelocity float64 = 2.0 / (1.0 / FPS) // In Box2D units
	MinVelocity float64 = -MaxVelocity      // in Box2D units

	// Default starting values
	InitialX      float64 = (ViewportW / Scale / 2)
	InitialY      float64 = 0.92 * (ViewportH / Scale)
	InitialRandom float64 = 1000.0 // Set 1500 to make game harder
)

var (
	LanderPoly [][]float64 = [][]float64{
		{-14, 17},
		{-17, 0},
		{-17, -10},
		{17, -10},
		{17, 0},
		{14, 17},
	}
)

// contactDetector detects and manages the contacts of objects in
// the lunarLander environment.
type contactDetector struct {
	env *lunarLander
}

// newContactDetector returns a new contactDetector
func newContactDetector(e *lunarLander) *contactDetector {
	return &contactDetector{e}
}

// BeginContact performs housekeeping for the lunarLander environment
// when two objects collide
func (c *contactDetector) BeginContact(contact box2d.B2ContactInterface) {
	// Check if lander touched the moon
	if c.env.lander == contact.GetFixtureA().GetBody() ||
		c.env.lander == contact.GetFixtureB().GetBody() {
		// If the body touches the ground or boundary, it's game over.
		// The ship should be landed gently.
		c.env.gameOver = true
	}

	// Check if leg 1 touched the ground
	if c.env.legs[0] == contact.GetFixtureA().GetBody() ||
		c.env.legs[0] == contact.GetFixtureB().GetBody() {
		c.env.leg1GroundContact = true
	}

	// Check if leg 2 touched the ground
	if c.env.legs[1] == contact.GetFixtureA().GetBody() ||
		c.env.legs[1] == contact.GetFixtureB().GetBody() {
		c.env.leg2GroundContact = true
	}
}

// EndContact performs housekeeping for the lunarLander environment
// when two objects which had previously collided separate
func (c *contactDetector) EndContact(contact box2d.B2ContactInterface) {
	// Check if leg 1 left the ground
	if c.env.legs[0] == contact.GetFixtureA().GetBody() ||
		c.env.legs[0] == contact.GetFixtureB().GetBody() {
		c.env.leg1GroundContact = false
	}

	// Check if leg 2 left the ground
	if c.env.legs[1] == contact.GetFixtureA().GetBody() ||
		c.env.legs[1] == contact.GetFixtureB().GetBody() {
		c.env.leg2GroundContact = false
	}

}

func (c *contactDetector) PreSolve(contact box2d.B2ContactInterface,
	oldManifold box2d.B2Manifold) {
}
func (c *contactDetector) PostSolve(contact box2d.B2ContactInterface,
	impulse *box2d.B2ContactImpulse) {
}

// lunarLander implements the lunar lander environment. In this
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
// Any Task used in this struct must have a specific range of values
// for its Starter. The Starter should return a vector of 3 elements
// in the following order:
//
//	1. The x position to start at in the Box2D world. The specific
//	   values that this element can take on must be in the interval
//	   [0.05 * (ViewportW / Scale), 0.95 * (ViewportW / Scale)].
//	   The default value to use in the Starter is InitialX for the
//	   lower and upper bounds.
//	2. The y position to start at in the Box2D world. The specific
//	   values that this element can take on must be in the interval
//	   [ViewportH / Scale / 2, InitialY].
//	   The default value to use in the Starter is InitialY for the
//	   lower and upper bounds.
//	3. The initial random force to apply to the lander. This can be any
//	   value, but the default is InitialRandom for the lower and
//	   upper bounds for the Starter.
//
// This struct itself does not implement the environment.Environment
// interface. To use the environment, use either the Discrete or
// Continuous structs defined in this package.
type lunarLander struct {
	environment.Task

	world box2d.B2World

	// Moon and sky
	moon         *box2d.B2Body
	moonVertices [][2]float64
	moonShade    color.Color
	skyShade     color.Color

	// Lander and its legs
	lander        *box2d.B2Body
	landerColour1 color.Color
	landerColour2 color.Color

	legs              []*box2d.B2Body
	leg1GroundContact bool
	leg2GroundContact bool
	legColour1        color.Color
	legColour2        color.Color

	// The position of the landing pad
	helipadX1 float64
	helipadX2 float64
	helipadY  float64

	// World boundary - bounds the x and y axes
	boundary       []*box2d.B2Body
	boundaryColour color.Color

	// Bounds on actions and state observations
	actionBounds   r1.Interval
	angleBounds    r1.Interval
	velocityBounds r1.Interval

	// Bounds on the starting state position
	startXBounds r1.Interval
	startYBounds r1.Interval

	// Whether the current episode resulted in a game over. A game
	// over occurs if the ship touches the moon or the boundary. In
	// such a case, the ship is considered to crash. Only the legs of
	// the ship can touch the moon or any boundary.
	gameOver bool

	// Random environment variables
	discount float64
	prevStep timestep.TimeStep
	seed     uint64
	rng      distuv.Uniform
	mPower   float64
	sPower   float64
}

// newLunarLander creates and returns a new base lunarLander struct
func newLunarLander(task environment.Task, discount float64,
	seed uint64) (*lunarLander, timestep.TimeStep) {
	l := lunarLander{}
	l.world = box2d.MakeB2World(box2d.B2Vec2{X: XGravity, Y: YGravity})
	l.boundaryColour = color.RGBA{R: 255, G: 166, B: 0, A: 255}

	l.moon = nil
	l.moonVertices = make([][2]float64, 0, 2*(Chunks-1))
	l.moonShade = color.RGBA{R: 255, G: 255, B: 255, A: 255}
	l.skyShade = color.RGBA{R: 30, G: 30, B: 30, A: 255}

	l.lander = nil
	l.landerColour1 = color.RGBA{R: 128, G: 102, B: 230, A: 255}
	l.landerColour2 = color.RGBA{R: 77, G: 77, B: 128, A: 255}
	l.legColour1 = color.RGBA{R: 128, G: 102, B: 230, A: 255}
	l.legColour2 = color.RGBA{R: 77, G: 77, B: 128, A: 255}

	l.gameOver = false
	l.discount = discount

	l.seed = seed
	src := rand.NewSource(seed)
	rng := distuv.Uniform{Min: 0, Max: 1.0, Src: src}
	l.rng = rng

	// Bounds on actions and state observations
	l.actionBounds = r1.Interval{
		Min: MinContinuousAction,
		Max: MaxContinuousAction,
	}
	l.angleBounds = r1.Interval{Min: MinAngle, Max: MaxAngle}
	l.velocityBounds = r1.Interval{Min: MinVelocity, Max: MaxVelocity}

	// Bounds on the starting state position
	l.startYBounds = r1.Interval{Min: ViewportH / Scale / 2,
		Max: InitialY}
	l.startXBounds = r1.Interval{Min: 0.05 * (ViewportW / Scale),
		Max: 0.95 * (ViewportW / Scale)}

	// Register the task if needed
	t, ok := task.(lunarLanderTask)
	if ok {
		t.registerEnv(&l)
		l.Task = t
	} else {
		l.Task = task
	}

	// Reset will set l.prevStep automatically
	step := l.Reset()
	return &l, step
}

// destroy performs housekeeping to destroy Box2D objects when the
// environment needs to be reset
func (l *lunarLander) destroy() {
	if l.moon == nil {
		return
	}
	l.world.SetContactListener(nil)
	l.world.DestroyBody(l.moon)
	l.moon = nil

	l.world.DestroyBody(l.lander)
	l.lander = nil

	l.world.DestroyBody(l.legs[0])
	l.world.DestroyBody(l.legs[1])

	l.world.DestroyBody(l.boundary[0])
	l.world.DestroyBody(l.boundary[1])
	l.world.DestroyBody(l.boundary[2])
	l.world.DestroyBody(l.boundary[3])
}

// Reset resets the environment and returns the first timestep of the
// next episode
func (l *lunarLander) Reset() timestep.TimeStep {
	l.destroy()
	l.world.SetContactListener(newContactDetector(l))
	l.gameOver = false
	l.prevStep = timestep.TimeStep{}
	l.mPower = 0.0
	l.sPower = 0.0

	// If we have a lunarLanderTask, its internal variables will need
	// to be reset when the environment is reset.
	t, ok := l.Task.(lunarLanderTask)
	if ok {
		t.reset()
	}

	// Get a starting state
	start := t.Start()
	err := validateStart(start, l.startXBounds, l.startYBounds)
	if err != nil {
		panic(fmt.Sprintf("reset: %v", err))
	}

	// Maximum W and H for Box2D world
	W := ViewportW / Scale
	H := ViewportH / Scale

	// Bounds
	l.boundary = make([]*box2d.B2Body, 4)
	for i := 0; i < 4; i++ {
		boundsDef := box2d.NewB2BodyDef()
		boundsDef.Type = 0 // Static body
		l.boundary[i] = l.world.CreateBody(boundsDef)
		boundsShape := box2d.NewB2EdgeShape()
		if i == 0 {
			boundsShape.Set(
				box2d.MakeB2Vec2(0.0, 0.0),
				box2d.MakeB2Vec2(0.0, ViewportH/Scale),
			)
		} else if i == 1 {
			boundsShape.Set(
				box2d.MakeB2Vec2(0.0, ViewportH/Scale),
				box2d.MakeB2Vec2(ViewportW/Scale, ViewportH/Scale),
			)
		} else if i == 2 {
			boundsShape.Set(
				box2d.MakeB2Vec2(ViewportW/Scale, ViewportH/Scale),
				box2d.MakeB2Vec2(ViewportW/Scale, 0.0),
			)
		} else {
			boundsShape.Set(
				box2d.MakeB2Vec2(ViewportW/Scale, 0.0),
				box2d.MakeB2Vec2(0.0, 0.0),
			)
		}
		boundsFix := box2d.MakeB2FixtureDef()
		boundsFix.Shape = boundsShape
		l.boundary[i].CreateFixtureFromDef(&boundsFix)

	}

	// Terrain
	height := make([]float64, Chunks+1)
	for i := 0; i < len(height); i++ {
		height[i] = l.rng.Rand() * (H / 2.0)
	}

	chunkX := make([]float64, Chunks)
	for i := 0; i < Chunks; i++ {
		chunkX[i] = float64(i) * (W / float64(Chunks-1))
	}

	l.helipadX1 = chunkX[Chunks/2-1]
	l.helipadX2 = chunkX[Chunks/2+1]
	l.helipadY = H / 4

	height[Chunks/2-2] = l.helipadY
	height[Chunks/2-1] = l.helipadY
	height[Chunks/2] = l.helipadY
	height[Chunks/2+1] = l.helipadY
	height[Chunks/2+2] = l.helipadY

	smoothY := make([]float64, Chunks)
	for i := 0; i < Chunks; i++ {
		if i == 0 {
			smoothY[i] = 0.33 * (height[Chunks-1] + height[i] + height[i+1])
		} else {
			smoothY[i] = 0.33 * (height[i-1] + height[i] + height[i+1])
		}
	}

	// Moon def
	moonDef := box2d.NewB2BodyDef()
	moonDef.Type = 0
	moonDef.Position.Set(0, 0)

	// Moon body
	moonBody := l.world.CreateBody(moonDef)
	l.moon = moonBody

	// Moon shape
	moonShape := box2d.NewB2EdgeShape()
	moonShape.Set(*box2d.NewB2Vec2(0.0, 0.0), *box2d.NewB2Vec2(W, 0.0))

	// Moon fixture
	moonFixture := box2d.MakeB2FixtureDef()
	moonFixture.Shape = moonShape

	// Attach moon shape to body with fixture
	moonBody.CreateFixtureFromDef(&moonFixture)

	l.moonVertices = make([][2]float64, 0, 2*(Chunks-1))
	for i := 0; i < Chunks-1; i++ {
		p1 := [2]float64{chunkX[i], smoothY[i]}
		p2 := [2]float64{chunkX[i+1], smoothY[i+1]}
		l.moonVertices = append(l.moonVertices, p1, p2)

		// Create edge shape
		edge := box2d.NewB2EdgeShape()
		edge.M_vertex1 = box2d.MakeB2Vec2(p1[0], p1[1])
		edge.M_vertex2 = box2d.MakeB2Vec2(p2[0], p2[1])

		// Create fixture
		edgeFixture := box2d.MakeB2FixtureDef()
		edgeFixture.Shape = edge
		edgeFixture.Density = 0.0
		edgeFixture.Friction = 0.1

		// Connect shape and body
		moonBody.CreateFixtureFromDef(&edgeFixture)
	}

	// Lander Body Def
	initialX := start.AtVec(0)
	initialY := start.AtVec(1)
	landerDef := box2d.MakeB2BodyDef()
	landerDef.Type = 2 // Dynamic body
	landerDef.Position = box2d.MakeB2Vec2(initialX, initialY)
	landerDef.Angle = 0.0

	// Create lander body
	landerBody := l.world.CreateBody(&landerDef)
	l.lander = landerBody

	// Lander shape
	landerShape := box2d.NewB2PolygonShape()
	vertices := make([]box2d.B2Vec2, len(LanderPoly))
	for i := 0; i < len(LanderPoly); i++ {
		vertices[i] = box2d.MakeB2Vec2(
			LanderPoly[i][0]/Scale,
			LanderPoly[i][1]/Scale,
		)
	}
	landerShape.Set(vertices, len(vertices))

	// Lander fixture
	landerFix := box2d.MakeB2FixtureDef()
	landerFix.Shape = landerShape
	landerFix.Density = 5.0
	landerFix.Friction = 0.1
	landerFix.Restitution = 0.0
	filter := box2d.MakeB2Filter()
	filter.CategoryBits = 0x0010
	filter.MaskBits = 0x001
	landerFix.Filter = filter

	// Attach shape to body
	landerBody.CreateFixtureFromDef(&landerFix)

	// Apply force to lander
	initialRandom := start.AtVec(2)
	initialForceX := (l.rng.Rand() * 2 * initialRandom) - initialRandom
	initialForceY := (l.rng.Rand() * 2 * initialRandom) - initialRandom
	initialForce := box2d.MakeB2Vec2(initialForceX, initialForceY)
	l.lander.ApplyForceToCenter(initialForce, true)

	// Lander legs
	l.legs = make([]*box2d.B2Body, 0, 2)
	for _, i := range []float64{-1.0, 1.0} {
		// Create leg body def
		legDef := box2d.NewB2BodyDef()
		legDef.Type = 2 // Dynamic body
		legDef.Position = box2d.MakeB2Vec2(initialX-i*LegAway/Scale,
			initialY)
		legDef.Angle = i * 0.05

		// Create leg body
		leg := l.world.CreateBody(legDef)
		l.legs = append(l.legs, leg)

		// Create leg shape
		legShape := box2d.NewB2PolygonShape()
		legShape.SetAsBox(LegW/Scale, LegH/Scale)

		// Leg fixture
		legFix := box2d.MakeB2FixtureDef()
		legFix.Density = 1.0
		legFix.Restitution = 0.0
		legFix.Shape = legShape
		filter := box2d.MakeB2Filter()
		filter.CategoryBits = 0x0020
		filter.MaskBits = 0x001
		legFix.Filter = filter

		// Attach shape to leg body
		leg.CreateFixtureFromDef(&legFix)

		// Create revolute joint for attaching to lander
		rjd := box2d.MakeB2RevoluteJointDef()
		rjd.BodyA = l.lander
		rjd.BodyB = leg
		rjd.LocalAnchorA = box2d.MakeB2Vec2(0., 0.)
		rjd.LocalAnchorB = box2d.MakeB2Vec2(i*LegAway/Scale, LegDown/Scale)
		rjd.EnableMotor = true
		rjd.EnableLimit = true
		rjd.MaxMotorTorque = LegSpringTorque
		rjd.MotorSpeed = 0.7 * i // OpenAI: 0.3 * i

		if i < 0 {
			rjd.LowerAngle = 0.9 - 0.5
			rjd.UpperAngle = 0.9
		} else {
			rjd.LowerAngle = -0.9
			rjd.UpperAngle = -0.9 + 0.5
		}
		l.world.CreateJoint(&rjd)
	}
	l.leg1GroundContact = false
	l.leg2GroundContact = false

	step, last := l.Step(mat.NewVecDense(2, []float64{0.0, 0.0}))
	step.StepType = timestep.First
	if last {
		panic("reset: environment ended as soon as it began")
	}
	return step
}

// Step takes one environmental step given some action to apply to
// the lander. This function returns the next step in the episode and
// whether this next step was the last in the episode.
func (l *lunarLander) Step(a *mat.VecDense) (timestep.TimeStep, bool) {
	// Clip actions
	for i := 0; i < a.Len(); i++ {
		a.SetVec(i, floatutils.ClipInterval(a.AtVec(i), l.actionBounds))
	}

	// Engines
	tip := [2]float64{
		math.Sin(l.lander.GetAngle()),
		math.Cos(l.lander.GetAngle()),
	}
	side := [2]float64{-tip[1], tip[0]}
	var dispersion [2]float64
	for i := range dispersion {
		dispersion[i] = (l.rng.Rand() / l.rng.Max) / Scale
	}

	// Main engine
	mPower := 0.0
	if a.AtVec(0) > 0.0 {
		mPower = (floatutils.Clip(a.AtVec(0), 0.0, 1.0) + 1.0) * 0.5
		if mPower < 0.5 || mPower > 1.0 {
			panic("step: illegal power for main engines")
		}

		ox := tip[0]*(4.0/Scale+2.0*dispersion[0]) + side[0]*dispersion[1]
		oy := -tip[1]*(4.0/Scale+2.0*dispersion[0]) - side[1]*dispersion[1]

		impulsePos := box2d.MakeB2Vec2(
			l.lander.GetPosition().X+ox,
			l.lander.GetPosition().Y+oy,
		)
		linearImpulse := box2d.MakeB2Vec2(
			-ox*MainEnginePower*mPower,
			-oy*MainEnginePower*mPower,
		)
		l.lander.ApplyLinearImpulse(linearImpulse, impulsePos, true)
	}
	l.mPower = mPower

	// Side engies
	sPower := 0.0
	if math.Abs(a.AtVec(1)) > 0.5 {
		// Orientation engines
		direction := floatutils.Sign(a.AtVec(1))
		sPower = floatutils.Clip(math.Abs(a.AtVec(1)), 0.5, 1.0)
		if sPower < 0.5 || sPower > 1.0 {
			panic("step: illegal value for orientation engines")
		}

		ox := tip[0]*dispersion[0] + side[0]*(3.0*dispersion[1]+direction*
			SideEngineAway/Scale)
		oy := -tip[1]*dispersion[0] - side[1]*(3.0*dispersion[1]+direction*
			SideEngineAway/Scale)

		impulsePos := box2d.MakeB2Vec2(
			l.lander.GetPosition().X+ox-tip[0]*17.0/Scale,
			l.lander.GetPosition().Y+oy+tip[1]*SideEngineHeight/Scale,
		)
		linearImpulse := box2d.MakeB2Vec2(
			-ox*SideEnginePower*sPower,
			-oy*SideEnginePower*sPower,
		)
		l.lander.ApplyLinearImpulse(linearImpulse, impulsePos, true)
	}
	l.sPower = sPower

	// Update the Box2D world
	l.world.Step(1.0/FPS, 6*int(Scale), 2*int(Scale))

	// Calculate the state observation
	pos := l.Lander().GetPosition()
	vel := l.Lander().GetLinearVelocity()
	var leg1GroundContact, leg2GroundContact float64
	if l.leg1GroundContact {
		leg1GroundContact = 1.0
	}
	if l.leg2GroundContact {
		leg2GroundContact = 1.0
	}

	state := []float64{
		(pos.X - ViewportW/Scale/2.0) / (ViewportW / Scale / 2.0),
		(pos.Y - (l.helipadY + LegDown/Scale)) / (ViewportH/Scale - l.helipadY),
		vel.X * (ViewportW / Scale / 2.0) / FPS,
		vel.Y * (ViewportH / Scale / 2.0) / FPS,
		floatutils.Wrap(l.lander.GetAngle(), l.angleBounds.Min, l.angleBounds.Max),
		20.0 * l.lander.GetAngularVelocity() / FPS,
		leg1GroundContact,
		leg2GroundContact,
	}
	stateVec := mat.NewVecDense(StateObservations, state)

	// Construct the next timestep in the episode
	reward := l.GetReward(l.prevStep.Observation, a, stateVec)
	t := timestep.New(timestep.Mid, reward, l.discount, stateVec,
		l.prevStep.Number+1)
	l.End(&t)

	l.prevStep = t
	return t, t.Last()
}

// DiscountSpec returns the discount specification of the environment
func (l *lunarLander) DiscountSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{l.discount})

	return spec.NewEnvironment(shape, spec.Discount, lowerBound, lowerBound,
		spec.Continuous)
}

// ObservationSpec returns the observation specification of the
// environment
func (l *lunarLander) ObservationSpec() spec.Environment {
	shape := mat.NewVecDense(StateObservations, nil)

	minXVelocity := l.velocityBounds.Min * (ViewportW / Scale / 2) / FPS
	minYVelocity := l.velocityBounds.Min * (ViewportH / Scale / 2) / FPS
	minAngularVelocity := l.velocityBounds.Min * 20.0 / FPS
	lowerBound := mat.NewVecDense(StateObservations, []float64{
		-1.,
		0.,
		minXVelocity,
		minYVelocity,
		l.angleBounds.Min,
		minAngularVelocity,
		0.,
		0.,
	})

	maxXVelocity := l.velocityBounds.Max * (ViewportW / Scale / 2) / FPS
	maxYVelocity := l.velocityBounds.Max * (ViewportH / Scale / 2) / FPS
	maxAngularVelocity := l.velocityBounds.Max * 20.0 / FPS
	upperBound := mat.NewVecDense(StateObservations, []float64{
		1.,
		1., // Approximate the true upper bound by 1.0
		maxXVelocity,
		maxYVelocity,
		l.angleBounds.Max,
		maxAngularVelocity,
		1.,
		1.,
	})

	return spec.NewEnvironment(shape, spec.Observation, lowerBound, upperBound,
		spec.Continuous)
}

// CurrentTimeStep returns the current timestep of the environment
func (l *lunarLander) CurrentTimeStep() timestep.TimeStep {
	return l.prevStep
}

// SPower returns the current power of the side/orientation engine that
// is active
func (l *lunarLander) SPower() float64 {
	return l.sPower
}

// MPower returns the current power of the main engine
func (l *lunarLander) MPower() float64 {
	return l.mPower
}

// IsAwake returns whether the lander is awake
func (l *lunarLander) IsAwake() bool {
	return l.Lander().IsAwake()
}

// Lander returns a pointer to the Box2D lander of the environment
func (l *lunarLander) Lander() *box2d.B2Body {
	return l.lander
}

// Ground contact returns wheter each of the lander's two legs are
// in contact with the ground
func (l *lunarLander) GroundContact() (bool, bool) {
	return l.leg1GroundContact, l.leg2GroundContact
}

// IsGameOver returns whether the current episode resulted in a game
// over
func (l *lunarLander) IsGameOver() bool {
	return l.gameOver
}

// Render saves the current environment as a PNG with
// the given filename.
func (l *lunarLander) Render(filename string) {
	dc := gg.NewContext(int(ViewportW), int(ViewportH))

	// Draw moon as background
	dc.SetColor(l.moonShade)
	dc.Clear()

	// Draw sky
	dc.ClearPath()
	trans := l.moon.M_xf
	startCoords := WorldToPixelCoord([2]float64{l.moonVertices[0][0], ViewportH / Scale})
	dc.MoveTo(startCoords[0], startCoords[1])
	for i := 0; i < len(l.moonVertices); i++ {
		vertices := box2d.MakeB2Vec2(l.moonVertices[i][0], l.moonVertices[i][1])
		vertices = box2d.B2TransformVec2Mul(trans, vertices)
		coords := WorldToPixelCoord([2]float64{vertices.X, vertices.Y})
		dc.LineTo(coords[0], coords[1])
	}
	last := len(l.moonVertices) - 1
	endCoords := WorldToPixelCoord([2]float64{l.moonVertices[last][0], ViewportH / Scale})
	dc.LineTo(endCoords[0], endCoords[1])
	dc.LineTo(startCoords[0], startCoords[1])
	dc.SetColor(l.skyShade)
	dc.Fill()

	// Boundary
	dc.ClearPath()
	dc.SetColor(l.boundaryColour)
	dc.SetLineWidth(5.0)
	for i := range l.boundary {
		fix := l.boundary[i].GetFixtureList()
		sh := fix.M_shape.(*box2d.B2EdgeShape)

		pixelCoords1 := WorldToPixelCoord([2]float64{sh.M_vertex1.X, sh.M_vertex1.Y})
		pixelCoords2 := WorldToPixelCoord([2]float64{sh.M_vertex2.X, sh.M_vertex2.Y})

		dc.DrawLine(pixelCoords1[0], pixelCoords1[1], pixelCoords2[0], pixelCoords2[1])
	}
	dc.Stroke()

	// Lander
	landerFix := l.lander.GetFixtureList()
	for landerFix != nil {
		shape := landerFix.M_shape.(*box2d.B2PolygonShape)
		path := make([][2]float64, 0, shape.M_count)
		for i, vertex := range shape.M_vertices {
			if i >= shape.M_count {
				break
			}
			trans := landerFix.M_body.M_xf
			vertex = box2d.B2TransformVec2Mul(trans, vertex)

			pixelCoords := WorldToPixelCoord([2]float64{vertex.X, vertex.Y})
			path = append(path, pixelCoords)
		}

		dc.ClearPath()
		for _, point := range path {
			dc.LineTo(point[0], point[1])
		}
		dc.LineTo(path[0][0], path[0][1])

		dc.SetColor(l.landerColour1)
		dc.Fill()
		landerFix = landerFix.M_next
	}

	// Legs
	for _, leg := range l.legs {
		legFix := leg.GetFixtureList()
		for legFix != nil {
			dc.ClearPath()
			shape := legFix.M_shape.(*box2d.B2PolygonShape)
			path := make([][2]float64, 0, shape.M_count)
			for i, vertex := range shape.M_vertices {
				if i >= shape.M_count {
					break
				}
				trans := legFix.M_body.M_xf
				vertex := box2d.B2TransformVec2Mul(trans, vertex)

				pixelCoords := WorldToPixelCoord([2]float64{vertex.X,
					vertex.Y})
				path = append(path, pixelCoords)
			}
			for _, point := range path {
				dc.LineTo(point[0], point[1])
			}
			dc.LineTo(path[0][0], path[0][1])

			dc.SetColor(l.legColour1)
			dc.Fill()
			legFix = legFix.M_next
		}
	}

	dc.SavePNG(fmt.Sprintf("%v.png", filename))
}

// validateStart validates a starting state to ensure that it is legal
func validateStart(state *mat.VecDense, startXBounds,
	startYBounds r1.Interval) error {
	if state.Len() != 3 {
		return fmt.Errorf("starting values should be 4-dimensional")
	}

	if state.AtVec(0) > startXBounds.Max || state.AtVec(0) < startXBounds.Min {
		return fmt.Errorf("x position out of bounds, expected x ϵ [%v, %v] "+
			"but got x = %v", startXBounds.Min, startXBounds.Max,
			state.AtVec(0))
	}

	if state.AtVec(1) > startYBounds.Max || state.AtVec(1) < startYBounds.Min {
		return fmt.Errorf("y position out of bounds, expected y ϵ (%v, %v) "+
			"but got y = %v", startYBounds.Min, startYBounds.Max, state.AtVec(1))
	}

	return nil
}

// WorldToPixelCoord changes the world coordinates to pixel coordinates
// for drawing to the screen
func WorldToPixelCoord(coords [2]float64) [2]float64 {
	x, y := coords[0], coords[1]

	pixelX := Scale * x

	pixelY := ViewportH - Scale*y

	return [2]float64{pixelX, pixelY}
}

// Display saves n PNG frames of a random agent on the environment
func Display(n int) {
	s := environment.NewUniformStarter([]r1.Interval{
		{Min: InitialX, Max: InitialX},
		{Min: InitialY, Max: InitialY},
		{Min: InitialRandom, Max: InitialRandom},
	}, uint64(time.Now().UnixNano()))
	task := NewLand(s, 500)
	l, _ := newLunarLander(task, 0.99, uint64(time.Now().UnixNano()))

	src := rand.NewSource(uint64(time.Now().UnixNano()))
	rng := distuv.Uniform{Min: -1.0, Max: 1.0, Src: src}

	for i := 0; i < n; i++ {
		l.Render(fmt.Sprint(i))
		action := mat.NewVecDense(2, []float64{rng.Rand(), rng.Rand()})
		l.Step(action)
	}
	l.Render(fmt.Sprint(n))
}
