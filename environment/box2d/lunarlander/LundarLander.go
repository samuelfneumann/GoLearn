// Package lunarlander provides an implementation of the Lunar Lander
// environment.
package lunarlander

import (
	"fmt"
	"image/color"
	"math"

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

// TODO: NOW, WE DEAL ONLY WITH CONTINUOUS ACTIONS
// TODO: NOW, WE DEAL ONLY WITH CONTINUOUS ACTIONS
// TODO: NOW, WE DEAL ONLY WITH CONTINUOUS ACTIONS
// TODO: NOW, WE DEAL ONLY WITH CONTINUOUS ACTIONS
// TODO: NOW, WE DEAL ONLY WITH CONTINUOUS ACTIONS
// TODO: NOW, WE DEAL ONLY WITH CONTINUOUS ACTIONS
// TODO: NOW, WE DEAL ONLY WITH CONTINUOUS ACTIONS
// TODO: NOW, WE DEAL ONLY WITH CONTINUOUS ACTIONS
// TODO: NOW, WE DEAL ONLY WITH CONTINUOUS ACTIONS

// TODO: Document starter start values

// TODO: Implement Tasks & Add constants for Starters

// TODO: moonVertices could be []box2d.B2Vec2

// PrevShaping should be put in the Task
// Every time reset is called, Task.resetPrevShaping() should be called
//
// Check if Task is a lunarLanderTask. If so, register the lunarLander
// environment with the task at the end of each Step() (since Reset
// calls Step, it will also re-register on environment reset).
// lunarLander env can then have methods for accessing its mPower, sPower,
// and any other fields that a lunarLanderTask may need for reward computation, etc.

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
	InitialY      float64 = ((ViewportH - ViewportH/25) / Scale)
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

func WorldToPixelCoord(coords [2]float64) [2]float64 {
	x, y := coords[0], coords[1]

	pixelX := Scale * x

	pixelY := ViewportH - Scale*y

	return [2]float64{pixelX, pixelY}
}

func Display() {
	s := environment.NewUniformStarter([]r1.Interval{
		{Min: InitialX, Max: InitialX},
		{Min: InitialY, Max: InitialY},
		{Min: InitialRandom, Max: InitialRandom},
	}, 12)
	task := NewLand(s, 500)
	l, _ := newLunarLander(task, 0.99, 12)

	src := rand.NewSource(123123)
	rng := distuv.Uniform{Min: -1.0, Max: 1.0, Src: src}

	for i := 0; i < 500; i++ {
		l.Render(i)
		// l.world.Step(0.02, int(6*Scale), int(2*Scale))

		action := mat.NewVecDense(2, []float64{rng.Rand(), rng.Rand()})
		l.Step(action)
	}
}

type contactDetector struct {
	env *lunarLander
}

func newContactDetector(e *lunarLander) *contactDetector {
	return &contactDetector{e}
}

func (c *contactDetector) BeginContact(contact box2d.B2ContactInterface) {
	// Check if lander touched the moon
	if c.env.lander == contact.GetFixtureA().GetBody() ||
		c.env.lander == contact.GetFixtureB().GetBody() {
		// If the body touches the ground, it's game over.
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

func (c *contactDetector) PreSolve(contact box2d.B2ContactInterface, oldManifold box2d.B2Manifold) {}
func (c *contactDetector) PostSolve(contact box2d.B2ContactInterface, impulse *box2d.B2ContactImpulse) {
}

// Continuous actions
type lunarLander struct {
	environment.Task

	world box2d.B2World

	boundary       []*box2d.B2Body
	boundaryColour color.Color
	xBounds        r1.Interval
	yBounds        r1.Interval

	moon         *box2d.B2Body
	moonVertices [][2]float64
	moonShade    color.Color

	skyShade color.Color

	lander        *box2d.B2Body
	landerColour1 color.Color
	landerColour2 color.Color

	legs              []*box2d.B2Body
	leg1GroundContact bool
	leg2GroundContact bool
	legColour1        color.Color
	legColour2        color.Color

	helipadX1 float64
	helipadX2 float64
	helipadY  float64

	gameOver bool
	seed     uint64
	rng      distuv.Uniform

	actionBounds   r1.Interval
	angleBounds    r1.Interval
	velocityBounds r1.Interval

	discount float64
	prevStep timestep.TimeStep
	mPower   float64
	sPower   float64
}

func (l *lunarLander) SPower() float64 {
	return l.sPower
}

func (l *lunarLander) MPower() float64 {
	return l.mPower
}

func (l *lunarLander) IsAwake() bool {
	return l.Lander().IsAwake()
}

func (l *lunarLander) Lander() *box2d.B2Body {
	return l.lander
}

func (l *lunarLander) GroundContact() (bool, bool) {
	return l.leg1GroundContact, l.leg2GroundContact
}

func (l *lunarLander) IsGameOver() bool {
	return l.gameOver
}

func (l *lunarLander) Render(j int) {
	dc := gg.NewContext(int(ViewportW), int(ViewportH))
	dc.SetColor(l.moonShade)
	dc.Clear()

	// Draw moon
	for i := 0; i < len(l.moonVertices)-1; i++ {
		v1 := WorldToPixelCoord(l.moonVertices[i])
		v2 := WorldToPixelCoord(l.moonVertices[i+1])
		dc.DrawLine(v1[0], v1[1], v2[0], v2[1])
	}

	dc.SetColor(l.moonShade)
	dc.SetLineWidth(5.0)
	dc.Stroke()

	dc.ClearPath()
	startCoords := WorldToPixelCoord([2]float64{l.moonVertices[0][0], ViewportH / Scale})
	dc.MoveTo(startCoords[0], startCoords[1])
	for i := 0; i < len(l.moonVertices); i++ {
		vertices := box2d.MakeB2Vec2(l.moonVertices[i][0], l.moonVertices[i][1])
		trans := l.moon.M_xf
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

	// Bounds
	dc.ClearPath()
	dc.SetColor(l.boundaryColour)
	dc.SetLineWidth(5.0)
	for i := range l.boundary {
		// var p [][2]float64
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

	dc.SavePNG(fmt.Sprintf("./LL%v.png", j))
}

func newLunarLander(task environment.Task, discount float64,
	seed uint64) (*lunarLander, timestep.TimeStep) {
	l := lunarLander{}
	l.world = box2d.MakeB2World(box2d.B2Vec2{X: XGravity, Y: YGravity})
	// l.boundaryColour = color.RGBA{R: 61, G: 53, B: 122, A: 255}
	l.boundaryColour = color.RGBA{R: 255, G: 166, B: 0, A: 255}
	// l.boundaryColour = color.RGBA{R: 255, G: 115, B: 76, A: 255}

	l.moon = nil
	l.moonVertices = make([][2]float64, 0, 2*(Chunks-1))
	l.moonShade = color.RGBA{R: 255, G: 255, B: 255, A: 255}

	l.skyShade = color.RGBA{R: 30, G: 30, B: 30, A: 255}

	l.lander = nil
	l.landerColour1 = color.RGBA{R: 128, G: 102, B: 230, A: 255}
	l.landerColour2 = color.RGBA{R: 77, G: 77, B: 128, A: 255}

	l.legColour1 = color.RGBA{R: 128, G: 102, B: 230, A: 255}
	l.legColour2 = color.RGBA{R: 77, G: 77, B: 128, A: 255}

	l.seed = seed
	l.gameOver = false

	src := rand.NewSource(seed)
	rng := distuv.Uniform{Min: 0, Max: 1.0, Src: src}
	l.rng = rng
	l.discount = discount

	l.actionBounds = r1.Interval{
		Min: MinContinuousAction,
		Max: MaxContinuousAction,
	}

	l.angleBounds = r1.Interval{Min: MinAngle, Max: MaxAngle}
	l.velocityBounds = r1.Interval{Min: MinVelocity, Max: MaxVelocity}
	l.yBounds = r1.Interval{Min: ViewportH / Scale / 2, Max: InitialY}
	l.xBounds = r1.Interval{Min: -ViewportW / 2, Max: ViewportW / 2}

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

func (l *lunarLander) destroy() {
	if l.moon == nil {
		return
	}
	l.world.SetContactListener(nil)
	// l.cleanParticles(true)
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

func (l *lunarLander) Reset() timestep.TimeStep {
	l.destroy()
	l.world.SetContactListener(newContactDetector(l))
	l.gameOver = false
	l.prevStep = timestep.TimeStep{}
	l.mPower = 0.0
	l.sPower = 0.0

	// If we have a lunarLanderTask, its internal variables wil lneed
	// to be reset when the environment is reset.
	t, ok := l.Task.(lunarLanderTask)
	if ok {
		t.reset()
	}
	start := t.Start()
	err := validateStart(start, l.xBounds, l.yBounds)
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
			boundsShape.Set(box2d.MakeB2Vec2(0.0, 0.0), box2d.MakeB2Vec2(0.0, ViewportH/Scale))
		} else if i == 1 {
			boundsShape.Set(box2d.MakeB2Vec2(0.0, ViewportH/Scale), box2d.MakeB2Vec2(ViewportW/Scale, ViewportH/Scale))
		} else if i == 2 {
			boundsShape.Set(box2d.MakeB2Vec2(ViewportW/Scale, ViewportH/Scale),
				box2d.MakeB2Vec2(ViewportW/Scale, 0.0))
		} else {
			boundsShape.Set(box2d.MakeB2Vec2(ViewportW/Scale, 0.0), box2d.MakeB2Vec2(0.0, 0.0))
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
	if last {
		panic("reset: environment ended as soon as it began")
	}

	return step
}

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

	if len(state) != StateObservations {
		panic(fmt.Sprintf("step: illegal number of state observations "+
			"\n\twant(%v) \n\thave(%v)", StateObservations, len(state)))
	}
	stateVec := mat.NewVecDense(StateObservations, state)

	reward := l.GetReward(l.prevStep.Observation, a, stateVec)
	t := timestep.New(timestep.Mid, reward, l.discount, stateVec,
		l.prevStep.Number+1)
	l.End(&t)

	l.prevStep = t

	return t, t.Last()
}

func (l *lunarLander) DiscountSpec() spec.Environment {
	shape := mat.NewVecDense(1, nil)
	lowerBound := mat.NewVecDense(1, []float64{l.discount})

	return spec.NewEnvironment(shape, spec.Discount, lowerBound, lowerBound,
		spec.Continuous)
}

func (l *lunarLander) ObservationSpec() spec.Environment {
	shape := mat.NewVecDense(StateObservations, nil)

	lowerBound := mat.NewVecDense(StateObservations, []float64{
		0.,
		0.,
		l.velocityBounds.Min,
		l.velocityBounds.Min,
		l.angleBounds.Min,
		l.velocityBounds.Min,
		0.,
		0.,
	})

	upperBound := mat.NewVecDense(StateObservations, []float64{
		1.,
		(ViewportH/Scale - (l.helipadY + LegDown)) /
			(ViewportH/Scale - l.helipadY),
		l.velocityBounds.Max,
		l.velocityBounds.Max,
		l.angleBounds.Max,
		l.velocityBounds.Max,
		1.,
		1.,
	})

	return spec.NewEnvironment(shape, spec.Observation, lowerBound, upperBound,
		spec.Continuous)
}

func (l *lunarLander) CurrentTimeStep() timestep.TimeStep {
	return l.prevStep
}

func validateStart(state *mat.VecDense, xBounds, yBounds r1.Interval) error {
	if state.Len() != 3 {
		return fmt.Errorf("starting values should be 4-dimensional")
	}

	if state.AtVec(0) > xBounds.Max || state.AtVec(0) < xBounds.Min {
		return fmt.Errorf("x position out of bounds, expected x ϵ (%v, %v) "+
			"but got x = %v", -ViewportW/Scale, ViewportW/Scale,
			state.AtVec(0))
	}

	if state.AtVec(1) > yBounds.Max || state.AtVec(1) < yBounds.Min {
		return fmt.Errorf("y position out of bounds, expected y ϵ (%v, %v) "+
			"but got y = %v", ViewportH/Scale/2, InitialY, state.AtVec(1))
	}

	return nil
}
