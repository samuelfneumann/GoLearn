# GoLearn: A Reinforcement Learning Framework in Go

# Algorithms
## Value-Based Algorithms
A number of value-based learning algorithms are included. So far, only
linear/tabular versions of `Q-Learning` and `Expected Sarsa` are implemented.
Value-based learning algorithms work with discrete action spaces. Actions
are selected from `(0, 1, 2, ..., N)` where `N` is the largest possible
action.

## Policy Gradient Algorithms
No policy gradient algorithms are implemented
# Environments
This library makes a separation between an `Environment` and a `Task`. An
`Environment` is simply some domain that can be acted in, but does not
return any reward for actions. An `Environment` has a state, and actions
can be taken in an `Environment` that affect the `Environment`'s state, but
no rewards are given for any actions in an `Environment`. The currently
implemented environments are in the following packages:

* `gridworld`: Implements gridworld environments and their tasks
* `classiccontrol/pendulum`: Implements the classic control problem
Pendulum and its tasks
* `classiccontrol/mountaincar`: Implements the classic control problem
Mountain Car and its tasks
* `classiccontrol/cartpole`: Implements the classic control problem
Cartpole and its tasks

Each package also defines public constants that determine the physical
parameters of the `Environment`. For example, `mountaincar.Gravity` is
the gravity used in the Mountain Car environment.

Although an `Environment` has no concept of rewards, an `Environment` does
have a `Task`, which determines the rewards taken for actions in the
`Environment`, the starting states in an`Environment`, and the end conditions
of an agent-environment interaction.

## `timestep` Package
## Task Interface

### Starter Interface
The `Starter` interface determines how a `Task` starts in an `Environment`:
```
type Starter interface {
	Start() mat.Vector
}
```
An environment will call the `Start()` method of its task, which will
return a starting state. The agent will then start from this state
to complete its task. If a `Starter` produces a starting state that is
not possible, then the environment will panic.

The main `Starter` for continuous-state environments is the `UniformStarter`
which selects starting states drawn from a multivariate uniform distribution,
where each dimension of the multivariate distribution is given bounds of
selection. For example:
```
var seed uint64 = 1
bounds := []r1.Interval{{Min: -0.6, Max:0.2}, {Min-0.02, Max: 0.02}}
starter := environment.NewUniformStarter(bounds, seed)
```
will create a `UniformStarter` which samples starting states `[x, y]`
uniformly, where `x ∈ [-0.6, 0.2]` and `y ∈ [-0.02, 0.02]`.

`Task`s can take in `Starter` structs to determine how and where the
`Task` will begin for each episode.


### Ender Interface
The `Ender` interface determines how a `Task` ends in an `Environment`:
```
type Ender interface {
	End(*timestep.TimeStep) bool
}
```
The `End()` method takes in a pointer to a `TimeStep`. The function checks
whether this `TimeStep` is the last in the episode. If so, the function
first changest the `TimeStep.StepType` field to `timestep.Last` and returns
`true`. If not, the function leaves the `TimeStep.StepType` field as
`timestep.Mid` and returns `false`.

The `environment` package implements global `Ender`s which can be used
with any `Environment`. Sub-packages of the `environment` package (such
as `environment/classiccontrol/mountaincar`) implement `Ender`s which
may be used for `Environment`s within that package. The global
`Ender`s are:

* `StepLimit`: ends episodes at a specific timestep limit.
* `IntervalLimit`: ends episodes when a state feature leaves an interval.

## Environment Wrappers
`Environment` wrappers can be founds in the `environment/wrappers` package.

`Environment` wrappers are themselves `Environment`s, but they somehow alter
the underlying `Environment` that they wrap. For example, a `SinglePrecision`
might wrap an `Environment` and convert all observations to `float32`.
A `NoiseMaker` might inject noise into the environmental observations. A
`TileCoder` might return tile-coded representations of environmental states.

So far, the following environment wrappers are implemented:

* `TileCoding`: Tile codes environmental observations


It is easy to implement your own environment wrapper. All you need to do
is create a struct that stores another `Environment` and have your
wrapper implement the `Environment` interface:
```
type Environment interface {
	Task
	Reset() timestep.TimeStep
	Step(action mat.Vector) (timestep.TimeStep, bool)
	DiscountSpec() spec.Environment
	ObservationSpec() spec.Environment
	ActionSpec() spec.Environment
}
```
With struct embedding this is even easier. Simply embed an `Environment` in
your `Environment` wrapper, and then "override" the necessary methods. Usually,
only the `Reset()`, `Step()`, and `ObservationSpec()` methods need to
be overridden if you embed an `Environment` in your wrapper.

`Environment` wrappers follow the `decorator` design pattern to ensure
easy extension and modification of `Environments`.

### wrappers.TileCoding
A `TileCoding` is an `Environment` wrapper, itself being an `Environment`.
The `TileCoding` struct will tile-code all observations before passing
them to an `Agent`. The `TileCoding` struct has the constructor:
```
func NewTileCoding(env environment.Environment, bins [][]int, seed uint64) (*TileCoding, ts.TimeStep)
```
The `env` parameter is an `Environment` to wrap. The `bins` parameter
determins both the number of tilings to use and the bins per each
dimension, per tiling. The length of the outer slice is the number of
tilings for the `TileCoding Environment` to use. The sub-slices determine
the number of tiles per dimension to use in each respective tiling.
For example, if we had:
```
bins := [][]int{{2, 3}, {16, 21}, {5, 6}}
```
Then the `TileCoding Environment` would tile code all environmental
observations using `3` tilings before passing the observations back to
the `Agent`. The first tiling has shape `2x3` tiles. The second tiling
will have shape `16x21` tiles. The third tiling will have 5 tiles
along the first dimension and 6 along the second dimension. Note that
the length of sub-slices should be equal to the environmental
observation vector's lengths. In the example above, we would expect the
embedded `Environment` to return `2D` vector observations.

Tile coding requires bounding the ranges of each feature dimension. The
`TileCoding Environment` takes care of this itself, and creates the tilings
automatically over these poritons of state feature space.

Tile coding also requires that tilings be offset from one another. Each
tiling is offset from the origin by uniformly randomly sampling an
offset between `[-tileWidth / OffsetDiv, tileWidth / OffsetDiv]` *for
each feature dimension*, where `tilecoder.OffsetDiv` is a global constant
that determines the degree to which tiles are offset from the origin.
Note that tilings are offset for *each state feature dimension*, meaning
that the tiling needs to be offset in each dimension that is tile coded.
If state features are `n-D`, then the offset of the tilings is also `n-D`.

# Experiments
## Trackers
`Trackers` define what data an `Experiment` will save to disk. For example,
the `trackers.Return Tracker` will cause an `Experiment` to save the episodic
returns seen during the `Experiment`. The `trackers.EpisodeLength Tracker`
will cause an `Experiment` to save the episode lengths during an
`Experiment`. Before running an `Experiment`, `Trackers` can be registered
with the `Experiment` by passing the required `Trackers` to the `Experiment`
constructor. Additionally, the `Register()` method can be used to
register a `Tracker` with an `Experiment` after the `Experiment` has already
been constructed.

Any struct that implements the `Tracker` interface can be passed to an
`Experiment`. For each timestep, the `Experiment`
will send the latest `TimeStep` returned from the `Environment` to
each of the `Trackers` registered with the `Experiment` using the `Tracker`'s
`Track()` method. This will cache the needed data in each `Tracker`. Once
the `Experiment` is done, the `Save()` method can then be called on each
`Tracker` to save the recorded experimental data to disk. Alternatively, each
`Experiment` implements a `Save()` method which will automatically save
all data for each of the `Tracker`s registered with the `Experiment`. This
is perhaps the easiest way to save experimental data.
The `Tracker` interface
is:
```
type Tracker interface {
	Track(t ts.TimeStep)
	Save()
}
```

`Trackers` follow the `observer-observable` design pattern to track data
generated from an experiment and save it later.


# ToDo

- [ ] UniformStarter needs documentation
- [ ] TileCoding and TileCoder could benefit from improved documentation

- [ ] Tasks and learners should both follow the Null Object pattern

- [ ] Cartpole needs `Render()` method


Documentation of AverageReward

Find some way to save the episodic return for average reward environments. Maybe create an AverageRewardTracker