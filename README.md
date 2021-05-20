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
The `Starter` interface determines how a task starts in an environment:
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

# Experiments
## Savers
`Savers` define what data an `Experiment` will save to disk. For example,
the `savers.Return Saver` will cause an `Experiment` to save the episodic
returns seen during the `Experiment`. The `savers.EpisodeLength Saver`
will cause an `Experiment` to save the episode lengths during an
`Experiment`. Before running an `Experiment`, `Savers` can be registered
with the `Experiment` by passing the required `Savers` to the `Experiment`
constructor. Additionally, the `Register()` method can be used to
register a `Saver` with an `Experiment` after the `Experiment` has already
been constructed.

Any struct that implements the `Saver` interface can be passed to an
`Experiment`. For each timestep, the `Experiment`
will send the latest `TimeStep` returned from the `Environment` to
each of the `Savers` registered with the `Experiment` using the `Saver`'s
`Track()` method. This will cache the needed data in each `Saver`. Once
the `Experiment` is done, the `Save()` method can then be called on each
`Saver` to save the recorded experimental data to disk. Alternatively, each
`Experiment` implements a `Save()` method which will automatically save
all data for each of the `Saver`s registered with the `Experiment`. This
is perhaps the easiest way to save experimental data.
The `Saver` interface
is:
```
type Saver interface {
	Track(t ts.TimeStep)
	Save()
}
```
`Savers` follow the `observer-observable` design pattern to track data
generated from an experiment and save it later.
