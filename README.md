# GoLearn: Reinforcement Learning in Go
GoLearn is a reinforcement learning Go module. It implements many environments
for reinforcement learning as well as agents. It also allows users to easily
run experiments through `JSON` configuration files without ever touching
the code.

# Algorithms
A number of algorithms are implemented. These are separated into the
`agent/linear` and `agent/nonlinear` packages.

The `agent/linear` package contains agents that **only** use linear function
approximation, and are implemented in GoNum. The `agent/nonlinear`
package contains agents that are implemented with Gorgonia for neural
network function approximation. Note that agents in the `agent/nonlinear`
package can use **either linear or nonlinear** function approximation
depending on how the neural network is set up. The *nonlinear* part
of the `agent/nonlinear` package refers to the ability of the package to use
nonlinear function approximation, but the package is not restricted to
only use nonlinear function approximation.

## Value-Based Algorithms
The following value based algorithms are implemented in the following
packages:

|          Agent          |                Package            |
|-------------------------|-----------------------------------|
|   `Linear Q-learning`   | `agent/linear/discrete/qlearning` |
| `Linear Expected SARSA` |   `agent/linear/discrete/esarsa`  |
|    `Deep Q-learning`    |  `agent/nonlinear/discrete/deepq` |

## Policy Gradient Algorithms
The following policy gradient algorithms are implemented in the following
packages:

|              Agent             |                 Package                |
|--------------------------------|----------------------------------------|
| `Linear-Gaussian Actor-Critic` | `agent/linear/continuous/actorcritic`  |
|   `Vanilla Policy Gradient`    | `agent/nonlinear/continuous/vanillapg` |

## Agent `Config`s and `ConfigList`s

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
* `classiccontrol/acrobot`: Implements the classic control problem
Acrobot and its tasks
* `box2d/lunarlander`: Implements the Lunar Lander environment

Each package also defines public constants that determine the physical
parameters of the `Environment`. For example, `mountaincar.Gravity` is
the gravity used in the Mountain Car environment.

In depth documentation for each environment is included in the source
files. You can view the documentation, which includes descriptions of
state features, bounds on state features, actions, dimensionality of
action, and more by using the `go doc` command or by viewing the source
files in a text editor.

Classic control environments were adapted from OpenAI gym's implementations.
All classic control environments have both discrete and continuous action
variants. Box2D environments were also adapted from OpenAI gym's
implementations and also have both discrete and continuous action variants.

Although an `Environment` has no concept of rewards, an `Environment` does
have a `Task`, which determines the rewards taken for actions in the
`Environment`, the starting states in an`Environment`, and the end conditions
of an agent-environment interaction.

## `envconfig` Package
The `envconfig` package is used to construct `Environment`s with specific
`Task`s with default parameters. For example, to create the Cartpole
environment with the `Balance` task with default parameters, one can
first construct a `Config` that describes the `Environment` and `Task`.
Once the `Config` has been constructed, the `CreateEnv()` method will
return the respective `Environment`:

```
c := envconfig.NewConfig(
	envconfig.Cartpole,		// Environment
	envconfig.Balance,		// Task
	true,					// Continuous actions?
	500,					// Episode cutoff
	0.99,					// Discount
	false,					// Use the OpenAI Gym (true) or the GoLearn (false) implementation
)

env, firstStep := c.CreateEnv()

// Use env in some agent-environment interaction
```

Currently, the following `Environment`-`Task` combinations are valid
configurations:
| Environment |               Task              |
|-------------|---------------------------------|
| MountainCar |               Goal              |
|  Cartpole   | Balance, SwingUp (soon to come) |
|  Pendulum   |             SwingUp             |
|   Acrobot   | SwingUp, Balance (soon to come) |
| LunarLander |               Land              |

Any other combination of `Environment`-`Task` will result in a panic
when calling `CreateEnv()`.

## `gym` Package
The `gym` package provides acces to OpenAI Gym's environments through an
`HTTP` server. Currently, the server is implemented but the Go client for
running and creating `Environment`s is not implemented.

## `timestep` Package
The `timestep` package manages environmental timesteps with the `TimeStep`
struct. Each time an `Environment` takes a step, it returns a new `TimeStep`
`struct`. A `TimeStep` contains a `StepType` (either `First`, `Middle`,
or `Last`), a reward for the previous action, a discount value, the observation
of the next state, the number of the timestep in the current episode, and the
`EndType` (either `TerminalStateReached`, `Timeout`, or `Nil`). `EndType`s have
the following meanings:

* `TerminalStateReached`: The episode ended because some terminal state (such
as a goal state) was reached. For example, in Mountain Car reaching the goal,
or in Cartpole `Balance` having the pole fall below some set angle.
* `Timeout`: The episode ended because a timestep limit was reached.
* `Nil`: The episode ended in some unspecified or unknown way.

The `timestep` package also contains a `Transition` `struct`. These are used
to model a transition of `(state, action, reward, discount, next state, next
action)`. Sometimes the `next action` is omitted. These `struct`s are sent to
`Agent`s for many different reasons, for example, to compute the TD error on a
transition in order to track the average reward.

## Task Interface
An `Environment` has a `Task` which outlines what the agent should acomplish.
For example, the `mountaincar` package implements a `Goal` `Task` where,
when added to a Mountain Car environment, rewards the agent for reaching a
goal state.

Each `Task` has a `Starter` and `Ender` (more on those later) which determine
how episodes start and end respectively. The `Task` computes rewards for
`state, action, next state` transitions and determines the goal states
that the agent should reach. The `Task` interface is:

```
type Task interface {
	Starter
	Ender
	GetReward(state mat.Vector, a mat.Vector, nextState mat.Vector) float64
	AtGoal(state mat.Matrix) bool
	Min() float64 // returns the min possible reward
	Max() float64 // returns the max possible reward
	RewardSpec() spec.Environment
}
```

All `Task`s have in-depth documentation in the `Go` source files. These
can be viewed with the `go doc` command.

New `Task`s can easily be implemented and added to existing environments
to change what the agent should accomplish in a given environment. This
kind of modularity makes it very easy to have a single environment with
different goals. For example, the Cartpole environment has a `Balance`
`Task`, but one could easily create a new `Task`, e.g. `SwingUp`, and
pass this task into the Cartpole constructor to create a totally new
environment that is not yet implemented by this module.

Each `Environment` can use any `struct` implementing the `Task` interface,
but commonly used `Task`s for each `Environment` are implemented in the
`Environment`'s respective package.

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

* `StepLimit`: Ends episodes at a specific timestep limit.
* `IntervalLimit`: Ends episodes when a state feature leaves an interval.
* `FunctionEnder`: Ends an episode when a function or method returns `true`.

## Environment Wrappers
`Environment` wrappers can be founds in the `environment/wrappers` package.

`Environment` wrappers are themselves `Environment`s, but they somehow alter
the underlying `Environment` that they wrap. For example, a `SinglePrecision`
might wrap an `Environment` and convert all observations to `float32`.
A `NoiseMaker` might inject noise into the environmental observations. A
`TileCoder` might return tile-coded representations of environmental states.

So far, the following environment wrappers are implemented:

* `TileCoding`: Tile codes environmental observations
* `IndexTileCoding`: Tile codes environmental observations and returns as
	state observations the indices of non-zero components in the tile-coded vector.
* `AverageReward`: Converts an environment to the average reward formulation,
	returning the differential reward at each timestep and tracking/updating
	the policy's average reward estimate over time. This wrapper easily converts
	any algorithm to its differential counterpart.


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

### wrappers.TileCoding and wrappers.IndexTileCoding
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

The implementation of tile coding in this library makes use of concurrency.
The tile coded vector is constructed by calculating the non-zero indices
generated by each tiling concurrently and then setting each of these
indices sequentially in a separate `goroutine`. A separate `goroutine` is
used to set the indices so that the indices can be set as soon as one of
the concurrent `goroutines` for calculating an index is finished.
This greatly reduces the computation time when there are many tilings and
many tiles per tiling.

The `IndexTileCoding` wrapper is conceptually identical to the `TileCoding`
wrapper except that the wrapper returns as state observations the indices
of non-zero components in the tile-coded original state observation. For
example if a state observation is tile coded and the tile coded representation
is `[1 0 0 1 0 0 0 0 0 0 1 0 1]`, then the state feature vector returned by
`IndexTileCoding` is `[3 10 12 0]` in no particular order except that the
bias unit will alway be the last index if a bias unit is used.

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

If an `Environment` is wrapped in such a way that the `TimeStep` returned
contains modified data, but the unmodified data is desired to be saved,
the underlying, wrapped `Environemnt` can be registered with a `Tracker`
using the `trackers.Register()` method. In this way, the data from the
underlying, wrapped `Environment` will be tracked instead of the data
from the wrapper `Environment`. For example, if an environment is wrapped
in an `wrappers.AverageReward` wrapper, then the differential reward is
returned on each timestep. In many cases, we desire to track the return of an
episode, not the differential return. In this case, the underlying, wrapped
`Environment` can be registered with a `Tracker` using the
`trackers.Register()` method. In this way, the underlying, un-modified reward
and the episodic return (rather than the episodic differential return) can be
tracked.

## Checkpointers
A `Checkpointer` checkpoints an `Agent` during an experiment by saving the
`Agent` to a binary file. The `Checkpointer` interface is:
```
type Checkpointer interface {
	Checkpoint(ts.TimeStep) error
}
```

Checkpointers can only work with `Serializable` `struct`s. A `struct` is
serializable if it implements the `Serializable` interface:
```
type Serializable interface {
	gob.GobEncoder
	gob.GobDecoder
}
```

Currently, the only implemented `Checkpointer` is an `nStep` `Checkpointer`
which checkpoints an `Agent` every `n` steps of an agent-environment
interaction. For more information, see the `checkpointer` package.

## Running the program
To run the program, simply run the `main.go` file. The program takes two
commandline arguments: a `JSON` configuration file for an `Experiment` and
a hyperparameter setting index for the `Agent` defined in the configuration
file. Example `JSON` `Experiment` configuration files are given in the
`experiments` directory.

An `Experiment` configuration file describes an `Environment` for the
`Experiment` as well as an `Agent` to run on the environment. For each
possible hyperparameter of the agent, the configuration file lists all
possible values that the user would like to test for that hyperparameter.
When running the program, the user must then specify a hyperparameter index
to use for the `Experiment`. The program will use that hyperparameter setting
index to get only the relevant hyperparameters from the configuration file,
construct an `Agent` using those hyperparameters, and then run the `Experiment`.
Hyperparameter settings indices wrap around, so that if there are `10` hyperparameter
settings, then hyperparameter settings indices `0 - 9` determine each hyperparameter
setting for the first run of the `Experiment`. Indices `10 - 19` determine each
hyperparameter setting for the second run of the `Experiment`. In general,
indices `10n -  10(n+1)` determine the hyperparameter settings for run
`n+1` of the `Experiment` and indices `10n + m` (for `m` fixed) refer to
sequential runs of hyperparameter setting `m` of the `Agent` in the
`Experiment`.


# ToDo


- [ ] Cartpole SwingUp would be nice to implement
- [ ] Would be nice to have the acitons in discrete pendulum determined by min and max discrete actions. E.g. Action i -> (action i / minDiscreteAction) * MinContinuousAction and similarly for max actions. Then, (MaxDiscreteAction - MinDiscreteAction) / 2 would be the 0 (do nothing) action which is the middle action.
- [ ] Eventually, it would be nice to have environments and tasks JSON serializable in the same manner as Solvers and InitWFns. This would make the config files super configurable...Instead of using default environment values all the time, we could have configurable environments through the JSON config files. This may prove problematic though with the gym-http-api...
- [ ] Eventually, it may be nice to have a Config and ConfigList for neural networks so that JSON config files can also determine the type of neural network to use with any agent. This is very low on the priority list.

- [ ] Readme should outline what exactly configs are and what they do for each package. Also, mention that environment Configs only allow environments with default behaviour, physical parameters, and task parameters to be created. To create a custom environemnt, you should use the relevant constructors with the relevant structs.
- [ ] Readme should mention that all configurations in a ConfigList should be compatible. E.g. if you have 3 hidden layers, then you must have 3 activations, etc.

- [ ] Would be cool if eventually the network package looked more like something like github.com/aunum/goro.
- [ ] Gridworld wrapper that returns features as [x, y] instead of one-hot. This is much harder than one-hot for NNs.
- [ ] Task AtGoal() -> argument should be Vector or *VecDense

- [ ] For many linear agents, action/state values are computed more than once. The Policy computes the action/state values at each timestep, and the Learner computes the same state/action values for the timestep when learning.

URGENT
- [ ] Accidentally changed specific, special -> environmentific in a bunch of files