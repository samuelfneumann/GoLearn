# GoLearn: Reinforcement Learning in Go

GoLearn is a reinforcement learning Go module. It implements many environments
for reinforcement learning as well as agents. It also allows users to easily
run experiments through `JSON` configuration files without ever touching
the code.

## Algorithms

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

### Value-Based Algorithms

The following value based algorithms are implemented in the following
packages:

|          Agent          |                Package            |
|-------------------------|-----------------------------------|
|   `Linear Q-learning`   | `agent/linear/discrete/qlearning` |
| `Linear Expected SARSA` |   `agent/linear/discrete/esarsa`  |
|    `Deep Q-learning`    |  `agent/nonlinear/discrete/deepq` |

### Policy Gradient Algorithms

The following policy gradient algorithms are implemented in the following
packages:

|                Agent               |                 Package                |
|------------------------------------|----------------------------------------|
|   `Linear-Gaussian Actor-Critic`   | `agent/linear/continuous/actorcritic`  |
|     `Vanilla Policy Gradient`      | `agent/nonlinear/continuous/vanillapg` |
|       `Vanilla Actor Critic`       | `agent/nonlinear/continuous/vanillaac` |

## Agent `Config`s and `ConfigList`s

Agents must be created with a configuration struct satisfying the `Config`
interface:

```go
// Config represents a configuration for creating an agent
type Config interface {
    // CreateAgent creates the agent that the config describes
    CreateAgent(env environment.Environment, seed uint64) (Agent, error)

    // ValidAgent returns whether the argument Agent is valid for the
    // Config. This differs from the Type() method in that an actual
    // Agent struct is used here, whereas the Type() method returns
    // a Type (string) describing the Agent's type. For example a
    // Config's Type may be "CategoricalVanillaPG-MLP" or "Gaussian
    // VanillaPG-TreeMLP", which are different Config types, but each of
    // which have *vanillapg.VanillaPG as a valid Agent since that
    // is the Agent the Configs describe.
    ValidAgent(Agent) bool

    // Validate returns an error describing whether or not the
    // configuration is valid.
    Validate() error

    // Type returns the type of agent which can be constructed from
    // the Config.
    Type() Type
}
```

Each `Config` struct uniquely determines the set of hyperparameters and
options of a specific agent. If you try to create an `Agent` for which
the `ValidAgent()` `Config` method would return `false`, the `Agent`
constructor will panic.

How can we create an agent? Imagine we have agent `X` defined in
package `xagent`. In package `xagent` then, there will be a Configuration
struct, for example imagine it is called `xConfig`. Then, if agent `X`
has the hyperparameters `learningRate`, `epsilon`, and `exploration`,
then `xConfig` might look something like:

```go
type xConfig struct {
    LearningRate float64
    Epsilon      float64
    Exploration  string
}
```

This configuration uniquely defines an `X` agent with specific hyperparameters.
There are then two ways of creating the agent `X`. The easiest way is to
call the `CreateAgent()` method on the `xConfig`:

```go
config := xagent.xConfig{
    LearningRate: 0.1,
    Epsilon: 0.01,
    Exploration: "Gaussian",
}

env := ... // Create some environment to train the agent in
seed := uint64(time.Now().UnixNano())

agent := config.CreateAgent(env, seed)

// Train the agent here
```

The other way is through the agent constructor:

```go
config := xagent.xConfig{
    LearningRate: 0.1,
    Epsilon: 0.01,
    Exploration: "Gaussian",
}

env := ... // Create some environment to train the agent in
seed := uint64(time.Now().UnixNano())

agent := xagent.New(config, env, seed)

// Train the agent here
```

Once the agent has been constructed, it is ready to train. Remember,
you must use a valid `Config` to construct a specific `Agent`.

A very common case we find in reinforcement learning is to sweep over
a bunch of different hyperparameter settings for a single agent type.
To do this, instead of creating a `[]Config`, we can create a `ConfigList`.
`ConfigList`s efficiently represent a set of all combinations of
hyperparameters. For example, a `ConfigList` for agent `X` might be:

```go
type xConfigList struct {
    LearningRate []float64
    Epsilon      []float64
    Exploration  []string
}
```

This `ConfigList` stores a list of all combinations of values found in the
fields `LearningRate`, `Epsilon`, and `Exploration`. For example, assume
a specific instantiation of an `xConfigList` looks like this:

```go
config := xConfigList{
    LearningRate []float64{0.1, 0.5},
    Epsilon      []float64{0.01, 0.001},
    Exploration  []string{"Gaussian},
}
```

Then, this list basically stores a slice of 5 `xConfig`s, each taking on
a different combination of `LearningRate`s, `Epsilon`s, and `Exploration`s.
On a `ConfigList`, you can then call an `ConfigAt(i int, c ConfigList)`
function, defined in the `agent` package, which will
return the `Config` at (0-indexed) position `i` in the list. The `ConfigAt()`
function *wraps around* the end of the list, so if the `ConfigList` has
`n` `Config`s in it, indices `nk + i, k ϵ Z, i < n ϵ Z`, all refer
to the same `Config` at index `i < n ϵ Z` in the `ConfigList`.

**Important**: A `ConfigList` **must** have its fields having the same name
as the corresponding ``Config`` of which it stores a list of. Otherwise,
the `At()` method will panic.

An `Experiment` can be `JSON` serialized (more on that later). In the
`Experiment` `JSON` file, the `Agent` configuration will be a specific
concrete type, the `TypedConfigList`, which provides a privitive way
of typing `ConfigList`s.

### `TypedConfigList`

A `TypedConfigList` provides a primitive typing mechanism of `ConfigList`s.
A `TypedConfigList` consists of a type descriptor and a concrete `ConfigList`
value. This might seem similar to how `interface` values are constructed
in `Go`, and that's because it is!

```go
TypedConfigList = | agent.Type | ConfigList |
```

Given a `JSON` serialization of a `TypedConfigList`, we can easily
`JSON` unmarshall this `TypedConfigList`, which will return the stored
`ConfigList` as its underlying concrete type. This is required because
we cannot easily unmarshall a concrete `ConfigList` into a
`ConfigList` (interface) value. For example, the following code will panic:

```go
var a ConfigList // ConfigList is an interface

err := json.Unmarshall(data, a) // Data holds a concrete type
if a == nil {
    panic(err)
}
```

A (possibly `JSON` marshalled) `Experiment` (more on this later) must store
an agent `ConfigList` describing which hyperparameters of a specific agent
should be run in the experiment. A problem is that the marshalled
`Experiment` will not know before runtime what kind of `Agent` is being
used in the experiment and therefore cannot know the concrete type
of the `ConfigList` of the `Agent`. The best the `Experiment` can do
is unmarshall the `ConfigList` into an interface value. But this will
result in the problem shown in the code block above.

The `JSON` marhsalled `TypedConfigList` will perform `ConfigList` typing
for the `Experiment`. The `Experiment` will then simply call `CreateAgent()`
and run.

## Environments

This library makes a separation between an `Environment` and a `Task`. An
`Environment` is simply some domain that can be acted in, but does not
return any reward for actions. An `Environment` has a state, and actions
can be taken in an `Environment` that affect the `Environment`'s state, but
no rewards are given for any actions in an `Environment`. The currently
implemented environments are in the following packages:

* `gridworld`: Implements gridworld environments
* `maze`: Implements randomly generated maze environments
* `classiccontrol`: Implements the classic control environments: Mountain Car, Pendulum, Cartpole, and Acrobot
* `box2d`: Implements environments using the [Box2D](https://box2d.org/) physics simulator [Go port](https://github.com/ByteArena/box2d)
* `mujoco`: Implements environments using the [MuJoCo](http://www.mujoco.org/) physics simulator
* `gym`: Provides access to [OpenAI Gym](https://gym.openai.com/)'s environments through [GoGym: Go Bindings for OpenAI Gym](https://github.com/samuelfneumann/GoGym).

Each package also defines public constants that determine the physical
parameters of the `Environment`. For example, `mountaincar.Gravity` is
the gravity used in the Mountain Car environment.

In depth documentation for each environment is included in the source
files. You can view the documentation, which includes descriptions of
state features, bounds on state features, actions, dimensionality of
action, and more by using the `go doc` command or by viewing the source
files in a text editor.

Classic control environments were adapted from [OpenAI Gym](https://gym.openai.com/)'s implementations.
All classic control environments have both discrete and continuous action
variants. Box2D environments were also adapted from [OpenAI Gym](https://gym.openai.com/)'s
implementations and also have both discrete and continuous action variants.

Although an `Environment` has no concept of rewards, an `Environment` does
have a `Task`, which determines the rewards taken for actions in the
`Environment`, the starting states in an`Environment`, and the end conditions
of an agent-environment interaction.

### `envconfig` Package

The `envconfig` package is used to construct `Environment`s with specific
`Task`s with default parameters. For example, to create the Cartpole
environment with the `Balance` task with default parameters, one can
first construct a `Config` that describes the `Environment` and `Task`.
Once the `Config` has been constructed, the `CreateEnv()` method will
return the respective `Environment`:

```go
c := envconfig.NewConfig(
    envconfig.Cartpole,        // Environment
    envconfig.Balance,        // Task
    true,                    // Continuous actions?
    500,                    // Episode cutoff
    0.99,                    // Discount
    false,                    // Use the OpenAI Gym (true) or the GoLearn (false) implementation
)

env, firstStep := c.CreateEnv()

// Use env in some agent-environment interaction
```

Currently, the following `Environment`-`Task` combinations are valid
configurations:
|  Environment |               Task              |
|--------------|---------------------------------|
|   Gridworld  |               Goal              |
|     Maze     |               Goal              |
|  MountainCar |               Goal              |
|   Cartpole   | Balance, SwingUp (soon to come) |
|   Pendulum   |             SwingUp             |
|    Acrobot   | SwingUp, Balance (soon to come) |
|  LunarLander |               Land              |
|    Hopper    |               Hop               |
|    Reacher   |               Reach             |

Any other combination of `Environment`-`Task` will result in an error
when calling `CreateEnv()`.

### `gym` Package

The `gym` package provides acces to [OpenAI Gym](https://gym.openai.com/)'s
environments through [GoGym: Go Bindings for OpenAI Gym](https://github.com/samuelfneumann/GoGym).
Currently, the package only supports the `MuJoCo` and `Classic Control` environment
suites since these are the only suites [GoGym: Go Bindings for OpenAI Gym](https://github.com/samuelfneumann/GoGym)
currently supports. Once [GoGym: Go Bindings for OpenAI Gym](https://github.com/samuelfneumann/GoGym)
has added more environments, this package will automatically work with the
new environments (given that you update [GoGym: Go Bindings for OpenAI Gym](https://github.com/samuelfneumann/GoGym)).

All [OpenAI Gym](https://gym.openai.com/) environments work with only their
default tasks and episode cutoffs.

### `timestep` Package

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

### Task Interface

An `Environment` has a `Task` which outlines what the agent should acomplish.
For example, the `mountaincar` package implements a `Goal` `Task` where,
when added to a Mountain Car environment, rewards the agent for reaching a
goal state.

Each `Task` has a `Starter` and `Ender` (more on those later) which determine
how episodes start and end respectively. The `Task` computes rewards for
`state, action, next state` transitions and determines the goal states
that the agent should reach. The `Task` interface is:

```go
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

```go
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

```go
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

```go
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

```go
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

```go
func NewTileCoding(env environment.Environment, bins [][]int, seed uint64) (*TileCoding, ts.TimeStep)
```

The `env` parameter is an `Environment` to wrap. The `bins` parameter
determins both the number of tilings to use and the bins per each
dimension, per tiling. The length of the outer slice is the number of
tilings for the `TileCoding Environment` to use. The sub-slices determine
the number of tiles per dimension to use in each respective tiling.
For example, if we had:

```go
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

## Experiments

### Trackers

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

```go
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

### Checkpointers

A `Checkpointer` checkpoints an `Agent` during an experiment by saving the
`Agent` to a binary file. The `Checkpointer` interface is:

```go
type Checkpointer interface {
    Checkpoint(ts.TimeStep) error
}
```

Checkpointers can only work with `Serializable` `struct`s. A `struct` is
serializable if it implements the `Serializable` interface:

```go
type Serializable interface {
    gob.GobEncoder
    gob.GobDecoder
}
```

Currently, the only implemented `Checkpointer` is an `nStep` `Checkpointer`
which checkpoints an `Agent` every `n` steps of an agent-environment
interaction. For more information, see the `checkpointer` package.

Currently, no agents implement the `Serializable` interface. This will
be added on an *as-needed* basis.

## Experiment Configs

An `experiment.Config` outlines what kind of `Experiment` should be run
and with which `Environment` and which `Agent` with which hyperparameters
(defined by the `Agent`'s `ConfigList`). The `Experiment` `Config` holds
an `Environment` `Config` and an `Agent` `ConfigList` (actually, it
holds a `TypedConfigList` so that it knows how to `JSON` unmarshall the
`ConfigList`). The `Experiment` `Config` also has a `Type` which outlines
what `Type` of experiment we are running (e.g. an `OnlineExp` is a
`Type` of `Experiment` that runs the `Agent` online and performs online
evaluation only):

```go
// Config represents a configuration of an experiment.
type Config struct {
    Type
    MaxSteps  uint
    EnvConf   envconfig.Config
    AgentConf agent.TypedConfigList
}
```

To create and run an `Experiment`, you can use the specific `Experiment`'s
constructor. For example, if we wanted an `Online` experiment:

```go
agent := ...         // Create agent
env := ...           // Create environment
maxSteps := ...      // Maximum number of steps for the experiment
trackers := ...      // Create trackers for the experiment
checkpointers := ... // Create checkpointers for the experiment

exp := NewOnline(env, agent, maxSteps, trackers, checkpointers)
exp.Run()  // Run the experiment
exp.Save() // Save the data generated
```

Or you can use an `Experiment` `Config` (these can be `JSON` serialized
to store and run later):

```go
agentConfig := ...         // Create agent configuration list
envConfig := ...           // Create environment configuration
maxSteps := ...      // Maximum number of steps for the experiment
trackers := ...      // Create trackers for the experiment
checkpointers := ... // Create checkpointers for the experiment

expConfig := experiment.Config{
    Type: experiment.OnlineExp,
    MaxSteps: 1_000_000,
    EnvConf: envConfig,
    AgentConf: agentConfig,
}

// JSON marshall the expConfig so that we can run in as many times as we
// want to later

var agentConfig int = ... // The index of the agent in the ConfigList to run
var seed uint64 = ...     // Seed for the agent and environment

exp := expConfig.CreateExperiment(agentConfig, seed, trackers, checkpointers)
exp.Run()  // Run the experiment
exp.Save() // Save the data generated
```

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

## ToDo

* [ ] Cartpole SwingUp would be nice to implement

* [ ] Would be nice to have the acitons in discrete pendulum determined by min and max discrete actions. E.g. Action i -> (action i / minDiscreteAction) - MinContinuousAction and similarly for max actions. Then, (MaxDiscreteAction * MinDiscreteAction) / 2 would be the 0 (do nothing) action which is the middle action.

* [ ] Eventually, it would be nice to have environments and tasks JSON serializable in the same manner as Solvers and InitWFns. This would make the config files super configurable...Instead of using default environment values all the time, we could have configurable environments through the JSON config files.

* [ ] Readme should mention that all configurations in a ConfigList should be compatible. E.g. if you have 3 hidden layers, then you must have 3 activations, etc.

* [ ] Task AtGoal() -> argument should be Vector or *VecDense

* [ ] Add `gym` to the `EnvConfig` structs.

* [ ] Add `TimeLimit` to `gym` package so that time limits can be altered

* [ ] Move GAEBuffer and ExpReplay to a new `buffer` package - in which case GAE buffer needs a public API
