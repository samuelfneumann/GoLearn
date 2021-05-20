# GoLearn: A Reinforcement Learning Framework in Go

# Agents
## Value-Method Agents
A number of value-based learning algorithms are included. So far, only
linear/tabular versions of `Q-Learning` and `Expected Sarsa` are implemented.
Value-based learning algorithms work with discrete action spaces. Actions
are selected from `(0, 1, 2, ..., N)` where `N` is the largest possible
action.

## Policy Gradient Agents
# Environments
This library makes a separation between an environment and a task. An
environment is simply some domain that can be acted in, but does not
return any reward for actions. An environment has a state, and actions
can be taken in an environment that affect the environment's state, but
no rewards are given for any actions in an environment. The currently
implemented environments are in the following packages:
```
- gridworld: Implements gridworld environments and their tasks
- classiccontrol/pendulum: Implements the classic control problem
Pendulum and its tasks
- classiccontrol/mountaincar: Implements the classic control problem
Mountain Car and its tasks
```

Although an environment has no concept of rewards, an environment does
have a Task, which determines the rewards taken for actions in the
environment, the starting states in an environment, and the end conditions
of an agent-environment interaction.

## Task Interface
### Starter Interface
### Ender Interface
## Environment Wrappers
Environment wrappers can be founds in the environment/wrappers package.

Environment wrappers are themselves environments, but they somehow alter
the underlying environment that they wrap. For example, a SinglePrecision
might wrap an environment and convert all observations to SinglePrecision.
A NoiseMaker might inject noise into the environmental observations. A
TileCoder might return tile-coded representations of environmental states.

So far, the following environment wrappers are implemented:
```
- TileCoding: Tile codes environmental observations
```

It is easy to implement your own environment wrapper. All you need to do
is create a struct that stores another environment, and have your
wrapper implement the Environment interface:
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
With struct embedding this is even easier. Simply embed an environment in
your environment wrapper, and then "override" the necessary methods. Usually,
only the `Reset()`, `Step()`, and `ObservationSpec()` methods need to
be overridden if you embed an environment in your wrapper.

Environment wrappers follow the decorator design pattern to ensure
easy extension and modification of environments.

### wrappers.TileCoding
A TileCoding is an environment wrapper, itself being an environment.
The TileCoding struct will tile-code all observations before passing
them to an agent. The TileCoding struct has the constructor:
```
func NewTileCoding(env environment.Environment, bins [][]int,
	seed uint64) (*TileCoding, ts.TimeStep)
```
The `env` parameter is an environment to wrap. The `bins` parameter
determins both the number of tilings to use and the bins per each
dimension, per tiling. The length of the outer slice is the number of
tilings for the TileCoding environment to use. The sub-slices determine
the number of tiles per dimension to use in each respective tiling.
For example, if we had:
```
bins := [][]int{{2, 3}, {16, 21}, {5, 6}}
```
Then the TileCoding environment would tile code all environmental
observations using `3` tilings before passing the observations back to
the agent. The first tiling has shape `2x3` tiles. The second tiling
will have shape `16x21` tiles. The third tiling will have 5 tiles
along the first dimension and 6 along the second dimension. Note that
the length of sub-slices should be equal to the environmental
observation vector's lengths. In the example above, we would expect the
embedded environment to return `2D` vector observations.

# Experiments
## Savers
Savers define what data an Experiment will save to disk. For example,
the `savers.Return` Saver will cause an Experiment to save the episodic
returns seen during the experiment. The `savers.EpisodeLength` Saver
will cause an Experiment to save the episode lengths during an
Experiment. Before running an Experiment, Savers can be registered
with the Experiment by passing the required Savers to the Experiment
constructor. Additionally, the Register() method can be used to
register a Saver with an Experiment.

Any struct that implements the Saver interface can be passed to an
experiment. The experiment will run. For each timestep, the experiment
will then send the latest TimeStep returned from the environment to
each of the Savers registered with the experiment using the Saver's
Track() method. This will cache the needed data in each Saver. Once
the experiment is done, the Save() method can then be called on each
Saver to save the recorded experimental data to disk. The Saver interface
is:
```
type Saver interface {
	Track(t ts.TimeStep)
	Save()
}
```
Savers follow the observer-observable design pattern to track data
generated from an experiment and save it later.
