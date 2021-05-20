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
## Task Interface
## Starter Interface
## Ender Interface
## Environment Wrappers
### Tile-Coding

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

# To Do
- [  ] Environments should deal with Starters gracefully. If given a starter that starts out-of-bounds, then clip/normalize so that the starting state is within bounds if possible.
-[  ] Rename Agent Spec struct to Config. Spec should describe something, config should determine a configuration of something.
-[  ] Add environment rendering
-[  ] Agent Spec/Config structs should work as the following
-[  ] Config structs should use JSON
-[  ] For now, env animations can just generate and save PNGs, but later we should do this with OpenGL
