# GoLearn: A Reinforcement Learning Framework in Go

# To Do
- [  ] Environments should deal with Starters gracefully. If given a starter that starts out-of-bounds, then clip/normalize so that the starting state is within bounds if possible. 
- [  ] Rename Agent Spec struct to Config. Spec should describe something, config should determine a configuration of something.
- [  ] Environments should have a max episode step limit
- [  ] Environment spec should say if observations/actions are continuous or discrete. 
- [  ] Add environment rendering
- [  ] EnvironmentLoop Struct controls the agent-environment interaction
- [  ] Agent Spec/Config structs should work as the following 
- [  ] Config structs should use JSON
- [  ] For now, env animations can just generate and save PNGs, but later we should do this with OpenGL
