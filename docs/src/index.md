```@meta
CurrentModule = RLTypes
```

# RLTypes

Documentation for [RLTypes](https://github.com/SvenDuve/RLTypes.jl).

This package provides shared functionality for the reinforcement learning framework. The package is not only required by the specific algorithm packages, but also exports names into the main module of Julia. The package contains important types and functions that are shared between the packages:

- DDPG
- DQN
- MBRL
- NNDynamics
- NODEDynamics
- ODERNNDynamics
- Rewards

The main purpose of the package is to provide

1. a type system for the reinforcement learning framework to enable multiple dispatch,
2. set hyperparameters automatically, or at least provide a common interface for setting hyperparameters,
3. to differentiate function signatures and hence enable multiple dispatch,
4. provide shared functionality, e.g. Replay Buffers.
   
The majority of the functionality is used internally, but certain function signatures set by the user depend on RLTypes. 

## Installation

In the julia REPL, run

```julia
using Pkg
Pkg.add(url="https://github.com/SvenDuve/RLTypes.jl")
```

Bring package into scope with

```julia
using RLTypes
```

## Example usage

```julia
# to be used within the function call to an RL Algorithm to set hyperparameters
AgentParameter(training_episodes=100, batch_size=128)
```

Check the source code for a full list of parameters.

## Function Reference

```@index
```

```@autodocs
Modules = [RLTypes]
```
