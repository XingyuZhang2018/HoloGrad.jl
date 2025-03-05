# HoloGrad.jl: Dynamic Hologram Generation with Gradient

[![Build Status](https://github.com/XingyuZhang2018/HoloGrad.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/XingyuZhang2018/HoloGrad.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/XingyuZhang2018/HoloGrad.jl/graph/badge.svg?token=jCbMe1oSPj)](https://codecov.io/gh/XingyuZhang2018/HoloGrad.jl)


This is a Julia project to evolute the hologram using gradient. 

## Overview
There are three main parts in this project:
- `src/`: the source code of the project
- `example/`: the example of the butterfly graph
- `project/`: Contains all plots and benchmark results presented in the paper [Dynamic Hologram Generation with Automatic Differentiation](https://arxiv.org/abs/...)

## Installation
```shell
> git clone https://github.com/XingyuZhang2018/HoloGrad.jl
```
move to the file and run `julia REPL`, press `]` into `Pkg REPL`
```julia
(@v1.11) pkg> activate .
Activating project at `~/HoloGrad.jl`
```

```julia
(HoloGrad) pkg> instantiate
```

## Basic Usage Example: Continuous Evolution of Butterfly Graph
Here's a step-by-step guide to generate the continuous evolution animation:

### Start Julia REPL and activate the project
```julia

$ julia

(@v1.11) pkg> activate .
Activating project at `~/HoloGrad`

(HoloGrad) pkg> 
```
### Load required packages
press `Backspace` into `Julia REPL`
```julia
julia> using HoloGrad
julia> using Random
julia> using CUDA  # Optional for GPU acceleration
```

###  Set random seed for reproducibility
```julia
julia> Random.seed!(42)
```
###  Include the layout example file
```julia
julia> include("layout_example.jl")
```
###  Choose array type (Array for CPU, CuArray for GPU)
```julia
julia> atype = Array
```
###  Create initial and target layouts
```julia
julia> layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
julia> layout_new = atype(layout_example(Val(:butterflycontinuous); α = 0.2))
```
###  Initialize SLM (Spatial Light Modulator)
```julia
julia> slm = atype(SLM(100))
```
###  Configure the WGS algorithm
```julia
julia> algorithm = WGS(maxiter=100, verbose=true, show_every=10, tol=1e-10, ratio_fixphase=0.8)
```
###  Run the evolution process
```julia
julia> layouts, slms = evolution_slm_flow(layout, layout_new, slm, algorithm)
```
more details about the `evolution_slm_flow` function can be found in the [annotation](https://github.com/XingyuZhang2018/HoloGrad/blob/main/src/evolution/gradient_flow.jl#L127-L153).
###  Generate and save animation
```julia
julia> HoloGrad.image_animation(slms, 200; file = "example/butterflycontinuous.gif")
```

The generated animation will look like this:

![Butterfly Continuous Evolution](butterflycontinuous.gif)