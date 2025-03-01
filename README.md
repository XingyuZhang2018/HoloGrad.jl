# qgh

[![Build Status](https://github.com/XingyuZhang2018/qgh/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/XingyuZhang2018/qgh.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/XingyuZhang2018/qgh/graph/badge.svg?token=jCbMe1oSPj)](https://codecov.io/gh/XingyuZhang2018/qgh)


This is a Julia project for quantum graph homology.

## Installation
```shell
> git clone https://github.com/XingyuZhang2018/qgh
```
move to the file and run `julia REPL`, press `]` into `Pkg REPL`
```julia
(@v1.11) pkg> activate .
Activating project at `~/../qgh`
```

```julia
(qgh) pkg> instantiate
```

## Review
There are four main parts in this project:
- `src/`: the source code of the project
- `example/`: the example of the butterfly graph
- `project/`: the plots of all the results in paper [Dynamic Hologram Generation with Automatic Differentiation](https://arxiv.org/abs/...)
- `benchmark/`: the benchmark of the project with CPU and GPU acceleration

