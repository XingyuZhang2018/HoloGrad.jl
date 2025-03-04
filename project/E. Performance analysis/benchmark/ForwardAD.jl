using qgh
using Random
using LinearAlgebra
using BenchmarkTools
using CUDA
using Test

@testset "butterfly" for atype in [Array, CuArray]
    Random.seed!(42)
    include("../../../example/layout_example.jl")

    N = 1024

    layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
    layout_end = atype(layout_example(Val(:butterflycontinuous); α = 0.01))
    slm = atype(SLM(N))

    algorithm = WGS(maxiter=50, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.9)

    Δx = layout_end.points - layout.points
    dxdt = Δx
    aditers = 15
    slm, cost, B = match_image(layout, slm, algorithm)
    println("WGS $atype time:")
    @btime CUDA.@sync match_image($layout, $slm, $algorithm)
    println("implicit $atype time:")
    @btime CUDA.@sync qgh.get_dϕdt($layout, $slm, $B, $dxdt)
    println("expand $atype time:")
    @btime CUDA.@sync qgh.get_dϕdt($layout, $slm, $B, $dxdt, $aditers)
end

@testset "circle" for atype in [Array, CuArray]
    Random.seed!(42)
    include("../../../example/layout_example.jl")

    N = 1024

    layout = atype(layout_example(Val(:circle); α = 0.0))
    layout_end = atype(layout_example(Val(:circle); α = 0.01))
    slm = atype(SLM(N))

    algorithm = WGS(maxiter=50, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.9)

    Δx = layout_end.points - layout.points
    dxdt = Δx
    aditers = 15
    slm, cost, B = match_image(layout, slm, algorithm)
    println("WGS $atype time:")
    @btime CUDA.@sync match_image($layout, $slm, $algorithm)
    println("implicit $atype time:")
    @btime CUDA.@sync qgh.get_dϕdt($layout, $slm, $B, $dxdt)
    println("expand $atype time:")
    @btime CUDA.@sync qgh.get_dϕdt($layout, $slm, $B, $dxdt, $aditers)
end