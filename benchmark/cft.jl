using HoloGrad
using HoloGrad: cft, icft
using Test
using BenchmarkTools

@testset "cft" begin
    Nx, Ny = 1024, 1024
    signal = randn(ComplexF64, Nx, Ny)
    layout = layout_example(Val(:butterflycontinuous); α = 0.0)
    @btime cft($layout, $signal)

    F = cft(layout, signal)
    @btime icft($layout, $F, $Nx, $Ny)
end