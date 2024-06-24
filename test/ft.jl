using Test
using qgh: ContinuousLayout, preloc_cft, cft, icft
using LinearAlgebra
using Random

@testset "continuous fourier transform" begin
    Random.seed!(42)
    f = randn(ComplexF64, 10, 10)
    points = [rand(2) for _ in 1:10]
    layout = ContinuousLayout(points)

    X, Y, iX, iY = preloc_cft(layout, f)
    F = cft(f, X, Y) 
    @test size(F,1) == layout.ntrap
    f_new = icft(F, iX, iY)
    @test size(f) == size(f_new)
end