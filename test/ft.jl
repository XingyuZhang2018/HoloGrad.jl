using Test
using qgh: ContinuousLayout, preloc_cft, cft, icft
using LinearAlgebra
using Random

@testset "continuous fourier transform with $atype" for atype in test_atypes
    Random.seed!(42)
    f = atype(randn(ComplexF64, 10, 10))
    points = atype(rand(10, 2))
    layout = ContinuousLayout(points)

    X, Y, iX, iY = preloc_cft(layout, f)
    F = cft(f, X, Y) 
    @test size(F,1) == layout.ntrap
    @test F isa atype
    f_new = icft(F, iX, iY)
    @test size(f) == size(f_new)
    @test f isa atype
end