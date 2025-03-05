using Test
using CUDA
using HoloGrad: ContinuousLayout, SLM

@testset "Layout with $atype" for atype in test_atypes
    points = atype(rand(5, 2))
    layout = ContinuousLayout(points)
    @test layout.ntrap == 5
end

@testset "default_slm with $atype" for atype in test_atypes
    slm = atype(SLM(10, 5))
    @test size(slm.A) == (10, 5)
    @test size(slm.ϕ) == (10, 5)
    @test slm.SLM2π == 208.0

    slm = atype(SLM(10))
    @test size(slm.A) == (10, 10)
    @test size(slm.ϕ) == (10, 10)

    @test slm.A isa atype
    @test slm.ϕ isa atype
end