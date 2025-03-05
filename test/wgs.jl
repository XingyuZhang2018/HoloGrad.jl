using HoloGrad
using Test

@testset "wgs cft with $atype" for atype in test_atypes
    Random.seed!(42)
    points = atype(rand(10, 2))
    layout = ContinuousLayout(points)
    slm = atype(SLM(10))
    algorithm = WGS()
    slm, cost = match_image(layout, slm, algorithm)
    @test cost < 1e-2
end