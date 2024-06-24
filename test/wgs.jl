using qgh
using Test

@testset "wgs cft" begin
    Random.seed!(42)
    points = [rand(2) for _ in 1:10]
    layout = ContinuousLayout(points)
    slm = SLM(10)
    algorithm = WGS()
    slm_padding, cost = match_image(layout, slm, algorithm)
    @test cost < 1e-6
end