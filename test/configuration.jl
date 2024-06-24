using Test
using qgh: ContinuousLayout, SLM

@testset "Layout" begin
    points = [rand(2) for _ in 1:5]
    layout = ContinuousLayout(points)
    @test layout.ntrap == 5
end

@testset "default_slm" begin
    slm = SLM(10)
    @test size(slm.A) == (10, 10)
    @test size(slm.ϕ) == (10, 10)
    @test slm.SLM2π == 208.0
end