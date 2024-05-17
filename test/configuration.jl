using Test
using qgh: GridLayout, SLM, extract_locations

@testset "Layout" begin
    mask = zeros(Bool, 10, 10)
    mask[2:2:8, 2:2:8] .= true
    layout = GridLayout(mask)
    @test layout.ntrap == 16
end

@testset "default_slm" begin
    slm = SLM(10)
    @test size(slm.A) == (10, 10)
    @test size(slm.ϕ) == (10, 10)
    @test slm.SLM2π == 208.0
end

