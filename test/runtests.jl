using HoloGrad
using Test
using CUDA

test_atypes = [Array]

@testset "HoloGrad.jl" begin
    include("configuration.jl")
    include("ft.jl")
    include("wgs.jl")
    include("fixedpoint.jl")
end
