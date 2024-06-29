using qgh
using Test
using CUDA

test_atypes = [Array]

@testset "qgh.jl" begin
    include("configuration.jl")
    include("ft.jl")
    include("wgs.jl")
    include("fixedpoint.jl")
end
