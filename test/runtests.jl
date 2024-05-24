using qgh
using Test

@testset "qgh.jl" begin
    include("configuration.jl")
    include("cft.jl")
    include("wgs.jl")
    include("fixedpoint.jl")
end
