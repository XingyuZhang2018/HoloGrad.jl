struct SLM
    A::AbstractArray
    ϕ::AbstractArray
    SLM2π::Float64
    function SLM(A, ϕ, SLM2π)
        ϕ = mod.(ϕ, SLM2π) # fix ϕ to [0, SLM2π]
        new(A, ϕ, SLM2π)
    end
end

SLM(A::AbstractArray{Float64,2}) = SLM(normalize(A), 2 * pi * rand(size(A)...), 208.0) 
function SLM(K::Int)
    norm_waist = round(Int, 12.36*1e-3/2/(12.5*1e-6)) # waist normalized by the SLM pixel size
    range_xy = LinRange(-K÷2+1, K÷2, K)
    x = repeat(range_xy, 1, K)
    y = repeat(range_xy', K, 1)
    d = sqrt.(x .^ 2 .+ y .^ 2)
    g = exp.((-2 / norm_waist^2) .* d .^ 2)  # Intensity
    return SLM(g)
end

function ϕdiff(A::SLM, B::SLM) 
    @assert A.SLM2π == B.SLM2π "SLM2π must be the same"
    diff = mod.(A.ϕ - B.ϕ .+ A.SLM2π/2, A.SLM2π) .- A.SLM2π/2 # fix ϕ to [-SLM2π/2, SLM2π/2]
    return diff
end

abstract type Layout end
"""
Specify the target trap locations as a list of points. The `x` and `y` locations should be normalized in [0, 1].

Args:
    mask (Tensor): the grid mask, only positions marked True have traps,
"""
struct ContinuousLayout <: Layout
    points::Vector
    ntrap::Int
    function ContinuousLayout(points::Vector)
        for p in points
            @assert (0 <= p[1] <= 1 && 0 <= p[2] <= 1) "All points must be normalized in the range [0, 1]"
        end
        new(points, length(points))
    end
end

function union(A::ContinuousLayout, B::ContinuousLayout)
    points = union(A.points, B.points)
    return ContinuousLayout(points)
end

function setdiff(A::ContinuousLayout, B::ContinuousLayout)
    points = setdiff(A.points, B.points)
    return ContinuousLayout(points)
end
