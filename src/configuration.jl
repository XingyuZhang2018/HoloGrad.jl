struct SLM
    A::AbstractArray
    ϕ::AbstractArray
    SLM2π::Float64
    function SLM(A, ϕ, SLM2π)
        ϕ = mod.(ϕ .+ SLM2π/2, SLM2π) .- SLM2π/2 # fix ϕ to [-SLM2π/2, SLM2π/2]
        new(A, ϕ, SLM2π)
    end
end

SLM(A::AbstractArray{Float64,2}) = SLM(normalize(A), 2 * pi * rand(size(A)...), 208.0) 
SLM(L::Int) = SLM(L, L)
function SLM(Lx::Int, Ly::Int)
    norm_waist = round(Int, 12.36*1e-3/2/(12.5*1e-6)) # waist normalized by the SLM pixel size
    x = repeat(LinRange(-Lx÷2+1, Lx÷2, Lx), 1, Ly)
    y = repeat(LinRange(-Ly÷2+1, Ly÷2, Ly)', Lx, 1)
    d = sqrt.(x .^ 2 .+ y .^ 2)
    g = exp.((-2 / norm_waist^2) .* d .^ 2)  # Intensity
    return SLM(g)
end

function ϕdiff(A::SLM, B::SLM; ifradian = true) 
    @assert A.SLM2π == B.SLM2π "SLM2π must be the same"
    diff = mod.(A.ϕ - B.ϕ .+ A.SLM2π/2, A.SLM2π) .- A.SLM2π/2 # fix dϕ to [-SLM2π/2, SLM2π/2]
    ifradian && (diff = diff / A.SLM2π * 2π)
    return diff
end

CuArray(slm::SLM) = SLM(CuArray(slm.A), CuArray(slm.ϕ), slm.SLM2π)
Array(slm::SLM) = SLM(Array(slm.A), Array(slm.ϕ), slm.SLM2π)


abstract type Layout end
"""
Specify the target trap locations as a list of points. The `x` and `y` locations should be normalized in [0, 1].

Args:
    points (`ntraps × 2` Matrix): the target trap locations, each row is a point.
"""
struct ContinuousLayout <: Layout
    points::AbstractArray
    ntrap::Int
    function ContinuousLayout(points::AbstractArray)
        @assert all(-0.01 .<= points .<= 1.01) "All points must be normalized in the range [0, 1]" # not so strict (0.01) for some cases
        new(points, size(points, 1))
    end
end

CuArray(layout::ContinuousLayout) = ContinuousLayout(CuArray(layout.points))
Array(layout::ContinuousLayout) = ContinuousLayout(Array(layout.points))