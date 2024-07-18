"""
compute the cost from image amplitude.

Args:
    A (Array): the amplitude on target traps (not normalized),

Returns:
    * std/mean, the smaller, the more evenly distributed target traps are.
"""
function compute_cost(A::AbstractArray)
    t = abs2.(A)
    mean = Statistics.mean(t)
    std_mean = std(t) / mean
    return std_mean
end

function compute_cost(slm::SLM, layout::Layout)
    A, ϕ, SLM2π = slm.A, slm.ϕ, slm.SLM2π

    # initial image space information
    field = A .* exp.(ϕ * (2im * π / SLM2π))
    X, Y, _, _ = preloc_cft(layout, field)
    trap = cft(field, X, Y)
    trap_A = abs.(trap)
    trap_A_mean = mean(trap_A)

    return compute_cost(trap_A), trap_A_mean
end

function compute_cost(slm::SLM, layout::Layout, trap_A_mean_origin::Real)
    cost, trap_A_mean = compute_cost(slm, layout)
    A_ratio = trap_A_mean / trap_A_mean_origin
    
    return cost, A_ratio
end

function kl_divergence(A::AbstractArray, B::AbstractArray) 
    A = abs.(A)
    A ./= sum(A)
    B = abs.(B)
    B ./= sum(B)

    div_log = A ./ B
    # @show div_log
    # div_log[div_log .== 0] .= 1   
    return sum(A .* log.(div_log))
end

find_max_intensity(slm::SLM, area, N::Int) = find_max_intensity(slm, area, N, N)
function find_max_intensity(slm::SLM, area, Nu::Int, Nv::Int)
    fourier = slm.A .* exp.(slm.ϕ * (2im * π / slm.SLM2π))
    X, Y = preloc_cft(fourier , Nu, Nv)
    image = cft_m(fourier, X, Y)

    max_intensity, index = findmax(abs.(image[area[1][1]:area[2][1], area[1][2]:area[2][2]]))
    return max_intensity, [area[1][1] + index[1] - 1, area[1][2] + index[2] - 1]
end