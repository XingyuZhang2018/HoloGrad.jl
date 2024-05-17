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

function plot(slm::SLM)
    fourier = slm.A .* exp.(slm.ϕ * (2im * π / slm.SLM2π))
    image = fftshift(fft(fourier))
    heatmap(abs.(image))
end