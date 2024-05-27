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

"""
compute the cost from image amplitude.

Args:
    A (Array): the amplitude on target traps (not normalized),
    appearindex (Array): the index of appearing traps,
    disappearindex (Array): the index of disappearing traps,
    α (Real): the ratio of appearing traps.

Returns:
    * std/mean, the smaller, the more evenly distributed target traps are.

Warning:
    Lose efficacy when α is close to 0 or 1.

"""
function compute_cost(A::AbstractArray, appearindex, disappearindex, α)
    t = abs2.(A)
    mean = Statistics.mean(t)
    std_mean2 = (std(t[appearindex]) + std(t[disappearindex])) / mean

    t[disappearindex] ./= (1 - α)^2
    t[appearindex] ./= α^2
    std_mean1 = std(t) / mean

    return isnan(std_mean1) ? std_mean2 : min(std_mean1, std_mean2)
end

function plot(slm::SLM)
    fourier = slm.A .* exp.(slm.ϕ * (2im * π / slm.SLM2π))
    image = fft(fourier)
    heatmap(abs.(image))
end

function plot(slms)
    p = Plots.plot(layout=(1, length(slms)))
    for i in 1:length(slms)
        fourier = slms[i].A .* exp.(slms[i].ϕ * (2im * π / slms[i].SLM2π))
        image = fft(fourier)
        Plots.heatmap!(p[i], abs.(image), colorbar=false, aspect_ratio=:equal, ticks=false, yticks=false)
    end
    heatmap!(p, size=(200*length(slms), 200))
end