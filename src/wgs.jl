@with_kw struct WGS
    show_every::Int = 10
    niters::Int = 50
    ratio_fixphase::Float64 = 0.8
    tol::Float64 = Defaults.tol
    maxiter::Int = Defaults.maxiter
    verbose::Bool = Defaults.verbose
end

function match_image(layout::Layout, slm::SLM, algorithm::WGS)
    # initial image space information
    field = slm.A .* exp.(1im * slm.ϕ)
    
    image = fftshift(fft(field))
    # @show image[1:10]
    trap = extract_locations(layout, image)
    trap_A = normalize!(abs.(trap))
    # @show trap_A[1:10]
    trap_ϕ = angle.(trap)
    target_A_reweight = ones(layout.ntrap) / sqrt(layout.ntrap)

    # ------ Iterative Optimization Start --------
    t0 = time()
    for i in 1:algorithm.niters
        target_A_reweight .*= clamp!(mean(trap_A) ./ trap_A, 0.1, 10)

        # image plane
        v_forced_trap = normalize(target_A_reweight) .* exp.(1im * trap_ϕ) # regenerate image
        v_forced_image = embed_locations(layout, v_forced_trap)
    
        # compute phase front at SLM
        t = ifft(ifftshift(v_forced_image))
        # phase = round.((angle.(t) .+ π) .* (0.5 / π) * slm.SLM2π)
        phase = (angle.(t) ) .* (0.5 / π) * slm.SLM2π
        fourier = slm.A .* exp.(2im * π * phase / slm.SLM2π)

        image = fftshift(fft(fourier))
        trap = extract_locations(layout, image)
        trap_A = normalize!(abs.(trap))
        trap_ϕ_new = angle.(trap)

        if algorithm.verbose && (i % algorithm.show_every == 0)
            @show (trap_ϕ_new - trap_ϕ)[1:10]
        end
        if i < algorithm.niters * algorithm.ratio_fixphase
            trap_ϕ = trap_ϕ_new
        end

        if algorithm.verbose && (i % algorithm.show_every == 0)
            Atotal, stdmean = compute_cost(trap_A)
            print(@sprintf("step = %5d\ttotal amplitude = %.5f\tcost = %.3e\n", i, Atotal, stdmean))
            if stdmean < algorithm.tol
                break
            end
        end
    end

    t1 = time()
    Atotal, stdmean = compute_cost(trap_A)
    algorithm.verbose && print(@sprintf("time = %.2f,\ttotal amplitude = %.5f\nfcost = %.3e\n", (t1-t0)/algorithm.niters, Atotal, stdmean))
    return trap_A, trap_ϕ
end

"""
compute the cost from image amplitude.

Args:
    A (Array): the amplitude on target traps (not normalized),

Returns:
    * total amplitude
    * std/mean, the smaller, the more evenly distributed target traps are.
"""
function compute_cost(A::AbstractArray)
    t = abs2.(A)
    Atotal = sum(t)
    mean = Atotal / length(t)
    std_mean = std(t) / mean
    return Atotal, std_mean
end