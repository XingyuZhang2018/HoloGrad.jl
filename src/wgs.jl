@with_kw struct WGS
    ratio_fixphase::Float64 = 0.8 # fix phase when `number of iteration > ratio_fixphase * maxiter`.
    show_every::Int = 10 # show the cost every `show_every` iterations.
    maxiter::Int = Defaults.maxiter # the number of iterations.
    tol::Float64 = Defaults.tol # the tolerance of the cost.
    verbose::Bool = Defaults.verbose # if print the cost every `show_every` iterations.
end

"""
    The fixed phase WGS algorithm.

    Args:

        layout (Layout): specify the layout in the image plane, it can be an instance of
            * GridLayout, a grid specified by a set of x locations, y locations and a mask,
            * PointLayout, a set of points specified by x locations and y locations.

        slm (Tensor): the (fixed) normalized amplitude and the phase in the SLM plane,
        algorithm (WGS): the algorithm parameters.

    Returns:
        SLM: the optimized SLM.
        cost: the cost of the optimized SLM.
        target_A_reweight: the reweighted target amplitude.

    References:
        https://www.osapublishing.org/ol/abstract.cfm?uri=ol-44-12-3178
"""
function match_image(layout::Layout, slm::SLM, algorithm::WGS)
    # initial image space information
    field = slm.A .* exp.(1im * slm.ϕ)
    image = fftshift(fft(field))
    trap = extract_locations(layout, image)
    trap_A = normalize!(abs.(trap))
    trap_ϕ = angle.(trap)
    target_A_reweight = ones(layout.ntrap) / sqrt(layout.ntrap)
    ϕ = similar(slm.ϕ)

    # ------ Iterative Optimization Start --------
    print("Start $algorithm")
    t0 = time()
    for i in 1:algorithm.maxiter
        target_A_reweight .*= clamp!(mean(trap_A) ./ trap_A, 0.1, 10)

        # image plane
        v_forced_trap = normalize(target_A_reweight) .* exp.(1im * trap_ϕ) # regenerate image
        v_forced_image = embed_locations(layout, v_forced_trap)
    
        # compute phase front at SLM
        t = ifft(ifftshift(v_forced_image))
        # ϕ = round.((angle.(t) .+ π) .* (0.5 / π) * slm.SLM2π)
        ϕ = (angle.(t) ) .* (0.5 / π) * slm.SLM2π
        fourier = slm.A .* exp.(2im * π * ϕ / slm.SLM2π)

        image = fftshift(fft(fourier))
        trap = extract_locations(layout, image)
        trap_A = normalize!(abs.(trap))

        if i < algorithm.maxiter * algorithm.ratio_fixphase
            trap_ϕ = angle.(trap)
        end

        if algorithm.verbose && (i % algorithm.show_every == 0)
            stdmean = compute_cost(trap_A)
            print(@sprintf("step = %5d\tcost = %.3e\n", i, stdmean))
            if stdmean < algorithm.tol
                break
            end
        end
    end

    t1 = time()
    stdmean = compute_cost(trap_A)
    print(@sprintf("Finish WGS\n  time = %.2f\tcost = %.3e\n", (t1-t0)/algorithm.maxiter, stdmean))

    return SLM(slm.A, ϕ, slm.SLM2π), stdmean, target_A_reweight
end