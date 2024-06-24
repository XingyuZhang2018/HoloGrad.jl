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
    target_A_reweight = ones(layout.ntrap) / sqrt(layout.ntrap)
    match_image(layout, slm, target_A_reweight, algorithm)
end

function match_image(layout::Layout, slm::SLM, target_A_reweight, algorithm::WGS)
    # preloc 
    A, ϕ, SLM2π = slm.A, slm.ϕ, slm.SLM2π

    # initial image space information
    field = A .* exp.(ϕ * (2im * π / slm.SLM2π))
    X, Y, iX, iY = preloc_cft(layout, field)
    trap = cft(field, X, Y)
    trap_A = normalize!(abs.(trap))
    trap_ϕ = angle.(trap)
    
    ϕ = similar(slm.ϕ)

    # ------ Iterative Optimization Start --------
    algorithm.verbose && print("Start $algorithm")
    t0 = time()
    iters = 0
    for i in 1:algorithm.maxiter
        reweight = clamp!(mean(trap_A) ./ trap_A, 0.1, 10)
        target_A_reweight .*= reweight
        trap_ϕ_new = one_step!(A, ϕ, field, SLM2π, trap, trap_A, trap_ϕ, target_A_reweight, X, Y, iX, iY)
        if i < algorithm.maxiter * algorithm.ratio_fixphase
            trap_ϕ = trap_ϕ_new
        end

        iters += 1
        if i % algorithm.show_every == 0
            stdmean = compute_cost(trap_A)
            algorithm.verbose && print(@sprintf("step = %5d\tcost = %.3e\n", i, stdmean))
            if stdmean < algorithm.tol
                break
            end
        end
    end

    t1 = time()
    stdmean = compute_cost(trap_A)
    print(@sprintf("Finish WGS at %d iterations\n  total_time = %.2f s\tsingle_time = %.2f ms\tcost = %.3e\n", iters, (t1-t0), (t1-t0)/iters*1000, stdmean))

    return SLM(slm.A, ϕ, slm.SLM2π), stdmean, target_A_reweight
end

function one_step!(A, ϕ, field, SLM2π, trap, trap_A, trap_ϕ, target_A_reweight, X, Y, iX, iY)
    # image plane
    v_forced_trap = normalize!(target_A_reweight) .* exp.(1im * trap_ϕ) 

    # compute phase front at SLM
    t = icft(v_forced_trap, iX, iY)
    ϕ .= angle.(t).* (0.5 / π) * SLM2π
    field .= A .* exp.(2im * π * ϕ  / SLM2π)

    trap .= cft(field, X, Y)
    trap_A .= normalize!(abs.(trap))
    trap_ϕ_new = angle.(trap)

    return trap_ϕ_new
end