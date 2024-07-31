function fixed_point_map(layout::Layout, slm::SLM, B)
    field = slm.A .* exp.(2im * π * slm.ϕ / slm.SLM2π)
    X, Y, iX, iY = preloc_cft(layout, field)
    trap = cft(field, X, Y)
    trap_ϕ = angle.(trap)
    trap_A = abs.(trap)
    reweight = mean(trap_A) ./ trap_A
    B = B .* reweight
    v_forced_trap = B .* exp.(1im * trap_ϕ)
    t = icft(v_forced_trap, iX, iY)
    ϕ = ((angle.(t) ) .* (0.5 / π) * slm.SLM2π)
    return [mod.(ϕ .+ slm.SLM2π/2, slm.SLM2π) .- slm.SLM2π/2, B]
end

"""
Get the time derivative of the phase in the SLM plane by the implicit function theorem.
"""
function get_dϕBdt(layout::Layout, slm::SLM, B, dxdt; atol=1e-12, maxiter=1, ϵ=1e-15)
    b = ForwardDiff.derivative(dt -> fixed_point_map(ContinuousLayout(layout.points + dxdt*dt), slm, B), 0.0)
    dϕBdt, info = linsolve(dϕBdt -> ((I(size(dϕBdt,1)) .* (1 + ϵ)) * dϕBdt - ForwardDiff.derivative(dt -> fixed_point_map(layout, SLM(slm.A, slm.ϕ + dϕBdt[1] * dt, slm.SLM2π), B + dϕBdt[2] * dt), 0.0)), b; atol, maxiter)
    info.converged == 0 && @warn "dϕBdt not converged."
    return dϕBdt
end

function get_dϕdt(layout::Layout, slm::SLM, B, dxdt; ϵ=1e-15)
    b = ForwardDiff.derivative(dt -> fixed_point_map(ContinuousLayout(layout.points + dxdt*dt), slm, B), 0.0)
    dϕdt, info = linsolve(dϕdt -> ((I(size(dϕdt,1)) .* (1 + ϵ)) * dϕdt - ForwardDiff.derivative(dt -> fixed_point_map(layout, SLM(slm.A, slm.ϕ + dϕdt * dt, slm.SLM2π), B), 0.0)), b)
    info.converged == 0 && @warn "dϕdt not converged."
    return dϕdt
end

"""
    Get the time derivative of the phase in the SLM plane by iterations.
"""
function get_dϕBdt(layout::Layout, slm::SLM, B, dxdt, iters::Int=5)
    function f(dt)
        layout = ContinuousLayout(layout.points + dxdt*dt)
        for _ in 1:iters
            ϕ, B = fixed_point_map(layout, slm, B)
            slm = SLM(slm.A, ϕ, slm.SLM2π)
        end
        return slm.ϕ, B
    end
    
    dϕdt = ForwardDiff.derivative(dt -> f(dt)[1], 0.0)
    dBdt = ForwardDiff.derivative(dt -> f(dt)[2], 0.0)
    return dϕdt, ForwardDiff.value.(dBdt)
end

function optimize_dt(layout::Layout, slm::SLM, dϕdt, dxdt; 
                     f_tol=1e-6, 
                     iters=100, 
                     show_every=10,
                     verbose=false
                     )
    loss(dt) = compute_cost(SLM(slm.A, slm.ϕ + dϕdt * dt[1], slm.SLM2π), ContinuousLayout(layout.points + dxdt*dt[1]))
    loss_g(dt) = Zygote.gradient(loss, dt[1])
    # loss_g(dt) = ForwardDiff.derivative(loss, abs(dt[1]))

    res = optimize(loss, loss_g, 
                   [rand()], LBFGS(m = 20), inplace = false,
                   Optim.Options(f_tol=f_tol, iterations=iters,
                   extended_trace=true,
                   callback=os->writelog(os, iters, show_every, verbose)),
                   )
    return res.minimizer[1]
end

function optimize_dt(layout::Layout, slm::SLM, dϕdt; 
                     f_tol=1e-6, 
                     iters=100, 
                     show_every=10,
                     verbose=false
                     )
    loss(dt) = compute_cost(SLM(slm.A, slm.ϕ + dϕdt * dt[1], slm.SLM2π), layout)[1]
    loss_g(dt) = Zygote.gradient(loss, dt[1])
    # loss_g(dt) = ForwardDiff.derivative(loss, abs(dt[1]))

    res = optimize(loss, loss_g, 
                   [1.0], LBFGS(m = 20), inplace = false,
                   Optim.Options(f_tol=f_tol, iterations=iters,
                   extended_trace=true,
                   callback=os->writelog(os, iters, show_every, verbose)),
                   )
    return res.minimizer[1]
end

function writelog(os::OptimizationState, iters, show_every, verbose)
    message = "$(round(os.metadata["time"],digits=1))    $(os.iteration)    $(round(os.value,digits=8))    $(round(os.g_norm,digits=8))\n"

    if verbose && (iters % show_every == 0)
        printstyled(message; bold=true, color=:blue)
        flush(stdout)
    end

    return false
end

function get_flow_params(slm, layout::Layout, layout_end::Layout, α, β)
    Nx, Ny = size(slm.A)
    one_piexl = 1 / Nx
    interval = layout_end.points - layout.points
    interval_norm = [norm(interval[i, :]) for i in 1:size(interval, 1)]
    interval_max, index = findmax(interval_norm)
    ratio = one_piexl / interval_max * α 
    dxdt = ratio * interval * β
    Δx = interval * ratio  
    keypoints = ceil(Int, 1/ratio)
    end_interps_ratio = mod(1/ratio, 1)
    return dxdt, keypoints, Δx, end_interps_ratio
end

"""
evolution slm by matching the image of layout and slm.

Args:
    layout (Layout): the layout of traps.
    layout_end (Layout): the layout of traps at the end.
    slm (SLM): the initial SLM.
    algorithm (Algorithm): the algorithm to match the image.
    
Keyword Args:
    interps (Int): the number of interpolation between two ket points.
    aditers (Int): the number of iterations to get the time derivative of the phase in the SLM plane.
    ifimplicit (Bool): if use the implicit function theorem to get the time derivative of the phase in the SLM plane.
    α (Float): the ratio decides the distance between two keypoints.
    β (Float): the speed of the flow.

Returns:
    layouts (Array{Layout}): the layouts of traps.
    slms (Array{SLM}): the SLMs.
"""
function evolution_slm_flow(layout::ContinuousLayout, layout_end::Layout, slm::SLM, algorithm;
                            interps=5, 
                            aditers=10,
                            ifimplicit=false,
                            α=0.5,
                            β=1.0
                            )
    
    dxdt, keypoints, Δx, end_interps_ratio = get_flow_params(slm, layout, layout_end, α, β)
    points = layout.points
    dt = 1

    slm, cost, B = match_image(layout, slm, algorithm)
    slm_new, cost, B_new = match_image(layout_end, slm, copy(B), algorithm)

    if ifimplicit
        dϕdt, dBdt = get_dϕBdt(layout, slm, B, dxdt) # by implicit function theorem
    else
        dϕdt, dBdt = get_dϕBdt(layout, slm, B, dxdt, aditers)
    end

    slms = [slm]
    layouts = [layout]
    
    for i in 1:keypoints
        println("keypoints Step $i/$keypoints")
 
        layout_new = ContinuousLayout(points + i*Δx)
        slm_new, cost, B = match_image(layout_new, slm, B, algorithm)

        t0 = time()
        if ifimplicit
            dϕdt_new, dBdt_new = get_dϕBdt(layout_new, slm_new, B, dxdt) # by implicit function theorem
        else
            dϕdt_new, dBdt_new = get_dϕBdt(layout_new, slm_new, B, dxdt, aditers)
        end
        t1 = time()
        print(@sprintf("    Finish dϕdt time = %.2f s\n", t1-t0))

        _, trap_A_mean = compute_cost(slm, layout)
        _, trap_A_mean_new = compute_cost(slm_new, layout_new)

        for j in 1:interps
            layout_interp = ContinuousLayout(layout_new.points - (interps - j) * Δx / interps)
            if j <= interps/2
                slm_interp = SLM(slm.A, slm.ϕ + j * dϕdt * dt/interps, slm.SLM2π)
                variance, decay_rate = compute_cost(slm_interp, layout_interp, trap_A_mean)
            else
                slm_interp = SLM(slm_new.A, slm_new.ϕ - (interps - j) * dϕdt_new * dt/interps, slm_new.SLM2π)
                variance, decay_rate = compute_cost(slm_interp, layout_interp, trap_A_mean_new)
            end

            print(@sprintf("\tinterpolation Step %d/%d: A variance = %.3e decay rate = %.8f\n", j, interps, variance, decay_rate))

            push!(slms, slm_interp)
            push!(layouts, layout_interp)
            i == keypoints && j >= interps * end_interps_ratio && break
        end

        slm = slm_new
        dϕdt = dϕdt_new
        layout = layout_new
    end
    
    return layouts, slms
end

"""
evolution slm by matching the image of layout and slm using flow only.

Args:
    layout (Layout): the layout of traps.
    layout_end (Layout): the layout of traps at the end.
    slm (SLM): the initial SLM.
    algorithm (Algorithm): the algorithm to match the image.
    
Keyword Args:
    interps (Int): the number of interpolation between two ket points.
    aditers (Int): the number of iterations to get the time derivative of the phase in the SLM plane.
    ifimplicit (Bool): if use the implicit function theorem to get the time derivative of the phase in the SLM plane.

Returns:
    layouts (Array{Layout}): the layouts of traps.
    slms (Array{SLM}): the SLMs.
"""
function evolution_slm_pure_flow(layout::ContinuousLayout, layout_end::Layout, slm::SLM, algorithm;
                                 interps=5, 
                                 aditers=10,
                                 ifimplicit=false,
                                 )

    Δx = layout_end.points - layout.points
    dxdt = Δx
    dt = 1

    t0 = time()
    slm, cost, B = match_image(layout, slm, algorithm)
    t1 = time()
    print(@sprintf("Finish wgs time = %.2f s\n", t1-t0))

    t0 = time()
    if ifimplicit
        dϕdt, dBdt = get_dϕBdt(layout, slm, B, dxdt) # by implicit function theorem
    else
        dϕdt, dBdt = get_dϕBdt(layout, slm, B, dxdt, aditers)
    end
    t1 = time()
    print(@sprintf("Finish dϕdt time = %.2f s\n", t1-t0))
    slms = [slm]
    layouts = [layout]
    
    _, trap_A_mean = compute_cost(slm, layout)
    for j in 1:interps
        layout_interp = ContinuousLayout(layout.points + j * Δx / interps)
        slm_interp = SLM(slm.A, slm.ϕ + j * dϕdt * dt/interps, slm.SLM2π)
        variance, decay_rate = compute_cost(slm_interp, layout_interp, trap_A_mean)

        print(@sprintf("\tinterpolation Step %d/%d: A variance = %.3e decay rate = %.8f\n", j, interps, variance, decay_rate))

        push!(slms, slm_interp)
        push!(layouts, layout_interp)
    end

    return layouts, slms
end