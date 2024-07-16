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
    return [mod.(ϕ, slm.SLM2π), B]
end

"""
Get the time derivative of the phase in the SLM plane by the implicit function theorem.
"""
function get_dϕBdt(layout::Layout, slm::SLM, B, dxdt; atol=1e-12, maxiter=1)
    b = ForwardDiff.derivative(dt -> fixed_point_map(ContinuousLayout(layout.points + dxdt*dt), slm, B), 0.0)
    dϕBdt, info = linsolve(dϕBdt -> ((I(size(dϕBdt,1)) .* (1 + 1e-10)) * dϕBdt - ForwardDiff.derivative(dt -> fixed_point_map(layout, SLM(slm.A, slm.ϕ + dϕBdt[1] * dt, slm.SLM2π), B + dϕBdt[2] * dt), 0.0)), b; atol, maxiter)
    info.converged == 0 && @warn "dϕBdt not converged."
    return dϕBdt
end

function get_dϕdt(layout::Layout, slm::SLM, B, dxdt)
    b = ForwardDiff.derivative(dt -> fixed_point_map(ContinuousLayout(layout.points + dxdt*dt), slm, B), 0.0)
    dϕdt, info = linsolve(dϕdt -> ((I(size(dϕdt,1)) .* (1 + 1e-10)) * dϕdt - ForwardDiff.derivative(dt -> fixed_point_map(layout, SLM(slm.A, slm.ϕ + dϕdt * dt, slm.SLM2π), B), 0.0)), b)
    info.converged == 0 && @warn "dϕdt not converged."
    return dϕdt
end

"""
    Get the time derivative of the phase in the SLM plane by iterations.
"""
function get_dϕdt(layout::Layout, slm::SLM, B, dxdt, iters::Int=5)
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

function writelog(os::OptimizationState, iters, show_every, verbose)
    message = "$(round(os.metadata["time"],digits=1))    $(os.iteration)    $(round(os.value,digits=8))    $(round(os.g_norm,digits=8))\n"

    if verbose && (iters % show_every == 0)
        printstyled(message; bold=true, color=:blue)
        flush(stdout)
    end

    return false
end

"""
evolution slm by matching the image of layout and slm.

Args:
    layout (Layout): the layout of traps.
    layout_end (Layout): the layout of traps at the end.
    slm (SLM): the initial SLM.
    algorithm (Algorithm): the algorithm to match the image.
    
Keyword Args:
    iters (Int): the number of interpolation between layout and layout_end.

Returns:
    layouts (Array{Layout}): the layouts of traps.
    slms (Array{SLM}): the SLMs.

Currently dx/dt and the time step are linearly fixed.
"""
function evolution_slm(layout::ContinuousLayout, layout_end::Layout, slm::SLM, algorithm; 
                        slices=5,
                        interps=5, 
                        aditers=10,
                        ifflow=true,
                        ifimplicit=false
                        )

    points = layout.points
    Δx = (layout_end.points - layout.points) / slices # linear interpolation 
    dt = 1
    dxdt = Δx

    slm, cost, B = match_image(layout, slm, algorithm)
    dϕdt, dBdt = get_dϕdt(layout, slm, B, dxdt, aditers)

    slms = [slm]
    layouts = [layout]
    for i in 1:slices
        println("Slices Step $i/$slices")
        if ifflow    
            layout_new = ContinuousLayout(points + i*Δx)
            slm_new, cost, B = match_image(layout_new, slm, B, algorithm)

            t0 = time()
            if ifimplicit
                dϕdt_new, dBdt_new = get_dϕBdt(layout, slm, B, dxdt) # by implicit function theorem
            else
                dϕdt_new, dBdt_new = get_dϕdt(layout_new, slm_new, B, dxdt)
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
            end

            slm = slm_new
            dϕdt = dϕdt_new
            layout = layout_new
        else
            layout = ContinuousLayout(points + i*Δx)
            slm, cost, B = match_image(layout, slm, B, algorithm)
            push!(slms, slm)
            push!(layouts, layout)
        end
        
    end
    
    return layouts, slms
end