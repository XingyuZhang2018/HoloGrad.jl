function fixed_point_map(layout::Layout, slm::SLM, B)
    field = slm.A .* exp.(2im * π * slm.ϕ / slm.SLM2π)
    X, Y, iX, iY = preloc_cft(layout, field)
    trap = cft(field, X, Y)
    trap_ϕ = angle.(trap)
    trap_A = normalize(abs.(trap))
    reweight = mean(trap_A) ./ trap_A
    B = B .* reweight
    v_forced_trap = normalize(B) .* exp.(1im * trap_ϕ)
    t = icft(v_forced_trap, iX, iY)
    ϕ = (angle.(t) ) .* (0.5 / π) * slm.SLM2π
    return [mod1.(ϕ, slm.SLM2π), B]
end

"""
Get the time derivative of the phase in the SLM plane by the implicit function theorem.
"""
function get_dϕBdt(layout::Layout, slm::SLM, B, dxdt; atol=1e-12, maxiter=1)
    b = ForwardDiff.derivative(dt -> fixed_point_map(ContinuousLayout(layout.points + dxdt*dt), slm, B), 0.0)
    dϕBdt, info = linsolve(dϕBdt -> ((I(size(dϕBdt,1)) .* (1 + 1e-6)) * dϕBdt - ForwardDiff.derivative(dt -> fixed_point_map(layout, SLM(slm.A, slm.ϕ + dϕBdt[1] * dt, slm.SLM2π), B + dϕBdt[2] * dt), 0.0)), b; atol, maxiter)
    info.converged == 0 && @warn "dϕBdt not converged."
    return dϕBdt
end

function get_dϕdt(layout::Layout, slm::SLM, B, dxdt)
    b = ForwardDiff.derivative(dt -> fixed_point_map(ContinuousLayout(layout.points + dxdt*dt), slm, B), 0.0)
    dϕdt, info = linsolve(dϕdt -> ((I(size(dϕdt,1)) .* (1 + 1e-10)) * dϕdt - ForwardDiff.derivative(dt -> fixed_point_map(layout, SLM(slm.A, slm.ϕ + dϕdt * dt, slm.SLM2π), B), 0.0)), b)
    info.converged == 0 && @warn "dϕdt not converged."
    return dϕdt
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
function evolution_slm(layout::ContinuousLayout, layout_end::Layout, slm::SLM, algorithm; slices=5, interps=5, ifflow=true)

    points = layout.points
    Δx = (layout_end.points - layout.points) / slices # linear interpolation 
    dt = 1
    dxdt = Δx

    slm, cost, B = match_image(layout, slm, algorithm)
    dϕdt, dBdt = get_dϕBdt(layout, slm, B, dxdt)

    slms = [slm]
    layouts = [layout]
    for i in 1:slices
        println("Step $i/$slices")
        if ifflow
            t0 = time()
            slm_old = slm
            layout = ContinuousLayout(points + i*Δx)
            slm_new, cost, B = match_image(layout, slm, B, algorithm)
            dϕdt_new, dBdt = get_dϕBdt(layout, slm_new, B, dxdt)
            for j in 1:interps
                if j < interps / 2
                    ϕ_mix = slm_old.ϕ + dϕdt * j/interps/5
                else
                    ϕ_mix = slm_new.ϕ + dϕdt_new * (interps-j)/interps/5
                end
                slm = SLM(slm.A, ϕ_mix, slm.SLM2π)
                push!(slms, slm)
            end
            t1 = time()

            dϕdt = dϕdt_new
            slm = slm_new
            layout = ContinuousLayout(points + i*Δx)
            cost = compute_cost(slm, layout)
            print(@sprintf("Finish dϕdt \n  time = %.2f s, cost = %.3e\n", t1-t0, cost))
        else
            layout = ContinuousLayout(points + i*Δx)
            slm, cost, B = match_image(layout, slm, B, algorithm)
        end
        push!(layouts, layout)
    end
    
    return layouts, slms
end