function fixed_point_map(layout::Layout, slm::SLM, target_A_reweight)
    field = slm.A .* exp.(2im * π * slm.ϕ / slm.SLM2π)
    X, Y, iX, iY = preloc_cft(layout, field)
    trap = cft(field, X, Y)
    trap_ϕ = angle.(trap)
    v_forced_trap = target_A_reweight .* exp.(1im * trap_ϕ)
    t = icft(v_forced_trap, iX, iY)
    ϕ = (angle.(t) ) .* (0.5 / π) * slm.SLM2π
    return mod.(ϕ .- slm.SLM2π, slm.SLM2π) .+ slm.SLM2π
end

"""
Get the time derivative of the phase in the SLM plan by the implicit function theorem.
"""
function get_dϕdt(layout::Layout, slm::SLM, B, dBdt, dxdt; atol=1e-12, maxiter=1)
    b = ForwardDiff.derivative(dt -> fixed_point_map(layout, slm, B + dBdt*dt), 0.0) + ForwardDiff.derivative(dt -> fixed_point_map(ContinuousLayout(layout.points + dxdt*dt), slm, B), 0.0)
    dϕdt, info = linsolve(dϕdt -> ((I(size(dϕdt,1)) .* (1 + 1e-10)) * dϕdt - ForwardDiff.derivative(dt -> fixed_point_map(layout, SLM(slm.A, slm.ϕ + dϕdt * dt, slm.SLM2π), B), 0.0)), b; atol, maxiter)
    info.converged == 0 && @warn "dϕdt not converged."
    return dϕdt
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
function evolution_slm(layout::ContinuousLayout, layout_end::Layout, slm::SLM, algorithm; iters=5)
    dt=1/iters # linear interpolation 
    slm, cost, target_A_reweight = match_image(layout, slm, algorithm)
    points = layout.points
    
    dxdt = (layout_end.points - layout.points) / iters # linear interpolation 

    slms = [slm]
    layouts = [layout]
    for i in 1:iters
        println("Step $i/$iters")
        layout = ContinuousLayout(points + (i-1)*dxdt)
        slm, cost, target_A_reweight = match_image(layout, slm, target_A_reweight, algorithm)

        dBdt = -target_A_reweight # without intermidiate target_A_reweight
        t0 = time()
        dϕdt = get_dϕdt(layout, slm, target_A_reweight, dBdt, dxdt)
        t1 = time()
        print(@sprintf("Finish dϕdt time = %.2f s\n", t1-t0))

        slm = SLM(slm.A, slm.ϕ + dϕdt * dt, slm.SLM2π)
        push!(slms, slm)
        push!(layouts, layout)
    end
    
    return layouts, slms
end