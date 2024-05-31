function fixed_point_map(layout::Layout, slm::SLM, target_A_reweight, algorithm)
    field = slm.A .* exp.(2im * π * slm.ϕ / slm.SLM2π)
    if isa(layout, GridLayout)
        ϵ = size(slm.A, 1) / size(layout.mask, 1)
    else
        ϵ = 1
    end
    trap = ft(layout, field, ϵ, Val(algorithm.ft_method))
    trap_ϕ = angle.(trap)
    v_forced_trap = target_A_reweight .* exp.(1im * trap_ϕ)
    if isa(layout, GridLayout)
        t = ift(layout, v_forced_trap, ϵ, Val(algorithm.ft_method))
    else
        t = ift(layout, v_forced_trap, size(slm.A)..., Val(algorithm.ft_method))
    end
    ϕ = (angle.(t) ) .* (0.5 / π) * slm.SLM2π
    return mod.(ϕ .- slm.SLM2π, slm.SLM2π) .+ slm.SLM2π
end

"""
Get the time derivative of the phase in the SLM plan by the implicit function theorem.
"""
function get_dϕdt(layout::Layout, slm::SLM, B, dBdt, dxdt, algorithm)
    b = ForwardDiff.derivative(dt -> fixed_point_map(layout, slm, B + dBdt*dt, algorithm), 0.0) + ForwardDiff.derivative(dt -> fixed_point_map(ContinuousLayout(layout.points + dxdt*dt), slm, B, algorithm), 0.0)
    dϕdt, info = linsolve(dϕdt -> ((I(size(dϕdt,1)) .* (1 + 1e-10)) * dϕdt - ForwardDiff.derivative(dt -> fixed_point_map(layout, SLM(slm.A, slm.ϕ + dϕdt * dt, slm.SLM2π), B, algorithm), 0.0)), b)
    # info.converged == 0 && error("dϕdt not converged")
    return dϕdt
end

function evolution_slm(layout::ContinuousLayout, layout_end::Layout, slm::SLM, algorithm; iters=5)
    dt=1/iters
    slm, cost, target_A_reweight = match_image(layout, slm, algorithm)
    points = layout.points
    diff = (layout_end.points - layout.points) / iters


    slms = [slm]
    layouts = [layout]
    for i in 1:iters
        println("Step $i/$iters")
        layout = ContinuousLayout(points + (i-1)*diff)
        slm, cost, target_A_reweight = match_image(layout, slm, target_A_reweight, algorithm)
        # layout_new = ContinuousLayout(points + i*diff)
        # slm_new, _, target_A_reweight_new = match_image(layout_new, slm, copy(target_A_reweight), algorithm)

        dBdt = -target_A_reweight
        dxdt = diff
        dϕdt = get_dϕdt(layout, slm, target_A_reweight, dBdt, dxdt, algorithm)
        # @show dϕdt slm_new.ϕ-slm.ϕ
        slm = SLM(slm.A, slm.ϕ + dϕdt * dt, slm.SLM2π)
        push!(slms, slm)
        push!(layouts, layout)
    end
    
    return layouts, slms
end