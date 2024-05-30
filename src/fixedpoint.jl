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
    return ϕ
end

"""
    get_dϕdt(slm::SLM, target_A_reweight, dAdt)

Get the time derivative of the phase in the SLM plan by the implicit function theorem.

"""
function get_dϕdt(layout::Layout, slm::SLM, target_A_reweight, dAdt, algorithm)
    dfdt = ForwardDiff.derivative(dt -> fixed_point_map(layout, slm, target_A_reweight + dAdt*dt, algorithm), 0.0)
    dϕdt, info = linsolve(dϕdt -> ((I(size(dϕdt,1)) .* (1 + 1e-10)) * dϕdt - ForwardDiff.derivative(dt -> fixed_point_map(layout, SLM(slm.A, slm.ϕ + dϕdt * dt, slm.SLM2π), target_A_reweight, algorithm), 0.0)), dfdt)
    # info.converged == 0 && error("dϕdt not converged")

    return dϕdt
end

function evolution_slm(layout::GridLayout, layout_new::Layout, slm::SLM, algorithm; iters=5, δ=1e-1, dt=1e-1)
    slm, cost, target_A_reweight = match_image(layout, layout_new, slm, 0.0, algorithm)

    layout_union, = deal_layout(layout, layout_new)
    dAdt = reshape(extract_locations(layout_union, (layout_new.mask - layout.mask)), prod(size(target_A_reweight)), )
    ϕ = slm.ϕ

    slms = [slm]
    for i in 1:iters
        @show i
        # slm, cost, target_A_reweight_new = match_image(layout, layout_new, slm, i/iters, abs.(target_A_reweight), algorithm)
        dϕdt = get_dϕdt(layout_union, slm, target_A_reweight, dAdt, algorithm)
        ϕ += dϕdt * dt
        slm = SLM(slm.A, ϕ, slm.SLM2π)
        push!(slms, slm)
        slm, cost, target_A_reweight = match_image(layout, layout_new, slm, i/iters, abs.(target_A_reweight + dAdt * dt), algorithm)
        @show target_A_reweight
    end
    
    return slms
end

function evolution_slm(layout::ContinuousLayout, layout_end::ContinuousLayout, slm::SLM, algorithm; iters=5)
    dt=1/iters
    points = layout.points
    diff = (layout_end.points - layout.points) / iters
    layout_new = ContinuousLayout(layout.points + diff)
    slm, cost, target_A_reweight = match_image(layout, layout_new, slm, 0.0, algorithm)

    slms = [slm]
    layouts = [layout]
    for i in 1:iters
        layout = ContinuousLayout(points + (i-1)*diff)
        layout_new = ContinuousLayout(points + i*diff)
        layout_union, layout_disappear, layout_appear = deal_layout(layout, layout_new)
        disapperindex = indexin(layout_disappear.points, layout_union.points)
        apperindex = indexin(layout_appear.points, layout_union.points)

        slm, cost, target_A_reweight = match_image(layout, layout_new, slm, 0.0, target_A_reweight, algorithm)

        dAdt = zeros(size(target_A_reweight))
        dAdt[apperindex] .= 1
        dAdt[disapperindex] .= -target_A_reweight[disapperindex]

        dϕdt = get_dϕdt(layout_union, slm, target_A_reweight, dAdt, algorithm)
        # @show dϕdt
        slm = SLM(slm.A, slm.ϕ + dϕdt * dt, slm.SLM2π)
        push!(slms, slm)
        push!(layouts, layout_new)
    end
    
    return layouts, slms
end