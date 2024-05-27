function fixed_point_map(layout::Layout, slm::SLM, target_A_reweight, algorithm)
    field = slm.A .* exp.(2im * π * slm.ϕ / slm.SLM2π)
    ϵ = size(slm.A, 1) / size(layout.mask, 1)
    trap = ft(layout, field, ϵ, Val(algorithm.ft_method))
    trap_ϕ = angle.(trap)
    v_forced_trap = target_A_reweight .* exp.(1im * trap_ϕ)
    t = ift(layout, v_forced_trap, ϵ, Val(algorithm.ft_method))
    ϕ = (angle.(t) ) .* (0.5 / π) * slm.SLM2π
    return ϕ
end

"""
    get_dϕdt(slm::SLM, target_A_reweight, dAdt)

Get the time derivative of the phase in the SLM plan by the implicit function theorem.

"""
function get_dϕdt(layout::Layout, slm::SLM, target_A_reweight, dAdt, algorithm)
    ∂f∂A = jacobian(x -> fixed_point_map(layout, slm, x, algorithm), target_A_reweight)[1]
    ∂f∂ϕ = jacobian(x -> fixed_point_map(layout, SLM(slm.A, x, slm.SLM2π), target_A_reweight, algorithm), slm.ϕ)[1]

    return (I(size(∂f∂ϕ, 1)) - ∂f∂ϕ) \ (∂f∂A * dAdt)
end

function evolution_slm(layout::Layout, layout_new::Layout, slm::SLM, algorithm; iters=5, δ=1e-1, dt=1e-1)
    slm, cost, target_A_reweight = match_image(layout, layout_new, slm, 0.0, algorithm)

    layout_union, = deal_layout(layout, layout_new)
    dAdt = reshape(extract_locations(layout_union, (layout_new.mask - layout.mask)), prod(size(target_A_reweight)), )
    ϕ = slm.ϕ

    slms = [slm]
    for i in 1:iters
        @show i
        # slm, cost, target_A_reweight_new = match_image(layout, layout_new, slm, i/iters, abs.(target_A_reweight), algorithm)
        dϕdt = reshape(get_dϕdt(layout_union, slm, target_A_reweight, dAdt, algorithm), size(slm.ϕ))
        ϕ += dϕdt * dt
        slm = SLM(slm.A, ϕ, slm.SLM2π)
        push!(slms, slm)
        slm, cost, target_A_reweight = match_image(layout, layout_new, slm, i/iters, abs.(target_A_reweight + dAdt * dt), algorithm)
        @show target_A_reweight
    end
    
    return slms
end