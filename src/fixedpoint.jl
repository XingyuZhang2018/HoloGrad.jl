function fixed_point_map(slm::SLM, target_A_reweight)
    field = slm.A .* exp.(2im * π * slm.ϕ / slm.SLM2π)
    image = fftshift(fft(field))
    trap_ϕ = angle.(image)
    v_forced_trap = target_A_reweight .* exp.(1im * trap_ϕ)
    t = ifft(ifftshift(v_forced_trap))
    ϕ = (angle.(t) ) .* (0.5 / π) * slm.SLM2π
    return ϕ
end

"""
    get_dϕdt(slm::SLM, target_A_reweight, dAdt)

Get the time derivative of the phase in the SLM plan by the implicit function theorem.

"""
function get_dϕdt(slm::SLM, target_A_reweight, dAdt)
    ∂f∂A = jacobian(x -> fixed_point_map(slm, x), target_A_reweight)[1]
    ∂f∂ϕ = jacobian(x -> fixed_point_map(SLM(slm.A, x, slm.SLM2π), target_A_reweight), slm.ϕ)[1]

    return (I(size(∂f∂ϕ, 1)) - ∂f∂ϕ) \ (∂f∂A * dAdt)
end

function evolution_slm(layout::Layout, layout_new::Layout, slm::SLM, algorithm, δ=1e-5, dt=1e-5)
    slm, cost, target_A_reweight = match_image(layout, slm, algorithm)
    dAdt = reshape(δ * (layout_new.mask - layout.mask), prod(size(slm.A)), 1)
    target_A_reweight = embed_locations(layout, target_A_reweight)
    ϕ = slm.ϕ

    for i in 1:10
        target_A_reweight += reshape(dAdt, size(slm.A)) * dt
        dϕdt = reshape(get_dϕdt(slm, target_A_reweight, dAdt), size(slm.ϕ))
        # ϕ = fixed_point_map(slm, normalize!(target_A_reweight))
        ϕ += dϕdt * dt
        # @show norm(dϕdt)
        if norm(fixed_point_map(SLM(slm.A, ϕ, slm.SLM2π), target_A_reweight) - ϕ) > 1e-5
            break
        end
    end
    slm = SLM(slm.A, ϕ, slm.SLM2π)
    return slm
end