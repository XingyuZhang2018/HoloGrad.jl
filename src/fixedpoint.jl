function fixed_point_map(layout::Layout, slm::SLM, target_A_reweight)
    field = slm.A .* exp.(2im * π * slm.ϕ / slm.SLM2π)
    image = fftshift(fft(field))
    trap = extract_locations(layout, image)
    trap_ϕ = angle.(trap)
    v_forced_trap = target_A_reweight .* exp.(1im * trap_ϕ)
    v_forced_image = embed_locations(layout, v_forced_trap)
    t = ifft(ifftshift(v_forced_image))
    ϕ = (angle.(t) ) .* (0.5 / π) * slm.SLM2π
    return ϕ
end