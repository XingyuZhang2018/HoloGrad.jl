using qgh
using CairoMakie
using Random

begin # slms_exact data
    Random.seed!(42)

    atype = Array # or CuArray
    N = 10

    layout = atype(ContinuousLayout([0.5 0.5]))
    layout_end = atype(ContinuousLayout([0.5 0.6]))
    slm = atype(SLM(N))

    algorithm = WGS(maxiter=100, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.9)

    #benchmark
    layouts_exact, slms_exact = evolution_slm_direct(layout, layout_end, slm, algorithm; keypoints=1)
end

begin # plot exact ϕ and B
    image_resolution = N

    fig = Figure(size = (480, 400))
    ax11 = Axis(fig[1, 1], aspect = DataAspect(), title = L"\phi_0")
    hm = CairoMakie.heatmap!(ax11, 1:N, 1:N, slms_exact[1].ϕ / slms_exact[1].SLM2π * 2π, colorrange = (-pi, pi))
    hidedecorations!(ax11)
    Colorbar(fig[1, 1][1, 2], hm, ticks = (-pi:pi/2:pi, [L"-\pi", L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}", L"\pi"]), ticklabelsize = 12)

    ax12 = Axis(fig[1, 2], aspect = DataAspect(), title = L"B_0")
    hm = qgh.heatmap!(ax12, slms_exact[1], image_resolution)
    Colorbar(fig[1, 2][1, 2], hm, ticklabelsize = 12)

    ax21 = Axis(fig[2, 1], aspect = DataAspect(), title = L"\phi_1")
    hm = CairoMakie.heatmap!(ax21, 1:N, 1:N, slms_exact[end].ϕ / slms_exact[end].SLM2π * 2π, colorrange = (-pi, pi))
    hidedecorations!(ax21)
    Colorbar(fig[2, 1][1, 2], hm, ticks = (-pi:pi/2:pi, [L"-\pi", L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}", L"\pi"]), ticklabelsize = 12)

    ax22 = Axis(fig[2, 2], aspect = DataAspect(), title = L"B_1")
    hm = qgh.heatmap!(ax22, slms_exact[end], image_resolution)
    Colorbar(fig[2, 2][1, 2], hm, ticklabelsize = 12)

    # display(fig)
    save("project/one_direction_move/one_point_phi_B.pdf", fig)
end

begin # plot flow implicit
    layouts_flow, slms_flow_implicit = qgh.evolution_slm_pure_flow(layout, layout_end, slm, algorithm; 
                                                          interps=1, 
                                                          aditers=5,
                                                          ifimplicit=true)

    fig = Figure(size = (580, 150))

    ax11 = Axis(fig[1, 1], aspect = DataAspect(), title = L"\Delta \phi_{\mathrm{exact}}")
    Δϕ_exact = qgh.ϕdiff(slms_exact[end], slms_exact[1])
    hm = CairoMakie.heatmap!(ax11, 1:N, 1:N, Δϕ_exact, colorrange = (-pi, pi))
    hidedecorations!(ax11)
    Colorbar(fig[1, 1][1, 2], hm, ticks = (-pi:pi/2:pi, [L"-\pi", L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}", L"\pi"]))

    ax12 = Axis(fig[1, 2], aspect = DataAspect(), title = L"\Delta \phi_{\mathrm{flow}}")
    Δϕ_flow = qgh.ϕdiff(slms_flow_implicit[end], slms_flow_implicit[1]) 
    hm = CairoMakie.heatmap!(ax12, 1:N, 1:N, Δϕ_flow, colorrange = (-pi, pi))
    hidedecorations!(ax12)
    Colorbar(fig[1, 2][1, 2], hm, ticks = (-pi:pi/2:pi, [L"-\pi", L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}", L"\pi"]))

    ax13 = Axis(fig[1, 3], aspect = DataAspect(), title = L"\Delta \phi_{\mathrm{exact}} - \Delta \phi_{\mathrm{flow}}")
    hm = CairoMakie.heatmap!(ax13, 1:N, 1:N, Δϕ_exact - Δϕ_flow, colorrange = (-2e-6, 2e-6))
    hidedecorations!(ax13)
    Colorbar(fig[1, 3][1, 2], hm, tickformat =  "{:.0e}")

    save("project/one_direction_move/one_point_Delta_phi_implicit.pdf", fig)
end

begin # plot flow without implicit
    layouts_flow, slms_flow = qgh.evolution_slm_pure_flow(layout, layout_end, slm, algorithm; 
                                                          interps=1, 
                                                          aditers=1,
                                                          ifimplicit=false)

    fig = Figure(size = (1400, 400))

    ax11 = Axis(fig[1, 1], aspect = DataAspect(), title = L"\Delta \phi_{\mathrm{exact}}")
    Δϕ_exact = qgh.ϕdiff(slms_exact[end], slms_exact[1])
    hm = CairoMakie.heatmap!(ax11, 1:N, 1:N, Δϕ_exact, colorrange = (-pi, pi))
    hidedecorations!(ax11)
    Colorbar(fig[1, 1][1, 2], hm, ticks = (-pi:pi/2:pi, [L"-\pi", L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}", L"\pi"]))

    ax12 = Axis(fig[1, 2], aspect = DataAspect(), title = L"\Delta \phi_{\mathrm{flow}}")
    Δϕ_flow = qgh.ϕdiff(slms_flow[end], slms_flow[1]) 
    hm = CairoMakie.heatmap!(ax12, 1:N, 1:N, Δϕ_flow, colorrange = (-pi, pi))
    hidedecorations!(ax12)
    Colorbar(fig[1, 2][1, 2], hm, ticks = (-pi:pi/2:pi, [L"-\pi", L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}", L"\pi"]))

    ax13 = Axis(fig[1, 3], aspect = DataAspect(), title = L"\Delta \phi_{\mathrm{exact}} - \Delta \phi_{\mathrm{flow}}")
    hm = CairoMakie.heatmap!(ax13, 1:N, 1:N, Δϕ_exact - Δϕ_flow, colorrange = (-2e-6, 2e-6))
    hidedecorations!(ax13)
    Colorbar(fig[1, 3][1, 2], hm, tickformat =  "{:.2e}")

    save("project/one_direction_move/one_point_Delta_phi.pdf", fig)
end