using qgh
using CairoMakie
using Random
using LinearAlgebra

begin # slms_exact data
    Random.seed!(42)

    atype = Array # or Array
    N = 100

    points = [0.4 0.4; 0.4 0.6; 0.6 0.4; 0.6 0.6]
    points_end = [0.4 0.5; 0.4 0.7; 0.6 0.5; 0.6 0.7]
    
    layout = atype(ContinuousLayout(points))
    layout_end = atype(ContinuousLayout(points_end))
    slm = atype(SLM(N))

    algorithm = WGS(maxiter=1000, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.9)

    #benchmark
    layouts_exact, slms_exact = evolution_slm_direct(layout, layout_end, slm, algorithm; keypoints=1)
end

begin # plot exact ϕ and B
    image_resolution = N

    fig = Figure(size = (480, 200))

    ax12 = Axis(fig[1, 1], aspect = DataAspect(), title = L"B_0")
    hm = qgh.heatmap!(ax12, slms_exact[1], image_resolution)
    Colorbar(fig[1, 1][1, 2], hm)

    ax22 = Axis(fig[1, 2], aspect = DataAspect(), title = L"B_1")
    hm = qgh.heatmap!(ax22, slms_exact[end], image_resolution)
    Colorbar(fig[1, 2][1, 2], hm)

    # display(fig)
    save("project/A. Movement in One Direction/one_direction_move/more_points_B.pdf", fig)
end

begin # plot flow implicit
    layouts_flow, slms_flow_implicit = qgh.evolution_slm_pure_flow(layout, layout_end, slm, algorithm; 
                                                          interps=1, 
                                                          aditers=5,
                                                          ifimplicit=true)

    fig = Figure(size = (500, 200))

    ax11 = Axis(fig[1, 1], aspect = DataAspect(), title = L"\Delta \phi_{\mathrm{WGS}}")
    Δϕ_exact = qgh.ϕdiff(slms_exact[end], slms_exact[1])
    hm = CairoMakie.heatmap!(ax11, 1:N, 1:N, Δϕ_exact, colorrange = (-pi, pi))
    hidedecorations!(ax11)
    Colorbar(fig[1, 1][1, 2], hm, ticks = (-pi:pi/2:pi, [L"-\pi", L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}", L"\pi"]))

    ax12 = Axis(fig[1, 2], aspect = DataAspect(), title = L"\Delta \phi_{\mathrm{flow}}")
    Δϕ_flow = qgh.ϕdiff(slms_flow_implicit[end], slms_flow_implicit[1]) 
    hm = CairoMakie.heatmap!(ax12, 1:N, 1:N, Δϕ_flow, colorrange = (-pi, pi))
    hidedecorations!(ax12)
    Colorbar(fig[1, 2][1, 2], hm, ticks = (-pi:pi/2:pi, [L"-\pi", L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}", L"\pi"]))

    save("project/A. Movement in One Direction/one_direction_move/more_points_Delta_phi_wgs_flow.pdf", fig)
end

begin # plot flow without implicit
    layouts_flow, slms_flow_1 = qgh.evolution_slm_pure_flow(layout, layout_end, slm, algorithm; 
                                                          interps=1, 
                                                          aditers=1,
                                                          ifimplicit=false)

    layouts_flow, slms_flow_10 = qgh.evolution_slm_pure_flow(layout, layout_end, slm, algorithm; 
                                                          interps=1, 
                                                          aditers=10,
                                                          ifimplicit=false)

    fig = Figure(size = (520, 400))
    # ax = Axis(fig[1, 1], aspect = DataAspect())
    # qgh.heatmap!(ax, slms_flow[end], image_resolution)
    # fig

    ax11 = Axis(fig[1, 1], aspect = DataAspect(), title = L"\Delta \phi_{\mathrm{implicit}}")
    Δϕ_flow_implicit = qgh.ϕdiff(slms_flow_implicit[end], slms_flow_implicit[1]) 
    hm = CairoMakie.heatmap!(ax11, 1:N, 1:N, Δϕ_flow_implicit, colorrange = (-pi, pi))
    hidedecorations!(ax11)
    Colorbar(fig[1, 1][1, 2], hm, ticks = (-pi:pi/2:pi, [L"-\pi", L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}", L"\pi"]))

    ax12 = Axis(fig[1, 2], aspect = DataAspect(), title = L"\Delta \phi_{\mathrm{expand-1}}")
    Δϕ_flow_1 = qgh.ϕdiff(slms_flow_1[end], slms_flow_1[1]) 
    hm = CairoMakie.heatmap!(ax12, 1:N, 1:N, Δϕ_flow_1, colorrange = (-pi, pi))
    hidedecorations!(ax12)
    Colorbar(fig[1, 2][1, 2], hm, ticks = (-pi:pi/2:pi, [L"-\pi", L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}", L"\pi"]))

    ax21 = Axis(fig[2, 1], aspect = DataAspect(), title = L"\Delta \phi_{\mathrm{expand-10}}")
    Δϕ_flow_10 = qgh.ϕdiff(slms_flow_10[end], slms_flow_10[1]) 
    hm = CairoMakie.heatmap!(ax21, 1:N, 1:N, Δϕ_flow_10, colorrange = (-pi, pi))
    hidedecorations!(ax21)
    Colorbar(fig[2, 1][1, 2], hm, ticks = (-pi:pi/2:pi, [L"-\pi", L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}", L"\pi"]))

    ax22 = Axis(fig[2, 2], aspect = DataAspect(), title = L"\Delta \phi_{\mathrm{implicit}} - \Delta \phi_{\mathrm{expand-10}}")
    hm = CairoMakie.heatmap!(ax22, 1:N, 1:N, Δϕ_flow_implicit - Δϕ_flow_10)
    hidedecorations!(ax22)
    Colorbar(fig[2, 2][1, 2], hm, tickformat =  "{:.2f}")

    save("project/A. Movement in One Direction/one_direction_move/more_points_Delta_phi_implicit_expand.pdf", fig)
end

begin 
    errs = []
    naditers = 20
    for aditers in 1:naditers
        layouts_flow, slms_flow = qgh.evolution_slm_pure_flow(layout, layout_end, slm, algorithm; 
                                                            interps=1, 
                                                            aditers=aditers,
                                                            ifimplicit=false)
        Δϕ_flow = qgh.ϕdiff(slms_flow[end], slms_flow[1])
        err = norm(Δϕ_flow_implicit - Δϕ_flow)
        push!(errs, err)
    end
    fig = Figure(size = (400, 400))
    ax11 = Axis(fig[1, 1], xlabel = "iterations for AD (expand terms)", ylabel = "error")
    CairoMakie.lines!(ax11, 1:naditers, errs, color = :black)
    CairoMakie.plot!(ax11, 1:naditers, errs, label = "error")

    save("project/A. Movement in One Direction/one_direction_move/more_points_error_aditers.pdf", fig)
end