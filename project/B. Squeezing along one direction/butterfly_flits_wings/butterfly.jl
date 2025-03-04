using qgh
using CairoMakie
using Random
using LinearAlgebra

begin # slms_exact data
    Random.seed!(42)

    include("../../../example/layout_example.jl")
    atype = Array # or CuArray
    N = 100

    layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
    layout_end = atype(layout_example(Val(:butterflycontinuous); α = 0.2))
    slm = atype(SLM(N))

    algorithm = WGS(maxiter=100, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.9)

    #benchmark
    layouts_exact, slms_exact = evolution_slm_direct(layout, layout_end, slm, algorithm; keypoints=1)
end

begin # plot exact ϕ and B
    image_resolution = N*3

    fig = Figure(size = (500, 200))
    areas = qgh.find_area(layout, layout_end, image_resolution, 2; area_size=0.05)

    ax11 = Axis(fig[1, 1], aspect = DataAspect(), title = L"B_0")
    hm = qgh.heatmap!(ax11, slms_exact[1], image_resolution)
    hidedecorations!(ax11)
    Colorbar(fig[1, 1][1, 2], hm)
    qgh.plot_rectange!(ax11, areas[1])

    ax12 = Axis(fig[1, 2], aspect = DataAspect(), title = L"B_1")
    hm = qgh.heatmap!(ax12, slms_exact[end], image_resolution)
    hidedecorations!(ax12)
    Colorbar(fig[1, 2][1, 2], hm)
    qgh.plot_rectange!(ax12, areas[end])

    # display(fig)
    save("project/B. Squeezing along one direction/butterfly_flits_wings/butterfly_B.pdf", fig)
end

begin # butterfly_flow_decay
    image_resolution = 1000

    layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
    layout_end = atype(layout_example(Val(:butterflycontinuous); α = 0.04))

    layouts_flow, slms_flow = qgh.evolution_slm_pure_flow(layout, layout_end, slm, algorithm; 
                                                          interps=5, 
                                                          aditers=15,
                                                          ifimplicit=false)

    areas = qgh.find_area(layout, layout_end, image_resolution, 5; area_size=0.01)

    fig = Figure(size = (400, 200))
    fa = fig[1, 1] = GridLayout()
    for i in 1:length(slms_flow)
        axi = Axis(fa[1, i], aspect = DataAspect(), limits = (70, 100, 810, 840))
        qgh.heatmap!(axi, slms_flow[i], image_resolution)
        hidedecorations!(axi)
        qgh.plot_rectange!(axi, areas[1])
    end
    colgap!(fa, 5)

    Label(fa[1, 1, Left()], "(a)",
            # fontsize = 26,
            padding = (0, 10, 0, 0),
            # halign = :right
            )

    fb = fig[2, 1] = GridLayout()

    positions = []
    intensities = Array{Float64}([])
    
    for i in 1:length(slms_flow)
        intensity, position = qgh.find_max_intensity(slms_flow[i], areas[i], image_resolution, image_resolution)
        push!(positions, position)
        push!(intensities, intensity)
    end
    distances = [norm(positions[i] - positions[1]) for i in 1:length(positions)] / (image_resolution / N)
    intensities /= maximum(intensities) 

    ax1 = Axis(fb[1, 1], xlabel = "steps", ylabel = L"\Delta x/\mathrm{pixel}")
    CairoMakie.lines!(ax1, 1:length(slms_flow), distances)
    CairoMakie.scatter!(ax1, 1:length(slms_flow), distances)

    ax2 = Axis(fb[1, 2], xlabel = "steps", ylabel = L"B_{\mathrm{max}}")
    CairoMakie.lines!(ax2, 1:length(slms_flow), intensities)
    CairoMakie.scatter!(ax2, 1:length(slms_flow), intensities)

    Label(fb[1, 1, TopLeft()], "(b)",
    # fontsize = 26,
    padding = (0, 0, 0, 0),
    # halign = :right
    )

    Label(fb[1, 2, TopLeft()], "(c)",
    # fontsize = 26,
    padding = (0, 25, 0, 0),
    # halign = :right
    )

    # display(fig)
    save("project/B. Squeezing along one direction/butterfly_flits_wings/butterfly_flow_decay.pdf", fig)
end

begin # flow date
    image_resolution = 1000

    layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
    layout_end = atype(layout_example(Val(:butterflycontinuous); α = 0.2))

    _, slms_flow_1 = qgh.evolution_slm_flow(layout, layout_end, slm, algorithm; 
                                            interps=10, 
                                            aditers=15,
                                            α = 1,
                                            β = 2,
                                            ifimplicit=false)

    _, slms_flow_2 = qgh.evolution_slm_flow(layout, layout_end, slm, algorithm; 
                                            interps=10, 
                                            aditers=15,
                                            α = 0.5,
                                            β = 1,
                                            ifimplicit=false)
end

begin # butterfly_flow_all
    
    fig = Figure(size = (400, 450))

    for (i, slms_flow) in zip(1:2, [slms_flow_1, slms_flow_2])
        areas = qgh.find_area(layout, layout_end, image_resolution, length(slms_flow) - 1; area_size=0.01)
        positions = []
        intensities = Array{Float64}([])
        
        for i in 1:length(slms_flow)
            intensity, position = qgh.find_max_intensity(slms_flow[i], areas[i], image_resolution, image_resolution)
            push!(positions, position)
            push!(intensities, intensity)
        end
        distances = [norm(positions[i] - positions[1]) for i in 1:length(positions)] / (image_resolution / N)
        intensities /= maximum(intensities) 

        ax1 = Axis(fig[i, 1], xlabel = "steps", ylabel = L"\Delta x/\mathrm{pixel}")
        CairoMakie.lines!(ax1, 1:length(slms_flow), distances)

        ax2 = Axis(fig[i, 2], xlabel = "steps", ylabel = L"B_{\mathrm{max}}")
        CairoMakie.lines!(ax2, 1:length(slms_flow), intensities)
    end

    colgap!(fig.layout, 2)
    rowgap!(fig.layout, 1)
    for (label, layout) in zip(["(a)", "(b)", "(c)", "(d)"], [fig[1,1], fig[1,2], fig[2,1], fig[2,2]])
        Label(layout[1, 1, TopLeft()], label, padding = (0, 10, 10, 0))
    end
    display(fig)
    save("project/B. Squeezing along one direction/butterfly_flits_wings/butterfly_flow_all.pdf", fig)
end

begin
    layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
    layout_end = atype(layout_example(Val(:butterflycontinuous); α = 0.004))
    
    layouts_exact, slms_exact = evolution_slm_direct(layout, layout_end, slm, algorithm; keypoints=1)

    fig = Figure(size = (400, 400))
    yticks = (-pi:pi/2:pi, [L"-\pi", L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}", L"\pi"])

    ax11 = Axis(fig[1, 1], aspect = DataAspect(), title = L"\Delta \phi_{\mathrm{WGS}}")
    Δϕ_exact = qgh.ϕdiff(slms_exact[2], slms_exact[1])
    hm = CairoMakie.heatmap!(ax11, 1:N, 1:N, Δϕ_exact, colorrange = (-pi, pi))
    hidedecorations!(ax11)
    # Colorbar(fig[1, 1][1, 2], hm, ticks = yticks)
    # colgap!(ax11, 1)

    ax12 = Axis(fig[1, 2], aspect = DataAspect(), title = L"\Delta \phi_{\mathrm{flow}}")
    Δϕ_flow = qgh.ϕdiff(slms_flow[2], slms_flow[1])
    hm = CairoMakie.heatmap!(ax12, 1:N, 1:N, Δϕ_flow, colorrange = (-pi, pi))
    hidedecorations!(ax12)
    # Colorbar(fig[1, 2][1, 2], hm, ticks = yticks)

    ax21 = Axis(fig[2, 1], aspect = 1, xlabel = "position", limits = (nothing, (-pi, pi)), yticks = yticks)
    qgh.plot_slms_ϕ_diff!(ax21, slms_exact[1:2]; color = :red)
    ax22 = Axis(fig[2, 2], aspect = 1, xlabel = "position", limits = (nothing, (-pi, pi)), yticks = yticks)
    qgh.plot_slms_ϕ_diff!(ax22, slms_flow[1:2]; color = :red)

    # fig
    save("project/B. Squeezing along one direction/butterfly_flits_wings/butterfly_Delta_phi_wgs_flow.pdf", fig)
end

