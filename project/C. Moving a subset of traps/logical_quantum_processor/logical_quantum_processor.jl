using HoloGrad
using CairoMakie
using Random
using LinearAlgebra

begin # slms exact move
    Random.seed!(42)
    include("../../../example/layout_example.jl")

    atype = Array # or CuArray
    N = 50
    image_resolution = 100
    keypoints = 1

    layout_0 = atype(layout_example(Val(:logical_array); α = 0.0, β = 0.0))
    layout_1 = atype(layout_example(Val(:logical_array); α = 0.0, β = 0.5))
    layout_2 = atype(layout_example(Val(:logical_array); α = 1.0, β = 0.5))
    slm = atype(SLM(5*N, N))

    algorithm = WGS(maxiter=100, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.9)

    slm_exact_0, cost, B = match_image(layout_0, slm, algorithm)
    slm_exact_1, cost, B = match_image(layout_1, slm, algorithm)
    slm_exact_2, cost, B = match_image(layout_2, slm, algorithm)
end

begin # plot exact ϕ and B
    fig = Figure(size = (500, 360))

    ax = Axis(fig[1, 1], aspect = DataAspect(), title = L"B_0")
    hm = HoloGrad.heatmap!(ax, slm_exact_0, 5*image_resolution, image_resolution)
    hidedecorations!(ax)
    Colorbar(fig[1, 1][1, 2], hm)
    HoloGrad.plot_rectange!(ax, [[450, 25], [600, 100]] .* 0.5)

    ax = Axis(fig[2, 1], aspect = DataAspect(), title = L"B_1")
    hm = HoloGrad.heatmap!(ax, slm_exact_1, 5*image_resolution, image_resolution)
    hidedecorations!(ax)
    Colorbar(fig[2, 1][1, 2], hm)
    HoloGrad.plot_rectange!(ax, [[450, 25], [600, 100]] .* 0.5)

    ax = Axis(fig[3, 1], aspect = DataAspect(), title = L"B_2")
    hm = HoloGrad.heatmap!(ax, slm_exact_2, 5*image_resolution, image_resolution)
    hidedecorations!(ax)
    Colorbar(fig[3, 1][1, 2], hm)
    HoloGrad.plot_rectange!(ax, [[290, 25], [440, 100]] .* 0.5)

    for (i,l) in zip(1:3, ["(a)", "(b)", "(c)"])
        Label(fig[i, 1, Left()], l)
    end

    display(fig)
    save("project/C. Moving a subset of traps/logical_quantum_processor/logical_quantum_processor_B.pdf", fig)
end

begin # plot flow move image
    _, slms_flow_0 = evolution_slm_flow(layout_0, layout_1, slm, algorithm; 
                                                 interps=10, 
                                                 aditers=15,
                                                 ifimplicit=false,
                                                 α = 0.5, β = 1)

    _, slms_flow_1 = evolution_slm_flow(layout_1, layout_2, slm, algorithm; 
                                                 interps=10, 
                                                 aditers=15,
                                                 ifimplicit=false,
                                                 α = 0.5, β = 1)
end

begin # decay
    image_resolution = 400
    areas = HoloGrad.find_area(layout_0, layout_1, [image_resolution*5, image_resolution], length(slms_flow_0) - 1; area_size=0.03)

    fig = Figure(size = (500, 500))
    fa = fig[1, 1] = GridLayout()
    for (i, j) in zip(1:6, LinRange(1, length(slms_flow_0), 6))
        axi = Axis(fa[i, 1], aspect = DataAspect())
        HoloGrad.heatmap!(axi, slms_flow_0[round(Int,j)], image_resolution*5, image_resolution)
        hidedecorations!(axi)
        HoloGrad.plot_rectange!(axi, areas[round(Int,j)])
    end
    colgap!(fa, 5)
    

    Label(fa[1, 1, Left()], "(a)",
            # fontsize = 26,
            padding = (0, 10, 0, 0),
            # halign = :right
            )

    fb = fig[1, 2] = GridLayout()
    image_resolution = 1000

    areas = HoloGrad.find_area(layout_0, layout_1, [image_resolution*5, image_resolution], length(slms_flow_0) - 1; area_size=0.03)
    positions = []
    intensities = Array{Float64}([])
    
    for i in 1:length(slms_flow_0)
        intensity, position = HoloGrad.find_max_intensity(slms_flow_0[i], areas[i], image_resolution*5, image_resolution)
        push!(positions, position)
        push!(intensities, intensity)
    end
    distances = [norm(positions[i] - positions[1]) for i in 1:length(positions)] / (image_resolution / N)
    intensities /= maximum(intensities) 

    ax1 = Axis(fb[1, 1], xlabel = "steps", ylabel = L"\Delta x/\mathrm{pixel}")
    CairoMakie.lines!(ax1, 1:length(slms_flow_0), distances)
    # CairoMakie.scatter!(ax1, 1:length(slms_flow_0), distances)

    ax2 = Axis(fb[1, 2], xlabel = "steps", ylabel = L"B_{\mathrm{max}}")
    CairoMakie.lines!(ax2, 1:length(slms_flow_0), intensities)
    # CairoMakie.scatter!(ax2, 1:length(slms_flow_0), intensities)

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

    rowgap!(fig.layout, 0)
    display(fig)
end

begin # butterfly_flow_all
    
    fig = Figure(size = (400, 450))

    for (i, slms_flow, layout) in zip(1:2, [slms_flow_0, slms_flow_1], [[layout_0, layout_1], [layout_1, layout_2]])
        areas = HoloGrad.find_area(layout[1], layout[2], [image_resolution*5, image_resolution], length(slms_flow) - 1; area_size=0.01)
        positions = []
        intensities = Array{Float64}([])
        
        for i in 1:length(slms_flow)
            intensity, position = HoloGrad.find_max_intensity(slms_flow[i], areas[i], image_resolution*5, image_resolution)
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
    save("project/C. Moving a subset of traps/logical_quantum_processor/logical_quantum_processor_flow_all.pdf", fig)
end