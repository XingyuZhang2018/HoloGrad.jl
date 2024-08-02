using qgh
using CairoMakie
using Random
using LinearAlgebra

begin # slms exact move
    Random.seed!(42)
    include("../../example/layout_example.jl")

    atype = Array # or CuArray
    N = 100
    image_resolution = 1000
    keypoints = 10

    layout = atype(layout_example(Val(:circle); α = 0.0))
    layout_end = atype(layout_example(Val(:circle); α = 0.5))
    slm = atype(SLM(N))

    algorithm = WGS(maxiter=100, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.9)

    #benchmark
    layouts_exact, slms_exact = evolution_slm_direct(layout, layout_end, slm, algorithm; keypoints)

    # qgh.heatmap(slms_exact[[1,end]], image_resolution)
    # qgh.image_animation(slms_exact, image_resolution; 
    #                     file = "project/shape_change/circle_to_ellipse_exact_move.mp4")
end

begin # plot exact ϕ and B
    image_resolution = N*3

    fig = Figure(size = (500, 200))
    areas = qgh.find_area(layout, layout_end, image_resolution, 2; area_size=0.03)

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
    save("project/shape_change/circle_to_ellipse_B.pdf", fig)
end

begin # plot flow move image
    layouts_flow, slms_flow = evolution_slm_flow(layout, layout_end, slm, algorithm; 
                                                 interps=10, 
                                                 aditers=15,
                                                 ifimplicit=false,
                                                 α = 0.5, β = 1)
    qgh.image_animation(slms_flow, image_resolution; 
                        file = "project/shape_change/circle_to_ellipse_flow_move.mp4")
end

begin # decay
    image_resolution = 120
    areas = qgh.find_area(layout, layout_end, image_resolution, length(slms_flow) - 1; area_size=0.01)

    fig = Figure(size = (400, 250))
    fa = fig[1, 1] = GridLayout()
    for (i, j) in zip(1:6, LinRange(1, length(slms_flow), 6))
        axi = Axis(fa[1, i], aspect = DataAspect())
        qgh.heatmap!(axi, slms_flow[round(Int,j)], image_resolution)
        hidedecorations!(axi)
        qgh.plot_rectange!(axi, areas[round(Int,j)])
    end
    colgap!(fa, 5)
    

    Label(fa[1, 1, Left()], "(a)",
            # fontsize = 26,
            padding = (0, 10, 0, 0),
            # halign = :right
            )

    fb = fig[2, 1] = GridLayout()
    image_resolution = 1000

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

    ax1 = Axis(fb[1, 1], xlabel = "steps", ylabel = L"\Delta x/\mathrm{pixel}")
    CairoMakie.lines!(ax1, 1:length(slms_flow), distances)
    # CairoMakie.scatter!(ax1, 1:length(slms_flow), distances)

    ax2 = Axis(fb[1, 2], xlabel = "steps", ylabel = L"B_{\mathrm{max}}")
    CairoMakie.lines!(ax2, 1:length(slms_flow), intensities)
    # CairoMakie.scatter!(ax2, 1:length(slms_flow), intensities)

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
    # display(fig)
    save("project/shape_change/circle_to_ellipse_all.pdf", fig)
end