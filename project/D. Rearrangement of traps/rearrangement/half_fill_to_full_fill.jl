using qgh
using CairoMakie
using Random
using LinearAlgebra
using Hungarian

begin # 20x20 half filled
    Random.seed!(45) # guarantee 16x16 points

    atype = Array # or CuArray
    N = 100
    image_resolution = 1000

    points = Array{Float64, 2}([[] []])
    for x = LinRange(0,1,20), y = LinRange(0,1,20)
        if rand() < 0.5
            points = vcat(points, [x y])
        end
    end

    layout_0 = atype(ContinuousLayout(points))
    slm = atype(SLM(N))

    algorithm = WGS(maxiter=100, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.9)

    slm_0, _, _ = match_image(layout_0, slm, algorithm)

    points = Array{Float64, 2}([[] []])
    l = ceil(Int, sqrt(layout_0.ntrap))
    for x = LinRange(0,1,l), y = LinRange(0,1,l)
        if size(points,1) < layout_0.ntrap
            points = vcat(points, [x y])
        end
    end

    layout_1 = atype(ContinuousLayout(points))
    slm_1, _, _ = match_image(layout_1, slm, algorithm)
    @show layout_0.ntrap layout_1.ntrap 
end

begin
    fig = Figure(size = (500, 200))

    ax = Axis(fig[1, 1], aspect = DataAspect(), title = L"B_0")
    hm = qgh.heatmap!(ax, slm_0, image_resolution)
    hidedecorations!(ax)
    Colorbar(fig[1, 1][1, 2], hm)

    ax = Axis(fig[1, 2], aspect = DataAspect(), title = L"B_1")
    hm = qgh.heatmap!(ax, slm_1, image_resolution)
    hidedecorations!(ax)
    Colorbar(fig[1, 2][1, 2], hm)

    display(fig)
    save("project/D. Rearrangement of traps/rearrangement/half_fill_to_full_fill.png", fig)
end

begin # optimize the order of layout_0 to layout_1 using Hungarian algorithm
    points_0 = layout_0.points
    points_1 = layout_1.points
    points_new = copy(points_0)

    # Compute the cost matrix
    cost_matrix = zeros(layout_1.ntrap, size(points_0, 1))
    for i in 1:layout_1.ntrap
        for j in 1:size(points_0, 1)
            cost_matrix[i, j] = norm(points_0[j, :] - points_1[i, :])
        end
    end

    # Apply Hungarian algorithm to find the optimal assignment
    assignment, cost = hungarian(cost_matrix)

    @show assignment 
    # Rearrange the points according to the assignment
    for i in 1:layout_1.ntrap
        points_new[i, :] = points_0[assignment[i], :]
    end

    layout_0 = atype(ContinuousLayout(points_new))
end

begin # plot flow move image
    layouts_flow, slms_flow = evolution_slm_flow(layout_0, layout_1, slm, algorithm; 
                                                 interps=10, 
                                                 aditers=15,
                                                 ifimplicit=false,
                                                 α = 0.5, β = 1.0)
    # layouts_exact, slms_flow, = evolution_slm_direct(layout_0, layout_1, slm, algorithm; keypoints=10)
    qgh.image_animation(slms_flow, image_resolution; 
                        file = "project/D. Rearrangement of traps/rearrangement/half_fill_to_full_fill_flow_move.mp4")
end

begin # decay
    image_resolution = 200
    areas = qgh.find_area(layout_0, layout_1, image_resolution, length(slms_flow) - 1; area_size=0.05)

    fig = Figure(size = (400, 250))
    fa = fig[1, 1] = GridLayout()
    for (i, j) in zip(1:6, LinRange(1, length(slms_flow), 6))
        axi = Axis(fa[1, i], aspect = DataAspect())
        qgh.heatmap!(axi, slms_flow[round(Int,j)], image_resolution)
        hidedecorations!(axi)
        qgh.plot_rectange!(axi, areas[round(Int,j)], strokewidth = 0.5)
    end
    colgap!(fa, 5)
    

    Label(fa[1, 1, Left()], "(a)",
            # fontsize = 26,
            padding = (0, 14, 0, 0),
            # halign = :right
            )

    fb = fig[2, 1] = GridLayout()
    image_resolution = 1000

    areas = qgh.find_area(layout_0, layout_1, image_resolution, length(slms_flow) - 1; area_size=0.01)
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
    display(fig)
    save("project/D. Rearrangement of traps/rearrangement/half_fill_to_full_fill_all.pdf", fig)
end