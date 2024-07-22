using qgh
using CairoMakie
using Random

begin # slms exact move
    Random.seed!(42)
    include("../../example/layout_example.jl")

    atype = Array # or Array
    image_resolution = 1000
    keypoints = 10

    layout = atype(layout_example(Val(:circle); α = 0.0))
    layout_end = atype(layout_example(Val(:circle); α = 0.5))
    slm = atype(SLM(100))

    algorithm = WGS(maxiter=100, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.9)

    #benchmark
    layouts_exact, slms_exact = evolution_slm_direct(layout, layout_end, slm, algorithm; keypoints)

    qgh.heatmap(slms_exact[[1,end]], image_resolution)
    qgh.image_animation(slms_exact, image_resolution; 
                        file = "project/shape_change/circle_to_ellipse_exact_move.mp4")
end

begin # plot exact half body image
    image_resolution = 1000

    fig = Figure(size = (500, 500))
    ax1 = Axis(fig[1, 1], aspect = DataAspect())
    qgh.heatmap!(ax1, slms_exact[1], image_resolution)
    # area = [(80, 820), (90, 830)]
    areas = qgh.find_area(layout, layout_end, image_resolution, keypoints)
    area = areas[1]
    qgh.plot_rectange!(ax1, area)
    display(fig)

    fig = Figure(size = (1000, 200))
    for i in 1:5
        axi = Axis(fig[1, i], aspect = DataAspect(), limits = (areas[i][1][1] - 200, areas[i][2][1] + 200, areas[i][1][2] - 200, areas[i][2][2] + 200))
        qgh.heatmap!(axi, slms_exact[i], image_resolution)
        qgh.plot_rectange!(axi, areas[i])
    end
    display(fig)
end

begin # plot flow move image
    layouts_flow, slms_flow = evolution_slm_flow(layout, layout_end, slm, algorithm; 
                                                 interps=10, 
                                                 aditers=10,
                                                 ifimplicit=false, β=1)
    qgh.image_animation(slms_flow, image_resolution; 
                        file = "project/shape_change/circle_to_ellipse_flow_move.mp4")
end

begin # plot more flow moves
    fig = Figure(size = (800, 800))
    areas = qgh.find_area(layout, layout_end, image_resolution, length(slms_flow))
    fa = fig[1, 1] = GridLayout()

    ax1 = Axis(fa[1, 1], aspect = DataAspect())
    qgh.heatmap!(ax1, slms_flow[1], image_resolution)
    ax2 = Axis(fa[1, 2], aspect = DataAspect())
    qgh.heatmap!(ax2, slms_flow[end], image_resolution)
    qgh.plot_rectange!(ax1, areas[1])
    qgh.plot_rectange!(ax2, areas[end])

    f2 = fig[2, 1] = GridLayout()
    qgh.plot_distance_intensity_decay!(f2, slms_flow[1:end], areas, image_resolution; color = :red, label = "flow")
    CairoMakie.colgap!(f2, -150)
    display(fig)
end