using qgh
using CairoMakie
using Random

begin # slms_exact data
    Random.seed!(42)
    include("../../example/layout_example.jl")

    atype = Array # or Array

    layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
    layout_new = atype(layout_example(Val(:butterflycontinuous); α = 0.0125))
    slm = atype(SLM(100))

    algorithm = WGS(maxiter=100, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.9)

    #benchmark
    layouts_exact, slms_exact = evolution_slm_direct(layout, layout_new, slm, algorithm; keypoints=5)
end

begin # plot exact intensity decay and move
    image_resolution = 1000
    areas = qgh.find_area(layout, layout_new, image_resolution, 5)
    area = areas[1]

    fig = Figure(size = (500, 500))
    fa = fig[1, 1] = GridLayout()
    ax1 = Axis(fa[1, 1], aspect = DataAspect(), limits = (40, 110, 790, 860))
    qgh.heatmap!(ax1, slms_exact[1], image_resolution)
    ax2 = Axis(fa[1, 2], aspect = DataAspect(), limits = (40, 110, 790, 860))
    qgh.heatmap!(ax2, slms_exact[end], image_resolution)
    area = [(50, 800), (100, 850)]
    qgh.plot_rectange!(ax1, area)
    qgh.plot_rectange!(ax2, area)

    fb = fig[2, 1] = GridLayout()
    qgh.plot_distance_intensity_decay!(fb, slms_exact, areas, image_resolution; color = :red, label = "exact")
    display(fig)
end

begin # plot flow intensity decay and move
    layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
    layout_new = atype(layout_example(Val(:butterflycontinuous); α = 0.015))

    layouts_flow, slms_flow = evolution_slm_flow(layout, layout_new, slm, algorithm; 
                                                 interps=10, 
                                                 aditers=10,
                                                 ifimplicit=false)

    fig = Figure(size = (500, 500))
    fa = fig[1, 1] = GridLayout()
    ax1 = Axis(fa[1, 1], aspect = DataAspect(), limits = (40, 110, 790, 860))
    qgh.heatmap!(ax1, slms_flow[1], image_resolution)
    ax2 = Axis(fa[1, 2], aspect = DataAspect(), limits = (40, 110, 790, 860))
    qgh.heatmap!(ax2, slms_flow[4], image_resolution)
    area = [(50, 800), (100, 850)]
    qgh.plot_rectange!(ax1, area)
    qgh.plot_rectange!(ax2, area)

    fb = fig[2, 1] = GridLayout()
    qgh.plot_distance_intensity_decay!(fb, slms_flow[1:5], areas, image_resolution; color = :red, label = "flow")
    display(fig)
end

begin # plot more flow moves
    layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
    layout_new = atype(layout_example(Val(:butterflycontinuous); α = 0.03))

    layouts_flow, slms_flow = evolution_slm_flow(layout, layout_new, slm, algorithm; 
                                                 interps=100, 
                                                 aditers=10,
                                                 ifimplicit=false)

    areas = qgh.find_area(layout, layout_new, image_resolution, length(slms_flow))

    fig = Figure(size = (500, 500))
    fa = fig[1, 1] = GridLayout()
    ax1 = Axis(fa[1, 1], aspect = DataAspect(), limits = (40, 110, 790, 860))
    qgh.heatmap!(ax1, slms_flow[1], image_resolution)
    ax2 = Axis(fa[1, 2], aspect = DataAspect(), limits = (40, 110, 790, 860))
    qgh.heatmap!(ax2, slms_flow[40], image_resolution)
    area = [(50, 800), (100, 850)]
    qgh.plot_rectange!(ax1, area)
    qgh.plot_rectange!(ax2, area)

    fb = fig[2, 1] = GridLayout()
    qgh.plot_distance_intensity_decay!(fb, slms_flow[1:end], areas, image_resolution; color = :red, label = "flow")
    display(fig)
end

begin # plot slms ϕ differece
    fig = Figure(size = (800, 400))
    ax1 = Axis(fig[1, 1], title = "exact", xlabel = "position", ylabel = "Δϕ")
    qgh.plot_slms_ϕ_diff!(ax1, slms_exact[1:2]; color = :red)
    ax2 = Axis(fig[1, 2], title = "flow", xlabel = "position", ylabel = "Δϕ")
    qgh.plot_slms_ϕ_diff!(ax2, slms_flow[1:2]; color = :red)
    display(fig)
end