using qgh
using CairoMakie
using Random

begin # slms_exact data
    Random.seed!(42)
    include("../../example/layout_example.jl")

    atype = Array # or Array

    layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
    layout_end = atype(layout_example(Val(:butterflycontinuous); α = 0.0125))
    slm = atype(SLM(100))

    algorithm = WGS(maxiter=100, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.9)

    #benchmark
    layouts_exact, slms_exact = evolution_slm_direct(layout, layout_end, slm, algorithm; keypoints=5)
end

begin # plot exact half body image
    image_resolution = 1000

    fig = Figure(size = (500, 500))
    ax1 = Axis(fig[1, 1], aspect = DataAspect())
    qgh.heatmap!(ax1, slms_exact[1], image_resolution)
    areas = qgh.find_area(layout, layout_end, image_resolution, 5)
    area = areas[1]
    qgh.plot_rectange!(ax1, area)
    display(fig)

    fig = Figure(size = (1000, 200))
    for i in 1:5
        axi = Axis(fig[1, i], aspect = DataAspect(), limits = (70, 100, 810, 840))
        qgh.heatmap!(axi, slms_exact[i], image_resolution)
        qgh.plot_rectange!(axi, areas[1])
    end
    display(fig)
end

begin # plot flow half body image
    layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
    layout_new = atype(layout_example(Val(:butterflycontinuous); α = 0.02))

    layouts_flow, slms_flow = evolution_slm_flow(layout, layout_new, slm, algorithm; 
                                                 interps=10, 
                                                 aditers=10,
                                                 ifimplicit=false,
                                                 α = 1)
    fig = Figure(size = (1000, 200))
    for i in 1:5
        axi = Axis(fig[1, i], aspect = DataAspect(), limits = (70, 100, 810, 840))
        qgh.heatmap!(axi, slms_flow[i], image_resolution)
        qgh.plot_rectange!(axi, areas[1])
    end
    display(fig)
end