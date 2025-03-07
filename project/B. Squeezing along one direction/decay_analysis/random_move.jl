using HoloGrad
using CairoMakie
using Random

begin # slms exact move
    Random.seed!(42)
    include("../../../example/layout_example.jl")

    atype = Array # or Array
    image_resolution = 1000

    layout_end = atype(layout_example(Val(:butterflycontinuous); α = 0.0))
    rand_θ = rand(size(layout_end.points, 1)) * 2π
    rand_r = 0.05 * [cos.(rand_θ) sin.(rand_θ)]
    layout = ContinuousLayout(layout_end.points + rand_r)
    slm = atype(SLM(100))

    algorithm = WGS(maxiter=100, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.9)

    #benchmark
    layouts_exact, slms_exact = evolution_slm_direct(layout, layout_end, slm, algorithm; keypoints=100)

    HoloGrad.image_animation(slms_exact, image_resolution; 
                        file = "project/B. Squeezing along one direction/decay_analysis/random_exact_move.mp4")
end

begin # plot flow move image
    layouts_flow, slms_flow = evolution_slm_flow(layout, layout_end, slm, algorithm; 
                                                 interps=10, 
                                                 aditers=10,
                                                 ifimplicit=false, β=1)
    HoloGrad.image_animation(slms_flow, image_resolution; 
                        file = "project/B. Squeezing along one direction/decay_analysis/random_flow_move.mp4")
end

# begin # plot more flow moves
#     fig = Figure(size = (800, 800))
#     fa = fig[1, 1] = GridLayout()

#     ax1 = Axis(fa[1, 1], aspect = DataAspect())
#     HoloGrad.heatmap!(ax1, slms_flow[1], image_resolution)
#     ax2 = Axis(fa[1, 2], aspect = DataAspect())
#     HoloGrad.heatmap!(ax2, slms_flow[end], image_resolution)
#     area = [(50, 800), (120, 890)]
#     HoloGrad.plot_rectange!(ax1, area)
#     HoloGrad.plot_rectange!(ax2, area)

#     f2 = fig[2, 1] = GridLayout()
#     HoloGrad.plot_distance_intensity_decay!(f2, slms_flow[1:end], area, image_resolution; color = :red, label = "flow")
#     CairoMakie.colgap!(f2, -150)
#     display(fig)
# end