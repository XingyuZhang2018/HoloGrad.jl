using qgh
using Random

let
    Random.seed!(100)
    include("layout_example.jl")
    layout = layout_example(Val(:butterflycontinuous); α = 0.0)
    layout_new = layout_example(Val(:butterflycontinuous); α = 0.2)
    slm = SLM(20)
    algorithm = WGS(verbose=false, ft_method=:cft)
    slms = []
    diff = layout_new.points - layout.points
    v = [[d[1] for d in diff], [d[2] for d in diff]]

    layouts, slms = evolution_slm(layout, slm, v, WGS(verbose=false, ft_method=:cft); iters=5, dt=0.2)
    # plot(layouts, slms)
    plot(slms)
end