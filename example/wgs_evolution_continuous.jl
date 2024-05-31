using qgh
using Random

let
    Random.seed!(100)
    include("layout_example.jl")
    layout = layout_example(Val(:butterflycontinuous); α = 0.0)
    layout_new = layout_example(Val(:butterflycontinuous); α = 0.2)
    slm = SLM(50)
    algorithm = WGS(verbose=false, ft_method=:cft)
    slms = []

    layouts, slms = evolution_slm(layout, layout_new, slm, WGS(verbose=false, ft_method=:cft); iters=10)
    plot(layouts, slms) 
    # plot(slms)
end