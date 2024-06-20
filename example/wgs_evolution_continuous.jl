using qgh
using Random
using qgh: evolution_slm
using ProfileView

let
    Random.seed!(100)
    include("layout_example.jl")
    layout = layout_example(Val(:butterflycontinuous); α = 0.0)
    layout_new = layout_example(Val(:butterflycontinuous); α = 0.2)
    slm = SLM(1000)
    algorithm = WGS(; maxiter=50, verbose=false, show_every=10, tol=1e-3, ratio_fixphase=0.5)
    slms = []

    layouts, slms = evolution_slm(layout, layout_new, slm, algorithm; iters=10)
    # ProfileView.@profview layouts, slms = evolution_slm(layout, layout_new, slm, WGS(verbose=false, ft_method=:cft); iters=1)
    # ProfileView.@profview layouts, slms = evolution_slm(layout, layout_new, slm, WGS(verbose=false, ft_method=:cft); iters=1)
    # plot(layouts, slms) 
    plot(slms)
end