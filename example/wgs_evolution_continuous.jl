using qgh
using Random
using qgh: evolution_slm
using ProfileView
using Statistics
using Plots

let
    Random.seed!(42)
    include("layout_example.jl")

    layout = layout_example(Val(:butterflycontinuous); α = 0.0)
    layout_new = layout_example(Val(:butterflycontinuous); α = 0.2)
    slm = SLM(50)

    # points = [[0.5, 0.5]]
    # move = [[0.1, 0.0]] 
    # @show move
    # layout = ContinuousLayout(points)
    # layout_new = ContinuousLayout(points + move)
    # slm = SLM(10)

    algorithm = WGS(maxiter=100, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.8)
    slms = []

    layouts, slms = evolution_slm(layout, layout_new, slm, algorithm; 
                                  slices=10, interps=5, ifflow=true)

    # for i in 1:length(slms)-1
    #     d = ϕdiff(slms[i+1], slms[i])
    #     println(mean(abs.(d)))
    # end
    # ProfileView.@profview layouts, slms = evolution_slm(layout, layout_new, slm, WGS(verbose=false, ft_method=:cft); iters=1)
    # ProfileView.@profview layouts, slms = evolution_slm(layout, layout_new, slm, WGS(verbose=false, ft_method=:cft); iters=1)
    # plot(layouts, slms) 
    # plot(slms)
    # display(plot(slms))
    display(plot_gif(slms))
    # display(plot_slms_ϕ_diff(slms))
end