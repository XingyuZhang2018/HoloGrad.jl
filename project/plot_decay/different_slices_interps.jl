using qgh
using Random
using qgh: evolution_slm
using ProfileView
using Statistics
using Plots
using CUDA

let
    Random.seed!(42)
    include("../../example/layout_example.jl")

    atype = Array # or Array

    layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
    layout_new = atype(layout_example(Val(:butterflycontinuous); α = 0.1))
    slm = atype(SLM(100))

    # layout = atype(ContinuousLayout([0.5 0.5]))
    # layout_new = atype(ContinuousLayout([0.9 0.5]))
    # slm = atype(SLM(10))

    algorithm = WGS(maxiter=100, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.8)
    
    p = Plots.plot(layout=(1, 2), size=(800, 450), margin=5Plots.mm)


    for slices in 5:5:20, interps in 10:2:10
        layouts, slms = evolution_slm(layout, layout_new, slm, algorithm; 
                                      slices, 
                                      interps, 
                                      aditers=5,
                                      ifflow=true)

        # display(plot_gif(slms))
        display(plot_decay(p[1], slms, 100))
        display(plot_decay(p[2], layouts, slms, slices, interps; linewidth=2))
    end

    #benchmark
    layouts, slms = evolution_slm(layout, layout_new, slm, algorithm; 
                                slices=100, 
                                interps=1, 
                                aditers=5,
                                ifflow=false)

    display(plot_decay(p[1], slms, 100))
    display(plot_decay(p[2], layouts, slms, 100, 1; linewidth=5))
end