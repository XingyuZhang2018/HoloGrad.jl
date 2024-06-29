using qgh
using Random
using qgh: evolution_slm
using ProfileView
using Statistics
using Plots
using CUDA

let
    Random.seed!(42)
    include("layout_example.jl")

    atype = CuArray # or Array

    layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
    layout_new = atype(layout_example(Val(:butterflycontinuous); α = 0.2))
    slm = atype(SLM(100))

    algorithm = WGS(maxiter=100, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.8)
    slms = []

    layouts, slms = evolution_slm(layout, layout_new, slm, algorithm; 
                                  slices=10, 
                                  interps=1, 
                                  aditers=5,
                                  ifflow=true)
    # display(plot(slms))
    display(plot_gif(slms))
    # display(plot_slms_ϕ_diff(slms))
end