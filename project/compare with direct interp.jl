using qgh
using Random
using qgh: evolution_slm
using Statistics
using Plots

begin
    Random.seed!(42)
    include("../example/layout_example.jl")

    atype = Array # or Array

    layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
    layout_new = atype(layout_example(Val(:butterflycontinuous); α = 0.1))
    slm = atype(SLM(100))

    # layout = atype(ContinuousLayout([0.5 0.5]))
    # layout_new = atype(ContinuousLayout([0.9 0.5]))
    # slm = atype(SLM(10))

    algorithm = WGS(maxiter=100, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.8)
    
    #benchmark
    layouts, slms = evolution_slm(layout, layout_new, slm, algorithm; 
                                slices=100, 
                                interps=1, 
                                aditers=5,
                                ifflow=false,
                                ifimplicit=false)

    layouts, slms_flow = evolution_slm(layout, layout_new, slm, algorithm; 
                                slices=10, 
                                interps=10, 
                                aditers=5,
                                ifflow=true,
                                ifimplicit=false)
end

begin
    xlims=(150, 200)
    ylims=(20, 70)
    interp = 45

    p = Plots.plot(layout=(1, 3), size=(2500, 450), right_margin=20Plots.mm)

    # plot_slms_ϕ_diff(p[1], slms[1], slms[2])
    # plot_slms_ϕ_diff(p[2], slms[2], slms[3])
    # plot_slms_ϕ_diff(p[3], slms[3], slms[4])
    # plot_slms_ϕ_diff(slms[1:10])
    plot_slms_ϕ_diff(slms_flow[1:11])
end

begin
    xlims=(30, 40)
    ylims=(4, 14)
    interp = 45

    p = Plots.plot(layout=(1, 3), size=(2500, 450), right_margin=20Plots.mm)

    heatmap!(p[1], slms[interp], 100; 
             xlims,
             ylims,
             yticks = true,
             colorbar=true,
             ytickfontsize = 20
             )


    heatmap!(p[2], slms_flow[interp], 100; 
             xlims, 
             ylims, 
             yticks = true,
             colorbar=true,
             ytickfontsize = 20)


    display(heatmap_slm_diff!(p[3], slms[interp], slms_flow[interp], 100; 
                              xlims, 
                              ylims, 
                              yticks = true,
                              ytickfontsize = 20,
                              ))
end