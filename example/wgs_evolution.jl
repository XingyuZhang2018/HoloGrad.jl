using qgh
using Random

let
    Random.seed!(100)
    layout = layout_example(Val(:butterfly); α = 0.0)
    slm = SLM(64)
    algorithm = WGS(verbose=false)
    
    slms = []
    for α in vcat(0:0.1:0.5,0.4:-0.1:0.0)
        layout_new = layout_example(Val(:butterfly); α = α)
        slm, cost, target_A_reweight = match_image(layout, slm, algorithm)
        layout = layout_new
        push!(slms, slm)
    end
    plot(padding.(slms, 64/512))
end