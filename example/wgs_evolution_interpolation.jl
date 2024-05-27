using qgh
using Random

let
    Random.seed!(100)

    layout = layout_example(Val(:butterfly); α = 0.0)
    slm = SLM(64)
    algorithm = WGS(verbose=false)
    slm, cost, target_A_reweight = match_image(layout, slm, algorithm)

    layout_new = layout_example(Val(:butterfly); α = 0.1)
    slms = []
    for α in 0:0.1:1.0
        if α == 0
            slm, cost, target_A_reweight = match_image(layout, layout_new, slm, α, algorithm)
        else
            slm, cost, target_A_reweight = match_image(layout, layout_new, slm, α, target_A_reweight, algorithm)
        end
        push!(slms, slm)
    end
    plot(padding.(slms, 64/512))
end