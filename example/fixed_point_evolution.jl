using qgh

let 
    Random.seed!(42)
    mask = zeros(Bool, 100, 100)
    mask[2,1] = true
    layout = GridLayout(mask)
    mask = zeros(Bool, 100, 100)
    mask[2,10] = true
    layout_new = GridLayout(mask)
    slm = SLM(10)

    slms= evolution_slm(layout, layout_new, slm, WGS(verbose=false); iters=10, δ=1, dt=0.1)
    for i in 1:length(slms)-1
        d = ϕdiff(slms[i+1], slms[i])
        println(maximum(abs.(d)))
    end
    plot(slms)
end