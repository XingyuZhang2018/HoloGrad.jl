using qgh
using Random
using FFTW
FFTW.set_num_threads(2)

let
    Random.seed!(100)

    mask = zeros(Bool, 100, 100)
    mask[20:20:80, 20:20:80] .= true
    layout = GridLayout(mask)
    slm = SLM(100)
    algorithm = WGS()
    slm, cost, target_A_reweight = match_image(layout, slm, algorithm)
    @test cost < 1e-6

    mask = zeros(Bool, 100, 100)
    mask[20:20:80, 40:20:100] .= true
    layout_new = GridLayout(mask)
    slms = []
    for α in 0:0.1:1
        slm, = match_image(layout, layout_new, slm, α, algorithm)
        push!(slms, slm)
    end
    @test cost < 1e-6
    qgh.plot(slms)
end