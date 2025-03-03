using qgh
using Random
using CUDA

let
    Random.seed!(42)
    include("layout_example.jl")

    atype = Array # or CuArray for GPU calculation

    layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
    layout_new = atype(layout_example(Val(:butterflycontinuous); α = 0.2))
    slm = atype(SLM(100))

    algorithm = WGS(maxiter=100, verbose=true, show_every=10, tol=1e-10, ratio_fixphase=0.8)
    slms = []

    layouts, slms = evolution_slm_flow(layout, layout_new, slm, algorithm)

    qgh.image_animation(slms, 200; file = "example/butterflycontinuous.gif")
end