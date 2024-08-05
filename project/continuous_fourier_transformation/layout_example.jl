using qgh
using CairoMakie
using Random
using CUDA

begin # 1600 Random points
    Random.seed!(42)

    atype = Array # or CuArray
    N = 100
    image_resolution = 1000

    layout = atype(ContinuousLayout(rand(400, 2)))
    slm = atype(SLM(N))

    algorithm = WGS(maxiter=100, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.9)

    slm_1, _, _ = match_image(layout, slm, algorithm)
end

begin # 40x40 half filled
    Random.seed!(42)

    atype = Array # or CuArray
    N = 100
    image_resolution = 1000

    points = Array{Float64, 2}([[] []])
    for x = LinRange(0,1,20), y = LinRange(0,1,20)
        if rand() < 0.5
            points = vcat(points, [x y])
        end
    end

    layout = atype(ContinuousLayout(points))
    slm = atype(SLM(N))

    algorithm = WGS(maxiter=100, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.9)

    slm_2, _, _ = match_image(layout, slm, algorithm)
end

begin
    fig = Figure(size = (500, 200))

    ax11 = Axis(fig[1, 1], aspect = DataAspect())
    hm = qgh.heatmap!(ax11, slm_1, image_resolution)
    hidedecorations!(ax11)
    Colorbar(fig[1, 1][1, 2], hm)

    ax12 = Axis(fig[1, 2], aspect = DataAspect())
    hm = qgh.heatmap!(ax12, slm_2, image_resolution)
    hidedecorations!(ax12)
    Colorbar(fig[1, 2][1, 2], hm)

    Label(fig[1, 1, TopLeft()], "(a)",
    # fontsize = 26,
    padding = (0, 0, 0, 0),
    # halign = :right
    )

    Label(fig[1, 2, TopLeft()], "(b)",
    # fontsize = 26,
    padding = (0, 0, 0, 0),
    # halign = :right
    )

    # display(fig)
    save("project/continuous_fourier_transformation/cft_example.png", fig)
end