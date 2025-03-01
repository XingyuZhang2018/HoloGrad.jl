using qgh
using CairoMakie
using Random
using LinearAlgebra

begin # slms_exact data
    Random.seed!(42)

    include("../../example/layout_example.jl")
    atype = Array # or CuArray
    N = 100

    layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
    layout_end = atype(layout_example(Val(:butterflycontinuous); α = 0.2))
    slm = atype(SLM(N))

    algorithm = WGS(maxiter=100, verbose=false, show_every=10, tol=1e-10, ratio_fixphase=0.9)

    #benchmark
    layouts_exact, slms_exact = evolution_slm_direct(layout, layout_end, slm, algorithm; keypoints=1)
end

begin # plot B_0 and B_1
    image_resolution = N*3

    fig = Figure(size = (400, 400), figure_padding = 0)
    ax11 = Axis(fig[1, 1], aspect = DataAspect())
    hm = qgh.heatmap!(ax11, slms_exact[1], image_resolution)
    hidedecorations!(ax11)

    save("project/butterfly_flits_wings/main_butterfly_B_1.png", fig)
end

begin # plot B_0 and B_1
    image_resolution = N*3

    fig = Figure(size = (400, 400), figure_padding = 0)
    ax11 = Axis(fig[1, 1], aspect = DataAspect())

    hm = qgh.heatmap!(ax11, slms_exact[end], image_resolution)
    hidedecorations!(ax11)

    xs = layout.points[:, 1] * image_resolution
    ys = layout.points[:, 2] * image_resolution
    d(x) = min(0.1 * abs(x-0.5), 0.02)
    move_arrow(x) = x < 0.5 ? x-d(x) : x+d(x)
    us = move_arrow.(layout_end.points[:, 1]) * image_resolution - xs 
    vs = layout_end.points[:, 2] * image_resolution - ys 
    CairoMakie.arrows!(ax11, xs, ys, us, vs, arrowsize = 12, lengthscale = 1, arrowcolor = :red, linecolor = :red)
    fig
    save("project/butterfly_flits_wings/main_butterfly_B_2.pdf", fig)
end

begin # plot B_0 and B_1
    image_resolution = N*3

    fig = Figure(size = (400, 400), figure_padding = 0)
    ax11 = Axis(fig[1, 1], aspect = DataAspect())

    hm = qgh.heatmap!(ax11, slms_exact[end], image_resolution)
    hidedecorations!(ax11)

    xs = layout.points[:, 1] * image_resolution
    ys = layout.points[:, 2] * image_resolution
    us = layout_end.points[:, 1]* image_resolution - xs 
    vs = layout_end.points[:, 2]* image_resolution - ys 
    CairoMakie.arrows!(ax11, xs, ys, us, vs, arrowsize = 12, lengthscale = 1, arrowcolor = :red, linecolor = :red)
    fig
    # save("project/butterfly_flits_wings/main_butterfly_B_2.png", fig)
end

begin # ϕ
    fig = Figure(size = (400, 400), figure_padding = 0)
    ax11 = Axis(fig[1, 1], aspect = DataAspect())
    hm = CairoMakie.heatmap!(ax11, 1:N, 1:N, slms_exact[1].ϕ/ slms_exact[1].SLM2π * 2π, colorrange = (-pi, pi))
    # Colorbar(fig[1, 1][1, 2], hm, ticks = (-pi:pi/2:pi, [L"-\pi", L"-\frac{\pi}{2}", L"0", L"\frac{\pi}{2}", L"\pi"]), ticklabelsize = 12)
    hidedecorations!(ax11)
    # fig
    save("project/butterfly_flits_wings/main_butterfly_phi.png", fig)
end


begin
    layout = atype(layout_example(Val(:butterflycontinuous); α = 0.00))
    layout_end = atype(layout_example(Val(:butterflycontinuous); α = 0.01))
    slm = atype(SLM(N))
    
    slm, cost, B = match_image(layout, slm, algorithm) 
    dxdt = layout_end.points - layout.points
    Δϕ = qgh.get_dϕdt(layout, slm, B, dxdt, 15)

    fig = Figure(size = (400, 400), figure_padding = 0)
    ax11 = Axis(fig[1, 1], aspect = DataAspect())
    hm = CairoMakie.heatmap!(ax11, 1:N, 1:N, Δϕ/ slms_exact[1].SLM2π * 2π, colorrange = (-pi, pi))
    hidedecorations!(ax11)
    # fig
    save("project/butterfly_flits_wings/main_butterfly_Delta_phi.png", fig)
end

