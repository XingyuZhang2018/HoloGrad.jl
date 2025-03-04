using qgh
using CairoMakie
using Random
using LinearAlgebra
using CSV
using DataFrames

begin # read data from experiment and plot
    data0 = CSV.read("project/IV. EXPERIMENT/frame_before.csv", DataFrame)[750:1850, 500:1600]
    data1 = CSV.read("project/IV. EXPERIMENT/frame_after.csv", DataFrame)[750:1850, 500:1600]
    x = 1:size(data0, 1)
    y = 1:size(data0, 2)

    fig = Figure(size = (500, 200))
    ax = Axis(fig[1, 1], aspect = DataAspect(), title = L"B_0")
    hm = CairoMakie.heatmap!(ax, x, y, Array(data0); colorrange =(0,500))
    hidedecorations!(ax)
    Colorbar(fig[1, 1][1, 2], hm)

    ax = Axis(fig[1, 2], aspect = DataAspect(), title = L"B_1")
    hm = CairoMakie.heatmap!(ax, x, y, Array(data1); colorrange =(0,500))
    hidedecorations!(ax)
    Colorbar(fig[1, 2][1, 2], hm)

    display(fig)
    save("project/IV. EXPERIMENT/half_fill_to_full_fill.png", fig)
end

begin # read data from experiment and plot
    data0 = CSV.read("project/IV. EXPERIMENT/frame_before.csv", DataFrame)[750:1850, 500:1600]
    data1 = CSV.read("project/IV. EXPERIMENT/frame_after.csv", DataFrame)[750:1850, 500:1600]
    x = 1:size(data0, 1)
    y = 1:size(data0, 2)
    fig = Figure(size = (600, 450))

    ax = Axis(fig[1, 1][1, 1] , aspect = DataAspect(), title = L"B_0")
    hm = CairoMakie.heatmap!(ax, x, y, Array(data0); colorrange =(0,500))
    hidedecorations!(ax)
    cb = Colorbar(fig[1, 1][1, 2], hm)
    cb.alignmode = Mixed(top = 0)
    # cb.alignmode = Mixed(right = 0)
    Label(fig[1, 1, TopLeft()], "(a)",
    # fontsize = 26,
    padding = (0, 14, 0, 0),
    # halign = :right
    )

    ax = Axis(fig[1, 2][1, 1], aspect = DataAspect(), title = L"B_1")
    hm = CairoMakie.heatmap!(ax, x, y, Array(data1); colorrange =(0,500))
    hidedecorations!(ax)
    Colorbar(fig[1, 2][1, 2], hm)
    # cb.alignmode = Mixed(right = 0)
    # colgap!(fa, 100)

    Label(fig[1, 2, TopLeft()], "(b)",
    # fontsize = 26,
    padding = (0, 14, 0, 0),
    # halign = :right
    )

    image_resolution = 2048
    N = 1100

    positions = vec(Array(CSV.read("project/IV. EXPERIMENT/displacement.csv", DataFrame)))
    intensities = vec(Array(CSV.read("project/IV. EXPERIMENT/intensity.csv", DataFrame)))
    
    distances = [norm(positions[i] - positions[1]) for i in 1:length(positions)] / (image_resolution / N)
    # intensities /= maximum(intensities) 

    ax1 = Axis(fig[2, 1], xlabel = "steps", ylabel = L"\Delta x/\mathrm{pixel}")
    CairoMakie.lines!(ax1, 1:length(distances), distances)

    ax2 = Axis(fig[2, 2], xlabel = "steps", ylabel = L"B_{\mathrm{max}}")
    CairoMakie.lines!(ax2, 1:length(intensities), intensities)

    Label(fig[2, 1, TopLeft()], "(c)",
    # fontsize = 26,
    padding = (0, 0, 0, 0),
    # halign = :right
    )

    Label(fig[2, 2, TopLeft()], "(d)",
    # fontsize = 26,
    padding = (0, 14, 0, 0),
    # halign = :right
    )

    rowgap!(fig.layout, 0)
    display(fig)
    save("project/IV. EXPERIMENT/half_fill_to_full_fill.pdf", fig)
end