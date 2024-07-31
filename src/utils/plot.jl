function plot(slm::SLM)
    fourier = slm.A .* exp.(slm.ϕ * (2im * π / slm.SLM2π))
    image = normalize(fft(fourier))
    heatmap(abs.(image))
end

function plot(layout::ContinuousLayout, slm::SLM)
    fourier = slm.A .* exp.(slm.ϕ * (2im * π / slm.SLM2π))
    Nx, Ny = size(slm.A)
    F = abs.(normalize(cft(layout, fourier)))
    image = zeros(size(slm.A)...)
    for i in 1:length(layout.points)
        image[(Int, clamp!([layout.points[i][1] * Nx, layout.points[i][2] * Ny], 1.0, Nx))...] = F[i]
    end
    heatmap(abs.(image))
end

function plot(slms)
    p = Plots.plot(layout=(1, length(slms)))
    for i in 1:length(slms)
        fourier = slms[i].A .* exp.(slms[i].ϕ * (2im * π / slms[i].SLM2π))
        image = normalize!(fft(fourier))
        Plots.heatmap!(p[i], Array(abs.(image)), colorbar=false, aspect_ratio=:equal, ticks=false, yticks=false)
    end
    heatmap!(p, size=(200*length(slms), 200))
end

function plot(layouts, slms)
    p = Plots.plot(layout=(1, length(slms)))
    Nx, Ny = size(slms[1].A)
    for i in 1:length(slms)
        fourier = slms[i].A .* exp.(slms[i].ϕ * (2im * π / slms[i].SLM2π))
        X, Y, iX, iY = preloc_cft(layouts[i], fourier)
        F = abs.(normalize(cft(fourier, X, Y)))
        image = zeros(size(slms[i].A)...)
        for j in 1:size(layouts[i].points, 1)
            image[round.(Int, clamp!([layouts[i].points[j, 1] * Nx, layouts[i].points[j, 2] * Ny], 1.0, Nx))...] = F[j]
        end
        Plots.heatmap!(p[i], abs.(image), colorbar=false, aspect_ratio=:equal, ticks=false)
    end
    heatmap!(p, size=(200*length(slms), 200))
end

heatmap(slms, N::Int; kwarg...) = heatmap(slms, N, N; kwarg...)
function heatmap(slms, Nu, Nv; kwarg...)
    fig = Figure(size = (200*length(slms), 200); kwarg...)
    for i in 1:length(slms)
        ax = Axis(fig[1, i])
        heatmap!(ax, slms[i], Nu, Nv; kwarg...)
    end
    Colorbar(fig[:, end+1]; kwarg...)
    fig
end

heatmap!(ax, slm::SLM, N::Int; kwarg...) = heatmap!(ax, slm::SLM, N, N; kwarg...)
function heatmap!(ax, slm::SLM, Nu::Int, Nv::Int; kwarg...)
    fourier = slm.A .* exp.(slm.ϕ * (2im * π / slm.SLM2π))
    X, Y = preloc_cft(fourier , Nu, Nv)
    image = abs.(cft_m(fourier, X, Y))
    image /= maximum(image)

    CairoMakie.heatmap!(ax, 1:size(image, 1), 1:size(image, 2), Array(image); kwarg...)
end

heatmap_slm_diff!(p, slm1::SLM, slm2::SLM, N::Int; kwarg...) = heatmap_slm_diff!(p, slm1, slm2, N, N; kwarg...)
function heatmap_slm_diff!(p, slm1::SLM, slm2::SLM, Nu::Int, Nv::Int; kwarg...)
    X, Y = preloc_cft(slm1.ϕ, Nu, Nv)
    fourier1 = slm1.A .* exp.(slm1.ϕ * (2im * π / slm1.SLM2π))
    fourier2 = slm2.A .* exp.(slm2.ϕ * (2im * π / slm2.SLM2π))
    image1 = cft_m(fourier1, X, Y)
    image2 = cft_m(fourier2, X, Y)
    Plots.heatmap!(p, Array(abs.(abs.(image1) .- abs.(image2))), 
                   aspect_ratio=:equal, 
                   ticks=false, 
                   frame=:none;
                   kwarg...
                   )
end

function plot_slms_ϕ_diff!(ax, slms; kwarg...)
    for i in 1:length(slms)-1
        d = ϕdiff(slms[i+1], slms[i])
        @show compute_cost(d), mean(abs.(d))
        CairoMakie.barplot!(ax, 1:length(d), Array(d[:]), strokewidth = 0.00001; kwarg...)
        CairoMakie.hlines!(ax, Array([mean(abs.(d))]); kwarg...)
    end
end

function plot_slms_ϕ_diff(p, slm::SLM, slm_new::SLM)
    d = ϕdiff(slm, slm_new)
    Plots.bar!(p, 1:length(d), Array(d[:]), legend=false, framestyle=:box, xticks=false)
    Plots.hline!(p, Array([mean(abs.(d))]), color=:red, linestyle=:dash)
end

function plot_gif(slms)
    p = Plots.plot(layout=(1, 2))
    anim = @animate for i in 1:length(slms)
        fourier = slms[i].A .* exp.(slms[i].ϕ * (2im * π / slms[i].SLM2π))
        image = normalize!(fft(fourier))
        Plots.heatmap!(p[1], Array(slms[i].ϕ), colorbar=true, clim=(0, slms[i].SLM2π), aspect_ratio=:equal, ticks=false, yticks=false, frame=:none)
        Plots.heatmap!(p[2], Array(abs.(image)), colorbar=true, aspect_ratio=:equal, ticks=false, yticks=false, frame=:none)
    end

    gif(anim, fps=length(slms)÷3)
end

image_animation(slms, N; file, kwarg...) = image_animation(slms, N, N; file, kwarg...)
function image_animation(slms, Nu, Nv; file, kwarg...)
    fig = Figure()
    ax = Axis(fig[1, 1], aspect = DataAspect())
    X, Y = preloc_cft(slms[1].ϕ, Nu, Nv)
    iter = Observable(1)
    fourier = @lift(slms[$iter].A .* exp.(slms[$iter].ϕ * (2im * π / slms[$iter].SLM2π)))
    image = @lift(Array(abs.(cft_m($fourier, X, Y))))
    CairoMakie.heatmap!(ax, 1:Nu, 1:Nv, image; kwarg...)
    hidedecorations!(ax)

    CairoMakie.record(fig, "$file", 1:length(slms); framerate=length(slms)÷3) do i
        iter[] = i
    end
end

plot_decay(slms_exact, slms_flow, N::Int; kwarg...) = plot_decay(slms_exact, slms_flow, N, N; kwarg...)
plot_decay(p, slms_exact, slms_flow, N::Int; kwarg...) = plot_decay(p, slms_exact, slms_flow, N, N; kwarg...)
function plot_decay(slms_exact, slms_flow, Nu::Int, Nv::Int; kwarg...)
    p = Plots.plot()
    plot_decay(p, slms_exact, slms_flow, Nu, Nv; kwarg...)
end

function plot_decay(p, slms_exact, slms_flow, Nu::Int, Nv::Int; kwarg...)
    y = []
    X, Y = preloc_cft(slms_exact[1].ϕ, Nu, Nv)
    for i in 1:length(slms_exact)
        fourier_exact = slms_exact[i].A .* exp.(slms_exact[i].ϕ * (2im * π / slms_exact[i].SLM2π))
        image_exact = cft_m(fourier_exact, X, Y)
        fourier_flow = slms_flow[i].A .* exp.(slms_flow[i].ϕ * (2im * π / slms_flow[i].SLM2π))
        image_flow = cft_m(fourier_flow, X, Y)
        push!(y, kl_divergence(image_exact, image_flow))
    end

    x = LinRange(0, 1, length(slms_exact))
    Plots.plot!(p, x, y, 
                label=false, 
                xlim=(0, 1), 
                xlabel="t", 
                ylabel="KL divergence";
                kwarg...)
    Plots.scatter!(p, x, y, label=false; kwarg...)
end

function plot_decay(layouts, slms, slices, interps)
    p = Plots.plot()
    plot_decay(p, layouts, slms, slices, interps)
end

function plot_decay(p, layouts, slms, slices, interps; linewidth=2)
    layout = layouts[1]
    A_ratios = [1.0]
    for i in 1:slices
        index = (i-1)*interps+1
        index_new = i*interps
        layout = layouts[index]
        layout_new = layouts[index_new]
        slm = slms[index]
        slm_new = slms[index_new]

        _, trap_A_mean = compute_cost(slm, layout)
        _, trap_A_mean_new = compute_cost(slm_new, layout_new)
        for j in 1:interps
            slm_interp = slms[index + j - 1]
            layout_interp = layouts[index + j - 1]
            if j <= interps/2
                variance, A_ratio = compute_cost(slm_interp, layout_interp, trap_A_mean)
            else
                variance, A_ratio = compute_cost(slm_interp, layout_interp, trap_A_mean_new)
            end
            push!(A_ratios,  clamp(A_ratio, 0, 1))
        end
    end

    x = LinRange(0, 1, length(slms))
    Plots.plot!(p, x, A_ratios, 
                xlim=(0, 1), 
                # ylim=(0, 1), 
                # seriestype = :scatter,
                label="key points = $slices interps = $interps", 
                xlabel="t", 
                ylabel="A_ratio",
                linewidth=linewidth
                )
    # scatter!(p, x, A_ratios, label=false)
end

plot_adjacent_KL(p, slms, N::Int; kwarg...) = plot_adjacent_KL(p, slms, N, N; kwarg...)
function plot_adjacent_KL(p, slms, Nu::Int, Nv::Int; kwarg...)
    y = []
    X, Y = preloc_cft(slms[1].ϕ, Nu, Nv)
    fourier = slms[1].A .* exp.(slms[1].ϕ * (2im * π / slms[1].SLM2π))
    image = cft_m(fourier, X, Y)
    for i in 2:length(slms)
        fourier = slms[i].A .* exp.(slms[i].ϕ * (2im * π / slms[i].SLM2π))
        image_new = cft_m(fourier, X, Y)
        push!(y, kl_divergence(image, image_new))

        image = image_new
    end

    x = 1:length(slms)-1
    Plots.plot!(p, x, y, 
                label=false, 
                xlim=(0, 1), 
                xlabel="t", 
                ylabel="KL divergence";
                kwarg...)
    Plots.scatter!(p, x, y, label=false; kwarg...)
end

plot_distance_intensity_decay!(f, slms, areas, N::Int; kwarg...) = plot_distance_intensity_decay!(f, slms, areas, N, N; kwarg...)
function plot_distance_intensity_decay!(f, slms, areas, Nu::Int, Nv::Int; kwarg...)
    positions = []
    intensities = Array{Float64}([])
    
    for i in 1:length(slms)
        intensity, position = find_max_intensity(slms[i], areas[i], Nu, Nv)
        push!(positions, position)
        push!(intensities, intensity)
    end
    distances = [norm(positions[i] - positions[1]) for i in 1:length(positions)]

    ax1 = Axis(f[1, 1], xlabel = "time", ylabel = "distance")
    ax2 = Axis(f[1, 2], xlabel = "intensity", ylabel = "distance", aspect = 0.25)
    linkyaxes!(ax1, ax2)

    t = LinRange(0, 1, length(slms))
    CairoMakie.lines!(ax1, t, distances; kwarg...)
    CairoMakie.scatter!(ax1, t, distances; kwarg...)

    CairoMakie.lines!(ax2, intensities, distances; kwarg...)
    CairoMakie.poly!(ax2, [intensities; fill(0, length(intensities))], [distances; reverse(distances)], color = (:blue, 0.3))
    CairoMakie.xlims!(ax2, low = 0)
    CairoMakie.hideydecorations!(ax2, grid = false)
end

function plot_rectange!(ax, area)
    vertices = Point2f[area[1], (area[2][1], area[1][2]), area[2], (area[1][1], area[2][2])]
    poly!(ax, vertices, color = (:red, 0.0), strokewidth = 1, strokecolor = :red)
end

function find_area(layout, layout_end, image_resolution, keypoints; area_size = 0.01)
    interval = layout_end.points - layout.points
    interval_norm = [norm(interval[i, :]) for i in 1:size(interval, 1)]
    interval_max, index = findmax(interval_norm)
    select_points = [layout.points[index, :] + interval[index, :] * i / keypoints for i in 0:keypoints]
    area = [[point .- area_size/2, point .+ area_size/2] .* image_resolution for point in select_points]

    return area
end