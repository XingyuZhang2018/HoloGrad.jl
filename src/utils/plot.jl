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
        Plots.heatmap!(p[i], abs.(image), colorbar=false, aspect_ratio=:equal, ticks=false, yticks=false)
    end
    heatmap!(p, size=(200*length(slms), 200))
end

plot(slms, N::Int) = plot(slms, N, N)
function plot(slms, Nu, Nv)
    p = Plots.plot(layout=(1, length(slms)))
    for i in 1:length(slms)
        fourier = slms[i].A .* exp.(slms[i].ϕ * (2im * π / slms[i].SLM2π))
        image = cft(fourier, Nu, Nv)
        Plots.heatmap!(p[i], abs.(image), colorbar=false, aspect_ratio=:equal, ticks=false, yticks=false)
    end
    heatmap!(p, size=(200*length(slms), 200))
end

function plot_slms_ϕ_diff(slms)
    p = Plots.plot(layout=(1, length(slms)-1))
    for i in 1:length(slms)-1
        d = ϕdiff(slms[i+1], slms[i])
        Plots.bar!(p[i], 1:length(d), Array(d[:]), legend=false, framestyle=:box, xticks=false)
        Plots.hline!(p[i], Array([mean(abs.(d))]), color=:red, linestyle=:dash)
    end
    Plots.bar!(p, size=(200*length(slms), 200))
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

plot_gif(slms, N) = plot_gif(slms, N, N)
function plot_gif(slms, Nu, Nv)
    p = Plots.plot(layout=(1, 2))
    X, Y = preloc_cft(slms[1].ϕ, Nu, Nv)
    anim = @animate for i in 1:length(slms)
        fourier = slms[i].A .* exp.(slms[i].ϕ * (2im * π / slms[i].SLM2π))
        image = cft_m(fourier, X, Y)
        Plots.heatmap!(p[1], Array(slms[i].ϕ), colorbar=true, clim=(0, slms[i].SLM2π), aspect_ratio=:equal, ticks=false, yticks=false, frame=:none)
        Plots.heatmap!(p[2], Array(abs.(image)), colorbar=true, aspect_ratio=:equal, ticks=false, yticks=false, frame=:none)
    end

    gif(anim, fps=length(slms)÷3)
end

plot_decay(slms, N::Int; kwarg...) = plot_decay(slms, N, N; kwarg...)
plot_decay(p, slms, N::Int; kwarg...) = plot_decay(p, slms, N, N; kwarg...)
function plot_decay(slms, Nu::Int, Nv::Int)
    p = Plots.plot()
    plot_decay(p, slms, Nu, Nv)
end

function plot_decay(p, slms, Nu::Int, Nv::Int; linewidth=2)
    y = []
    X, Y = preloc_cft(slms[1].ϕ, Nu, Nv)
    for i in 1:length(slms)
        fourier = slms[i].A .* exp.(slms[i].ϕ * (2im * π / slms[i].SLM2π))
        image = cft_m(fourier, X, Y)
        push!(y, compute_cost(image))
    end

    x = LinRange(0, 1, length(slms))
    Plots.plot!(p, x, y, 
                label=false, 
                xlim=(0, 1), 
                xlabel="t", 
                ylabel="A variance",
                linewidth=linewidth
                )
    # scatter!(p, x, y, label=false)
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
                ylim=(0, 1), 
                # seriestype = :scatter,
                label="key points = $slices interps = $interps", 
                xlabel="t", 
                ylabel="A_ratio",
                linewidth=linewidth
                )
    # scatter!(p, x, A_ratios, label=false)
end