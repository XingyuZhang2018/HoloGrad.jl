using qgh

let 
    include("layout_example.jl")
    Nu = 512
    Nx = 64

    layout = layout_example(Val(:butterfly))
    slm = SLM(Nx)
    algorithm = WGS(ft_method = :fft)
    slm, cost, target_A_reweight = match_image(layout, slm, algorithm)
    # heatmap(embed_locations(layout, target_A_reweight)) # show the original obtained target image
    plot(padding(slm, Nx/Nu))
end