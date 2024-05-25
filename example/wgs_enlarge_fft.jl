using qgh

let 
    Nu = 512
    Nx = 64
    mask = zeros(Bool, Nu, Nu)
    for i in [[43, 90], [55, 151], [76, 186], [92, 238], [116, 278], [134, 
        317], [138, 377], [127, 417], [148, 475], [182, 471], [211, 
        427], [231, 384], [254, 324], [271, 374], [294, 425], [315, 
        461], [355, 480], [365, 445], [365, 383], [356, 339], [386, 
        285], [408, 242], [431, 211], [446, 159], [461, 108], [431, 
        68], [401, 67], [364, 75], [330, 114], [299, 145], [270, 185], [238,
         196], [211, 151], [183, 117], [156, 87], [119, 60], [82, 51]]
        mask[512-i[2],i[1]] = true
    end
    # heatmap(mask) # show the target image
    layout = GridLayout(mask)
    slm = SLM(Nx)
    algorithm = WGS(ft_method = :fft)
    slm, cost, target_A_reweight = match_image(layout, slm, algorithm)
    # heatmap(embed_locations(layout, target_A_reweight)) # show the original obtained target image
    plot(padding(slm, Nx/Nu))
end