using qgh
using Random
using FFTW
FFTW.set_num_threads(2)

let  
    Random.seed!(42)

    # initial SLM and layout
    mask = zeros(Bool, 1024, 1024)
    mask[100:20:924, 100:20:924] .= true
    layout = GridLayout(mask)
    slm = SLM(1024)

    # run the WGS algorithm
    algorithm = WGS(maxiter = 1000, ratio_fixphase = 0.5, ft_method = :fft)
    slm, cost = match_image(layout, slm, algorithm)

    # plot the result
    plot(slm)
end