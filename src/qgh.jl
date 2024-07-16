module qgh

    using CUDA
    using FFTW
    using ForwardDiff
    using KrylovKit
    using LinearAlgebra
    using Statistics
    using Parameters
    using Plots
    using Printf
    using Optim
    using OMEinsum
    using Zygote
    
    import Base: intersect, union, setdiff, Array
    import Plots: plot, plot!, heatmap, heatmap!
    import CUDA: CuArray
    
    export ContinuousLayout, SLM, ϕdiff
    export match_image, WGS, evolution_slm
    export plot, plot_slms_ϕ_diff, plot_gif, plot_decay, heatmap, heatmap!, heatmap_slm_diff!
    
    module Defaults
        const atype = Array 
        const maxiter = 1000
        const tol = 1e-12
        const verbose = true
        const infolder = "./data/"
        const outfolder = "./data/"
    end

    include("configuration.jl")
    include("utils/error_analysis.jl")
    include("utils/plot.jl")
    include("ft.jl")
    include("wgs.jl")
    include("evolution/fixedpoint.jl")
end
