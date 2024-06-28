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
    
    import Base: intersect, union, setdiff
    import Plots: plot
    
    export ContinuousLayout, SLM, ϕdiff
    export match_image, WGS, evolution_slm
    export plot, plot_slms_ϕ_diff, plot_gif
    
    module Defaults
        const atype = Array 
        const maxiter = 1000
        const tol = 1e-12
        const verbose = true
        const infolder = "./data/"
        const outfolder = "./data/"
    end

    include("configuration.jl")
    include("utils.jl")
    include("ft.jl")
    include("wgs.jl")
    include("fixedpoint.jl")
end
