module qgh

    using CUDA
    using FFTW
    using LinearAlgebra
    using Statistics
    using Parameters
    using Plots
    using Printf
    using OMEinsum
    using Zygote
    
    import Base: intersect, union, setdiff
    import Plots: plot
    
    export GridLayout, SLM
    export match_image, WGS
    export plot
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
    include("cft.jl")
    include("wgs.jl")
    include("fixedpoint.jl")
end
