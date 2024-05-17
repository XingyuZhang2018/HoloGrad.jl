module qgh

    using CUDA
    using FFTW
    using LinearAlgebra
    using Statistics
    using Parameters 
    using Printf
    using Zygote
    
    export GridLayout, SLM
    export match_image, WGS
    module Defaults
        const atype = Array 
        const maxiter = 100
        const tol = 1e-12
        const verbose = true
        const infolder = "./data/"
        const outfolder = "./data/"
    end

    include("configuration.jl")
    include("wgs.jl")
end
