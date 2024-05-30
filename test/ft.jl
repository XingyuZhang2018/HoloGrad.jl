using qgh
using qgh: mft, imft, smft, ismft, padding, padding_extract, ft, ift, cft, icft
using LinearAlgebra
using Test
using Random
using FFTW

@testset "padding" begin
    Random.seed!(42)
    f = randn(ComplexF64, 10, 10)
    @test padding_extract(padding(f, 0.5),0.5) == f
end

@testset "matrix fourier transform" begin
    Random.seed!(42)
    f = randn(ComplexF64, 10, 10)
    @test mft(f, 10, 10) ≈ fft(f)

    f = randn(ComplexF64, 10, 10)
    @test ifft(fft(f)) ≈ f
    @test imft(mft(f, 10, 10), 10, 10) ≈ f

    @test mft(f, 20, 20) ≈ fft(padding(f, 0.5))
    @test imft(mft(f, 20, 20), 10, 10) ≈ f
end

@testset "sparse matrix fourier transform" begin
    Random.seed!(42)
    f = randn(ComplexF64, 10, 10)

    mask = zeros(Bool, 20, 20)
    mask[2:2:20, 2:2:20] .= true
    layout = GridLayout(mask)

    F = smft(layout, f)
    @test size(F,1) == layout.ntrap
    f_new = ismft(layout, F, 10, 10)
    @test normalize(f) ≈ normalize(f_new)
end

@testset "continuous fourier transform" begin
    Random.seed!(42)
    f = randn(ComplexF64, 10, 10)
    points = [rand(2) for _ in 1:10]
    layout = ContinuousLayout(points)

    F = cft(layout, f)
    @test size(F,1) == layout.ntrap
    f_new = icft(layout, F, 10, 10)
    # @test normalize(f) ≈ normalize(f_new)
    # @show normalize(f) - normalize(f_new)
end

@testset "ft api with $ft_method" for ft_method in [:fft, :mft, :smft] 
    f = randn(ComplexF64, 10, 10)

    mask = zeros(Bool, 20, 20)
    mask[2:2:20, 2:2:20] .= true
    layout = GridLayout(mask)

    F = ft(layout, f, 0.5, Val(ft_method))
    f_new = ift(layout, F, 0.5, Val(ft_method))
    @test normalize(f) ≈ normalize(f_new)
end