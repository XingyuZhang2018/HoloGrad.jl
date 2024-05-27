using qgh
using qgh: mft, imft, smft, ismft, padding, padding_extract, ft, ift
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

@testset "ft api" begin
    f = randn(ComplexF64, 10, 10)

    mask = zeros(Bool, 20, 20)
    mask[2:2:20, 2:2:20] .= true
    layout = GridLayout(mask)

    F_fft = ft(layout, f, 0.5, Val(:fft))
    F_mft = ft(layout, f, 0.5, Val(:mft))
    F_smft = ft(layout, f, 0.5, Val(:smft))

    f_fft = ift(layout, F_fft, 0.5, Val(:fft))
    f_mft = ift(layout, F_mft, 0.5, Val(:mft))
    f_smft = ift(layout, F_smft, 0.5, Val(:smft))

    @test normalize(f) ≈ normalize(f_fft) ≈ normalize(f_mft) ≈ normalize(f_smft)
end