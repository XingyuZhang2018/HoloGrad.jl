using qgh
using qgh: ft_enlarge, ift_reduce, padding, padding_extract, ft, ift
using LinearAlgebra
using Test
using Random
using FFTW


@testset "compare with fft" begin
    Random.seed!(42)
    f = randn(ComplexF64, 10, 10)
    @test ft_enlarge(f, 10, 10) ≈ fft(f)

    f = randn(ComplexF64, 10, 10)
    @test ifft(fft(f)) ≈ f
    @test ift_reduce(ft_enlarge(f, 10, 10), 10, 10) ≈ f
end

@testset "padding and enlarge" begin
    Random.seed!(42)
    f = randn(ComplexF64, 10, 10)
    @test padding_extract(padding(f, 0.5),0.5) == f
    @test ft_enlarge(f, 20, 20) ≈ fft(padding(f, 0.5))
    @test ift_reduce(ft_enlarge(f, 20, 20), 10, 10) ≈ f
end

@testset "ft with layout" begin
    Random.seed!(42)
    f = randn(ComplexF64, 10, 10)

    mask = zeros(Bool, 20, 20)
    mask[1:1:20, 1:1:20] .= true
    layout = GridLayout(mask)

    F = ft(layout, f)
    @test size(F,1) == layout.ntrap

    f_new = ift(layout, F, 10, 10)
    @test normalize(f) ≈ normalize(f_new)
end