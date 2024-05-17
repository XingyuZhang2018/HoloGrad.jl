using FFTW
using LinearAlgebra
using qgh
using Random
using Test

@testset "fft" begin
    A = randn(ComplexF64, 1024, 1024)
    B = fft(A)
    C = ifft(B)
    @test A ≈ C

    A = randn(ComplexF64, 1024, 1024)
    B = fftshift(fft(A))
    C = ifft(ifftshift(B))
    @test A ≈ C

    A = normalize(randn(ComplexF64, 1024, 1024))
    B = normalize(fftshift(fft(A)))
    C = normalize(ifft(ifftshift(B)))
    @test A ≈ C
end

@testset "benchmark" begin
    Random.seed!(100)
    mask = zeros(Bool, 1024, 1024)
    mask[100:20:924, 100:20:924] .= true
    layout = GridLayout(mask)
    slm = SLM(1024)
    algorithm = WGS(niters = 1000)
    match_image(layout, slm, algorithm)
end