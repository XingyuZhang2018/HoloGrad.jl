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

@testset "wgs" begin
    Random.seed!(100)
    mask = zeros(Bool, 10, 10)
    mask[2:2:8, 2:2:8] .= true
    layout = GridLayout(mask)
    slm = SLM(10)
    algorithm = WGS()
    slm, cost = match_image(layout, slm, algorithm)
    @test cost < 1e-6
end