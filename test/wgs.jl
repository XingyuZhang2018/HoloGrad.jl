using FFTW
using LinearAlgebra
using qgh
using qgh: deal_layout, extract_locations
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

@testset "deal_layout" begin
    layout = GridLayout(Array{Bool}([0 0 0; 
                                     1 1 0; 
                                     1 0 0]))
    layout_new = GridLayout(Array{Bool}([0 0 1; 
                                         0 1 1; 
                                         0 0 0]))

    layout_union, layout_disappear, layout_appear = deal_layout(layout, layout_new)
    @test layout_union.mask == [0 0 1; 
                                1 1 1; 
                                1 0 0]
    @test layout_disappear.mask == [0 0 0; 
                                    1 0 0; 
                                    1 0 0]
    @test layout_appear.mask == [0 0 1; 
                                 0 0 1; 
                                 0 0 0]

    @test extract_locations(layout_union, layout_appear.mask) == [0,0,0,1,1]
    @test extract_locations(layout_union, layout_disappear.mask) == [1,1,0,0,0]
end

@testset "wgs evolution" begin
    Random.seed!(100)
    mask = zeros(Bool, 10, 10)
    mask[2:2:8, 2:2:8] .= true
    layout = GridLayout(mask)
    slm = SLM(10)
    algorithm = WGS()
    slm, cost, target_A_reweight = match_image(layout, slm, algorithm)
    @test cost < 1e-6

    mask = zeros(Bool, 10, 10)
    mask[2:2:8, 4:2:10] .= true
    layout_new = GridLayout(mask)
    slm, = match_image(layout, layout_new, slm, 0.0, algorithm)
    @test cost < 1e-6
end