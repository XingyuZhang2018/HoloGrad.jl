using qgh
using qgh: fixed_point_map
using LinearAlgebra
using Random
using Test
using Zygote

function test_config()
    mask = zeros(Bool, 10, 10)
    mask[2:2:8, 2:2:8] .= true
    layout = GridLayout(mask)
    slm = SLM(10)
    algorithm = WGS(verbose=false)
    slm, cost, target_A_reweight = match_image(layout, slm, algorithm)
    return layout, slm, target_A_reweight
end

function num_grad(f, K; δ::Real=1e-5)
    if eltype(K) == ComplexF64
        (f(K + δ / 2) - f(K - δ / 2)) / δ + 
            (f(K + δ / 2 * 1.0im) - f(K - δ / 2 * 1.0im)) / δ * 1.0im
    else
        (f(K + δ / 2) - f(K - δ / 2)) / δ
    end
end

function num_grad(f, a::AbstractArray; δ::Real=1e-5)
    b = Array(copy(a))
    df = map(CartesianIndices(b)) do i
        foo = x -> (ac = copy(b); ac[i] = x; f(ac))
        num_grad(foo, b[i], δ=δ)
    end
    return (df)
end

@testset "fixed_point_map" begin
    Random.seed!(42)
    layout, slm, target_A_reweight = test_config()
    ϕ = fixed_point_map(layout, slm, target_A_reweight)
    @test ϕ ≈ slm.ϕ 
end

@testset "∂f/∂A and ∂f/∂ϕ " begin
    Random.seed!(42)
    layout, slm, target_A_reweight = test_config()
    foo1(target_A_reweight) = norm(fixed_point_map(layout, slm, target_A_reweight))
    @test Zygote.gradient(foo1, target_A_reweight)[1] ≈ num_grad(foo1, target_A_reweight) atol = 1e-7

    function foo2(ϕ) 
        s = SLM(slm.A, ϕ, slm.SLM2π)
        norm(fixed_point_map(layout, s, target_A_reweight))
    end

    @test Zygote.gradient(foo2, slm.ϕ)[1] ≈ num_grad(foo2, slm.ϕ) atol = 1e-7
end