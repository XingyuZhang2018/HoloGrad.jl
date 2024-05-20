using qgh
using qgh: fixed_point_map, get_dϕdt, embed_locations, evolution_slm
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
    target_A_reweight = embed_locations(layout, target_A_reweight)
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
    plot(slm)
    ϕ = fixed_point_map(slm, target_A_reweight)
    @test ϕ ≈ slm.ϕ 
end

let 
    Random.seed!(42)
    layout, slm, target_A_reweight = test_config()
    plot(slm)
end

@testset "∂f/∂A and ∂f/∂ϕ " begin
    Random.seed!(42)
    layout, slm, target_A_reweight = test_config()
    foo1(target_A_reweight) = norm(fixed_point_map(slm, target_A_reweight))
    @test Zygote.gradient(foo1, target_A_reweight)[1] ≈ num_grad(foo1, target_A_reweight) atol = 1e-7

    function foo2(ϕ) 
        s = SLM(slm.A, ϕ, slm.SLM2π)
        norm(fixed_point_map(s, target_A_reweight))
    end

    @test Zygote.gradient(foo2, slm.ϕ)[1] ≈ num_grad(foo2, slm.ϕ) atol = 1e-7
end

@testset "dϕ/dt " begin
    Random.seed!(42)
    layout, slm, target_A_reweight = test_config()
    dAdt = 1e-5 * randn(prod(size(slm.A)), 1)
    dϕdt = get_dϕdt(slm, target_A_reweight, dAdt)
    @test size(dϕdt) == size(dAdt)
end

@testset "evolution_slm " begin
    Random.seed!(42)
    mask = zeros(Bool, 10, 10)
    mask[2:2:8, 2:2:8] .= true
    layout = GridLayout(mask)
    mask = zeros(Bool, 10, 10)
    mask[2:2:8, 4:2:10] .= true
    layout_new = GridLayout(mask)
    slm = SLM(10)

    slm_new = evolution_slm(layout, layout_new, slm, WGS())
    plot(slm_new)
end

let 
    Random.seed!(42)
    mask = zeros(Bool, 10, 10)
    mask[2:2:8, 2:2:6] .= true
    layout = GridLayout(mask)
    mask = zeros(Bool, 10, 10)
    mask[2:2:8, 4:2:8] .= true
    layout_new = GridLayout(mask)
    slm = SLM(10)

    # slm, cost, target_A_reweight = match_image(layout, slm, WGS())
    # plot(slm)

    slm_new = evolution_slm(layout, layout_new, slm, WGS())
    plot(slm_new)
end