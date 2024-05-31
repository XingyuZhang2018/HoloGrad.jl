using qgh
using qgh: fixed_point_map, get_dϕdt, embed_locations, evolution_slm
using ForwardDiff
using KrylovKit
using LinearAlgebra
using Random
using Test
using Zygote

function test_config_grid()
    mask = zeros(Bool, 10, 10)
    mask[2:2:8, 2:2:8] .= true
    layout = GridLayout(mask)
    slm = SLM(10)
    algorithm = WGS(verbose=false)
    slm, cost, target_A_reweight = match_image(layout, slm, algorithm)
    return layout, slm, target_A_reweight
end

function test_config_continuous()
    points = [rand(2) for _ in 1:10]
    layout = ContinuousLayout(points)
    slm = SLM(10)
    algorithm = WGS(verbose=false, ft_method=:cft)
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
    layout, slm, target_A_reweight = test_config_grid()
    ϕ = fixed_point_map(layout, slm, target_A_reweight, WGS())
    @test ϕ ≈ slm.ϕ 

    layout, slm, target_A_reweight = test_config_continuous()
    ϕ = fixed_point_map(layout, slm, target_A_reweight, WGS(ft_method=:cft))
    @test ϕ ≈ slm.ϕ 
end

@testset "∂f/∂A and ∂f/∂ϕ " begin
    Random.seed!(42)
    layout, slm, target_A_reweight = test_config_continuous()
    algorithm = WGS(ft_method=:cft)
    ∂f∂A = jacobian(x -> fixed_point_map(layout, slm, x, algorithm), target_A_reweight)[1]

    dAdt = 1e-5 * randn(size(target_A_reweight)...)
    dfdt = ForwardDiff.derivative(dt -> fixed_point_map(layout, slm, target_A_reweight + dAdt*dt, algorithm), 0.0)
    @test ∂f∂A * dAdt ≈ reshape(dfdt, prod(size(dfdt)))

    ∂f∂ϕ = jacobian(x -> fixed_point_map(layout, SLM(slm.A, x, slm.SLM2π), target_A_reweight, algorithm), slm.ϕ)[1]

    dϕdt = rand(size(slm.ϕ)...)
    function linemap(dϕdt)
        dϕdt - ForwardDiff.derivative(dt -> fixed_point_map(layout, SLM(slm.A, slm.ϕ + dϕdt * dt, slm.SLM2π), target_A_reweight, algorithm), 0.0)
    end
    dϕdt1,  = linsolve(x -> linemap(x), dfdt)
    dϕdt2,  = linsolve(dϕdt -> dϕdt - ∂f∂ϕ * dϕdt, ∂f∂A * dAdt)

    @test dϕdt2 ≈ reshape(dϕdt1, prod(size(slm.ϕ)))
end

@testset "dϕ/dt " begin
    Random.seed!(42)
    layout, slm, target_A_reweight = test_config_continuous()
    dAdt = 1e-5 * randn(size(target_A_reweight)...)
    dxdt = 1e-5 * layout.points
    dϕdt = get_dϕdt(layout, slm, target_A_reweight, dAdt, dxdt, WGS(ft_method=:cft))
    @test size(dϕdt) == size(slm.ϕ)
end

@testset "evolution_slm continuous" begin
    Random.seed!(42)
    points = [rand(2) for _ in 1:5]
    layout = ContinuousLayout(points)
    points_new = points .+ 0.1 * [[0, 1]]
    layout_new = ContinuousLayout(points_new)
    slm = SLM(10)

    # v =  0.1 * [zeros(1), ones(1)]
    layouts, slms = evolution_slm(layout, layout_new, slm, WGS(verbose=false, ft_method=:cft); iters=5)
    for i in 1:length(slms)-1
        d = ϕdiff(slms[i+1], slms[i])
        println(maximum(abs.(d)))
    end
    @test length(slms) == 6
    # plot(layouts, slms)
    # plot(slms)
end