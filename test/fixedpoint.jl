using qgh
using qgh: fixed_point_map, get_dϕdt
using ForwardDiff
using KrylovKit
using Test
using Zygote

function test_config_continuous()
    points = [rand(2) for _ in 1:10]
    layout = ContinuousLayout(points)
    slm = SLM(10)
    algorithm = WGS(verbose=false)
    slm, cost, B = match_image(layout, slm, algorithm)
    return layout, slm, B
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

    layout, slm, B = test_config_continuous()
    ϕ = fixed_point_map(layout, slm, B)
    @test ϕ ≈ slm.ϕ 
end

@testset "∂f/∂A and ∂f/∂ϕ " begin
    Random.seed!(42)
    layout, slm, B = test_config_continuous()
    ∂f∂B = jacobian(x -> fixed_point_map(layout, slm, x), B)[1]

    dBdt = 1e-5 * randn(size(B)...)
    dfdt = ForwardDiff.derivative(dt -> fixed_point_map(layout, slm, B + dBdt*dt), 0.0)
    @test ∂f∂B * dBdt ≈ reshape(dfdt, prod(size(dfdt)))

    ∂f∂ϕ = jacobian(x -> fixed_point_map(layout, SLM(slm.A, x, slm.SLM2π), B), slm.ϕ)[1]

    dϕdt = rand(size(slm.ϕ)...)
    function linemap(dϕdt)
        dϕdt - ForwardDiff.derivative(dt -> fixed_point_map(layout, SLM(slm.A, slm.ϕ + dϕdt * dt, slm.SLM2π), B), 0.0)
    end
    dϕdt1,  = linsolve(x -> linemap(x), dfdt)
    dϕdt2,  = linsolve(dϕdt -> dϕdt - ∂f∂ϕ * dϕdt, ∂f∂B * dBdt)

    @test dϕdt2 ≈ reshape(dϕdt1, prod(size(slm.ϕ)))
end

@testset "dϕ/dt " begin
    Random.seed!(42)
    layout, slm, B = test_config_continuous()

    dBdt = 1e-5 * randn(size(B)...)
    dxdt = 1e-5 * layout.points
    dϕdt = get_dϕdt(layout, slm, B, dBdt, dxdt)
    @test size(dϕdt) == size(slm.ϕ)
end

@testset "evolution_slm continuous" begin
    Random.seed!(42)
    points = [rand(2) for _ in 1:5]
    layout = ContinuousLayout(points)
    points_new = points .+ 0.1 * [[0, 1]]
    layout_new = ContinuousLayout(points_new)
    slm = SLM(10)

    layouts, slms = evolution_slm(layout, layout_new, slm, WGS(verbose=false); iters=5)
    # for i in 1:length(slms)-1
    #     d = ϕdiff(slms[i+1], slms[i])
    #     println(maximum(abs.(d)))
    # end
    @test length(slms) == 6
    # plot(layouts, slms)
    # plot(slms)
end