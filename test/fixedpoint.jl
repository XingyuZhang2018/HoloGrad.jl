using qgh
using qgh: fixed_point_map, get_dϕBdt
using ForwardDiff
using KrylovKit
using Test
using Zygote
using Random

function test_config_continuous(atype)
    points = atype(rand(10, 2))
    layout = ContinuousLayout(points)
    slm = atype(SLM(10))
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

@testset "fixed_point_map with $atype" for atype in test_atypes
    Random.seed!(42)

    layout, slm, B = test_config_continuous(atype)
    ϕ_new, B_new = fixed_point_map(layout, slm, B)
    @test ϕ_new ≈ slm.ϕ 
    @test B_new ≈ B
end

@testset "∂f/∂A and ∂f/∂ϕ with $atype" for atype in test_atypes
    Random.seed!(42)
    layout, slm, B = test_config_continuous(atype)
    # ∂f∂x = ForwardDiff.jacobian(x -> fixed_point_map(ContinuousLayout(x), slm, B), layout.points)[1]

    # dxdt = [1e-5 * randn(2) for _ in 1:length(layout.points)]
    # dfdt = ForwardDiff.derivative(dt -> fixed_point_map(ContinuousLayout(layout.points + dxdt*dt), slm, B), 0.0)
    # @test size(dfdt[1]) == size(slm.ϕ)
    # @test size(dfdt[2]) == size(B)

    # ∂f∂ϕ = jacobian(x -> fixed_point_map(layout, SLM(slm.A, x, slm.SLM2π), B), slm.ϕ)[1]

    # dϕdt = rand(size(slm.ϕ)...)
    # function linemap(dϕdt)
    #     dϕdt - ForwardDiff.derivative(dt -> fixed_point_map(layout, SLM(slm.A, slm.ϕ + dϕdt * dt, slm.SLM2π), B), 0.0)
    # end
    # dϕdt1,  = linsolve(x -> linemap(x), dfdt)
    # dϕdt2,  = linsolve(dϕdt -> dϕdt - ∂f∂ϕ * dϕdt, ∂f∂B * dBdt)

    # @test dϕdt2 ≈ reshape(dϕdt1, prod(size(slm.ϕ)))

end

@testset "dϕ/dt with $atype" for atype in [Array]
    Random.seed!(42)
    layout, slm, B = test_config_continuous(atype)

    dxdt = 1e-5 * layout.points
    dϕBdt = get_dϕBdt(layout, slm, B, dxdt)
    @test size(dϕBdt[1]) == size(slm.ϕ)
    @test size(dϕBdt[2]) == size(B)
end

@testset "evolution_slm continuous with $atype" for atype in test_atypes
    Random.seed!(42)
    points = atype(rand(5, 2))
    layout = ContinuousLayout(points)
    points_new = points + 1e-5 * atype(randn(size(points)))
    layout_new = ContinuousLayout(points_new)
    slm = atype(SLM(10))

    layouts, slms = evolution_slm(layout, layout_new, slm, WGS(verbose=false);
                                   slices=5, 
                                   interps=1, 
                                   aditers=5,
                                   ifflow=true)
    #     d = ϕdiff(slms[i+1], slms[i])
    #     println(maximum(abs.(d)))
    # end
    @test length(slms) == 6
    # plot(layouts, slms)
    # plot(slms)
end