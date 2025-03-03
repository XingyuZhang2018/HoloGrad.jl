using qgh
using qgh: fixed_point_map, get_dϕBdt
using ForwardDiff
using KrylovKit
using Test
using Zygote
using Random
using LinearAlgebra

function test_config_continuous(atype)
    points = atype(rand(5, 2))
    layout = ContinuousLayout(points)
    slm = atype(SLM(5))
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
    @test sum(abs.(ϕ_new - slm.ϕ)) < length(slm.ϕ)
    @test B_new ≈ B atol = 1e-2
end

@testset "implicit function theorem only with ϕ with $atype" for atype in test_atypes
    Random.seed!(42)
    layout, slm, B = test_config_continuous(atype)
    ∂f∂B = ForwardDiff.jacobian(x -> fixed_point_map(layout, slm, x)[1], B)

    dBdt = 1e-5 * randn(size(B)...)
    dfdt = ForwardDiff.derivative(dt -> fixed_point_map(layout, slm, B + dBdt*dt)[1], 0.0)
    @test ∂f∂B * dBdt ≈ reshape(dfdt, prod(size(dfdt)))

    ∂f∂ϕ = ForwardDiff.jacobian(x -> fixed_point_map(layout, SLM(slm.A, x, slm.SLM2π), B)[1], slm.ϕ)

    function linemap(dϕdt)
        dϕdt - ForwardDiff.derivative(dt -> fixed_point_map(layout, SLM(slm.A, slm.ϕ + dϕdt * dt, slm.SLM2π), B)[1], 0.0)
    end
    dϕdt1,  = linsolve(x -> linemap(x), dfdt)
    dϕdt2,  = linsolve(dϕdt -> dϕdt - ∂f∂ϕ * dϕdt, ∂f∂B * dBdt)

    @test dϕdt2 ≈ reshape(dϕdt1, prod(size(slm.ϕ)))
end

@testset "implicit function theorem with ϕ and B with $atype with $atype" for atype in test_atypes
    Random.seed!(42)
    layout, slm, B = test_config_continuous(atype)
    ∂fϕ∂x = ForwardDiff.jacobian(x -> fixed_point_map(ContinuousLayout(x), slm, B)[1], layout.points)
    ∂fB∂x = ForwardDiff.jacobian(x -> fixed_point_map(ContinuousLayout(x), slm, B)[2], layout.points)
    
    dxdt = 1e-5 * randn(size(layout.points)...) 
    dfdt = ForwardDiff.derivative(dt -> fixed_point_map(ContinuousLayout(layout.points + dxdt*dt), slm, B), 0.0)
    @test ∂fϕ∂x * dxdt[:] ≈ reshape(dfdt[1], prod(size(dfdt[1]))) 
    @test ∂fB∂x * dxdt[:] ≈ reshape(dfdt[2], prod(size(dfdt[2]))) 

    ∂fϕ∂ϕ = ForwardDiff.jacobian(x -> fixed_point_map(layout, SLM(slm.A, x, slm.SLM2π), B)[1], slm.ϕ)
    ∂fB∂ϕ = ForwardDiff.jacobian(x -> fixed_point_map(layout, SLM(slm.A, x, slm.SLM2π), B)[2], slm.ϕ)
    ∂fϕ∂B = ForwardDiff.jacobian(x -> fixed_point_map(layout, slm, x)[1], B)
    ∂fB∂B = ForwardDiff.jacobian(x -> fixed_point_map(layout, slm, x)[2], B)

    function linemap(dϕBdt)
        dϕBdt - ForwardDiff.derivative(dt -> fixed_point_map(layout, SLM(slm.A, slm.ϕ + dϕBdt[1] * dt, slm.SLM2π), B + dϕBdt[2] * dt), 0.0)
    end
    dϕBdt1,  = linsolve(x -> linemap(x), dfdt)
    dϕBdt2,  = linsolve(dϕBdt ->  dϕBdt - [∂fϕ∂ϕ * dϕBdt[1] + ∂fϕ∂B * dϕBdt[2], ∂fB∂B * dϕBdt[2] + ∂fB∂ϕ * dϕBdt[1]], [∂fϕ∂x * dxdt[:], ∂fB∂x * dxdt[:]], maxiter = 1)
    dϕBdt3 = get_dϕBdt(layout, slm, B, dxdt, 10; ifdBdt=true)

    @test reshape(dϕBdt2[1], size(slm.ϕ)) ≈ dϕBdt1[1] 
    @test dϕBdt1[1] ≈ dϕBdt3[1] atol=1e-2
    @test reshape(dϕBdt2[2], size(B)) ≈ dϕBdt1[2] 
    @test dϕBdt1[2] ≈ dϕBdt3[2] atol=1e-4
end

@testset "evolution_slm continuous with $atype" for atype in test_atypes
    Random.seed!(42)
    points = atype(rand(5, 2))
    layout = ContinuousLayout(points)
    points_new = points + 1e-1 * atype(randn(size(points)))
    layout_new = ContinuousLayout(points_new)
    slm = atype(SLM(10))

    layouts, slms = evolution_slm_flow(layout, layout_new, slm, WGS(verbose=false);
                                       interps=6, 
                                       aditers=5)

    @test length(layouts) == length(slms)
end