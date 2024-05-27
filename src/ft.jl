"""
fourier transform functions for SLM optimization.

three methods are provided: fft, mft, smft.

    * fft: fast fourier transform.

    * mft: dense matrix fourier transform.

    * smft: sparse matrix fourier transform.

"""
function ft(layout, signal, ϵ, ::Val{:fft})
    image = fft(padding(signal, ϵ))
    extract_locations(layout, image)
end

function ft(layout, signal, ϵ, ::Val{:mft})
    Nx, Ny = size(signal)
    Nu = Nx ÷ ϵ
    Nv = Ny ÷ ϵ
    image = mft(signal, Nu, Nv)
    extract_locations(layout, image)
end

function ft(layout, signal, ϵ, ::Val{:smft})
    smft(layout, signal)
end

function ift(layout, frequency, ϵ, ::Val{:fft})
    image = embed_locations(layout, frequency)
    padding_extract(ifft(image), ϵ)
end

function ift(layout, frequency, ϵ, ::Val{:mft})
    image = embed_locations(layout, frequency)
    Nu, Nv = size(image)
    Nx = Nu * ϵ
    Ny = Nv * ϵ
    imft(image, Nx, Ny)
end

function ift(layout, frequency, ϵ, ::Val{:smft})
    Nu, Nv = size(layout.mask)
    Nx = round(Int, Nu * ϵ)
    Ny = round(Int, Nv * ϵ)
    ismft(layout, frequency, Nx, Ny)
end

"""
Padding the matrix A with zeros.

Args:

    A (Array): the matrix to be padded,
    ϵ (Real): the padding factor, 0<ϵ≤1.

Returns:

    A_enlarge, the padded matrix. the original matrix is at the center of the padded matrix.
    
"""
function padding(A::AbstractArray{T,2}, ϵ) where T
    ϵ == 1 && return A
    Nx, Ny = size(A)
    r = round(Int, 1 / ϵ)
    Nu = Nx * r
    Nv = Ny * r
    A_enlarge = zeros(T, Nu, Nv)
    # A_enlarge[(r÷2+1):r:Nu, (r÷2+1):r:Nv] = A
    A_enlarge[((Nu-Nx)÷2+1):((Nu-Nx)÷2+Nx), ((Nv-Ny)÷2+1):((Nv-Ny)÷2+Ny)] = A
    return A_enlarge
end

padding(slm::SLM, ϵ) = SLM(padding(slm.A, ϵ), padding(slm.ϕ, ϵ), slm.SLM2π)

function padding_extract(A::AbstractArray{T,2}, ϵ) where T
    ϵ == 1 && return A
    Nu, Nv = size(A)
    r = round(Int, 1 / ϵ)
    Nx, Ny = Nu ÷ r, Nv ÷ r
    return A[((Nu-Nx)÷2+1):((Nu-Nx)÷2+Nx), ((Nv-Ny)÷2+1):((Nv-Ny)÷2+Ny)]
end

"""
    dense matrix fourier transform.
"""
function mft(signal, Nu, Nv)
    Nx, Ny = size(signal)
    @assert Nu >= Nx && Nv >= Ny
    u = 0:Nu-1
    v = 0:Nv-1
    x = (0:Nx-1) .+ ((Nu - Nx) ÷ 2)
    y = (0:Ny-1) .+ ((Nv - Ny) ÷ 2)
    X = exp.(-2im * π * u * x' / Nu)
    Y = exp.(-2im * π * y * v' / Nv)
    return X * signal * Y
end

"""
    sparse matrix fourier transform.
"""
function smft(layout, signal)
    Nx, Ny = size(signal)
    mask = layout.mask
    Nu, Nv = size(mask)
    @assert Nu >= Nx && Nv >= Ny
    ind = CartesianIndices(mask)[mask]
    u = [uv[1] - 1 for uv in ind]
    v = [uv[2] - 1 for uv in ind]
    x = (0:Nx-1) .+ ((Nu - Nx) ÷ 2)
    y = (0:Ny-1) .+ ((Nv - Ny) ÷ 2)
    X = exp.(-2im * π * u * x' / Nu)
    Y = exp.(-2im * π * y * v' / Nv)
    return ein"(ab,bc),ca->a"(X, signal, Y)
end

function imft(frequency::AbstractArray{T, 2}, Nx, Ny) where T
    Nu, Nv = size(frequency)
    @assert Nu >= Nx && Nv >= Ny
    u = 0:Nu-1
    v = 0:Nv-1
    x = (0:Nx-1) .+ ((Nu - Nx) ÷ 2)
    y = (0:Ny-1) .+ ((Nv - Ny) ÷ 2)
    X = exp.(2im * π * x * u' / Nu)
    Y = exp.(2im * π * v * y' / Nv) 
    return (X * frequency * Y) / (Nu * Nv)
end

function ismft(layout, F::AbstractArray{T, 1}, Nx, Ny) where T
    mask = layout.mask
    Nu, Nv = size(mask)
    @assert Nu >= Nx && Nv >= Ny
    ind = CartesianIndices(mask)[mask]
    u = [uv[1] - 1 for uv in ind]
    v = [uv[2] - 1 for uv in ind]
    x = (0:Nx-1) .+ ((Nu - Nx) ÷ 2)
    y = (0:Ny-1) .+ ((Nv - Ny) ÷ 2)
    X = exp.(2im * π * x * u' / Nu)
    Y = exp.(2im * π * v * y' / Nv) 
    return ein"(ab,b),bc->ac"(X, F, Y) / (Nu * Nv)
end
