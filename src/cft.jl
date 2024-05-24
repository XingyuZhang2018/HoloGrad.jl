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

function padding_extract(A::AbstractArray{T,2}, ϵ) where T
    ϵ == 1 && return A
    Nu, Nv = size(A)
    r = round(Int, 1 / ϵ)
    Nx, Ny = Nu ÷ r, Nv ÷ r
    return A[((Nu-Nx)÷2+1):((Nu-Nx)÷2+Nx), ((Nv-Ny)÷2+1):((Nv-Ny)÷2+Ny)]
end

function ft_enlarge(signal, ϵ)
    Nx, Ny = size(signal)
    Nu = Nx ÷ ϵ
    Nv = Ny ÷ ϵ
    ft_enlarge(signal, Nu, Nv)
end

"""
    fft_enlarge(signal, Nu, Nv)

Enlarge the signal to a frequency with a larger size.
"""
function ft_enlarge(signal, Nu, Nv)
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

function ft(layout, signal)
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

function ift_reduce(frequency, ϵ)
    Nu, Nv = size(frequency)
    Nx = Nu * ϵ
    Ny = Nv * ϵ
    ift_reduce(frequency, Nx, Ny)
end

"""
    ifft_reduce(frequency, Nx, Ny)

Reduce the frequency to a signal with a smaller size.
"""
function ift_reduce(frequency::AbstractArray{T, 2}, Nx, Ny) where T
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

function ift(layout, F, ϵ)  
    Nu, Nv = size(layout.mask)
    Nx = round(Int, Nu * ϵ)
    Ny = round(Int, Nv * ϵ)
    ift(layout, F, Nx, Ny)
end

function ift(layout, F, Nx, Ny)
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
