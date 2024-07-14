function preloc_cft(layout, signal)
    Nx, Ny = size(signal)
    points = layout.points
    u = @view points[:, 1]
    v = @view points[:, 2]
    x = isa(points, CuArray) ? CuArray(0:Nx-1) : 0:Nx-1
    y = isa(points, CuArray) ? CuArray(0:Ny-1) : 0:Ny-1
    X = exp.(-2im * π * u * x')
    Y = exp.(-2im * π * y * v')
    iX = exp.(2im * π * x * u')
    iY = exp.(2im * π * v * y') 
    return X, Y, iX, iY
end

function preloc_cft(signal, Nu, Nv)
    Nx, Ny = size(signal)
    u = isa(signal, CuArray) ? CuArray(0:Nu-1) : 0:Nu-1
    v = isa(signal, CuArray) ? CuArray(0:Nv-1) : 0:Nv-1
    x = isa(signal, CuArray) ? CuArray(0:Nx-1) : 0:Nx-1
    y = isa(signal, CuArray) ? CuArray(0:Ny-1) : 0:Ny-1
    X = exp.(-2im * π * u * x' / Nu)
    Y = exp.(-2im * π * y * v' / Nv)
    return X, Y
end

"""
    continuous fourier transform.
"""
cft(signal, X, Y) = ein"(ab,bc),ca->a"(X, signal, Y)
icft(F, iX, iY) = ein"(ab,b),bc->ac"(iX, F, iY)

cft_m(signal, X, Y) = X * signal * Y
