function preloc_cft(layout, signal)
    Nx, Ny = size(signal)
    points = layout.points
    u = [p[1] for p in points]
    v = [p[2] for p in points]
    x = 0:Nx-1
    y = 0:Ny-1
    X = exp.(-2im * π * u * x')
    Y = exp.(-2im * π * y * v')
    iX = exp.(2im * π * x * u')
    iY = exp.(2im * π * v * y') 
    return X, Y, iX, iY
end

"""
    continuous fourier transform.
"""
cft(signal, X, Y) = ein"(ab,bc),ca->a"(X, signal, Y)
icft(F, iX, iY) = ein"(ab,b),bc->ac"(iX, F, iY)

