using FFTW
using Test
using BenchmarkTools

@testset "fft for threads number $n" for n in 1:10
    FFTW.set_num_threads(n)
    A = randn(ComplexF64, 1024, 1024)
    @btime fftshift(fft($A));
end

@testset "ifft for threads number $n" for n in 1:10
    FFTW.set_num_threads(n)
    A = randn(ComplexF64, 1024, 1024)
    @btime ifft($A);
end