using Test
using DFTK: PlaneWaveBasis, load_psp, build_local_potential, Hamiltonian
using DFTK: lobpcg, eval_psp_local_fourier, PreconditionerKinetic
using DFTK: lobpcg, eval_psp_local_fourier, build_nonlocal_projectors

include("silicon_testcases.jl")

@testset "Diagonalisation of a free-electron Hamiltonian" begin
    # Construct a free-electron Hamiltonian
    Ecut = 5
    grid_size = [15, 15, 15]
    pw = PlaneWaveBasis(lattice, grid_size, Ecut, kpoints, kweights, ksymops)
    ham = Hamiltonian(pw)

    tol = 1e-8
    nev_per_k = 10
    ref_λ = [
        [0.00000000000, 0.56219939834, 0.56219939834, 0.56219939834, 0.56219939834,
         0.56219939834, 0.56219939834, 0.56219939834, 0.56219939834, 0.74959919778],
        [0.06246659981, 0.24986639926, 0.49973279852, 0.49973279852, 0.49973279852,
         0.56219939834, 0.56219939834, 0.56219939834, 0.74959919778, 0.74959919778],
        [0.08328879975, 0.33315519901, 0.39562179883, 0.39562179883, 0.39562179883,
         0.39562179883, 0.83288799753, 0.83288799754, 0.83288799754, 0.83288799754],
        [0.16657759951, 0.22904419932, 0.22904419932, 0.41644399877, 0.41644399877,
         0.66631039803, 0.72877699784, 0.72877699784, 0.72877699784, 0.72877699784],
    ]

    @test length(ref_λ) == length(kpoints)
    @testset "without Preconditioner" begin
        res = lobpcg(ham, nev_per_k, tol=tol)

        @test res.converged
        for ik in 1:length(kpoints)
            @test ref_λ[ik] ≈ res.λ[ik]
            @test maximum(res.residual_norms[ik]) < 100tol  # TODO Why the 100?
            @test res.iterations[ik] < 200
        end
    end

    @testset "with Preconditioner" begin
        res = lobpcg(ham, nev_per_k, tol=tol, backend=:lobpcg_qr,
                     prec=PreconditionerKinetic(ham, α=0.1))

        @test res.converged
        for ik in 1:length(kpoints)
            @test ref_λ[ik] ≈ res.λ[ik]
            @test maximum(res.residual_norms[ik]) < 100tol  # TODO Why the 100?
            @test res.iterations[ik] < 50
        end
    end
end

@testset "Diagonalisation of a core Hamiltonian" begin
    Ecut = 10
    grid_size = [21, 21, 21]
    pw = PlaneWaveBasis(lattice, grid_size, Ecut, kpoints, kweights, ksymops)
    Si = Species(atnum, psp=load_psp("si-pade-q4.hgh"))
    ham = Hamiltonian(pw, pot_local=build_local_potential(pw, Si => positions),
                      pot_nonlocal=build_nonlocal_projectors(pw, Si => positions))
    res = lobpcg(ham, 5, tol=1e-8, prec=PreconditionerKinetic(ham, α=0.1),
                 backend=:lobpcg_qr)

    ref = [
        [0.067955741977536, 0.470244204908046, 0.470244204920801,
         0.470244204998022, 0.578392222232969],
        [0.111089041747288, 0.304724122513625, 0.445322298067717,
         0.445322298101198, 0.584713217756577],
        [0.129419322499919, 0.293174377882115, 0.411932220567084,
         0.411932220611853, 0.594921264868345],
        [0.168662148987539, 0.238552367551507, 0.370743978236562,
         0.418387442903058, 0.619797227001203],
    ]
    for ik in 1:length(kpoints)
        @test res.λ[ik] ≈ ref[ik] atol=0.02
    end
end