using LinearMaps
using IterativeSolvers

# Mixing rules: (ρin, ρout) => ρnext, where ρout is produced by diagonalizing the
# Hamiltonian at ρin These define the basic fix-point iteration, that are then combined with
# acceleration methods (eg anderson).
# All these methods attempt to approximate the inverse Jacobian of the SCF step,
# ``J^-1 = (1 - χ0 (vc + fxc))^-1``, where vc is the Coulomb and fxc the
# exchange-correlation kernel. Note that "mixing" is sometimes used to refer to the combined
# process of formulating the fixed-point and solving it; we call "mixing" only the first part
#
# The interface is `mix(m; basis, ρin, ρout, kwargs...) -> ρnext`

@doc raw"""
Kerker mixing: ``J^{-1} ≈ \frac{α G^2}{k_F^2 + G^2}``
where ``k_F`` is the Thomas-Fermi wave vector.

Notes:
  - Abinit calls ``1/k_F`` the dielectric screening length (parameter *dielng*)
"""
struct KerkerMixing
    α::Real
    kF::Real
end
# Default parameters suggested by Kresse, Furthmüller 1996 (α=0.8, kF=1.5 Ǎ^{-1})
# DOI 10.1103/PhysRevB.54.11169
KerkerMixing(;α=0.8, kF=0.8) = KerkerMixing(α, kF)
function mix(mixing::KerkerMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray; kwargs...)
    T = eltype(basis)
    Gsq = [sum(abs2, basis.model.recip_lattice * G) for G in G_vectors(basis)]
    ρin = ρin.fourier
    ρout = ρout.fourier
    ρnext = @. ρin + T(mixing.α) * (ρout - ρin) * Gsq / (T(mixing.kF)^2 + Gsq)
    # take the correct DC component from ρout; otherwise the DC component never gets updated
    ρnext[1] = ρout[1]
    from_fourier(basis, ρnext; assume_real=true)
end

@doc raw"""
Simple mixing: ``J^{-1} ≈ α``
"""
struct SimpleMixing
    α::Real
end
SimpleMixing(;α=1) = SimpleMixing(α)
function mix(mixing::SimpleMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray; kwargs...)
    T = eltype(basis)
    if mixing.α == 1
        return ρout
    else
        ρin + T(mixing.α) * (ρout - ρin)
    end
end

@doc raw"""
We use a simplification of the Resta model DOI 10.1103/physrevb.16.2717 and set
``χ_0(q) = \frac{C_0 G^2}{4π (1 - C_0 G^2 / k_F^2)}
where ``C_0 = 1 - ε_r`` with ``ε_r`` being the macroscopic relative permittivity.
We neglect ``f_\text{xc}``, such that
``J^{-1} ≈ α \frac{k_F^2 - C_0 G^2}{ε_r k_F^2 - C_0 G^2}``

By default it assumes a relative permittivity of 10 (similar to Silicon).
`εr == 1` is equal to `SimpleMixing` and `εr == Inf` to `KerkerMixing`.
"""
struct RestaMixing
    α::Real
    εr::Real
    kF::Real
end
RestaMixing(;α=0.8, kF=0.8, εr=10) = RestaMixing(α, εr, kF)
function mix(mixing::RestaMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray; kwargs...)
    T = eltype(basis)
    εr = T(mixing.εr)
    kF = T(mixing.kF)
    εr == 1               && return mix(SimpleMixing(α=α), basis, ρin, ρout)
    εr > 1 / sqrt(eps(T)) && return mix(KerkerMixing(α=α, kF=kF), basis, ρin, ρout)

    ρin = ρin.fourier
    ρout = ρout.fourier
    C0 = 1 - εr
    Gsq = [sum(abs2, basis.model.recip_lattice * G) for G in G_vectors(basis)]

    ρnext = @. ρin + T(mixing.α) * (ρout - ρin) * (kF^2 - C0 * Gsq) / (εr * kF^2 - C0 * Gsq)
    # take the correct DC component from ρout; otherwise the DC component never gets updated
    ρnext[1] = ρout[1]
    from_fourier(basis, ρnext; assume_real=true)
end


@doc raw"""
We model ``χ_0(r, r') = -LDOS(εF, r) δ(r, r') + LDOS(εF, r) LDOS(εF, r') / DOS(εF)``
"""
struct HybridMixing
    α               # Damping parameter
    ldos_nos        # Minimal NOS value in for LDOS computation
    ldos_maxfactor  # Maximal factor between electron temperature and LDOS temperature
    G_blur          # Width of Gaussian filter applied to LDOS in reciprocal space.
end

function HybridMixing(;α=1, ldos_nos=20, ldos_maxfactor=10, G_blur=Inf)
    HybridMixing(α, ldos_nos, ldos_maxfactor, G_blur)
end

function mix(mixing::HybridMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray;
             ldos=nothing, kwargs...)
    ldos === nothing && return mix(SimpleMixing(α=α), basis, ρin, ρout)

    # blur the LDOS
    if mixing.G_blur < Inf
        blur_factor(G) = exp(-(norm(G)/mixing.G_blur)^2)
        ldos_fourier = r_to_G(basis, complex.(ldos))
        ldos_fourier .*= blur_factor.(basis.model.recip_lattice * G for G in G_vectors(basis))
        ldos = real.(G_to_r(basis, ldos_fourier))
    end

    # F : ρin -> ρout has derivative χ0 vc
    # a Newton step would be ρn+1 = ρn + (1 -χ0 vc)^-1 (F(ρn) - ρn)
    # We approximate -χ0 by a real-space multiplication by LDOS
    # We want to solve J Δρ = ΔF with J = (1 - χ0 vc)
    ΔF = ρout.real - ρin.real
    devec(x) = reshape(x, size(ρin))
    function Jop(x)
        den = devec(x)
        Gsq = [sum(abs2, basis.model.recip_lattice * G) for G in G_vectors(basis)]
        Gsq[1] = Inf  # Don't act on DC
        den_fourier = from_real(basis, den).fourier  # TODO r_to_G ??
        pot_fourier = 4π ./ Gsq .* den_fourier
        pot_real = from_fourier(basis, pot_fourier).real  # TODO G_to_r ??

        # apply χ0
        den_real = real(ldos .* pot_real - sum(ldos .* pot_real) .* ldos ./ sum(ldos))
        vec(den + den_real)
    end
    J = LinearMap(Jop, length(ρin))
    x = gmres(J, ΔF)
    Δρ = devec(x)
    from_real(basis, real(ρin.real + mixing.α * Δρ))
end

@doc raw"""
Mixture of HybridMixing, KerkerMixing and RestaMixing
```math
\chi_0(r, r') = w_{LDOS}   (-LDOS(εF, r) + LDOS(εF, r) LDOS(εF, r') / DOS(εF))
              + w_{Kerker} IFFT (-\frac{k_F^2}{4π}) FFT
              + w_{Resta}  \sqrt{L(x)} IFFT \frac{C_0 G^2}{4π (1 - C_0 G^2 / k_F^2)} FFT \sqrt{L(x)}
```
where ``C_0 = 1 - ε_r`` and the same convention for parameters is used as before.
Additionally there are the weights `w_{LDOS}` `w_{Kerker}` `w_{Resta}`
and the localizer function `L(r)`. For now `localizer` can take the values `:ρ`
and `:one` corresponding to `L(r) = 1` and `L(r) = ρ(r) / sum(ρ)`.
"""
struct CombinedMixing
    α::Real
    εr::Real
    kF::Real
    localizer::Symbol
    w_ldos::Real
    w_kerker::Real
    w_resta::Real

    # These are needed for compatibility now
    ldos_maxfactor::Real
    ldos_nos::Real
end

function CombinedMixing(;α=1, εr=10, kF=1, localizer=:one, w_ldos=0, w_resta=1, w_kerker=0)
    ldos_maxfactor = 1
    ldos_nos = 1
    CombinedMixing(α, εr, kF, localizer, w_ldos, w_kerker, w_resta, ldos_maxfactor, ldos_nos)
end

function mix(mixing::CombinedMixing, basis, ρin::RealFourierArray, ρout::RealFourierArray;
             ldos=nothing, kwargs...)
    T = eltype(basis)
    εr = T(mixing.εr)
    kF = T(mixing.kF)
    C0 = 1 - εr
    Gsq = [sum(abs2, basis.model.recip_lattice * G) for G in G_vectors(basis)]

    @assert mixing.localizer in (:one, :ρ)
    if mixing.localizer == :ρ
        L = sqrt.(ρout.real ./ sum(ρout.real))
        apply_localizer(x) = from_real(basis, L .* x.real)
    else
        apply_localizer = identity
    end

    # Solve J Δρ = ΔF with J = (1 - χ0 vc) and χ_0 given as in the docstring of the class
    ΔF = ρout.real - ρin.real
    devec(x) = reshape(x, size(ρin))
    function Jop(x)
        δρ = devec(x)
        Jδρ = complex(δρ)

        # Apply Hartree kernel
        δV_fourier = 4T(π) ./ Gsq .* r_to_G(basis, Jδρ)
        δV_fourier[1] = 0  # Zero DC component
        δV = from_fourier(basis, δV_fourier)

        # Apply χ0 term by term to δV and accumulate in δρ
        if mixing.w_kerker > 0
            Jδρ .-= mixing.w_kerker .* G_to_r(basis, (-kF^2 / 4T(π)) .* δV.fourier)
        end

        if mixing.w_resta > 0
            loc_δV = apply_localizer(δV).fourier
            resta_loc_δV =  @. (mixing.w_resta * C0 * kF^2 * Gsq / 4T(π)
                                               / (kF^2 - C0 * Gsq) * loc_δV)
            Jδρ .-= apply_localizer(from_fourier(basis, resta_loc_δV)).real
        end

        if mixing.w_ldos > 0 && ldos !== nothing
            Jδρ .-= @. mixing.w_ldos * (-ldos * δV.real
                                        + sum(ldos .* δV.real) * ldos / sum(ldos))
        end

        # Poor man's zero DC component before return
        vec(real(Jδρ .-= sum(Jδρ) / length(Jδρ)))
    end
    J = LinearMap(Jop, length(ρin))
    x = gmres(J, ΔF)
    Δρ = devec(x)
    Δρ .-= sum(Δρ) / length(Δρ)  # Poor man's zero DC component

    from_real(basis, real(@. ρin.real + T(mixing.α) * Δρ))
end


struct χ0Mixing
    α               # Damping parameter
    ldos_nos        # Minimal NOS value in for LDOS computation
    ldos_maxfactor  # Maximal factor between electron temperature and LDOS temperature
    droptol         # Tolerance for dropping contributions in χ0
    sternheimer_contribution  # Use Sternheimer for contributions of unoccupied orbitals
end
function χ0Mixing(α=1; ldos_nos=20, ldos_maxfactor=10, droptol=Inf, sternheimer_contribution=false)
    χ0Mixing(α, ldos_nos, ldos_maxfactor, droptol, sternheimer_contribution)
end

function mix(mixing::χ0Mixing, basis, ρin::RealFourierArray, ρout::RealFourierArray;
             ldos, ham, ψ, occupation, εF, eigenvalues, ldos_temperature)
    # TODO Duplicate code with HybridMixing
    #
    # F : ρin -> ρout has derivative χ0 vc
    # a Newton step would be ρn+1 = ρn + (1 -χ0 vc)^-1 (F(ρn) - ρn)
    # We approximate -χ0 by a real-space multiplication by LDOS
    # We want to solve J Δρ = ΔF with J = (1 - χ0 vc)
    ΔF = ρout.real - ρin.real
    devec(x) = reshape(x, size(ρin))
    function Jop(x)
        den = devec(x)
        Gsq = [sum(abs2, basis.model.recip_lattice * G) for G in G_vectors(basis)]
        Gsq[1] = Inf  # Don't act on DC
        den_fourier = from_real(basis, den).fourier  # TODO r_to_G ??
        pot_fourier = 4π ./ Gsq .* den_fourier
        pot_real = from_fourier(basis, pot_fourier).real  # TODO G_to_r ??

        # apply χ0
        den_real = apply_χ0(ham, real(pot_real), ψ, occupation, εF, eigenvalues;
                            droptol=mixing.droptol,
                            sternheimer_contribution=mixing.sternheimer_contribution,
                            temperature=ldos_temperature)
        vec(den - den_real)
    end
    J = LinearMap(Jop, length(ρin))
    x = gmres(J, ΔF)
    Δρ = devec(x)
    from_real(basis, real(ρin.real + mixing.α * Δρ))
end
