var documenterSearchIndex = {"docs":
[{"location":"#","page":"Home","title":"Home","text":"EditURL = \"https://github.com/JuliaMolSim/DFTK.jl/blob/master/docs/src/index.jl\"","category":"page"},{"location":"#DFTK.jl:-The-density-functional-toolkit.-1","page":"Home","title":"DFTK.jl: The density-functional toolkit.","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"DFTK is a Julia package for playing with plane-wave density-functional theory algorithms. In its basic formulation it solves periodic Kohn-Sham equations.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The following documentation is an overview of the structure of the code, and of the formalism used. It assumes basic familiarity with the concepts of plane-wave density functional theory. Users wanting to simply run computations or get an overview of features should look at examples directory in the main code.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"In the following we will illustrate the concepts on the example of computing the LDA ground state of the Silicon crystal.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"using DFTK\nusing Plots\nusing LinearAlgebra\n\n# 1. Define lattice and atomic positions\na = 10.26  # Silicon lattice constant in Bohr\nlattice = a / 2 * [[0 1 1.];\n                   [1 0 1.];\n                   [1 1 0.]]\n\n# Load HGH pseudopotential for Silicon\nSi = ElementPsp(:Si, psp=load_psp(\"hgh/lda/Si-q4\"))\n\n# Specify type and positions of atoms\natoms = [Si => [ones(3)/8, -ones(3)/8]]\n\n# 2. Select model and basis\nmodel = model_LDA(lattice, atoms)\nkgrid = [4, 4, 4]  # k-Point grid (Regular Monkhorst-Pack grid)\nEcut = 15          # kinetic energy cutoff in Hartree\nbasis = PlaneWaveBasis(model, Ecut; kgrid=kgrid)\n\n# 3. Run the SCF\nscfres = self_consistent_field(basis, tol=1e-8);\nnothing #hide","category":"page"},{"location":"#Notations-and-conventions-1","page":"Home","title":"Notations and conventions","text":"","category":"section"},{"location":"#Units-1","page":"Home","title":"Units","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"DFTK uses atomic units throughout: lengths are in Bohr, energies in Hartree. In particular, hbar = m_e = e = 4pi epsilon_0 = 1. In this convention the Schrödinger equation for the electron of the hydrogen atom is","category":"page"},{"location":"#","page":"Home","title":"Home","text":"ipartial_t psi = -frac 1 2 Delta psi - frac 1 r psi","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Useful conversion factors can be found in DFTK.units:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"DFTK.units.eV","category":"page"},{"location":"#Coordinates-1","page":"Home","title":"Coordinates","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Computations take place in the unit cell of a lattice defined by a 3  3 matrix (model.lattice) with lattice vectors as columns. Note that Julia stores matrices as column-major, so care has to be taken when interfacing with other libraries in row-major languages (e.g. Python). The reciprocal lattice model.recip_lattice (the lattice of Fourier coefficients of functions with the periodicity of the lattice) is defined by the matrix","category":"page"},{"location":"#","page":"Home","title":"Home","text":"B = 2pi A^-T = 2π A^T^-1","category":"page"},{"location":"#","page":"Home","title":"Home","text":"where A is the unit cell.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"model.recip_lattice' * model.lattice","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Vectors in real space are denoted by r if they reside inside the unit cell and by R for lattice vectors. Vectors in reciprocal space are analogously k (for vectors in the Brillouin zone) and G for vectors on the reciprocal lattice. Commonly q is used to refer to k + G. If not denoted otherwise the code uses reduced coordinates for these vectors. One switches to Cartesian coordinates by","category":"page"},{"location":"#","page":"Home","title":"Home","text":"x_textcart = M x_textred","category":"page"},{"location":"#","page":"Home","title":"Home","text":"where M is either model.lattice (for real-space vectors) or model.recip_lattice (for reciprocal-space vectors). A useful relationship is","category":"page"},{"location":"#","page":"Home","title":"Home","text":"b_textcart cdot a_textcart=2pi b_textred cdot a_textred","category":"page"},{"location":"#","page":"Home","title":"Home","text":"if a and b are real-space and reciprocal-space vectors respectively. Other names for reduced coordinates are integer coordinates (usually for G-vectors) or fractional coordinates (usually for k-points).","category":"page"},{"location":"#Naming-conventions-1","page":"Home","title":"Naming conventions","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"DFTK liberally uses Unicode characters to represent Greek characters (e.g. ψ, ρ, ε...). Input them at the Julia REPL by their latex command and press \"TAB\". For all major editors there are great Julia plugins offering easy support for such characters as well.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Reciprocal-space vectors: k for vectors in the Brillouin zone, G for vectors of the reciprocal lattice, q for general vectors\nReal-space vectors: R for lattice vectors, r and x are usually used for unit for vectors in the unit cell or general real-space vectors, respectively. This convention is, however, less consistently applied.\nOmega is the unit cell, and Omega (or sometimes just Omega) is its volume.\nA are the real-space lattice vectors (model.lattice) and B the Brillouin zone lattice vectors (model.recip_lattice).\nThe Bloch waves are\npsi_nk(x) = e^ikcdot x u_nk(x)\nwhere n is the band index and k the k-point. In the code we sometimes use psi and u interchangeably.\nvarepsilon are the eigenvaluesi, varepsilon_F is the Fermi level.\nrho is the density.\nIn the code we use normalized plane waves:\ne_G(r) = frac 1 sqrtOmega e^i G cdot r\nY^l_m are the complex spherical harmonics, and Y_lm the real ones.\nj_l are the Bessel functions. In particular, j_0(x) = fracsin xx.","category":"page"},{"location":"#Basic-datastructures-1","page":"Home","title":"Basic datastructures","text":"","category":"section"},{"location":"#Model-1","page":"Home","title":"Model","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"The physical model to be solved is defined by the Model datastructure. It contains the unit cell, number of electrons, atoms, type of spin polarization and temperature. Each atom has an atomic type (Element) specifying their number of valence electrons and the potential (or pseudopotential) it creates. The Model structure also contains the list of energy terms defining the model. These can be of the following types (for now), defined in the src/terms directory:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Kinetic energy\nLocal potential energy, either given by analytic potentials or specified by the type of atoms.\nNonlocal potential energy, for norm-conserving pseudopotentials\nNuclei energies (eg Ewald or pseudopotential correction)\nHartree energy\nExchange-correlation energy\nPower nonlinearities (useful for Gross-Pitaevskii type models)\nMagnetic field energy\nEntropy term","category":"page"},{"location":"#","page":"Home","title":"Home","text":"By mixing and matching these terms, the user can create custom models. Convenience constructors are provided for commonly used models:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"model_DFT: density-functional theory Hamiltonian using  any of the LDA or GGA functionals of the  libxc  library.\nmodel_LDA: LDA Hamiltonian using the Teter parametrisation","category":"page"},{"location":"#","page":"Home","title":"Home","text":"For the silicon example above the following terms were used[1]:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"typeof.(model.term_types)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"[1]: If you are not familiar with Julia syntax, this is equivalent to [typeof(t) for t in model.term_types].","category":"page"},{"location":"#","page":"Home","title":"Home","text":"DFTK computes energies for all terms of the model Hamiltonian. In the silicon example from above:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"display(scfres.energies)","category":"page"},{"location":"#PlaneWaveBasis-1","page":"Home","title":"PlaneWaveBasis","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"The PlaneWaveBasis datastructure handles the discretization of a given Model in a plane-wave basis. As usual in plane-wave methods the discretization is twofold: Once the k-point grid, which determines how the Brillouin zone is sampled in the discrete basis and once the sampling of the reciprocal-space lattice, which is restricted to a finite set of plane waves. The former aspect is controlled by the kgrid agrument (or by an explicit list of k-points) and the latter is controlled by the cutoff energy parameter Ecut.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The periodic parts of Bloch waves are expanded in a set of normalized plane waves e_G:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"beginaligned\n  psi_k(x) = e^i k cdot x u_k(x)\n  = sum_G in mathcal R^* c_G  e^i  k cdot  x e_G(x)\nendaligned","category":"page"},{"location":"#","page":"Home","title":"Home","text":"where mathcal R^* is the set of reciprocal lattice vectors. The c_G are ell^2-normalized. The summation is truncated to a \"spherical\", k-dependent basis set","category":"page"},{"location":"#","page":"Home","title":"Home","text":"  S_k = leftG in mathcal R^* middle frac 1 2 k+ G^2 le E_textcutright","category":"page"},{"location":"#","page":"Home","title":"Home","text":"where E_textcut is the cutoff energy.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Densities involve terms like psi_k^2 = u_k^2 and therefore products e_-G e_G for G G in S_k. To represent these we use a \"cubic\", k-independent basis set large enough to contain the set G-G  G G in S_k. We can obtain the coefficients of densities on the e_G basis by a convolution, which can be performed efficiently with FFTs (see G_to_r and r_to_G functions). Potentials are discretized on this same set.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The normalization conventions used in the code is that quantities stored in reciprocal space are coefficients in the e_G basis, and quantities stored in real space use real physical values. This means for instance that wavefunctions in the real space grid are normalized as fracOmegaN sum_r psi(r)^2 = 1 where N is the number of grid points.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"For example let us check the normalization of the first eigenfunction at the first k-Point in reciprocal space:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"ψtest = scfres.ψ[1][:, 1]\nsum(abs2.(ψtest))","category":"page"},{"location":"#","page":"Home","title":"Home","text":"We now perform an IFFT to get ψ in real space. The k-Point has to be passed because ψ is expressed on the k-dependent basis. Again the function is normalised:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"ψreal = G_to_r(basis, basis.kpoints[1], ψtest)\nsum(abs2.(ψreal)) * model.unit_cell_volume / prod(basis.fft_size)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The list of k points can be obtained with basis.kpoints.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"basis.kpoints","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The G vectors of the \"spherical\", k-dependent grid can be obtained with G_vectors(basis.kpoints[ik]) with an index ik:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"[length(G_vectors(k)) for k in basis.kpoints]","category":"page"},{"location":"#","page":"Home","title":"Home","text":"ik = 1\nG_vectors(basis.kpoints[ik])[1:4]","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The list of G vectors (Fourier modes) of the \"cubic\", k-independent basis set can be obtained with G_vectors(basis).","category":"page"},{"location":"#","page":"Home","title":"Home","text":"length(G_vectors(basis)), prod(basis.fft_size)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"collect(G_vectors(basis))[1:4]","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Analogously the list of r vectors (real-space grid) can be obtained with r_vectors(basis):","category":"page"},{"location":"#","page":"Home","title":"Home","text":"length(r_vectors(basis))","category":"page"},{"location":"#","page":"Home","title":"Home","text":"collect(r_vectors(basis))[1:4]","category":"page"},{"location":"#","page":"Home","title":"Home","text":"As seen above, wavefunctions are stored in an array ψ as ψ[ik][iG, iband] where ik is the index of the kpoint (in basis.kpoints), iG is the index of the plane wave (in G_vectors(basis.kpoints[ik])) and iband is the index of the band. Densities are usually stored in a special type, RealFourierArray, from which the representation in real and reciprocal space can be accessed using ρ.real and ρ.fourier respectively.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"rvecs = collect(r_vectors(basis))[:, 1, 1]  # slice along the x axis\nx = [r[1] for r in rvecs]                   # only keep the x coordinate\nplot(x, scfres.ρ.real[:, 1, 1], label=\"\", xlabel=\"x\", ylabel=\"ρ\", marker=2)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"G_energies = [sum(abs2.(model.recip_lattice * G)) ./ 2 for G in G_vectors(basis)][:]\nscatter(G_energies, abs.(scfres.ρ.fourier[:]);\n        yscale=:log10, ylims=(1e-12, 1), label=\"\", xlabel=\"Energy\", ylabel=\"|ρ|^2\")","category":"page"},{"location":"#","page":"Home","title":"Home","text":"(the density has no components on wavevectors above a certain energy, because the wavefunctions are limited to frac 1 2k+G^2  E_rm cut)","category":"page"},{"location":"#Useful-formulas-1","page":"Home","title":"Useful formulas","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"The Fourier transform is","category":"page"},{"location":"#","page":"Home","title":"Home","text":"widehatf( q) = int_mathbb R^3 e^-i q cdot  x dx","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Plane wave expansion formula","category":"page"},{"location":"#","page":"Home","title":"Home","text":"e^i q cdot r =\n     4 pi sum_l = 0^infty sum_m = -l^l\n     i^l j_l(q r) Y_l^m(qq) Y_l^mast(rr)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Spherical harmonics orthogonality","category":"page"},{"location":"#","page":"Home","title":"Home","text":"   int_mathbbS^2 Y_l^m*(r)Y_l^m(r) dr\n     = delta_ll delta_mm","category":"page"},{"location":"#","page":"Home","title":"Home","text":"This also holds true for real spherical harmonics.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Fourier transforms of centered functions: If","category":"page"},{"location":"#","page":"Home","title":"Home","text":"f(x) = R(x) Y_l^m(xx), then","category":"page"},{"location":"#","page":"Home","title":"Home","text":"beginaligned\n  hat f( q)\n  = int_mathbb R^3 R(x) Y_l^m(xx) e^-i q cdot x dx \n  = sum_l = 0^infty 4 pi i^l\n  sum_m = -l^l int_mathbb R^3\n  R(x) j_l(q x)Y_l^m(-qq) Y_l^m(xx)\n   Y_l^mast(xx)\n  dx \n  = 4 pi Y_l^m(-qq) i^l\n  int_mathbb R^+ r^2 R(r)  j_l(q r) dr\n endaligned","category":"page"},{"location":"#","page":"Home","title":"Home","text":"This also holds true for real spherical harmonics.","category":"page"},{"location":"#Crystal-symmetries-1","page":"Home","title":"Crystal symmetries","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"In this discussion we will only describe the situation for a monoatomic crystal mathcal C subset mathbb R^3, the extension being easy. A symmetry of the crystal is a real-space unitary matrix tildeS and a real-space vector tildeτ such that","category":"page"},{"location":"#","page":"Home","title":"Home","text":"tildeS mathcalC + tildetau = mathcalC","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The symmetries where tilde S = 1 and tildeτ is a lattice vector are always assumed and ignored in the following.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"We can define a corresponding unitary operator U  L^2_textper to L^2_textper with action","category":"page"},{"location":"#","page":"Home","title":"Home","text":" (Uu)(x) = uleft( S^-1 (x-tau) right)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"where we set","category":"page"},{"location":"#","page":"Home","title":"Home","text":"beginaligned\nS = tildeS^-1\ntau = -tildeS^-1tildetau\nendaligned","category":"page"},{"location":"#","page":"Home","title":"Home","text":"This unitary operator acts on the Fourier coefficients of lattice-periodic functions as","category":"page"},{"location":"#","page":"Home","title":"Home","text":"(Uu)(G) = e^-i G cdot tau u(S^-1 G)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"and so","category":"page"},{"location":"#","page":"Home","title":"Home","text":"U (-i + k) U^* = (-i + Sk)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Furthermore, since the potential V is the sum over radial potentials centered at atoms, it is easily seen that U V U^* = V, i.e. that U and V commute.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"It follows that if the Bloch wave ψ_k = e^ikcdot x u_k is an eigenfunction of the Hamiltonian, then e^i (Sk) cdot x (Uu_k) is also an eigenfunction, and so we can take","category":"page"},{"location":"#","page":"Home","title":"Home","text":"u_Sk = U u_k","category":"page"},{"location":"#","page":"Home","title":"Home","text":"This is used to reduce the computations needed. For a uniform sampling of the Brillouin zone (the reducible k-Points), one can find a reduced set of k-Points (the irreducible k-Points) such that the eigenvectors at the reducible k-Points can be deduced from those at the irreducible k-Points.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"basis_irred = basis\nscfres_irred = scfres\n# Redo the same computation but disabling symmetry handling\nbasis_red = PlaneWaveBasis(model, Ecut; kgrid=kgrid, enable_bzmesh_symmetry=false)\nscfres_red = self_consistent_field(basis_red, tol=1e-8)\n(norm(scfres_irred.ρ.real - scfres_red.ρ.real),\n norm(values(scfres_irred.energies) .- values(scfres_red.energies)))","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The results are identical up to the convergence threshold, but compared to the first calculation on the top of the page, disabling symmetry leads to a substantially larger computational time, since more k-Points are explicitly treated:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"(length(basis_red.kpoints), length(basis_irred.kpoints))","category":"page"},{"location":"#","page":"Home","title":"Home","text":"!!!note \"The tol argument in self_consistent_field\"     The tol argument to self_consistent_field is a convergence threshold     in the total energy, such that less agreement is found in the density.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"To demonstrate the mapping, let us consider an example:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"ikpt_irred = 2 # pick an arbitrary kpoint in the irreducible BZ\nkpt_irred_coord = basis_irred.kpoints[ikpt_irred].coordinate\nbasis_irred.ksymops[ikpt_irred]","category":"page"},{"location":"#","page":"Home","title":"Home","text":"This is a list of all symmetries operations (Stau) that can be used to map this irreducible kpoint to reducible kpoints. Let's pick the third symmetry operation of this k-Point and check.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"S, τ = basis_irred.ksymops[ikpt_irred][3]\nkpt_red_coord = S * basis_irred.kpoints[ikpt_irred].coordinate\nikpt_red = findfirst(kcoord -> kcoord ≈ kpt_red_coord,\n                     [k.coordinate for k in basis_red.kpoints])\n(scfres_irred.eigenvalues[ikpt_irred], scfres_red.eigenvalues[ikpt_red])","category":"page"}]
}
