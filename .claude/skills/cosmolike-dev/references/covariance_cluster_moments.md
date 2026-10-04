# Selected halo moments for cluster covariance

## Contract and physics

`moments_cluster_cov.c/h` integrates caller-supplied selected mass weights
and mass-weighted Fourier profiles. The cluster-only C++ binding exposes
`covariance_cluster_moments`. It chooses no mass function, selection,
neutrino prescription, low-mass completion or background normalization.
The weight contains dn*S once; the profile is (M/rho)*u, in L^3.

Outputs retain state and observed-category axes: abundance, biased
abundance, J01, J11, J02(K,Q), J03(K,K,Q), J03(K,Q,Q). Unnormalized
dimensions are L^-3, L^-3, 1, 1, L^3, L^6, L^6. Normalizing J01 by
the selected abundance produces the one-halo cluster-matter power.
J11 supplies the abundance contribution to its response, not a complete
profile/growth/dilation or observed-mean response.

References checked directly: To et al. (2021), arXiv:2008.10757,
Eqs. 20--21; Schaan, Takada & Spergel (2014), arXiv:1406.3330, Eq. 35.
The latter's non-SSC count-matter kernel is
J02(K,K)+2 P_lin(K) I11(K) J11(K), with all-halo I11. The shared Python
projection below implements this matter cross block. Catalog normalization
for discrete cluster partners and the full joint generator remain open.

For exclusive observed categories, membership indicators obey I_i^2=I_i,
and I_i I_j=0 for i!=j. Therefore same-halo terms use one selection
probability, not its powers or the product of two exclusive probabilities.
This does not eliminate different-halo cross-category covariance or SSC.
Never reuse the old equal-redshift-only exclusion for those terms.

## Numerics and checks (2026-10-04)

The C mass sums use unconditional SIMDe and keep mass-node order within
each result. Single/pair loops collapse state, selection and k/pair groups
for OpenMP; the abundance loop collapses state and selection. Only a
triangular integer pair map is allocated. Profiles are shared across
selections and products. No MPI, global cache or data-vector C change.
Thread determinism is measured; performance/scaling claims await a quiet
machine. These correctness runs overlapped the project regression runner.

The optimized DES covariance suite passes 19 checks, including six new
selected-moment cases. Closed polynomial integrals establish bias/profile
powers and one-factor selection weighting. A partition of selection
probabilities recovers the unselected moment. Independent NumPy sums
check signed profiles, small/odd/even grids, distinct unit powers, owned
outputs and malformed-input rejection. Results are bitwise identical at
1/2/4/8 threads. Comparisons use rtol 2e-14 and atol 1e-14 for signed sums.

An isolated -O0 production-C build with undefined-behavior and
float-divide-by-zero sanitizers passes 16 cases: (nk,nmass) equal to
(1,1), (2,6), (3,9), (5,257), each at 1/2/4/8 threads. It agrees with
the optimized binding at the same tolerance and is bitwise across thread
counts. The installed interface stays optimized.

The DES supplied-weight pilot uses fixed-alpha Tinker (0.368), fixed
lognormal richness selection, four richness categories, redshifts
0.25/0.45/0.60, mass range 1e12--1e16 Msun/h and k=0.001--10 h/Mpc.
It shares public sigma, slope, bias and profile readers, so it is an
integration/normalization comparison, not an independent halo-fit oracle.
At 512 mass nodes, maximum relative differences to the existing cluster
tables are 2.07e-5 for abundance, 4.01e-6 for bias, 1.48e-5 for P_cm^1h.
The 256-to-512 changes in the new density/single/pair moments are
1.47e-6/2.40e-6/4.97e-6. Refinement is not monotone through the shared
interpolated readers; no survey accuracy setting or FoM claim follows.

External evidence: `test/covariance_reference/cluster_moments_pilot.py`,
`check_cluster_moments_debug.py`, their results and session logs. Public
tests are `des_cluster/tests/covariance/test_moments_cluster.py`.

## Non-SSC count-matter cross projection

`counts_cluster.count_matter_cross` projects the above kernel using
dchi W_A W_B/f_K^2. The selected moment weight already contains the
observed count selection, which must not be applied again. A supplied
transfer fixes source-leg conventions. Output keeps one- and two-halo
contributions separate; its total means ONLY this non-SSC cross block.
The explicit footprint area cancels here, unlike the SSC dependence.
Only projected matter and a constant linear-bias galaxy approximation
are covered. Discrete cluster partners and shared-object noise are not.

The mass kernel is assembled with vectorized NumPy. All radial sums reuse
the existing SIMDe/OpenMP C weighted projection in one batch per component;
there is no new C loop, scalar fallback or duplicate projection engine.
Four independent tests pass: closed shell integrals and spin transfers;
bitwise 1/2/4/8-worker projection; length-unit conversion; malformed inputs;
and exact Poisson enumeration of detected/missed/other halo populations.
The latter computes count-power covariances from factorial halo-pair
estimators in two volumes. It independently recovers the two-halo factor
two and volume cancellation, with 3e-13 relative agreement. Other algebra
comparisons use 2e-14. It is not a survey accuracy or timing measurement.

The subsequent manual didactic review checked the first versus second
mass-moment indices, full-halo I11 versus selected J11, why a count's
area cancels, the f_K^-2 projection, source transfers and the distinction
between matter partners and discrete cluster partners. The docstrings and
public README state units and output shapes and do not call this a full
cluster covariance. The notebook package exports both count helpers.

## Didactic review of the mass integrator

Completed after the analytic, debug and physical checks, before the next
major implementation ticket. The manual pass checked dimensions, selection
versus catalog normalization, the incomplete response interpretation,
exclusive-category versus different-halo correlations, ownership and odd
SIMD tails. Every intrinsic explains lane contents and the mathematical
operation. Loop overviews explain the physical sum and independent work.
C/header and added C++ lines fit 80 columns; declarations and predicates
are separated. Public README text defines the selected moments and links
the primary papers, with no machine-local harness or bot-skill dependency.
