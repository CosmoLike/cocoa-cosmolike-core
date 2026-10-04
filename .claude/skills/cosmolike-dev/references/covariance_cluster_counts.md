# Cluster count responses: first integration boundary

## Scope

The supplied-abundance component is the first cluster covariance ticket.
It is not a completed 6x2pt+N generator. It adds:

- `counts_cluster_cov.c/h`: count-shell volume conversion, unconditional
  SIMDe, independent (bin,node-pair) OpenMP work with collapse(2).
- `generic_interface_cluster_cov.cpp/hpp`: checked NumPy access, linked
  only by des_cluster; no cluster data-vector globals are read or mutated.
- `covariance/counts_cluster.py::count_statistics`: mean counts, exclusive
  count-bin Poisson noise, count SSC and optional count-two-point SSC,
  using the existing C weighted contraction.
- `des_cluster/tests/covariance/test_counts.py`: independent analytic
  volume/SSC integrals, unit changes, observable transforms, positivity,
  thread repeatability, array tails/ownership and invalid-input boundaries.

All new C is inside covariances/ and ends `_cluster_cov.c`. Existing
data-vector C is unchanged. The notebook's galaxy/shear forecast still
does not include cluster fields or counts.

## Independently checked physics

Read Takada & Spergel (2014), arXiv:1307.4399, Sec. 4.1, and Schaan,
Takada & Spergel (2014), arXiv:1406.3330, Eqs. 33 and 35. Also read
To et al. (2021), arXiv:2008.10757, Secs. 4.1.3 and 4.2, and the DES
joint-model update arXiv:2503.13631, Sec. II.4. The latter requires the
same localizing Y transform on the mean and both sides of the covariance.
Legacy code is a comparison target, not the physical reference.

The implementation follows directly from a comoving shell's volume:

    S_i = dN_i/dchi = Omega f_K^2 n_i
    Phi_Ni = Omega f_K^2 B_i,  B_i = d n_i / d delta_b
    Cov_SSC(N_i,N_j) = integral dchi sigma_b^2 Phi_Ni Phi_Nj.

Here n_i already includes the richness/redshift selection and completeness.
The collapsed long-mode variance sigma_b^2 has units length, not the
dimensionless variance of a finite shell. Counts are absolute, so no
observed-mean subtraction belongs to Phi_N. A two-point response does
retain its estimator's observed-mean subtraction.

For count-two-point SSC use the same Phi_AB as in two-point SSC, whose
local term is W_A W_B D/f_K^2. Multiplication by Phi_N cancels the two
distance powers in that local term. The old core's project_cov_shear_N
multiplies dN/dchi by W_A W_B D without that denominator. Its convention
is not copied. The analytic test and the independent change of length
units detect that error. No conclusion about a historical published
matrix's provenance or parameter bias follows from this code comparison.

Distinct observed bins are Poisson categories even if their underlying
true-redshift or mass distributions overlap. Their Poisson covariance is
diagonal, while their SSC can be nonzero. Shared-object, weighted-count
catalogs need a different shot-noise model and are outside this contract.

Schaan Eq. 35 additionally has a non-SSC count-spectrum term, including
one- and two-halo contributions. The current cross_ssc output deliberately
names its component and must not be promoted to the complete cross block.

## Validation and didactic review (2026-10-04)

Optimized des_cluster build: six checks passed with
`python -m pytest projects/des_cluster/tests/covariance/test_counts.py -q`.
The complete data-vector regression run is tracked with the project
rollout; the new call does not run during likelihood evaluation.

Closed-integral test: constant selected abundances between chi=0.2 and
0.8 give mean counts proportional to (0.8^3-0.2^3)/3 and count SSC to
(0.8^5-0.2^5)/5. A supplied two-point response A/chi^2 makes cross SSC
proportional to (0.8-0.2), with both positive and negative A. Independent
expected values agree within 2e-14 relative. The joint response Gram
matrix is positive semidefinite to roundoff without eigenvalue repair.
Rescaling all lengths by 3000, including the dimensional background
variance and abundances, leaves every integrated output invariant.
Linear recombination on the two-point side transforms the cross block
on that side only.

One/two/four/eight-thread outputs are bitwise identical for one, even
and odd node counts. A subsequent call leaves earlier output arrays
unchanged. Zero abundance response produces Poisson-only count covariance.
Malformed shapes, negative abundance/variance, zero radial weights and
nonfinite values raise Python exceptions before entering C.

An isolated O0 debug build with undefined-behavior and floating-division
instrumentation passes 16 additional cases: 12 count bins, 1/2/9/257
nodes, 1/2/4/8 workers, signed responses and exact scalar-array agreement.
Developer-only artifacts are /tmp/counts_cluster_cov_debug.dylib and
/tmp/check_counts_cluster_debug.py. No project was relinked to debug mode.

The separate manual didactic pass checked: the shell-volume derivation;
selected abundance versus fitted lensing bias; finite-shell versus
collapsed background variance; the local distance-factor cancellation;
absolute counts versus normalized density; exclusive observed bins
versus overlapping true distributions; every SIMDe lane/load/multiply/
store; ownership; loop overview; one comparison per C line; and C/C++
lines within 80 columns. Python setup stays explicit, while array
contractions use the existing C implementation.

No performance measurement was taken during the project regressions.
This ticket establishes a correct boundary, not an optimized full-cluster
runtime or 8-core speedup. No production accuracy setting is certified.

## Next physical integration

1. Supply n_i and B_i from the chosen richness-selection/HMF model on
   common covariance nodes. Do not equate a fitted selection-modified
   lensing bias with a count response without deriving that choice.
2. Build every cluster/galaxy/shear crossed spectrum inside covariances/.
   Check the noisy field matrix and resulting G blocks for positivity.
3. Implement the selected halo moments for SSC/cNG and the non-SSC
   count-two-point contribution, with independent references.
4. Apply the project's Sigma=Y gamma_t convention and selection factors
   consistently, including all count cross blocks, and assemble the actual
   2812-entry layout (3 cluster redshift, 4 richness, 6 lens, 4 source bins).
5. Test full/selected matrices, refinement and Fisher information before
   replacing or recommending a likelihood covariance.
