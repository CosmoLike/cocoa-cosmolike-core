# Roman covariance negative-mode review, 2026-10-03

## The delivered matrix

The active `roman_real/data/example1.dataset` selects `example1_cov`:
2115 entries, with 1950 retained by `example1.mask`. The older file named
`cov_roman_real` has 1160 entries and a different layout; it is not the
matrix analyzed here. No covariance, mask or frozen reference was edited.

The external `test/covariance_reference/diagnose_roman_eigenvalues.py`
reads all entries, separates columns 8 and 9 into Gaussian G and combined
non-Gaussian NG, checks completeness and duplicate agreement, and tests
the full matrix and selected principal submatrix with one BLAS thread.
There are 2,237,670 unique unordered entries and 14,805 duplicate entries.
The duplicates agree exactly in both G and NG. Symmetry is exact.

For numerical diagnosis use the correlation matrix
`R_ij = C_ij / sqrt(C_ii C_jj)`. This diagonal rescaling preserves the
number of positive/negative directions while avoiding the enormous range
of units and amplitudes in the raw matrix.

| Matrix | Smallest eigenvalue | Negative eigenvalues | Cholesky |
|---|---:|---:|---|
| Full C | -4.2741810162e-12 | 1 | fails |
| Full R | -0.49847819898 | 1 | fails |
| Gaussian C alone | 8.5210941479e-16 | 0 | passes |
| Selected R, 1950 entries | 0.00079735042777 | 0 | passes |
| Shear-only R, 1080 entries | 0.00153096357914 | 0 | passes |
| Galaxy family R, gamma_t + w | 0.00121700634168 | 0 | passes |
| xi_minus + gamma_t principal submatrix | -0.39073387196 | 1 | fails |

Every individual probe submatrix is positive definite. Of the six
two-probe combinations, only xi_minus + gamma_t fails. This localizes
the problem to their joint correlations, rather than either auto block
alone. It does not establish whether one auto block is underestimated,
the cross block overestimated, or both.

## Where the failing direction lives

For the unit eigenvector v of the full R's negative mode:

- `v^T G_scaled v = +0.73766224803`;
- `v^T NG_scaled v = -1.23614044701`;
- their sum is `-0.49847819898`, with eigen-equation residual 7.8e-14.

The failure requires the stored NG contribution; G alone is a valid matrix.
This does not establish which physical or numerical prescription is wrong.
The negative direction is not a small floating-point rounding error.
An NG contribution may be indefinite, because a connected cumulant need not be a
covariance; it is the negative **total** variance that is unacceptable.

Squared mode weights are 55.47% gamma_t, 35.46% xi_minus, 4.64% xi_plus
and 4.44% w. The first three angular bins account for 57.54% of the mode;
44.83% lies outside the production mask. The largest individual weights
are gamma_t at 2.972 arcmin, especially lens bin 0 with source bins 7,
6 and 5 (zero-based catalog indices).

Keeping every shear datum, removing only the smallest gamma_t/w bins gives:

| Smallest bins removed per gamma_t/w row | Matrix size | Minimum eigenvalue of R |
|---:|---:|---:|
| 0 | 2115 | -0.4984782 |
| 1 | 2046 | -0.2338292 |
| 2 | 1977 | -0.0246594 |
| 3 | 1908 | +0.00079735 |

These removals are diagnostics, not proposed cuts or a repaired covariance.
The existing project mask is a different selection and also passes.

## A test independent of comparing raw eigenvalue sizes

Partition the matrix into shear and galaxy families. Both auto blocks
are positive definite, so write them as `A = L_A L_A^T` and
`B = L_B L_B^T`. For their cross block X, form

`Q = L_A^-1 X L_B^-T`.

The transformed joint covariance is `[[I,Q],[Q^T,I]]`. A singular value
s of Q produces eigenvalues `1+s` and `1-s`; hence positivity requires
every s below one. The delivered cross block has one singular value
above one: **1.0409946763**. This proves that its cross correlations are
too strong relative to its auto blocks.

Keeping both auto blocks fixed and varying only the NG part of X gives
maximum singular values 0.99495 at fraction 0, 0.99500 at 0.9, 1.00128
at 0.95, 1.02394 at 0.98, and 1.04099 at 1. This is another diagnostic;
rescaling the cross block would not be a justified physical correction.

## Cross-lens NG omission: a stronger structural lead

The delivered file has exactly zero stored NG in all 936,450 symmetric
matrix positions coupling galaxy observables with different lens-bin
labels. Their Gaussian terms are generally nonzero (927,900 positions).
The zero test uses column 9 directly, avoiding cancellation in C-G.
Adjacent lens redshift distributions overlap: normalized dot products
range from 0.186 to 0.420. Disjoint physical support cannot explain the
whole cross-lens zero pattern.

The local legacy `run_covariances_real_fullsky.c` explicitly writes NG
only for equal lens bins: clustering/clustering at line 79, galaxy-shear
with galaxy-shear at line 131, and clustering with galaxy-shear at line
358. This drops cross-lens results even when the underlying projected
integrand can calculate them. The shipped zero pattern is consistent
with these output rules.

Holding all 1080 shear entries, and retaining galaxy observables from
just one lens bin at a time, gives a positive-definite principal submatrix
for each of the eight choices. The smallest minima range from 0.0007425
to 0.0015308. But the union of the first two lens families already fails:

| Galaxy lens bins included alongside all shear | Minimum correlation eigenvalue |
|---|---:|
| 0 alone | +0.00074255 |
| 1 alone | +0.00153076 |
| 0 and 1 | -0.31628726 |
| 0, 1 and 2 | -0.43355445 |
| All eight | -0.49847820 |

Thus combining lens families creates the failing direction even though
each separately has a valid joint covariance with shear. This makes the
omitted cross-lens NG terms a particularly direct suspect. It does not
prove that restoring them alone will make the full matrix accurate or
positive; no missing term has been guessed or inserted.

A three-observable SSC example explains the risk without quadrature or
interpolation. Give one shear observable and two galaxy observables unit
response to one common background mode. Their covariance is an all-ones
outer product, with eigenvalues 0, 0, 3. Deleting only the galaxy/galaxy
cross term leaves `[[1,1,1],[1,1,0],[1,0,1]]`, whose eigenvalues are
`1-sqrt(2), 1, 1+sqrt(2)`. The negative variance appears solely because
the retained couplings no longer describe a consistent shared fluctuation.

The rewrite must compute cross-lens covariance when the windows overlap,
including correlations absent from the chosen data vector. They remain
covariance-owned calculations. Check this omission before attributing the
historical failure solely to interpolation or integration precision.

## A demonstrated legacy interpolation failure mechanism

The local CosmoCov checkout's `theory/covariances_fourier.c` uses different
numerics for different NG blocks:

- shear/shear: 20 by 20 nodes, interpolation of log covariance, followed
  by a shear-only high-ell taper (`bin_cov_NG_shear_shear_tomo`, lines
  283--315 in the inspected checkout);
- galaxy/shear and galaxy/galaxy: 40 by 40 nodes, interpolation of values,
  without that taper (lines 318--352 and 455--488).

Separate accurate-looking interpolants need not represent one consistent
covariance. The external `check_mixed_interpolation.py` makes this concrete
using a single SSC background mode with positive smooth response
`R(ell)=ell^-1`. Its exact covariance is an outer product and is therefore
positive semidefinite. At ell=4729.72, below the taper, the mixed legacy
rules produce correlation 1.00963638 and eigenvalue -0.00963638. Forming
all blocks from the same interpolated responses instead gives correlation
one and eigenvalues zero and two, as the rank-one example requires.

This establishes a possible numerical failure mechanism. It does **not**
prove which algorithm/version/configuration generated the delivered Roman
matrix, or exclude quadrature and physical-model errors. Its file records
G and SSC+cNG together, and the original cosmology, survey inputs and
generation settings have not been located. The challenge repository supplied
by the owner gives a closely related configuration, discussed below.
A faithful regeneration is needed to assign
the historical root cause, refining interpolation and integration
separately and retaining SSC and cNG as separate outputs.

## The owner's Roman data-challenge lead

The owner supplied the
[Fourier medium-tier challenge](https://github.com/CosmoLike/roman_cpip_data_challenge/tree/d1c1a2cffee1ad9167aae8e4d80666cb295cf963/data_challenge1_fourier_medium).
The same repository has a
[real-space medium-tier README](https://github.com/CosmoLike/roman_cpip_data_challenge/blob/d1c1a2cffee1ad9167aae8e4d80666cb295cf963/data_challenge1_real_medium/README.md)
specifying 2415 square degrees, total n_eff=41.3 per square arcminute,
eight bins, and 15 angular bins from 2.5 to 250 arcmin. It identifies
CosmoCov as the generator and explicitly reports the unmasked matrix's
failure of positive definiteness. Its supplied cut uses a 1.5 Mpc/h scale.
These are facts about that release, not a numerical-convergence criterion.

At repository commit `d1c1a2c`:

- The real-space mask is byte-identical to local `example1.mask`.
- Every covariance index, angle and tomographic-bin label is identical.
- The redshift grid agrees, but the bin densities differ: their relative
  Euclidean norm difference is 4.53%, excluding the redshift column.
- The covariance differs numerically, not just in formatting or headers.
  For example, its Gaussian/local diagonal ratios range from 0.1613 to
  1.2368; a single survey-area rescaling cannot explain the difference.
- The challenge's full correlation matrix has minimum eigenvalue
  -0.4789368542; its second eigenvalue is +0.0006391950. After applying
  the common mask, the minimum is +0.0006391960.
- Its negative eigenvector has absolute overlap 0.9937138580 with the
  local matrix's negative eigenvector. The failure direction is therefore
  very similar in two numerically different matrices.

The public history and archived challenges were also inspected. None of
their covariance LFS fingerprints or redshift-file blobs matches the local
inputs. The example MCMC YAML contains priors and sampling reference
points; these do not establish the covariance's generation cosmology.
It is not a CosmoCov input configuration, and supplies no quadrature or
interpolation settings. Thus the lead establishes a documented family
of inputs and a known positivity problem, but does not close provenance
or identify the historical numerical/physical cause.

SHA-256 fingerprints:

| Input | Local | Medium-tier challenge |
|---|---|---|
| covariance | `e279f38c1c415b4866b39d330191e33c319f508f2ac5798592833947aa237cd1` | `9b0cb0182a14fb5cb6597a4bc21b5beeb43cf5baa738e28a3a1f9323cad7e347` |
| redshift distribution | `142a7068e93e432cad126c8d21b0285988210fd6e7d44fe2a4868f6d8c3576b6` | `54267f52d08bf27c430a6a335db2b6559e1b06ce43e9b6a8b3b393778fd46e3c` |
| mask (both) | `de63095b81c2cb01e94a5e6628c0c52b5925d4cc6917239fc79371b462063779` | identical |

Read the real-space directory for Roman Real; the Fourier directory's
bin selection and matrix dimension differ. No local input was replaced
with the challenge release during this review.

## Consequences for the rewrite

For SSC, interpolate shared responses and form their covariance on a
common positive radial rule. If rows B contain responses multiplied by
the square roots of those weights, `C_SSC = B B^T` is positive semidefinite
even when individual responses are negative. Interpolating each finished
auto/cross block with different nonlinear rules loses this property.
Likewise, a common linear interpolation operator S preserves positivity
through `S C S^T`; changing individual covariance entries independently
does not generally preserve it. Cubic interpolation can be linear in its
input values, but taking logarithms of separate covariance blocks is not.

This structure follows the response formulation of
[Takada & Hu (2013), Section II.C and Appendix A](https://arxiv.org/pdf/1302.6994).
It is also consistent with the validation lesson of
[Fang, Eifler & Krause (2021), Section 4.2.1](https://arxiv.org/pdf/2004.04833):
their LSST quadrature covariance had a negative eigenvalue and was excluded
from subsequent likelihood analysis. Neither paper identifies the cause
of this particular Roman file.

Retain checks on the full delivered and selected total covariance. Do not
clip eigenvalues, add a ridge, silently change masks, or require cNG alone
to be positive definite. A positive matrix still needs convergence tests
in parameter errors/FoM before it is an accurate covariance.

## Evidence and review

External artifacts in `test/covariance_reference/results/`:
`roman_eigenvalue_diagnosis.json` (including the input SHA-256),
`roman_negative_mode.npz`, `roman_negative_mode.png`, and
`mixed_interpolation.json`. `diagnose_cross_lens.py` and
`roman_cross_lens_diagnostic.json` record the lens-family selection and
stored-zero checks. Challenge evidence is in
`roman_challenge_comparison.json`, `roman_challenge_tree.json`, and the
downloaded, pinned files in `roman_challenge_source/`. The diagnostic scripts and source review
were read again for units, matrix scaling, interpretation of signed
contributions, and the distinction between localization and root cause.
No production computation changed in this ticket.

## Future-generator regression: physical Roman SSC recomputation

The owner clarified that the priority is a reliable new Roman covariance,
not recovering the provenance of the old file. The old negative mode is
now a demanding test direction, not a prerequisite for proceeding.

The external `recompute_roman_ssc_mode.py` computes actual halo-response
SSC for the Roman redshift distributions and 15 real-space angular bins.
It compresses the 2115 measurements into nine variables: shear and the
eight lens families. Within each family the weights are the stored
correlation eigenvector divided by the original standard deviations.
Summing the nine variables therefore preserves the exact failing
direction; the stored compressed total variance is -0.498478198980569.

This is a new controlled physical calculation. It uses the Roman
fiducial from the notebook adapter with massless neutrinos, so cb and
total matter coincide, and a 2415-square-degree spherical-cap footprint.
Magnification and RSD are off. The initialized NLA/redshift/bias choices,
cosmology and input covariance fingerprint are saved with each result.
These are explicit modeling inputs, not a reconstruction of the old file.

At each distance the calculation:

1. Reads common covariance-owned lens/source windows.
2. Computes the halo moments needed for the two-halo slope and the
   one-halo abundance response. All halo inputs refer to the same field.
3. Constructs a smooth linear-power row from the supplied CAMB table,
   with explicit power-law endpoints, before evaluating the dilation slope.
4. Computes both the published isotropic halo response and the alternative
   projected-tree coefficients as separate modeling choices.
5. Splines the logarithm of the positive response onto a dense log-k grid.
   The integer-multipole transform then uses only linear table lookups.
6. Applies the full-sky bin operators on ell=2..ell_max, with the observed
   shear transfer factors, and subtracts the complete projected
   survey-mean response.
7. Forms SSC from common radial response factors. Cross-lens terms are
   present automatically. No finished covariance entry is log-splined.

The response has 129 exact nodes plus four padding nodes in the pilot,
then 4097 dense nodes; refinement uses 257+4 exact and 8193 dense nodes.
At a fixed distance log(ell+1/2) differs from log physical k by a constant.
This experiment tests interpolation at fixed a, not the rejected coarse
evolution table along a moving Limber wavenumber.

| Setting | Pilot | Refined |
|---|---:|---:|
| Radial nodes, four a panels | 256 | 512 |
| Mass nodes, eight log-M panels | 1024 | 2048 |
| Exact response nodes, with padding | 133 | 261 |
| Dense response nodes | 4097 | 8193 |
| Maximum integer ell | 50000 | 100000 |
| Angular nodes per bin | 512 | 1024 |
| Maximum mask multipole | 4096 | 8192 |

The refined nine-variable **SSC correlation matrix** has:

| Response | Complete minimum eigenvalue | Cross-lens terms deleted |
|---|---:|---:|
| Isotropic | +0.0069079983 | -0.0898583558 |
| Projected tree | +0.0063019857 | -0.0943281163 |

Even the three-variable subset containing shear and the first two lens
families reproduces the failure: +0.0165668554 becomes -0.0610020237 for
the isotropic response. The projected-tree case behaves similarly.
The removed cross-lens contribution in the stored failing direction is
+1.2581760903 or +1.4260918933, respectively. These are variances in that
explicitly normalized diagnostic direction, not corrections inserted into
the shipped covariance.

The complete physical SSC is positive, while deleting only the cross-lens
terms makes it indefinite. Thus the unsafe omission has now been tested
with the actual radial/halo/real-space calculation, beyond the earlier
rank-one toy example. A future generator must compute every requested
cross block; different lens labels are not a reason to set NG to zero.

Refinement changes the matrix by 4.19e-7 and 1.42e-6 in relative
Frobenius norm. More stringently, generalized refined/pilot covariance
eigenvalues lie in [0.99994047, 1.00013202] for isotropic and
[0.99993738, 1.00010638] for projected tree. Every direction in this
nine-dimensional space is therefore stable within 1.33e-4 for this
combined refinement. This does not replace a full-matrix or Fisher FoM
test and does not assess nonlinear response-model uncertainty.

The archived physical responses were also contracted with production
`gaussian_project_cov`. C agrees with independent NumPy to at most
7.97e-16 after diagonal scaling. Uneven rectangular subblocks reconstruct
the same result bitwise at 1, 2, 4 and 8 threads. No C routine calls MPI;
the arrays already support separate Python/C++ block dispatch.

A permanent LSST SSC regression now checks uneven subblock assembly
against an independent response contraction, all cross entries, positive
definiteness and 1/2/4/8-thread consistency. The new halo regression checks
small (one a, two k) batches against the corresponding larger table.
The physical response calculations remain external, with results
`roman_physical_ssc_{pilot,refined}.{json,npz}`,
`roman_physical_ssc_refinement.json` and
`roman_physical_ssc_C_projection.json`.

Remaining generator work is full G+SSC+cNG assembly with complete pair
coverage, including non-Limber spectra, and positivity/convergence of both
the full Roman matrix and the likelihood selection. Positivity of these
nine compressed SSC variables does not prove positivity of the final
2115-dimensional total covariance. No covariance or mask was replaced,
and no eigenvalue clipping or diagonal regularization was applied.
