# Covariance didactic review, 2026-10-03

This pass covers all eight covariance C files, their public headers and
the C++ binding. It is a self-review, not a Fable review. The requested
Fable model was not available in this session.

## Explanation and layout

- Every substantial loop has an overview before the loop or OpenMP pragma.
  It explains the physical reasoning behind the weighting, traversal or
  approximation, then connects that reasoning to the iteration and output.
  Nested loops have explanations at their own level.
- The lensing example derives `g = A - chi*B` from the individual galaxy's
  factor `1 - chi/chi'`. A counts the fraction behind the foreground shell;
  B weights that fraction by inverse distance. Increasing scale factor
  extends both accumulated integrals toward the observer. The trapezoidal
  rule is explained as the area under a straight line between samples.
- All 172 SIMDe calls have immediately adjacent explanations, including
  repeated calls. Comments identify the physical values in the lanes,
  argument order, arithmetic, fused rounding, load/store addresses and
  valid endpoints. Loop overviews distinguish independent outputs from
  partial sums that must eventually be added.
- Allocations, geometric normalization, quadrature, recurrence updates,
  integration and output are separated by comments and blank lines.
  All covariance C/C++/header lines fit within 80 columns.
- The observer endpoint assumption is stated accurately: setting n_z(0)
  to zero alone does not prove that n_z/chi has a zero limit. The existing
  endpoint contract and behavior are unchanged.

The module README gives a reading order from fields and spectra through
Gaussian, halo, SSC and observable projection. The development skill now
requires explanation rather than narration in loop overviews, and an
explanation before every SIMDe call.

## Arithmetic preservation

Only `operators_cov.c`, `perturbation_cov.c` and `non_gaussian_cov.c` have
executable-text changes: nested vector expressions are split into named
intermediate values. Substituting these names recovers the original
expressions exactly, including association and fused operations. All
other reviewed C/C++/header edits are comments or whitespace.

Saved pre-edit libraries and freshly built libraries receive identical
inputs in external `test/covariance_reference/check_didactic_equivalence.py`.
Comparisons use the uint64 views of double outputs, checking actual bit
patterns at one and eight OpenMP threads. The real-space and band
operators, perturbative angular averages, five halo contributions and
both response normalizations are identical. Cases include odd counts,
the exact K=Q diagonal and ratios on either side of it.

## Validation

Strict optimized and debug isolated libraries build successfully, as does
the LSST project binding. The 43 focused covariance checks pass with the
seven isolated debug libraries and the rebuilt optimized spectra binding.
Source checks confirm the operation expressions, per-call comments and
line lengths; `git diff --check` also passes.

All seven project regression suites pass, with their covariance-library
environment variables set and the slow halo checks enabled:

| Project | Passed |
|---|---:|
| roman_real | 104 |
| roman_kl | 49 |
| roman_fourier | 45 |
| des_y3 | 63 |
| lsst_y1 | 100 |
| desy1xplanck | 45 |
| des_cluster | 29 |
| Total | 435 |

No reference was refrozen. The project suites ran concurrently in pairs
for regression validation; their elapsed times are not benchmark evidence.

Raw outputs live outside git under
`test/covariance_reference/results/didactic_review/`: `source_audit.log`,
`equivalence.log`, `debug_covariance.log`, `lsst_build.log`,
`project_tests.json` and the seven `project_*.log` files.

This pass changes no physics, grids, accuracy controls, threading layout
or allocation policy. It does not establish a full Roman covariance,
numerical convergence or performance improvement. Those remain separate
measured reviews.

# C/C++ comment pass over covariances/, 2026-10-08

Nine comment-only commits, f44a1b6 through c9143c9 on `bugfix`, cover
all 47 C/C++ files of `cosmolike/covariances/`. Every batch was drafted
by a subagent and verified by Fable before committing: token-stream
equality against the previous commit (c_strip --collapse, rerun
independently for all 47 files), no new over-80 lines, style scans, and
a full diff read with the physics re-derived where it could be checked
(mask pair-area normalization, TATT E/B coefficients, F3 planar
branches, the 47/21 separate-universe sum, Wick and partition counts,
the Mellin kernel recurrences, DLMF Jacobi coefficients, STS Eq. 33/35
and the fixed-alpha 0.368/alpha(a) identity).

What the pass added, uniformly: SIMDe vocabulary per file and lane
narration at all ~260 call sites, `scalar:` blocks with exact
statements, per-platform fused-rounding facts (fmadd/fnmadd fuse on
arm64 NEON and FMA x86; fmsub fuses only on FMA x86 - verified against
the SIMDe source and the generated arm64 assembly), halo.c-style
function headers with formulas, axes, units and threading, and written
derivations in place of references to study reports.

Comment corrections worth remembering (code unchanged): the SSC
`signal` contract is the zero-IA mean model (matches covariance_ssc.md);
source window[0] is the TATT radial weight, not audit-only;
`get_FPT_IA` runs in ia_cov.c regardless of nuisance.IA_code; I11 adds
tail panels downward after the upper masses; CosmoCov's tri_2h_13_cov
holds half of the published T_13; the core real-space kernels apply the
spin factor a second time relative to CCL full-sky (flagged, not
changed - see the findings ledger); the guard widths and FFTW 1/N in
fftlog; "production bindings" in the README now names
components_interface_cov.cpp.

Report-only code issues moved to
`test/cosmocov_port_study/14_code_findings_ledger.md` (section
"covariances/ C-comment pass"). Post-pass validation: full rebuild and
the lsst_y1 covariance suite (128 passed) ran on the committed tree
together with the halo-menu additions recorded in
`halo_model_options.md`.

## Data-vector C-comment pass, 2026-10-08

Nine review-mode batches over the 50-file data-vector corpus (61,670
lines), Opus-drafted and Fable-verified, committed as 7c7842c (baryons),
77373fa (cluster interface), ea613ca (cluster data-vector), 9452d7d
(scuts/wrappers), a86fb0b (generic_interface), 07ce152
(redshift/IA/pt), 953d312 (cosmo3D/basics/structs), 0a761b5 (halo),
c8e05b6 (cosmo2D). Protocol per batch: independent token-equality rerun
(c_strip --collapse vs the pre-batch commit), over-80 and style scans,
full diff read, and re-derivation or reproduction of the load-bearing
claims (alpha integrations, frozen-reference values, paper LaTeX,
disassembly, GSL/HDF5 probes).

Durable lessons (verified, do not relearn):
- SIMDe rounding is width- and intrinsic-specific. 128-bit
  fmadd/fnmadd: fused by SIMDe on NEON (vfmaq_f64/vfmsq_f64) and on
  x86+FMA. 128-bit fmsub and the 256-bit fmadd/fmsub: composed of
  mul and add/sub INTRINSICS without native FMA - fixed two roundings,
  not compiler-contractible. 256-bit fnmadd/fnmsub: per-lane C loops -
  the compiler contracts them like scalar code (fused under the
  project's clang 19 arm64 flags; two roundings with
  -ffp-contract=off). Check fma.h for the exact intrinsic before any
  rounding claim.
- The omp simd refusals in this codebase come from the strict FP flags
  (-frounding-math -ftrapping-math), not from pointer-to-pointer
  tables (reproduced both ways).
- set_blas_single_threaded holds only for a pthreads OpenBLAS; the
  macOS conda env pins the OpenMP build, which resizes from
  omp_get_max_threads() on every BLAS call outside a parallel region.
  Serial BLAS on this Mac rests on OMP_NUM_THREADS=1 at load.
- The halo option menu couples through tinker_alpha: the SMT01 bias
  rescales the default Tinker 2010 amplitude (0.3207 vs 0.3684 at
  z = 0); see halo_model_options.md caveats.
- MNRAS papers number appendix equations per appendix (eq F1, B1-B2,
  C1-C9): main-text equation numbers in comments that exceed the paper
  count are a red flag worth checking against the LaTeX.

## bfmt feedback-study port, 2026-10-08

The owner's wavenumber-unit fix (roman_real 7c83681, lsst_y1 dfd145c)
was extended fleet-wide: all eight likelihood prototypes request the
bfmt suppression on 10^log10k_interp_2D in 1/Mpc and convert to h/Mpc
only at ci.set_cosmology (audited - the likelihood path never had the
bug); the six projects without the notebook study received it (des_y3
4bae3fc, des_cluster 024fa96, desy1xplanck a45a641, roman_fourier
8e130bb, roman_kl c90f2de - the project's first data-vector notebook -
and des_y6 df63058, whose notebooks were rebuilt on cnu + inline
wrappers and whose bespoke des_y6_notebook.py was deleted). All seven
notebooks executed green and are committed with their chi2 tables.

Fleet conventions (keep for any future feedback work):
- The CAMB helper (cnu.get_camb_cosmology) returns log10k in h/Mpc;
  get_baryon_suppression and the likelihoods' bfmt requirement read
  1/Mpc. Convert ONCE, at the returned grid, with the standard comment
  (see any port's compute_probes). Never hand the helper grid over raw.
- Apply ln S to a PRIVATE copy of lnPNL with the stride loop
  lnPNL[i :: len(z)] += ln S(z_i); linear tables never change (cluster
  counts and the halo model ride on lnPL_cb - verified at run time:
  des_cluster's per-block table shows the counts column at 0.000).
- The chi2 table is three columns everywhere: method | chi2 |
  Delta chi2 | chi2 of shift, no-feedback row first, shift contracted
  with the full-layout zeroed inverse (the cluster path needs
  get_inv_cov_masked_cluster). Against measured data the columns
  separate the cross-term from the shift; against synthetic data they
  coincide.
- BARYON_METHODS is copied verbatim from lsst_y1 EXAMPLE_EVALUATE3
  cell 10; BACCOemu stays commented with a project-true box statement
  (Omega_b floor 0.04001, read from the emulator's parameter_ranges).
- Run pytest per SUBDIRECTORY (tests/data_vector alone):
  covariance/test_forecast.py lazily imports camb from site-packages
  and poisons the cobaya-model tests when collected in the same
  process (its docstring says "Run separately from tests/data_vector").
