# Scientific code documentation: explanation and verification

Comments should connect the mathematical reason for an operation to its
implementation. A description of loop mechanics alone does not explain
why the discretization, weighting or memory ownership is correct. Equations,
units and array axes belong beside the calculation they define.

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
- Every SIMDe call needs an immediately adjacent explanation, including
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
Gaussian, halo, SSC and observable projection. The development skill
requires explanation rather than narration in loop overviews, and an
explanation before every SIMDe call.

## Arithmetic-preserving documentation changes (2026-10-03)

In the measured review, only `operators_cov.c`, `perturbation_cov.c` and `non_gaussian_cov.c` have
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

## Validation evidence (2026-10-03)

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

## Covariance-specific interpretation

Several distinctions are essential when documenting the kernels:

- The SSC `signal` contract is the zero-IA mean model.
- Source window[0] supplies the TATT radial weight; it is not audit-only.
- `get_FPT_IA` runs in `ia_cov.c` independently of `nuisance.IA_code`.
- I11 accumulates low-mass panels downward after the upper masses.
- CosmoCov's `tri_2h_13_cov` contains half of the published T_13 convention.
- FFTLog comments must distinguish guard widths from the FFTW 1/N factor.
- Production component bindings live in `components_interface_cov.cpp`.

The reported extra real-space spin factor relative to CCL remains a code
finding, not a correction established by comment editing. Its evidence is
in `test/cosmocov_port_study/14_code_findings_ledger.md`. Documentation work
must preserve that distinction rather than silently changing arithmetic.

Token-stream comparisons across the 47 covariance and 50 data-vector C/C++
files establish arithmetic preservation for the 2026-10-08 comment review.
That check must be complemented by mathematical review: token equality
cannot prove that a comment's physical explanation is correct. The covariance
rebuild and 128-test suite also cover the accompanying halo-menu changes,
whose separate scope is recorded in `halo_model_options.md`.

## Platform and physics facts that comments must preserve

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

## Baryonic-feedback units and comparison conventions

- The CAMB helper (cnu.get_camb_cosmology) returns log10k in h/Mpc;
  get_baryon_suppression and the likelihoods' bfmt requirement read
  1/Mpc. Convert ONCE, at the returned grid, with the standard comment
  (see any port's compute_probes). Never hand the helper grid over raw.
- Apply ln S to a PRIVATE copy of lnPNL with the stride loop
  lnPNL[i :: len(z)] += ln S(z_i); linear tables never change (cluster
  counts and the halo model ride on lnPL_cb - verified at run time:
  des_cluster's per-block table shows the counts column at 0.000).
- The chi2 table distinguishes: method | chi2 |
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
