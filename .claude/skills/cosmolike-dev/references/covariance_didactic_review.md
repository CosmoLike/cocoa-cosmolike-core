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
