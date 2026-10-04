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
