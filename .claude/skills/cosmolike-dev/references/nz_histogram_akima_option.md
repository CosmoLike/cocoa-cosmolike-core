# Optional histogram/Akima reconstruction

The production `bugfix` default remains the original midpoint n(z).
`photoz_histogram_akima: 1` is an explicit alternative in every likelihood
YAML of LSST Y1, Roman real/Fourier/KL, DES Y3, DES×Planck, DES cluster and
DES Y6. The two experimental branches remain available; do not replace the
default or restore the earlier area-correction experiment without a request.

## Three-, five- and seven-point experiment

The user authorized `experimental/nz-akima-3-5-7` from the clean
`bugfix` commits. Core and all eight project repositories use that branch.
`photoz_histogram_akima_factor: 3`, `5` or `7` chooses the first preparation,
independently of every accuracy control. The original midpoint curve is still
mode 0, and mode 1 still defaults to three. The source/lens loader caches and
Akima workspace sizes must change when only this factor changes. Test
3 -> 5 -> 3, 3 -> 7 -> 3 and fresh YAML agreement, plus lookup refinement at all three factors.

The study record is `test/nz_akima_three_five/monitor_state.json`.
All eight factor-5 and factor-7 scans completed. Each preserved the
threefold level-6 vector bitwise and matched fresh YAML with the live factor
switch. All 24 permanent reconstruction/factor tests and 17 existing frozen
default checks passed without failures, errors or skips. Six synthetic
reader tests passed in both the strict build and isolated DEBUG/UBSan build,
with no sanitizer findings. The permanent plot script reproduces the
published PNG and numerical checks bitwise.

The independently recomputed accuracy audit and test counts are saved in
`dev_scripts/nz_akima_factor_validation.json`. Raw vector distances for
every level match the study records; fresh selected-level tests reproduce
them. The next lower level exceeds 0.1 for both factors in every physical
project. These are measured practical levels for the stored fiducials,
not a parameter-space accuracy guarantee.
The complete model table and example levels are in `cosmolike/README.md`;
each project saves `tests/nz_akima_factor.json` and its reference vectors.
DES Y6 is identity-metric software checking only. This remains an experiment;
no likelihood integration defaults, frozen inputs or references are changed.

## Implementation contract

- YAML -> likelihood initialization -> `init_photoz_histogram_akima` ->
  `Ntable.photoz_histogram_akima` -> the two public photo-z readers.
- Mode 0 retains their original arithmetic, including existing interpolation
  and z-column controls. Mode 1 ignores `photoz_interpolation_type` but keeps
  `photoz_zmid_convention`.
- `nz_histogram_akima` shares numerical preparation between source and lens,
  with separate cached storage records. Its first preparation is exactly
  `preparation_factor * nzbins`; the chosen factor must NEVER depend on an
  accuracy setting. The experimental branch allows three, five or seven.
- Irregular input bins are overlap-averaged into a uniform preparation grid.
  Fit GSL Akima through its centres and repeated endpoint samples. Subdivide
  its cubic coefficients algebraically for the final uniform lookup; never
  refit a natural cubic. No binary search or allocation in the hot lookup.
- Caller-owned cached GSL workspaces are separate for each tomographic bin.
  Prepare independent rows under `omp parallel for schedule(static)`.
  Allocate/release workspaces serially; preserve bitwise thread determinism.
- A live selector switch reloads existing histogram copies through the usual
  sample setters. Their reconstruction stamps force recomputation of common
  bounds, per-bin support, lens means and dependent kernels. Test 0 -> 1 -> 0.
- Explain cache ownership, every allocation block, threshold scans and the
  coefficient transformations beside the code. Use braces and separate lines
  for every conditional and loop; do not compress control flow.

## Scientific interpretation

Mode 1 is a reconstruction choice, not an accuracy boost. The original
midpoint cubic and the fixed-threefold Akima define different smooth n(z).
The uniform-bin threefold construction preserves total area to rounding;
individual bin areas change. Irregular remapping preserves total area before
Akima, but a general Akima fit need not preserve it or positivity.
No added area correction or clipping is applied by this option.

For uniform cells with normalized masses p and factor m = 3, 5 or 7,
the ideal curve has q_j = p_j + 3/(32m) * (p_(j-1) - 2p_j + p_(j+1));
repeat the endpoint mass for its missing neighbour. At m=3 this is
15 p_j/16 + (p_(j-1) + p_(j+1))/32. This identity explains total-area
conservation and neighbour exchange, not physical superiority of the model.

## Permanent checks and records

Every project has `tests/data_vector/test_nz_reconstruction.py`, using its
frozen NLA fiducial at a measured practical integration level, and the
shared `cocoa_nz_testing.py`. It compares vectors in the same masked covariance,
checks YAML versus live selection and exact default round trips, and audits
integrated cell areas using the actual C readers. Two- and four-point Gauss
rules split at every polynomial knot must agree within 2e-13 per cell.
LSST also exercises uniform/irregular inputs, accuracy boosts and thread counts
in `test_nz_histogram_reader.py`. No tests-only C function was added.

Public explanations, input fingerprints, measured model distances and area
figures live in each project's test README and `tests/nz_reconstruction.json`.
DES Y6's identity-covariance distance is a software metric only. Keep the
actual frozen signed input cells in these tests; three earlier controlled
studies clipped negatives in both arms, so do not silently substitute their
measurements for the permanent raw-input case.

The external integration record is `test/nz_akima_yaml/`. All jobs are
sequential; see its monitor_state.json and run/followups ledgers before
resuming anything. The earlier 1024/2048/4096 study is retained separately in
`test/nz_histogram_experiment/`. No scientific tolerance or frozen reference
is changed. Covariance-generation accuracy is a separate question: the
coarse experimental Akima covariance window/radial refinement checks were
not converged, and the data-vector Delta chi2 test does not validate them.

## Validation and example accuracy

The reference calculation uses level 6 (4096 points). Routine comparison
examples use the lowest measured level near a vector-distance target of 0.1.
This is an explicitly adopted practical test budget, not a strict universal
physics tolerance. Roman Fourier level 3 is retained at 0.106799. Both curves
and their model displacement were checked against the same reference.

| Project | Test level | Akima numerical Delta chi2 vs level 6 | Refined model Delta chi2 |
|---|---:|---:|---:|
| lsst_y1 | 2 | 0.0315961152 | 0.990789032 |
| roman_real | 3 | 0.0332351054 | 0.253947593 |
| roman_fourier | 3 | 0.106798981 | 0.491419608 |
| roman_kl | 2 | 0.00295310751 | 0.636057262 |
| des_y3 | 2 | 0.0454913695 | 0.184156599 |
| desy1xplanck | 3 | 0.0187756726 | 0.249582294 |
| des_cluster | 3 | 0.0166744456 | 0.481671865 |

DES Y6 uses level 0 only as a software check; its identity covariance supplies
no statistical accuracy threshold. All eight mode-selection tests passed at
their practical settings and at level 6. Exact 0->1->0 and YAML/live agreement,
unchanged covariance and the original compiled-reader arithmetic were
verified. All 17 existing frozen default checks passed without refreezing.
The eight 2048->4096 refinements have maximum physical Akima Delta chi2
0.000121638 (DES cluster).

The LSST synthetic uniform/irregular readers preserve the Akima curve under
lookup-grid refinement and are bitwise equal at 1/8 threads. Both checks pass
with changed C/C++ objects compiled at O0 with DEBUG and undefined-behavior /
float-division sanitizers; no sanitizer finding occurred. The eight
covariance-operator checks pass after replacing the stale expectation that
2048-point rules are rejected: 2048/4096 composite rules now receive the same
polynomial-exactness tests, with unsupported sizes still rejected.

Each project README has one two-panel figure: compiled reader curves at an
interior histogram edge and all-bin probability redistribution. Dense edge
samples are verified against the original reader audit. Total-area errors
remain in a table. All saved measurements retain the full 4096 reference;
the public test can reproduce it through COCOA_NZ_INTEGRATION_ACCURACY=6.
