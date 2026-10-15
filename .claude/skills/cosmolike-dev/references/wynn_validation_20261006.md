# Wynn extrapolation: stability mechanisms and validation

## Regression evidence (2026-10-06)

The sequential `stability_rerun.sh` record contains 571 passing project
tests, with no failures, errors or skips. Every
`test_*.py` module in both sectors of all seven projects has a passing
XML result. This validates the regression sweep for implementation
`d95867f`, including FFTLog bias 0.8 and the growing-correction Wynn guard.

| Project | Data-vector tests | Covariance tests |
| --- | ---: | ---: |
| LSST Y1 | 57 | 124 |
| Roman real | 104 | 1 |
| Roman Fourier | 45 | 1 |
| Roman KL | 49 | 1 |
| DES Y3 | 63 | 1 |
| DES x Planck | 45 | 1 |
| DES cluster | 29 | 50 |
| Total | 392 | 179 |

Roman's independent variance check passes its unchanged tolerance.
Only its halo snapshots and associated manifest entries were regenerated,
after the corrected probes differed from the old snapshots by at most
6.80e-10. The original files were preserved before using the documented
`--halo` generator; likelihood references and survey data were unchanged.

The two changed-cosmology I11 checks have maximum fractional differences
9.59198e-8 and 4.86316e-7 against the deep finite control, through z=39
and k=300 h/Mpc. Both pass the existing 1e-5 criterion. The extracted C
recurrence agrees with the three high-precision sequences within 3.40e-12.

The isolated debug library passed 12 halo tests. The covariance-disabled
library reproduced NLA and TATT data-vector chi-squared values exactly:
0.023705928404497645 and 0.011953501626625753. Neither isolated build
overwrote the installed production interface.

### Full LSST matrix

The validated production interface generated all 1560 by 1560 entries of
G, SSC, cNG and total. The comparisons retained every entry and verified
finiteness, symmetry, component sums, matching inputs and Cholesky
positivity. All three total matrices are positive definite without repair.

The table gives the largest absolute generalized variance change for
each component difference, using the positive **total reference** as the
metric. These values are fractional, not percentages. Components are
not inverted separately, and their maxima need not add to the total's.

| Reference | G | SSC | cNG | Total |
| --- | ---: | ---: | ---: | ---: |
| Previous production, 1e4 cutoff | 0 | 3.71889e-7 | 1.14586e-6 | 1.26732e-6 |
| Initial Wynn, before stability fixes | 0 | 1.34337e-5 | 2.47536e-6 | 1.32221e-5 |

G is bitwise identical in both comparisons. Relative to previous
production, total variance ratios span 0.999999508356 to 1.000001267319:
a maximum change of 1.27 parts per million (0.0001267%). The candidate's
smallest eigenvalue after reference-diagonal normalization is 2.26985e-4.
Against initial Wynn, the ratios span 0.999986777860 to 1.000012275071.

The candidate uses core `a35dbc9` (a documentation descendant of
`d95867f`) and LSST `601f9cf`. Artifact SHA256 values are:

- Interface: 22e2e3499ac4de7bdcd52eaee62fb148a30d3773d1c49e684d827e8943d079cc.
- Covariance: 23135f138c14b3dd50abf3ac5ec30fce4e31f161359e189780bb037c961dbc78.
- Power tables: 26ed1a29b47f82e2aa3de13de93e5ccd3bf88bc7ceb802125157ca12211fff70.

Saved comparison records in OneCov-benchmark- are
`results/wynn_stability_vs_production4_20261006.json` and
`results/wynn_stability_vs_initial_20261006.json`, with matching four-panel
figures. The source run is `work/cutoff_full_wynn_stability/`.
Test XML, logs, snapshot backups and `final_verification.json` are in
`test/wynn_validation/20261006/stability_rerun/`.

## Numerical failure mechanisms

The two relevant mechanisms are FFT periodic error in the variance table
and amplification of nearly converged partial sums by high epsilon order.
They require separate diagnostics; increasing one integration rule cannot
resolve both.

- **Variance:** bias 0.5 leaves a periodic FFT error at large radii. Bias
  0.65, 0.8, 1.0 and a doubled FFT interval at bias 0.5 were checked in an
  isolated interface. Bias 0.8 reduces the maximum Roman variance error
  from 1.82652e-4 to 6.98870e-6, below the existing 2e-5 criterion. It
  agrees with the doubled-interval check within 4.98e-14 on the nine
  resolved masses and 1.32e-8 on the sampled low-mass tail. Bias 1.0 has
  worse tiny-radius roundoff, so retain 0.8.
- **I11:** 96/128/256/512-node rules leave the failed discrepancy near
  9.3e-5. Captured partial sums show that the highest epsilon order
  amplifies tiny profile differences; 60-digit arithmetic reproduces it.
  A small guard retains the previous estimate when the next extrapolation
  correction grows. The three original saved fiducial sequences keep
  their highest-order estimate. This guard is a numerical stability check,
  not a calibration of the unobserved halo population.

The changed-cosmology I11 failure occurred at Omega_m=0.25, n_s=0.92,
A_s=1.7e-9, with a fractional discrepancy of 9.34546e-5 at z=10 and
k=24.08225 h/Mpc against the deep finite control. Its persistence under
quadrature refinement and reproduction at 60-digit precision identify
extrapolation sensitivity rather than ordinary finite-sum roundoff.
I11(0)=1 alone did not expose this finite-k failure.

For Roman variance, refining the direct reference from two to four million
samples changed it by only 2.13e-10 relatively while the discrepancy remained
above the 2e-5 criterion. Independent reference convergence is therefore
essential before changing production sampling or refreshing snapshots.
After FFTLog weighting correction, the seven halo-reference differences
were at most 6.80e-10; likelihood references and survey data were unchanged.

## Validation scope and reusable checks

- Compare Wynn with deep finite controls at multiple cosmologies, redshifts
  and finite k. Test the zero-wavenumber normalization separately.
- Diagnose FFTLog weighting, interval continuation and interpolation
  independently. Identical outputs under a nominal boost are not evidence
  of convergence unless the intended table actually refines.
- Keep failed attempts and snapshot provenance in the raw records. A skipped
  covariance test is not validation: the DES × Planck build key is
  `IGNORE_COSMOLIKE_DESXPLANCK_COVARIANCE`, not a name containing DESY1XPLANCK.
- Separate covariance-disabled, debug and full-matrix checks. Small halo
  fixtures do not establish the stability of every survey matrix.
- Treat single-run or concurrent-job elapsed times as diagnostics, not quiet
  machine performance benchmarks.

The recorded checks support regression acceptance of the stabilized
implementation, not interpolation, integration or Fisher convergence, nor
physical calibration of unresolved halo masses. The intermittent
supplied-covariance inversion issue has a separate diagnosis; a passing
sweep alone cannot establish its resolution.

Detailed pre-correction measurements remain in
`test/wynn_validation/20261006/` and the OneCov-benchmark- records
`results/wynn40_vs_production4_20261006.json` and
`results/wynn40_vs_native6_20261006.json`. The stabilized full-matrix
comparisons and hashes above identify the accepted configurations. Changes
in sigma-table coverage and FFTLog settings accompany the tail prescription,
so an old/new matrix difference cannot be attributed to Wynn alone.
