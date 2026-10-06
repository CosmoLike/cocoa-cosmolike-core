# Wynn validation checkpoint — 2026-10-06

The sequential background checks finished at approximately 18:36 UTC.
This is a validation record of the uncommitted Wynn implementation on
base commit 8125fa358e3c93cef3be83b6ba38ad24e84255d6, not acceptance of
the production change. Roman halo and changed-cosmology checks remain
unresolved. Do not refreeze those checks or weaken their tolerances.

External logs and XML are in test/wynn_validation/20261006. The original
runner (session 3921) stopped at variants.py. The resumed runner (16488)
preserved that failure, corrected the DES x Planck compilation flag, and
completed the independent matrix/build checks. Both sessions have ended;
no numerical job remains from these runners.

## Project suites

| Project | Data-vector sector | Covariance sector |
| --- | --- | --- |
| LSST Y1 | 57 passed | 121 passed on the final rerun |
| Roman real | 96 passed, 8 failed | 1 passed |
| Roman Fourier | 45 passed | 1 passed |
| Roman KL | 49 passed | 1 passed |
| DES Y3 | 63 passed | 1 passed |
| DES x Planck | 45 passed | 1 passed after enabling covariance |
| DES cluster | 29 passed | 50 passed |

LSST's two initial covariance failures checked the previous mass-panel
defaults. Both are covered by the successful 121-test final sector run.
DES x Planck initially skipped covariance because the external runner
unset IGNORE_COSMOLIKE_DESY1XPLANCK_COVARIANCE; the actual Makefile option
is IGNORE_COSMOLIKE_DESXPLANCK_COVARIANCE. A corrected rebuild and the
forecast test passed. Do not count the earlier skip as validation.

The environment requested eight OpenMP threads and one BLAS thread.
Individual test modules retain their own settings, including explicit
thread-count sweeps. No simultaneous numerical jobs were used.

## Complete LSST matrix

The production interface generated all four 1560x1560 components with
no entries masked. Construction took 49.9666 s in one run at eight threads;
initialization including CAMB took a separate 0.3982 s. These are single
measurements, not evidence of a precise speedup over the earlier 50.6281 s
run with a 1e4 mass cutoff.

| Reference | Largest total generalized variance change | Positive totals |
| --- | ---: | --- |
| Previous production, 1e4 cutoff | 1.3342528613e-5 | Both |
| Original production, 1e6 cutoff | 1.3250417699e-5 | Both |

Changes are fractional, not percentages. The first comparison corresponds
to 0.0013343%. Gaussian matrices are bitwise identical in both comparisons.
Relative to the previous production total, the largest absolute variance
mode of each component difference is 1.346858596e-5 for SSC and
1.718991164e-6 for cNG. Components are kept separate; neither component
is inverted independently. The smallest eigenvalue of the candidate,
rescaled by the previous total's diagonal, is 2.269851575e-4.

The comparisons verified matching cosmology, survey geometry, ordering,
power inputs and accuracy controls, apart from the documented mass
panels/execution diagnostic. They checked finiteness, symmetry, component
sums and Cholesky positivity without repairing matrices. These are
old/new implementation comparisons, not independent accuracy or Fisher
convergence proofs. The sigma-table domain and FFTLog settings also changed;
do not attribute every difference to Wynn alone.

Saved records in OneCov-benchmark-:

- work/cutoff_full_production_wynn40/report.json and covariance.npz.
- results/wynn40_vs_production4_20261006.json.
- results/wynn40_vs_native6_20261006.json.
- Matching four-panel figures in results/figures/.

Rechecked SHA256 values:

- covariance.npz: 7cecb9be107caf9e0d4e002cd74cbe3d1cb3583316a6f87ee7abdf1d329a3abb.
- power_tables.npz: 26ed1a29b47f82e2aa3de13de93e5ccd3bf88bc7ceb802125157ca12211fff70.

## Build-mode and recurrence checks

The isolated debug library passed all 12 covariance halo tests. The
isolated covariance-disabled library loaded without covariance bindings
and reproduced both NLA and TATT data-vector reference chi-squared values
exactly. These builds did not relink the production LSST interface.

The extracted production C epsilon recurrence matched the three saved
high-precision sequences within 3.40e-12 absolutely. This verifies that
recurrence implementation for those inputs, not stability for arbitrary
cosmologies or partial-integral sequences.

## Unresolved checks and next diagnosis

1. Roman real: seven frozen halo probes changed. The independent variance
   check also fails at M=2.43287e15 Msun/h: its relative discrepancy is
   2.01673e-5 against the unchanged 2e-5 criterion. Refining the direct
   integral from two to four million samples changes that reference by
   only 2.13e-10 relatively, so this is not explained by reference sampling.
   A diagnostic query at 1e17 has a larger discrepancy, 1.82652e-4. The
   current diagnostic's boost calls returned identical values; verify
   actual grid/cache refinement before interpreting that as convergence.
   Diagnose FFTLog weighting, padding and interpolation separately before
   refreshing snapshots.

2. Changed-cosmology I11: Omega_m=0.25, n_s=0.92, A_s=1.7e-9 has a maximum
   fractional difference of 9.34546e-5 at z=10 and k=24.08225 h/Mpc between
   default Wynn and the deep finite 256-node control. Wynn gives
   1.0000034750 and the control 0.99991002885. The original criterion is
   1e-5. Lower-redshift maxima through z=3 are at most 2.36e-6.
   The second changed cosmology passes, with maximum 1.15821e-6.
   Separate finite-rule refinement, low-mass interpolation and epsilon
   sensitivity before changing production code. I11(0)=1 passes both.

Preserve all completed runs. Future fixes require the affected tests and
appropriate full-matrix rechecks; this checkpoint alone does not authorize
calling the Wynn implementation fully validated.

## Follow-up diagnosis and targeted fixes

The original failures above are preserved as the pre-fix record. The
follow-up identified two numerical causes without changing the halo fits,
mass domain, completion prescription or scientific test tolerances.

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

After rebuilding the production interfaces, 15 LSST halo tests passed,
including a permanent three-cosmology regression through z=39 and
k=300 h/Mpc with the unchanged 1e-5 threshold. All 25 non-frozen Roman
halo tests passed, including variance, fit/integral identities, caches
and thread repeatability. The old seven-reference discrepancy shrinks
to at most 6.80e-10 after correcting the FFTLog weighting; fitted fnu,
hb1nu and the profile probe are unchanged.

These checks justify regenerating Roman's halo snapshots with its
documented generator. The prior snapshot and manifest are preserved in
the external validation folder. No likelihood chi-squared reference or
survey data needs regeneration for these fixes.

The sequential `stability_rerun.sh` rechecks all seven project suites,
the full LSST component matrices, debug and covariance-disabled builds.
It uses fresh output paths under `stability_rerun/`; monitor its status
before claiming final acceptance. The implementation and project updates
are committed after the targeted checks at the explicit request to make
incremental commits. The broader rerun continues unchanged; its results
will be recorded in a follow-up commit. Committing is not a claim that
the full validation sweep has finished.
