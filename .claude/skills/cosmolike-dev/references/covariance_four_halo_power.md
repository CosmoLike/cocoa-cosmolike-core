# Global covariance power refinement

## Production decision

The selective four-halo C prototype was retired after its measured full
run was slower than global input refinement. Its patch and completed tests
are preserved outside git in the external study. No new C API is shipped.

The adopted shared Python preparation uses `power_accuracyboost=8`;
`power_refinement` is this factor times the global `accuracy_boost`.
The initializer fills all linear, nonlinear and cb log-power tables using
a natural cubic spline at fixed z, preserving original nodes. It returns
installed tables in the existing interchange format. The C readers retain
linear lookups. AB1 gives 11,993 k nodes and AB2 gives 23,985 from the
original 1,500. Integration level is independent; data-vector preparation
is unchanged. Notebook boost loops reinitialize power before assembly.

## Accuracy measurements, 2026-10-06

The controlled scan uses 45 unordered pairs from nine k values between
0.001 and 10 h/Mpc, at z=0.1, 0.5 and 1. The same C angular kernel receives
either dense linear lookups or direct natural-cubic power inputs. The
latter is a smooth-input control, not a new trispectrum implementation or
an independent CAMB calculation. Of the 45 pairs, 36 have every internal
sample inside the original power domain; they contain the largest errors.
The other pairs probe extrapolation and are recorded separately.

| k nodes for 4h | Maximum fractional 4h difference from smooth input |
| ---: | ---: |
| 1,500 | 61.6316% |
| 2,999 | 19.3879% |
| 5,997 | 1.73172% |
| 11,993 | 0.487531% |
| 23,985 | 0.388319% |
| 47,969 | 0.290414% |

These numbers do not establish convergence at all k or of a Fisher matrix.
The earlier TJPCov diagnostic used not-a-knot spline boundaries; this
experiment consistently uses natural boundaries in both local/global cases.

All three complete LSST Y1 real-space matrices have 1,560 entries and are
positive definite. The following comparisons use generalized eigenvalues
against the positive total, so every measured-vector direction is included.

| Comparison | Maximum total variance-mode change |
| --- | ---: |
| Selective factor eight versus native | 0.00243077% |
| Global factor eight versus native | 0.0189755% |
| Selective versus global factor eight | 0.0189115% |

Selective versus native preserves G and SSC bitwise. Its cNG relative
Frobenius change is 1.49209e-5; the total's is 5.53550e-6.
Selective versus global has component changes, relative to the total
variance, of 8.42732e-5 (G), 1.87876e-4 (SSC) and 5.25431e-6 (cNG).
Global refinement modifies linear, nonlinear and cb input tables, including
their edge extrapolation secants; those extra component changes are expected.

## Quiet-machine timings

Apple M2 Pro, eight OpenMP threads, BLAS one. Three fresh processes per
case ran sequentially in rotating order. No numerical/compiler jobs
overlapped. The table reports mean and sample standard deviation of
preparation plus full covariance construction; CAMB initialization,
writing and eigenvalue diagnostics are excluded. Each case's four saved
matrix components repeat bitwise across all three processes.

| Scope | Seconds | Mean process peak memory |
| --- | ---: | ---: |
| Original 1,500-node input | 51.79 +/- 1.82 | 1,797 MiB |
| 11,993 nodes for 4h only | 67.87 +/- 1.44 | 1,819 MiB |
| 11,993 nodes globally | 56.62 +/- 0.22 | 1,892 MiB |

The selective path is 19.9% slower than global refinement on this laptop.
It saves about 73 MiB of whole-process peak memory. These memory numbers
include all libraries and covariance arrays, not only power-table storage.
They do not establish the ordering on x86 hardware.

Preparing one selective 11,993-node table takes 0.122 +/- 0.015 ms in
25 warm helper calls, with free excluded. Global preparation averages
0.132 seconds. The selective slowdown therefore is not the cubic setup:
the shared angular worker evaluates native powers for lower halo orders
and extra refined powers/terms for 4h. The selective experiment is retired. The user chose globally refined
power as the covariance production default.

## Retired selective-prototype validation

Targeted compiled tests passed: three new interpolation/API/thread tests,
14 notebook utility tests, five existing angular tests and two production
workflow tests. The legacy C entry agrees bitwise with the saved pre-change
library on an unequal/equal-pair test with an odd SIMD remainder. Both APIs
and one/eight-thread results agree bitwise. The new 4h result agrees with
the globally densified natural-cubic control at the tested unequal pairs.

The selective all-project run was intentionally stopped when the user
chose global refinement. Its completed outputs are retained; exit143 is
not a numerical failure. The adopted global checks are recorded below;
do not treat the selective tests as their replacement.
No frozen references or scientific tolerances have been changed.

## Adopted global-default checks

The ordinary LSST YAML and shared initializer now reproduce the measured
global control exactly: all installed inputs and every entry of G, SSC,
cNG and total are bitwise equal. The 1,560-entry total is positive definite.
Construction took 55.91 seconds in this additional single run; initialization
including CAMB and cubic preparation took 0.516 seconds. The earlier
three-run timing table remains the controlled timing comparison.

Twenty targeted tests passed: four preparation tests, fourteen notebook
utility checks and two production/notebook workflow checks. The rebuilt
LSST library has the original pre-prototype SHA256, confirming that no
selective C change remains. All-project suites and executed notebooks
passed as recorded below; quiet cross-code timings are also complete.

The main cross-code refreshes completed: 32 OneCov stages, including
complete small Fourier and real-space G/SSC/cNG/total matrices, and 63
TJPCov stages covering Gaussian, SSC, halo moments and separated matter
trispectra. TJPCov native 4h maxima are now 1.1853%, 5.7215% and 7.1214%
at z=0.1, 0.5 and 1. The last occurs at K=Q=10 h/Mpc, where CCL 4h is
only 0.00604% of its summed trispectrum. Identical power and moments
still agree within 4.983e-6 fractionally. The natural-cubic control gives
0.4875% at 11,993 nodes and 0.3883% at 23,985. Native TJPCov projected
cNG/full totals have not yet been compared; do not infer those results
from the separated matter terms.

All 40 OneCov dependent controls also completed. The small complete
Fourier and real-space totals are positive definite. Their generalized
variance-ratio ranges, CoCoA relative to OneCov, are [0.954018, 1.006960]
and [0.984788, 1.023480], respectively. These retain each native model's
choices; the real-space comparison includes full-sky versus flat-sky
transforms. They are not full-survey or Fisher convergence certificates.

The input exporter preceded a source-text edit to the preparation helper.
Both source hashes remain in the comparison provenance. Applying the
committed helper to the archived native inputs reproduces every installed
power/grid array byte exactly; no source-hash equality is assumed.

Validated scripts and results are committed locally in OneCov-benchmark-
as 89190ed/87ba6ba and in tjcovbenchmark as f98e2ce/b03301b. The production
helper is core commit 1d4428d.

### All-project regression, 2026-10-07

Every data-vector and covariance test module passed, with no skips or
changes to frozen references or scientific tolerances:

| Project | Modules | Tests |
| --- | ---: | ---: |
| LSST Y1 | 36 | 185 |
| Roman real | 22 | 105 |
| Roman Fourier | 14 | 46 |
| Roman KL | 15 | 50 |
| DES Y3 | 17 | 64 |
| DES x Planck | 14 | 46 |
| DES cluster | 21 | 79 |
| Total | 139 | 575 |

The accepted XML records were checked against every expected module.
The first Roman real fresh-process cache check stopped because the
sandbox denied OpenMPI's local socket bind. Its unchanged retry passed
with local socket permission. The failed attempt is preserved separately;
the resumed runner reused 48 successful records with verified SHA256
hashes and ran only unfinished modules. No failed XML enters the totals.

The covariance-disabled LSST interface also reproduced its NLA and TATT
reference likelihoods exactly. This check uses the previously built
isolated library: global power refinement changes no C sources.

The first notebook launch stopped before executing any cell because
Jupyter did not forward the temporary kernel search path from its client
configuration. Setting that path on the actual kernel-spec manager passed
a startup/execute/shutdown smoke check. Only notebook execution resumed;
all suite outputs and the unsuccessful launcher attempt are preserved.

All seven covariance notebooks then completed: 57 code cells and 21
embedded figures. Every figure was visually inspected; executed file
hashes, source-cell retention and absence of error outputs were checked.
All seven computed totals passed their notebook positivity diagnostic
after the likelihood selection. The original likelihood covariances are
only comparison inputs; this is not a claim of reproducing their physics.
The outputs are committed in their respective project repositories.
Notebook execution times are not the controlled timing benchmark.

### Quiet cross-code timing completion, 2026-10-07

OneCov-benchmark- completed 36 sequential timing commands after every
regression and notebook job had finished. Eight OpenMP threads and one
BLAS thread were used; the launch check found no other numerical workers.
The published `results/global_power_20261006/timings.json` retains source
and installed-power fingerprints, per-process samples and numerical checks.

All four timed complete matrices exactly reproduce their accuracy archives:
G, SSC, cNG, total, coordinates and signals, including OneCov's native
real-space antisymmetry. Native settings were checked separately from
output-directory and timing metadata. Component outputs are bitwise equal
between the two fresh processes; common-input cross-code differences remain
below 4.21e-15 of the variance scale.

| Selected shear matrix | CoCoA construction | OneCov construction |
| --- | ---: | ---: |
| Fourier 100x100 | 20.9901 s | 71.5482 s |
| Real-space 16x16 | 45.7640 s | 491.4991 s |

These are single first-use measurements, excluding plotting and writing.
Setup is separate in the published tables. Native halo choices differ,
and CoCoA uses full-sky transforms while OneCov uses flat-sky transforms.
The supplied-trispectrum 8x8 cNG projection favors OneCov by 2.09x/3.13x
at 300/601 radial nodes; do not imply all individual kernels favor CoCoA.
The measurements do not establish native cNG or Fisher convergence and
do not time a full 1560-entry OneCov survey matrix.

## Reproduction record

External study: `test/covariance_reference/four_halo_power_study/`.
`run_full.py` runs the three full production cases in fresh processes;
`compare.py` checks every component and full generalized modes;
`diagnose.py` scans power refinement and measures the C preparation;
`summarize.py` checks bitwise repetition and aggregates the timing samples.
`run_regressions.sh` runs project modules sequentially and preserves logs.
Only the parent launches numerical/compiler work; timing jobs never overlap.

The adopted path is checked by
`test/covariance_reference/global_power_validation/full_default.py`. Its
`run_checks.py` runs all project modules sequentially without recompiling
unchanged C sources or refreezing; `run_notebooks.py` regenerates every
project example with the installed dense inputs.
