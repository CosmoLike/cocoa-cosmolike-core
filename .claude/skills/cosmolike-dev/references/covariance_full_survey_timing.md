# Full LSST Y1 and Roman real covariance measurements

Historical Limber notebook baseline. For current production CLI timings
with Gaussian non-Limber, see [the seven-project measurement](covariance_cli_timing.md).

Measured 2026-10-03 on Apple M2 Pro, macOS 13.7.5, eight OpenMP workers.
Both loaded OpenBLAS libraries were pinned to one thread, including the
OpenMP-built library. The runtime inventory confirmed an eight-thread
OpenMP team. LSST and Roman ran sequentially with no overlapping numerical
jobs, builds or regression tests. Desktop/window processes remained active.
These are one cold complete computation per survey, not repeated means or
an x86 scaling claim. Regression jobs began only after both timings ended.

## Model and layout

Both matrices contain Gaussian, SSC and all five connected halo terms.
The model has zero neutrino mass, linear galaxy bias, zero IA, magnification
and RSD, and Limber spectra. A spherical cap models each footprint. Gaussian
signal/mixed noise uses the fsky approximation; pure pair noise uses the
cap's angular pair area. This is not an exact cut-sky covariance.
SSC uses the isotropic fractional halo response transferred to Pdelta,
including galaxy survey-mean subtraction. Long-mode SSC also uses Limber.
There is no calibrated nonlinear tidal response or connected galaxy
shot-noise trispectrum. These are explicit supported forecasts, not
reproductions or replacements of the supplied likelihood covariances.

- LSST Y1: five lens and five source bins, 26 angular bins from 2.5 to 900
  arcmin, 60 observable rows and a 1560 by 1560 matrix. Area 12300 deg2,
  lens densities 3.6/bin and source densities 2/bin per arcmin2, shape
  dispersion 0.26/component. The selected project mask retains 959 rows.
- Roman real: eight lens and eight source bins, 15 angular bins from 2.5
  to 250 arcmin, 141 observable rows and a 2115 by 2115 matrix. Excluded
  measured gammat pairs are (6,0),(7,0),(7,1), zero based. Area 2415 deg2,
  lens and source density 41.3/8 per bin per arcmin2, dispersion
  0.30/component. The example1 mask retains 1950 rows.

All internal field pairs and cross-lens non-Gaussian blocks are retained.
Rows follow xi+, xi-, gammat, w; theta is the innermost index. Source pairs
are upper triangular and lens-major ordering is used for gammat. The
projected first/second lens NG block contains 676/676 nonzero entries for
LSST and 225/225 for Roman.

The LSST cosmology is Omega_m=0.3, Omega_b=0.05, H0=70, ns=0.965,
As=2.1e-9, w=-1. Roman uses Omega_b=0.04, H0=67.32, ns=0.96605 with the
same Omega_m, As and w. Both use Takahashi Halofit and the project's
shipped n(z) and photo-z interpolation convention. Resolved inputs and
CAMB tables were saved with each matrix.

## Resolution and measured times

Signal ell_max=100000; mask ell_max=32768; 128 NG samples uniform in
ln(ell+1/2); four scale-factor panels with 128 GL nodes each; 16385 lensing
window samples; 512 GL nodes per angular bin; eight log-mass panels from
1e6 to 1e17 Msun/h with 512 GL nodes each; 20 relative-angle panels with
256 GL nodes each. The response finite difference uses delta ln(k)=5e-5.
Scale-factor panels cover 1/4.5 to 1-1e-6 for LSST and 0.2 to 1-1e-6 for
Roman. These are explicit timing settings, overriding the small notebook
configuration; they are not an accuracy_boost certification.

| Stage (seconds) | LSST Y1 | Roman real |
|---|---:|---:|
| All-pairs Limber spectra | 2.232508 | 3.962908 |
| Angular and mask geometry | 0.905128 | 0.550892 |
| Complete Gaussian block assembly | 25.295259 | 75.022428 |
| Observable signal projection | 0.042847 | 0.093093 |
| Shared halo response and trispectrum | 133.729024 | 134.397532 |
| SSC and connected radial projection | 0.721956 | 0.359169 |
| **Complete covariance** | **162.929569** | **214.391137** |
| CAMB/initialization and input saving | 0.412998 | 0.426187 |
| Matrix output saving | 0.021259 | 0.044348 |
| Full and selected eigenvalue checks | 0.251977 | 0.715572 |
| **Overall elapsed** | **163.657858** | **215.622518** |

Full covariance time includes allocations, Python/C++ boundary validation,
all integer-multipole projections and cold core table construction. Setup,
I/O and eigenproblems are separated. Overall time also includes small
configuration/manifest overhead, so its total is not exactly the displayed
stage sum. The earlier Gaussian kernel extrapolation omitted substantial
block assembly and boundary overhead; these complete measurements include it.

The assembler hoists angle geometry in multipole units, batches power
reads across independent OpenMP rows and shares matter calculations across
all catalog pairs. Signed linear interpolation of the coarse NG multipole
table is projected algebraically through common operators. It does not
sparsely sample oscillating real-space kernels. The measured fixed-physical-k
coarse/cubic-dense/linear radial lookup experiment is not integrated here:
each of the 512 shells still evaluates its own matter calculation.

## Positivity and validation

No clipping, jitter or diagonal regularization was applied.

| Correlation-matrix minimum eigenvalue | Full | Selected |
|---|---:|---:|
| LSST Y1 | 2.2685418339063421e-4 | 5.959828783994409e-3 |
| Roman real | 5.910644347210254e-4 | 1.1334723601599168e-3 |

Both totals are finite, exactly symmetric and have zero negative modes.
This verifies these full supported-model realizations, not positivity for
all physical choices and cosmologies. Fisher/FoM convergence, independent
physical model comparisons and more stringent integration refinement remain
open. The high-ell angular quadrature and coarse NG table also need that
refinement; the timing profile is not designated a numerical source of truth.

Independent small NumPy checks compare the compressed angular transform
with an explicitly interpolated dense signed matrix and the complete
connected projection with a direct tensor sum, including all cross-lens
entries. Default power-row reads match serial interpolation bitwise for
linear/nonlinear power at 1/2/4/8 threads. The isolated O0 sanitizer build
passes the same uneven-row check. Full low-resolution smoke matrices,
with identical saved CAMB tables, agree bitwise at one/eight threads in G,
SSC, cNG, total, signal and radial geometry for both surveys. Smoke timing
is not used as performance evidence.

The didactic self-review covered array axes/units, the interpolation/projection
identity, the common matter calculation, the survey-mean subtraction,
all-cross-pair assembly and the independent-row OpenMP rationale. This is
a manual self-review, not a Fable review. New covariance C remains inside
covariances/ and respects 80 columns; no new SIMD arithmetic was added.

## Local measurement artifacts

The external reference directory contains `run_full_survey.py`, with
`--project lsst_y1|roman_real --profile refined --output <new-directory>`.
Run in Cocoa's activated environment with the core package, project
interface and CAMB checkout on PYTHONPATH. OPENBLAS_NUM_THREADS=1,
OMP_NUM_THREADS=8 and OMP_PROC_BIND=disabled were set; the harness also
pins BLAS explicitly. Each output directory contains report.json,
settings.json, covariance.npz and its CAMB inputs. Raw measured directories:

- `test/covariance_reference/results/full_lsst_refined_8threads/`
- `test/covariance_reference/results/full_roman_refined_8threads/`
- `test/covariance_reference/results/full_covariance_verification.json`

These local harness paths are developer evidence, not public README
instructions. Production assembly is in the tracked shared survey.py;
its independent algebra and reader tests are in LSST's covariance suite.

After the measured runs, the standard project regression commands completed:

```text
python -m pytest projects/lsst_y1/tests/data_vector projects/lsst_y1/tests/covariance -q
  exit 0: 114 passed, 15 warnings
python -m pytest projects/roman_real/tests -q
  exit 0: 80 passed, 24 skipped
```

Roman's 24 opt-in slow halo/HOD/halo-IA tests were not enabled
(`COCOA_HALO_SLOW` was unset). No frozen reference was changed. These two
regression jobs overlapped each other after all reported benchmarks ended;
their elapsed times are not performance measurements. Their local logs are
`/tmp/full-covariance-lsst-regressions.log` and
`/tmp/full-covariance-roman-regressions.log`.
The isolated debug check is `check_power_rows_debug.py` against the
`build_roman_review.sh candidate debug` build; both returned zero.
Full supported optimized interfaces were rebuilt for LSST and Roman.
Other project suites were not repeated for this covariance-only addition.
