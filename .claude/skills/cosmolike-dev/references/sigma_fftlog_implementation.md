# Evolving variance and cb halo statistics: implementation record

2026-10-03. Owner-approved Phases 2 and 3 of `neutrino_growth_plan.md`.
Implementation and validation are complete for the supported builds.
Local scripts and raw logs: `test/neutrino_growth_study/implementation/`.
That directory is not tracked, so this record preserves the decisions
and measurements with the source. No GitHub push is authorized.

## Calculation and public interface

`cosmo3D.c` builds sigma_m^2(M,a), sigma_cb^2(M,a), and their logarithmic
mass slopes from the evolving linear spectra. It uses FFTLog bias 1.5,
the supplied uniform logarithmic k spacing, edge-power continuation to
1e-7 through 1e5 h/Mpc, a smooth taper over the upper quarter of Fourier
frequencies, and the analytic Mellin transform of the spherical top-hat
window squared. A second inverse transform differentiates that kernel.
Sixteen extra radial nodes at each mass boundary isolate the natural
spline's end condition. The final tables use the existing dense mass
and scale-factor grids (1024 by 256 at the default settings), storing
ln(sigma^2/a^2) to reduce early-time interpolation curvature.

The Python diagnostic is `sigma2(M, a=1, field=0)` (0: total; 1: cb).
Production C `sigma2(M,a)` always selects cb. `dlognudlogm(M,a)` reads
the evolving cb slope. `conc(M,a)` takes scale factor, not a growth
factor, and uses D_cb(M,a) = sigma_cb(M,a)/sigma_cb(M,1).
The halo-field initializer and YAML option have been retired.

All halo peak heights and mass-function densities use cb. The M200m
NFW radius still uses the total mean density defining that mass, and
lensing weights and nonlinear matter spectra retain total matter.
The Bhattacharya growth prescription is an adopted extension of a fit
without a massive-neutrino calibration. EMUL2 retains its pre-existing
approximation P_cb = P_lin/(1-f_nu)^2. CAMB supplies the actual cb table.
The external DES reference keeps its non-halo growth convention but now
integrates the evolving cb spectrum independently for halo statistics.

## Threading experiment

Hardware: Apple M2 Pro. Strict IEEE default build. Each timing warms the
FFTW plans first, then changes Omega_m between refills so the table is
rebuilt. Timers surround the C table construction only: no CAMB or input
copying is included. FFTW plans are created serially on allocation and
reused across cosmologies with separate, aligned arrays per OpenMP worker.
The Gamma-function factors depend only on the grid and are cached too.

The production layout completes a (field,a) row in one worker: one forward
FFT, two inverse FFTs, normalization, and splining. The alternative saves
all forward coefficients, splits inverse transforms with collapse(3) over
(field,a,derivative), then normalizes and splines after a barrier. It
performs the same number of FFTs but needs additional arrays and copies.

Mean milliseconds per refill, three repeats of 100 refills:

| Threads | Complete rows | Split inverse transforms |
|---:|---:|---:|
| 1 | 20.5542 | 20.7780 |
| 4 | 5.4216 | 5.5512 |
| 8 | 2.9569 | 3.1638 |

A longer run reversed the variant order to check warming/order effects:
three repeats of 1000 refills at eight threads gave 2.9993 ms for complete
rows (3.0100, 3.0255, 2.9623), versus 3.1809 ms for the staged layout
(3.1743, 3.1757, 3.1926). The staged layout was about 6% slower there.
Both layouts and one/eight-thread results were bit-identical across both
fields, two neutrino masses, four scale factors and 1024 mass nodes,
including the mass slopes. Keep complete rows: 2 x 256 = 512 independent
rows already provide 64 rows per worker at eight threads.

These are variance-table timings, not whole-likelihood speedups. Linux
`perf` hardware counters are unavailable on this macOS host.

## Documentation and numerical review

The comments derive smoothing, biasing, the Mellin integral, and its
mass derivative before discussing implementation. The spline helper
explains its tridiagonal equations and boundary conditions. The FFT-size
helper explains radix decomposition. The reader gives a fractional-node
example, the upper-edge bracket rule, table-index meanings, and the two
steps of bilinear interpolation. Like-shaped work arrays share a role
dimension; struct members have one declaration and explanation per line.
Cache allocation conditions follow cosmo2D.c directly in the `if`.
The owner’s simple-guards/no-speculative-recovery rule is in SKILL.md.

A manual second review covered the halo callers as well as the new
engine. The skill's named Fable 5 reviewer is not available in this
session; no Fable review is claimed.

Checks completed before refreezing:

- C against the independently implemented Python FFTLog study: maximum
  relative discrepancy 1.83e-13 in variance and 5.5e-13 in slope.
- Independent direct Simpson quadrature at five cluster masses, four
  scale factors and two neutrino densities: maximum variance discrepancy
  1.81e-5 (acceptance 1e-4).
- Roman halo invariants, including the optional slow integrals and
  spectra: 25 passed (31.69 seconds).
- Saved-reference chi2 drift measured before regeneration, with only
  the retired configuration key removed in memory:

| Project | Maximum absolute chi2 drift |
|---|---:|
| lsst_y1 | 0.033724 |
| roman_real | 0.076432 |
| roman_fourier | 0.003509 |
| roman_kl | 0.006486 |
| des_y3 | 0.052772 |
| desy1xplanck | 0.054075 |
| des_cluster | 0.151836 |

All are below the existing 0.2 threshold. Refreezes also remove retired
options and update the halo probe interface. Only the documented
reference generator writes frozen data and manifests; survey observations
and shipped synthetic vectors are not regenerated for this change.

## Unconditional direct indexing (owner request, same session)

The FFTLog redshift lookup now reuses `piecewise_index` and the existing
lnPL segment metadata. A before/after comparison at every one of the 256
scale-factor rows, 33 masses from 1e6 to 1e17 M_sun/h, both fields and both
saved neutrino masses was bit-identical for variance and slope (67,584
values). This check precedes the retirement of the build flag below.

At the owner's request, COSMO3D_ASSUME_PIECEWISE_UNIFORM is retired:
cosmo3D.c retains only the direct-index implementations. The segment
metadata, detector and setter calls are unconditional, including in debug
builds; all seven Makefiles remove the flag. The retained C/C++ tokens in
cosmo3D.c, generic_interface.cpp, basics.c, basics.h and structs.h were
checked against the former enabled branch and are unchanged. The existing
bucket lookup of a_chi is retained because its chi grid is not uniform.
No binary-search implementation remains in cosmo3D.c.

The full unmasked Roman 3x2pt vectors also agree bit for bit with the
pre-retirement library: NLA and TATT, 2115 entries each, at the fiducial
cosmology, a joint Omega_m/H0 detour, and the return to the fiducial point.
All six vectors include points excluded by the survey mask. A temporary
all-ones mask and unit covariance are installed before initialization;
no frozen or survey data is edited. The current side includes permanent
OpenBLAS pinning too. Raw result: full_vector_comparison.log.

The drift table measures all changes since each frozen snapshot, not the
isolated effect of Phases 2 and 3. In particular, the non-halo configurations
have HOD and halo IA disabled: their drift includes the previously committed
Phase 1 growth-k change to 0.05/Mpc. Their September frozen configurations
do not pin growth_k and therefore resolve that value from the live code.
The cluster configurations additionally change through the evolving cb halo
statistics. Reference-update commits must state both contributions.

## Independent cluster comparison

The independent Python reference self-tests passed (30 tests). With
matched settings, counts agree to 9.27e-6 relative, richness-bin number
densities to 1.39e-5, and richness-bin biases to 3.12e-6. The joint cluster
data-vector difference has delta chi2 = 7.512e-6 with the Y6 cuts and
3.031e-5 with the extended positive-definite mask. With nonzero NLA and
shear calibration, the corresponding cluster-lensing differences are
6.984e-6 and 2.815e-5. Production settings give joint differences of
5.037e-5 and 0.001865, respectively, below the 0.2 budget.

The stricter 1e-4 row-level target is not met everywhere: three of the
33 matched-setting rows (cluster-lensing C_ell, gamma_t and its Y-space
transform) miss it. These numerical limitations remain visible in the
comparison report; a passing data-vector budget does not imply that every
individual row passes. The historical unshifted source-tail diagnostic
intentionally compares different source-support conventions. The nuisance
check instead matches the core's existing photo-z-shifted support.

The full joint vector, its cosmology detour, and its return to the original
point are bit-identical at one and eight OpenMP threads. The separate
massless Lighthouse comparison still shows the documented coarse-CQUAD
precision differences (about 2.29% in counts and 0.759% in bias); it is
reported separately from the independent matched-settings oracle.

## Permanent OpenBLAS thread limit

The owner requires OpenBLAS to stay at one thread; parallelism belongs to
the explicit CosmoLike OpenMP loops. A shared runtime lookup calls
openblas_set_num_threads(1) at module import, initial setup, OpenMP team
configuration, and before both covariance inversion paths. There is no
restoration of a larger BLAS team. Other BLAS backends need not export this
entry point. Linux builds link libdl for the runtime lookup.

The ordinary covariance path also checks the inverse in correlation units:
R_ij = C_ij/(sigma_i sigma_j), (R^-1)_ij = (C^-1)_ij sigma_i sigma_j,
with max |R R^-1 - I| < 1e-8, before applying the final inverse mask.
The cluster path retains its corresponding check. No retry or alternative
factorization is added.

The controlled cluster check deliberately sets OpenBLAS to eight threads
before each of three inversions of the real survey covariance. Each leaves
OpenBLAS at one and the CosmoLike OpenMP team at four; inverses are
bit-identical and their correlation residual is 3.93e-13 or smaller.
Three fresh LSST model initializations with the same deliberately raised
BLAS limit likewise leave BLAS at one and OpenMP at four, produce
bit-identical inverses, and have residual 1.19e-13 or smaller.
The same controlled three-inversion check passes in all seven projects,
including Roman-Real, Roman-Fourier, Roman-KL, DES-Y3 and DESxPlanck:
OpenBLAS remains at one, the explicitly requested OpenMP team remains at
four, and repeated inverses agree bitwise. Calls requesting one, four
and eight OpenMP threads likewise preserve that team while pinning BLAS.

The aggressive build still fails its covariance residual check despite
the enforced BLAS limit. An isolated build of the pre-change core
(4adf735) and cluster interface (aa32535) reproduces this failure, with
OpenBLAS explicitly at one thread and OMP_NUM_THREADS=1: max |R R^-1 - I| = 4 during dummy-data
initialization, before any cosmology or variance calculation. This is a
pre-existing aggressive-mode limitation, not a passing validation; the
default build passes and the residual guard correctly stops the bad
aggressive result. Raw reproduction: baseline_aggressive_cluster_init.log.

The owner retired aggressive mode on 2026-10-03. All seven Makefiles now
reject its environment flag instead of compiling with fast-math. The
ordinary Roman covariance guard also catches an aggressive-build residual
of 83919.9135. Its selected tests had 34 passes and five failures (four
inverse guards and one 1.47e-12 halo rounding mismatch against a 1e-12
frozen tolerance). These failures are recorded, not counted as passing
validation. The supported configurations are the optimized strict-IEEE
default and debug. Installation guidance and the development skill list
the unsupported floating-point flags; no individual flag was isolated
as the cause. All seven retired-mode rejection checks passed.

## Final project validation

Every suite runs in a started cocoapy311 Cocoa environment, with
OMP_NUM_THREADS=4 and COCOA_HALO_SLOW=1. BLAS is held at one by the
compiled interface as well as the test shell. Command from Cocoa/:
`python -m pytest projects/PROJECT/tests -q`.

Completed on the final default build (including permanent BLAS pinning):

| Project | Result | Elapsed seconds |
|---|---:|---:|
| des_cluster | 29 passed | 577.39 |
| lsst_y1 | 57 passed | 891.94 |
| roman_real | 104 passed | 1367.35 |
| roman_fourier | 45 passed | 1093.93 |
| roman_kl | 49 passed | 1564.64 |
| des_y3 | 63 passed | 1191.31 |
| desy1xplanck | 45 passed | 948.81 |

All seven suites passed: 392 tests total. The independent Python
cluster reference suite adds 30 passing tests. Frozen references were
regenerated only with the documented project generators and are committed
separately from implementation changes.

The final debug builds also pass the cluster neutrino check (1 test,
45.24 seconds) and Roman halo plus NLA/TATT example checks (39 tests,
459.72 seconds), with the configured undefined-behavior and floating-point
division sanitizers enabled.
