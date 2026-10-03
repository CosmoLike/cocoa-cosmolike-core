# Covariance rewrite: constraints, first implementation, and remaining gates

Owner request: 2026-10-03. Local source baseline: core `852296f`, after the
sigma_cb(M,a) implementation. The study lives outside git in
`test/cosmocov_port_study/` and the independent references in
`test/covariance_reference/`. This record keeps the constraints and current
status available in a fresh core checkout; it does not replace the study.

## Boundaries

- New C code only in `cosmolike/covariances/`, with filenames ending
  `_cov.c`. Existing data-vector C files are not modified for this rewrite.
  The later explicit request to retire the global SIMD switches is a
  separate maintenance ticket; it does not relax this port boundary.
- Covariance tables, grids, model choices and cache ownership are separate
  from the data vector. Reading public APIs is allowed. Private copies of
  internal helpers stay private to the covariance module when needed.
- OpenBLAS remains at one thread. Parallelism belongs to explicit OpenMP
  loops, with static scheduling and no sum split across workers.
- Default strict-IEEE and debug builds only. Aggressive/fast-math mode is
  retired. No new binary searches on uniform or piecewise-uniform grids.
- Use small guards for unsupported inputs. No speculative recovery system.
- Production SIMD paths are unconditional. Scalar comparison builds live
  only in the external test harness. Keep C lines within 80 columns and
  put each comparison or predicate on its own line, including assignments.
- After each tested major ticket, complete a separate didactic red-eye
  review for an advanced undergraduate reader before starting the next
  major ticket. This applies to a substantial component, not each helper.
- Commit locally; the owner pushes. Never push.

## What the study review changes

The September plan used `covariance/`, assumed the old sigma(M,1) halo
convention, and referred to three compiler modes and a concurrent cluster
port. Those descriptions are obsolete. The directory is `covariances/`;
halo statistics use sigma_cb(M,a) and rho_cb, concentration uses scale-
dependent cb growth, and only strict default/debug builds are supported.
The M200m radius and lensing mass weight still use total matter. Do not
bring back the retired total/cb halo switch or evolve all masses with a
single growth factor.

Review anchors: reports 01/09 for Gaussian, noise and Fourier conventions;
02/10/11/12 for cNG, SSC, derivations and the contract gates; 03/04/08 for
the node/halo/side-module architecture; 05/06/07 for project requirements,
existing covariance variants and literature. Their line numbers refer to
older source. Verify the actual code before copying a method.

The current source establishes these implementation patterns:

| Existing source | Covariance consequence |
|---|---|
| `cosmo2D.c::legendre_sums` | Group output rows to reuse kernel reads; keep each ell sum within one worker. |
| `cosmo2D.c::cfftlog_ells_p1/p2` | Plan serially, reuse transforms and plans, own thread scratch separately. Covariance pair coverage cannot use data-vector exclusions or per-bin early exits. |
| `cosmo3D.c::sigma2_fields_build` | Separate workspace geometry from cosmology values; group arrays with the same lifetime; warm lazy tables before parallel work. |
| `halo.c::nfw_fmadd4` and mass-node loops | Hoist row invariants, use explicit SIMDe, and preserve fused-operation rounding. Measure how vector values move through registers. |

## First validated slice

`cosmolike/covariances/gaussian_cov.c` contains stateless production
building blocks: integer-ell Wick covariance, projection by two supplied
operators, stable spherical annulus pair area, and analytic pure-noise
covariance. The caller owns and reuses scratch. There is no allocation,
FFTW planning, BLAS call, global cache, or spectrum computation here.

These are provisional numerical interfaces, not the frozen full module
contract. They exercise the Gaussian algebra without introducing a survey
driver or modifying any project covariance. They do not complete Phase 0,
Phase 1 or Phase 2 of the study.

Physics sources checked directly:

- [Krause & Eifler, CosmoLike](https://arxiv.org/html/1601.05779v1),
  Appendix A, Eq. 28 in v1: the two Gaussian Wick pairings and mode count.
- [Friedrich et al., DES Y3 covariance](https://arxiv.org/html/2012.08568v3),
  Section 4 and Sections 6.10.1–6.10.3: angular transforms, pair-count
  noise, effective dispersion per shear component, and ordered versus
  unordered galaxy pairs.
- [Takada & Hu, super-sample covariance](https://arxiv.org/html/1302.6994v3),
  corrected Eq. 44 and Appendix A: anchors for the later response and
  line-of-sight covariance, not implemented by the Gaussian primitives.

Noise is supplied in the observed field: N_g=1/n and
N_s=sigma_component^2/n, with n in sr^-1. No shear transfer factor is
applied to white noise. Signal inputs already carry their transfer
factors. Independent catalogs may overlap in redshift but cannot share
objects under the analytic-noise contract. General cross-catalog noise,
mask operators, integral constraints, and angular-bin overlap require an
explicit extension at the driver boundary.

For the same angular bin, define ordered-pair area
A_pair=Omega_s 2 pi (cos(theta_low)-cos(theta_high)). Then pure noise is
N_A N_B/A_pair times: the two catalog Kronecker pairings for w; twice
those pairings for xi+xi+ and xi-xi-; only the direct lens/source pairing
for gamma_t. Xi+xi- and other cross-probe pure-noise blocks vanish.
Measured or mask-derived pair areas can replace the uniform-survey area.

The real-space harmonic part computes CC+CN+NC directly, avoiding both a
finite-ell approximation to NN and catastrophic subtraction when N >> C.
The harmonic option retains NN, suitable for a later Fourier-band driver.
Signal E/B bookkeeping and the removal of low angular modes must be
resolved with the actual estimator before real-space survey integration.

## Measurements (Apple M2 Pro, 2026-10-03)

Clang 19.1.7, strict project-style flags, single OpenBLAS thread.
Projection workload: 20 left bins, 20 right bins, ell=0..50000, spherical
scalar kernels for 2.5–250 arcmin and a specified smooth test variance.
This is a supplied-spectrum kernel benchmark, not a full covariance time.
Three warm-ups excluded, 51 timed calls; allocation and Python setup excluded,
but the C weighting pass is included. Raw measurements are in external
`covariance_reference/results/final_timing.json`.

| OpenMP threads | Scalar median (ms) | Tiled SIMDe median (ms) |
|---:|---:|---:|
| 1 | 24.041 | 1.848 |
| 4 | 6.359 | 0.616 |
| 8 | 3.296 | 0.398 |

One-, two- and four-row tiles were compared. Four rows by four columns
won on this workload. Two 128-bit fused accumulators per row avoided
register moves seen in the experimental 256-bit split/join path. The
retained loop vectorizes across output columns, so it preserves scalar
ell order. Scalar/SIMDe and 1/4/8-thread benchmark outputs are bitwise
identical. This is not evidence for an optimal x86 tile; measure there.
Linux `perf` hardware counters were not available on this macOS host.

Seven independent checks cover production SIMDe in optimized and debug
builds, plus optimized/debug scalar comparisons compiled externally from
the pinned baseline `6f055d0` with an explicit `fma` sum. Debug checks use
UBSan/float-divide-by-zero instrumentation:
all field pairs and per-ell positive semidefiniteness; exact Fourier-band
mode counts; signed projection
operators and padded edge groups through ell=50000; repeated and 1/4/8
thread calls; ordered-pair/component factors; narrow annuli and unit
conversion against mpmath; noise-dominated signal against 70-digit
arithmetic; and the spherical white-noise completeness limit.
The last check uses broad rings and a 3e-4 tail tolerance; it does not
choose a production ell cutoff. NumPy longdouble equals float64 on this
host; high-precision claims come only from the mpmath checks.

## Didactic red-eye review of the Gaussian foundation

Completed a separate manual review after the seven primitive checks passed
in optimized and debug SIMD builds. This was a self-review, not a Fable
model review. Read the whole C/header pair, tests and module documentation
as one major ticket before beginning another implementation component.

- The Wick comments derive both surviving pairings and explain the
  integer-multipole mode count, observed-field noise, and NN separation.
- The projection comments define operator normalization, array ownership,
  worker ownership, vector-lane meanings and why no ell reduction changes
  order. Weighting and projection now have separate visual sections.
- The pair-count comments distinguish repeated catalog IDs from distinct
  catalogs, ordered from unordered pairs, and per-component shear noise.
  The two boolean assignments are explicitly mapped to Kronecker deltas.
- Removed the production scalar switch. Every C/header line is at most
  80 characters; guards and combined assignments have one predicate per
  line. The external scalar baseline remains available for checking.
- Remaining scientific limits are explicit: uniform-footprint pair area,
  independent catalogs, supplied operators/spectra, and no survey driver.

The final rebuilt project run passed **399 tests across all seven
projects**, including Roman's slow halo checks and the seven new LSST
primitive checks. Another 24 debug checks passed for the common and
cluster paths. No references were refrozen. Counts, logs and the separate
SIMD-retirement review are recorded in
[the SIMD retirement record](simd_retirement.md).
The final comments also make the shared ell grid and disjoint writable
rows explicit; vector names follow the existing `v2d`/`v` convention.

## Next gates, before a survey covariance can be used

The first survey-input ticket now has its own
[implementation, validation and didactic-review record](covariance_survey_inputs.md).
It supplies all-pairs Limber spectra and covariance-owned efficiencies;
the full non-Limber and Phase-0 accuracy gates below remain open.
The [SSC mask/shell-response ticket](covariance_ssc.md) adds independently
tested mask normalization and response projection, including a general
correlated-radial-kernel path through the existing SIMDe contraction.
Its supplied response can now come from the explicit halo prescriptions below.
The [cNG angular ticket](covariance_perturbation.md) now checks reduced
SIMDe tree averages against explicit Wick diagrams and high-precision
corner integrals.
The subsequent [halo-moment ticket](covariance_halo.md) and
[response/trispectrum assembly ticket](covariance_non_gaussian.md) now
implement and independently test those node-level ingredients. Full
survey integration and the Phase-0 physics/accuracy gates remain open.
The external [projection diagnostic](covariance_projection_measurements.md)
connects these ingredients on the pinned survey without replacing a project
covariance or claiming that band centers are exact band averages.

After these five tickets, all seven project suites passed: 405 tests with
21 external-library skips. Enabling every isolated covariance library
passed all 34 focused tests, including those 21, for **426 distinct passing
tests**. The 27 new focused tests also pass with debug instrumentation;
the isolated debug LSST interface passes ten existing example/non-Limber
regressions as well. No references were refrozen. Component timing and
accuracy measurements are recorded with each ticket, including failures
of the full survey refinement gate.

The following [angular-operator ticket](covariance_operators.md) adds six
optimized/debug checks, for 432 distinct passing tests including the earlier
project run. It builds unit-normalized full-sky spin-bin averages and exact
integer Fourier bands. This isolated library does not relink the projects;
their survey-operator integration and accuracy gates remain open.

1. Complete the pinned 2-lens/2-source configuration's convergence study.
   The overlapping samples, signed magnification, per-bin noise, CAMB dump
   and raw spherical-cap mask are available outside git. The preliminary
   projected diagnostic does not yet meet the full numerical contract.
2. Implement and validate covariance-owned all-pairs spectra: cross-bin
   gg, excluded gs, non-Limber gg and gamma_t, full magnification foreground,
   and consistent matched linear Limber subtraction. Keep every field pair.
3. Complete the Phase-0 independent SSC/cNG references, the CosmoCov oracle
   gate, units tests, realistic-node PSD checks and physics-delta metrics.
   Preserve separate Gaussian, SSC and cNG arrays; never clip negative
   eigenvalues or discard negative magnification products.
4. Before freezing SSC choices, distinguish the study's derived forms from
   published formulae: Takada & Hu's corrected Eq. 44 differentiates the
   two-halo spectrum, while the study's squeezed-tree derivation uses the
   linear-spectrum slope. Keep the derivation and measure that difference.
   The projected tree coefficients (17/7,1/2) are not a calibrated nonlinear
   tidal response. No inferred paper typo is a published erratum.
5. Freeze full headers only after the study's gates pass. Then wire the
   dataset inputs, row ordering, separate covariance outputs, writer and
   project generators. New covariance files coexist with existing ones
   until validated. Do not refreeze a likelihood to hide a failed check.

The full rewrite remains open in Cocoa's execution backlog. No claim of
delta-chi-squared < 0.2 or of survey covariance convergence follows from
the primitive tests or timing table.
