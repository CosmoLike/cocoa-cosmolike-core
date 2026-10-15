# Covariance architecture, physical conventions and validation

Covariance construction combines supplied survey inputs, spectra and halo
responses with estimator-specific projection operators. The implementation
separates these layers so that their physical assumptions, numerical errors
and computational costs can be checked independently.

## Module boundaries

- Covariance C routines live in `cosmolike/covariances/`, with filenames
  ending `_cov.c`. Keep covariance-specific calculations out of data-vector
  C files. Changes to shared primitives require their own regression checks.
- Covariance grids, model choices and ownership are separate from the data
  vector. Public shared readers can be reused, but data-vector pair cuts and
  cache lifetimes do not define covariance behavior.
- OpenBLAS uses one thread. Explicit OpenMP loops own parallelism, with
  static scheduling and complete sums assigned to individual workers.
- Production SIMD is unconditional. Independent scalar comparisons belong
  in external tests. Use strict-IEEE or debug builds; fast-math is unsupported.
- Uniform or piecewise-uniform grids use direct indexing. Unsupported inputs
  receive explicit guards rather than speculative recovery paths.
- Comments explain equations, units, memory ownership and numerical choices.
  Review each substantial component as a coherent calculation, including its
  header and tests. C lines stay within 80 columns, with separate predicates
  and readable control-flow blocks.
- Commit locally; publishing remains a separate action. Never push.

Halo statistics use sigma_cb(M,a) and rho_cb, with scale-dependent cb growth
in concentration. The M200m radius and lensing mass weight use total matter.
A total/cb selector or one growth factor for every halo mass would change
this physical prescription. The neutrino-free limit does not justify that
simplification for massive-neutrino cosmologies.

## Numerical design

| Existing source | Covariance consequence |
|---|---|
| `cosmo2D.c::legendre_sums` | Group output rows to reuse kernel reads; keep each ell sum within one worker. |
| `cosmo2D.c::cfftlog_ells_p1/p2` | Plan serially, reuse transforms and plans, own thread scratch separately. Covariance pair coverage cannot use data-vector exclusions or per-bin early exits. |
| `cosmo3D.c::sigma2_fields_build` | Separate workspace geometry from cosmology values; group arrays with the same lifetime; warm lazy tables before parallel work. |
| `halo.c::nfw_fmadd4` and mass-node loops | Hoist row invariants, use explicit SIMDe, and preserve fused-operation rounding. Measure how vector values move through registers. |

These patterns explain why workspaces and physical values have separate
lifetimes, and why all-pairs spectra must be prepared before projection.
They are design constraints, not a claim that one tile size or loop order
is fastest on every architecture.

## Gaussian foundation and estimator conventions

`cosmolike/covariances/gaussian_cov.c` contains stateless production
building blocks: integer-ell Wick covariance, projection by two supplied
operators, stable spherical annulus pair area, and analytic pure-noise
covariance. The caller owns and reuses scratch. There is no allocation,
FFTW planning, BLAS call, global cache, or spectrum computation here.

These supplied-input primitives isolate the Gaussian algebra from survey
assembly. Their validation must be distinguished from the convergence of
the complete covariance.

Physics references:

- [Krause & Eifler, CosmoLike](https://arxiv.org/html/1601.05779v1),
  Appendix A, Eq. 28 in v1: the two Gaussian Wick pairings and mode count.
- [Friedrich et al., DES Y3 covariance](https://arxiv.org/html/2012.08568v3),
  Section 4 and Sections 6.10.1–6.10.3: angular transforms, pair-count
  noise, effective dispersion per shear component, and ordered versus
  unordered galaxy pairs.
- [Takada & Hu, super-sample covariance](https://arxiv.org/html/1302.6994v3),
  corrected Eq. 44 and Appendix A: anchors for response and
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
The harmonic option retains NN, suitable for Fourier-band projection.
Signal E/B bookkeeping and the removal of low angular modes must be
resolved with the actual estimator before real-space survey integration.

## Component organization

| Layer | Responsibility and reference |
|---|---|
| Survey inputs | [Radial inputs and all-pairs spectra](covariance_survey_inputs.md): covariance-owned efficiencies and complete field-pair coverage. |
| Gaussian non-Limber and IA | [Hybrid spectra and alignment](covariance_nonlimber_ia_plan.md): matched Limber subtraction, density/RSD kernels and E/B bookkeeping. |
| Angular and mask operators | [Bin operators](covariance_operators.md) and [mask geometry](covariance_mask.md): estimator normalization, exact integer bands and pair areas. |
| SSC | [Mask and shell responses](covariance_ssc.md), with explicit long-mode and local/global-mean conventions. |
| cNG | [Perturbation averages](covariance_perturbation.md), [halo moments](covariance_halo.md) and [response/trispectrum assembly](covariance_non_gaussian.md). |
| Project interfaces | [Production and exploration interfaces](covariance_interface_rollout.md), with shared C calculations and separate output ownership. |
| Cluster forecasts | [Joint covariance model](covariance_cluster_joint.md), including its selected-halo and count-cross approximations. |

These references own the implemented capabilities and limitations of their
components. A Gaussian non-Limber or IA calculation does not imply the same
extension of SSC/cNG. A joint cluster matrix is not automatically a complete
selected/discrete-halo covariance.

## Survey-level validation

A correct primitive is necessary but insufficient for a trustworthy survey
covariance. Validation must cover the following distinct questions:

1. Do inputs, units, row ordering, noise and physical conventions match the
   independent comparison? Keep Gaussian, SSC and cNG arrays separate.
2. Are all required field pairs present, including cross-bin gg and gs pairs
   excluded by data-vector scale or overlap cuts? Match the linear Limber
   term subtracted from the non-Limber density calculation.
3. Are radial, angular, mask and mass grids independently resolved? A
   spherical-cap pilot showed a 1.05e-5 pair-area error from its mask cutoff
   and SSC-shell changes up to 1.30e-6 after mask refinement. Geometry
   resolution must therefore be tested alongside the covariance integrals.
4. Is the total matrix finite and positive on the intended estimator space,
   and are parameter errors and Figure of Merit stable under refinement?
   Use the [covariance accuracy protocol](covariance_accuracy.md). Do not
   transfer the data-vector Δχ² threshold directly to covariance convergence.
5. Are physical approximations identified separately from numerical error?
   Takada & Hu's corrected Eq. 44 differentiates the two-halo spectrum;
   the squeezed-tree alternative uses the linear-spectrum slope. Their
   difference must be measured. The tree coefficients (17/7, 1/2) are not
   a calibrated nonlinear tidal response, and an inferred paper typo is
   not a published erratum.

Negative magnification products are physical signed contributions. Never
clip them, repair eigenvalues or refreeze a likelihood to hide a failed
comparison. Cross-code agreement and regression tests establish different
facts from convergence or physical calibration.

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

## Reproducible Gaussian kernel benchmark

The independent NumPy/mpmath reference is external to the core repository,
in `test/covariance_reference/`. Its `build_primitives.sh` creates isolated
libraries without relinking a project. The following commands require that
checkout and the Cocoa conda environment on macOS; a fresh core clone alone
does not provide these artifacts.

```bash
bash covariance_reference/build_primitives.sh scalar
bash covariance_reference/build_primitives.sh simd
bash covariance_reference/build_primitives.sh debug
bash covariance_reference/build_primitives.sh debug_simd
export OPENBLAS_NUM_THREADS=1
export COSMOLIKE_COVARIANCE_REFERENCE="$PWD/covariance_reference"
export COSMOLIKE_COVARIANCE_LIBRARY="$PWD/covariance_reference/results/gaussian_simd.dylib"
python cocoa/Cocoa/projects/lsst_y1/tests/covariance/test_covariance_primitives.py -v
```

Run the same test with each library. The project's pytest suite discovers
it too, and reports a clear skip when the external paths are not set.
For the kernel benchmark, with no other CPU-heavy jobs running:

```bash
OMP_PROC_BIND=disabled python covariance_reference/benchmark_projection.py \
  covariance_reference/results/gaussian_scalar.dylib \
  covariance_reference/results/gaussian_simd.dylib \
  --repeats 51 --output covariance_reference/results/final_timing.json
```

The benchmark includes the C weighting pass and excludes input generation
and allocation. It measures a supplied-spectrum projection, not CAMB,
spectrum generation, or a full covariance. It does not establish a
production accuracy setting.

The scalar comparison compiles the pinned core baseline `6f055d0` with
an explicit scalar `fma` sum; the production projection always uses SIMDe.
Bitwise agreement was checked on native FMA/NEON. Targets that emulate
FMA require their own rounding comparison. This pinned workflow
covers the Gaussian primitives, not the full covariance assembly.
