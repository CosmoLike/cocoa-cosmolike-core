# Analytic covariance rewrite

This directory separates covariance computation from the data-vector
engines. Every new C filename ends in `_cov.c`; existing core C files
outside this directory are not changed for the port.

The initial implementation in `gaussian_cov.c` provides four building
blocks for supplied spectra and kernels:

- `gaussian_wick_cov`: the two Gaussian four-point pairings at integer ell,
  with an explicit choice to include or separate pure noise.
- `gaussian_project_cov`: projection by two supplied linear operators,
  using caller-owned scratch, OpenMP output groups and SIMDe.
- `annulus_pair_area_cov`: stable spherical angular-pair geometry.
- `gaussian_noise_pair_cov`: analytic pure noise for xi±, gamma_t and w,
  with ellipticity dispersion defined per component.

The C function comments derive the equations and state dimensions, units,
array ownership, and threading rules. Production always uses the SIMDe
projection, including debug builds. Scalar comparisons are available only
in the external test harness, which compiles the pinned baseline from
core commit `6f055d0` with an explicit scalar `fma` sum. OpenBLAS is never
called here. Bitwise scalar/SIMDe agreement is checked on native FMA/NEON;
SIMDe targets that emulate FMA need their own rounding comparison.

This is **not yet a survey covariance generator**. All-pairs spectra,
production spin kernels, masks, SSC, connected non-Gaussian covariance,
dataset inputs, file output and project bindings remain to be integrated
and independently validated. The full module contract is not frozen.

The [rewrite record](../../.claude/skills/cosmolike-dev/references/covariance_rewrite.md)
contains the paper references, measured loop comparisons, and remaining
physics gates. The detailed external study is
`test/cosmocov_port_study/PLAN.md`.

## Local validation

The independent NumPy/mpmath reference lives outside git, at the owner's
requested `test/covariance_reference/`. Its `build_primitives.sh` compiles
these production C functions into an isolated library without relinking
any project. From `test/`, with the Cocoa conda environment active on macOS:

```bash
bash covariance_reference/build_primitives.sh scalar
bash covariance_reference/build_primitives.sh simd
bash covariance_reference/build_primitives.sh debug
bash covariance_reference/build_primitives.sh debug_simd
export OPENBLAS_NUM_THREADS=1
export COSMOLIKE_COVARIANCE_REFERENCE="$PWD/covariance_reference"
export COSMOLIKE_COVARIANCE_LIBRARY="$PWD/covariance_reference/results/gaussian_simd.dylib"
python cocoa/Cocoa/projects/lsst_y1/tests/test_covariance_primitives.py -v
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
