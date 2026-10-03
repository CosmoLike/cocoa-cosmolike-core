# SIMD switches retired (owner request, 2026-10-03)

Production compiles the existing SIMDe paths in optimized and debug builds.
`COSMO2D_NOT_USE_SIMD`, the halo-specific `HALO_NOT_USE_SIMD`, and the new
covariance scalar switch are retired. Do not add a replacement opt-out.
Scalar single-point functions and incomplete-vector tails are still part
of the algorithms; they are not optional whole-library scalar modes.

This is a separate maintenance request made after the Gaussian foundation
was tested and didactically reviewed. It explicitly authorizes the switch
retirement outside `cosmolike/covariances/`; the covariance port itself
still must not modify the existing data-vector C files.

## What changed

- Removed guarded scalar alternatives from the angular, scale-cut, halo,
  cluster, basic-vector and unfinished halo source files. Retained SIMD
  branches have exactly the same C tokens as before the deletion.
- Made SIMDe headers/types unconditional and removed the global-to-halo
  scalar-switch forwarding. Native architecture choices within SIMDe and
  the existing fused-operation helpers are preserved.
- All seven project Makefiles include SIMDe unconditionally. Debug builds
  keep their existing `-O0` and sanitizer settings; they now exercise the
  SIMD implementations used by the optimized builds.
- Updated comments and this skill so they no longer prescribe scalar
  preprocessor fallbacks. Existing algorithm-specific controls unrelated
  to the retired SIMD switches are unchanged.

## Validation and final didactic review

- Isolated debug builds compiled successfully for LSST Y1 and DES cluster,
  covering the common core and cluster extensions with native SIMDe at
  `-O0`. Their NLA/TATT and non-Limber regressions passed: 10 LSST checks
  and 14 cluster checks, with no sanitizer errors or refreezing.
- Every newly added C/header line fits within 80 columns. The Gaussian
  C/header pair also satisfies that limit throughout.
- Seven project Makefiles have one unconditional SIMDe include path and
  no scalar opt-out. Their whitespace checks pass.

All seven optimized interfaces were rebuilt from the final source, then
all project tests passed without refreezing (399 total):

| Project | Passed | Test runtime (s) |
|---|---:|---:|
| roman_real | 104 | 1285.27 |
| roman_kl | 49 | 1513.83 |
| roman_fourier | 45 | 1029.51 |
| des_y3 | 63 | 1090.45 |
| lsst_y1 | 64 | 824.30 |
| desy1xplanck | 45 | 897.60 |
| des_cluster | 29 | 517.33 |

Roman's optional slow halo tests were enabled. The LSST count includes
seven Gaussian primitive tests. These are test-suite runtimes, including
CAMB and repeated models, not likelihood or covariance benchmarks.
The separate 24 debug checks covered the common and cluster C paths;
the installed project libraries remain optimized.

Completed a separate manual didactic red-eye review after the tests.
Checked the removed branches, retained single-point/tail calculations,
thread ownership, unchanged summation order, Makefiles and public comments.
Replaced stale references to deleted branches with the equations being
computed. Updated the skill's conflicting fallback instructions. The
Gaussian follow-up also uses house `v2d`/`v` names and explicitly states
that operators share one ell grid and writable rows must be disjoint.
These final clarifications change only comments; the retained-token check
still passes. This was a self-review, not a Fable model review.

## External checks

The owner keeps studies and independent tests outside git, under `test/`.
`simd_reference/check_retirement.py` compares the 11 existing C/header
files affected by global-switch removal with core commit `6bc8cb7`,
selecting that revision's SIMD branches using
`unifdef -UCOSMO2D_NOT_USE_SIMD -UHALO_NOT_USE_SIMD`. It then compares every
retained token, ignoring comments and whitespace. All 11 changed files
match. The Gaussian folder has its separate primitive tests. This checks
branch deletion, not numerical convergence; the unfinished halo sources
remain uncompiled and are not being reactivated by this change.

`simd_reference/run_validation.py` builds with `make -j1`, then runs the
project tests against the newly built library. Default mode runs every
suite and enables Roman's slow halo checks. Debug mode builds externally
and runs NLA/TATT examples plus non-Limber gg/gamma_t; DES cluster also
runs its additional combined-probe examples. Each subprocess verifies
which shared library it imports. Results and logs are saved externally.

For an optional scalar data-vector experiment, use an isolated copy of
core `6bc8cb7` with both C and C++ compilers defining the historical
`COSMO2D_NOT_USE_SIMD`. That definition is meaningful only in the old
source. Keep the same strict optimization flags, survey inputs and model
settings in both builds; compare full unmasked vectors at several points,
not chi-squared alone. The Gaussian scalar benchmark is already automated
by `covariance_reference/build_primitives.sh`, using core `6f055d0` and an
explicit scalar `fma` sum. No production switch is needed for either test.
