---
name: cosmolike-dev
description: >-
  Development, optimization, review and debugging of CosmoLike/Cocoa C and
  Python. Use for C patches, hot loops, OpenMP/SIMD, GSL, FFTW/FAST-PT,
  nondeterministic chi2, perf benchmarks and performance claims. Applies to
  Limber and non-Limber calculations, TATT, NLA, 3x2pt, Legendre sums,
  tomographic C_ell, halo models and HOD, even without optimization work.
  Also covers cosmolike_notebook_utils (data-vector and covariance plotting,
  CAMB and Fisher helpers), cocoa_testing.py, and each project's likelihoods,
  notebook wrappers, notebooks, tests and scripts.
---

# CosmoLike Development

This skill encodes how the CosmoLike C core was optimized 6x (v4.07: 670.6s →
~111s per 1000 likelihood evaluations, 92% fewer instructions) without a single
physics regression, between late 2025 and mid 2026. Follow the same discipline:
the speedups came from measurement and algorithmic restructuring, validated at
every step — not from cleverness applied blindly.

Before writing any optimization, read `references/patterns.md`.

Before reviewing any patch, read `references/pitfalls.md` — it is the catalog
of bugs this codebase has actually had, and doubles as the review checklist.

Before writing or reviewing any Python — a plotting function, a notebook
wrapper, a notebook cell, a likelihood method, a test, a script — read
`references/python.md`. It is the style contract for Python in these
repositories and holds the conventions of the data-vector plotting
functions.

Before changing or explaining the growth factor (growfac, f_growth, the
G_growth table the likelihoods build), the IA amplitudes (1/D, 1/D^2), the
one-loop D^4 factors or sigma(M, z) = D sigma(M, 0), read
`references/fable_review_growth_factor.md` (Fable 5, 2026-10-02): where
each D comes from physically, every consumer with file:line, and why the
table's sampling k (5e-4/Mpc, a horizon scale) matters at w != -1.
The measurements behind it (dark-energy perturbations on/off, the k
scan, neutrino mass dependence vs Eisenstein & Hu 1999) are in
`references/growth_factor_measurements.md`.
Before changing sigma^2(M), the halo-model consumers (mass function,
bias, concentration, HOD, cluster counts) or anything
that chooses P_cb vs P_mm, read `references/fable_review_neutrino_halos.md`
(Fable 5, literature verified with arXiv section/figure): which field each
consumer needs with massive neutrinos, and the state of the halo.c fits.
Which growth factor the D^1.15 of the Bhattacharya concentration takes with
neutrinos (no paper says; recommended: the cb growth at halo scales in both
the prefactor and nu): `references/fable_review_concentration_growth.md`.

**Neutrino halo model.**
Plan and decisions: `references/neutrino_growth_plan.md`. Implementation
and measured checks: `references/sigma_fftlog_implementation.md`.
The C FFTLog tables expose both matter and cb variances and mass slopes
at (M,a). Halo statistics always use cb; `halo_matter_field` is retired.
Concentration uses D_cb(M,a) = sigma_cb(M,a)/sigma_cb(M,1). The M200m
radius and lensing mass weight retain total matter. Serial FFTW planning,
reused plans, and complete (field,a) rows follow the non-Limber design;
a split collapse(3) experiment was slower at 1, 4 and 8 threads.
The independent DES reference retains its external non-halo growth
convention; changing that convention and the future CosmoCov halo
implementation remain separate decisions. p_mm/p_my/p_yy remain outside
the compiled code. The cfastpt frequency-window comments already describe
the tapered top fraction correctly.

**Covariance rewrite.** Before covariance work,
read `references/covariance_rewrite.md` and the external study at
`test/cosmocov_port_study/PLAN.md`. The implementation lives in the plural
`cosmolike/covariances/` directory. Do not modify any existing C file
outside that directory for the port. Every new covariance C filename ends
in `_cov.c`, including future cluster extensions (`*_cluster_cov.c`).
Data-vector and covariance numerical choices remain separately owned.
Covariance generation is an offline calculation and never runs inside
MCMC. Reuse explicit shared inputs across its blocks; do not add persistent
cosmology caches just to imitate the data-vector lifetime. Shared core
readers can still have caches and require serial initialization before
parallel reads. Explain this distinction in `cosmolike/README.md`.
The non-Limber and NLA follow-up is detailed in
`references/covariance_nonlimber_ia_plan.md`. It is planning only:
implementation is deferred until requested. The data-vector non-Limber
path and low-level covariance NLA windows already exist, but the full
forecast uses Limber and zero IA. Do not describe either follow-up as
complete or switch defaults without its independent checks. Non-Limber
Gaussian spectra do not remove the separate SSC/cNG approximations.
Study the actual `cosmo2D.c`, `cosmo3D.c`, and `halo.c` implementations;
some older study and pattern descriptions predate their current behavior.
Carry over serial FFTW planning/reuse, precomputed node tables, direct grid
indexing, caller-owned grouped scratch, and deterministic OpenMP loops.
Explicitly evaluate SIMDe vectorization: do not assume an OpenMP SIMD
pragma vectorizes strict floating-point arithmetic. Measure loop layouts,
thread counts, precision, and generated instructions before choosing them.
Covariance production code always uses SIMDe for its bulk arithmetic.
Keep scalar comparisons in the external test harness, with no covariance
preprocessor fallback. The same rule now applies to existing data-vector
SIMDe paths; see `references/simd_retirement.md`.
Krause and Takada papers are primary physics sources; CosmoCov code and the
study's inferred corrections are comparison targets, not a physics oracle.
Keep the implementation simple, with short guards for unsupported cases;
do not build elaborate recovery paths. Never push; local commits are allowed.
Cluster count responses and all-pairs Limber projection are recorded in
`references/covariance_cluster_counts.md` and
`references/covariance_cluster_spectra.md`. They are validated components,
not a completed cluster 6x2pt+N covariance. Retain the distinct count and
two-point response conventions and test all internal field cross spectra.
Selected mass moments and their exclusive-category convention are in
`references/covariance_cluster_moments.md`; do not replace one membership
probability by its powers when several legs belong to the same halo.
Joint cluster-lensing localization and its deterministic zero rows are
recorded in `references/covariance_cluster_localization.md`. Transform
every cross block before applying the likelihood's scale selection.
The joint angular notebook forecast and its explicit approximations are
recorded in `references/covariance_cluster_joint.md`. Its connected term
uses linearly biased matter tracers and its count crosses contain SSC
only. It is not a complete selected/discrete-halo covariance. Archive
those omissions with the matrix and retain the defined Y null-row map.

**Optional covariance build.** Each project's
`IGNORE_COSMOLIKE_<PROJECT>_COVARIANCE` installation key defaults to 1.
Its Makefile omits `_cov` sources and objects and defines
`COSMOLIKE_NO_COVARIANCE` so the project interface omits covariance bindings.
Keep ordinary data-vector evaluation and supplied-covariance inversion usable
in that build. Document activation/recompilation in each project README;
test both build modes and do not add covariance dependencies to data-vector
C files. The module's `has_covariance` attribute reports the compiled mode.

**Production interfaces and notebook wrappers are separate layers.**
`_interface` serves CLI/production runs; covariance production bindings
borrow contiguous NumPy arrays without Armadillo/CARMA conversions.
`_wrapper` serves Jupyter exploration, exposing intermediate quantities
through readable Armadillo types. Wrappers perform validation, allocation
and layout conversion only. Heavy integration, table construction, SIMD,
OpenMP scheduling and matrix assembly belong in shared covariance C
routines, called by both layers. Never duplicate a physical calculation
or maintain a second optimized implementation in either C++ layer.
Keep the shared Python survey workflow common too; select its numerical
backend explicitly. Test agreement of both paths, array ownership and
one/eight-thread determinism. Explain this division clearly in human
READMEs, with separate production and notebook subsections.

**Notebook C++ wrappers.** Follow `halo_wrapper_cluster.cpp` and
`cosmo2D_wrapper.cpp`: numeric inputs, results and working arrays use
`arma::Col`, `arma::Mat` and `arma::Cube`, with named axes and units.
Keep Python conversion in the binding files. Copy inputs without mutation;
CARMA exports the Armadillo results at the return boundary. Do not
use `py::array_t` aliases, `std::vector` numeric containers, or vectors of
row/plane pointers in notebook wrappers. If a batched C routine requires
C storage, use a short, explicit copy into/out of its ordinary workspace,
as in the data-vector wrappers; do not invent pointer-adapter machinery.
Keep callable C++ declarations in the wrapper header. Dictionaries and
tuples may group independently meaningful Armadillo results; keep
conversion at the return/binding boundary. Prefer named quantities over
packing a fourth numerical axis into a Python-specific container.
These wrappers support readable Jupyter experimentation. Copies and modest
wrapper overhead are acceptable. Production optimization belongs in the
normal interface and C kernels; never obscure the notebook API to avoid
an array copy. Preserve axis order, ownership and input immutability, and
test C-order, Fortran-order and sliced NumPy inputs through CARMA.

**Integration validation limits.** The maximum covariance integration
level to test is 4. On the M2 Pro laptop, Roman tests stop at level 3;
reserve Roman level 4 for a server. Do not escalate above these limits
when assessing default settings. Keep completed matrices and compare
against the highest completed permitted level, stating which reference
was actually used.

**Public documentation.** READMEs are
for human readers, including advanced undergraduate physics students.
Explain the physics, define symbols and approximations, and describe each
source file and how its calculation fits into the module. Cite the papers
directly. Never send readers to Claude/bot skills or untracked study
directories for an explanation. Port the necessary physics into the
README itself. Keep machine-local library paths, build commands, internal
benchmark recipes and development history in the skill references.
Document test suites in public documentation when they are reproducible
from the repository; do not present a developer's external harness as a
public test interface. State implemented capabilities and remaining limits
without turning the README into a development log.

Write skill instructions as impersonal project guidance for all
contributors. State the requirement and its rationale directly, without
personal names, quotations or conversational attributions. Retain
scientific citations and dates that identify measurements or sources.

**Survey accuracy and Roman.** The
old study's 1e-6 per-entry reference-refinement target is not a universal
production covariance requirement. Do not transfer the data-vector
|delta chi2| < 0.2 rule to covariance convergence. Assess numerical
refinement through marginalized Figure of Merit and parameter errors,
with positive-definiteness checks and relative covariance-mode diagnostics.
The proposed 1e-3 scale is a starting numerical target, not a
literature-mandated accuracy of the physical covariance model. See
`references/covariance_accuracy.md` for papers, proposals and limitations.
Keep tight algebra, units and determinism checks separate.
The immediate target is **roman_real**, with eight lens and eight source
bins. Eifler et al., arXiv:2004.05271, provides survey guidance; do not
replace the project's layout with the paper's ten-bin Fourier analysis.
See `references/covariance_roman.md`. Every new all-pairs cross-bin and
non-Limber C implementation belongs in `cosmolike/covariances/`.
Start runtime estimates with small representative components, separating
shared tables from work repeated per bin pair; do not start an hours-long
full covariance just to estimate its cost. See
`references/covariance_roman_timing.md` for the measured pilot and its limits.
The module-wide didactic pass is recorded in
`references/covariance_didactic_review.md`. Follow it with measured small
component profiles and optimization experiments, including cubic-spline
upsampling as used in `halo.c`, `cosmo2D.c` and `pt_cfastpt.c`. Verify both
runtime and accuracy; do not infer a speedup from adding SIMD or unrolling.
The intended spline architecture is coarse exact evaluation, cubic-spline
upsampling during table construction, then linear interpolation on the
dense table in the hot path. Do not replace the hot lookup with a cubic
solve/evaluation. Test off-grid linear queries against direct calculations,
in addition to checking the dense nodes, and distinguish interpolation in
physical wavenumber from interpolation along k=(ell+1/2)/chi(a).
When an accuracy boost refines interpolation tables, preserve existing
sample positions: double intervals, not endpoint-inclusive point counts.
If a cutoff grows, extend the same grid rather than stretching it. Test
actual node retention and a high-boost convergence sequence; separate
grid movement from interpolation density, cutoff error and fixed input
power-table resolution. Gauss--Legendre quadrature nodes are a different
case: refine their nodes and weights together and measure convergence.
See `references/covariance_accuracy.md` for the measured grid audit.
Study `cosmo2D.c::limber_fill_interp` and the `legendre_sums`/`xipm`
transform helpers when designing covariance lookup and projection:
share grid indices across tables, prefer SIMDe for bulk linear reads,
and reuse spectra/kernels across several output sums.
Measure lookup and contraction separately; preserve each sum's order
and keep the covariance implementation inside its own directory.
Production chains run on x86 supercomputers. Treat Apple/NEON timings as
local diagnostics, not a reason to discard SIMDe or an x86-oriented
optimization. Vector widths, gather costs, FMA mappings, cache sizes and
register pressure differ. Retain unconditional SIMDe and benchmark the
actual x86 production compiler/CPU before selecting architecture-sensitive
unrolling, tiling or lookup layouts. Never label Mac timings as x86 gains.
SIMDe is the default design choice for independent bulk arithmetic;
measurements refine its layout rather than treating an unimpressive Mac
result as a veto. Run accuracy checks first. Timing reports require a
quiet machine and one benchmark at a time, with no concurrent tests,
builds, CAMB jobs or other computational experiments. Do not report
contended preflight timings as optimization evidence.

**Covariance parallelism.** Strong
scaling to 8--10 OpenMP cores per process is a primary requirement.
Measure 1, 2, 4 and 8 threads on this laptop; do not accept good 2--3-core
scaling as sufficient. Audit small outer-loop counts, serial setup and
load imbalance. Test collapse over independent indices or larger input
batches, preserving deterministic sums and the unconditional SIMDe paths.
The laptop has eight performance cores; qualify any 10-thread result
because it also uses efficiency cores. Production x86 scaling still needs
measurement on that hardware. Keep BLAS at one thread.
Covariance C must never call MPI. Cocoa/Cobaya's Python layer owns MPI;
a future C++/Python interface can dispatch independent matrix subblocks
to processes, each using OpenMP internally. Example 40-core layouts are
five MPI processes times eight threads or four times ten. Keep shared
tables reusable within a process and make block inputs explicit; do not
implement an MPI layer or a speculative block framework before needed.
The complete Gaussian C assembler assigns whole observable blocks to
workers; its C primitives suppress inner parallel teams when called from
that outer region. Preserve the fixed multipole sum order and per-worker
scratch ownership. Measurements and checks are recorded in
`references/covariance_gaussian_scaling.md`; the complete survey still
requires separate scaling measurements, especially its shared matter tables.
The seven-project real/Fourier baseline is recorded in
`references/covariance_survey_scaling.md`, including its correction for
three skipped project rebuilds. Check the actual compile flag and build
log: most projects use upper-case `IGNORE_COSMOLIKE_*_CODE` names, while
DES cluster uses lower-case `des_cluster`. A zero shell exit alone does
not establish that a skip-guarded build script compiled anything.
Shared matter preparation now
groups eight radial shells and requests only I11 at displaced derivative
endpoints; see `references/covariance_halo_scaling.md` for measurements,
bitwise checks and the bounded-memory choice. The final connected
projection also assigns complete angular blocks to one worker team;
see `references/covariance_connected_scaling.md`. Keep its shared radial
weights, per-worker scratch and exact triangular ownership. Shared-table
power reads and smaller projections still need profiling; do not claim
that the complete scaling problem is solved.

**Covariance CLI workflows.** Each project supplies an
`EXAMPLE_EVALUATE_COVARIANCE.yaml` and a thin Python runner. Use Cobaya's
`yaml_load_file` and `Parameterization`, keeping familiar `theory`, `params`,
`sampler: evaluate` and `output` blocks. Evaluate one explicit cosmology;
never silently sample priors. `covariance` contains measurement, thread and
accuracy controls, inheriting the project's usable `default.yaml` baseline.
Shared reading and assembly belong in `cosmolike_notebook_utils`; runners
select the optimized production interface, not notebook wrappers. Document
HPC usage independently of Jupyter. Explain Armadillo through the Python
notebook API it makes convenient; C++ is a thin bridge to shared C physics.

**Notebook covariance workflows.**
Develop the first public examples in `projects/lsst_y1/covariance/`, using
explicit LSST Y1 survey inputs. Keep reusable Python calculations in
`cosmolike_notebook_utils`, with the initialized project interface passed
by the caller. Project folders own survey choices and thin examples;
design them for replication across projects without copying algorithms.
Port useful Python from the external covariance reference work, not its
bash scripts, machine-specific library paths or benchmark scaffolding.
Keep independent numerical test references independent of production
calculations. Separate test modules into `tests/data_vector/` and
`tests/covariance/`, with clear commands for each sector. The ordinary
documented test command selects data-vector tests. Do not refreeze or
modify likelihood snapshots merely to reorganize the test files.
The public entry point is `EXAMPLE_EVALUATE_COVARIANCE.ipynb` in LSST Y1.
Expose one covariance `accuracy_boost`, resolving numerical controls in a
shared helper; do not ask ordinary notebook users to tune a list of grids.
**Boost 1 must be usable.** Each project's `covariance/default.yaml` owns
the fine-tuned base controls needed for its bins and scale range. Establish
their convergence against a high-resolution calculation and further
refinement; a fast smoke configuration must not be the public default.
Study that project's likelihood YAML accuracy choices when setting the
starting baseline, without assuming data-vector convergence certifies a
covariance. Keep test-only small grids explicit in tests.
The global boost multiplies every internal refinement, rather than replacing
or bypassing it: base factors 2 and 3 become effective factors 4 and 6 when
the global boost changes from 1 to 2. This applies to internally tuned
non-Gaussian and window tables and shared core reader refinements.
**Quadrature is separate.** `integration_accuracy` selects precomputed GSL
rules through an explicit level ladder, independently of `accuracy_boost`.
Do not multiply radial, mass or angular rule orders by the global boost.
Do not expose arbitrary rule sizes in project defaults. Covariance rules
must use GSL's precomputed nodes, with 64 as the absolute minimum even if
a 32-node test appears adequate. The notebook ladder is 96/128/256/512/1024
for levels 0/1/2/3/4; low-level testing also accepts 64. Python-prepared
production integrals obtain the same GSL rules through the C++ binding;
independent references may generate their own rules. Split oscillatory
angular integrals into physical panels rather than requesting generated
2048-node rules. Level zero must be useful: test every consuming sector,
including cluster selection and count responses, against higher levels.
Check levels 2, 3 and 4 as well as 1, comparing the default directly with
the highest level and verifying stability of the last refinement.
Current implementation and evidence: `references/covariance_defaults.md`.
Keep this level unchanged under global table refinements. Multiply interval
counts in likelihoods too: `nonlimber_accuracyboost: 2` and
`pk_z_refinement: 3` mean effective factors 4 and 6 at `accuracyboost: 2`.
For covariance interpolation, multiply interval
counts, preserving existing interpolation nodes under doubling; divide
finite-difference step sizes by the boost. Do not multiply fixed physical
bin edges or survey inputs. Expose and save both base controls and resolved
effective settings, and test non-unit internal factors as well as defaults.
For example, verify base factor 3 at global boosts 1 and 2, not only powers
of two. Any implementation limit must raise a clear error, never silently
cap a resolution and make the global boost ineffective.
The notebook must compute several boosts and show changes in the covariance,
its error bars and relative modes. A largest tested boost is a comparison
reference, not an automatic claim of convergence. Covariance READMEs follow
Cocoa's numbered contents, anchors, assumptions and Step flows; teach setup,
compilation, notebook execution, accuracy changes and separate tests.
Commit completed pieces incrementally; avoid commits of many thousands of
lines. Keep mechanical test moves separate from numerical changes.
Shared covariance plotting belongs in `cosmolike_notebook_utils`; consult
Krause's papers for interpretable layouts, cite exact figures, and render
and inspect the resulting panels. The plots must preserve signs and
visibly mask undefined ratios, never repair eigenvalues or fabricate a
missing component. See `references/covariance_notebook_workflows.md`.

Then diagnose the negative eigenvalues reported in the existing Roman
covariance, distinguishing the full matrix from the likelihood selection
and tracing the responsible scales and components without clipping modes.
The shipped-matrix localization and a controlled legacy interpolation
failure are recorded in `references/covariance_roman_negative_modes.md`.
The same review finds zero NG between different lens bins despite window
overlap; combining lens families introduces the failing mode. Never copy
the legacy writer's equal-lens-only NG rule into the rewrite. Compute
cross-lens covariance terms even when those spectra are absent from the
data vector; their C implementation stays covariance-owned.
Prioritize the future Roman generator over recovering the old file's
provenance. Use its failure to design regression tests: recompute
physical cross-lens responses, retain complete subblock coverage, and
check full and selected total matrices. Do not make historical attribution
a prerequisite for developing and validating the new covariance.
For SSC, interpolate common responses before forming their weighted outer
products. Do not independently interpolate auto/cross covariance blocks
with inconsistent value/log prescriptions and assume positivity survives.
Then establish a stable high-resolution numerical reference and use Fisher
FoM/errors to select practical settings. A chain is not needed for the
initial local Fisher test; check several cosmologies before generalizing.

Before doing any Docker work — Dockerfile edits, GPU-container debugging, 
image size diagnosis, or container build failures — 
read `references/docker-reference.md`. It contains 
the methodology and conventions for editing Dockerfiles, 
the GPU stack model, dependency-resolution patterns, and image-size diagnostics.

## Core principles

1. **Correctness is non-negotiable, at the scale physics can see.** The
   pass criterion of a frozen-reference test is |chi2 - reference| < 0.2
   (`CHI2_TOLERANCE`): no physics is detectable below that (CAMB settings,
   CAMB versus CLASS already move chi2 by that much). Keep this threshold
   unchanged. Record chi2 and its difference to at least four decimals so
   drift below the threshold stays visible. Separately, an optimization or
   refactor that is not meant to change the physics is checked on the full
   unmasked data vector, not on chi2 alone (see the validation protocol).
2. **Measure, never guess.** Every optimization starts with a profile and ends
   with `perf stat -r 3`. Single-evaluation timings on shared nodes (SeaWulf,
   NVWulf) are noise; never accept or report them as evidence.
3. **Algorithms first, SIMD last.** The dominant wins in this codebase came
   from precomputation and factoring redundant work out of loops (cosmo_nodes,
   cfftlog forward-FFT hoisting, trapezoid factorization), not from intrinsics.
   Only vectorize a loop after its algorithm is already minimal.
4. **One change at a time.** Each change is validated and measured in
   isolation. Never bundle a refactor with an optimization in one commit.
5. **Keep a checkable reference for optimizations.** Scalar comparison
   implementations belong in external tests, not selectable production
   branches. `COSMO2D_NOT_USE_SIMD`, `HALO_NOT_USE_SIMD` and the covariance
   scalar switch are retired: optimized and debug builds always
   compile the existing SIMDe paths. Keep scalar single-point kernels and
   vector tails where the algorithm needs them. See the retirement record
   for the pinned historical source and independent validation checks.
6. **Preserve existing conventions.** Keep the file's variable naming, struct
   layout, and code structure when modifying. Renames happen only as their own
   dedicated, mechanical commits (e.g. `zdistr_photoz` → `nz_source_photoz`).
   Before adding a grid lookup, read and reuse the existing indexing helpers.
   In cosmo3D.c, use `piecewise_index` and the setter-provided segment metadata
   for piecewise-uniform redshift grids, and direct arithmetic for uniform
   grids. Do not reintroduce binary searches into these optimized paths:
   they add branches and scattered table reads that the existing metadata
   was designed to avoid. This applies to every build, including debug:
   COSMO3D_ASSUME_PIECEWISE_UNIFORM and its binary-search alternatives are
   retired. Setters always construct and validate the metadata.
7. **Determinism is a correctness test.** If chi2 varies run-to-run or with
   `OMP_NUM_THREADS`, there is a race or uninitialized memory. Full stop. Do
   not proceed until it is found.
8. **OpenBLAS always uses one thread.** Parallelism belongs to the explicit
   CosmoLike OpenMP loops. Pin OpenBLAS directly at interface initialization
   and before both normal and cluster covariance checks/inversions; do not
   restore a larger BLAS team afterward. Environment variables alone are
   insufficient for an OpenMP-built OpenBLAS. Keep the inverse residual
   checks so a corrupted inverse stops initialization.

## The change loop (mandatory workflow)

```
profile → baseline → change ONE thing → validate → measure → document → commit
```

1. **Profile.** `perf record` / `perf report` on the benchmark to identify the
   actual hot function. Do not optimize code that isn't hot. Sanity-check hot
   percentages against physics combinatorics (e.g. xi_pm at 36 bin pairs x 2
   spins legitimately dominates w_gg at 8 effective combinations).
2. **Baseline.** Default build. Record the reference chi2 at the validation
   point and a `perf stat -r 3` run of the 1000-evaluation benchmark.
3. **Change one thing.**
4. **Validate** per the protocol below. No exceptions, even for "trivial"
   changes — the memset and inline-linkage bugs both came from trivial changes.
5. **Measure.** `perf stat -r 3` again. Report wall time, instructions, IPC,
   and FP-vectorization breakdown. A regression in instructions with flat wall
   time still matters (memory-bound code hides it until the cache budget moves).
6. **Document.** Comments must explain *why* an idiom exists, especially when
   the fast version looks gratuitous (see the `restrict` example below).

## Validation protocol

Run all of these before declaring a change correct:

- **Frozen references:** |chi2 - reference| < 0.2 in the default IEEE build
  (the tests' `CHI2_TOLERANCE`), with chi2 and the difference printed to at
  least four decimals so drift is visible.
- **Full data vector, for optimizations and refactors:** chi2 is one
  covariance-weighted number after the mask; it cannot see a bug in masked
  entries (small scales cut by the scale cuts, which notebooks and other
  masks do use) or in a branch the frozen point never reaches (table edges,
  large photo-z shifts, out-of-grid fallbacks). Compare the full unmasked
  data vector against an external reference build at several parameter
  points. Bitwise equality is the default
  expectation when the operation order is unchanged: it is free and it
  catches a single misplaced rounding. Where an optimization cannot be
  bitwise (a vector libm replacement, reordered sums), say so in a comment
  and in the commit, and state the per-entry tolerance the comparison used.
- **Determinism sweep:** repeat the evaluation several times within one
  process and across processes, at `OMP_NUM_THREADS=1` and a high count
  (e.g. 8 or node-width). Any variation = race / uninitialized memory.
- **Both supported build modes** (Makefile):
  - `COSMOLIKE_DEBUG_MODE`: `-O0` + sanitizers. Catches linkage bugs
    (`inline` vs `static inline`), out-of-bounds writes, double frees.
  - default: strict IEEE-754 (`-fno-fast-math -frounding-math
    -ftrapping-math -fsignaling-nans`) with LTO + unrolling. This is the
    bit-reproducibility reference.
  - Aggressive mode is retired. Makefiles
    reject `COSMOLIKE_AGGRESSIVE_MODE`: the fast-math build produced incorrect
    covariance inverses even with OpenBLAS at one thread. Do not reintroduce
    it or enable `-ffast-math`, `-Ofast`, `-funsafe-math-optimizations`,
    `-fassociative-math`, `-ffinite-math-only`, `-freciprocal-math`,
    `-fno-signed-zeros`, or `-fno-trapping-math`. These flags relax the
    floating-point contract; the failing bundle was tested, not each flag
    independently. Retain the optimized strict-IEEE default and debug mode.
- **Both Limber paths** (C_ss and C_gs), **both real-space projections**
  touched (xi_pm, gammat, w_gg), **both theory paths** (emulator and exact
  CAMB), and **both IA branches** (NLA and TATT) when relevant.
- **FFT/FAST-PT changes:** call the function at least twice in the same
  process. Plan-caching and static-state bugs only show on the second call.
- **Non-Limber changes:** verify the per-bin early-exit still converges to the
  same C_ell as the no-early-exit fallback.

## Benchmarking standard

**Time cosmolike, never CAMB (a repeated mistake).** Every speed
judgment is made against cosmolike's OWN per-step time, with the
Boltzmann/theory time taken out: production will replace CAMB by the
emul2 output emulator, so CAMB seconds are not the denominator. A
whole-`evaluate_chi2` or `logposterior` wall time mixes both and
understates every cosmolike cost (2026-09-29: the halo spectra were
judged "0.1 s of 1.6 s, negligible" - but the 1.6 s was mostly CAMB;
against cosmolike's own 150 ms/step the halo builds were a third of
it). Recipe: cobaya `info["timing"] = True`, one warm-up evaluation,
then N evaluations at jittered cosmologies (e.g. As x (1 + 1e-3 i))
so every table refills, and read the likelihood component's timer
separately from the theory components'. Reference numbers (M2, 4
threads, NLA 3x2pt, 2026-09-29): roman_real cosmolike 150 ms/step
(CAMB 241 ms), lsst_y1 89 ms (CAMB 233 ms).

- Benchmark = 1000 likelihood evaluations of `roman_real.combo_3x2pt` via
  `cobaya-run` under the MPI wrapper.
- Always `perf stat -r 3` (3 repeats, mean ± stddev) with hardware counters:
  cycles, instructions, IPC, `fp_arith_inst_retired` scalar/128/256/512,
  LLC-loads and LLC-load-misses.
- FLOPs = N_scalar + 2*N_128 + 4*N_256 + 8*N_512. Vectorized FP work % =
  (2*N_128 + 4*N_256 + 8*N_512) / FLOPs. The optimized code sits near ~78%
  (CCL, for comparison, ~2.5–31% depending on configuration).
- Confirm auto-vectorization with GCC `-fopt-info-vec-all`; look for
  "loop vectorized using 32 byte vectors" on the loop you care about.
- Under the default strict flags (`-frounding-math`, no
  `-fassociative-math`) clang auto-vectorizes no FP loop, and says
  nothing about it: check `-Rpass=loop-vectorize` /
  `-Rpass-missed=loop-vectorize` remarks before believing a loop is
  vectorized. SIMDe intrinsics do compile to vector instructions (verify
  by disassembly; the `u_KS` S/Q sums, now in `future_port_unfinished/`:
  `v4d` mul then add). SIMDe and scalar
  may agree to ~1e-12 rather than bitwise when operation order changes.
  Keep the scalar comparison in the external test harness. Native fused
  operations that preserve order should still be checked bitwise.
- Mind IPC interpretation: this workload is memory-bound (~1.1 IPC, ~25% LLC
  miss rate is normal). Low IPC is not by itself a problem to "fix".
- Landmarks (June 2026 snapshot; re-measure, don't trust): full benchmark
  ~111s; per-eval exact CAMB ~165ms, emulator path ~259ms; TATT `get_FPT_IA`
  ~72ms dominates the emulator path; non-Limber ~12ms (was ~150ms).
  Hot path: `xi_pm_tomo` → Limber fill → Legendre summation.

## Which model reviews and writes documentation

Documentation passes and reviews go to **Fable 5** (model id
`claude-fable-5`), not Fable 5.1. The Agent tool's generic `fable` setting
does not pin the version: use the `fable5` agent type
(`.claude/agents/fable5.md`, frontmatter `model: claude-fable-5`) for
every Fable task.

## Clean & Human-Readable Code Style Guide

**Ticket completion gate.** After the
implementation and tests for each major ticket, make a separate didactic
red-eye review pass before starting the next major ticket. A ticket is a
substantial component, such as the Gaussian covariance foundation; it is
not every helper function or intermediate edit. Read the complete changed
component as an advanced undergraduate physics student: verify that the
physics, units, array roles, numerical steps, threading and vectorization
can be followed without unstated specialist knowledge. Fix unclear prose
and dense code, and rerun relevant checks if the review changes behavior.
Record the review and any remaining limitations with the ticket's results.

### Mission
You MUST prioritize human scannability, structural clarity, and
junior-developer (i.e., student) readability over compact or clever code
syntax. Optimized code must remain understandable to a physics student;
runtime performance does not replace clear explanations.

### Non-Negotiable Formatting Boundaries
1. **Vertical Breathing Room:** Always separate logical blocks, variable
   initialization phases, and calculation sequences with single blank
   lines.
2. **Visual Banners:** Use distinct, short uppercase comment banners to
   section out complex algorithms (e.g., `// --- 1. CONFIGURATION ---`).
   A banner is a navigation label, not an explanation. Before a substantial
   sequence, follow it with a short paragraph connecting the physical
   purpose to the calculation: define the quantities, explain why these
   inputs or terms belong together, and state the result this stage supplies
   to the next one. For SIMD, distinguish physical indices from vector lanes
   and explain which quantities interact within one lane. Keep the detailed
   per-call comments too. During didactic review, read each stage introduction
   without its code: it must explain the reasoning, not repeat the heading
   or list operations such as "pack, multiply, store."
3. **Assignment Alignment:** Where clear and practical, vertically align
   consecutive `=` assignment operators to keep variable declarations
   neat and organized.
   Declare one variable per line, including struct members and FFTW
   plans. Give each struct member a short trailing `//` explanation;
   document array dimensions and index meanings as well. Do not pack
   several declarations onto one line: the reader should process one
   quantity at a time.
4. **No Dense Logic Chains:** Break down complex multi-conditional
   statements or ternary operators into individual, well-named temporary
   variables or multi-line structures.
   For cache reuse/reallocation, follow cosmo2D.c: put the cache-change
   conditions directly in the `if`, one comparison or predicate per line.
   This also applies to assignments combining `&&` or `||`, not only guards.
   Use line breaks rather than unnecessary single-use boolean variables.
   Do not introduce a single-use `rebuild` flag for that condition.
5. **Guided Context:** Add bite-sized, purposeful inline comments before
   mathematical equations or data-transformation loops explaining *why*
   the code is performing that action, not just *what* it is doing.
6. **80-Character Lines:** Keep C code and comments within 80 columns.
   Wrap function arguments, comparisons, and intrinsic calls at natural
   boundaries. Check line lengths during the ticket's didactic review.

Variable names say the physics (`n_gal`, `b_gal`,
not `ng`, `bg`, `tq`, `occ`); logs are `ln<quantity>` (`lnk`, `lnx`,
`ln1c` — a bare `l` prefix like `l1c` or `lc` is banned); one statement
per line. Speed is never the excuse: names, blank lines and comments
cost nothing at run time.

### Visual Code Geography

- **Section banners.** Major logical sections are wrapped in distinct
  banners:

```c
// ============================================================================
// [SECTION] COSMOLOGICAL INTEGRATION & HALO BIAS
// ============================================================================
```

- **Vertical whitespace.** 3 blank lines between major algorithmic
  modules / distinct physical steps; 2 blank lines between helper
  functions or mathematical definitions; 1 blank line inside a function
  between phases (pre-computation vs the integration loop).

### SIMD code a student can read

Split nested expressions such as
`nfw_um4(simde_mm256_loadu_pd(conc_gal + q),
simde_mm256_mul_pd(vk, simde_mm256_loadu_pd(r_sg + q)), ...)` into named
steps. Explain **every SIMDe call** immediately before it, including
repetitions of a previously explained intrinsic.
SIMD means applying the same operation to several numbers at once; each
number occupies a vector position called a lane. Do not assume a physics
student already knows these terms or the intrinsic naming conventions.

- Immediately above each substantial SIMD block, show the analogous scalar
  calculation as a short commented C example using the surrounding array
  names. Explain why that calculation gives the physical quantity, then map
  its indices to lanes: adjacent nodes, different bins, or independent sums.
  Keep the scalar example in comments, not a production fallback. Use `fma`
  for fused steps, and state when a summary describes the mathematics rather
  than the exact reduction order. The example complements the per-call
  explanations below; it does not replace them.
- Put one intrinsic per statement, with named intermediate results.
  Immediately before **each call**, explain its inputs, operation and
  result in terms of those physical quantities. A glossary elsewhere,
  or one explanation for the entire loop, does not satisfy this rule.
- Explain broadcasts, loads, arithmetic and stores individually. For
  `set_pd(high, low)`, explicitly give lane order; for `loadu/storeu`,
  give the array indices and explain that no vector-aligned address is
  required. A store still requires enough valid array elements.
- For fused calls, give the exact scalar expression, including the sign
  of `fmsub` or `fnmadd`, and explain the fused rounding convention.
  For reductions, say which entries are added and in what order.
- Explain the two-at-a-time loop bounds, scalar remainder, or repeated
  final lane. Identify where a duplicate result is discarded.
- Use blank lines between loading, physical arithmetic, accumulation
  and output. Within a long block, separate each conceptual step with a
  short explanatory comment and whitespace. Apply this also to scalar
  setup: allocation, boundaries, normalization, quadrature and storage
  are separate steps, not one dense paragraph of code.
- Immediately above every substantial loop, give an overview of its
  purpose, what one iteration represents, which inputs it reads, and
  what it computes or stores. For OpenMP put the overview before the
  pragma. Explain nested loops at their own level too. When SIMD is used,
  the overview must say what each lane represents, what work happens
  together, and whether the lanes are eventually added or remain separate.
  Detailed comments inside a loop do not replace this overview.
- An overview must **explain the reasoning**, not merely narrate operations
  such as "advance a sample, append a trapezoid, save the result." Define
  the physical quantities and explain why this traversal, weighting or
  approximation computes the desired quantity. For example, explain that
  the lensing integrals count galaxies behind a foreground distance, and
  that a trapezoid integrates a straight-line approximation between two
  sampled endpoint values. State what the SIMD lanes mean in that reasoning.
  A list of variable names and programming verbs is not a substitute.

Preserve the original operation graph and test numerical equivalence
when exposing nested calls as named steps. Do not combine operations or
change a fused call into separately rounded multiplication and addition.

### Equation-to-Code Blueprinting

Write for an advanced undergraduate physics student. Introduce the physical
quantity, derive the change of variables, define each new symbol, and only
then explain the numerical operation. Separate those steps into paragraphs
and displayed equations in the comment. A compressed formula plus labels
such as "Mellin kernel", "house spline", or "FFT-friendly tail" is not an
explanation. Describe what is integrated, what an array holds, where extra
nodes are placed, and why the calculation needs them. Use the detailed
function headers in `halo.c` as the documentation model, including inputs,
outputs, units, cache ownership, and a map from equations to code.
This applies to numerical helpers too: explain the algorithm and define
its terminology. For example, an FFT-size helper must explain transform
factorization, what a radix is, why the selected small factors help FFTW,
and what the returned padded length changes. Two lines naming the
algorithm are not sufficient documentation for a student.

Before any complex mathematical loop or physics derivation, insert a
comment block titled `/* PHYSICAL DERIVATION & LOGIC FLOW */` mapping
the code's math back to the textbook formulas:

```c
/* PHYSICAL DERIVATION & LOGIC FLOW
   1. Calculate halo mass m from log-mass space: m = exp(lnM)
   2. Compute peak height: nu = delta_c / (sigma(m) D(a))
   3. Compute the HOD expected number: <N> = fc Nc + Ns
   4. Integrate the weighted bias contribution over the mass function. */
```

### Cognitive Complexity Limits (the "physics student" standard)

Assume the reader is a physics student who knows the math and needs
absolute clarity on how variables map to formulas.

- **Do not overengineer or add code bloat for obscure failures.** When
  a rare unsupported condition can be checked with a simple `if` guard,
  check it and stop with a clear error. Do not build recovery machinery,
  retry paths, compatibility layers or speculative fallbacks for cases
  that almost never occur. Keep the implementation focused on the
  supported scientific calculation. This does not remove the numerical
  reference paths required to validate an algorithm change.

- No nested ternary operators; explicit `if / else` blocks.
- No single-line blocks: always braces `{}` on loops and conditionals.
- No unexplained magic numbers: every physical constant, integration
  bound, or unit conversion gets a `const` with a descriptive name.
- Banned: cryptic ultra-short names (`nq`, `sn`, `sb`, `tq`) that hide
  the physics. Required: names mapping to physical concepts or clear
  code spellings of the LaTeX symbols.
- Dated measurements ("Measured 2026-09-29 ... one refill 0.5 s") never
  appear in source comments — they live here, in the skill file, or in
  session notes. Source comments serve one purpose: the connection to
  the physics, which optimization obscured and the comments restore.

## Code style

- **2-space indentation. Never tabs.**
- `//` comments. Section separators in utility files use `// ---` lines.
  Comments explain *why*, with enough detail that the next person doesn't
  "simplify" a load-bearing idiom. Canonical example:

```c
// Local restrict pointers: without these, GCC cannot prove that Pl[i] and
// Cl[nz] don't alias (pointer-to-pointer indirection inside a collapse(2)
// OpenMP region), so it emits conservative reload-checking code -> ~2x
// slowdown on this loop. The body is a single FMA with nothing to hide the
// overhead behind, so aliasing pessimization dominates.
const double* restrict pl = Pl[i];
const double* restrict cl = Cl[nz];
double sum = 0.0;
#pragma omp simd reduction(+:sum)
for (int l=lmin; l<Ntable.LMAX; l++) {
  sum += pl[l]*cl[l];
}
```

- `static inline` always. Bare `inline` breaks linkage in `-O0` DEBUG builds.
- SIMDe typedefs: `typedef simde__m256d v4d;`, `typedef simde__m128d v2d;`,
  `typedef simde__m128i v4i;`. Vector variable names like `vWK1` (prefix `v`,
  no underscore). SIMDe gives AVX2 on Linux and NEON on Apple Silicon from the
  same source — never use raw `_mm256_*` intrinsics.
- Allocation only through the custom `malloc1d/2d/3d/4d` (posix_memalign,
  64-byte cache-line padded rows). Each returns one block, pointer rows
  included: one `free` per table. Zero only through `zero1d/2d/3d/4d`.
  **Never** a flat `memset` over a padded multi-dim allocation (see pitfalls).
- Combine work arrays with matching dimensions and lifetimes into one
  higher-dimensional allocation. For example, the real FFT input, variance
  output and derivative output belong in `malloc3d(threads, 3, nfft)`;
  document the role index and use named local pointers inside the loop.
  Keep arrays with different shapes separate; do not add an allocator
  abstraction merely to combine them.
- Compile production SIMDe paths unconditionally, including in debug
  builds. Keep scalar comparison implementations in external tests; do not
  reintroduce production SIMD opt-out macros.
- Fortran (custom CAMB): all modifications fenced with `!VM BEGINS` /
  `!VM ENDS` so they survive upstream rebases.
- Function naming for the established decompositions: `<name>_work` for
  batched tomographic-block computation, `<name>_fill` for gather-based
  interpolation table fills, `int_for_<name>_core` for scalar integrand cores
  callable from both SIMD bodies and scalar tails, `<name>_params_at` /
  `<name>_core` for a fit split into its per-axis coefficients (once per
  a) and its per-node remainder (halo.c: `hb1nu`, `fnu`). Batched and
  scalar callers both run params_at then core, with the same arithmetic
  in the same order, so their results are bitwise identical.

## Codebase map (hot-path oriented)

- `cosmo2D.c` — Limber C_ell (C_ss, C_gs, C_gg) and real-space projections
  (`xi_pm_tomo`, `w_gammat_tomo`, `w_gg_tomo`). Contains cosmo_nodes
  precomputation, `_work`/`_fill` batch functions, Legendre summation loops.
  Note: the cosmo_nodes refactor was applied to ss and gs but **not** gg —
  known remaining work.
- `pt_cfastpt.c` — FAST-PT wrappers: `get_FPT_IA` (TATT, the single most
  expensive function on the emulator path) and `get_FPT_bias`. Single
  `J_abl`/`J_abl_ar` dispatch, static FFTW plan cache, `next_fft_size`.
- `cfastpt.c` — FFTLog core used by FAST-PT.
- `IA.c` — intrinsic alignment kernels (NLA, TATT amplitudes).
- `redshift_spline.c` — `nz_source_photoz`, `nz_lens_photoz` (uniform
  fine-grid linear interpolation, no GSL search), lens efficiencies
  `g_tomo`/`g2_tomo`/`g_lens` (factored cumulative trapezoid, P − chi*Q).
- `cfftlog/` — non-Limber pipeline; `cfftlog_ells_cocoa0` hoists the
  ell-independent forward FFT out of the convergence loop.
- `halo.c` — halo model: Tinker multiplicity and bias (`tinker_alpha`,
  `fnu`, `hb1nu`, `bias_norm`), the NFW profile (`u_nfw_c` on the f/G
  table), HOD tables (`hod_tables`: `ngal`, `bgal`), spectra `p_gm`/`p_gg`
  (`p_mm`, `p_my`, `p_yy`, the KS gas profiles `u_KS`, `frac_bnd`,
  `frac_ejc`, `u_y_ejc` and `u_c` are kept, not compiled, in
  `future_port_unfinished/`). Every lazily built table
  is warmed by `halo_warmup`. Numerics: "halo.c numerics" below.
- `basics.c` — allocators, `zero*d`, interpolation utilities
  (`spline_coeffs_uniform` + direct-index Horner is the house spline;
  `spline2d_upsample_uniform` its tensor-product 2D form).

## Accuracy knobs

`init_accuracy_boost(accuracy_boost, integration_accuracy)`
(generic_interface.cpp) is the GLOBAL SUPER FUNCTION for sampling
accuracy: one call scales every sampling knob from its first-call
baseline (repeated calls rescale the same baselines — they never
compound):

- ceil(baseline x boost): `Ntable.N_a`, `N_ell`, `N_ell_internal`,
  `dCX_dlnk_nlnk`, `dCX_dlnk_nlnk_internal`, `N_M_internal`,
  `halo_uks_nc`, `halo_uks_nz`, `halo_nfw_n`, `NL_Nchi`,
  `nz_fine_sampling_factor`
- baseline x boost (double): `Ntable.FPT_internal_accuracy_boost`
- also written: `Ntable.FPTboost` (int(boost − 1) for boost > 1;
  FAST-PT grids) and `Ntable.high_def_integration =
  integration_accuracy` (the hdi quadrature-order ladders)

The internal coarse grids scale together with their dense tables, so
the coarse/dense ratios are boost-invariant, and a knob whose
baseline is 0 (disabled) stays
0 under any boost. The dedicated setters (`init_ntable_ell_internal`,
`init_ntable_dcx_dlnk_nlnk_internal`, `init_fpt_internal_boost`, ...)
are individual overrides: called before the first boost call they
define the baseline, called after they overwrite the boosted value.

**When adding a new sampling knob, wire it into
init_accuracy_boost's ladder in the same change** — users must never
need a second call to shift overall accuracy.

## Deep unrolling (the `_work` technology)

A table build is ONE explicit loop nest in the function that owns the
table, not a chain of calls. The pattern to remove:

```
table[i][j] = X_nointerp(x_i, y_j)     per table point
  -> GSL fixed-order integrator        per table point
    -> int_for_X(node, void* params)   callback, per node
      -> more table lookups            per node
```

It hides the loop nest from the compiler and from the reader.
Invariants get recomputed at the innermost level (e.g. a profile power
theta(x)^p evaluated inside a per-(c, y) integrand callback for every
(c, y, node), although it depends only on (c, node): redundant by the
size of the y axis).
Callbacks through function pointers cannot vectorize. And only the
outermost loop can be threaded.

Rules:

- Outer loops = table axes; innermost loop = quadrature nodes.
- Fold single-caller helpers (`*_nointerp`, `int_for_*` callbacks) into
  the owner and delete them. A `*_work` batch whose only caller is the
  table owner folds in too (`sigma2_work` into `sigma2`; `bias_norm_work`
  into `bias_norm`).
- Hoist every quantity to the outermost loop level it depends on: per
  Ntable (nodes, weights), per cosmology (sigma(M), nu), per parameter
  change (HOD occupations, profile powers), per axis value (fit
  parameters at each a), per node (the rest).
- Innermost loop = a plain multiply-add over node arrays: local
  `restrict` pointers + `omp simd` (SIMDe only where `omp simd` fails).
- Thread/collapse the loops that carry enough independent work, with no
  cross-thread reductions (determinism).
- Quadrature nodes placed on an existing table's grid turn its
  interpolation into a gather with weights 0/1 (e.g. a halo-model mass
  integral quadratured on the sigma2 ln M nodes reads sigma2 exactly).
- Code duplication across consumers is acceptable when it buys speed.
- Gauss-Legendre sizes: always a size GSL has precomputed (tabulated).
  The minimum accepted size is 64, even if a 32-node check seems adequate.
  The hdi ladders use 64, 96, 128, 256, 512, 1024, written inline at
  each site, e.g. redshift_spline.c:
  `(0 == hdi) ? 256 : (1 == hdi) ? 512 : 1024; // predefined GSL tables`.
  `malloc_gslint_glfixed` (basics.c) accepts any n and silently computes
  a non-tabulated rule on the fly, with weights good to only ~5e-7: it
  does not enforce the rule, the caller does.

Worked example: `sigma2` (cosmo3D.c) — the lobe-node cache is built in
its Ntable rebuild block, and one threaded lobe-sum loop refills the
table per cosmology.

## halo.c numerics

Developer record for halo.c: quadrature choices, table designs,
accuracy protocols, measured accuracy and cost, and the rules they
imply. Source comments state what the code does and its invariants; the
numbers and their reasons live here. Dated figures are snapshots:
re-measure before relying on them.

### Quadrature

Every Gauss-Legendre rule in halo.c is a size GSL has precomputed (the
rule above). `Ntable.high_def_integration` (hdi) selects the size per
integral family:

- `u_KS` gas integrals: 96 / 128 / 256 / 512 / 1024; 96 nodes converge
  F0 to 2e-14, so the ladder buys nothing.
- `bias_norm`: 128 / 256 / 512; 128 nodes converged to 3e-15 (powers
  and one exponential; `sigma2` enters only at the end points). The
  ladder never needs raising.
- `ngal` / `bgal` (`hod_tables`), GL in ln M: 128 / 256 / 512 / 1024.
  Node study (2026-09-29; 5 bins x a = 0.5 / 0.75 / 0.95 vs 32-node
  panels 0.05 wide): 5e-7 / 1.3e-7 / 3e-8 / 5e-9 at 128 / 256 / 512 /
  1024 nodes. The floor is the linear reads of `sigma2` and
  `dlognudlogm` (a kink per cell), not the HOD shape; splitting at
  M_min / M_0 does not help. The linear read in a (<= 1.4e-5) dominates
  at 128 nodes: raising hdi buys nothing for `ngal` / `bgal` until
  `N_a` is raised.
- Mass integrals of the spectra (`p_gm`, `p_gg`; `p_mm` measured too
  before it moved to `future_port_unfinished/`): 64 / 128 / 256 / 1024 (the largest tabulated size) at hdi
  0 / 1 / 2 / >= 3. Chi2 ladder at 64 nodes vs 1024 (2026-09-29, HOD
  gg+gs 3x2pt, per-point fiducial): roman_real 8.6e-6, lsst_y1 8.2e-11;
  spectrum builds at 64 nodes: p_mm 0.04 s, p_gm 0.06 s, p_gg 0.05 s
  (4 threads, M2) - no longer the MCMC bottleneck. Halo a/k grid cuts
  measured the same day and REJECTED (budget-to-speed ratio): a-grid /2
  0.047 (roman), k-grid /2 0.070 roman / 0.131 lsst_y1, /4 either
  > 1; all three together 0.09 / 0.13 for ~0.1 s - most of the whole
  code's 0.2 budget for nothing. Re-open only if a profile shows the
  halo builds hot again (then with coarse-exact + spline upsample, not
  plain linear reads). Acceptance depends on the total numerical error
  budget, |delta chi2| < 0.2 across the code. The 128-node result (2.1e-8)
  is far below that threshold; additional numerical precision must be
  justified against its MCMC runtime cost. Keep at least 64 nodes.
  Source comments explain the physics and algorithm rather than quoting
  a knob's current value; the code and this reference hold the settings.
  P(k)-level convergence is slow at HIGH k only (the NFW ringing is
  sampled in ln M; worst over k up to 330 h/Mpc: I02 1e-4 / 8e-4 at
  512 / 256 nodes, and at 256 nodes `p_gg` moves by up to 2.4e-3), but
  the observables never reach those k. Measured 2026-09-29 (roman_real
  3x2pt data vector with HOD gg live, delta^T C^-1 delta vs the
  1024-node build, per-point generated fiducial): 512 -> 4.4e-11,
  256 -> 5.4e-10, 128 -> 2.1e-8; largest single datavector entry moves
  by 1.2e-8 (256) / 8.6e-8 (128) relative. 256 is therefore also
  viable if the builds must halve again. The y spectra (p_my/p_yy) are
  kept, not compiled, in future_port_unfinished/halo_tsz.c; restoring
  them needs their own chi2-level tolerance check. The integrands read `sigma2` and
  `dlognudlogm` by linear interpolation in ln M (a kink per cell,
  algebraic GL convergence); the gather bullet above is the way to
  make a small rule exact at every k.

Trapezoid rules, uniform in a log variable:

- `u_KS` Q and P in s = ln t: hdi 0: [-32, 4], step 0.2 (181 nodes);
  hdi 1: [-40, 4], 0.2 (221); hdi >= 2: [-40, 4], 0.1 (441). Dropped
  tail of Q ~ z e^{smin}: 3e-9 at z = ZHI = 2.5e5 (hdi 0), 1e-12
  (hdi >= 1); smax = 4 is generous (e^-164 at z = 3). End weights are
  not halved (the integrand is negligible at both ends).
- `tinker_alpha` in s = ln nu: [-90, 3.5], DS = 0.1 (936 nodes), exact
  to 2.9e-12 vs DS = 0.01 on [-200, 5] (the difference is the dropped
  lower tail).

### Halo-model IA (ia_tables; Fortuna et al. 2021)

- Kernel: closed-form l = 2, 4, 6 satellite-alignment profile (the
  derivation: scratchpad ia_kernel/KERNEL.md of 2026-09-29, to be moved
  into the halo.c header by a Fable pass), series branch below a
  per-edge switch. The l sum does NOT converge for the de-projected
  sin^-2 theta profile (prefactors -1.875, -1.875, -2.13, -2.39, ...):
  l <= 6 is F21's model choice, knob Ntable.halo_ia_lmax.
- f, g read: cubic Hermite on the nfw_ table with the exact slopes
  df/dln t = -t g, dG/dln t = t f (approved 2026-09-29 for the IA kernel
  only): f to 1e-14, g to 1e-11, against 5e-9 / 2e-8 for the linear read,
  which the l = 4, 6 closed forms amplify by up to 1e6. nfw_um keeps the
  linear read.
- gamma_hat vs mpmath: 2e-11 (l = 2), ~1e-9 worst (l <= 6, at the c = 20
  switch gap), ~1e-10 elsewhere.
- Refill cost ~15-40 ms at 4 threads for l <= 6 (M2); 10-15 ms for l = 2.
- Mass nodes below the IA HOD's M_0 carry no red satellites and are
  skipped in the kernel sums (about half of the spectra's mass range);
  mapping the rule from M_0 upward (as p_gm) is an open improvement.

### Tables and splines

- House spline (`spline_coeffs_uniform` + direct-index Horner)
  instances: `tinker_alpha`'s table, `ks_upsample1d` (the `u_KS`
  tables), the `p_*` coarse ln k grid; `sigma2`'s refill in cosmo3D.c.
- Natural-spline padding: S'' = 0 at the ends is wrong for a curved
  function (alpha''(0.25) = -2.9 gives a 2e-5 miss unpadded). The
  [1 4 1] rows damp an end error by 2 - sqrt(3) = 0.268 per interval,
  so PAD = 6 leaves 4e-4 of it. Spline the quantity that is read back
  (alpha = 1/I, not I). A `*_shape` helper evaluated at padding nodes
  must not clamp; the clamp lives in the caller.
- NFW f/G table (`nfw_table`): NFW_TASY = 50 is the smallest switch to
  the asymptotic series at table accuracy: the series stops after
  8!/t^8 (f) and 9!/t^8 (g), and the first omitted terms are 4e-11 and
  4e-10 at t = 50. Below NFW_TMIN = 1e-10, f and G are flat to 3e-9, so
  the clamp is safe. `nfw_pos` readers stay at ln t <= ln NFW_TASY, so
  the last-interval clamp extrapolates by at most one ulp.
- `u_KS` axes: the ln c axis is 1e2-1e4x more accurate than uniform c
  at equal nodes; PAD = 6 (ln z padded below only) gives 20-200x less
  edge error than no padding. ZHI = 2.5e5 must exceed
  k_max r_v(M_max) = 3e6 x 3.8e-3 ~ 1.1e4; the c clamp [0.05, 100] is
  a safety margin (c ~ 0.16 at a = 1/41).
- `hod_tables` builds all lens bins at once, so every bin's HOD must be
  set before the first call; `hod_.lim[0]` is a placeholder a for
  `HOD_nc`'s range check (the HOD does not depend on a).

### Accuracy tests must see the small scales (masks hide them)

HOD effects are important on small scales. Tests using conservative
scale cuts can miss numerical errors there and incorrectly support
lower accuracy settings. The data vector is evaluated with the mask:
the model is not computed at cut points, so delta is exactly zero there.
A production-mask test alone cannot assess accuracy on those scales.

Protocol for any halo/HOD/small-scale knob:
1. Evaluate the model with no cuts: a scratch dataset with ones.mask
   (and a diagonal covariance so the interface accepts it - the full
   unmasked covariance can be non-positive-definite and the interface
   aborts on it).
2. Score with chi2 (reference arm injected as truth) on the most
   aggressive POSITIVE-DEFINITE mask: re-admit cut points while the
   correlation matrix's smallest eigenvalue stays >= 1e-4 (raw
   condition numbers mix units; Schur-complement tests per point do
   not bound the smallest eigenvalue). 2026-09-29: roman_real's
   unmasked correlation matrix has an eigenvalue of -0.5; the
   aggressive set re-admits 85 of 165 cut points (small-scale gammat
   theta bins 0-3 and w); lsst_y1 is usable fully unmasked
   (correlation min eig 5e-4).
3. Report the production-mask number next to it, never alone.

Measured the same day (coarse-k spline step 8 on the HOD spectra):
lsst_y1 chi2 9e-9 under its production mask, 1.87 with no cuts.

### The b_mag = 0 a-range trap (cosmo2D/redshift_spline)

The lens bins' Limber a-range is gated on whether magnification is
active: `redshift_spline.c` tests `gbmag(0, ni) != 0` and extends the
range when it is. Crossing b_mag = 0 is therefore a DISCRETE quadrature
change, not a smooth limit: any accuracy or consistency sweep in the
magnification amplitude (e.g. the quadratic-structure and
second-difference oracle tests of test_hod_cell.py) must keep every
b_mag value nonzero, and a "magnification off" comparison arm uses a
tiny amplitude (1e-3) instead of 0. Symptom of getting this wrong: a
constant offset in the b_mag = 0 arm that mimics non-quadratic
structure (identical third-difference residual at every spacing).

### Accuracy protocols

Independent mpmath references; the record is the last run. Re-run when
the named knobs change.

- `tinker_alpha`: table vs exact Eq. 7 at 997 values of a. Record: max
  4.7e-8 (a = 0.2545), median 1.4e-9, set by the linear read of the
  ND = 4096 grid (alpha'' dx^2/8): to go lower raise ND, not NC. Re-run
  when ND, NC, PAD or the trapezoid window change.
- `u_nfw_c`: 3000 random (c, k, m), c in [0.05, 100], vs Eq. 81 of
  astro-ph/0206508 at 30 digits. Record: max 6.1e-7 (c < 0.1, where
  m(c) ~ c^2/2 amplifies the table error), median 1e-10. Re-run when
  `halo_nfw_n`, NFW_TMIN or NFW_TASY change.
- `u_KS`: 20000 random (c, z), c in [0.05, 100], z in [1e-6, 1.2e4],
  Gamma = 1.17, vs an mpmath-verified evaluator. Record: boost 1: max
  4.8e-6 of the local envelope (median 8e-7), 1.8e-5 relative where
  |u| > 1e-2; boost 2: 1.1e-6; hdi changes nothing (the floor is the
  table reads, not the quadrature). Re-run when a `u_KS` grid size,
  PAD, ZHI, a trapezoid window or a contour ray changes.

### Complex helpers (`ks_ctheta`, `ks_cg`)

- Complex ln(1 + x) near x = 0 (`ks_ctheta`):
  0.5 log1p(2 Re x + |x|^2) + i atan2(Im x, 1 + Re x), not clog(1 + x);
  below |x| = 1e-4 the 5-term Taylor series of ln(1 + x)/x (dropped
  term ~1e-21). Same recipe for any future complex profile helper.
- `ks_cg` takes theta^p as cexp(p clog theta) on the principal branch;
  |arg theta| < pi/2 is verified on both `u_KS` contour rays for c in
  [0.05, 100]. Changing the rays or the concentration range needs the
  check redone: a branch jump makes `u_KS` silently wrong.

### Constants under -frounding-math

- Compile-time physics constants (the Tinker bias coefficients at
  Delta = 200) are numeric literals, never log10/exp/pow expressions:
  under `-frounding-math` an inexact constant expression is not folded
  and runs on every call. Derive offline (mpmath, 40 digits; write 21
  significant digits) and guard with `#if Delta != 200` / `#error`.

### Warm-up and determinism

- Every lazily built static table is first called outside any parallel
  region. halo.c does this through one function, `halo_warmup`: it
  builds every lazy table halo.c reads, serially, before any threaded
  loop runs. A new lazy table is added to `halo_warmup`, not warmed at
  its call sites.
- Each table value is one serial sum over its quadrature nodes, and
  threading runs only across table nodes (`tinker_alpha`: each coarse
  value ye[i] is a serial sum over the NS nodes, threaded across coarse
  nodes). No cross-thread reductions, so no table depends on the thread
  count.

### Sizes and costs (snapshots; re-measure)

- `u_KS` at boost 1: NC = 40, NZ = 64, NW = 6, NY = 191, N1 = 60,
  PAD = 6; refinement MC = 12, MW = 32, MZ = 16, MY = 115, M1 = 70;
  dense grids 613 x 545 (S) and 613 x 1105 (each Q), ~13.8 MB total.
- `u_KS` at 7362e15, 4 threads: refill per Gamma change 1.1 ms (six
  upsamplings under `schedule(dynamic, 1)`, so the two Q jobs run side
  by side); one read 46 ns (cos, sin, 3 log, 3 exp, 5 table reads); the
  shared ln tau grid needs 563 `ks_cg` calls for P (36743 without it).
- `hod_tables` refill, 2026-09-29, 4 threads, 10 lens bins x
  N_a = 256: 3.9 ms.
- `tinker_alpha` build, 4 threads: 0.9 ms, once per process.

## Python code

The full contract is `references/python.md`; read it before touching
Python. The rules that are broken most often:

- **The reader** is a library user or physics student who reads C-like
  control flow but may not know advanced Python idioms.
- **Cold paths** (set-up, validation, file handling, figure layout) use
  explicit loops, plain `if` blocks, named intermediate variables and
  named arguments. No walrus operator, nested comprehensions, chained
  ternaries, or chained calls that mix selection, conversion and mutation.
- **Hot paths** (vectorized numpy, the per-evaluation body of a
  likelihood) stay vectorized; they get a comment with the mathematical
  reason or shape invariant, and a numerical check when they change.
- **No monkey patches**, in tests included. Pass the replacement as an
  argument, subclass, or use a separate process configured before import.
- **Validate before mutating**; a failure message says what failed, the
  observed value, the required condition and the repair. No silent
  fallback.
- **A changed return shape, tuple order or unit is an interface change**:
  every project's wrappers, notebooks and tests change with it.
- **Text explains the current code**: reasons, invariants, shapes, units.
  No names, dates, review history or "now does X".
- **`cosmolike_notebook_utils` never imports a project's compiled
  interface**; its plotting modules are pure numpy and matplotlib; cluster
  code goes in `_cluster` files and is imported explicitly.
- **A new plotting function copies the conventions of `plot_datavectors`**
  (signature order, `*_ref` ratio mode, `param` + `colorbarlabel` sweeps,
  glued panels, `show = None` returning `(fig, axes)`, malformed input
  printing one message and returning 0), and its figures are rendered and
  looked at in every mode before it is accepted.
- Python files keep their own indentation (4 spaces); the 2-space rule is
  for C.


## No C code only for tests

A C function whose only caller is a Python binding used by tests (a
`*_nointerp` point diagnostic, plus its header declaration, its
generic_interface/halo_wrapper wrapper and one binding per project) is
maintenance surface in the core with no production value. Delete all of
it and test the production function — the cached table — against an
independent Python (numpy/mpmath) reference:

- at the table's own nodes (interpolation weights 0/1, so exact node
  values: checks the quadrature);
- between nodes (checks the interpolation).

`*_nointerp` functions with real C callers (table fills, other
integrands) stay until deep unrolling folds them into their owner.

## When a request is unclear

If a request is ambiguous, or its interpretation repeatedly needs
correction, do not act on a guess. Before asking for another explanation,
launch a subagent with `model: "fable"`, supplying the request verbatim,
the relevant code paths and the current interpretation. Ask it to identify
the intended scope, exclusions and concrete next action. Act on that
interpretation and state it to the requester in one line.
Clear requests need no consult.

## Patch review checklist

Reject or push back unless all of these hold (details in
`references/pitfalls.md`):

- [ ] Frozen chi2 within 0.2 of the reference (printed to four decimals);
      full unmasked data vector bitwise equal to the reference build, or
      within a stated tolerance where the change documents why it cannot
      be bitwise; determinism sweep clean across thread counts.
- [ ] Builds and runs clean in both supported modes (strict and DEBUG);
      DEBUG sanitizers quiet. Aggressive mode remains retired.
- [ ] New hot loops inside `collapse(2)` regions use local `restrict` pointers.
- [ ] No flat `memset`/`memcpy` over padded multi-dim allocations; uses
      `zero*d`.
- [ ] `static inline`, not `inline`.
- [ ] No new lazily-initialized `static` state without real synchronization
      (double-checked locking on a plain `static int` sentinel is a known,
      previously-shipped race — see pitfalls).
- [ ] FFTW plans created once, cached, and creation is serialized; sizes
      passed through `next_fft_size`.
- [ ] Production SIMDe remains unconditional; any scalar comparison stays
      in external tests, without restoring a retired fallback switch.
- [ ] Comments explain why, not what.
- [ ] PR cites `perf stat -r 3` numbers (mean ± stddev), never single-eval
      timings; claims of "no perf change" are backed by counters, not vibes.
- [ ] Naming and structure of the surrounding code preserved; 2-space indent.
- [ ] Python in the patch passes the checklist of `references/python.md`
      (Section 12).

## Communication norms for reviews

Flag every issue once, clearly and concretely, with the failing case or the
exact fix. If the author explicitly decides to keep something as-is, note it
"for the record" and move on — do not re-litigate. When the reviewer (human or
Claude) makes an error, say so plainly and correct the record. No hedging, no
filler: code and numbers over prose.
