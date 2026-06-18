---
name: cosmolike-dev
description: Development, optimization, review, and debugging practices for the CosmoLike/Cocoa C codebase (cosmo2D.c, pt_cfastpt.c, redshift_spline.c, cfastpt.c, cfftlog, IA.c, basics.c). Use this skill whenever working on CosmoLike or Cocoa C code in any way — writing or reviewing patches, optimizing hot loops, adding OpenMP/SIMD, replacing GSL calls, touching FFTW/FAST-PT code, debugging non-deterministic chi2, benchmarking with perf, or evaluating performance claims. Also use it when terms like Limber, non-Limber, TATT, NLA, 3x2pt, FAST-PT, Legendre summation, or tomographic C_ell appear in a C-code context, even if optimization isn't mentioned explicitly.
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

Before doing any Docker work — Dockerfile edits, GPU-container debugging, 
image size diagnosis, or container build failures — 
read `references/docker-reference.md`. It contains 
the methodology and conventions for editing Dockerfiles, 
the GPU stack model, dependency-resolution patterns, and image-size diagnostics.

## Core principles

1. **Correctness is non-negotiable.** In the default (strict IEEE-754) build,
   chi2 must match the reference value exactly — to the last printed digit —
   before and after a change. A change that alters chi2 is a bug until proven
   to be a deliberate, documented accuracy improvement.
2. **Measure, never guess.** Every optimization starts with a profile and ends
   with `perf stat -r 3`. Single-evaluation timings on shared nodes (SeaWulf,
   NVWulf) are noise; never accept or report them as evidence.
3. **Algorithms first, SIMD last.** The dominant wins in this codebase came
   from precomputation and factoring redundant work out of loops (cosmo_nodes,
   cfftlog forward-FFT hoisting, trapezoid factorization), not from intrinsics.
   Only vectorize a loop after its algorithm is already minimal.
4. **One change at a time.** Each change is validated and measured in
   isolation. Never bundle a refactor with an optimization in one commit.
5. **Every risky optimization gets an escape hatch.** Preprocessor guards
   (`COSMO2D_NOT_USE_SIMD`, `DONT_NZ_FAST_SUMBSAMPLE`, ...) must allow falling
   back to the simple reference implementation. The fallback is also the
   ground truth for debugging.
6. **Preserve existing conventions.** Keep the file's variable naming, struct
   layout, and code structure when modifying. Renames happen only as their own
   dedicated, mechanical commits (e.g. `zdistr_photoz` → `nz_source_photoz`).
7. **Determinism is a correctness test.** If chi2 varies run-to-run or with
   `OMP_NUM_THREADS`, there is a race or uninitialized memory. Full stop. Do
   not proceed until it is found.

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

- **chi2 exact match** against the recorded reference in the default IEEE
  build. Print enough digits (e.g. `9.67405...`) that drift is visible.
- **Determinism sweep:** repeat the evaluation several times within one
  process and across processes, at `OMP_NUM_THREADS=1` and a high count
  (e.g. 8 or node-width). Any variation = race / uninitialized memory.
- **All three build modes** (Makefile):
  - `COSMOLIKE_DEBUG_MODE`: `-O0` + sanitizers. Catches linkage bugs
    (`inline` vs `static inline`), out-of-bounds writes, double frees.
  - default: strict IEEE-754 (`-fno-fast-math -frounding-math
    -ftrapping-math -fsignaling-nans`) with LTO + unrolling. This is the
    bit-reproducibility reference.
  - `COSMOLIKE_AGGRESSIVE_MODE`: `-ffast-math -flto -funroll-loops`. chi2 may
    differ in late digits; posteriors must be statistically identical (this
    was verified for one full model — re-verify if your change touches
    summation order).
- **Both Limber paths** (C_ss and C_gs), **both real-space projections**
  touched (xi_pm, gammat, w_gg), **both theory paths** (emulator and exact
  CAMB), and **both IA branches** (NLA and TATT) when relevant.
- **FFT/FAST-PT changes:** call the function at least twice in the same
  process. Plan-caching and static-state bugs only show on the second call.
- **Non-Limber changes:** verify the per-bin early-exit still converges to the
  same C_ell as the no-early-exit fallback.

## Benchmarking standard

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
- Mind IPC interpretation: this workload is memory-bound (~1.1 IPC, ~25% LLC
  miss rate is normal). Low IPC is not by itself a problem to "fix".
- Landmarks (June 2026 snapshot; re-measure, don't trust): full benchmark
  ~111s; per-eval exact CAMB ~165ms, emulator path ~259ms; TATT `get_FPT_IA`
  ~72ms dominates the emulator path; non-Limber ~12ms (was ~150ms).
  Hot path: `xi_pm_tomo` → Limber fill → Legendre summation.

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
  64-byte cache-line padded rows). Zero only through `zero1d/2d/3d/4d`.
  **Never** a flat `memset` over a padded multi-dim allocation (see pitfalls).
- Every SIMD/fast-path block is wrapped in a preprocessor guard with the slow
  reference path in the `#else`/guarded branch.
- Fortran (custom CAMB): all modifications fenced with `!VM BEGINS` /
  `!VM ENDS` so they survive upstream rebases.
- Function naming for the established decompositions: `<name>_work` for
  batched tomographic-block computation, `<name>_fill` for gather-based
  interpolation table fills, `int_for_<name>_core` for scalar integrand cores
  callable from both SIMD bodies and scalar tails.

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
- `basics.c` — allocators, `zero*d`, interpolation utilities.

## Patch review checklist

Reject or push back unless all of these hold (details in
`references/pitfalls.md`):

- [ ] chi2 identical to reference in default build; determinism sweep clean
      across thread counts.
- [ ] Builds and runs clean in all three modes; DEBUG sanitizers quiet.
- [ ] New hot loops inside `collapse(2)` regions use local `restrict` pointers.
- [ ] No flat `memset`/`memcpy` over padded multi-dim allocations; uses
      `zero*d`.
- [ ] `static inline`, not `inline`.
- [ ] No new lazily-initialized `static` state without real synchronization
      (double-checked locking on a plain `static int` sentinel is a known,
      previously-shipped race — see pitfalls).
- [ ] FFTW plans created once, cached, and creation is serialized; sizes
      passed through `next_fft_size`.
- [ ] Preprocessor fallback guard exists and the fallback still works.
- [ ] Comments explain why, not what.
- [ ] PR cites `perf stat -r 3` numbers (mean ± stddev), never single-eval
      timings; claims of "no perf change" are backed by counters, not vibes.
- [ ] Naming and structure of the surrounding code preserved; 2-space indent.

## Communication norms for reviews

Flag every issue once, clearly and concretely, with the failing case or the
exact fix. If the author explicitly decides to keep something as-is, note it
"for the record" and move on — do not re-litigate. When the reviewer (human or
Claude) makes an error, say so plainly and correct the record. No hedging, no
filler: code and numbers over prose.
