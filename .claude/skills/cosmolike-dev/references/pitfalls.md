# CosmoLike pitfalls: bugs we actually shipped or nearly shipped

Every entry here happened. When reviewing a patch, scan this list against the
diff. When debugging, match the symptom column first — most "mystery" failures
in this codebase are reruns of one of these.

## 1. ~2x slowdown after "simplifying" a SIMD helper into an inline loop

- **Symptom:** wall time roughly doubles on a Legendre-summation loop after
  replacing a `restrict`-qualified helper with an inline
  `#pragma omp simd reduction` loop indexing `Pl[i][l]`, `Cl[nz][l]` directly.
- **Cause:** pointer aliasing. Inside a `collapse(2)` region GCC cannot prove
  the rows of two pointer-to-pointer arrays don't overlap and emits
  reload-checking code.
- **Fix:** hoist `const double* restrict pl = Pl[i];` etc. and index through
  the locals. Declaring the locals but still indexing `Pl[i][l]` in the body
  does nothing (also a real mistake that was made — the compiler only honors
  `restrict` on the accesses that go through the qualified pointer).

## 2. Non-deterministic chi2 (run-to-run, or thread-count dependent)

Two independent root causes have produced this exact symptom:

- **2a. Uninitialized FFTW work-buffer gap.** `get_FPT_IA`'s `fb` buffer had
  an unwritten region between the data and the padded end; FFTW read it.
  chi2 differed across runs depending on heap garbage. **Fix:** `memset` /
  explicit zeroing of the full allocated buffer (here a flat buffer, so
  memset is fine) before the transform. **Rule:** any buffer handed to FFTW
  is fully initialized, including padding.
- **2b. Double-checked locking race on a `static int N[]` sentinel** in
  `ZL()`/`ZS()` (`redshift_spline.c`). Plain reads/writes of a static
  sentinel under OpenMP are a data race: threads observed the sentinel set
  before the table writes were visible. **Fix:** proper synchronization
  (single-threaded init before the parallel region, or `omp critical` with
  re-check, or atomics with correct ordering). **Rule:** no new
  lazily-initialized `static` state in code reachable from parallel regions
  without real synchronization. "It's just an int flag" is exactly the bug.

Triage when chi2 wobbles: (1) rerun at `OMP_NUM_THREADS=1` — if it
stabilizes, it's a race (2b-type); if it still wobbles, it's uninitialized
memory (2a-type). (2) DEBUG build with sanitizers. (3) Diff against the
preprocessor-guard fallback path.

## 3. Flat `memset`/`memcpy` over padded multi-dim allocations

- **Symptom:** corruption, sanitizer reports, or silent wrong values near row
  boundaries after zeroing a `malloc2d/3d/4d` array with one
  `memset(a[0], 0, n1*n2*sizeof(double))`.
- **Cause:** rows are 64-byte padded; the logical array is not contiguous and
  the flat size arithmetic is wrong. A related over-allocation bug existed in
  `malloc2d` itself (padding arithmetic).
- **Fix:** `zero1d/2d/3d/4d` from `basics.c` — they exist precisely for this.
  Review rule: any `memset`/`memcpy` whose target came from `malloc2d+` is
  wrong until proven otherwise.

## 4. Bare `inline` breaks DEBUG builds

- **Symptom:** undefined-reference link errors only in `COSMOLIKE_DEBUG_MODE`
  (`-O0`).
- **Cause:** C99 `inline` without `static` provides no external definition
  when the compiler doesn't inline (which it doesn't at -O0).
- **Fix:** `static inline`, always.

## 5. Pointer comparison where content comparison was meant

- **Symptom:** FAST-PT self-convolution path taken (or not taken) incorrectly;
  subtly wrong P(k) terms.
- **Cause:** `if (a == b)` on array pointers to decide "same input spectrum".
- **Fix:** compare configuration/contents. Pointers tell you about storage,
  not physics.

## 6. Double free after changing an init/allocator signature

- **Symptom:** crash in teardown (or under sanitizers) after removing an
  `init` parameter / changing ownership in a refactor.
- **Cause:** two sites both believed they owned the buffer.
- **Fix/rule:** any patch that changes an allocation signature must audit
  every `free` of that object, and run the DEBUG build to completion —
  teardown is part of the test.

## 7. "Round" FFT sizes that are terrible FFT sizes

- **Symptom:** FAST-PT mysteriously slow; profile dominated by FFTW.
- **Cause:** transform length like 10201 (= 101^2) has large prime-ish
  factors.
- **Fix:** pad through `next_fft_size` (10201 → 10240). Rule: never hand a
  raw data length to FFTW; always pass it through `next_fft_size` and zero
  the padding (see #2a).

## 8. FFTW plan churn and plan-creation races

- **Symptom:** per-call planning overhead dominating; or crashes/garbage when
  multiple threads create plans.
- **Cause:** creating plans per (bin, config, ell) call; FFTW plan creation
  is not thread-safe.
- **Fix:** static plan cache keyed by size, creation serialized, plans
  reused. Validate with two calls in one process (first call exercises
  creation, second exercises the cache).

## 9. Grid alignment off-by-one in fine/coarse grids

- **Symptom:** small but real chi2 shift after a resampling optimization.
- **Cause:** fine grid sized `x*N_a` instead of `x*(N_a - 1) + 1`, so coarse
  nodes don't coincide with fine nodes and the interpolant differs at the
  reference points.
- **Fix:** `Na = x*(N_a - 1) + 1`. Review any grid-refinement factor for this
  pattern.

## 10. Uniform-grid fast paths applied to non-uniform grids

- **Symptom:** wrong interpolation values after copying the n(z) direct-index
  trick to another table.
- **Cause:** the `idx = (x - x0)*inv_dx` lookup assumes a uniform grid. The
  chi(a) grid is only piecewise uniform — that's why `a_chi` was deliberately
  left on binary search.
- **Fix/rule:** before converting a lookup, verify grid uniformity; if it's
  piecewise, either build the multi-segment dispatch or leave it alone and
  write down why.

## 11. Trusting single-evaluation timings

- **Symptom:** optimization "wins" that vanish, or regressions that were
  never real.
- **Cause:** shared HPC nodes; one evaluation is noise.
- **Fix:** `perf stat -r 3` on the 1000-eval benchmark is the only accepted
  evidence. Instruction counts and FP-op counters are far more stable than
  wall time — quote both.

## 12. Sweeping renames/restructures mixed into functional patches

- **Symptom:** unreviewable diffs; regressions hidden inside style noise.
- **Cause:** combining a rename (e.g. `pf_photoz` → `nz_lens_photoz`) with an
  algorithmic change.
- **Fix/rule:** renames are separate, mechanical commits. Functional patches
  preserve surrounding naming and structure exactly.
