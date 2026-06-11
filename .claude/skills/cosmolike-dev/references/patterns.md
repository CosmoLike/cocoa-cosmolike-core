# CosmoLike optimization patterns

The established, validated transformations. Each one was measured on the
1000-evaluation `roman_real.combo_3x2pt` benchmark and verified chi2-exact.
Apply them in this order of priority — the list is roughly sorted by payoff.

## 1. Node precomputation (`cosmo_nodes`)

The single biggest class of win. Inside Limber loops, anything that depends
only on the integration node `(a, z, chi)` and the tomographic bin — IA
kernels, lens efficiencies `W_kappa`, growth factors, `n(z)` values — gets
evaluated once into per-node tables (e.g. `PK[zl][i][p]`, window-function
tables) before the ell loop, instead of per `(ell, node)`.

- Structure: a `cosmo_nodes` struct holding the tables, filled by a `_work`
  function, consumed by the per-ell integrand.
- Applied to the C_ss and C_gs paths. **Not yet applied to C_gg** — its perf
  share is slightly inflated because of this; it is the natural next target.
- When reviewing: any call inside an ell loop whose arguments don't contain
  ell is a candidate. Ask why it isn't hoisted.

## 2. Batched `_work` functions with `collapse(2)`

Compute whole tomographic blocks at once:

```c
#pragma omp parallel for collapse(2) schedule(static)
for (int nz=0; nz<NSIZE; nz++) {
  for (int i=0; i<Ntable.Ntheta; i++) {
    ...
  }
}
```

`collapse(2)` over (bin, angle/ell) gives enough parallel slack for full
nodes. But it creates the aliasing trap (pattern 3) — the two always travel
together.

## 3. Thin inner loops: `omp simd reduction` + local `restrict`

For dot-product-shaped inner loops (Legendre summation
`sum += Pl[i][l]*Cl[nz][l]`), do NOT hand-write intrinsics and do NOT index
the 2D arrays directly. The validated idiom:

```c
const double* restrict pl = Pl[i];
const double* restrict cl = Cl[nz];
double sum = 0.0;
#pragma omp simd reduction(+:sum)
for (int l=lmin; l<Ntable.LMAX; l++) {
  sum += pl[l]*cl[l];
}
```

History, so nobody relearns it: hand-written SIMDe versions
(`simd_dot_product`, `simd_xipm_dot_product`, `simd_array_sum`,
`simd_horizontal_sum`) were written, then deleted, once it was proven that the
~2x gap between intrinsics and the plain pragma was **pointer aliasing**, not
FMA latency (1-, 2-, and 3-accumulator intrinsic variants all timed identical
~0.106s; the inline pragma without `restrict` was ~2x slower; with `restrict`
locals it matched). GCC cannot prove `Pl[i]` and `Cl[nz]` don't overlap
through pointer-to-pointer indirection inside a `collapse(2)` region.

Multi-accumulator unrolling is NOT needed — the workload is memory-bound and
the hardware hides FMA latency behind cache misses.

## 4. Gather-based `_fill` interpolation: the one place intrinsics survive

GCC will not auto-emit gather instructions. Interpolation-table fill loops
(`C_ss_tomo_limber_fill`, `C_gs_tomo_limber_fill`, ...) that load from
scattered indices keep hand-written SIMDe gather + FMA bodies, with a comment
explaining the auto-vectorization blocker, a scalar tail that calls the same
`int_for_*_core` scalar function, and a `COSMO2D_NOT_USE_SIMD` fallback.

Rule of thumb: contiguous-access loop → pragma + restrict; gather-access
loop → SIMDe intrinsics. Confirm with `-fopt-info-vec-all` either way.

## 5. Heavy integrand loops (TATT)

When the loop body is large (the TATT EE/BB integrand has ~25 input arrays),
use `#pragma omp simd reduction(+:sum_EE, sum_BB)` on the inner per-`p` loop
with `collapse(2)` on the outer `(i, k)` loops, hoist all row pointers to
`restrict` locals, and keep the physics in scalar
`int_for_C_ss_tomo_limber_tatt_EE_core` / `_BB_core` functions called from
both the vector body and the tail. Intrinsics bought only ~2.3% here — not
worth the maintenance. Heavy bodies hide vector overheads; thin bodies don't.

## 6. GSL elimination

Two validated replacements:

- **Binary search → uniform fine grid + direct index.** `n(z)` lookups
  (`nz_source_photoz`, `nz_lens_photoz`) resample onto a uniform fine grid at
  init; the hot path is then `idx = (z - z0)*inv_dz` plus **linear**
  interpolation — pure linear was tested sufficient, no spline needed.
  Fallback guard: `DONT_NZ_FAST_SUMBSAMPLE`. Caveat: this only works on
  uniform grids. The `a_chi` lookup was deliberately NOT converted because
  chi(a) is only piecewise uniform and the multi-segment dispatch wasn't
  worth the 3.5% — leave it unless that calculus changes.
- **Per-entry Gauss-Legendre quadrature → factored cumulative trapezoid.**
  Lens efficiency g(z_l) = ∫ dz_s n(z_s) (1 - chi_l/chi_s) factors (flat
  cosmology) as P(z_l) - chi(z_l)*Q(z_l) where P, Q are single cumulative
  integrals of n(z_s) and n(z_s)/chi(z_s). This collapsed
  O(nbin x N_a x szint) `n(z)` evaluations to O(nbin x N_a) in
  `g_tomo`/`g2_tomo`/`g_lens`. Grid alignment matters:
  `Na = x*(N_a - 1) + 1` so the fine grid's nodes coincide with the coarse
  grid's (an off-by-one here was a real bug).

## 7. FAST-PT (`pt_cfastpt.c`)

- Collapse N separate `fastpt_scalar` calls into one `J_abl` / `J_abl_ar`
  dispatch over the term table (done for `get_FPT_IA`: five calls → one; same
  pattern in `get_FPT_bias`). One pass over the FFT machinery, not five.
- **Static FFTW plan caching:** plans are created once per size, stored in
  static state, reused. FFTW planning is not thread-safe — serialize creation.
- **FFT-friendly sizes:** pad transform lengths through `next_fft_size`
  (e.g. 10201 → 10240 gave a large win — 10201 = 101^2 is an awful FFT size).
- In the self-convolution path, equality of input spectra must be decided by
  comparing **contents/configuration, not pointer values** (pointer comparison
  was a shipped bug).

## 8. Non-Limber (`cfftlog`)

- The forward FFT of the source function is ell-independent: hoisted into
  `cfftlog_ells_cocoa0`, called once before the convergence loop instead of
  per (bin, config, ell).
- Per-(bin, config, ell) FFTW plans replaced with shared `SIZE2` plans.
- Per-bin early convergence exit: once a bin's non-Limber correction
  converges, stop iterating it (validate against the no-early-exit path).
- Gamma-function recurrence Γ(z+1) = zΓ(z) to step between adjacent ell
  instead of fresh Lanczos evaluations.
- Net: ~150ms → ~12ms.

## 9. Memory discipline

- All multi-dim arrays through custom `malloc1d/2d/3d/4d`: `posix_memalign`,
  rows padded to 64-byte cache lines (kills false sharing between OpenMP
  threads writing adjacent rows).
- Padding means the logical array is NOT contiguous: zero with
  `zero1d/2d/3d/4d`, never flat `memset` (see pitfalls #3).
- Precompute layouts so the hot loop walks contiguously in the fastest index
  (`PK[zl][i][p]` with `p` innermost).

## 10. When to stop

- IPC ~1.1 with ~25% LLC miss rate is the steady state of this workload —
  it's memory-bound. Don't chase IPC.
- Check remaining hot-spot percentages against bin combinatorics before
  declaring them "inefficient" (xi_pm: 36 pairs x 2; gammat: ~55; w_gg: ~8).
  If the ratio matches the combinatorics, the code is already balanced.
- A rejected optimization with a documented reason (like `a_chi`) is a result.
  Record it so it isn't re-attempted.
