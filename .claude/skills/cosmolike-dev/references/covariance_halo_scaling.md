# Shared covariance halo tables

Implemented and checked on 2026-10-04. This changes scheduling and omits
unused work; it does not change a physical or numerical accuracy setting.

## Remove work that the response never consumes

The isotropic SSC prescription differentiates the two-halo matter power
I11(k)^2 P_linear(k) with respect to ln(k). Its centered finite difference
needs I11 at k exp(-h) and k exp(+h). The one-halo power and its response
use higher moments at the central k, already computed for the connected
trispectrum. No higher moment at either displaced endpoint is used.

`halo_moments_cov` therefore accepts a null pair-output pointer for an
I11-only request. The Python boundary exposes `pair_moments=False` and
returns `(I11, None)`. The default remains the complete five pair moments.
Both paths use the same mass quadrature, profiles, completion and SIMD
sum. This is an omission of unused work, not a new halo approximation.

The previous response call also repeated the same scale factor in one
row per central k. Moving all derivative endpoints onto their parent
scale-factor row lets the abundance, bias and concentration be prepared
once per shell, while retaining each physical wavenumber.

For the DES pilot's 16 central wavenumbers, the old central-plus-response
calls prepared mass statistics for 17 scale-factor rows per shell. The
new calls prepare two. The number of distinct profile rows falls from
64 to 48 because the derivative's unused central point disappears.
Pair integrations fall from 136+16*6=232 to 136 per shell. These are
operation counts for that grid, not measured runtime speedups.

## Group independent shells for OpenMP

At a fixed shell, the physical wavenumbers are (ell+1/2)/chi. Different
shells still require their own power spectra and halo statistics. Grouping
several shells changes how these independent rows reach C; it does not
interpolate along the changing Limber wavenumber or share a physical
value between different redshifts.

Mass-statistics loops distribute (a,M), profile loops distribute (a,k)
and, for small requests, disjoint mass chunks. Moment loops distribute
(a,k) or (a,pair), leaving each complete mass sum on one worker. Grouping
shells reduces repeated quadrature construction and parallel-region
launches, and supplies more independent work to eight workers.

The central five-moment table and the displaced I11 table are held only
for one bounded shell group. Every final power read, tree-level angular
integral, projection and radial output keeps its previous order. No
static covariance cache, extra precision parameter or MPI layer is added.

## Measured group sizes and selection

Apple M2 Pro, DES pilot: 704 shells, 16 central wavenumbers and 128
Gaussian nodes on each of eight mass panels. BLAS used one thread.
Initialization and reference construction were excluded. Each case had
one excluded warm-up and three timed calls, with no other numerical job.
The component includes central moments and derivative endpoints, not
power reads, tree averages, projections or a complete covariance.

| Threads | Previous calls (s) | I11-only, group 8 (s) |
|---:|---:|---:|
| 1 | 4.895885 | 1.874365 |
| 2 | 2.671991 | 0.991379 |
| 4 | 1.519511 | 0.529803 |
| 8 | 1.277814 | 0.352525 |

Every central moment and derivative endpoint is bitwise unchanged. At
eight threads the new preparation is 3.62 times faster than the previous
calls, and scales by 5.32 from one to eight threads. Four to eight gives
1.50. This is an Apple component result, not an x86 measurement.

At eight threads, I11-only group sizes 1/2/4/8/16/32 took
0.7420/0.5169/0.4194/0.3525/0.3194/0.2908 s. Retain group 8: group 32
saves another 0.0617 s at this pilot resolution but quadruples the profile
scratch. Scratch grows with both the mass rule and k count as accuracy
rises. Eight is a bounded-memory tradeoff, not a claim that it minimizes
runtime for every grid or architecture. No user precision knob is added.

## Complete DES joint comparison

The 2812-entry, boost-1 joint forecast retains its 48 defined Y null rows
and the previously documented physical approximations. Each thread count
has an excluded warm-up and three complete warm-reader recomputations;
CAMB and initialization are excluded. All G, SSC, cNG, total and mean
entries are bitwise equal to the saved pre-optimization reference.

| Threads | Full matrix mean (s) | Sample std. (s) |
|---:|---:|---:|
| 1 | 10.714666 | 0.122955 |
| 2 | 6.610318 | 0.014528 |
| 4 | 4.599150 | 0.021248 |
| 8 | 4.189486 | 0.294833 |

The earlier Gaussian-only eight-thread mean was 5.1310 s. The new mean
is 4.1895 s, with appreciable repetition scatter. Shared tables average
1.6006 s, versus 2.5714 s in that earlier full run. Complete four-to-eight
scaling remains only 1.10: this change does not finish the scaling work.

## Verification and didactic review

- LSST's complete covariance suite: 74 passed, including two new tests
  for I11-only and grouped derivative endpoints at 1/2/4/8 threads.
- An isolated O0 UBSan/float-division build: 24 cases passed against the
  optimized DES interface. It covers zero and high k, odd/even lengths,
  single/multiple scale factors and untouched output-row sentinels.
- All benchmark outputs and complete joint components match bitwise.
  No frozen reference was changed.
- Changed C/C++/header files respect 80 columns; Python respects 90.
  Diff checks and Python parsing pass.

The review explains why only I11 is needed at displaced k, why each
shell keeps its own physical wavenumbers, which moments remain central,
and how every mass sum retains its order. New array inputs are listed by
physical role. No new SIMD operation is introduced; the existing
lane-by-lane explanations remain. The public README explains the optional
I11-only interface without referring to internal test files.

The complete all-project baseline is in covariance_gaussian_scaling.md.
Rerun every covariance suite and frozen likelihood example after the
remaining shared projection changes, before completing this work.

Internal records live in test/covariance_reference/: the shell-group
benchmark, I11 debug harness, complete DES before/after arrays and timing
JSON. They are development records, not public prerequisites.
