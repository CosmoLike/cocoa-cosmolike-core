# Covariance optimization review, 2026-10-03

This review follows the module-wide didactic pass. It profiles small
Roman components and tests proposed changes externally; no new production
accuracy setting, interpolation API or compiler mode is introduced.
Experiments use the compiled Roman Real interface, strict Clang builds,
explicit OpenMP and single-threaded OpenBLAS on Apple M2 Pro.
Production chains target x86 supercomputers. These laptop measurements
cannot decide x86 SIMD gains: AVX2/AVX-512 widths and gather/FMA mappings,
register pressure, cache and memory bandwidth differ from Apple NEON.
Keep unconditional SIMDe. Re-measure architecture-sensitive choices on
the actual x86 CPU and strict production compiler before selecting them.

The owner subsequently confirmed that the laptop was quiet enough.
The completed measurements and the selected halo-loop adjustment appear
in the final section below; they supersede the earlier timing deferrals.

## Patterns taken from the existing engines

`pt_cfastpt.c::fpt_regrid` separates the expensive FFTLog sampling from
the dense output grid. Its uniform log-k grid gives direct indices;
caller-owned spline scratch, Horner evaluation and shared offsets avoid
searches and repeated setup. Signed outputs are interpolated as values.
The spline build fills a dense table, whose hot readers stay linear.

`halo.c` and `cosmo2D.c` use the same separation between exact integration
nodes and dense interpolation nodes. Padding places the natural spline's
zero-second-derivative endpoints outside the physically queried range.
The appropriate object to spline depends on its smoothness and sign;
copying only a coarse/fine node ratio does not establish accuracy.

`cosmo2D.c::limber_fill_interp` shares direct grid indices and fractions
across tables, then performs four linear queries with SIMDe gathers and
FMA. The real-space transform fills the spectra in batches before the
weighted sums. Its `legendre_sums` uses 4 spectra by 4 angular bins, and
`legendre_sums_xipm` uses 2 shear pairs by 4 angular bins, to reuse input
loads across independent sums. Each sum retains its multipole order.
The covariance projection already applies this reuse principle with
explicit SIMDe. Do not infer vectorization merely from an OpenMP pragma.

For covariance tables the intended sequence is:

1. Perform expensive exact calculations on a validated coarse grid.
2. Build cubic coefficients once and populate a dense table.
3. In the hot path, calculate direct uniform-grid indices and linearly
   interpolate adjacent dense values, using SIMDe for bulk reads.

Validation must include off-grid queries in step 3. Agreement at the
dense nodes tests construction but misses linear-lookup error.

## Existing hotspot evidence and prepared threading measurements

The earlier Roman pilot identified the angular-input/integration stage as
dominant: 666.9 ms of a 700.0 ms shared row at its largest sampled setting.
That included power interpolation, geometry, Python allocation and rule
construction; it did not isolate the C perturbation integrator.

The prepared profile separates those stages for 128 multipoles (8256 unique
pairs), 5120 angle nodes and z=0.75. It compares the existing serial
power reader with disjoint OpenMP batches at 1, 2, 4 and 8 threads, and
the existing angular kernel with compiler unroll factors 2 and 4.
SIMDe arithmetic is already present; disassembly contains native vector
FMA, multiplication, addition and division instructions.

At the owner's earlier request, this pass prioritized interpolation accuracy
and the negative-mode investigation. The queued timing job was stopped
before it launched. A lookup-validation preflight invoked timing helpers
while project tests were running; those elapsed times were discarded.
No performance result was retained from that attempt. Run the prepared profile
only on a quiet machine, one benchmark at a time, after all competing
computational jobs have exited. Do not use noisy Mac results to reject
the default SIMDe design for production x86.
After all 435 project tests passed and their processes exited, two CPU
activity snapshots five seconds apart still showed substantial WebKit and
desktop activity. No benchmark was launched. The snapshots are retained
locally in `results/quiet_machine_check.json`; production timing remains
deferred until the host is quiet.

## Cubic construction in physical wavenumber

The external `review_optimization.py accuracy` first tests uniform ln-k
tables spanning the k values corresponding to ell=2--100000, at z=0.25,
0.75 and 1.5. The dense comparison has 129 nodes. Coarse tables add three
padding nodes at either end. The five halo moments use 512 mass nodes;
angular averages use 2560 angle nodes, separately refined to 5120.

For positive halo moments, spline their logarithms, preserving the two
different orientations of I13 when expanding the triangular table.
The worst errors over the three redshifts and five moments are:

| Coarse spacing / target spacing | Exact nodes including padding | Maximum error / reference peak | Relative norm error |
|---:|---:|---:|---:|
| 2 | 71 | 2.19e-5 | 5.57e-6 |
| 4 | 39 | 6.03e-4 | 1.84e-4 |
| 8 | 23 | 7.96e-3 | 2.58e-3 |

These are construction tests, not a full covariance validation. Even the
first setting has relative errors up to 0.0067 in entries larger than
1e-6 of the reference peak. Projection determines whether those matter.

Applying a raw two-dimensional value spline to AvgP/AvgB/AvgT on the
same Cartesian K,Q grid fails: even spacing ratio 2 gives a worst peak
error of 8.47%. The near-diagonal angular structure must be resolved;
smooth halo moments do not imply smooth perturbative averages on the
same grid. Separate angular-rule refinement changes peak-normalized
values by at most 3.22e-7 in these rows, so it cannot explain that failure.

## Cubic construction in scale factor, then linear lookup

At fixed K,Q, test 65 a samples over 0.25--0.875, retaining all 561 pairs
from 33 k nodes. Removing a^2, a^4 and a^6 reduces dominant growth, but
the actual pivot power is more effective in these cosmologies. Define

`g(a) = P_lin(k_pivot,a) / P_lin(k_pivot,a_ref)`.

Construct coarse tables for AvgP/g, AvgB/g^2 and AvgT/g^3, spline these
residuals, and restore the factors at dense nodes. The remaining scale
dependence is still calculated from the supplied massive-neutrino power
tables; scale-independent growth is not imposed. The pivot is the middle
k node and a_ref=1/1.75. Twenty-one exact rows include two padded rows at
each end; seventeen lie in the target interval.

The fiducial dense-node test has maximum errors divided by each role's
own peak of 6.74e-7 (AvgP), 4.96e-7 (AvgB), and 2.45e-6 (AvgT).
Repeating at m_nu=0.30 eV gives a worst peak error of 3.68e-6; at
w0=-0.8, wa=0.2 it gives 2.52e-6. No significant entries change sign.
Relative errors in small entries are larger; these are not FoM bounds.

The separate `check_dense_linear_lookup.py` tests the owner's complete
construction/lookup sequence. It compares 48 off-grid Gauss-Legendre
query points with direct integrals, using the same 21 exact build rows:

| Dense a nodes | AvgP peak error | AvgB peak error | AvgT peak error |
|---:|---:|---:|---:|
| 65 | 3.86e-5 | 5.47e-5 | 1.20e-4 |
| 257 | 5.04e-6 | 8.63e-6 | 1.05e-5 |
| 1025 | 6.63e-7 | 6.22e-7 | 2.72e-6 |

Here the hot path reads the dense physical values and linearly combines
two rows. It calls neither a cubic spline nor the power reader. The
1025-node case has a worst relative error of 8.61e-4 among entries above
1e-6 of the appropriate role's peak. Densifying the cheap table removes
most of the linear-lookup error without extra exact angular integrals.
This test covers a interpolation only; it does not validate the additional
K,Q interpolation required to query arbitrary physical modes.

The complete lookup was repeated with an external C SIMDe reader. At each
query it shares the a index/fraction across all physical columns, loads
adjacent columns contiguously and uses a fused multiply-add in each lane.
The same accuracy results hold. Separate checks with five odd/even column
counts, signed inputs, exact endpoints and one/eight threads agree bitwise
with scalar libm fma. A second validation uses the existing cosmo2D
four-query gather reader. These are correctness results, not speed claims.

This layout comparison matters for x86: shared a with many physical
outputs can use contiguous SIMD loads, whereas unrelated query locations
may benefit from cosmo2D's gathered reads. Keep both experiments outside
production until the survey driver's actual access pattern is established;
do not add an unused generic lookup framework to the C library.

## Why the tested coordinate matters

A separate experiment holds angular ell fixed while a changes, so
`k(a)=(ell+1/2)/chi(a)` moves through the power spectrum. The same coarse
a grid then under-resolves the function. Densifying its interpolant cannot
recover missing coarse-grid information.

Even removing the measured external pair powers gives a 14.2% peak
AvgT error at 1025 dense nodes, followed by linear off-grid queries.
Using one moving pivot for every pair is much worse. This rejects these
specific coarse fixed-ell tables, not the coarse-cubic/dense-linear design.
The fixed-physical-k result must not be advertised as a measured speedup
or accuracy certificate for a complete Limber projection.

## Decisions and next integration check

- Keep strict arithmetic, explicit independent-output OpenMP and SIMDe.
- Do not reject an x86 SIMD optimization solely because the M2 shows a
  small gain. This review has no x86 timing evidence.
- Reuse shared angular geometry and rules outside the radial loop.
- Use the measured coarse-construction/dense-linear strategy where the
  complete lookup passes, with an independently refined reference.
- Do not install coarse Cartesian K,Q splines for angular averages or
  the failed coarse fixed-ell evolution table.
- For SSC, interpolate common responses and form weighted outer products;
  do not independently spline auto/cross covariance blocks. The negative
  mode review demonstrates why consistency across blocks matters.
- Finish survey projection and validate total covariance positivity and
  Fisher FoM/errors before selecting production coarse/dense counts.

No claimed timing gain here is an end-to-end Roman covariance speedup.
The pilot still excludes unfinished all-pairs non-Limber survey stages.

## Evidence and didactic self-review

External experiments under `test/covariance_reference/` are
`review_optimization.py`, `check_dense_linear_lookup.py`,
`power_batch_cov.c`, `linear_rows_cov.c`, `check_linear_rows.py`,
`build_power_batch.sh`, and `build_unroll_trials.sh`.
Results include `optimization_accuracy.json`,
`optimization_extra_cosmologies.json`, `optimization_limber_spline.json`,
`dense_linear_lookup.json`, `linear_rows_accuracy.log`, and the isolated
unroll check logs.
The two unrolled libraries each pass all five perturbation tests.
The new external reader's six SIMDe calls were reviewed individually;
its loop overviews explain the interpolation and lane ownership, each
intrinsic has adjacent step-by-step comments, and all C lines fit 80
columns. Scalar arithmetic remains only a comparison/tail, with no
production SIMD fallback switch.

The review checked coordinate definitions, array orientation, padding,
signed quantities, initialization/warm-up scope and the distinction
between construction and hot lookup. It also separates interpolation
errors from quadrature refinement and from projected covariance accuracy.
This is a didactic self-review; Fable 5 was unavailable.

## Follow-up: eight-core scaling and the process boundary

The owner requires useful scaling to 8--10 OpenMP workers per process,
with 1/2/4/8-thread measurements on this laptop. C must not call MPI.
A later Python driver, through the C++ interface, may divide larger
matrix subblocks among MPI processes. A 40-core node can then run five
processes with eight threads or four with ten, keeping BLAS single
threaded. Core-state initialization and table warm-up remain serial
within each process before independent OpenMP work begins.

The review checks the number of *independent output tasks*, not merely
whether a loop has an OpenMP pragma:

| Calculation | Existing task count | Eight/ten-thread implication |
|---|---|---|
| Halo mass weights | na | One-shell calls made this expensive setup serial. |
| NFW profiles | na * nk | A two/three-k derivative batch underfilled the team. |
| Halo moment sums | na * ceil(npair/2) | Large pair tables fill the team; batch small independent requests rather than split deterministic sums. |
| Perturbation averages | ceil(npair/2) | Thousands of pair tasks for realistic tables; angular reductions stay within each lane. |
| Lensing efficiency | nfield | Roman has 16 fields, enough for eight; each radial sweep is cumulative and cannot be naively collapsed. |
| Limber spectra | nell * ceil(npair/2) | Already collapsed over independent outputs. |
| Real-space operators | 4 * ntheta | Roman has 60 complete recurrence tasks; multipole recurrence itself is dependent. |
| Gaussian projection | ceil(nleft/4) * ceil(nright/4) | A 15-by-15 block gives 16 tiles, suited to eight; larger dispatched blocks provide more tasks for ten. |
| Mask pair area | ceil(ntheta/2) | Fifteen bins give only eight tasks, but this geometry is reusable; profile before changing its sum order. |
| SSC mask variance | ceil(nradial/2) | Supply a radial batch, not one call per shell. |
| SSC shell response | nrow | Small row batches can underfill; larger independent matrix blocks avoid that limit. |
| Halo response/trispectrum assembly | ceil(npoint/2) | Batch independent physical points; retain SIMDe. |

Two minimal halo-loop candidates were implemented and kept separately in
the external comparison:

- Collapse mass-weight preparation over (a,mass). Compute the small
  low-mass completion sum afterward, in the original mass order.
- Collapse profile preparation over (a,k,mass), because every NFW value
  is independent. This exposes enough work even for a small k batch.

No integration-node order, physics reader, quadrature, profile formula or
SIMDe moment sum changes. There is no thread reduction, MPI call, new
cache, extra workspace or duplicated fallback implementation. The
existing mass-weight table supplies the subsequent completion sum.

The external `check_halo_scaling.py` compares the pinned `f497d1b`
baseline, the mass-only change and both changes. Workloads cover
(na,nk,nmass)=(1,2,4096), (1,32,4096), (1,128,4096), (8,16,2048).
Every result is bitwise equal across the three layouts and 1/2/4/8
threads. The dedicated small-batch project test also compares the
one-a/two-k result with its entries in a larger table.

The first timing attempt encountered desktop activity and was archived
as `halo_thread_scaling_contended.json`. It is explicitly excluded from
performance evidence. A five-second CPU-time measurement subsequently
found 1.27 cores of desktop/background work, so clean timings were deferred
and correctness work continued. No speedup should be inferred from those
discarded samples. The eight performance cores are the primary scaling
target; the laptop also has four efficiency cores.

A future SSC builder has another clear reuse opportunity: the external
response adapter requests a separate small shifted-k halo batch per
wavenumber. It consequently repeats mass-weight preparation at the same
scale factor. Design the production response-table build around shared
mass weights and profiles, and measure a diagonal-moment path before
adding it; computing an entire quadratic pair table solely for its
diagonal would waste work. This is an identified next experiment, not an
implemented API or claimed speedup.

The separate didactic review checked that the new loop overviews explain
why mass weights and profiles are independent, why the completion sum
stays ordered, and how small batches use all workers. New C lines stay
within 80 columns. Existing SIMD calls and their lane explanations are
unchanged. Debug tests cover all halo and SSC checks, including physical
mass refinement, thread repeatability and complete rectangular subblock
assembly. This remains a self-review; no unavailable model review is
claimed.

The two loop changes were saved separately in commit `33a5fc5` after the
bitwise checks and the default/debug covariance tests. This commits the
parallel work distribution, not a measured speedup. After all project
regressions finished, a fresh five-second CPU-time sample still measured
1.18 cores of desktop activity, principally WindowServer and the Codex
renderer/service. No new timing run was started. The external record is
`results/quiet_check_final.json`; the later measurements follow below.

## Completed quiet-machine measurements

The owner confirmed that the machine was available for timing. Each job
ran alone: no concurrent build, test suite, CAMB calculation or second
benchmark. Ordinary desktop services remained active. Hardware was Apple
M2 Pro (eight performance and four efficiency cores), macOS 13.7.5,
Clang 19.1.7, arm64. OpenMP used 1/2/4/8 threads with
`OMP_PROC_BIND=disabled`; the operating system chose core placement.
Every loaded BLAS pool was checked to have one thread.

Isolated C variants use the same strict flags: `-O3 -march=native`,
`-fno-fast-math -fno-associative-math -frounding-math -ftrapping-math
-fno-reciprocal-math`. Halo and angular-integrator variants also use
`-flto=auto`. The separate power/lookup experiment does not use LTO.
These are wall-clock component measurements, not Linux perf hardware
counters, production x86 results, or full-covariance timings.

### Halo preparation: split rows only when the team needs more work

The comparison uses the real Roman input setup with Omega_m=0.3,
Omega_b=0.04, H0=67.32, ns=0.96605, As=2.1e-9, w=-1 and mnu=0.06 eV.
It measures cb halo moments; this does not certify a total-matter
massive-neutrino covariance prescription. CAMB and the first lazy core
tables are excluded. Each timed call includes caller output allocation,
copies and padding checks, C workspace construction, all halo stages,
and workspace release. No halo output is cached between calls.

The pinned baseline is `f497d1b`, before either mass-axis change.
Four layouts were compared: baseline, mass-weight splitting only,
`33a5fc5` (weights and every profile mass node separately), and a chunked
profile layout. The final confirmation alternates baseline/chunked order,
uses two warmups per configuration, and retains 101 samples per layout
and thread count. All use the same eight logarithmic mass panels between
1e6 and 1e17 M_sun/h and the same physical inputs.

Final medians, milliseconds per complete halo call:

| (a rows, k per row, mass nodes) | Layout | 1 thread | 2 threads | 4 threads | 8 threads |
|---|---|---:|---:|---:|---:|
| (1, 2, 4096) | Baseline | 1.322 | 1.230 | 1.268 | 1.425 |
| (1, 2, 4096) | Selected | 1.327 | 0.794 | 0.507 | 0.562 |
| (1, 32, 4096) | Baseline | 7.749 | 4.653 | 2.995 | 2.255 |
| (1, 32, 4096) | Selected | 7.764 | 4.241 | 2.321 | 1.534 |
| (1, 128, 4096) | Baseline | 43.428 | 23.803 | 13.145 | 7.799 |
| (1, 128, 4096) | Selected | 43.444 | 23.281 | 12.357 | 7.023 |
| (8, 16, 2048) | Baseline | 16.455 | 8.598 | 4.622 | 2.888 |
| (8, 16, 2048) | Selected | 16.468 | 8.609 | 4.633 | 2.932 |

The eight-thread means and sample standard deviations retain the observed
spread rather than presenting the median as a guaranteed runtime:

| (a rows, k per row) | Baseline mean +/- SD (ms) | Selected mean +/- SD (ms) |
|---|---:|---:|
| (1, 2) | 1.427 +/- 0.042 | 0.587 +/- 0.069 |
| (1, 32) | 2.363 +/- 0.242 | 1.598 +/- 0.198 |
| (1, 128) | 8.043 +/- 0.601 | 7.350 +/- 0.723 |
| (8, 16) | 2.978 +/- 0.334 | 3.046 +/- 0.386 |

Relative to one thread, selected eight-thread medians improve by 2.36x,
5.06x, 6.19x and 5.62x respectively. Relative to the old eight-thread
baseline, gains are 2.54x, 1.47x and 1.11x for the one-a requests.
The already well-filled eight-a request has no improvement (1.5% slower
by median). The smallest request is faster at four than eight threads;
batching small independent requests remains important. Do not promise
linear scaling for tiny batches or extrapolate these results to ten cores.

The 51-repeat four-layout trial explains the adjustment to `33a5fc5`:
collapsing every profile mass node added index overhead to larger calls.
Whole profile rows avoid that overhead when enough rows exist. Otherwise
each row gets ceil(threads/(na*nk)) contiguous mass chunks, giving at least
one task per worker. This adds no arrays or caches and changes no sums.
For (8,16), its eight-thread median was 2.946 ms versus 3.165 ms for
the every-mass layout; for (1,128), 7.065 versus 7.184 ms. The 101-repeat
baseline comparison confirms the selected layout without cherry-picking
a best repetition. An earlier contended file remains excluded.

The full default comparison checks uint64 views, including four additional
boundary shapes: (na,nk)=(1,1),(1,3),(3,1),(2,5). Every output is bitwise
identical to baseline at 1/2/4/8 threads. A separate O0 build with undefined
behavior and floating-divide-by-zero sanitizers checks five small shapes,
zero-k completion, odd pair counts and output padding. It also matches
the strict baseline bitwise, with no sanitizer report.

### Angular input preparation dominates the existing C integral

At z=0.75, 128 wavenumbers give 8,256 unique pairs and 5,120 angular
nodes per pair. Each row reads P(|K+Q|,a). Nine samples follow two warmups.
The existing serial power reader takes 442.845 ms with output preallocated.
The external experiment partitions identical rows across workers and
calls that same reader; its output is bitwise equal to the serial batch.

| Threads | Power median (ms) | Power mean +/- SD (ms) | SIMDe integral median (ms) | Integral mean +/- SD (ms) |
|---:|---:|---:|---:|---:|
| 1 | 443.134 | 443.005 +/- 0.902 | 40.064 | 40.123 +/- 0.269 |
| 2 | 226.502 | 226.722 +/- 2.196 | 20.340 | 20.375 +/- 0.212 |
| 4 | 118.682 | 128.690 +/- 29.992 | 10.619 | 10.676 +/- 0.209 |
| 8 | 63.091 | 63.071 +/- 1.058 | 5.642 | 6.316 +/- 1.041 |

The power stage scales 7.02x and the existing SIMDe integral 7.10x from
one to eight threads. Constructing the Python wavenumber geometry with
allocation takes a further 166.737 ms (166.415 +/- 2.999 ms), compared
with 7.051 ms for the angular rule and 10.927 ms for Python row pointers.
Fixed physical wavenumbers let a future table builder reuse this geometry
across scale-factor rows. Reusing it and batching power inputs matter
more than tuning the already short contraction. The isolated power
experiment remains external until that production table builder exists.

Explicit unroll factors two/four pass bitwise comparisons. Eight-thread
medians are 6.821/5.506 ms, versus 5.642 ms without the extra pragma;
means are 6.475 +/- 1.215, 6.140 +/- 1.175, and 6.316 +/- 1.041 ms.
The observed differences do not justify a production unroll change.
Keep unconditional SIMDe; revisit unrolling on the intended x86 target.

### Coarse exact rows, cubic construction, then dense linear queries

The complete two-stage test uses 33 fixed physical wavenumbers (561 pairs),
2,560 angular nodes, and 65 off-grid scale factors over [0.25,0.875].
It compares direct calculation at every query with 21 exact samples
including boundary padding, cubic construction of 1,025 physical-value
rows, then ordinary linear interpolation. Both paths use eight threads
and reuse the same precomputed pair geometry and angular rule.
The spline timing includes exact coarse integrals, pivot-power growth
factors, dense construction, output allocation and all 65 queries.

| Path | Median (ms) | Mean +/- SD (ms), five samples |
|---|---:|---:|
| 65 direct rows | 183.521 | 183.872 +/- 1.744 |
| Complete two-stage calculation | 66.604 | 66.183 +/- 1.260 |

The measured gain is 2.76x, including construction. Maximum errors over
each reference component's peak are 7.15e-7 (AvgP), 4.77e-7 (AvgB) and
2.18e-6 (AvgT). Maximum pointwise relative errors among entries larger
than 1e-6 of their component peak are 7.13e-6, 1.10e-5 and 6.11e-4.
No significant entry changes sign. These reproduce the accuracy tests;
they do not certify a covariance or justify splining directly along the
moving k=(ell+1/2)/chi(a) coordinate.

Once the dense table exists, the 65-query hot lookup medians are:

| Reader | Threads | Median (microseconds) | Mean +/- SD, 19 samples |
|---|---:|---:|---:|
| NumPy, allocates output | Serial | 253.8 | 255.5 +/- 21.4 |
| Existing cosmo2D SIMDe gather, preallocated | 8 | 161.8 | 185.7 +/- 63.0 |
| Contiguous-column SIMDe experiment, preallocated | 1 | 30.6 | 31.4 +/- 1.9 |
| Contiguous-column SIMDe experiment, preallocated | 8 | 48.3 | 52.2 +/- 13.7 |

These compare readers and layouts, not a controlled scalar-versus-SIMD
instruction experiment. Table transposition is outside the gather timing;
neither C reader allocates output. Contiguous lanes represent adjacent
physical columns at the same a and share its bracket. The gather reader
instead represents different query coordinates in its lanes. Both agree
with the same linear formula, and contiguous results are bitwise equal
at one/eight threads. A short cache-resident lookup does not benefit from
eight workers here. Keep the contiguous SIMDe candidate for the eventual
table design; do not add a standalone production API with no caller.

### Evidence and remaining limits

The local raw records are `results/halo_thread_scaling_chunked.json`
(51 repeats), `results/halo_thread_scaling_final.json` (101 repeats),
`results/halo_thread_accuracy.json`, `results/halo_chunks_debug.log`, and
`results/optimization_timing.json`. Harnesses are
`check_halo_scaling.py`, `check_halo_chunks_debug.py` and
`review_optimization.py timing`, under the external covariance reference.
All samples are retained, including the slower repetitions.

The didactic self-review checked the new profile-loop overview, its
two-rows/eight-workers example, chunk endpoint explanation and low-mass
completion slot. Every C line is within 80 columns; the SIMD moment sums
and their explanations remain unchanged. This is a self-review, not an
unavailable Fable review. Production changes are confined to `halo_cov.c`.
Only LSST currently compiles that routine into its public interface.

After installing the measured code, the standard LSST interface rebuild
and 61 project checks passed: all 53 covariance tests plus eight frozen
NLA/TATT and repeated-cosmology tests for cosmic shear and 3x2pt
(`test_example1.py` and `test_example2.py`). The run took 90.02 seconds;
the four frozen chi2 differences and four repeatability differences all
print 0.000000. No reference was refrozen. The small-batch regression now
checks one, two and three k rows at 1/2/4/8 threads using uint64 views.
The production source also passes a compile-only check without OpenMP.
The earlier full seven-project results remain recorded separately; those
full suites were not repeated for this covariance-only layout change.

These measurements identify component costs and support one small halo
layout change. Full Roman G+SSC+cNG generation, full/selected-matrix
positivity and Fisher/FoM convergence remain separate unfinished work.
