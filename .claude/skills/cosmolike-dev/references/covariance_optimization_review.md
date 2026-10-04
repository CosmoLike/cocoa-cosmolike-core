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

At the owner's request, this pass now prioritizes interpolation accuracy
and the negative-mode investigation. The queued timing job was stopped
before it launched. A lookup-validation preflight invoked timing helpers
while project tests were running; those elapsed times were discarded.
No new performance result is reported here. Run the prepared profile
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
