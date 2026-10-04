# Gaussian covariance block scheduling (2026-10-04)

The complete Gaussian wrapper used to visit measured observable pairs
serially and start three OpenMP regions inside each pair. DES has 140
observable rows, hence 9870 triangular blocks and about 29610 small
parallel regions. The new wrapper flattens the triangular block list and
distributes complete blocks among workers. Each worker owns its scratch
and both symmetric output blocks. Standalone C primitives retain their
bin/multipole parallelism; calls inside an active team run on that worker.
Neither the arithmetic nor the increasing-multipole sum order changes.

## Controlled component comparison

Apple M2 Pro, BLAS fixed at one thread, 140 observables, 20 angular bins,
22 internal fields, multipoles 2 through 10000. Supplied positive field
spectra exercise the entire 2800-entry Gaussian assembly, including
allocation, Wick contractions, projection and analytic pair noise.
They are synthetic inputs, not a complete physical covariance forecast.
Each setting has one excluded warm-up and three measured calls. Before
and after libraries ran sequentially, with no other numerical jobs.

| Threads | Before mean (s) | After mean (s) | After sample std. (s) |
|---:|---:|---:|---:|
| 1 | 3.778862 | 3.743424 | 0.029403 |
| 2 | 2.358472 | 1.926090 | 0.040660 |
| 4 | 1.554988 | 1.006872 | 0.002998 |
| 8 | 1.766546 | 0.585849 | 0.001031 |

Every output entry is bitwise identical to the previous one-thread
reference. Eight-thread component speedup is 3.02 relative to the old
eight-thread layout and 6.39 relative to the new one-thread layout.
Four to eight threads gives 1.72. These are local Apple measurements;
production x86 scaling remains to be measured.

## Full DES joint check

The massless, Limber, boost-1, 2812-entry joint forecast retains its 48
defined Y null rows. All G, SSC, cNG, total and mean entries are bitwise
identical to the saved pre-change arrays at 1, 2, 4 and 8 threads.
Initialization and CAMB are excluded; each setting has an excluded
warm-up and three complete matrix recomputations at fixed cosmology.

| Threads | Mean (s) | Sample std. (s) |
|---:|---:|---:|
| 1 | 13.963389 | 0.422411 |
| 2 | 8.423946 | 0.041212 |
| 4 | 5.689045 | 0.048475 |
| 8 | 5.131000 | 0.279223 |

The eight-thread Gaussian stage is 0.6033 s, but shared halo tables take
2.5714 s in this noisier full run. Overall four-to-eight scaling remains
only 1.11. The optimization therefore fixes the Gaussian scheduling
problem, not the complete forecast's scaling. Profiled baseline shared
matter tables took 2.201 s; inspect redundant shifted-k halo moments next.

## Review and checks

The didactic review explains why observable blocks are independent,
which worker owns each symmetric block, how per-worker scratch prevents
races, and why fixed per-entry sum order preserves thread determinism.
No new SIMD operation was introduced; existing lane-by-lane explanations
remain. Both edited C/C++ files respect 80 columns. Diff checks pass.
An isolated unoptimized UBSan build passed 64 real/Fourier comparisons
against the optimized project build, covering 1/5 observables, 1/3/5/20
bins and 1/2/4/8 workers. This includes partial SIMD tiles and the
single-observable fallback to bin parallelism.

The external development record contains `benchmark_gaussian_blocks.py`,
`check_gaussian_block_debug.py`, and `benchmark_cluster_joint_scaling_after.py`
under `test/covariance_reference/`. Their JSON/NumPy results are internal
measurement records, not public test prerequisites. Project regression
results are appended after the sequential rebuild and test pass.

## Complete project regression pass

Every project was rebuilt and checked sequentially on 2026-10-04. The
ordinary OpenMP build passed all covariance and data-vector suites:

| Project | Covariance tests | Data-vector tests |
|---|---:|---:|
| lsst_y1 | 72 | 57 |
| des_cluster | 46 | 29 |
| roman_real | 1 | 104 |
| roman_fourier | 1 | 45 |
| roman_kl | 1 | 49 |
| des_y3 | 1 | 63 |
| desy1xplanck | 1 | 45 |
| Total | 123 | 392 |

Roman's optional slow halo checks were enabled. No frozen result or
manifest was changed. Existing advisory accuracy scans report numerical
differences without imposing the covariance convergence criterion; their
completion is not a certification of an inference accuracy setting.

The Gaussian C file also compiled successfully without OpenMP. Its
OpenMP header is conditional, and ignored parallel pragmas introduce no
serial-build dependency on the runtime.

Subsequent covariance-only scheduling changes should rerun every project's
covariance suite and frozen likelihood examples. Repeating all advisory
baryon/accuracy scans is unnecessary unless the change or a failure gives
a reason to revisit the data-vector baseline.
