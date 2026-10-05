# Connected covariance block scheduling

Measured on 2026-10-04, Apple M2 Pro. This changes the scheduling of the
existing supplied-trispectrum projection, not the model or its sampling.

## Why the previous loop scaled poorly

After angular projection, the matter trispectrum has axes
`[4*nbin,4*nbin,nradial]`. Catalogs enter through `W_A*W_B` on the same
radial nodes. The previous Python loop called `covariance_project` for
each pair of angular bins and probe types. For the LSST layout this meant
5460 calls, each starting two OpenMP regions. The per-call work could not
amortize their synchronization, particularly at eight workers.

`covariance_project_connected` now creates one team for all angular
blocks. Each task forms the shared `measure*T` weights and calls the
same `gaussian_project_cov` primitive for its catalog pairings. That
primitive suppresses nested teams. Each worker owns its weight, weighted
window and catalog-block scratch; no new cache or cosmology state exists.

Catalog order, triangular ownership, SIMDe sums and radial addition order
are unchanged. Mirrored entries use one computed value. The implementation
preserves signed trispectra and retains every cross-catalog combination.
The Python helper becomes a shape-checked call to this compiled wrapper.

## Controlled scheduling experiment

Inputs are fixed synthetic signed tables with the LSST layout
(60 observables, 26 angular bins, 512 radial nodes) and the DES joint
layout (140 observables, 20 bins, 704 nodes). No CAMB, halo calculation or
input generation is timed. Calls include validation, output allocation,
weight construction and projection. BLAS is fixed to one thread.
Each case has one excluded warm-up and seven timed repetitions. Only one
numerical process runs; no concurrent builds or tests enter the timings.

| Layout | Threads | Previous Python loop (s) | Cyclic whole blocks (s) |
|---|---:|---:|---:|
| LSST | 1 | 0.243164 | 0.086623 |
| LSST | 2 | 0.307999 | 0.045361 |
| LSST | 4 | 0.369335 | 0.025091 |
| LSST | 8 | 0.572600 | 0.014826 |
| DES joint | 1 | 0.504890 | 0.261094 |
| DES joint | 2 | 0.467506 | 0.135066 |
| DES joint | 4 | 0.448736 | 0.071790 |
| DES joint | 8 | 0.544636 | 0.040427 |

At eight threads, sample standard deviations are 0.000624 s for the new
LSST layout and 0.001081 s for DES. One-to-eight scaling is 5.84 and 6.46;
four-to-eight scaling is 1.69 and 1.78. These are projection-kernel results,
not full survey speedups or x86 performance measurements.

The external experiment also compiles two alternative OpenMP schedules:

| Layout, eight threads | Contiguous static (s) | Cyclic static (s) | Dynamic, one block (s) |
|---|---:|---:|---:|
| LSST | 0.021167 | 0.014826 | 0.014173 |
| DES joint | 0.112794 | 0.040427 | 0.040954 |

Contiguous assignment is imbalanced because successive probe groups have
very different catalog counts. Keep `schedule(static, 1)`: cyclic blocks
distribute those groups across workers without dynamic dispatch. Dynamic
assignment has no consistent advantage across these two measured layouts.
Do not infer a universal optimum for different CPUs or grids.

## Complete DES joint forecast

The boost-1 forecast has 2812 entries and retains its 48 defined Y null
rows. It uses the same massless Limber, biased-tracer cNG and SSC-only
count-cross model as the earlier baseline. These physical omissions are
unchanged. CAMB and initialization are excluded; each timed call rebuilds
the complete covariance with warm core readers. There is one excluded
warm-up and three timed calls at each worker count.

| Threads | Full mean (s) | Sample std. (s) |
|---:|---:|---:|
| 1 | 10.490320 | 0.017502 |
| 2 | 6.451193 | 0.047947 |
| 4 | 4.364951 | 0.102685 |
| 8 | 3.520034 | 0.222216 |

Every G, SSC, cNG, total and mean entry is bitwise equal to the saved
pre-optimization reference at all four thread counts. The final SSC/cNG
stage averages 0.2032 s at eight workers, versus 0.7370 s in the preceding
halo-only run. Shared matter preparation is still 1.4668 s and all-pairs
spectra 1.0222 s. Full one-to-eight scaling is 2.98 and four-to-eight is
1.24; these remaining stages need attention. Do not substitute the much
larger kernel speedup for this complete forecast measurement.

## Correctness and didactic review

- Every scheduling variant matches the previous supplied-table output
  bit for bit at 1/2/4/8 workers.
- LSST's covariance suite: 87 passed. Thirteen added tests cover signed
  tables, unsorted probe order, empty groups, one/multiple angular bins,
  odd/even radial lengths, partial SIMD tiles and invalid input rejection.
  A separate NumPy contraction checks the equation independently.
- An isolated O0 UBSan/float-division build passes 48 connected cases and
  all 64 existing Gaussian wrapper comparisons.
- No frozen reference or physical setting changes.

The code review explains the shared matter weight before the parallel
region, defines the combined probe/bin index and radial measure, and
states which task owns each matrix entry and its transpose. Each new
SIMDe operation describes its two radial lanes. Paragraphs separate
validation, grouping, weighting, contraction and copying. Public guides
explain the interface and physics without internal build paths.

Internal harnesses and JSON are in `test/covariance_reference/`: the
connected schedule experiment, old Python loop and sanitizer checks.
They are development records, not prerequisites for using the API.

## Project regressions after shared halo and projection changes

All seven interfaces contain the new I11-only and connected-block APIs.
The build runner's incorrectly cased skip flags initially left Roman
Fourier stale; its covariance test rejected the missing keyword. After
correcting the flags, Roman Fourier, Roman KL, DES Y3 and DES x Planck
were rebuilt and checked. LSST, DES cluster and Roman real had already
rebuilt successfully. No numerical setting or frozen reference changed.

| Project | Covariance tests | Frozen likelihood tests |
|---|---:|---:|
| lsst_y1 | 87 | 12 |
| des_cluster | 46 | 16 |
| roman_real | 1 | 12 |
| roman_fourier | 1 | 12 |
| roman_kl | 1 | 12 |
| des_y3 | 1 | 24 |
| desy1xplanck | 1 | 12 |
| Total | 138 | 100 |

The full earlier data-vector suites are recorded in
covariance_gaussian_scaling.md with their build-provenance correction.
For these covariance-only changes, the fresh pass repeats all covariance
tests and frozen likelihood examples, not unrelated advisory emulator
scans. All passed; no refreezing was needed. The sequential timing pass
also checks loaded bindings and records the actual library and relevant
source SHA-256 hashes, alongside repository revisions.
