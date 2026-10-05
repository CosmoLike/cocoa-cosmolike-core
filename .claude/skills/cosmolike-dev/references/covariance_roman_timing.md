# Roman covariance: small-component runtime estimate

Measured 2026-10-03 on Apple M2 Pro, eight OpenMP threads, strict IEEE
Clang 19.1.7 builds. OpenBLAS is pinned to one thread; runtime inspection
confirmed both loaded OpenBLAS libraries report one. No full covariance
or long reference run was launched. These are laptop wall times, not
Linux hardware-counter measurements or a measured CosmoCov speedup.

## What was measured

The external `test/covariance_reference/benchmark_roman_subset.py` loads
the actual compiled roman_real interface. It installs the project's eight
lens/source distributions, NLA parameters, biases and fiducial cosmology
(Omega_m=0.3, Omega_b=0.04, H0=67.32, A_s=2.1e-9, n_s=0.96605,
m_nu=0.06 eV, w0=-1, wa=0). CAMB is generated once and excluded.
This does not validate the covariance halo model's neutrino prescription.

The covariance halo library was compiled separately against that Roman
interface, with the same strict flags as `build_halo.sh`; the original
LSST-linked library remains available. The external `SurveyNGInputs`
adapter accepts this library explicitly. No production C or project
interface was changed.

At each sampled redshift, logarithmically spaced multipoles span
2--100000. Masses span 10^6--10^17 solar masses/h in eight log intervals;
the angle rule has 20 graded intervals. Counts below are total nodes,
not nodes per interval. Each stateless helper recomputes its output:
the first call and two further warm-ups precede five timed calls. Timings
include Python adapter allocations and the C workspaces. The shared core
tables remain warm, and their first-use cost is recorded separately.

| Multipole nodes | Mass nodes | Angle nodes | Halo + angular + SSC response per radial node |
|---:|---:|---:|---:|
| 16 | 512 | 1280 | 5.4--7.1 ms |
| 32 | 512 | 1280 | 14.1--14.7 ms |
| 64 | 512 | 1280 | 43.2--44.3 ms |
| 64 | 1024 | 2560 | 84.7--89.4 ms |
| 128 | 512 | 1280 | 176.9 ms |
| 64 | 4096 | 5120 | 192.2 ms |
| 128 | 4096 | 5120 | 700.0 ms |

The first four rows span z=0.25, 0.75 and 1.5; the last three use z=0.75.
The last three include the measured five-term assembly (0.13--0.17 ms);
the earlier rows omit that small step. At the finest sampled setting,
angular work is 666.9 +/- 11.1 ms, halo moments 7.90 +/- 0.22 ms, and
SSC response 25.01 +/- 0.26 ms (mean +/- sample standard deviation).
This identifies the angular stage as the current dominant cost. It also
includes the external angular-rule construction and power-input adapter;
it is not a pure timing of the C perturbation kernel.

## Projection cost, counted separately

The real-space geometry uses Roman's 15 log theta bins, 2.5--250 arcmin.
It computes all four probe operators once: about 0.22 s to ell=50000 and
0.41 s to ell=100000 with 512 angular-bin quadrature nodes. One 15 by 15
Gaussian block takes 0.31 ms and 0.67 ms, respectively (21 timed calls).
Its supplied smooth harmonic variance is a cost test, not Roman physics.

Roman has 36 xi+ rows, 36 xi- rows, 61 gamma_t rows and eight w rows;
each row has 15 theta bins. Thus there are 141 observable rows and
141*142/2 = 10011 upper-triangle Gaussian blocks. Straight multiplication
gives about 3.1 s or 6.7 s for their projections. This is an extrapolation
from a repeatedly used block, not a full scan of changing spectra.

`benchmark_roman_contractions.py` measures the existing C weighted-dot
kernel on small supplied arrays for the planned non-Gaussian factorization:
first transform the shared trispectrum into 60 angular rows, then apply
the tomographic radial weights. Two multipole contractions cost about
0.15 ms per radial node at 64--128 multipole nodes. Small radial products
suggest well below one second for their arithmetic at 512--1024 radial
nodes. These are arithmetic-only estimates: they do not measure a finished
NG interpolator, survey driver or its allocation and output costs.

Do not multiply the shared 3D table time by the number of bin pairs. Those
tables depend on redshift and multipoles, not tomography. A design that
repeats them inside every covariance block would have a different cost.

## Conditional budget, not an end-to-end result

Multiplying the measured per-node cost by a proposed radial count, and
adding the ell=100000 Gaussian projection estimate, gives:

| Proposed sampling | 512 total radial nodes | 1024 total radial nodes |
|---|---:|---:|
| 64 multipoles, 512 mass, 1280 angle nodes | about 30 s | about 52 s |
| 128 multipoles, 4096 mass, 5120 angle nodes | about 6 min | about 12 min |

These budgets cover the measured shared ingredients plus projection
arithmetic. They omit all-pairs spectra (especially the unfinished
non-Limber extension), survey-window/mask setup, interpolation, final
assembly, matrix validation and I/O. They are neither an upper bound on
the full runtime nor a certified accuracy/runtime tradeoff. The radial
counts are planning examples, not a measured converged Roman grid. A
super-strict run may require more nodes and take longer. No numerical
reference, FoM convergence or old/new speedup has been established here.

Next: finish the missing survey stages, time another representative subset,
then choose a high-resolution reference budget. Refine that reference
again before using Fisher/FoM comparisons to choose practical settings.

## Evidence and didactic self-review

External raw outputs, with every timed sample and configuration:

- `results/roman_subset_8threads.json` and its log;
- `results/roman_subset_refined_8threads.json` and its log;
- `results/roman_contractions.json` and its log;
- `inputs/roman_timing_camb.npz`;
- `results/roman_subset_checks.log` and `results/adapter_regression.log`.

After implementation and measurement, a separate didactic self-review
checked both benchmark files and the small adapter change. Definitions
distinguish multipole nodes from angular/mass quadrature nodes, per-node
work from bin-pair work, physical inputs from cost-only arrays, and
measured timings from extrapolations. Warm-up and allocation scopes are
explicit. This is a self-review, not a Fable review.

Validation: all four changed external Python files pass `py_compile`;
Roman halo moments, angular averages, five-term assembly and SSC responses
are bitwise identical at one and eight threads for the checked row; the
default LSST adapter reproduces archived I11 and all five halo moments
bitwise. The supplied-array contraction cases agree with NumPy within
2e-13 relative tolerance. Production code did not change; this pilot does
not replace project regression suites or the future survey physics tests.
