# Covariance convergence: parameter constraints and valid matrices

Owner clarifications, 2026-10-03: Figure of Merit (FoM) convergence is the
scientific objective; a usable total covariance cannot have negative
eigenvalues. The owner suggests trying 1e-3 convergence of important modes.
The mode definition and final acceptance protocol remain to be measured
on roman_real. No production covariance has passed this new protocol yet.

## Why the earlier criteria were inappropriate

The 1e-6 elementwise target originated in the external study's
`12_fable_phase0_review.md`, C2 item 2(a), then entered `PLAN.md`'s
reference exit criteria. It was a reference-quality proposal, not a
requirement derived from Roman parameter constraints. Carrying it into
production as a universal blocker was inappropriate. Relative errors of
near-zero SSC cross terms are particularly poor science criteria.

The subsequent suggestion to adopt |delta chi2| < 0.2 as the main
covariance criterion was also withdrawn. That is the owner's data-vector
criterion. It is not a covariance-convergence standard. Retain quadratic
form comparisons as diagnostics, without making that number the gate.

## What Krause and collaborators actually tested

- [Friedrich et al. (2021), DES Y3 covariance](https://arxiv.org/html/2012.08568v3),
  Sections 5.1--5.3 and Table 1: propagate covariance changes into posterior
  widths, actual scatter of maximum-posterior parameters, and the mean and
  scatter of the best-fit chi2. Equations 33--36 distinguish the reported
  parameter covariance from the actual estimator covariance when the data
  have a different covariance. Priors and nuisance parameters matter.
- [Barreira, Krause & Schmidt (2018)](https://arxiv.org/html/1807.04266v2),
  Section 3.1, Figures 2--4 and footnote 6: compare parameter errors,
  marginalized contours and parameter-volume FoM for different G/SSC/cNG
  combinations. In their five-parameter Euclid-like shear example,
  removing cNG raises FoM by about 14%; removing SSC raises it by about a
  factor two. Goodness-of-fit can change little even when errors are wrong.
  These are physical-model comparisons, not quadrature stopping rules.
- [Fang, Eifler & Krause (2021), 2D-FFTLog](https://arxiv.org/pdf/2004.04833),
  Sections 4.2.1--4.2.3: compare correlation matrices, modes ranked by
  their signal-to-noise contribution, expected chi2 shifts and simulated
  parameter constraints. Figure 2 displays 100 DES eigenmodes and 200 LSST
  singular modes, not a universal mode-count rule. They exclude the
  quadrature LSST covariance with a negative eigenvalue from inference.

These papers do not establish a universal 1e-6 entrywise or 1e-3 eigenvalue
criterion. The latter is our proposed numerical target for fixed physics.

## Proposed Roman numerical protocol

1. Hold cosmology, nuisance values, priors, mask, model and data-vector
   derivatives fixed. Independently converge derivative step sizes before
   judging covariance refinement. Refine each covariance grid and cutoff,
   then refine the combination; demonstrate a stable sequence.
2. Check symmetry and positive definiteness of the total G+SSC+cNG matrix,
   including a Cholesky factorization of its diagonally scaled form. Check
   the full delivered matrix and the selected likelihood submatrix. Do not
   clip negative eigenvalues or add a ridge to make a failing result pass.
   G and SSC are positive semidefinite individually; the connected cNG
   cumulant need not be. Do not reject cNG solely for being indefinite.
3. For a Jacobian J of the mean data vector and prior precision P, compute
   F = J^T C^-1 J + P, using solves rather than an explicit inverse.
   Invert the full parameter Fisher matrix before extracting the cosmology
   block, so nuisance parameters are marginalized. For dark energy use
   FoM = 1/sqrt(det(S_w0wa)), where S is that parameter covariance.
4. Try relative stability of 1e-3 in FoM and the important marginalized
   parameter errors. FoM alone can hide opposite changes of ellipse axes;
   also compare generalized eigenvalues of the marginalized cosmological
   parameter covariances. This is a small matrix, so inspect all its modes
   instead of choosing an arbitrary top-N subset. Test the proposed 1e-3
   scale there too, recording which criterion sets the numerical cost.
5. Diagnose the full data-covariance change with
   C_fine v = lambda C_coarse v. Values near one measure relative changes
   in common directions, including rotations missed by comparing sorted
   eigenvalue lists. Extremal generalized eigenvalues cover all directions;
   they are not the largest raw covariance eigenvalues. Try 1e-3 as a
   conservative diagnostic, but do not automatically demand expensive
   refinement of parameter-insensitive directions when the science metrics
   have converged. Any such decision needs the measured parameter effects.
6. Keep G, SSC and cNG separate, and refine them independently to expose
   accidental cancellation. Maintain existing strict algebra, units,
   analytic-limit and thread-determinism checks. Physical approximations
   (e.g. SSC response prescriptions) get separate parameter-impact tests;
   numerical convergence is not proof that a model is physically accurate.

For a wrong covariance's uncertainty calibration, supplement Fisher widths
with the actual best-fit scatter of Friedrich et al., Eqs. 34/36. Its prior
term assumes the prior centers are noisy independent measurements; state
that convention rather than silently applying it to fixed prior centers.
Final posterior spot-checks test the local Fisher approximation.

## Fisher does not require a chain

At a chosen fiducial theta_0, J contains the derivatives of the predicted
data vector with respect to cosmological and nuisance parameters. Holding
the likelihood covariance fixed, F = J^T C^-1 J + P is a local Gaussian
forecast. The existing `cosmolike_notebook_utils/fisher.py` provides
derivative, Fisher, prior and FoM helpers; the covariance comparison must
use the same converged J, mask and priors for every candidate C.

For C v_i = lambda_i v_i, mode i contributes
(v_i^T J_alpha)(v_i^T J_beta)/lambda_i to F_alpha,beta. Its importance
therefore depends on parameter sensitivity and degeneracies, not on a
large raw covariance eigenvalue. Include nuisance parameters before
marginalizing; fixing them would overstate the cosmological information.

Repeat the local comparison at a few representative cosmologies. Existing
chain samples can supply those points but are not required. A later chain
comparison tests posterior non-Gaussianity. This fixed-covariance forecast
does not add information from parameter derivatives of C itself.

## What the existing refined diagnostic establishes

The common-mask 256-to-512 radial comparison has total generalized
covariance eigenvalues [0.99999952830575, 1.000001042368917]. All modes in
that small diagnostic therefore pass a 1e-3 relative-mode comparison.
The SSC entrywise ratios of 7.03e-4 and 9.18e-4 are retained as diagnostics.

This is a two-lens/two-source, band-center Limber calculation with an
external smooth-power interpolant. It does not establish Roman FoM
convergence. The still-valid precision bound max|1/lambda-1| is
1.04237e-6, recorded by external `compare_survey_refinement.py` in
`results/smooth_power_512_likelihood_bounds.json`; it is not the new
production stopping rule.

## Nested interpolation-grid audit (2026-10-04)

Before the correction, the covariance non-Gaussian table used 16*b points
uniform in ln(ell+1/2) between ell=2 and ell=10000*b. This changed both
the cell spacing and the upper endpoint at each boost. Only the first old
node survived a doubling. Even with a fixed endpoint, doubling n points
gives 2*n-1 intervals instead of the intended 2*(n-1).

`covariance_accuracy` now starts with 15 intervals across ell=2..10000.
Each doubling halves that same logarithmic step. Increasing the signal
cutoff appends cells, rather than stretching existing ones. Resolved
settings hold the explicit ng_ell array instead of ng_ell_nodes; the full
arrays have 16/35/73/153 samples at boosts 1/2/4/8. Every old sample is
bitwise retained at an even index of the next grid, including both original
anchors. The saved settings preserve those positions without requiring a
reader to reconstruct them from current defaults.

Both survey assemblers validate the supplied grid before expensive work
and trim unused upper nodes with a linear array count. They keep the first
node at or beyond the signal cutoff, with its original value. This avoids
extra halo work above fixed Fourier bands without changing the remaining
cell locations. No new C code, binary search or data-vector table is used.

The lensing-window table already has 4096*b+1 points on a fixed interval
and is nested. Signal/mask sums use integer multipoles, so raising their
cutoffs retains the old modes. Mass, radial and angular Gauss--Legendre
rules change their nodes and weights together; those are quadrature rules,
not interpolation grids, and were not replaced merely to force nesting.
DenseLogTable has caller-owned sampling choices rather than boost defaults;
its documentation now explains refinement of both interval counts.

CAMB tables remain fixed under the covariance boost. The notebook CAMB
helper's redshift copy grid is nested, but its log-k copy grid still uses
1250+250*CLAccuracyBoost*AccuracyBoost points. This separate nonnested grid
does not move in these covariance-only tests. Its refinement and CAMB's own
accuracy are independent possible floors; neither was changed in this fix.

The first isolated response experiment holds mass integration at 1024
nodes/panel and the logarithmic response step at 0.00125. It compares 512
fixed random log-spaced query locations with direct halo-response calls at
a=0.55,0.75,0.9, using a fixed distance of 1000 Mpc/h to map angular to
physical modes. The original moving-range grid's boost-8 maximum relative
response errors are 2.48%, 2.24%, 2.17%; the nested grid gives 1.74%, 1.57%,
1.49%. This comparison also changes node density, so it is not evidence
that node retention alone caused the improvement. It is not a full survey
or a certified response-accuracy setting. External evidence:
`covariance_reference/check_nested_boost.py` and
`covariance_reference/results/nested_boost_response.json`.

### Fixed-range control and high-boost components

The fixed-range comparison holds ell=2..10000 on both ladders. At a=0.75,
the nested 16/31/61/121/241/481-node tables retain every old node, while
16/32/64/128/256/512-node tables retain only the two endpoints. The nested
sequence generally has fewer individual queries whose error grows on
refinement (56/30/45/49/102 of 512, versus 62/81/56/95/124), but it does not
win every maximum or RMS comparison. Nesting is not a proof of monotonic
error: curvature, narrow features and fixed input-table interpolation
remain. See results/fixed_range_boost_response.json and its Python runner.

The actual LSST first-source Gaussian xi+/xi- covariance has 52 entries,
with the full 26 angular bins from 2.5 to 900 arcminutes. CAMB, cosmology,
catalogs and scientific bins stay fixed; the complete covariance boost
changes signal/mask cutoffs, radial/angular rules and window sampling.
All four totals are positive definite. Maximum generalized variance
changes on successive refinements are:

| Boost pair | Maximum fractional variance change |
|---|---:|
| 1 to 2 | 1.02147743905e-4 |
| 2 to 4 | 1.10252220802e-5 |
| 4 to 8 | 2.28962303339e-6 |

The highest run uses bounded multipole batches; the lower-boost numbers
agree with the earlier unbatched run. The storage fix, bitwise tests and
separate quiet benchmark are in `covariance_limber_batches.md`.
This is one source's Gaussian covariance, not all-probe FoM convergence.
Evidence: check_gaussian_boost.py and results/gaussian_boost_sequence.*.

A separate physical-shell test projects all five matter cNG terms and the
SSC response at a=0.75 and a fixed representative distance of 1000 Mpc/h.
Four density band operators cover 30..99, 100..399, 400..1199 and 1200..4000.
There is no line-of-sight integral in this diagnostic. It raises mass/tree
quadrature, response-step and nested multipole resolution together, then
refines only the multipole table once more at fixed boost-8 quadrature.

| Refinement | SSC shell change | cNG shell change |
|---|---:|---:|
| 1 to 2 | 7.9911e-2 | 1.3690e-1 |
| 2 to 4 | 6.1263e-2 | 4.3712e-2 |
| 4 to 8 | 5.4588e-3 | 1.1407e-2 |
| Table-only doubling above 8 | 1.4192e-3 | 2.6133e-3 |

Here the metric is max|Delta C_ij|/sqrt(|C_ii*C_jj|), using the finer
component's diagonal. These are component diagnostics, not generalized
modes of a total covariance. The extra table doubling is not a new public
boost=16. See check_ng_shell_boost.py and results/ng_shell_boost.*.

### Remaining copied-power derivative floor

The fixed-range nested response ladder still stalls near its worst query:
at a=0.75, 241/481/961 response nodes give maximum relative interpolation
errors 0.00793978 / 0.00793059 / 0.00786841. Increasing a covariance-table
count alone is therefore insufficient in this example.

An input-copy control retains all 1500 original log-k nodes and their
linear, cb and nonlinear log-power samples exactly, then fills a nested
11993-node table in two different ways. Linear midpoint insertion preserves
the same piecewise-linear power function and leaves the response-table
floor essentially unchanged. Direct responses move by at most 1.42e-6;
changing the supplied k sampling also changes internal halo preparation.

Cubic construction between the original samples, followed by ordinary
linear reads in C, instead gives:

| Response nodes | Original power copy | Cubic-dense power copy |
|---:|---:|---:|
| 241 | 7.9398e-3 | 4.5312e-3 |
| 481 | 7.9306e-3 | 1.1844e-3 |
| 961 | 7.8684e-3 | 3.1093e-4 |

Errors compare interpolation against direct responses using the same
power-copy choice. Direct responses between those choices change by up to
0.477%; cubic construction changes the function between supplied samples.
This isolates a practical floor associated with the slope changes of the
copied power interpolant, which feed the SSC dilation derivative. It does
not prove that this cubic reconstruction is the exact CAMB derivative.
The production CAMB/power-copy policy was deliberately not changed by this
covariance-grid fix. Validate dense nested copying against CAMB's own
interpolator and physical covariance/Fisher results before adopting it.

Evidence: check_response_input_floor.py and results/response_input_floor.*.
The measured maximum/RMS curves were rendered and visually inspected in
results/response_grid_convergence.png and .pdf, including the logarithmic
axes and the distinction between each power-copy choice's own reference.
All paths in these measurement sections are within the external
covariance_reference directory. The tests compare numerical choices within
the supported massless model, not the accuracy of its physical prescription.

### Regression and didactic review

All 116 project covariance checks pass: LSST 65, DES cluster 46, and one
shared-adapter check in each of the five remaining projects. New checks
cover exact inherited coordinates, cutoff bracketing without stretching,
owned trimmed arrays, invalid grids and saved-grid roundtrips. The existing
tests retain all component/reference and one/eight-thread comparisons.

The manual didactic pass checked intervals versus points, fixed coordinates
versus values refined by quadrature, table support versus measured cutoff,
Fourier trimming, distinction from Gaussian rules and fixed CAMB inputs.
There are no new C loops or SIMD intrinsics. Public documentation and all
seven notebooks explain the nested sampling and its limits without claiming
that positive-definiteness or a single boost establishes convergence.
