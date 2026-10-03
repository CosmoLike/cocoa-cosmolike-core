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
