# Histogram reconstruction: model, implementation and numerical validation

A histogram specifies probability in redshift cells, but does not determine
its distribution within each cell. A smooth n(z) therefore introduces a
reconstruction model. The original midpoint curve and histogram preparation
followed by Akima are two such models; numerical stability alone cannot
establish which is physically more accurate.

## Model definition

Every likelihood YAML in the eight survey projects exposes:

| Control | Meaning |
|---|---|
| `photoz_histogram_akima: 0` | Original midpoint reconstruction; the default. |
| `photoz_histogram_akima: 1` | Uniform histogram preparation followed by Akima. |
| `photoz_histogram_akima_factor: 3` | Preparation factor; allowed values are 3, 5 and 7. |

Mode 0 retains the existing interpolation and redshift-column conventions.
Mode 1 always uses Akima, so `photoz_interpolation_type` has no effect on
that mode; `photoz_zmid_convention` still determines the input cell edges.
Z_LOW supplies left edges, with the last width repeated. Z_MID supplies
centres, with edges halfway between centres and half-spacing outer edges.

For an input histogram with N cells, preparation creates mN uniform cells,
where m is the selected factor. On uniform input this repeats each density
m times. Odd factors retain a prepared centre at each original centre.
For irregular input, each prepared density is the sum of overlapping input
probabilities divided by the prepared cell width. This remapping conserves
total area, but need not retain the original centres.

Akima joins the prepared centres with cubic polynomials. A repeated density
half a prepared cell outside each histogram edge allows evaluation over the
whole support without extrapolation. Queries outside that support return
zero. Input normalization precedes interpolation; no subsequent clipping or
area correction is applied. Signed input cells are retained.

The preparation factor is a **model choice**, independent of every accuracy
control. Raising it narrows the transitions between histogram cells and
changes n(z). An accuracy scan must keep it fixed.

## Uniform lookup and cache ownership

`nz_histogram_akima` in `redshift_spline.c` shares the source/lens algorithm
while keeping separate cached storage for the two samples. The public
selector `init_photoz_histogram_akima` connects likelihood initialization
to the reconstruction mode and preparation factor in `Ntable`.

Final lookup refinement subdivides the fitted polynomials algebraically.
If a piece is P(u), a smaller interval stores the coefficients of
P(u0 + h t). This is the same curve, not a second spline fit. Uniform
spacing gives direct interval indexing without binary search or allocation
in the reader. Both reconstruction modes require uniform lookup grids.

Storage is reused while its dimensions match. A changed histogram, edge
convention, preparation factor or lookup factor triggers reconstruction.
Selector/factor changes must also refresh support bounds, lens means and
dependent kernels through the sample setters. Cosmology changes and the
usual photo-z shifts/stretch do not refit n(z); the latter change query
coordinates and the density Jacobian.

Each tomographic row owns a GSL workspace and output coefficients. Allocate
and release workspaces serially, then prepare independent rows with
`omp parallel for schedule(static)`. There is no cross-thread reduction.
Comments should explain ownership, the relative-density tail cutoff and
coefficient transformations. The tail cutoff selects integration bounds;
it does not clip the stored distribution or bound the integrated tail mass.

## Probability conservation and redistribution

For uniform input cells, let p_i be the normalized mass in cell i and q_i
the area of the ideal reconstructed curve over that same cell. Repeated
endpoint densities and m = 3, 5 or 7 give

```math
q_i = p_i + \frac{3}{32m}(p_{i-1}-2p_i+p_{i+1}).
```

Use the endpoint mass for a missing neighbour. At m = 3 this becomes
q_i = 15p_i/16 + (p_{i-1}+p_{i+1})/32. The correction is a discrete second
difference: neighbouring gains and losses cancel in the sum, preserving
total area while changing individual cell probabilities. Increasing m
reduces the exchange by narrowing the smoothed transition.

This identity depends on uniform cells and the repeated-density
construction. For irregular input, conservation by the overlap remapping
does not imply conservation by the subsequent Akima fit. General Akima
interpolation also has no positivity guarantee.

Measure total area separately from redistribution. For nonnegative input
and reconstructed cell masses, the redistributed percentage is

```math
R = 100 \times \frac{1}{2}
    \sum_i \left|\frac{q_i}{\sum_j q_j}-p_i\right|.
```

The denominator sums reconstructed areas over every original redshift cell
of one tomographic distribution. The outer sum compares those normalized
areas with the input probabilities in the same cells. The half prevents
counting each transfer twice, as both a loss and a gain. This diagnostic
normalization does not renormalize the production reader. For signed
inputs, the same expression measures normalized area differences rather
than a literal fraction of probability transferred.

## Separating model sensitivity from numerical error

The survey metric is

```math
\Delta\chi^2 = (d_b-d_a)^{\mathsf T}C^{-1}(d_b-d_a),
```

with fixed physical inputs, mask and supplied covariance C. Comparing
reconstructions measures model sensitivity. Comparing accuracy settings
within one reconstruction measures numerical sensitivity. Neither is a
change in goodness of fit to observed data.

The reproducible study uses `accuracyboost: 5` and
`integration_accuracy: 7` for original/3/5/7, and separately evaluates the
original reconstruction at its YAML defaults. All quadrature rules are
capped at 8,192 points, including source/lens redshift moments. Native GSL
allocation and the independence of quadrature from interpolation controls
are described in [the integration reference](integration_accuracy_4096.md).

The original-default numerical distances range from 0.0021653 to 0.0249203
across the seven supplied physical covariances. In these refined
comparisons, every 5→7 distance is smaller than its 3→5 distance. The
results support practical stability relative to a Δχ² scale of 0.2 at the
stored NLA points; they do not establish a continuum limit or a preferred
physical reconstruction. Global boosts can still affect other interpolation
tables even when n(z) itself is invariant under lookup refinement.

DES Y6 supplies an identity metric rather than a physical survey covariance
and has no statistical Δχ² row. Data-vector convergence does not establish
covariance-generation convergence: coarse Akima window/radial refinements
have shown unresolved sensitivity and require their own checks.

## Validation and reproducibility

Routine regression coverage retains synthetic-reader checks and frozen
likelihood checks. The expensive reconstruction study is invoked separately
because its purpose is to compare modeling choices, not to enforce a frozen
model displacement.

- Check original → Akima → original and factor 3 → 5 → 3 / 3 → 7 → 3
  round trips, including agreement with fresh YAML initialization.
- Check source/lens readers on uniform and irregular histograms, with fixed
  preparation factor under lookup refinement. Boosts 1–4 preserve the curve;
  one/eight-thread evaluations are bitwise equal in the recorded checks.
- Audit areas from the compiled readers, splitting at every polynomial
  knot. Two- and four-point Gauss rules agree within 2e-13 per original cell.
- Keep physical inputs and covariance fixed in vector comparisons. Verify
  finite results, unchanged covariance and exact restoration of the original
  vector. Do not replace signed inputs with clipped inputs or refreeze a
  likelihood to conceal a numerical change.

Six reader cases and 17 frozen-default checks pass; strict and DEBUG/UBSan
reader checks show no sanitizer findings. All seven explicit study tests
reproduce the saved vectors, inverse covariances, area arrays and distances.
This reproducibility evidence is specific to the recorded configurations.

`cocoa_nz_study.py` implements the shared calculation and area audit.
Each physical-covariance project exposes
`tests/studies/run_nz_reconstruction.py` and
`tests/studies/study_nz_reconstruction.py`. Results and the interpretation of
figures belong in `tests/studies/README.md` and `tests/studies/results/`.
The shared plotter is `cosmolike_notebook_utils/plot_nz_reconstruction.py`;
project `generate_nz_reconstruction_figures.py` scripts redraw saved results.
Explanations and settings belong in the guide, not in figure footers.

The maintained numerical summary is `dev_scripts/nz_akima_manual_study.json`.
`dev_scripts/nz_akima_prior_native_refinement.json` records independent
boost/rule sensitivity. Earlier factor scans remain in the validation and
raised-accuracy JSON records; they are configuration-specific evidence,
not universal minimum integration levels. Composite-rule, clipped-input
and native-GSL results must not be mixed as interchangeable references.
