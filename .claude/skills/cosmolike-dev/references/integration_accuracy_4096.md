# Extended quadrature: numerical controls and validation

## Native GSL rules

`integration_accuracy` selects quadrature independently of interpolation
accuracy. Level 7 selects 8,192 points in ordinary Limber paths, cluster
projections/counts and the covariance accuracy resolver. Existing lower-level
choices remain intact. No fixed rule exceeds 8,192 points.

`malloc_gslint_glfixed` delegates to `gsl_integration_glfixed_table_alloc`
at every supported size and handles allocation failure. GSL owns node
generation and storage; callers own the cached table and its invalidation.
This avoids maintaining a separate implementation of GSL's internal storage
layout. Generated high-order rules are single Gauss-Legendre rules over
each physical panel, not copies of a lower-order rule on subpanels.

Source/lens mean-redshift integration in `redshift_spline.c` also refines:
levels 0–5 select 256/512/1024/2048/4096/8192 points, with levels 6–7 capped
at 8192. `redshift_spline_cluster.c` instead uses refining trapezoid grids.
Uniform n(z) lookup is mandatory; there is no compile-time GSL-reader bypass.
The reconstruction factor remains independent of both quadrature and lookup
refinement, as explained in [the histogram reference](nz_histogram_akima_option.md).

## Accuracy criteria

Generated GSL rules have measured absolute polynomial-moment errors below
4e-10. Their analytic checks use an absolute 1e-9 tolerance; stored rules
retain tighter checks. Relative errors are inappropriate for moments near
zero. This quadrature-unit criterion does not relax survey-level Δχ² or
covariance-convergence requirements.

Checks cover symmetric ordered nodes, positive weights, polynomial moments,
affine interval maps and both GSL consumers. Targeted covariance checks also
exercise radial spectra, mass moments, angular operators and tree angles.
Successful rule checks establish the numerical building blocks; they do not
certify the resolution of every physical integrand.

The histogram study holds physical inputs and covariance fixed while varying
interpolation and quadrature separately. Its practical stability scale is
Δχ² = 0.2 at the stored fiducial points. Escalation beyond the tested settings
requires evidence of a material unresolved error, rather than an assumption
that every small remaining difference warrants another global refinement.
Covariance convergence requires its own matrix and inference diagnostics.

## Evidence and reference compatibility

Native-rule and targeted covariance-sector checks pass, together with six
histogram-reader cases and 17 frozen-default checks. Seven explicit
reconstruction studies preserve the original default vectors bitwise and
check exact restoration after changing reconstruction. Their maintained
summary is `dev_scripts/nz_akima_manual_study.json`; DES Y6 is excluded from
statistical distances because its identity metric is not a survey covariance.

Detailed native-rule measurements and test outputs are in the external
`test/nz_akima_three_five/native_8192/`, `native_checks_8192_final/` and
`native_refinement_8192/` records. The committed histogram reference explains
the corresponding model comparisons and their limits.

Composite rules assembled from the stored 1024-point rule use different
nodes from native GSL rules of the same total size. Their saved results
therefore provide a method comparison, not a bitwise target for the native
implementation. Preserve that distinction when reading the earlier
`test/nz_histogram_experiment/results/extended_rules_checks.json` and
`extended_covariance_sector_checks.json` records. Neither implementation's
small-fixture tests alone establish full-survey covariance convergence.
