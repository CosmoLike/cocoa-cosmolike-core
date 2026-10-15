# Covariance defaults and independent integration controls

Each project owns `covariance/default.yaml`. Its resolved settings must
identify the physical model, the base controls and the effective numerical
grids. A numerical comparison is meaningful only when the physical model
and estimator remain fixed.

## Independent controls

The global `accuracy_boost` refines interpolation intervals, multipole
cutoffs and shared reader tables. Internal factors multiply this boost;
old interpolation nodes survive doubling. Notebook refinement cells pass
saved `accuracy_parameters` back to the resolver rather than silently
reverting to generic defaults.

`integration_accuracy` selects quadrature independently. Native GSL rules
extend through 8,192 points; the supported levels and allocation contract
are in [the integration reference](integration_accuracy_4096.md). The
initializer also passes the level to shared halo/cluster readers, which
retain their own lower-level ladders. The low-mass Wynn tail has a dedicated
lower baseline, justified by its smooth integrand and separate checks in
[covariance_mass_cutoff.md](covariance_mass_cutoff.md).

Low-level bindings expose supported rule sizes for component checks.
`covariance_integration_rule` exports GSL nodes and weights for the
Python tree-angle and selected-cluster response calculations, keeping
both sides on the same quadrature convention. A small isolated integral
passing at fewer nodes is not sufficient to lower a shared production floor.

## Resolving angular oscillations

One fixed rule across LSST's widest angular bin aliases the high-ell
oscillations: the tested 512/1024-node rules fail at ell=100000.
`realspace_operator_cov` therefore divides each measured bin into panels
with ell_max × panel_width <= 128. Their boundaries depend on the physical
bin and cutoff, not integration level. Refinement changes the rule within
those panels, while normalization uses the full measured annulus area.

An integration panel is not a new observable bin. The spin factors,
recurrence and output ordering remain those of the measured estimator.
Scratch accommodates the largest panelled node count; each bin evaluates
only its own nodes, with one worker owning the complete sum. This separates
sampling requirements from the estimator definition.

## LSST diagnostic: integration error versus table error

The following 2026-10-04 measurements use the fixed Limber, zero-IA model.
They are not current production timing figures. Non-Limber/NLA/TATT
validation is separate in `covariance_nonlimber_ia_plan.md`; comparing a
different Gaussian model with these arrays would mix physics and numerics.

Eight-thread full LSST runs use identical archived CAMB inputs and the
same 128-node non-Gaussian table. Both 1560-square totals are positive
definite, without repair. Level 0 takes 67.2448 s; level 1 takes 120.3022 s.
The generalized variance ratios C_level0/C_level1 span
[0.9998014740,1.0002049439], a maximum change of 2.04944e-4.
Level 1 compared with the previous radial128/mass512/tree256 reference has
maximum full-mode change 1.41127e-7. These are integration diagnostics,
not validated production timings; these values do not resolve interpolation error.

At fixed earlier high quadratures, refining the non-Gaussian table from
128 to 255 nodes gives total variance ratios [0.9880792314,1.0404942380].
Thus the apparently small Frobenius change (3.04e-4) hides a 4.05% change
in some full-matrix directions. Both totals remain positive definite.
The fine run took 901.3341 s before the GSL-only integration changes.
Evidence is in external `covariance_reference/results/default_lsst_ng_refined`
and `gsl_lsst_level0`/`gsl_lsst_level1`; these paths are development records,
not public README instructions.

Levels 2 and 3 completed at 476.4301 s and 2173.2408 s respectively.
Both complete totals are positive definite without repair. Maximum full-mode
changes are 1.39029465e-4 (level 1 against 2) and 8.26002339e-5 (2 against 3).
For the latter, component changes normalized to the total level-3 covariance
are 1.56556e-6 (G), 8.22305e-5 (SSC) and 3.91331e-6 (cNG). This diagnoses
SSC as the dominant remaining integration difference in this comparison.

The level-4 directory contains an incomplete run and provides no matrix for
comparison. The valid conclusions are limited to the completed levels;
this reference does not schedule another calculation.

The integration implementation's recorded regression included 143
covariance checks and 100 frozen data-vector checks across seven projects,
without reference changes. Such regression results establish compatibility,
not convergence of every covariance direction.

## Implications for accuracy assessment

The interpolation example above changes a full-matrix variance direction
by 4.05% despite a small Frobenius difference and positive totals. Test
interpolation, cutoff and integration controls separately; retain G, SSC,
cNG and total so that a dominant error can be located. Positive definiteness
and a high nominal resolution are necessary diagnostics, not convergence
certificates. Final scientific assessment follows parameter errors and
Figure of Merit as described in `covariance_accuracy.md`.
