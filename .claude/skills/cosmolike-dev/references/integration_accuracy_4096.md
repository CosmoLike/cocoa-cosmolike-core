# Extended integration levels

## Current experimental branch: native GSL through 8192 points

On `experimental/nz-akima-3-5-7`, level 7 selects 8192 points in all
ordinary Limber paths, cluster projections/counts and the covariance
accuracy resolver. Existing lower-level choices remain unchanged.
`malloc_gslint_glfixed` now calls `gsl_integration_glfixed_table_alloc`
directly at every supported size. GSL owns node generation and storage;
the wrapper only handles allocation failure. Callers retain their caches.

The user requested this simpler implementation instead of constructing
composite rules. The generated high-order rules differ from the earlier
composite rules, so those earlier vectors are historical results, not
bitwise regression targets for current high-order calculations.

Native GSL's generated rules have absolute polynomial-moment errors below
4e-10 in the local checks. The user approved a practical 1e-9 absolute
tolerance for these checks; stored rules retain their tighter tolerance.
Do not turn this into a relaxation of the survey Delta chi2 requirements.

Current records are under `test/nz_akima_three_five/`: `native_8192/`,
`native_checks_8192_final/` and `native_refinement_8192/`. The native rule,
angular/mass/tree-sector checks and all 17 frozen-default checks passed.
The six histogram-reader cases also passed at interpolation boosts 1–4.
The user accepts practical stability relative to Delta chi2 0.2. Do not
continue global-boost escalation in pursuit of an exact continuum limit.
The permanent manual study is `cocoa_nz_study.py`, called explicitly by each
survey's `tests/studies/run_nz_reconstruction.py`; it is not collected by
routine pytest. All seven `manual_study_cap8192/` runs completed and were
verified with fixed inputs/covariances and original cache round trips.
All seven original default vectors remain bitwise unchanged. Six reader
checks and 17 frozen-default checks passed; ordinary collection in all
eight projects excludes these expensive studies. Current published records
are in `dev_scripts/nz_akima_manual_study.json`.
DES Y6 has no physical covariance and is excluded from statistical distances.

The source/lens mean-redshift rules in `redshift_spline.c` now refine too:
256/512/1024/2048/4096/8192 at levels 0/1/2/3/4/5, capped at 8192 for
levels 6 and 7. The user explicitly rejected extending any rule beyond8192.
The defaults remain unchanged. `redshift_spline_cluster.c` uses refining
trapezoid grids and has no capped fixed-Gauss selector to extend.
The old `DONT_NZ_FAST_SUMBSAMPLE` bypass is retired: all galaxy n(z) readers
prepare uniform lookup grids, whether the reconstruction is cubic or Akima.

## Historical composite implementation (retired)

`integration_accuracy` retains the existing defaults and levels 0–4.
Level 5 selects 2048 points and level 6 selects 4096 points in every
ordinary data-vector Limber path, including cluster projections and
cached derivative paths. The shared covariance accuracy resolver exposes
the same levels for radial, angular, mass and tree-angle integration.

The new sizes are **composite rules**, not single Gauss-Legendre rules of
orders 2048 or 4096. `malloc_gslint_glfixed` maps GSL's stored 1024-point
nodes onto two or four equal subintervals. It stores the positive half
of the symmetric result in GSL's public descriptor layout, with owned
arrays and `precomputed=0` for ordinary GSL destruction. Existing sizes
still call the original allocator. The caller's cached descriptor is
rebuilt only when its existing numerical-settings cache invalidates.
GSL's public implementation documents the symmetric storage and free:
https://github.com/ampl/gsl/blob/master/integration/glfixed.c

The old rule choices differ between consumers: clustering and cluster
counts already reach 1024 at level 3, while shear and most other Limber
paths reach it at level 4. Those choices are preserved. The covariance
ladder is 96/128/256/512/1024/2048/4096; its Wynn-tail ladder is
32/64/128/256/512/1024/2048. The latter therefore also uses the composite
helper at its final level. Arbitrary generated orders remain unsupported
at covariance interfaces. The global interpolation `accuracy_boost`
does not select quadrature sizes.

Validation in `nz_histogram_experiment/results/`:

- `extended_rules_checks.json`: all supported stored rules plus 2048/4096
  have ordered symmetric nodes, positive weights, accurate analytic
  moments and affine-map integrals through both GSL consumers. Rules
  through 1024 reproduce GSL arrays bitwise.
- `akima_fourier_refined.json`: Roman Fourier and KL, both original and
  Akima n(z), levels 4/5/6, matched inputs/masks/covariances. All four
  level-4 vectors reproduce the previous modules bitwise. Level-6
  representation distances are 0.491420 and 0.636057; the last Akima
  refinement distances are 4.49e-5 and 4.81e-7.
- `extended_covariance_sector_checks.json`: both new sizes passed public
  halo-moment, real-space-operator and tree-angle checks. Maximum mass
  changes from 1024 were 2.22e-7 fractionally; angular changes were below
  6.7e-15 absolute. These are small-fixture checks, not full survey
  covariance convergence.
- `extended_covariance_original_checks.xml`: 23 tests passed in the LSST
  notebook-tools and covariance-spectra modules, including new extended
  rule tests. The Akima counterpart retains two coarse-resolution
  failures described in `nz_threefold_akima.md`; do not relax those tests
  or call Akima production-validated.

External builds preserve original strict compilation commands, source
and module hashes. Installed interfaces are unchanged. This validates the
new rules and selected consuming paths, not convergence of every survey
at a chosen level, a full new regression release, or a timing claim.
