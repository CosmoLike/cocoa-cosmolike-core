# Covariance defaults and independent integration levels

Status (2026-10-04): candidate defaults, with full-matrix convergence still
in progress. Do not publish the retired ell_max=10000 pilot timings as
usable covariance timings. The 128-node non-Gaussian table at ell_max=100000
also fails the proposed 1e-3 full-mode refinement diagnostic.

## Control contract

Each project owns `covariance/default.yaml`. Its loader resolves and saves
the base parameters and effective settings. The global `accuracy_boost`
refines interpolation intervals, multipole cutoffs and shared reader tables.
Internal factors multiply it; old interpolation nodes survive doubling.
Notebook refinement cells must pass the saved `accuracy_parameters` back
to the resolver, rather than silently reverting to generic defaults.

Quadrature is independent: `integration_accuracy` levels 0/1/2/3/4 select
the precomputed GSL 96/128/256/512/1024-node rules. The global boost does not
change these orders. The high-level initializer also passes the level to
the shared core for the halo/cluster readers. Existing core cluster readers
retain their own precomputed ladders, including the 64-node mass baseline.
Covariance C files outside `covariances/` were not changed.

The public component bindings retain explicit rule sizes for low-level
checks, accepting only 64, 96, 128, 256, 512 and 1024. The 64-node floor
applies even when a smaller rule happens to pass an isolated experiment.
`covariance_integration_rule` copies GSL's precomputed nodes and weights
to NumPy. The tree-angle helper and selected-cluster profile response use
that binding instead of generating SciPy rules. The latter uses two mass
panels, each with the selected rule, instead of one generated double-size rule.

## Angular convergence and didactic review

One Gaussian rule across LSST's widest bin aliases the high-ell oscillations:
at ell=100000 the old 512/1024-node rules fail badly. A generated 2048-node
rule was studied but is not an accepted production solution.

`realspace_operator_cov` now splits each measured angular bin into panels
with ell_max*panel_width <=128. Panel boundaries depend on the physical
bin and cutoff, not integration level. Each panel uses a precomputed rule;
all contributions divide by the full measured annulus area. No measured
bin changes. The recurrence, spin factors, SIMD lane order and deterministic
per-row ownership remain. Scratch uses the largest panelled node count;
each bin evaluates only its own real nodes.

The didactic review checked the distinction between an integration panel
and a measured bin, area normalization, the polynomial-resolution rationale,
flat node indexing and scratch ownership. New C lines remain within 80
columns. No SIMD operation was added or left without its existing explanation.

## Evidence so far

With the Cocoa environment active, the rebuilt LSST interface passed:

`python -m pytest projects/lsst_y1/tests/covariance -q`

92 tests, 27.48 seconds; log `/tmp/cov-gsl-ladder-lsst-tests.log`.
The high-ell scalar antiderivative check passes with composite rules of
64, 96 and 128 nodes per panel. Tests check GSL polynomial integrals,
rejection of 32/65/384/2048-node rules, the independent integration ladder,
global/internal table factors and exact interpolation-node retention.

Eight-thread full LSST runs use identical archived CAMB inputs and the
same 128-node non-Gaussian table. Both 1560-square totals are positive
definite, without repair. Level 0 takes 67.2448 s; level 1 takes 120.3022 s.
The generalized variance ratios C_level0/C_level1 span
[0.9998014740,1.0002049439], a maximum change of 2.04944e-4.
Level 1 compared with the previous radial128/mass512/tree256 reference has
maximum full-mode change 1.41127e-7. These are integration diagnostics,
not validated production timings: levels 2/3/4 and interpolation remain open.

At fixed earlier high quadratures, refining the non-Gaussian table from
128 to 255 nodes gives total variance ratios [0.9880792314,1.0404942380].
Thus the apparently small Frobenius change (3.04e-4) hides a 4.05% change
in some full-matrix directions. Both totals remain positive definite.
The fine run took 901.3341 s before the GSL-only integration changes.
Evidence is in external `covariance_reference/results/default_lsst_ng_refined`
and `gsl_lsst_level0`/`gsl_lsst_level1`; these paths are development records,
not public README instructions.

The linked interfaces were rebuilt sequentially. Frozen data-vector examples
passed in LSST Y1 (12), Roman real (12), Roman Fourier (12), Roman KL (12),
DES Y3 (24), DES Y1 x Planck (12) and DES cluster (16), without changing
frozen references. Each non-LSST galaxy project also passed its covariance
adapter test. DES cluster passed all 46 covariance tests. The totals are
143 covariance tests and 100 frozen data-vector tests across seven projects.
Notebook outputs were cleared because they described the retired pilot;
the revised cells have not been executed as a full notebook.

Pending: levels 2/3/4, including default
versus level4 and level3 versus4; converged non-Gaussian interpolation and
cutoff checks; validated project baselines and quiet eight-thread timing
table. Keep G, SSC, cNG and total comparisons separate. Do not equate a
positive matrix or the highest tested resolution with convergence.
