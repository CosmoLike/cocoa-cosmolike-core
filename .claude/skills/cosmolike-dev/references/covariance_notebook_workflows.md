# Portable covariance notebook workflow

Shared algorithms live in `cosmolike_notebook_utils`; each project's
`EXAMPLE_EVALUATE_COVARIANCE.ipynb` imports a thin survey adapter. Public
examples depend on the installed project interface rather than external
shell harnesses or machine-local shared libraries. The production CLI and
notebook boundaries are described in `covariance_interface_rollout.md`.

## Responsibilities

- Covariance bindings validate shapes and physical domains before C calls
  and return arrays with a defined lifetime. Every supported project
  binds the shared components; algorithms do not live in project copies.
- `covariance/geometry.py`, `gaussian.py`, `halo.py`, `sampling.py` and
  `diagnostics.py` receive explicit interfaces or arrays. The package
  never imports a particular project's compiled interface.
- `plot_covariances.py` uses NumPy and Matplotlib, follows the existing
  show=1 / show=None convention and leaves global styling to the notebook.
- Independent `reference/` implementations provide numerical oracles.
  Production routines never compute their own expected test result.
- Survey adapters own bin layouts, physical inputs and initialization.
  Forecast initialization does not load the likelihood covariance or
  modify the ordinary data-vector grids.

## Example interpretation

The original LSST one-source pilot used five lens/source distributions,
12300 deg2, total lens/source densities 18/10 arcmin^-2 split equally,
sigma_e=0.26 per component, massless neutrinos and zero IA, photo-z shifts
and magnification. These are explicit forecast choices, not a claim to
reproduce the frozen likelihood model. The SRD guides the total densities,
not the assumed equal per-bin allocation.

Its 52×52 xi+/xi- Gaussian matrix used all crossed angular spectra,
Limber projection, full-sky bin operators and cap-mask pair noise. The
saved comparison with boosts 1/2/4 at fixed CAMB inputs gave maximum
variance-ratio departures from boost 4 of 1.043470e-4 and 1.102522e-5.
Maximum error-bar changes were 2.410347e-5 and 2.644153e-6. All three
totals were positive definite. This is one-source Gaussian validation,
not full-survey, SSC/cNG or Fisher convergence.

Global boost and quadrature refinement are separate controls. See
`covariance_defaults.md` for their current definitions. An old notebook's
saved settings identify that measurement; current defaults must never be
substituted into its provenance.

## Figures and explanations

Useful diagnostics include split correlation triangles, component maps,
standard deviations, percent error changes and covariance differences.
The component-map convention follows Barreira, Krause & Schmidt (2018),
arXiv:1807.04266 Fig. 1; split triangles follow Friedrich et al. (2021),
arXiv:2012.08568 Fig. 6. Plot actual computed components and state each
denominator. Mask near-zero element denominators; offer variance-scale
normalization when relative component errors are misleading.

Use shared frameless legends and shared labels where meanings match.
Keep formulas and interpretation in the guide rather than figure footers.
Inspect labels, colorbars, signs, matrix boundaries and zero-reference
handling on saved images. Numerical artist checks complement visual
inspection; neither replaces the other.

## Reproduction and regression coverage

From an activated Cocoa installation, run the covariance tests with
`python -m pytest -q projects/lsst_y1/tests/covariance`. Execute the actual
notebook cells through nbconvert and retain outputs from that execution.
A kernel that cannot bind a local socket has not executed a numerical
check; fix the launch environment without altering scientific settings.

Tests use the normal project extension. Direct C ABI calls are reserved
for buffer-canary tests or independent sampling. Temporary artificial
inputs remain separate from supplied data and frozen references. The
covariance and likelihood paths must select the same CAMB installation
before import; a site-packages/external-modules mismatch caused the
recorded collection failure without a numerical-code defect.

The pilot validation passed 53 covariance tests and a combined 109-test
LSST run. The six other projects passed 104/45/49/63/45/29 checks for
Roman real/Fourier/KL, DES Y3, DES×Planck and DES cluster respectively.
The original failing import log and successful unchanged rerun remain in
`results/covariance_scaling/project_tests_final.json` under the external
reference directory. Counts identify that validation configuration, not
the current test collection. No frozen output changed.

Notebook execution times and concurrent test runtimes do not measure a
speedup. Controlled component benchmarks belong to
`covariance_optimization_review.md`; they run without competing builds,
notebooks or numerical tests.
