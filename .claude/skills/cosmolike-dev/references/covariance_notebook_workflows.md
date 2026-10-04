# Portable covariance notebook workflow

The owner requested public Python examples in LSST Y1, reusable algorithms
in `cosmolike_notebook_utils`, an `EXAMPLE_EVALUATE_COVARIANCE.ipynb` entry
point, and covariance plotters informed by Krause papers. Bash harnesses and
machine-local libraries are not part of the public workflow.

## Boundaries

- `python_components_cov.cpp` binds the existing eight covariance C
  components to owned NumPy arrays. Shapes, dimensions and physical
  domains are checked before entering C. Project-local compilation links
  the shared sources. Only LSST currently enables all these bindings.
- `covariance/geometry.py`, `gaussian.py`, `halo.py`, `sampling.py` and
  `diagnostics.py` receive an explicit interface or numeric arrays.
  Setup is cold; numerical array contractions, dense queries and C calls
  are hot. No project interface is imported by the package.
- `plot_covariances.py` uses only NumPy/Matplotlib. It follows the existing
  show=1 / show=None convention and does not set global plotting styles.
- `reference/` contains separately implemented numerical oracles.
  Production code must never use them to compute an expected result.
- LSST's `covariance/lsst_y1_covariance.py` owns survey choices and the
  initializer. The notebook imports its thin wrapper. Initialization does
  not load the project's likelihood covariance or modify data-vector grids.

## Example physics

Five lens/source bins use the shipped normalized n(z) files. The explicit
forecast assumes total lens/source densities 18/10 arcmin^-2 equally split
between five bins, 12300 deg^2, sigma_e=0.26 per component, mnu=0, no IA,
photo-z shifts or magnification. These are not the frozen likelihood's
parameters. The SRD guides the totals, not the per-bin allocation.

The first example computes one source's 52x52 xi+/xi- Gaussian covariance,
with all crossed angular spectra available, Limber spectra, full-sky bin
operators and cap-mask pair noise. It does not claim SSC/cNG or non-Limber
survey completeness. Both test resolutions use the same physical model.

The initial headless notebook execution produced minimum correlation
lambda=0.3376751679 and max generalized variance-ratio departure 1.02148e-4
when doubling ell, radial, angular and mask resolution. This is a pilot
numerical check, not a certified FoM setting or a speed benchmark.

## Figures and didactic review

Inspected five rendered outputs: split correlation triangles, two component
maps plus histogram, xi+/xi- standard deviations, percent error changes, and covariance-difference maps.
Labels, colorbars, signs and zero-reference handling were checked.
The component-plot design follows Barreira, Krause & Schmidt (2018)
arXiv:1807.04266 Fig. 1; split correlation triangles follow Friedrich et al.
(2021), arXiv:2012.08568 Fig. 6. The notebook shows its own computed Gaussian
pieces, not invented SSC/cNG or the papers' data. Element denominators near
zero are explicitly masked; diagonal normalization is also available.

The self-review checked source/field ordering, shape-noise convention,
analytic white-noise replacement, cubic construction vs linear queries,
input ownership, signed response handling and cross-block completeness.
C++ lines stay <=80 columns. No separate Fable review was available.

## Tests and execution

The old eight component suites no longer require external environment
paths. They call the normal LSST extension; raw C ABI calls are retained
only for output-canary tests and independent physical sampling. Temporary
artificial survey/CAMB files are deleted after tests. All 53 covariance
tests pass, including seven notebook API/plot tests and the new public halo
batch/response check. The notebook was executed headlessly with nbconvert;
its outputs are from committed cells, with no hand-entered scientific data.

The first combined sector run found a CAMB import-path conflict: covariance
setup chose site-packages, then Cobaya required external_modules/code/CAMB.
The covariance conftest now chooses the same CAMB source before import.
This is path selection, not replacement of an imported module. The combined
rerun passed all 109 collected tests. One additional notebook-helper check
was added after that run started; the final covariance-only run includes it
and passes all 53 checks.

Data-vector modules are in tests/data_vector; covariance modules in
tests/covariance. Frozen files, hashes and reference generator stay at
parent tests and are unchanged. The ordinary documented command selects
only data_vector. A covariance-only command locates the built interface
through its own conftest. The plot tests draw a noninteractive canvas and
inspect numerical artist data; manual image inspection complements them.

No runtime speedup is claimed for the new Python workflows. Quiet-machine
halo loop comparisons belong to covariance_optimization_review.md, and
must not overlap notebook execution or test runs.

## One accuracy boost

Supported boosts 1,2,4,8 resolve projection and halo preparation controls.
Every doubling raises signal/mask cutoffs, radial and angular quadrature,
lensing-window intervals, halo mass and angular counts. Tree angular panels
increase by one; the centered response step halves. The interpolation and
physical model still constrain attainable accuracy; no automatic production
FoM certificate is claimed. Invalid boosts fail before initialization.

The final executed notebook compares boosts 1,2,4 at fixed CAMB inputs and
physical settings. The generalized variance changes relative to boost 4 were
1.043470e-4 (boost 1) and 1.102522e-5 (boost 2). Maximum error-bar changes were
2.410347e-5 and 2.644153e-6. All three matrices were positive definite. The
reference's self-comparison differs from unity only at roundoff. These are
Gaussian one-source example results, not full-survey accuracy statements.

Final plots use Cocoa's STIX/retina notebook settings, distinct line styles,
legends above panels and percent changes. All five final PNG outputs were
visually inspected. LSST's covariance README has a numbered contents list,
explicit anchors and self-contained Step flows; MarkdownIt rendered its
three tables and four callouts and found no broken local links or anchors.

Validation commands from configured Cocoa:

- `python -m pytest -q projects/lsst_y1/tests/covariance`: exit 0, 53 passed.
- `python -m jupyter nbconvert --to notebook --execute --inplace
  --ExecutePreprocessor.timeout=600
  projects/lsst_y1/EXAMPLE_EVALUATE_COVARIANCE.ipynb`: exit 0; 13 cells,
  five figures, outputs committed from execution. Jupyter needed local
  socket permission in the agent sandbox; this is not a project limitation.
- Default LSST compilation including all eight C components and both
  covariance C++ sources: exit 0.
- `git diff --check`: passed before commits.

All seven project regression runs completed successfully without refreezing:

| Project | Passed checks |
|---|---:|
| roman_real | 104 |
| roman_fourier | 45 |
| roman_kl | 49 |
| des_y3 | 63 |
| desy1xplanck | 45 |
| des_cluster | 29 |
| lsst_y1, combined covariance/data-vector rerun | 109 |

The final covariance-only run separately passes 53 checks. The combined
LSST run reports 15 warnings from the existing emulator dependencies;
none is a test failure. The original multi-project coordinator retains
its nonzero exit for the first LSST import-path failure. Keep that log:
the explicit combined rerun, not a suppressed failure, establishes the
fixed result. External `results/covariance_scaling/project_tests_final.json`
records both attempts. Concurrent regression elapsed times are not timing
evidence. The main repository and all seven project working trees are clean.

The seven final README render/link checks passed, including the subsequently
edited test instructions. No frozen data or reference manifest changed.
