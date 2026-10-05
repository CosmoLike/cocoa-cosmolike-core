# Production CLI covariance timings, 2026-10-05

Apple M2 Pro, macOS 13.7.5; eight OpenMP workers, BLAS fixed to one.
All numerical jobs ran sequentially, with no concurrent builds, tests or
other numerical jobs. Desktop applications remained open. Each sample
used a fresh Python process and the shipped EXAMPLE_EVALUATE_COVARIANCE.yaml.
There were no covariance warm-ups. Three runs per project used the direct
production backend and retained their full output archives.

## Timing definition and models

Construction includes first-use CosmoLike tables, all spectra, halo response
and trispectrum tables, transformations and full G + SSC + cNG assembly.
CAMB/initialization, imports, output writing and diagnostics are excluded.
The separately measured CLI wall time includes imports, setup and output.
All projects use accuracy_boost=1, integration_accuracy=0, massless neutrinos
and zero IA. No numerical setting was changed for this benchmark.

The six galaxy/shear examples enable Gaussian non-Limber gg and gs.
Shear-shear and SSC/cNG retain their Limber model. The joint DES cluster
6x2pt + N example still requires Limber throughout. DESxPlanck generates
only its 1500-entry galaxy/shear sector; there are no CMB-lensing blocks.

| Project | Space | Size | Construction mean +/- sample SD (s) | Range (s) | CLI wall mean (s) |
| --- | --- | ---: | ---: | ---: | ---: |
| lsst_y1 | real | 1560 | 68.336549 +/- 0.958020 | 67.231806--68.938504 | 69.813259 |
| roman_real | real | 2115 | 74.760804 +/- 0.275426 | 74.570379--75.076614 | 76.000021 |
| roman_fourier | fourier | 1485 | 33.089472 +/- 0.193284 | 32.977021--33.312655 | 34.270121 |
| roman_kl | fourier | 2200 | 41.949737 +/- 0.698427 | 41.149481--42.436427 | 43.234545 |
| des_y3 | real | 900 | 64.807446 +/- 0.216941 | 64.665388--65.057159 | 66.070394 |
| desy1xplanck | real | 1500 | 67.236778 +/- 0.414921 | 66.767199--67.553917 | 68.451642 |
| des_cluster | real | 2812 | 125.244230 +/- 1.571918 | 123.660763--126.804340 | 126.556696 |

## Validation

G, SSC, cNG and total are finite, exactly symmetric and bitwise identical
across all three fresh processes for each project. The total equals the
component sum. All six galaxy/shear totals are positive definite before
likelihood cuts. The cluster layout contains 48 exact Y-localization null
rows; its documented 2764-entry complement is positive definite. No
eigenvalue clipping, jitter or diagonal regularization was used.

| Project | Minimum total correlation eigenvalue |
| --- | ---: |
| lsst_y1 | 0.000226984910404 |
| roman_real | 0.000591024228587 |
| roman_fourier | 0.000251301085683 |
| roman_kl | 5.96663314328e-06 |
| des_y3 | 0.0191960569164 |
| desy1xplanck | 0.00698882400783 |
| des_cluster | 1.782053692e-05 |

These checks establish repeatability and positivity for these saved
realizations. They do not establish interpolation, integration or Fisher
convergence. The cancelled LSST integration sweep was not resumed.

## Matched LSST notebook-interface comparison

A fresh notebook-backend call used exactly the same resolved inputs and
eight-thread setup as the CLI. It ran outside Jupyter and excluded plotting
and file writing, isolating the numerical interface choice. Every saved
covariance component, signal, geometry and measurement metadata agreed
bitwise with the production output.

| Interval | Production CLI mean, three runs (s) | Notebook backend, one run (s) |
| --- | ---: | ---: |
| Full construction | 68.336549 | 177.743270 |
| Shared matter response and trispectrum | 61.435975 | 169.442774 |

The measured full-runtime ratio is 2.600999. Almost all extra time is in
the shared matter-table stage. Its 672 shells repeatedly exchange arrays
with 8256 mode pairs and 1920 angular samples. The notebook bindings make
private column-major copies for Armadillo, and the wrappers pack row-major
C workspaces. The direct bindings pass contiguous NumPy rows to the same
C kernels. This comparison measures the combined interface overhead; it
does not separately time each copy, validation pass or allocation.

Use the production CLI as the public runtime baseline. Historical notebook
runtimes alone are not a controlled measurement of optimization gains.

## Reproduction and saved evidence

The normal project compute_covariance.py runners read their shipped YAMLs.
All thread counts came from OMP_NUM_THREADS=8, not YAML keys. Real-space
ell_max is 100000; Fourier bands end at each project's fixed measured range.
Exact resolved settings, stage times, commands and per-run logs are saved
with the full matrices. Binary hashes and source commits are in validation.json.

External development artifacts under test/covariance_reference:

- benchmark_nonlimber_cli.py: sequential fresh-process timing runner.
- validate_nonlimber_cli_timing.py: post-timing matrix and repeatability checks.
- benchmark_notebook_same_settings.py: matched LSST notebook-backend call.
- results/nonlimber_cli_20261005/: first five projects, validation.json,
  timing.json and notebook_comparison.json.
- results/nonlimber_cli_all_projects_20261005/: DES Y3 and joint DES cluster.

The timed core commit is f9c84ea. Project source and binary hashes are
recorded separately. Subsequent repository edits only update documentation.
