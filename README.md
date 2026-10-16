# Table of contents

1. [Overview](#overview)
2. [Running a likelihood example](#likelihood)
3. [Running the notebooks](#notebooks)
4. [Finding the code](#files)
5. [Covariance calculations](#covariance)
6. [Shared Python tools](#python)
7. [Running the tests](#tests)
8. [Appendix](#appendix)
   1. [FAQ: How is a data vector calculated?](#prediction)
   2. [FAQ: How do halos use massive neutrinos?](#neutrinos)
   3. [FAQ: What does the cluster code calculate?](#clusters)
   4. [FAQ: Which units do the interfaces use?](#units)
   5. [FAQ: How are accuracy and parallelism controlled?](#numerics)
   6. [FAQ: Why does the cNG read the power spectrum in log k?](#logk)
   7. [FAQ: What establishes a usable covariance?](#positivity)

# Overview <a name="overview"></a>

This repository contains the C calculations, C++ interfaces and shared
Python tools used by [Cocoa](https://github.com/CosmoLike/cocoa)'s
CosmoLike projects. A project supplies its survey data, redshift bins,
likelihood configuration and compiled Python interface. This repository
supplies the calculations that those projects share.

> [!WARNING]
> **CLI for production; notebook wrappers for exploration.**
>
> Run production and HPC calculations from YAML through the optimized
> `_interface` bindings. Notebook `_wrapper` APIs expose intermediate
> quantities for exploration; copying and rearranging their arrays adds
> overhead. Both routes call the same C kernels.
>
> Use project CLI entry points as the production performance baseline.
> The measured wrapper/CLI runtime ratio and its conditions are in
> [Cocoa's main README](https://github.com/CosmoLike/cocoa#overview).

The principal observables are galaxy clustering, galaxy–galaxy lensing
and cosmic shear, collectively called **3x2pt**. Galaxy clustering
measures correlations of galaxy positions. Galaxy–galaxy lensing measures
background-galaxy distortions around foreground galaxies. Cosmic shear
measures correlations between the distortions of background galaxies.

| Measurement | Angular spectrum | Real-space statistic |
| --- | --- | --- |
| Galaxy clustering | $`C_\ell^{gg}`$ | $`w(\theta)`$ |
| Galaxy–galaxy lensing | $`C_\ell^{g\gamma}`$ | $`\gamma_t(\theta)`$ |
| Cosmic shear | $`C_\ell^{\gamma\gamma}`$ | $`\xi_+(\theta),\xi_-(\theta)`$ |

A **data vector** collects these measurements across angular scales and
redshift-bin pairs. Its theoretical prediction is their expected mean.
A **covariance matrix** describes how their fluctuations are related.
The likelihood uses a supplied covariance to compare the prediction with
the data; constructing that covariance is a separate calculation.

The core also contains CMB-lensing cross correlations, halo-occupation
calculations, and a separate cluster sector. Which observables and model
options are available in an analysis depends on the project's interface
and likelihood. [Krause & Eifler](https://arxiv.org/abs/1601.05779)
describe the CosmoLike multiprobe framework.

> [!NOTE]
> Each project provides a real/Fourier G+SSC+cNG notebook for its galaxy
> and shear fields. Gaussian spectra support non-Limber gg/gs and NLA/TATT,
> retaining every internal cross-bin spectrum. SSC/cNG keep their zero-IA
> Limber model. Forecasts use massless neutrinos and a spherical-cap
> footprint. Numerical and Fisher convergence remain to be established
> before inference. DES cluster also has a joint angular 6x2pt+N forecast
> with explicitly limited cluster cNG and count-cross approximations.
> DES×Planck covers galaxy–shear only; CMB covariance remains separate work.

# Running a likelihood example <a name="likelihood"></a>

We assume users have installed Cocoa and its LSST Y1 project using the
[Cocoa installation guide](https://github.com/CosmoLike/cocoa#required_packages_conda),
activated the Conda environment with `conda activate cocoa`, and opened
Bash in `cocoa/Cocoa`. The core is compiled through each project; it is
not a standalone likelihood executable.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: compile the LSST Y1 interface against this core.

    source ./projects/lsst_y1/scripts/compile_lsst_y1.sh

**Step :three:**: evaluate the LSST Y1 cosmic-shear likelihood.

    cobaya-run ./projects/lsst_y1/EXAMPLE_EVALUATE1.yaml

The YAML file selects the cosmology, nuisance parameters, data, covariance
and numerical settings. The evaluate sampler prints the likelihood result
and writes its configured output. Other surveys have their own examples
and compilation scripts; see the
[Cocoa project guide](https://github.com/CosmoLike/cocoa/tree/main/Cocoa/projects).

> [!TIP]
> For the other LSST Y1 observables and sampler examples, see the
> [LSST Y1 project guide](https://github.com/CosmoLike/cocoa_lsst_y1).

# Running the notebooks <a name="notebooks"></a>

We assume users have installed Cocoa and LSST Y1, run `conda activate cocoa`,
and are using Bash in `cocoa/Cocoa`.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: compile the interface for the chosen notebook.

For the data-vector notebooks:

    unset IGNORE_COSMOLIKE_LSST_Y1_CODE
    source ./projects/lsst_y1/scripts/compile_lsst_y1.sh

For `EXAMPLE_EVALUATE_COVARIANCE.ipynb`, enable covariance generation:

    unset IGNORE_COSMOLIKE_LSST_Y1_CODE
    unset IGNORE_COSMOLIKE_LSST_Y1_COVARIANCE
    source ./projects/lsst_y1/scripts/compile_lsst_y1.sh

**Step :three:**: start Jupyter.

    jupyter notebook --no-browser --port=8888

**Step :four:**: open the URL printed by Jupyter and choose a notebook.

| Notebook in `projects/lsst_y1/` | Purpose |
| --- | --- |
| `EXAMPLE_EVALUATE1.ipynb` | Calculate and inspect cosmic-shear predictions. |
| `EXAMPLE_EVALUATE2.ipynb` | Calculate and inspect the joint 3x2pt prediction. |
| `EXAMPLE_EVALUATE_COVARIANCE.ipynb` | Construct, refine and plot real/Fourier G, SSC, cNG and total covariances. |

**Step :five:**: select **Kernel → Restart Kernel and Run All Cells**.

The covariance notebook uses the project's five lens and five source
redshift distributions and retains all 3x2pt measured rows. Its survey
assumptions are explicit in
`projects/lsst_y1/covariance/lsst_y1_covariance.py`; instructions are in
that folder's `README.md`. It does not overwrite the likelihood covariance.

# Finding the code <a name="files"></a>

Within a Cocoa installation this repository is located at
`Cocoa/external_modules/code/cosmolike_core/`. Paths below are relative
to this repository. The C headers describe the available calls; function
comments define their inputs, units and equations.

| Folder or file | Responsibility |
| --- | --- |
| [cosmolike/](cosmolike/README.md) | Source guide to data-vector physics, halo statistics, cluster extensions and C++ interfaces. |
| [cosmolike/covariances/](cosmolike/covariances/README.md) | Covariance components with their own numerical grids and integrations. |
| [cfastpt/](cfastpt/) | C implementation of FAST-PT mode-coupling integrals. |
| [cosmolike_notebook_utils/](cosmolike_notebook_utils/) | Shared cosmology preparation, plotting, Fisher and covariance tools. |
| [cocoa_hybrid_sampling.py](cocoa_hybrid_sampling.py) | Shared CPU/MPI hybrid minimization, profiles and Nautilus entry points, reading each project's evaluate YAML. |
| [cocoa_testing.py](cocoa_testing.py) | Shared machinery for project reference checks, independent processes and repeatability tests. |
| [log.c/](log.c/README.md) | C logging library. |
| [dev_scripts/](dev_scripts/) | Source-comment and function-header checks. |
| [future_port_unfinished/](future_port_unfinished/) | Uncompiled matter/tSZ halo work and experimental analytic Fisher derivatives. |

## Data-vector C files

| Source | What to look for |
| --- | --- |
| [cosmo3D.c](cosmolike/cosmo3D.c) | Supplied power spectra, distances and growth; FFTLog mass variances and their mass derivatives. |
| [redshift_spline.c](cosmolike/redshift_spline.c) | Lens/source redshift distributions, photo-z changes and allowed tomographic pairs. |
| [radial_weights.c](cosmolike/radial_weights.c) | Galaxy-density and lensing weights along the line of sight. |
| [cosmo2D.c](cosmolike/cosmo2D.c) | Angular spectra, Limber and non-Limber projections, full-sky real-space transforms and angular-bin averages. |
| [bias.c](cosmolike/bias.c) | Galaxy-bias parameters and redshift dependence. |
| [IA.c](cosmolike/IA.c) | Intrinsic-alignment amplitudes and their redshift parameterizations. |
| [pt_cfastpt.c](cosmolike/pt_cfastpt.c) | Perturbation-theory and intrinsic-alignment tables built with the shared C FAST-PT engine. |
| [halo.c](cosmolike/halo.c) | Halo abundance, bias, concentration, profiles, galaxy occupation and halo-model intrinsic alignment. |
| [baryons.c](cosmolike/baryons.c) | Baryonic modifications of the matter power spectrum. |
| [cosmo2D_scuts.c](cosmolike/cosmo2D_scuts.c) | Responses to physical wavenumber, used to diagnose which scales contribute to a data-vector point. |
| [tinker_emulator.c](cosmolike/tinker_emulator.c) | Legacy mass-function and halo-bias emulator code; excluded from the current project builds. |
| [basics.c](cosmolike/basics.c) | Integration, interpolation, allocation and numerical utilities. |
| [structs.c](cosmolike/structs.c), [structs.h](cosmolike/structs.h) | Shared cosmological, survey, nuisance and numerical state. |

The cluster files are described [separately below](#clusters). Covariance
C implementations live inside `cosmolike/covariances/` and end in `_cov.c`.
Additional covariance cross-bin calculations belong there, so their
requirements do not change data-vector pair selection or integration grids.

## C++ interfaces: production and notebooks

We chose **Armadillo** to make the Python notebook API easy to develop
and use: spectra, halo quantities and covariance components appear as
arrays with clear physical axes. Armadillo is a C++ library for vectors,
matrices and three-dimensional arrays called cubes.

C++ serves as a thin connecting layer. **pybind11** exposes the notebook
functions to Python, and **CARMA** converts between Armadillo and NumPy
arrays. The numerical calculations remain in the shared C code.

The C files own the numerical calculations. Two C++ layers expose those
same calculations for different uses:

| Layer | Intended use | Array handling |
| --- | --- | --- |
| `_interface` | Command-line runs and production likelihoods. | Keep numerical work in C and avoid notebook conversions. The covariance production path borrows contiguous NumPy inputs directly. |
| `_wrapper` | Jupyter exploration and intermediate quantities. | Accept and return named Armadillo vectors, matrices and cubes. Copies are acceptable when they make experimentation easier. |

**The wrappers are adapters, not a second numerical implementation.**
Integration, SIMD arithmetic, table building and matrix-assembly loops
belong in the shared C code. A fix to a physical calculation must therefore
reach both entry points without being implemented twice.

For example, both covariance paths call `gaussian_matrix_cov`:

```text
Command-line script -> production _interface -> shared covariance C
Jupyter notebook    -> Armadillo _wrapper    -> shared covariance C
```

### Production interface

[generic_interface.cpp](cosmolike/generic_interface.cpp) installs
cosmology and survey inputs, assembles data vectors and evaluates the
likelihood. Loading a supplied likelihood covariance does not generate
a new survey covariance.

Covariance command-line scripts use the project's `ci.covariance`
submodule. This path bypasses the Armadillo notebook wrappers. Inputs use
contiguous NumPy storage; returned arrays own their values.

| Covariance production source | Boundary it exposes |
| --- | --- |
| [production_interface_cov.cpp](cosmolike/covariances/production_interface_cov.cpp) | Radial inputs and all-pairs angular spectra. |
| [components_interface_cov.cpp](cosmolike/covariances/components_interface_cov.cpp) | Halo, mask, transform and response components. |
| [matrix_interface_cov.cpp](cosmolike/covariances/matrix_interface_cov.cpp) | Gaussian matrices and connected projections through the shared C assemblers. |
| [cluster_interface_cov.cpp](cosmolike/covariances/cluster_interface_cov.cpp) | Selected-cluster quantities and count components. |

The shared Python survey workflow prepares inputs, calls these interfaces
and combines G, SSC and cNG. Each project's
`covariance/compute_covariance.py` runs that workflow with its
`EXAMPLE_EVALUATE_COVARIANCE.yaml` and saves the components without Jupyter.
Cobaya reads the YAML's familiar `theory`, `params`, `sampler: evaluate`
and `output` blocks; the driver evaluates one specified cosmology.

### Notebook wrappers

Wrappers expose individual calculation steps for inspection: spectra,
halo moments, radial windows, integration operators and matrix components.
Their C++ signatures document the physical axes and units.

| Notebook wrapper | Quantities to inspect |
| --- | --- |
| [cosmo2D_wrapper.cpp](cosmolike/cosmo2D_wrapper.cpp) | Angular spectra and real-space correlations. |
| [halo_wrapper.cpp](cosmolike/halo_wrapper.cpp) | Halo statistics and mass-dependent quantities. |
| [cosmo2D_scuts_wrapper.cpp](cosmolike/cosmo2D_scuts_wrapper.cpp) | Scale-response diagnostics. |
| [components_wrapper_cov.cpp](cosmolike/covariances/components_wrapper_cov.cpp) | Covariance spectra, windows, halo moments, masks and responses. |
| [covariance_wrapper_cov.cpp](cosmolike/covariances/covariance_wrapper_cov.cpp) | Armadillo access to the shared C matrix assemblers. |
| [cluster_wrapper_cov.cpp](cosmolike/covariances/cluster_wrapper_cov.cpp) | Count shells, cluster spectra and selected halo moments. |

The Python bindings convert notebook arrays to Armadillo without changing
the caller's arrays or views. Outputs remain valid after later calls.
This convenience costs copies; production covariance runs avoid them.

### Project bindings and optional compilation

Each project registers its entry points in `interface/interface.cpp` and
selects sources in `interface/MakefileCosmolike`. The project exposes both
covariance paths only when covariance generation is enabled.

The default `IGNORE_COSMOLIKE_<PROJECT>_COVARIANCE=1` omits covariance
kernels and bindings. Unset the project's key and recompile to enable them;
`ci.has_covariance` reports the compiled mode.

Ordinary likelihoods still read and invert supplied covariance matrices
in a build that omits covariance generation. See the
[covariance source guide](cosmolike/covariances/README.md) for the shared
C calculations and their physical assumptions.

# Covariance calculations <a name="covariance"></a>

The calculation separates three contributions:

```math
\mathcal C = \mathcal C^{\mathrm G}
           + \mathcal C^{\mathrm{SSC}}
           + \mathcal C^{\mathrm{cNG}}.
```

The Gaussian term describes fluctuations determined by two-point spectra,
including galaxy shot noise and intrinsic shape noise. SSC, or
**super-sample covariance**, describes how fluctuations larger than the
survey change the structures observed inside it. The connected
non-Gaussian term, cNG, describes the remaining connected four-point
correlations. [Takada & Hu](https://arxiv.org/abs/1302.6994) explain the
background-response description of SSC.

The [covariance physics guide](cosmolike/covariances/README.md) derives
the equations, conventions and approximations for each component.

| C source | Calculation |
| --- | --- |
| [fftlog_cov.c](cosmolike/covariances/fftlog_cov.c), [nonlimber_cov.c](cosmolike/covariances/nonlimber_cov.c) | Shared-plan FFTLog and all-pairs Gaussian gg/gs non-Limber corrections. |
| [ia_cov.c](cosmolike/covariances/ia_cov.c) | Higher-order Gaussian TATT E and B spectra from core IA amplitudes and FAST-PT kernels. |
| [spectra_cov.c](cosmolike/covariances/spectra_cov.c) | Common radial windows, all lens/source Limber cross spectra, and the shifted-log linear-power reader of the cNG angle grid ([why log k](#logk)). |
| [operators_cov.c](cosmolike/covariances/operators_cov.c) | Full-sky, bin-averaged real-space transformations and multipole-band weights. |
| [gaussian_cov.c](cosmolike/covariances/gaussian_cov.c) | Gaussian spectrum pairings, rectangular matrix projection and analytic pair noise. |
| [assembly_cov.c](cosmolike/covariances/assembly_cov.c) | Complete Gaussian matrices and connected projections shared by production and notebook calls. |
| [mask_cov.c](cosmolike/covariances/mask_cov.c) | Angular pair area from the survey footprint. |
| [halo_cov.c](cosmolike/covariances/halo_cov.c) | Halo mass integrals needed for responses and four-point correlations. |
| [perturbation_cov.c](cosmolike/covariances/perturbation_cov.c) | Angular averages of gravitational mode-coupling terms. |
| [non_gaussian_cov.c](cosmolike/covariances/non_gaussian_cov.c) | Halo trispectrum contributions and matter-power responses. |
| [ssc_cov.c](cosmolike/covariances/ssc_cov.c) | Mask-dependent background variance and projected responses. |

A covariance between measured spectra AB and CD requires the crossed
spectra AC, BD, AD and BC. Some of those spectra may be excluded from the
data vector. Their absence from the list of measured observables does not
make their contribution to the covariance zero.

The LSST Y1 notebook combines these tools into real-space and Fourier
G+SSC+cNG forecasts with full-sky angular-bin averages and spherical-cap
pair noise. Gaussian gg/gs can use non-Limber spectra and NLA/TATT;
SSC/cNG retain their zero-IA Limber model. The forecast specifies number
densities and massless neutrinos explicitly. The cap is an example
footprint, not a measured survey mask.

Gaussian physics is selected separately from numerical accuracy:

```python
settings = survey.configuration(
    gaussian={"nonlimber": True, "ia": "NLA", "A1": 0.6},
)
```

The amplitude is an illustrative constant; a list supplies one value per
source bin. TATT also accepts `A2` and `B_TA`. Its B modes enter real-space
shear covariance with the appropriate xi+/xi- signs. The CLI uses the same
mapping under `covariance.gaussian` in its Cobaya-style YAML.

The joint selected-cluster forecast retains its separate Limber, zero-IA
restriction. Its galaxy/shear-only adapter supports the new Gaussian options.

The notebook's configuration cell exposes one covariance accuracy setting:

```python
settings = survey.configuration(accuracy_boost=1)
```

Here `survey` is the LSST Y1 adapter imported by the notebook.

The supported boosts are 1, 2, 4 and 8. They refine multipole cutoffs and
interpolation tables from the baseline in each project's
`covariance/default.yaml`.

A separate `integration_accuracy` selects
fixed quadrature rules: levels 0/1/2/3/4/5/6/7 use
96/128/256/512/1024/2048/4096/8192 nodes per panel. GSL supplies one
Gauss–Legendre rule over each physical panel, generating the nodes and
weights when a stored table is unavailable. The global boost leaves this rule
unchanged. Wide angular bins use several panels to resolve oscillations.
Neither setting reruns CAMB or changes the likelihood's YAML files.

With `boosts = [1, 2]`, the notebook refines fixed physical inputs and
plots changes in correlations, error bars and covariance entries. It also
compares variance ratios across all matrix directions. The highest tested
boost is a numerical comparison reference, not a guarantee of convergence.

The DES cluster notebook also uses
[`forecast_cluster.py`](cosmolike_notebook_utils/covariance/forecast_cluster.py)
to assemble the angular cluster $`6\times2\mathrm{pt}+N`$ matrix, with
absolute counts, all Gaussian and SSC cross blocks, and the same Y
localization used by its mean model.

Its cluster cNG treats clusters as
linearly biased matter tracers; count cross covariance contains SSC only.
Selected-cluster one-halo cNG and non-SSC count–spectrum terms are omitted
and recorded in each output. The [DES cluster running guide](../../../projects/des_cluster/covariance/README.md#joint)
explains those limits, the known Y null rows, and the saved row positions.

> [!NOTE]
> The example computes all three covariance components. The shared
> assembler retains separate Gaussian, SSC and cNG model choices.
> Gaussian non-Limber/IA support does not extend those effects into SSC/cNG.
> Survey cutoff, interpolation and Figure-of-Merit checks remain separate.
> The combined matter halo-response and
> trispectrum helpers currently require massless neutrinos; the cb-aware
> halo statistics alone do not supply a massive-neutrino covariance model.

# Shared Python tools <a name="python"></a>

The project `EXAMPLE_EMUL2_MINIMIZE`, `PROFILE` and `NAUTILUS` scripts share
[`cocoa_hybrid_sampling.py`](cocoa_hybrid_sampling.py). Each loads the matching
hybrid evaluate YAML; `--check` evaluates its fiducial without sampling.
The annealed Emcee minimization and profiles include priors. Nautilus uses
Cobaya's independent prior distributions and includes external prior factors
once. MPI ranks own their models locally and exchange parameter arrays and
scalar scores. See the [LSST Y1 hybrid examples](https://github.com/CosmoLike/cocoa_lsst_y1#cobaya_base_code_examples_emul2)
for CPU setup, output formats, convergence limits and multi-node launch steps.
The [validation record](dev_scripts/hybrid_sampling_validation_20261007.json)
covers all 16 project configurations and short serial/two-rank sampler checks;
it does not establish full-chain convergence or multi-node scaling.


The notebooks keep their survey inputs and initialization in their
project. Reusable calculations live here. The shared package does not
import a particular project's compiled module: the caller supplies an
initialized interface or a data-vector function where needed.

| File or package | Purpose |
| --- | --- |
| [camb_cosmology.py](cosmolike_notebook_utils/camb_cosmology.py) | Run CAMB and prepare linear/nonlinear matter power, cb power, growth and distance tables for the interface. |
| [plot_datavectors.py](cosmolike_notebook_utils/plot_datavectors.py) | Plot tomographic galaxy and shear predictions and baryonic changes. |
| [plot_datavectors_cluster.py](cosmolike_notebook_utils/plot_datavectors_cluster.py) | Plot the cluster observables. |
| [plot_response.py](cosmolike_notebook_utils/plot_response.py) | Show which physical wavenumbers contribute to a prediction. |
| [fisher.py](cosmolike_notebook_utils/fisher.py) | Calculate numerical derivatives, Fisher matrices, priors, marginalized Figures of Merit and forecast contours. |
| [covariance/](cosmolike_notebook_utils/covariance/README.md) | Prepare geometry and noise, assemble Gaussian blocks and halo inputs, construct dense interpolation tables and check covariance modes. |
| [plot_covariances.py](cosmolike_notebook_utils/plot_covariances.py) | Plot split correlation matrices, component maps/histograms and angular standard deviations. |
| [covariance/reference/](cosmolike_notebook_utils/covariance/reference/) | Independent numerical implementations used by the component tests. |

The covariance plots preserve negative correlations and visibly mask
undefined ratios. They do not alter a matrix to make it appear positive.
The split-triangle comparisons follow
[Friedrich et al., Fig. 6](https://arxiv.org/abs/2012.08568); the component
maps and histograms adapt
[Barreira, Krause & Schmidt, Fig. 1](https://arxiv.org/abs/1807.04266).

# Running the tests <a name="tests"></a>

Data-vector tests check predictions against stored references and test
numerical refinement, parameter changes and repeatability. Covariance
tests check the separate numerical components and notebook tools.

## Data-vector tests

We assume users have run `conda activate cocoa`, use Bash in `cocoa/Cocoa`,
and have compiled the LSST Y1 interface.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: run the data-vector tests.

    python -m pytest projects/lsst_y1/tests/data_vector

Their explanations and reference-update procedure are in
`projects/lsst_y1/tests/data_vector/README.md`. The stored inputs and
reference vectors remain under `projects/lsst_y1/tests/frozen/`.

## Covariance tests

We assume users have run `conda activate cocoa`, use Bash in `cocoa/Cocoa`,
and have compiled LSST Y1 with the covariance sources.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: enable and compile covariance generation.

    unset IGNORE_COSMOLIKE_LSST_Y1_CODE
    unset IGNORE_COSMOLIKE_LSST_Y1_COVARIANCE
    source ./projects/lsst_y1/scripts/compile_lsst_y1.sh

**Step :three:**: run the covariance tests.

    python -m pytest projects/lsst_y1/tests/covariance

The tests use the normal project extension and tracked independent Python
references. Their file guide is
`projects/lsst_y1/tests/covariance/README.md`. No developer-local library
or external study directory is required.

### The project forecast check <a name="cov_forecast_check"></a>

Every other project's covariance sector runs one shared check,
[cocoa_covariance_testing.py](cocoa_covariance_testing.py)'s
`check_project_forecast`, against its own galaxy/shear forecast adapter
and compiled bindings, using the project's redshift files and catalog
inputs with small numerical grids and a few measured rows:

| Check | Purpose |
| --- | --- |
| Project layout | Verify the full angular and Fourier vector lengths from the measured row map. |
| Accuracy refinement | Keep measurement bins fixed while increasing quadrature resolution. |
| Real-space components | Check finite, symmetric G, SSC, cNG and total matrices for a measured subset. |
| Fourier components | Check the same properties for bandpowers. |
| Thread repeatability | Compare every component bitwise with one and eight OpenMP threads. |
| Positive total | Check variance positivity for every direction of the tested subset. |
| Output archive | Read arrays and resolved survey metadata without pickle. |

The independent algebra and projection references live once, in LSST
Y1's covariance component suite; the per-project check covers that
project's binding and inputs and does not duplicate those references.
Use each project's covariance notebook for full matrices and numerical
refinement: subset positivity alone does not establish full-matrix
positivity or parameter-error convergence, and no stored covariance or
reference value is replaced by the check. With the default
data-vector-only build the sector reports skips; a skip in a
deliberately disabled build is expected and is not a successful
covariance check.

A data-vector-only build reports skips for the optional covariance tests.
Each project separates `tests/data_vector` from `tests/covariance`.
[cocoa_testing.py](cocoa_testing.py) supplies the data-vector test machinery;
project `tests/cocoa_test_utils.py` files supply their configurations.
[cocoa_covariance_testing.py](cocoa_covariance_testing.py) checks each
project's galaxy/shear forecast adapter with small measured subsets.
The independent component references remain in LSST Y1's covariance tests.

## The shared data-vector checks <a name="dv_checks"></a>

Every project runs the same families of data-vector checks from the
shared machinery; a project README documents only its configurations,
test numbers, measured values and deviations, and links here for the
mechanism. Surveys without a family's subject (no baryon block, no
TATT examples) simply do not instantiate it.

### The test-file naming convention <a name="dv_naming"></a>

A test file's name means the same thing in every project, so a reader
who knows one suite knows them all:

| file | meaning everywhere |
|---|---|
| `test_exampleN.py` | the project's `EXAMPLE_EVALUATEN` configuration: the frozen-reference drift check and the race check, in NLA and in TATT (four tests). In the galaxy surveys `EXAMPLE1` is cosmic shear and `EXAMPLE2` is 3x2pt. |
| `test_exampleN_2x2pt.py` | the `EXAMPLEN` 3x2pt configuration reduced to galaxy clustering plus galaxy-galaxy lensing: the same four checks |
| `test_fastpt.py` | the frozen FAST-PT references and the [CFASTPT vs FAST-PT sweep](#dv_fastpt) |
| `test_nonlinear.py` | the [Halofit vs EuclidEmulator2 checks](#dv_halofit_ee2) |
| `test_ee2.py` | the [EE2 race check, plus the modification pin where the project carries it](#dv_ee2) |
| `test_accuracy.py` | the advisory [accuracy scans](#dv_accuracy) and, where shipped, the opt-in [N-random-models check](#dv_nmodels) |
| `test_accuracy_baryons.py` | the [baryonic-feedback accuracy checks](#dv_baryons) BF0-BF7 |
| `test_baryons.py` | the [baryonic-feedback drift tests](#dv_baryons) BD1-BD7 |
| `test_photoz_conventions.py` | the [photo-z convention checks](#dv_photoz) |
| `test_nonlimber_ggl.py`, `test_nonlimber_gg.py` | the [non-Limber checks](#dv_nonlimber) |
| `test_cache_consistency.py` | the [sector-ladder cache check](#dv_cache_ladder) |
| `test_nz_cache_roundtrip.py` | the shared [n(z) selection cache round trips](cocoa_nz_cache_testing.py) |
| `test_scale_cut_diagnostics.py` | the notebook scale-cut diagnostics, where the project ships them |
| `test_emul2.py` | the hybrid-emulator advisories, where the project ships trained emulators |

`test_exampleN.py` always tests the project's `EXAMPLE_EVALUATEN`
configuration, and what `EXAMPLE_EVALUATEN` is follows one of two
conventions:

- **Galaxy surveys**: `EXAMPLE1` is cosmic shear, `EXAMPLE2` the
  flagship 3x2pt (or 6x2pt) combination, and `test_example2_2x2pt.py`
  its 2x2pt reduction. Further examples extend the list (DES Y3 adds
  its DES-Y1 configurations as `EXAMPLE3`/`EXAMPLE4`).
- **Cluster surveys**: the examples lead with the cluster flagships -
  `EXAMPLE1` is the 4x2pt + N combination, `EXAMPLE2` the 6x2pt + N
  one, `EXAMPLE3` cosmic shear and `EXAMPLE4` the galaxy 3x2pt, with
  `test_example4_2x2pt.py` its 2x2pt reduction. A future cluster
  project follows this layout and inherits this explanation.

One documented deviation remains: DES Y6 consolidates its example
checks in one `test_examples.py`. A project README names which
convention it follows next to its roster.

### Frozen references and the race check <a name="dv_frozen"></a>

The standard configurations get four tests each: a chi2 drift check
and a race-condition check, in both intrinsic-alignment models.

| check | pass limit | a failure means |
|-------|------------|-----------------|
| drift | the recomputed chi2 must stay within 0.2 of the value stored in `frozen/reference_chi2.json` | code or data changed the numbers |
| race condition | the fiducial evaluated on its own versus evaluated again after nine other cosmologies; the two must agree within 1e-4 | leftover state or an OpenMP race |

Everything the tests compare against lives under the project's
`frozen/`: one snapshot of configurations, data and reference values,
captured together when the references were generated and unchanged
since. The tests read nothing from the live project folders - the
frozen configuration modules write out every option and every
parameter, including the ones that normally come from the default
yaml files, so editing those files cannot change what the tests
evaluate. A SHA-256 manifest (`tests/manifest_sha256.json`) pins
every byte of the snapshot; each test verifies it first and refuses
to run when a file under `frozen/` was edited, added or removed, naming the file.
Maintainers refresh a snapshot only for a deliberate change, with
`tests/generate_frozen_reference.py --overwrite` (it deletes and
rebuilds `frozen/` from the current project and re-pins the
manifest), reviewing the printed chi2 values before committing them;
the `_meta` block of `frozen/reference_chi2.json` records when the
current references were generated. Each project README lists what its
snapshot holds.

The TATT drift tests compare against synthetic data vectors generated
at freeze time rather than the shipped survey vector: against the
shipped vector the TATT chi2 sits away from its minimum, where it
responds linearly to tiny numerical changes; at its own minimum the
response is quadratic and the drift bounds stay meaningful. Each
frozen TATT configuration therefore evaluates against its own
generated vector, and the reference chi2 is zero by construction.

### The zero-baseline mechanism <a name="dv_zero_baseline"></a>

Several families need a comparison that no stored vector can supply.
They all use the same construction: the default-settings model writes
its own theory vector during evaluation, that vector becomes the data
of a temporary dataset (so the default chi2 against it is zero by
construction, and nothing new is stored in `frozen/`), and the
alternative evaluation - pushed accuracy, another implementation,
another random point - scores against it. The single reported number,
delta chi2 = chi2(alternative) - 0, is then a pure response to what
changed, with the full masked inverse covariance as the metric.

### The CFASTPT vs FAST-PT sweep <a name="dv_fastpt"></a>

Cosmolike computes the TATT perturbation-theory integrals with two
implementations: cfastpt, the C code built into the interface
(`IA_code: 0`, the reference), and the python FAST-PT package through
the fastpt theory block (`IA_code: 1`). The sweep evaluates both at 30
fixed points across the intrinsic-alignment prior (20 drawn across the
prior plus a one-parameter-at-a-time family), cosmology at the
fiducial, on each likelihood the project ships. At every point the
cfastpt data vector is the fiducial, so the reported delta chi2 is
zero for identical vectors and quadratic in their difference; the pass
limit is 0.2. A second FAST-PT evaluation at doubled boosts repeats
the measurement as an advisory, so the residual grid response of the
defaults shows next to the pass quantity.

The fastpt block computes on two grids: `accuracyboost` multiplies the
density of the output table cosmolike reads with linear interpolation,
and `internal_accuracyboost` the density of the internal grid the
FFTLog convolutions run on, with a cubic spline in log k upsampling
the terms from one grid onto the other. Both boosts are rebased so 1.0
is the converged configuration; each project records its own
convergence evidence.

Two shared pytest options steer the sweep (registered by each
project's conftest): `--high=1` applies the pushed camb/cosmolike
settings of the accuracy checks to every sweep block, and `--mask`
reruns the sweep under another scale-cut mask - each choice is a
frozen TATT dataset variant differing only in its `mask_file` line,
with `--mask=ones` keeping every data point (the strictest
comparison) and the 0.2 pass rule unchanged. Under a non-frozen mask
the sweep regenerates its baseline: the cfastpt fiducial vector under
that mask becomes the data vector, keeping the zero baseline.

### Halofit vs EuclidEmulator2 <a name="dv_halofit_ee2"></a>

The likelihood can source the nonlinear matter power from CAMB's
Takahashi halofit (`non_linear_emul: 2`) or from EuclidEmulator2
(`non_linear_emul: 1`). The checks evaluate the project's data vectors
with both sources at ten fixed cosmologies across the omegam/ns/As
space (every other parameter at the frozen fiducial), reporting per
cosmology the delta chi2 of the Halofit vector against the EE2 vector;
the EE2 vector is that cosmology's fiducial, so the baseline is zero
and no stored data vector enters the metric. The checks are advisory -
no pass limit: the numbers say how much of the statistical error
budget the Halofit-versus-emulator difference consumes under the
chosen scale cuts. The `--mask` option applies.

### The EE2 modification tests <a name="dv_ee2"></a>

Cocoa pins a modified EuclidEmulator2 (OpenMP threading, a
1,010-redshift capacity, the `get_boost2` API with a pre-built
emulator, memory-leak fixes, and a bilinear interpolation with a
border fix; see `external_modules/code/euclidemu2/README.md`). The
physics test compiles the pre-modification build (commit `ff59f66`)
at test time into a
temporary prefix from the local clone (offline; `.local` untouched)
and evaluates both builds at the ten shared cosmologies: the
per-cosmology delta chi2 of the original against the modified build
must stay below 0.2. The original lacks `get_boost2` and silently
overflows beyond 101 redshifts, so its worker carries a compatibility
adapter that leaves its numerics untouched. A second test is the race
check with EE2 on, since EE2's compute is OpenMP-threaded.

### Accuracy checks <a name="dv_accuracy"></a>

An advisory family, no pass/fail: each check reports the delta chi2
between a high-accuracy and a default evaluation - the numerical error
of the default settings. A first scan changes one accuracy setting at
a time, so a large delta can be attributed to the setting causing it;
the remaining checks push every setting at once on each configuration
the project ships. The settings and what they control:

| setting | what it controls |
|---------|------------------|
| `accuracyboost` (cosmolike) | sizes of cosmolike's internal lookup tables, including the dyadic z grid of the power-spectrum tables: every coarser grid's nodes are a subset of every finer grid's, so a higher boost tightens the same interpolation instead of moving the nodes |
| `integration_accuracy` (cosmolike) | the quadrature level: each integral family selects a fixed Gauss-Legendre rule from its ladder. Since the native-GSL extension the ladders reach 8,192 nodes at level 7, and any higher level (projects push 10) selects the 8,192-node rule |
| `internal_accuracyboost` (cosmolike) | density of the C-FAST-PT convolution grid relative to the output table the likelihood interpolates; the default 1 keeps the two grids equal (the exact single-grid path) |
| `lmax` (cosmolike) | highest multipole of the internal harmonic-space tables transformed into the real-space correlation functions; arcminute scales need very high multipoles |
| `kmax_boltzmann` (cosmolike) | the k cutoff of the power spectrum the likelihood requests from CAMB |
| `AccuracyBoost` (CAMB) | CAMB's overall accuracy multiplier: denser sampling in every internal CAMB grid, the most expensive setting |
| `k_per_logint` (CAMB) | k samples CAMB computes per logarithmic interval of the transfer functions |
| `kmax` (CAMB) | highest k of CAMB's matter power spectrum; one physical cutoff with `kmax_boltzmann`, seen from the CAMB side |

When several settings move the chi2, settle them in cost order: raise
cosmolike `accuracyboost` first (cheap), then CAMB `k_per_logint`, and
CAMB `AccuracyBoost` last (expensive at run time, and able to
masquerade for the cheap settings). `kmax_boltzmann` and CAMB `kmax`
are one physical cutoff seen from two sides; move them together.
High-accuracy evaluations take minutes.

Advisory results recorded before the native quadrature extension
(2026-10-15) pushed `integration_accuracy` into ladders whose largest
rule was 1,024 nodes; the same pushed level now selects 8,192-node
rules, so a re-measured delta can differ from a dated record even with
nothing else changed. Each project's dated measurements state what was
measured then; re-measure before quoting them as current.

### The N-random-models check (opt-in) <a name="dv_nmodels"></a>

This check repeats the accuracy measurement at N reproducible random
points drawn across the prior: per point, draw, generate a synthetic
data vector at that point with the default settings (the zero-baseline
mechanism), and evaluate the high-accuracy chi2 against it. The draws
are seeded, so every run draws exactly the same points and the numbers
compare across reruns and machines. The report streams one block per
model and ends with the min/median/max delta chi2; advisory, the
deltas only have to be finite. Each model costs a default and a
high-accuracy build-plus-evaluation, minutes per model, so the check
is off by default: with `COCOA_ACCURACY_NMODELS` unset (or 0) it
prints how to enable itself and passes.

### Baryonic feedback: accuracy and drift <a name="dv_baryons"></a>

Two complementary families cover every feedback method the `bfmt`
theory block implements (SP(k) through three baryon-fraction
relations, BCEmu, Flamingo, BACCOemu, and BCemu2025), at a fixed,
measurable parameter point per method (the points live in the shared
`BARYON_METHODS` table; for example, the BACCOemu point keeps the
baryon density inside that emulator's training box, and the
double-power-law point keeps the baryon fraction inside SP(k)'s
calibrated band over the full redshift grid).

The accuracy checks regenerate their vector on the fly per run (the
zero-baseline mechanism) and report the default-versus-pushed delta
chi2 with feedback on - a pure numerics response that can never see
drift; a first check also runs the one-setting-at-a-time scan with
one feedback method on, so a large delta names the setting causing
it. The drift tests hold a vector frozen at freeze time
(`generate_frozen_reference.py --baryons`) still, so they measure
drift and nothing else: a chi2 above the tolerance means cosmolike or
the `bfmt` block changed its prediction since the freeze.

### Photo-z convention checks <a name="dv_photoz"></a>

The likelihood exposes runtime knobs for how the n(z) table files
become the smooth distributions the Limber integrals consume, declared
in the likelihood yamls and defaulting to the historical behavior:
`photoz_interpolation_type` (0 = cubic spline, 1 = linear, 2+ =
Steffen monotone, which cannot overshoot below zero around a sharp
feature) and `photoz_zmid_convention` (0 = the z column holds Z_LOW
left bin edges, so the tabulated value belongs at the cell center
z + dz/2; 1 = the column holds Z_MID sample points; the two readings
differ by a rigid dz/2 shift of every distribution). The test
evaluates the frozen fiducial under the default, each alternative,
and the default again in one process, measuring every alternative
against the default with delta chi2 = delta^T C^-1 delta under the
masked inverse covariance. The assertions are a dead-flag floor on
each alternative (a stale n(z) cache would give exactly zero), the
frozen-reference check on the default, and a bit-identical round trip
back to the default. The n(z) reconstruction switch has its own
[shared study](cosmolike_notebook_utils/README.md#nz_study) and a
routine cache round-trip test
([cocoa_nz_cache_testing.py](cocoa_nz_cache_testing.py)).

### Non-Limber checks <a name="dv_nonlimber"></a>

The likelihood yaml keys `adopt_limber_gs` and `adopt_limber_gg`
choose how the galaxy-galaxy lensing and galaxy clustering spectra are
computed: `0` takes the multipoles below 150 from the exact
projection, computed with the split of Fang, Krause, Eifler &
MacCrann (arXiv:1911.11947) - an FFTLog integral of the linear power
spectrum plus, in Limber, what linear theory misses - and `1` uses
the Limber approximation at every multipole. Each test evaluates the
frozen 3x2pt fiducial with the default, the other setting, and the
default again in one process and reports
delta chi2 = delta^T C^-1 delta of the non-Limber minus the Limber
vector, with per-bin (clustering) or per-pair (lensing)
contributions. The assertions are a dead-flag floor, that only the
expected sector's entries change, a bit-identical round trip back to
the default, agreement with the project's recorded measurement to 5%,
and the frozen-reference check on the default evaluation. The lens
distributions are narrow, so Limber fails at low multipoles for the
clustering auto spectra; each project records its measured deltas.

### The sector-ladder cache check <a name="dv_cache_ladder"></a>

cosmolike caches every expensive stage behind its own key, and a
partial-invalidation bug - one sector's update path leaving a stale
static another sector consumes - produces wrong data vectors only in
MIXED update sequences, which the per-point checks never exercise.
The test walks a deterministic ladder in one process (cosmology-only
steps, then IA-only, source-photo-z, lens-photo-z and
shear-calibration steps), evaluating after every step, then scrambles
every sector at once and returns to the ladder's final point: the
pipeline must reproduce the recorded data vector and chi2 bit for
bit, and a second instance walking the mirrored sector order must
land on the same vector. The shear-calibration steps must equal the
analytic (1+m_i)(1+m_j) block rescale to 1e-12, a no-op update must
change nothing, and both intrinsic-alignment models run (the TATT
ladder exercises the FAST-PT rebuild machinery).

# Appendix <a name="appendix"></a>

## FAQ: How is a data vector calculated? <a name="prediction"></a>

The input cosmology supplies three-dimensional power spectra and the
relation between redshift and distance. A power spectrum describes the
variance of density fluctuations at each spatial wavenumber $`k`$;
larger wavenumbers describe smaller structures.

Redshift distributions specify where the selected galaxies lie. Lensing
weights also account for all foreground matter that can distort a source.
Combining these weights with the power spectrum gives angular spectra,
where multipole $`\ell`$ labels angular scale.

For example, the Limber approximation has the form

```math
C_\ell^{AB} = \int d\chi\,
\frac{W_A(\chi)W_B(\chi)}{f_K(\chi)^2}
P\!\left(\frac{\ell+1/2}{f_K(\chi)},a(\chi)\right).
```

Here $`\chi`$ is radial comoving distance, $`f_K`$ is transverse comoving
distance, $`W_A,W_B`$ are projection weights, and $`a=1/(1+z)`$ is scale
factor. The weights and additional terms depend on the chosen observable,
galaxy-bias and intrinsic-alignment model.

The Limber approximation simplifies radial mode coupling. On large angular
scales, the non-Limber galaxy and galaxy–shear calculations retain radial
oscillations through spherical-Bessel transforms. Their FFTLog algorithm
uses fast Fourier transforms on a logarithmic radial grid. A non-Limber
data-vector implementation does not by itself provide all the crossed
spectra required by a covariance.

Angular spectra can be returned directly or transformed into angular-bin
averages of the real-space correlations. The project then assembles their
order, applies calibration and nuisance parameters, and selects the
measurements retained by its likelihood mask.

Intrinsic alignments describe coherent galaxy shapes generated by their
environment. NLA uses the nonlinear alignment prescription; TATT adds
tidal alignment and tidal torquing terms. The C FAST-PT implementation
computes the required mode-coupling integrals, following the algorithm of
[McEwen et al.](https://arxiv.org/abs/1603.04826). Model support and the
choice between C FAST-PT and Python FAST-PT belong to the project.

## FAQ: How do halos use massive neutrinos? <a name="neutrinos"></a>

The core distinguishes total matter, denoted $`m`$, from cold dark matter
plus baryons, denoted $`cb`$. Neutrino free streaming makes their power
spectra and growth different. Halo abundance and bias use the cold-field
prescription discussed by
[Castorina et al.](https://arxiv.org/abs/1311.1212).

For either field $`X`$, the linear mass variance is

```math
\sigma_X^2(M,a) = \int d\ln k\,
\frac{k^3 P_X^{\rm lin}(k,a)}{2\pi^2}
W^2(kR_X),
\qquad
R_X = \left(\frac{3M}{4\pi\bar\rho_{X,0}}\right)^{1/3}.
```

$`W(x)=3(\sin x-x\cos x)/x^3`$ is the Fourier transform of a spherical
top-hat: it averages the density inside a sphere. Thus $`\sigma_X^2`$
measures the strength of fluctuations after smoothing on the scale
containing mass $`M`$. The field choice affects both the power spectrum
and the present-day mean density $`\bar\rho_{X,0}`$ used to assign that
comoving smoothing radius.

[cosmo3D.c](cosmolike/cosmo3D.c) evaluates both fields with FFTLog from
the supplied evolving linear spectra and tabulates their mass slopes.
FFTLog expands the spectrum on a logarithmic wavenumber grid, allowing
the smoothings at many radii to be computed together.

The calculation retains the redshift dependence of the spectrum rather
than reconstructing every redshift from one present-day variance and a
single growth factor.

Halo peak heights, abundance, bias and halo-occupation mass integrals use
$`\sigma_{cb}`$ and $`\bar\rho_{cb,0}=\rho_{\rm crit,0}(\Omega_m-\Omega_\nu)`$.
The total-matter variance remains available for diagnostics. The former
`halo_matter_field` YAML option and initializer are retired; halo
statistics always use cb.

In the notebook variance diagnostic, `field=0` selects total matter and
`field=1` selects cb. This diagnostic choice does not change the field
used internally by halo abundance or bias.

The existing $`M_{200m}`$ profile-radius definition still refers to the
total mean matter density. Total-matter lensing weights and supplied
nonlinear matter spectra also retain their meaning. Using cb for halo
statistics does not turn every matter spectrum in the likelihood into
$`P_{cb}`$.

Concentration is the ratio of a halo's outer radius to its profile scale
radius; it describes how centrally concentrated the halo is. The
prescription in [halo.c](cosmolike/halo.c) uses

```math
\nu = \frac{1.686}{\sigma_{cb}(M,a)},
\qquad
D_{cb}(M,a) = \frac{\sigma_{cb}(M,a)}{\sigma_{cb}(M,1)},
\qquad
c(M,a) = 9\,\nu^{-0.29}D_{cb}(M,a)^{1.15}.
```

The peak height $`\nu`$ compares the collapse threshold with the typical
fluctuation on that mass scale. The same cold variance sets the peak
height and its growth.

This is the code's adopted extension of the
[Bhattacharya et al. concentration fit](https://arxiv.org/abs/1112.5479),
whose calibration did not include massive neutrinos. It is not a new
simulation calibration of concentration in neutrino cosmologies.

The cosmology interface therefore accepts `omegan2`, meaning
$`\Omega_\nu h^2`$, and `lnP_linear_cb` on the same grid and in the same
layout as `lnP_linear`. Halo calculations require a supplied cb table;
an empty argument is not permission to substitute total-matter power.

The shared CAMB helper supplies the table using CAMB's `delta_nonu` field.
The EMUL2 likelihood path retains an approximate conversion from its
matter spectrum; it should not be confused with a CAMB cb spectrum.

## FAQ: What does the cluster code calculate? <a name="clusters"></a>

The `des_cluster` project combines cluster counts with cluster lensing,
cluster clustering, cluster–galaxy clustering and the galaxy/shear
observables. Richness is an observed proxy for halo mass; the model
integrates the mass–richness distribution over the selected bins.
[To et al.](https://arxiv.org/abs/2503.13631) describe the DES multiprobe
cluster modeling framework. Implementing these observables does not
transfer that paper's validation to an arbitrary survey configuration.

| Cluster source | Responsibility |
| --- | --- |
| [structs_cluster.c](cosmolike/structs_cluster.c), [header](cosmolike/structs_cluster.h) | Cluster state and defaults. |
| [redshift_spline_cluster.c](cosmolike/redshift_spline_cluster.c) | Cluster selection, redshift distributions and pair maps. |
| [radial_weights_cluster.c](cosmolike/radial_weights_cluster.c) | Cluster-density and magnification weights. |
| [halo_cluster.c](cosmolike/halo_cluster.c) | Mass–observable relation, abundance, bias and richness-bin halo terms. |
| [cosmo2D_cluster.c](cosmolike/cosmo2D_cluster.c) | Counts, angular spectra and real-space correlations. |
| [generic_interface_cluster.cpp](cosmolike/generic_interface_cluster.cpp) | Cluster configuration, joint data-vector assembly and supplied covariance handling. |
| [cosmo2D_wrapper_cluster.cpp](cosmolike/cosmo2D_wrapper_cluster.cpp), [halo_wrapper_cluster.cpp](cosmolike/halo_wrapper_cluster.cpp) | Cluster arrays for Python notebooks. |

Cluster-specific state and extensions live in the `_cluster` files.
They use shared cosmology and halo readers while keeping cluster sample
selection separate from the ordinary galaxy/source code. The unfinished
matter and thermal Sunyaev–Zel'dovich work in `future_port_unfinished/`
is not part of the compiled cluster or 3x2pt prediction.

## FAQ: Which units do the interfaces use? <a name="units"></a>

The cosmology setter accepts astronomical units and converts them to the
core's units. The covariance component calls expose the core units
directly. Mixing these two conventions changes the physical scale being
calculated even when the arrays have the correct shape.

Here $`h=H_0/(100\,\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1})`$
is the dimensionless Hubble parameter, and $`c`$ is the speed of light.

| Quantity | Cosmology input tables | Core and covariance components |
| --- | --- | --- |
| Wavenumber | `log10k_2D` is $`\log_{10}k`$ with $`k`$ in $`h/\mathrm{Mpc}`$. | $`k`$ in $`(c/H_0)^{-1}`$. |
| Power spectrum | `lnP_linear`, `lnP_nonlinear`, `lnP_linear_cb` are natural logarithms of power in $`(\mathrm{Mpc}/h)^3`$. | Dimensional power in $`(c/H_0)^3`$, unless a call explicitly requests logarithms. |
| Comoving distance | `chi` in $`\mathrm{Mpc}/h`$. | Distance in $`c/H_0`$. |
| Halo mass | $`M_\odot/h`$ for halo queries. | $`M_\odot/h`$. |
| Angle and area | Project inputs may use arcminutes and square degrees. | Covariance angles in radians; areas in steradians. |
| Number density | Project inputs may use galaxies per square arcminute. | Angular noise uses galaxies per steradian. |

Since $`c/H_0=2997.92458\,\mathrm{Mpc}/h`$, multiply a wavenumber in
$`h/\mathrm{Mpc}`$ by 2997.92458 before a core covariance power call.
The Python covariance helpers document each conversion and array layout.

Power tables use the interface's flattened layout: a table indexed
`[redshift, wavenumber]` is flattened in Fortran order. Growth uses its
own redshift grid, `z_G`; it need not have the power table's redshift
sampling. Use [camb_cosmology.py](cosmolike_notebook_utils/camb_cosmology.py)
and the project notebook to keep those inputs paired correctly.

Covariance component bindings accept contiguous `float64` arrays;
field and band indices use `int32`. Returned arrays own their data.
[The Python covariance guide](cosmolike_notebook_utils/covariance/README.md)
defines field ordering, shapes, mask normalization and noise conventions.

## FAQ: How are accuracy and parallelism controlled? <a name="numerics"></a>

The project YAML files and notebooks expose data-vector accuracy settings.
Covariance has its own `accuracy_boost` for tables/cutoffs and
`integration_accuracy` for the native GSL quadrature rules. Its default
`power_accuracyboost=8` prepares 11,993 wavenumber samples from 1,500 CAMB
samples for all covariance components. A natural cubic spline fills those
tables once; the C readers continue to use linear interpolation.

The global boost multiplies this refinement and preserves existing nodes.
Reinitialize the forecast after changing it, as the project notebooks do.
The cosmology and CAMB settings stay fixed in these refinement comparisons.
Numerical convergence and the validity of a physical approximation are
separate questions.

The code reuses expensive work across many outputs. FFTW plans are
created serially and reused; independent transforms use separate worker
buffers. Expensive functions can be sampled on a coarse grid, expanded
with a cubic spline during table construction, and read by linear
interpolation on the dense table. The dense lookup cannot recover a
physical feature missing from the coarse samples.

Power and growth readers use the supplied piecewise-uniform redshift
grids with arithmetic indexing; the setters validate their grid metadata.
SIMDe supplies portable vector operations, allowing one instruction to
perform the same arithmetic on several independent values. These paths
are enabled in the supported builds, including debug builds.

CosmoLike's explicit OpenMP loops distribute independent work within a
process. OpenBLAS stays at one thread, including normal and cluster
covariance inversions. This prevents a matrix operation from launching
another thread team inside the numerical calculation.

The covariance C code does not call MPI. Independent process scheduling
belongs to Python/Cobaya. A future covariance driver can assign matrix
blocks to separate processes, each using OpenMP, but an automatic MPI
covariance generator is not currently supplied.

> [!NOTE]
> The supported optimized build keeps strict floating-point arithmetic.
> `COSMOLIKE_AGGRESSIVE_MODE` is retired. Do not add `-ffast-math`,
> `-Ofast` or unsafe reassociation flags to the project Makefiles.
> `COSMOLIKE_DEBUG_MODE` remains available for diagnostics.

The compiled interface holds mutable cosmology and survey state. Complete
initialization before parallel evaluation; do not change that state from
concurrent Python threads. Use separate processes for independent
cosmologies and size the OpenMP team for the cores assigned to each process.
Timing comparisons require a quiet machine and one calculation at a time.

## FAQ: Why does the cNG read the power spectrum in log k? <a name="logk"></a>

The cNG angle integrals evaluate the linear power at the internal
momentum $`\lvert\mathbf{k}+\mathbf{q}\rvert`$ for every wavenumber
pair and every angular node: at the baseline grids that is 15,852,480
table reads per radial shell, the largest single cost of a covariance
run.

The standard reader (`p_lin_at_a`, [cosmo3D.c](cosmolike/cosmo3D.c))
takes one `log10(k)` per sample to locate the table cell. That
logarithm is the first link of each sample's dependency chain: its
result is the table index, so the four table loads and the bilinear
interpolation all wait for it, and out-of-order execution cannot hide
the first link of a chain. The `exp` that returns from the stored
$`\ln P`$ to $`P`$ sits at the chain's end, where its latency hides
under the next sample's loads; removing it changes the cost by about
one percent, so it stays.

Under the Limber projection the internal-momentum grid is the same at
every radial shell up to one factor: $`k = \mathrm{magnitude}/f_K`$,
with the magnitude fixed in multipole units. The logarithm is
therefore shift-invariant,

```math
\log_{10}(k) = \log_{10}(\mathrm{magnitude}) - \log_{10}(f_K),
```

so [survey.py](cosmolike_notebook_utils/covariance/survey.py) takes
the table of logarithms once per run and each shell supplies one
scalar shift to the reader `linear_power_logk_rows_cov`
([spectra_cov.c](cosmolike/covariances/spectra_cov.c)). This also
removes the per-shell division of the whole grid by $`f_K`$, which
existed only to feed the reader.

Measured on 2026-10-08 (Apple M2 Pro, 8 OpenMP threads): the
per-sample `log10` was 46% of the read. The before/after construction-time
ratios were **1.65** for LSST Y1 real space and **1.44** for the DES cluster
joint covariance. These ratios describe the measured configurations.

The shifted sum is not bitwise the logarithm of the quotient:
$`\log_{10}(m) - \log_{10}(f_K)`$ and $`\log_{10}(m/f_K)`$ differ in
the last bits, so cNG entries move by at most 5.0e-18 in absolute
value. The Gaussian and SSC blocks do not pass through this reader
and are bitwise unchanged.

> [!Warning]
> Do not route the internal-momentum read back through
> `covariance_power` with `k = magnitude/distance`: that restores the
> per-sample `log10` and the per-shell division of the full grid,
> 30% to 40% of the construction time.

## FAQ: What establishes a usable covariance? <a name="positivity"></a>

For any coefficients $`v`$, the combination $`v^T\hat d`$ of measured data
has variance $`v^T\mathcal C v`$. A valid covariance cannot give that
combination a negative variance. Positive diagonal entries alone do not
establish this: correlations between different blocks also matter.

SSC is assembled from common response vectors with positive integration
weights. Keeping their complete outer products preserves positive
semidefiniteness. Deleting selected cross-lens entries while retaining
correlations with shear can destroy it. Cross-bin terms must therefore
be computed consistently, even when the corresponding spectra are not
part of the data vector.

Check the full Gaussian + SSC + cNG matrix and the submatrix selected for
the likelihood. An invertible total covariance must be positive definite;
eigenvalue and Cholesky checks assess this. An individual connected cNG
contribution need not itself be positive definite. Clipping eigenvalues
or adding a diagonal correction does not validate the physics.

Positivity is only one requirement. Refine the numerical settings and
compare marginalized parameter errors and Fisher Figures of Merit.

A Figure of Merit measures the inverse size of a chosen parameter
confidence region; a larger value means tighter constraints.
A Fisher calculation uses derivatives of the mean data vector at a chosen
cosmology and can make this comparison without running a chain. Generalized
covariance eigenvalues provide a complementary check of variance changes
in every data-space direction.

[Friedrich et al.](https://arxiv.org/abs/2012.08568) examine covariance
approximations through parameter estimation and goodness of fit. The
data-vector $`\lvert\Delta\chi^2\rvert<0.2`$ regression criterion is not
a covariance convergence criterion. The new Roman generator still needs
positivity checks for each generated configuration and parameter-constraint
convergence tests; the existing supplied Roman covariance has not been
replaced by this work.
