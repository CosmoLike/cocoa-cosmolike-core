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
   6. [FAQ: What establishes a usable covariance?](#positivity)

# Overview <a name="overview"></a>

This repository contains the C calculations, C++ interfaces and shared
Python tools used by [Cocoa](https://github.com/CosmoLike/cocoa)'s
CosmoLike projects. A project supplies its survey data, redshift bins,
likelihood configuration and compiled Python interface. This repository
supplies the calculations that those projects share.

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
> and shear fields. The matrices use massless neutrinos, Limber spectra
> and a spherical-cap footprint, retaining every internal cross-bin
> spectrum. Numerical and Fisher convergence remain to be established
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

**Step :two:**: compile the LSST Y1 interface.

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
| [cosmolike/](cosmolike/) | Data-vector physics, halo statistics, cluster extensions and C++ interfaces. |
| [cosmolike/covariances/](cosmolike/covariances/README.md) | Covariance components with their own numerical grids and integrations. |
| [cfastpt/](cfastpt/) | C implementation of FAST-PT mode-coupling integrals. |
| [cosmolike_notebook_utils/](cosmolike_notebook_utils/) | Shared cosmology preparation, plotting, Fisher and covariance tools. |
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

## C++ interfaces

[generic_interface.cpp](cosmolike/generic_interface.cpp) installs inputs,
assembles data vectors, applies the likelihood mask, and manages the
supplied covariance and likelihood evaluation. It does not generate a new
survey covariance when a likelihood loads a covariance file.

| Wrapper | Arrays exposed to project notebooks |
| --- | --- |
| [cosmo2D_wrapper.cpp](cosmolike/cosmo2D_wrapper.cpp) | Angular spectra and real-space correlations. |
| [halo_wrapper.cpp](cosmolike/halo_wrapper.cpp) | Halo statistics and mass-dependent quantities. |
| [cosmo2D_scuts_wrapper.cpp](cosmolike/cosmo2D_scuts_wrapper.cpp) | Scale-response diagnostics. |
| [components_wrapper_cov.cpp](cosmolike/covariances/components_wrapper_cov.cpp) | Covariance spectra, radial inputs, halo moments, mask, transform and SSC components as Armadillo arrays. |
| [covariance_wrapper_cov.cpp](cosmolike/covariances/covariance_wrapper_cov.cpp) | Whole real/Fourier Gaussian matrices and connected projections from supplied matter tables and catalog windows. |
| [cluster_wrapper_cov.cpp](cosmolike/covariances/cluster_wrapper_cov.cpp) | Count shells, cluster spectra and named selected halo moments. |

Notebook C++ functions take and return Armadillo vectors, matrices and
cubes, with physical axes documented in their headers. Python array
conversion belongs to `generic_interface_cov.cpp`,
`generic_interface_cluster_cov.cpp` and `python_components_cov.cpp`.
These bindings copy inputs so existing notebook arrays and views remain
unchanged; outputs retain their values after later calls. This readable
notebook API is separate from the optimized likelihood interface and C
kernels. The [covariance source guide](cosmolike/covariances/README.md)
describes the individual quantities and their units.

Each project binds the supported wrappers with pybind11 in its own
`interface/interface.cpp`. Its `interface/MakefileCosmolike` selects the
sources to compile. The LSST Y1, DES Y3, DES×Planck, DES cluster,
Roman real, Roman Fourier and Roman KL interfaces bind the galaxy/shear
covariance components. C++ returns whole Gaussian matrices and projects
connected matter tables through every catalog pair. Shared Python
prepares those tables and assembles G, SSC and cNG into the forecast.

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
| [spectra_cov.c](cosmolike/covariances/spectra_cov.c) | Common radial windows and all lens/source Limber cross spectra. |
| [operators_cov.c](cosmolike/covariances/operators_cov.c) | Full-sky, bin-averaged real-space transformations and multipole-band weights. |
| [gaussian_cov.c](cosmolike/covariances/gaussian_cov.c) | Gaussian spectrum pairings, rectangular matrix projection and analytic pair noise. |
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
G+SSC+cNG forecasts with Limber spectra, full-sky angular-bin averages and
spherical-cap pair noise. It uses explicit forecast number densities,
zero intrinsic alignment and massless neutrinos. The cap is an example
footprint, not a measured survey mask.

The notebook's configuration cell exposes one covariance accuracy setting:

```python
settings = survey.configuration(accuracy_boost=1)
```

Here `survey` is the LSST Y1 adapter imported by the notebook. The
supported boosts are 1, 2, 4 and 8. They refine multipole cutoffs and
interpolation tables from the baseline in each project's
`covariance/default.yaml`. A separate `integration_accuracy` selects
precomputed GSL quadrature rules: levels 0/1/2/3/4 use
96/128/256/512/1024 nodes per panel. The global boost leaves this rule
unchanged. Wide angular bins use several panels to resolve oscillations.
Neither setting reruns CAMB or changes the likelihood's YAML files.

The notebook computes boosts 1 and 2 at fixed physical inputs, then
plots changes in correlations, error bars and covariance entries. It also
compares variance ratios across all matrix directions. The highest tested
boost is a numerical comparison reference, not a guarantee of convergence.

The DES cluster notebook also uses
[`forecast_cluster.py`](cosmolike_notebook_utils/covariance/forecast_cluster.py)
to assemble the angular cluster $`6\times2\mathrm{pt}+N`$ matrix, with
absolute counts, all Gaussian and SSC cross blocks, and the same Y
localization used by its mean model. Its cluster cNG treats clusters as
linearly biased matter tracers; count cross covariance contains SSC only.
Selected-cluster one-halo cNG and non-SSC count–spectrum terms are omitted
and recorded in each output. The [DES cluster running guide](../../../projects/des_cluster/covariance/README.md#joint)
explains those limits, the known Y null rows, and the saved row positions.

> [!NOTE]
> The example computes all three covariance components. The shared
> assembler provides full G+SSC+cNG matrices for the supported Limber
> forecast. All-pairs non-Limber spectra and Roman Figure-of-Merit
> convergence remain unfinished. The combined matter halo-response and
> trispectrum helpers currently require massless neutrinos; the cb-aware
> halo statistics alone do not supply a massive-neutrino covariance model.

# Shared Python tools <a name="python"></a>

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

**Step :two:**: run the covariance tests.

    python -m pytest projects/lsst_y1/tests/covariance

The tests use the normal project extension and tracked independent Python
references. Their file guide is
`projects/lsst_y1/tests/covariance/README.md`. No developer-local library
or external study directory is required.

Each project separates `tests/data_vector` from `tests/covariance`.
[cocoa_testing.py](cocoa_testing.py) supplies the data-vector test machinery;
project `tests/cocoa_test_utils.py` files supply their configurations.
[cocoa_covariance_testing.py](cocoa_covariance_testing.py) checks each
project's galaxy/shear forecast adapter with small measured subsets.
The independent component references remain in LSST Y1's covariance tests.

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
height and its growth. This is the code's adopted extension of the
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
`integration_accuracy` for precomputed quadrature rules. Changing them
does not rerun the Boltzmann calculation or the likelihood prediction.
Numerical convergence
and the validity of a physical approximation are separate questions.

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
