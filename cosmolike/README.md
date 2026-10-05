# Table of contents

1. [What this directory calculates](#overview)
2. [Following a prediction through the code](#prediction)
3. [C files: cosmology, galaxies and shear](#c_files)
4. [C++ production interfaces](#interfaces)
5. [C++ wrappers for Python notebooks](#wrappers)
6. [Cluster observables](#clusters)
7. [Why data vectors and covariances manage work differently](#ownership)
8. [Running an example and its tests](#running)
9. [Appendix](#appendix)
   1. [Which units and array axes matter?](#units)
   2. [How are expensive calculations reused?](#numerics)
   3. [Where do covariance calculations belong?](#covariances)

# What this directory calculates <a name="overview"></a>

These C and C++ files predict the **mean data vector** used by Cocoa's
CosmoLike likelihoods. A data vector collects measurements at several
angular scales and in several redshift bins. Splitting a galaxy sample
into redshift bins is called **tomography**: it lets the analysis follow
structure at different distances.

The ordinary galaxy and shear analysis combines three kinds of correlation,
usually called **3x2pt**:

| Measurement | Physical question | Spectrum | Real-space correlation |
| --- | --- | --- | --- |
| Galaxy clustering | How often do galaxy positions occur together? | $`C_\ell^{gg}`$ | $`w(\theta)`$ |
| Galaxy–galaxy lensing | How are background shapes distorted around foreground galaxies? | $`C_\ell^{g\gamma}`$ | $`\gamma_t(\theta)`$ |
| Cosmic shear | How are two background shapes distorted by shared foreground matter? | $`C_\ell^{\gamma\gamma}`$ | $`\xi_+(\theta),\xi_-(\theta)`$ |

Here $`\theta`$ is angular separation, and multipole $`\ell`$ labels angular
scale in a spherical-harmonic expansion. Larger multipoles describe finer
angular structure. The two shear correlations account for the orientation
of each galaxy's shape relative to the pair separation.

The directory also contains CMB-lensing correlations, halo-occupation
calculations and the [cluster extension](#clusters). A project's interface
and likelihood choose which of these calculations enter its prediction.
[Krause & Eifler](https://arxiv.org/abs/1601.05779) describe the multiprobe
CosmoLike framework and its cosmological and nuisance parameters.

The [repository guide](../README.md) covers installation entry points and
shared Python tools. This page explains the source files. The
[covariance guide](covariances/README.md) explains the separate calculation
of fluctuations around the mean prediction.

# Following a prediction through the code <a name="prediction"></a>

The prediction combines three-dimensional structure with the survey's
selection of galaxies. Each stage answers a different physical question:

| Stage | What it supplies | Main source |
| --- | --- | --- |
| Cosmological input | Matter power, growth and the relation between distance and redshift. | [cosmo3D.c](cosmo3D.c) |
| Galaxy selection | The redshift distribution of each lens or source bin. | [redshift_spline.c](redshift_spline.c) |
| Projection weights | How strongly each distance contributes to the observed field. | [radial_weights.c](radial_weights.c) |
| Angular prediction | Spectra and their averages over measured angular bins. | [cosmo2D.c](cosmo2D.c) |
| Likelihood assembly | The ordered prediction, calibration factors, retained entries and comparison with data. | [generic_interface.cpp](generic_interface.cpp) |

## From spatial fluctuations to angular spectra

A Boltzmann solver such as CAMB supplies power-spectrum and background
tables through Cocoa's Python layer. A power spectrum describes the
strength of fluctuations at spatial wavenumber $`k`$. The C readers in
`cosmo3D.c` interpolate these supplied tables; they do not run CAMB.

The galaxy distributions determine which distances the survey samples.
For galaxy density, the projection weight includes the abundance of
selected galaxies and their bias relative to matter. For shear, it also
includes the lensing geometry: matter can distort only sources behind it.

For a projected matter contribution, the Limber approximation has the form

```math
C_\ell^{AB}=\int d\chi\,
\frac{W_A(\chi)W_B(\chi)}{f_K^2(\chi)}
P\!\left(\frac{\ell+1/2}{f_K(\chi)},a(\chi)\right).
```

Here $`A,B`$ label the observed fields, $`\chi`$ is radial comoving distance,
$`f_K`$ is transverse comoving distance, and $`a=1/(1+z)`$ is scale factor.
The weights $`W_A,W_B`$ describe the selection and projection of those
fields. Additional galaxy-bias and intrinsic-alignment terms depend on
the selected model.

Limber associates an angular mode with the transverse wavenumber
$`k=(\ell+1/2)/f_K`$. On large angular scales, the non-Limber galaxy and
galaxy–shear paths retain radial oscillations through spherical-Bessel
transforms. `cosmo2D.c` computes these with FFTLog, which uses Fourier
transforms on a logarithmic grid. The project chooses the relevant
non-Limber settings.

## From angular spectra to the likelihood vector

For a real-space prediction, `cosmo2D.c` combines many multipoles with
full-sky angular kernels. These kernels include the average over each
measured angular bin. The result is a bin average, rather than simply the
correlation evaluated at the bin's displayed center.

The C++ production interface places the predicted observables in the
project's required order and applies its calibration and selection rules.
A **likelihood mask** selects which data-vector entries survive the scale
cuts. It is distinct from a sky-footprint mask used to describe survey
geometry.

With a supplied covariance $`\mathcal{C}`$, the Gaussian comparison uses

```math
\chi^2=(d-t)^T\mathcal{C}^{-1}(d-t).
```

$`d`$ is the measured vector and $`t`$ the prediction, with matching order
and selection. Loading and inverting that covariance is part of likelihood
setup. Generating a new covariance belongs to `covariances/`.

# C files: cosmology, galaxies and shear <a name="c_files"></a>

Each `.c` file contains calculations; its matching `.h` file declares
the functions that other files can call. Start with the header to locate
a quantity, then read its function comment for assumptions and units.

## Inputs and radial projection

| Source and header | Responsibility |
| --- | --- |
| [cosmo3D.c](cosmo3D.c), [header](cosmo3D.h) | Read linear/nonlinear power, cb power, distances and growth. Build the FFTLog mass-variance tables and their logarithmic mass slopes. |
| [redshift_spline.c](redshift_spline.c), [header](redshift_spline.h) | Interpolate lens/source redshift distributions, apply photo-z changes, integrate lensing efficiencies and map the retained redshift-bin pairs. |
| [radial_weights.c](radial_weights.c), [header](radial_weights.h) | Form the density, source, lensing, magnification and redshift-space-distortion weights used in angular projections. |
| [cosmo2D.c](cosmo2D.c), [header](cosmo2D.h) | Integrate angular spectra, implement galaxy non-Limber corrections and transform spectra into real-space bin averages. It also contains the CMB-lensing spectra and beam/pixel factors. |

Photometric-redshift, or **photo-z**, parameters change the inferred
redshift distribution of a sample. Magnification changes observed galaxy
counts because lensing changes apparent flux and area. Redshift-space
distortions describe the effect of peculiar velocities on inferred radial
positions. Their weights and inclusion depend on the observable and model.

## Galaxy and halo physics

| Source and header | Responsibility |
| --- | --- |
| [bias.c](bias.c), [header](bias.h) | Evaluate galaxy-bias coefficients and their redshift dependence. These specify how the galaxy field responds to the underlying matter field. |
| [IA.c](IA.c), [header](IA.h) | Evaluate intrinsic-alignment amplitudes and their bin or redshift dependence. The angular calculation combines these amplitudes with the required spectra. |
| [pt_cfastpt.c](pt_cfastpt.c), [header](pt_cfastpt.h) | Build perturbative galaxy-bias and intrinsic-alignment tables through the [C FAST-PT engine](../cfastpt/). Upsample its results for repeated interpolation. |
| [halo.c](halo.c), [header](halo.h) | Calculate halo abundance, bias, concentration, density profiles, halo occupation and the corresponding galaxy–matter, galaxy–galaxy and halo-alignment terms. |
| [baryons.c](baryons.c), [header](baryons.h) | Load and interpolate baryonic power-ratio tables used to modify the matter spectrum. |
| [cosmo2D_scuts.c](cosmo2D_scuts.c), [header](cosmo2D_scuts.h) | Measure how spatial wavenumbers contribute to shear and CMB-lensing–shear observables, for physical-scale-cut diagnostics. |

**Intrinsic alignment** is a correlation of galaxy shapes with their local
environment, in addition to gravitational lensing. The nonlinear alignment
model (NLA) and tidal alignment and tidal torquing model (TATT) supply
different contributions. [Blazek et al.](https://arxiv.org/abs/1708.09247)
describe the alignment expansion underlying TATT.

The perturbative terms require integrals coupling different spatial
modes. FAST-PT evaluates these using Fourier transforms of logarithmically
sampled spectra. `pt_cfastpt.c` connects that engine to the tables consumed
by CosmoLike. [McEwen et al.](https://arxiv.org/abs/1603.04826) explain the
FAST-PT method.

**Halo occupation** specifies how many central and satellite galaxies
occupy a halo of a given mass. `halo.c` integrates that population with
halo profiles to predict galaxy statistics. Halo abundance and bias use
the cold-dark-matter-plus-baryon field, called **cb**, including its
mass variance. Total-matter variance is also available for diagnostics.
The [neutrino and halo guide](../README.md#neutrinos) explains the field
choices, profile-mass convention and concentration prescription.

## Shared numerical support

| Source and header | Responsibility |
| --- | --- |
| [basics.c](basics.c), [header](basics.h) | Numerical integration, interpolation, splines, padded array allocation and other shared helpers. |
| [structs.c](structs.c), [header](structs.h) | Cosmology, survey, nuisance and numerical state, including defaults, resets and table settings. |
| [tinker_emulator.c](tinker_emulator.c), [header](tinker_emulator.h) | Legacy halo mass-function and bias emulator implementation. The current project builds exclude this file. |

Use the interface setters to change cosmology and nuisance parameters.
Those setters update the state used to decide which cached calculations
must be rebuilt. Changing a structure directly can leave a previously
computed table associated with the wrong inputs.

# C++ production interfaces <a name="interfaces"></a>

The `_interface` layer serves the normal Python likelihood and command-line
workflow. It prepares the C state, assembles the requested prediction and
evaluates the comparison with the supplied data. It calls the C physics
functions directly, without going through the notebook `_wrapper` API.

| Source and header | Responsibility |
| --- | --- |
| [generic_interface.cpp](generic_interface.cpp), [header](generic_interface.hpp) | Cosmology/survey setters, numerical configuration, prediction assembly, supplied data and covariance handling, and likelihood evaluation. |
| [generic_interface_cluster.cpp](generic_interface_cluster.cpp), [header](generic_interface_cluster.hpp) | Cluster-model setters, joint block ordering, cluster-lensing transformation, selection factors and joint likelihood data handling. |

Each Cocoa project has its own `interface/interface.cpp`. It uses
**pybind11**, the library that exposes C++ functions to Python, to register
the functions available in that project's compiled module. Its
`interface/MakefileCosmolike` selects the sources to compile.

```text
Project likelihood and YAML
    -> project's Python bindings
    -> generic_interface functions
    -> shared C calculations
    -> ordered, selected prediction and likelihood
```

The C++ layer handles the connection to the analysis. The C files own the
power-spectrum projections, halo integrals and angular transforms. The
same physical calculation is therefore available to both production runs
and notebook inspection.

# C++ wrappers for Python notebooks <a name="wrappers"></a>

We chose **Armadillo** to make the Python notebook API easy to develop
and use. Spectra, halo quantities and intermediate calculations can be
exposed as arrays with physical axes, such as multipole and redshift bin.
Armadillo is a C++ library for vectors, matrices and three-dimensional
arrays called cubes.

C++ serves as a thin connecting layer. **pybind11** exposes the functions
to Python, and **CARMA** converts between Armadillo and NumPy arrays.
The wrappers arrange inputs and outputs around calls to the shared C
calculations. Their purpose is to support notebook experimentation.

| Source and header | Quantities available to inspect |
| --- | --- |
| [cosmo2D_wrapper.cpp](cosmo2D_wrapper.cpp), [header](cosmo2D_wrapper.hpp) | Angular spectra, real-space correlations, bin centers and tomographic pair maps. |
| [halo_wrapper.cpp](halo_wrapper.cpp), [header](halo_wrapper.hpp) | Halo abundance, bias, concentration, profiles and halo-occupation statistics, with scalar and array queries. |
| [cosmo2D_scuts_wrapper.cpp](cosmo2D_scuts_wrapper.cpp), [header](cosmo2D_scuts_wrapper.hpp) | Wavenumber-response and cumulative scale-cut diagnostics. |
| [cosmo2D_wrapper_cluster.cpp](cosmo2D_wrapper_cluster.cpp), [header](cosmo2D_wrapper_cluster.hpp) | Cluster counts, spectra, correlations and cluster-lensing outputs before or after the likelihood's transformation. |
| [halo_wrapper_cluster.cpp](halo_wrapper_cluster.cpp), [header](halo_wrapper_cluster.hpp) | Richness-selection probabilities, selected abundance/bias, halo profiles and cluster radial weights. |

For example, `ci.p_gm(k, a, ni)` is registered by the project, enters
`p_gm_cpp` in `halo_wrapper.cpp`, and calls `p_gm` in `halo.c`.
Here `k` is wavenumber, `a` scale factor and `ni` the lens-bin index.
The `_cpp` suffix identifies the C++ adapter, rather than a second
galaxy–matter power model.

Read the matching wrapper header before interpreting an array. The
ordinary angular wrappers expose selected tomographic pairs; unused
entries in a rectangular output can be zero because no pair was selected.
Such a zero is not a computed covariance cross spectrum. Cluster outputs
with four physical axes use explicitly documented NumPy arrays, since an
Armadillo cube has only three axes.

# Cluster observables <a name="clusters"></a>

Cluster counts and correlations require a selection by observed
**richness**, a proxy for halo mass. A halo of a given mass can enter
different richness bins with different probabilities. Integrating those
probabilities over the halo population determines the selected abundance
and its clustering bias.

| Source and header | Responsibility |
| --- | --- |
| [halo_cluster.c](halo_cluster.c), [header](halo_cluster.h) | Mass–richness probabilities, richness-bin abundance and bias, and the selected one-halo cluster–matter power. |
| [redshift_spline_cluster.c](redshift_spline_cluster.c), [header](redshift_spline_cluster.h) | Cluster redshift selection, normalized distributions, lensing efficiencies and cluster/source/galaxy pair maps. |
| [radial_weights_cluster.c](radial_weights_cluster.c), [header](radial_weights_cluster.h) | Cluster-density and magnification weights for angular projection. |
| [cosmo2D_cluster.c](cosmo2D_cluster.c), [header](cosmo2D_cluster.h) | Expected counts, cluster–shear, cluster–galaxy and cluster–cluster spectra and correlations. |
| [structs_cluster.c](structs_cluster.c), [header](structs_cluster.h) | Cluster sample, model and nuisance state. |

The cluster C++ interface combines these observables with the ordinary
galaxy/shear vector. It also applies the configured cluster-lensing
transformation and selection factors. A raw tangential-shear prediction
and the transformed cluster-lensing quantity used by a likelihood can
therefore differ; the notebook wrappers expose both stages.

# Why data vectors and covariances manage work differently <a name="ownership"></a>

The two calculations answer different questions. The **mean data vector**
predicts the measurements for a proposed cosmology and nuisance parameters.
The **covariance** describes how those measurements fluctuate together
across possible realizations of the survey.

In this workflow, a covariance is generated before inference and supplied
to the likelihood. **Covariance generation never runs inside MCMC.**
MCMC, or Markov chain Monte Carlo, explores parameter values by repeatedly
calling the likelihood. Those calls change the mean prediction while
using the same supplied covariance.

| Design question | Data-vector calculation | Covariance generation |
| --- | --- | --- |
| What repeats? | Likelihood evaluations at proposed parameter values. | Many matrix blocks at one chosen model and survey configuration. |
| What is reused? | Tables whose dependencies did not change, and tables queried many times within an evaluation. | Spectra, halo quantities and projection operators shared by different blocks. |
| Who retains the main tables? | Often the C function, through a persistent cache. | The calling calculation, through explicit arrays and workspaces. |
| When are results released? | A cache can persist until replacement or process termination. | A calculation can release its workspace when assembly finishes. |
| Where is parallel work? | Independent table entries or predicted observables. | Independent table entries and complete covariance blocks. |

## A data-vector table remembers its last inputs

Suppose many angular-spectrum queries need the same three-dimensional
power table. Integrating the underlying model for every query would repeat
work. The function can build the table once, then interpolate it for each
requested wavenumber and redshift.

A local C variable declared `static` retains its value after the function
returns. This lets a function retain a table pointer and information about
the inputs used to fill that table. On a later call, it can decide whether
the stored values still apply.

For a dynamically allocated table, `static` preserves the **pointer**;
the allocation itself remains valid until it is freed. Declaring the
pointer `static` does not automatically fill the table, track its physics
dependencies, or free its memory. Those are explicit responsibilities of
the function that owns it.

The usual sequence is:

1. Check whether the required workspace exists and has the right shape.
2. Check whether the table's cosmology, nuisance or numerical inputs changed.
3. Rebuild the affected values when necessary.
4. Interpolate the table for the requested quantity.

These checks matter because different parameter changes affect different
calculations. A distance table need not change when only a galaxy-bias
parameter changes. A galaxy spectrum that depends on that bias does need
updating. Interface setters update the state used by these checks.

This usually retains the most recently built table; it does not store
every cosmology visited by a chain. When the cosmology changes, a rebuild
may be necessary. The new table still saves work because many queries
within that likelihood evaluation read it.

## A covariance shares a prepared set of inputs across blocks

A covariance contains many related calculations at the same model point.
For example, the covariance between two galaxy–shear spectra depends on
galaxy clustering and shear correlations as well as galaxy–shear spectra.
Several blocks can therefore need exactly the same field-pair spectra.

The covariance driver prepares those spectra once and retains the arrays
while it assembles the matrix. Gaussian assembly reads those shared
spectra. SSC and cNG assembly similarly reuse their halo responses,
trispectra and integration operators. A block does not need to discover
whether another block has already calculated its inputs.

```text
Choose one model, survey and accuracy configuration
    -> prepare common spectra, halo tables and projection operators
    -> retain them while calculating all requested matrix blocks
    -> assemble and save Gaussian, SSC, cNG and total matrices
    -> release temporary workspaces

Later, start MCMC
    -> read the saved covariance during likelihood setup
    -> reuse it while repeatedly calculating the mean data vector
```

**Ownership** means responsibility for an array's lifetime: who creates
it, keeps it available and eventually releases it. An assembly function
can borrow an input array during a call without owning it. It must finish
using that array before its owner releases the storage.

Explicit inputs also help notebook work. A student can inspect a supplied
spectrum or operator and then examine the covariance it produces. The
notebook wrapper exposes these quantities as Python arrays; the production
interface supplies inputs directly to the same C calculations. Neither
C++ layer needs its own implementation of the physics.

The caller can reuse a retained operator in a separate calculation when
its bin edges, multipole range and numerical settings are unchanged.
Spectra require more: the cosmology, catalog selections and relevant
nuisance settings must also agree. Explicit ownership makes these
dependencies visible, but does not check them automatically.

## Why this helps OpenMP

OpenMP workers share a process's memory. A function-local `static` table
is normally shared by all those workers; each worker does not receive its
own private copy merely because the variable is declared inside a function.

Reading a completed shared table is straightforward. Building or replacing
that table while another worker reads it requires coordination. Two
workers entering a lazy first-call builder together can otherwise write
the same storage or observe an unfinished result.

The covariance workflow prepares shared inputs before the block loops.
Workers then read those inputs, use their own temporary buffers and write
separate output blocks. Keeping a complete sum on one worker also avoids
changing its addition order when the number of workers changes.

This is not a claim that covariance code is independent of all persistent
state. It reads initialized cosmology and survey state, and some shared
data-vector readers have their own caches. Those readers must be prepared
before parallel work that relies on them. The same care is needed when
making FFTW plans: plan construction is serial, and workers execute with
their own buffers.

## Choosing the lifetime that matches the calculation

Persistent caching pays when later calls reuse a result enough to justify
retaining its memory and maintaining its dependency checks. This is common
in repeated likelihood work. A separately generated covariance can instead
reuse its inputs throughout one assembly and release large arrays when
that calculation ends.

The tradeoff appears when separate covariance runs repeat the same setup.
An explicit workspace can be retained by the caller, or a measured need
may justify a small persistent cache. Neither approach is intrinsically
faster: speed comes from avoiding repeated work, organizing memory and
distributing independent calculations effectively.

The `static` keyword also appears on helper **functions**. There it limits
the function's visibility to its source file; it does not create a cache.
Likewise, OpenMP's `schedule(static)` assigns loop iterations to workers;
it says nothing about whether an array persists between function calls.

Keep these lifetime choices separate from numerical accuracy. The
data-vector and covariance calculations can use the same physical readers
while owning different grids, interpolation tables and integration rules.
A converged mean prediction does not establish a converged covariance.

# Running an example and its tests <a name="running"></a>

We assume Cocoa and LSST Y1 are installed, users have run
`conda activate cocoa`, and Bash is open in `cocoa/Cocoa`. These shared
sources are compiled through the project, not as a standalone program.

**Step :one:**: activate Cocoa's private Python environment.

    source start_cocoa.sh

**Step :two:**: enable and compile the LSST Y1 data-vector interface.

    unset IGNORE_COSMOLIKE_LSST_Y1_CODE
    source ./projects/lsst_y1/scripts/compile_lsst_y1.sh

**Step :three:**: evaluate the cosmic-shear likelihood at the YAML cosmology.

    cobaya-run ./projects/lsst_y1/EXAMPLE_EVALUATE1.yaml

**Step :four:**: run the separate data-vector tests.

    python -m pytest ./projects/lsst_y1/tests/data_vector

The project's YAML and dataset choose the model, numerical settings,
binning, data, covariance and scale cuts. Other projects supply their own
examples and tests; see the [Cocoa project guide](https://github.com/CosmoLike/cocoa/tree/main/Cocoa/projects).
The [notebook instructions](../README.md#notebooks) show how to inspect
intermediate predictions interactively.

# Appendix <a name="appendix"></a>

## Which units and array axes matter? <a name="units"></a>

The cosmology setter converts astronomical input units to the units used
by the C calculations. A direct halo or spectrum query can instead expect
those internal units. The wrapper header specifies which convention its
arguments use.

| Quantity | Convention to check |
| --- | --- |
| Redshift and time | $`z`$ is redshift; $`a=1/(1+z)`$ is scale factor. They are different arguments. |
| Wavenumber | Core power/halo calls use $`(c/H_0)^{-1}`$; scale-response notebook wrappers use $`h/\mathrm{Mpc}`$. |
| Power | Cosmology tables contain natural logarithms of power in $`(\mathrm{Mpc}/h)^3`$; core power readers return power in $`(c/H_0)^3`$. |
| Mass | Halo queries use $`M_\odot/h`$; `lnM` means its natural logarithm. |
| Angle | Core integrations use radians; the real-space bin-center notebook helper returns arcminutes. |
| Bin index | C and Python indices start at zero. Tomographic pair indices use the project's pair maps. |

Here $`h=H_0/(100\,\mathrm{km}\,\mathrm{s}^{-1}\,\mathrm{Mpc}^{-1})`$,
and $`c/H_0=2997.92458\,\mathrm{Mpc}/h`$. Multiply a wavenumber in
$`h/\mathrm{Mpc}`$ by 2997.92458 before passing it to a core halo-power
query. The [input-table guide](../README.md#units) also explains flattening
order and the separate growth grid.

## How are expensive calculations reused? <a name="numerics"></a>

Many calls read a cached table: an expensive calculation is performed
when its inputs change, then reused for many queries. For example,
`halo.c` integrates over halo mass to build spectra on scale-factor and
wavenumber grids. Later calls interpolate those spectra. A first call
can consequently include work that later calls reuse.

Several builders evaluate a coarse set of samples, use cubic splines to
fill a denser table, and use linear interpolation for repeated reads.
`pt_cfastpt.c` uses this separation to avoid performing a larger FFTLog
calculation solely to obtain more output samples. Increasing the dense
table alone cannot recover structure absent from the coarse samples.

**OpenMP** distributes independent work among CPU threads. **SIMD**
performs the same arithmetic on several values in one instruction;
**SIMDe** provides portable forms of those vector operations. They address
different levels of parallelism and are used together in the C code.
FFTW transform plans are prepared serially and reused with worker buffers.

The project YAML controls table resolution, integration accuracy and
non-Limber sampling. Keep its tested base settings when varying the global
accuracy boost; a larger interpolation table does not by itself establish
convergence of an angular transform or physical model. See the
[accuracy and parallelism guide](../README.md#numerics) for the supported
build modes and the division between OpenMP and Python process scheduling.

## Where do covariance calculations belong? <a name="covariances"></a>

The files described above optimize the mean prediction and its likelihood
evaluation. `covariances/` owns covariance-specific integrations, tables
and field-pair requirements. Its [README](covariances/README.md) explains
Gaussian, super-sample and connected non-Gaussian terms.

A likelihood can read and invert a supplied covariance even when covariance
generation is omitted from the build. Conversely, a covariance calculation
can require crossed redshift-bin spectra that are absent from the measured
data vector. Keep those additional calculations in `covariances/`, with
their own numerical settings.
