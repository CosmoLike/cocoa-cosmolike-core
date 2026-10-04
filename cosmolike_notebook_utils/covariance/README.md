# Covariance tools for notebooks

This package prepares inputs for CoCoA's covariance components and assembles
requested matrix blocks. The caller supplies its initialized project
interface; the package never imports a survey's compiled module or loads
a likelihood covariance. The first complete teaching example is LSST Y1's
`EXAMPLE_EVALUATE_COVARIANCE.ipynb`.

A covariance describes the joint scatter of measured two-point functions.
The Gaussian part follows from pairs of power spectra. Connected
non-Gaussian covariance (cNG) describes the remaining connected four-point
function; super-sample covariance (SSC) describes changes caused by matter
fluctuations larger than the survey. See
[Krause & Eifler](https://arxiv.org/abs/1601.05779),
[Takada & Hu](https://arxiv.org/abs/1302.6994), and the
[C implementation and physics guide](../../cosmolike/covariances/README.md).

## Files

| File | Calculation and responsibility |
| --- | --- |
| `accuracy.py` | One `accuracy_boost` resolves the signal/mask cutoffs and radial, angular and window sampling together. |
| `gaussian.py` | Complete Wick pairings, conversion of source spectra to observed shear, rectangular Gaussian projection, and real-space pair noise. `shear_gaussian` is the shared small single-source example. |
| `geometry.py` | Convert number densities to noise powers, construct a raw spherical-cap mask spectrum, and resolve nearly opposite wavevectors with a planar angular quadrature. |
| `halo.py` | Arrange physical power and halo moments for the five trispectrum contributions and isotropic density response. The combined matter prescription requires massless neutrinos. |
| `survey.py` | Assemble complete real-space G, SSC and cNG matrices with all cross-bin blocks under the specified massless, Limber forecast model. |
| `sampling.py` | `DenseLogTable`: coarse exact samples → cubic construction of a dense uniform log-k table → linear lookup by arithmetic index. Signed quantities remain signed. |
| `diagnostics.py` | Check symmetry, diagonal variances, raw/correlation eigenvalues and generalized covariance ratios. No clipping or diagonal correction is applied. |
| `reference/` | Independent NumPy/SciPy/mpmath algorithms for component tests. Production assembly never calls these oracles. |
| [`../plot_covariances.py`](../plot_covariances.py) | Matplotlib correlation comparisons, component maps/histograms and scale-dependent standard deviations. |

Project folders own survey numbers, redshift files, cosmology, nuisance
parameters, initialization and numerical settings. Copy that small adapter
when enabling another project; reuse these calculations and plotters.
The component bindings require a project interface built with the shared
covariance C/C++ sources.

## Array and unit conventions

Fields are numbered from zero, with lenses preceding sources. Supplied
signal matrices use `[multipole, field, field]`; noise is a separate
`[field]` array. A covariance between measured AB and CD needs AC, BD, AD
and BC, including cross-bin spectra excluded from the data vector.
Catalog shot noise is $1/n$; source shape noise is
$\sigma_\epsilon^2/n$, with $n$ per steradian and dispersion per component.

The low-level `interface.covariance_*` calls accept contiguous `float64`
arrays (field/band IDs use `int32`) and return owned NumPy arrays. The
returned values survive later interface calls or cosmology changes.
Distances use $c/H_0$, wavenumbers its inverse, and matter power
$(c/H_0)^3$. Thus a value of $k$ in $h/{\rm Mpc}$ is multiplied by
2997.92458 before a core power/halo call. Angles are radians and survey
areas steradians; project adapters make the conversion from arcminutes
and square degrees explicit.

## Gaussian projection

`gaussian_block` receives complete spectra, noise, four field IDs and
left/right operators. Each output is the sum of left operator × harmonic
covariance × right operator. Rectangular inputs support independently
requested subblocks; Python can later assign these to MPI processes.
C uses OpenMP within a process and never calls MPI. BLAS stays at one thread.

`realspace_block` treats pure white noise separately. Its infinite
multipole sum becomes a pair-count expression in angular space; using
that expression avoids losing noise power above a finite multipole cutoff.
Signal and mixed signal-noise retain the supplied finite multipole sum.
Both sides must use the same disjoint angular bins and footprint.

For a mask, the raw angular power has monopole
$C_0^W=\Omega^2/(4\pi)$. `cap_mask` is an illustrative footprint, not an
approximation to every survey. `covariance_mask_pair_area` and
`covariance_ssc_mask_variance` both use this raw normalization.

## Halo responses and SSC

`halo_trispectrum` retains 1-halo, 2-halo (1+3), 2-halo (2+2), 3-halo
and 4-halo terms separately. `halo_power_response` returns a dimensional
isotropic $dP/d\delta_b$ using the fractional halo response transferred
to the supplied nonlinear power. This halo approximation is not a
calibration of nonlinear tidal responses or a massive-neutrino model.

The component interface accepts that response through
`covariance_ssc_shell_response`, along with common radial windows and the
projected catalog-mean response. `covariance_ssc_mask_variance` supplies
the long-mode Limber background strength, with units of length. Multiplying
it by the radial integration weight and contracting common shell responses
with `covariance_project` gives the SSC matrix. Keep every cross-lens
block. Removing selected cross correlations can make a covariance indefinite.

## Real-space and Fourier assembly

`survey.realspace_covariance` and `survey.fourier_covariance` receive an initialized project interface,
resolved integration settings, the observable row map and catalog noise.
It returns separate Gaussian, SSC, cNG and total matrices, the projected
mean signals and elapsed times by calculation stage. CAMB setup, output
writing and eigenvalue diagnostics remain outside this function.
`survey.observable_rows` puts angular bins inside each tomographic row,
with xi+, xi-, galaxy--shear and galaxy clustering in that order.
Fourier rows omit xi-: one E-mode spectrum supplies both real-space shear
correlations. Integer band endpoints are inclusive, and each multipole
receives weight proportional to its mode count, $2\ell+1$. Refinement
holds those endpoints fixed so it compares the same measurement.

Fourier means average the core angular spectra directly. Real-space
means retain Cocoa's extra source-leg factor when using unit-normalized
spin kernels. The chosen convention is applied consistently to Gaussian,
SSC and cNG signals; white noise receives neither conversion.
The low-level supplied-spectrum wrappers let users state their own field
conventions explicitly; see their Python `help(...)` documentation.

The supported model uses massless neutrinos, linear galaxy bias, zero
intrinsic alignment, magnification and RSD, Limber spectra and a spherical
cap footprint. SSC uses the isotropic fractional halo response transferred
to the chosen nonlinear matter power, with galaxy survey-mean subtraction.
The five halo cNG contributions are projected with every cross-lens block
retained. These approximations define a forecast; they do not reproduce
all physical choices of a supplied likelihood covariance.

The costly matter trispectrum depends on distance and two wavenumbers,
not on the observed catalog pair. The assembler computes it once per
radial shell and shares it across all catalog pairs. Its multipole table
is linearly interpolated in ln(ell+1/2), preserving signed values. The
angular kernels still sum every integer multipole: the code first projects
the interpolation weights, then contracts the smaller trispectrum table.
This is algebraically the same angular projection of the interpolated
table. Refining the coarse table is still necessary to test its accuracy.

Numerical controls for this low-level assembly are explicit in its
function documentation. The notebook's single-source Gaussian boost is
not a certified full-survey setting. Complete matrices must pass positivity
checks and resolution/Fisher comparisons before use for inference.

## One accuracy boost

Use `covariance_accuracy(accuracy_boost=1)`, or set `accuracy_boost` in the
project's configuration function. Supported boosts are 1, 2, 4 and 8.
Each step doubles signal and mask multipole cutoffs, radial and angular
quadrature nodes, and window-grid intervals. The resolved dictionary is
available for inspection and saving; ordinary notebook users need only
change the boost. This control belongs to covariance and leaves CAMB and
data-vector accuracy settings unchanged.

Boost 1 is the teaching pilot. Boost 8 reaches 80,000 signal multipoles,
32,768 mask modes, 512 radial nodes per panel, 1,024 angular nodes per bin
and 32,769 window nodes. These are numerical resolutions, not certified
survey-accuracy labels. Check refinement and FoM/errors for the intended
physical model. The same boost raises mass/angular resolution in the halo preparation
helpers and reduces the response finite-difference step. It does not
certify the full survey assembly or its physical choices.

## Interpolation and numerical checks

Sample expensive quantities at **fixed physical wavenumber** before
constructing a `DenseLogTable`. Cubic interpolation only builds its dense
table. Repeated queries read adjacent entries linearly; queries outside
the sampled range fail explicitly. Increase both coarse and dense counts,
and compare off-grid values to direct calculations. Increasing dense
sampling cannot restore a feature missing from coarse samples.

A positive diagonal or a few positive subblocks is insufficient. Check
the complete total matrix, any selected likelihood matrix, and complete
cross-block coverage. An individual cNG contribution need not itself be
positive definite. Generalized eigenvalues bound variance changes across
all directions, but final accuracy should be assessed with parameter
errors and marginalized Fisher Figures of Merit at representative
cosmologies. The data-vector $|\Delta\chi^2|<0.2$ rule is not a covariance
convergence criterion.

The included single-source Gaussian example is executable. A validated
full survey G+SSC+cNG generator, all-pairs non-Limber corrections and
production FoM convergence remain separate work.
