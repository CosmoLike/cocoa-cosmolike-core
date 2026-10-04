# Table of contents

1. [Covariance tools for notebooks](#covariance_tools_for_notebooks)
2. [Files](#files)
3. [Array and unit conventions](#array_and_unit_conventions)
4. [Gaussian projection](#gaussian_projection)
5. [Halo responses and SSC](#halo_responses_and_ssc)
6. [Real-space and Fourier assembly](#real_space_and_fourier_assembly)
7. [One accuracy boost](#one_accuracy_boost)
8. [Interpolation and numerical checks](#interpolation_and_numerical_checks)

# Covariance tools for notebooks <a name="covariance_tools_for_notebooks"></a>

This package prepares inputs for CoCoA's covariance components and assembles
requested matrix blocks. The caller supplies its initialized project
interface; the package never imports a survey's compiled module or loads
a likelihood covariance. Each project supplies an
`EXAMPLE_EVALUATE_COVARIANCE.ipynb`.

A covariance describes the joint scatter of measured two-point functions.
The Gaussian part follows from pairs of power spectra. Connected
non-Gaussian covariance (cNG) describes the remaining connected four-point
function; super-sample covariance (SSC) describes changes caused by matter
fluctuations larger than the survey. See
[Krause & Eifler](https://arxiv.org/abs/1601.05779),
[Takada & Hu](https://arxiv.org/abs/1302.6994), and the
[C implementation and physics guide](../../cosmolike/covariances/README.md).

## Files <a name="files"></a>

| File | Calculation and responsibility |
| --- | --- |
| `accuracy.py` | One `accuracy_boost` resolves the signal/mask cutoffs and radial, angular and window sampling together. |
| `gaussian.py` | Complete Wick pairings, conversion of source spectra to observed shear, rectangular Gaussian projection, and real-space pair noise. `shear_gaussian` is the shared small single-source example. |
| `geometry.py` | Convert number densities to noise powers, construct a raw spherical-cap mask spectrum, and resolve nearly opposite wavevectors with a planar angular quadrature. |
| `halo.py` | Arrange physical power and halo moments for the five trispectrum contributions and isotropic density response. The combined matter prescription requires massless neutrinos. |
| `forecast.py` | Initialize a project forecast, bind survey settings, compute either space and save arrays with resolved settings. |
| `survey.py` | Assemble real/Fourier G, SSC and cNG matrices with all cross-bin blocks under the specified massless, Limber forecast model. |
| `survey_cluster.py` | Prepare the joint cluster row layout, absolute count densities, normalized cluster windows and every internal cluster cross spectrum. |
| `forecast_cluster.py` | Assemble the joint angular forecast with count Poisson noise, common SSC, biased-tracer cNG and the optional Y transformation. Archive its omitted physics with the result. |
| `counts_cluster.py` | Integrate supplied selected abundances into count means, Poisson noise and SSC. Project the separate non-SSC count–matter-spectrum cross terms from selected halo moments. This is not a full cluster forecast. |
| `transform_cluster.py` | Apply the cluster-lensing localization to both sides of a supplied joint covariance, including every count and two-point cross block. |
| `sampling.py` | `DenseLogTable`: coarse exact samples → cubic construction of a dense uniform log-k table → linear lookup by arithmetic index. Signed quantities remain signed. |
| `diagnostics.py` | Check symmetry, diagonal variances, raw/correlation eigenvalues and generalized covariance ratios. No clipping or diagonal correction is applied. |
| `reference/` | Independent NumPy/SciPy/mpmath algorithms for component tests. Production assembly never calls these oracles. |
| [`../plot_covariances.py`](../plot_covariances.py) | Matplotlib correlation comparisons, component maps/histograms and scale-dependent standard deviations. |

Project folders own survey numbers, redshift files, cosmology, nuisance
parameters, initialization and numerical settings. Copy that small adapter
when enabling another project; reuse these calculations and plotters.
The component bindings require a project interface built with the shared
covariance C/C++ sources.

## Array and unit conventions <a name="array_and_unit_conventions"></a>

Fields are numbered from zero, with lenses preceding sources. Supplied
signal matrices use `[multipole, field, field]`; noise is a separate
`[field]` array. A covariance between measured AB and CD needs AC, BD, AD
and BC, including cross-bin spectra excluded from the data vector.
Catalog shot noise is $`1/n`$; source shape noise is
$`\sigma_\epsilon^2/n`$, with $`n`$ per steradian and dispersion per component.

The low-level `interface.covariance_*` calls accept contiguous `float64`
arrays (field/band IDs use `int32`) and return owned NumPy arrays. The
returned values survive later interface calls or cosmology changes.
Distances use $`c/H_0`$, wavenumbers its inverse, and matter power
$`(c/H_0)^3`$. Thus a value of $`k`$ in $`h/{\rm Mpc}`$ is multiplied by
2997.92458 before a core power/halo call. Angles are radians and survey
areas steradians; project adapters make the conversion from arcminutes
and square degrees explicit.

## Gaussian projection <a name="gaussian_projection"></a>

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
$`C_0^W=\Omega^2/(4\pi)`$. `cap_mask` is an illustrative footprint, not an
approximation to every survey. `covariance_mask_pair_area` and
`covariance_ssc_mask_variance` both use this raw normalization.

## Halo responses and SSC <a name="halo_responses_and_ssc"></a>

`halo_trispectrum` retains 1-halo, 2-halo (1+3), 2-halo (2+2), 3-halo
and 4-halo terms separately. `halo_power_response` returns a dimensional
isotropic $`dP/d\delta_b`$ using the fractional halo response transferred
to the supplied nonlinear power. This halo approximation is not a
calibration of nonlinear tidal responses or a massive-neutrino model.

The component interface accepts that response through
`covariance_ssc_shell_response`, along with common radial windows and the
projected catalog-mean response. `covariance_ssc_mask_variance` supplies
the long-mode Limber background strength, with units of length. Multiplying
it by the radial integration weight and contracting common shell responses
with `covariance_project` gives the SSC matrix. Keep every cross-lens
block. Removing selected cross correlations can make a covariance indefinite.

## Real-space and Fourier assembly <a name="real_space_and_fourier_assembly"></a>

`survey_cluster.selected_windows` distinguishes counts from density
contrasts. A shell contains $`dN_i=\Omega f_K^2 n_i\,d\chi`$ objects,
where $`n_i`$ includes the observed redshift and richness selections.
The normalized window is $`q_i=f_K^2 n_i/\bar n_i`$, with
$`\bar n_i=\int f_K^2 n_i\,d\chi`$ per steradian. Thus a count retains
its absolute abundance, while $`\int q_i\,d\chi=1`$ for clustering.
The fixed-selection response uses $`n_i b_i`$, where $`b_i`$ is the
selected halo bias. See [To et al., Sec. 4.1](https://arxiv.org/abs/2008.10757).

`all_pairs_spectra` combines the ordinary galaxy/shear fields with all
cluster categories. Cluster clustering and cluster–galaxy spectra use
biased nonlinear matter power; cluster lensing also includes the selected
halo's own mass profile. Noise remains separate. The field order is
galaxies, clusters, then sources; cluster categories run through richness
inside each observed redshift bin. `observable_layout` supplies the
measured rows and the positions of counts in the joint vector. Cross-bin
spectra needed by Gaussian pairings remain available even when they are
absent from that measured row list. These helpers prepare inputs; they
do not by themselves compute a complete cluster covariance.

`forecast_cluster.compute_forecast` combines these inputs into the full
angular matrix. Its cNG approximation multiplies the matter trispectrum
by one linear bias per density leg; count cross terms contain SSC only.
Selected-cluster one-halo cNG corrections and non-SSC count–spectrum
terms remain outside this forecast. Every output records those omissions.
For an executable example and its physical choices, use the
[DES cluster guide](../../../../../projects/des_cluster/covariance/README.md#joint).
The result includes count means and positions, all six two-point families,
separate components, and the optional Y transformation on every cross block.
`valid_indices` excludes only its defined last-bin null modes. Numerical
and Fisher convergence, physical scale cuts and the omitted terms still
need assessment before inference.

For cluster counts, `count_statistics` integrates supplied selected
abundances and their long-mode responses. It returns count means,
Poisson noise, count SSC and optional count–two-point SSC separately.
`count_matter_cross` adds the non-SSC cross correlation with projected
matter spectra, returning separate one- and two-halo terms. It receives
the selected moments from `interface.covariance_cluster_moments`, along
with the same radial selection, distances, matter windows and spin
conventions. Its two-halo term contains the full-population moment
$`I_{11}`$, not another selected cluster moment. See
[Schaan, Takada & Spergel, Eq. 35](https://arxiv.org/abs/1406.3330).

This cross-spectrum helper covers matter/shear fields and a constant
linear-bias galaxy approximation. It does not supply discrete cluster
legs or shared-object noise terms. A full cluster joint matrix also
needs cluster SSC/cNG, count–cluster-spectrum terms, consistent catalog
normalizations and the project's estimator transforms.

Cluster lensing can be expressed as the localized statistic
$`Y(R)=\Sigma(R)-\Sigma(R_{\max})`$. The angular transformation removes
the dependence on mass interior to the measured radius. If a mean vector
changes as $`y=A x`$, its covariance changes as $`C_y=A C_x A^{\mathsf T}`$.
Here $`A`$ acts on cluster-lensing angular rows and leaves the other
measurements unchanged. Consequently a count–lensing block receives one
transformation, and a lensing–lensing block receives two. See
[Park, Rozo & Krause, Eqs. 9–12](https://arxiv.org/abs/2004.07504) and
[the DES covariance model, Sec. II.4](https://arxiv.org/abs/2503.13631).

`transform_cluster.py` receives the same angular operator used by the
project's mean calculation. Supply every unmasked angular bin: the
derivative stencil can read neighboring bins outside the final scale
selection. The last output bin is exactly zero because it subtracts
$`\Sigma(R_{\max})`$ from itself. That known null mode remains in the
returned matrix. Apply the likelihood's selection afterwards; no
eigenvalue correction is part of the transformation.

`survey.realspace_covariance` and `survey.fourier_covariance` receive an initialized project interface,
resolved integration settings, the observable row map and catalog noise.
It returns separate Gaussian, SSC, cNG and total matrices, the projected
mean signals and elapsed times by calculation stage. CAMB setup, output
writing and eigenvalue diagnostics remain outside this function.
`survey.observable_rows` puts angular bins inside each tomographic row,
with xi+, xi-, galaxy--shear and galaxy clustering in that order.
Fourier rows omit xi-: one E-mode spectrum supplies both real-space shear
correlations. Integer band endpoints are inclusive, and each multipole
receives weight proportional to its mode count, $`2\ell+1`$. Refinement
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
function documentation. The notebook's full G+SSC+cNG accuracy boost is
not a certified inference setting. Complete matrices must pass positivity
checks and resolution/Fisher comparisons before use for inference.

## One accuracy boost <a name="one_accuracy_boost"></a>

Use `covariance_accuracy(accuracy_boost=1)`, or set `accuracy_boost` in the
project's configuration function. Supported boosts are 1, 2, 4 and 8.
Each step doubles signal and mask multipole cutoffs, radial and angular
quadrature nodes, and window-grid intervals. The resolved dictionary is
available for inspection and saving; ordinary notebook users need only
change the boost. This control belongs to covariance and leaves CAMB and
data-vector accuracy settings unchanged.

Boost 1 is a teaching resolution. For example, boost 8 resolves:

| Numerical quantity | Boost 8 |
| --- | --- |
| Real-space signal maximum multipole | 80,000 |
| Mask maximum multipole | 32,768 |
| Radial nodes per panel | 512 |
| Angular nodes per bin | 1,024 |
| Lensing-window nodes | 32,769 |

The same boost refines halo mass and angular integration and reduces the
response finite-difference step. These are resolutions, not survey-accuracy
labels. Check parameter errors and Fisher FoM for the intended physical model.

## Interpolation and numerical checks <a name="interpolation_and_numerical_checks"></a>

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
cosmologies. The data-vector $`|\Delta\chi^2|<0.2`$ rule is not a covariance
convergence criterion.

`covariance_modes` tests positivity using the correlation matrix: each
observable is divided by its own standard deviation. This invertible
change of units preserves the signs of the covariance modes. It is useful
when shear, clustering and counts have very different variances: rounding
in a raw eigensolver can overwhelm the smallest eigenvalues. The function
also returns the raw eigenvalues for inspection. Rescaling neither removes
negative modes nor adds variance to make a matrix pass.

The notebook computes full G+SSC+cNG forecasts in both spaces. All-pairs
non-Limber corrections and production FoM convergence remain separate work.
