# Roman covariance: requested survey reference

Owner request, 2026-10-03: use
[Eifler et al., arXiv:2004.05271](https://arxiv.org/html/2004.05271v1)
as background for a covariance of the existing **roman_real** project,
not a reproduction of the paper's ten-bin analysis (owner clarification).
Section 2.1, Table 2 and Section 3 specify the
imaging 3x2pt case. Section 2.2 points to Krause & Eifler (2017), Appendix
A, for covariance equations. The paper's broader cluster, spectroscopic
and supernova forecast is not part of this first 3x2pt generator.

| Input | Paper value |
|---|---|
| Area | 2000 deg^2 |
| Total lens / source density | 66 / 51 arcmin^-2 |
| Lens / source tomography | 10 / 10 equal-count bins |
| Per-bin lens / source density | 6.6 / 5.1 arcmin^-2 |
| Shape dispersion | 0.26 |
| Lens redshift minimum | 0.25 |
| Photo-z scatter alternatives | 0.01(1+z), 0.05(1+z) |
| Omega_m, Omega_b, h | 0.3156, 0.0492, 0.6727 |
| sigma_8, n_s, w_0, w_a | 0.831, 0.9645, -1, 0 |
| Galaxy bias | 1.3 + 0.1 i, one per lens bin |
| Fourier binning before cuts | 25 logarithmic bins, 30 to 15000 |
| Shear cutoff | ell_max = 4000 |
| Lens-probe cutoff | R_min = 21 Mpc/h |

The paper uses Limber/flat-sky spectra and omits IA and baryonic nuisance
models in this analysis. Full-sky real-space transforms and non-Limber
galaxy spectra are explicit extensions. State which configuration a
generated covariance represents rather than calling every version a
reproduction of the paper.

## Mapping to the installed examples

The installed `roman_real/data/example1.dataset` has eight lens and eight
source bins from one shared n(z) file, 15 theta bins over 2.5--250 arcmin,
and the example's three excluded lens/source pairs. Its unmasked layout
has 2115 entries. `roman_fourier` also has eight bins per sample, but its
n(z), exclusions and angular sampling differ from `roman_real`.

The target is roman_real: eight lens/source bins and its existing
redshift distributions, angular bins, mask and pair exclusions. The eight
n(z) columns each integrate to 0.125. The existing port study, PLAN.md
Section 10 item 6, records working defaults of 2415 deg^2, total lens and
source densities each 41.3 arcmin^-2 (5.1625 per bin), and shape dispersion
0.30 per component. It distinguishes these from the older paper's survey
and from the incompletely established provenance of the shipped files.
Preserve these explicit working defaults unless the owner changes them;
the paper is background, not authorization to replace them. Do not
reconvolve the existing n(z) or import the paper's Fourier scale cuts.

Local candidate parent histograms exist in
`test/CosmoCov_Fourier/zdistris/zdistri_WFIRST_LSST_*_fine_bin_norm`.
They contain four columns (bin edges, midpoint, distribution), not ten
tomographic distributions. Their association with this paper has not been
verified; do not silently certify or adopt them as its exact input data.

## Implementation boundary and acceptance

Every new cross-bin/non-Limber C routine belongs in
`cosmolike/covariances/`, with filename suffix `_cov.c`. Do not add these
pairs, caches or numerical choices to the ordinary data-vector files.
Project wrappers may call that isolated implementation.

Keep G, SSC and cNG separate, with an explicit row map and input manifest.
New output must coexist with shipped covariances until validated.
Use the [FoM and parameter-error criteria](covariance_accuracy.md),
not a universal 1e-6 relative error on every small SSC entry. Validate the
noise convention, actual pair selection, survey footprint, output layout,
positive definiteness and refinement on the chosen Roman configuration.
This record specifies the new target; it is not a generated covariance.

## Runtime before expensive convergence runs

The owner requested a subset estimate instead of a potentially hours-long
full run. The [small-component timing record](covariance_roman_timing.md)
separates shared three-dimensional tables from tomographic projection.
After that estimate, use a deliberately refined configuration as a
candidate numerical reference, verify another refinement, and assess the
cheaper configurations by the Fisher/FoM protocol. A high boost alone is
not evidence of convergence or correct physical modeling.
