# DES cluster covariance: original code and adopted approximations

Source audit, 2026-10-04. Distinguish a general selected-halo covariance
from the covariance approximations adopted for a particular DES analysis.
An omitted physical term is not automatically a missing DES requirement.
This audit does not claim that a local checkout reproduces the exact
unpublished generation configuration of a supplied DES covariance.

## Sources

- Original CosmoLike core checkout: `1b861b1fbe73a2d62dfce3acf29970cff20f5e30`.
- Lighthouse checkout: `f55bb004178ed907314583e0265d7c905a00a3db`.
- [To et al., DES Y1 validation, Section 4.2](https://arxiv.org/html/2008.10757).
- [To et al., DES Y6 modeling, Section II.4 and Appendix F](https://arxiv.org/html/2503.13631v1).

## Selected-cluster one-halo cNG

The original core's `theory/covariances_cluster.c` defines
`tri_1h_cmcm` and `tri_1h_mmcm`. The active `project_tri_cgl_cgl` uses
the selected one-halo cmcm term plus biased multihalo matter terms.
However, its shear/galaxy cross routines comment out the selected mmcm
alternative and use biased total matter trispectra. Merely finding a
helper in the original core therefore does not prove it was used in every
block, or in the later DES analysis.

Lighthouse's `cpp/cov_clusters_fullsky.c::inner_project_tri_cov_cs_cs_tomo`
and its other cluster two-point integrands use `tri_matter_cov`, with
tracer biases carried by the windows. This is the biased-matter cNG
approximation used in the current forecast. The separate selected-cluster
one-halo correction would extend that inspected DES implementation.
The Y6 paper does not enumerate these individual trispectrum choices;
it describes the covariance construction and its relationship to Y1.

## Count--spectrum cross covariance

Original `project_cov_cgl_N`, `project_cov_cl_N`, `project_cov_ggl_N`
and `project_cov_shear_N` contain background-density responses times
`survey_variance`: SSC. The inspected Lighthouse `project_cov_cg_N`,
`project_cov_cs_N`, `project_cov_gg_N`, `project_cov_gs_N`,
`project_cov_ss_N` and `project_cov_cc_N` also contain SSC-only crosses.
Non-SSC count--spectrum terms are a broader physical extension, not a
term found in those legacy paths. The Y1 paper gives the cross-covariance
transformation and references the general framework; it does not establish
that every possible term of that framework was enabled.

## Non-Limber spectra

This is an explicit DES Y6 requirement: Appendix F states that non-Limber
calculations are used for covariance between different tomographic bins.
The inspected Lighthouse Gaussian cluster-lensing covariance calls
`C_clusterxclusterclustering_mix_interp`, whose implementation builds
non-Limber-to-Limber mixed density spectra. This does not establish that
every shear cross spectrum uses non-Limber. The current all-Limber joint
forecast is consequently missing a documented DES covariance ingredient.

## Higher responses and selection effects

The inspected covariance integrands use a scalar density response and
linear tracer-bias factors. They do not implement a general tidal SSC
response or an explicit derivative of selection probability with respect
to a long environmental perturbation. Original `delP_SSC` also documents
an omitted two-halo halo-sample-variance contribution involving b2.

DES nevertheless models selection bias, and Lighthouse's windows carry
its effective `weighted_bias`. This must not be described as DES omitting
selection effects altogether. Likewise, the Y6 paper's nonlinear-bias
tests do not by themselves establish a nonlinear-bias covariance response.
The current fixed-selection, zero-IA/magnification/RSD forecast is not a
reproduction of every DES model setting. Keep adoption claims tied to
the paper's stated covariance choices and the inspected active routines.
