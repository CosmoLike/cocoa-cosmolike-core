#ifndef COSMOLIKE_NONLIMBER_CLUSTER_COV_H
#define COSMOLIKE_NONLIMBER_CLUSTER_COV_H

#include "spectra_cov.h"

#ifdef __cplusplus
extern "C" {
#endif

// The non-Limber correction of the joint cluster forecast: for every pair
// of the fields lenses, selected clusters (cluster redshift bin major,
// richness bin minor) and sources, at l = 2..lmax,
//   correction[p][l-2] = exact[p] - matched[p]
// of nonlimber_pairs_cov (nonlimber_cov.h), with the clusters as
// density fields of window q_c b_c and, with include_rsd, the Kaiser term
// on every lens and cluster. Rows of two sources are zero: shear-shear
// stays Limber, as in apply_nonlimber_cov. The caller adds the rows to
// its nonlinear Limber spectra of the same fields and windows
// (cosmolike_notebook_utils.covariance.survey_cluster.all_pairs_spectra).
//
// radial is the galaxy/source quadrature snapshot of radial_inputs_cov
// without IA, built from a_edges with amin = a_edges[0] and nwindow. The
// cluster model must be initialized without magnification; the
// cosmology must be flat with massless neutrinos. Call outside OpenMP
// regions.
void cluster_nonlimber_cov(
    const struct radial_cov* radial, // galaxy/source quadrature snapshot
    const double amin,               // far boundary, a_edges[0] of radial
    const int nwindow,               // efficiency samples used for radial
    const int include_rsd,           // 1 adds the Kaiser term, 0 omits it
    const int lmax,                  // last corrected integer multipole
    const int nchi,                  // log-distance samples, 2^n+1 >= 65
    const double chi_min,            // positive near distance in c/H0
    double* const* correction        // [joint triangular pair][lmax-1]
  );

#ifdef __cplusplus
}
#endif
#endif
