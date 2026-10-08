#ifndef COSMOLIKE_HALO_CLUSTER_COV_H
#define COSMOLIKE_HALO_CLUSTER_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// Sample the initialized lognormal richness model on supplied mass nodes:
//   weight  = dlnM (dn/dlnM) P(richness bin|M,z), one probability factor,
//   bias    = linear halo bias b_h(M,a),
//   profile = (M/rho_m) u_NFW(k|M).
// The caller validates sizes, domains and initialized massless/NFW state,
// including the Tinker et al. (2010) mass function (HMF_TINKER_2010)
// that the fixed-alpha mode assumes.
// k[state][mode] uses inverse c/H0; lnm is ln(M/[Msun/h]); dlnm > 0.
// Outputs own disjoint storage. All calls enter serially, outside OpenMP.
// No radial/photo-z selection or catalog normalization is included.
void halo_samples_cluster_cov(
    const int na,                      // independent scale factors
    const double* a,                   // scale factors [na]
    const int nk,                      // wavenumbers per scale factor
    const double* const* k,            // wavenumbers [na][nk]
    const int nmass,                   // supplied mass nodes
    const double* lnm,                 // logarithmic masses [nmass]
    const double* dlnm,                // quadrature weights [nmass]
    double** const* weight,            // selected dn [na][richness][mass]
    double* const* bias,               // linear bias [na][mass]
    double** const* profile            // (M/rho_m)u [na][nk][mass]
  );

#ifdef __cplusplus
}
#endif
#endif
