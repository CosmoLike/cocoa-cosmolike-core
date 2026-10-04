#ifndef COSMOLIKE_COUNTS_CLUSTER_COV_H
#define COSMOLIKE_COUNTS_CLUSTER_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// Convert selected comoving abundances into counts per radial distance.
// density and density_response already include the redshift selection.
// They have units L^-3; distance uses L; both outputs have units L^-1.
// No dchi integration weight or catalog-mean normalization is included.
// The output rows are caller-owned and disjoint from all input rows.
void counts_shell_cluster_cov(
    const int ncount,                       // observed count bins
    const int nnode,                        // common radial nodes
    const double area_sr,                  // angular survey area
    const double* distance,                // f_K at each node
    const double* const* density,          // selected abundance
    const double* const* density_response, // d(abundance)/d(delta_b)
    double* const* shell,                   // dN/dchi
    double* const* response                 // d(dN/dchi)/d(delta_b)
  );

#ifdef __cplusplus
}
#endif
#endif
