#ifndef COSMOLIKE_SPECTRA_CLUSTER_COV_H
#define COSMOLIKE_SPECTRA_CLUSTER_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// Project every cluster-base and cluster-cluster pair, including pairs
// absent from the measured vector. This supplied-table boundary reads no
// global state. The detailed physical model is in spectra_cluster_cov.c.
// Distances use one unit L; windows use L^-1 and power/profile use L^3.
// Inputs and writable output must occupy disjoint memory.
void limber_cluster_cov(
    const int nell,                   // multipole count
    const double* ell,                // [nell], finite ell >= 2
    const int nnode,                  // common radial-node count
    const double* distance,           // [nnode], positive f_K
    const double* dchi,               // [nnode], positive radial weights
    const int nbase,                  // galaxy plus source fields
    const int nlens,                  // leading galaxy fields in base
    const double* const* base,        // [nbase][nnode], biased galaxy/lens
    const int ncluster,               // observed cluster categories
    const double* const* window,      // [ncluster][nnode], normalized q_c
    const double* const* bias,        // [ncluster][nnode], cluster bias
    const double* const* power,       // [nell][nnode], nonlinear matter P
    const double* const* const* p1h,  // [nrichness][nell][nnode], P_cm^1h
    const int* richness,              // [ncluster], profile-row indices
    double* const* spectra            // [npair][nell], see ordering below
  );

// First ncluster*nbase rows: cluster-major (cluster,base) pairs.
// Remaining ncluster*(ncluster+1)/2 rows: cluster upper triangle,
// (0,0),(0,1),...,(1,1),... . No catalog noise is included.

#ifdef __cplusplus
}
#endif
#endif
