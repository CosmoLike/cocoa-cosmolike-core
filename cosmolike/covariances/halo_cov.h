#ifndef COSMOLIKE_HALO_COV_H
#define COSMOLIKE_HALO_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// Halo moments of the cold-dark-matter-plus-baryon field. Both the halo
// abundance and M/rho normalization use rho_cb. The public concentration
// and M200m NFW profile retain the core's mass definition. With massive
// neutrinos these are cb moments, not a complete total-matter trispectrum.
// k[a][node] uses (c/H0)^-1. a is inside [limits.a_min,1), mass-panel
// edges are increasing ln(M/[M_sun/h]) inside the core sigma-table range.
// All arrays are supplied by the caller; writable arrays do not overlap.
// moments may be NULL to request only I11, without pair integrations.
void halo_moments_cov(
    const int na,                  // number of scale factors
    const double* a,              // [na] scale factors
    const int nk,                  // wavenumbers per scale factor
    const double* const* k,       // [na][nk], finite k >= 0
    const int npanel,              // number of logarithmic mass intervals
    const double* lnm_edges,      // [npanel+1], natural logarithms of mass
    const int nquad,               // tabulated GL nodes per mass interval
    double* const* i11,            // [na][nk], with M_min completion
    double** const* moments        // [5][na][nk*(nk+1)/2], see below
  );

// Pair rows use i-major upper-triangle order. Moment roles:
// 0: I02(K,Q), 1: I12(K,Q), 2: I13(K,Q,Q), 3: I13(K,K,Q),
// 4: I04(K,K,Q,Q). Their units are L^3,L^3,L^6,L^6,L^9, L=c/H0.
// I11 is dimensionless. Only I11 receives the unresolved-low-mass term.

#ifdef __cplusplus
}
#endif
#endif
