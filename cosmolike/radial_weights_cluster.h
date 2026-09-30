#ifndef __COSMOLIKE_RADIAL_WEIGHTS_CLUSTER_H
#define __COSMOLIKE_RADIAL_WEIGHTS_CLUSTER_H
#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// [SECTION] CLUSTER RADIAL WEIGHTS (the radial_weights.c conventions)
// ============================================================================

// Cluster density kernel per unit comoving distance, the W_gal analog:
//   W_cluster = nz_cluster(z(a), ni, nl) * H(a)/H0
// The richness-weighted bias b_nl(a) multiplies it in the 2-halo terms.
double W_cluster(const double a, const int ni, const int nl,
  const double hoverh0);

// Cluster magnification kernel, the W_mag analog:
//   W_mag_cluster = 1.5 Omega_m f_K(chi)/a * g_cluster(a, ni, nl)
// It enters with the coefficient cluster.magnification (C_c = -2).
double W_mag_cluster(const double a, const double fK, const int ni,
  const int nl);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
