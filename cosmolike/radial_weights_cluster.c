// ============================================================================
// radial_weights_cluster.c
//
// Radial weights of the cluster sample in the Limber integrals of the
// cluster two-point functions (cosmo2D_cluster.c), with the conventions
// of radial_weights.c: W_cluster is the W_gal analog (a density per unit
// comoving distance chi, in c/H0 units), W_mag_cluster the W_mag analog.
// Neither applies a bias or an amplitude: the callers multiply W_cluster
// by the richness-weighted bias b_nl(a) (eq 21 of 2503.13631) and
// W_mag_cluster by the magnification coefficient cluster.magnification
// (C_c, eq 28).
// ============================================================================

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "log.c/src/log.h"

#include "radial_weights_cluster.h"
#include "redshift_spline_cluster.h"
#include "structs.h"
#include "structs_cluster.h"



// ============================================================================
// [SECTION] CLUSTER RADIAL WEIGHTS
// ============================================================================


// ---------------------------------------------------------------------------
// Radial density kernel of the clusters of redshift bin ni (richness bin
// nl for the abundance kernel):
//
//   W_cluster(a) = n(z(a)) dz/dchi = nz_cluster(1/a - 1, ni, nl) H(a)/H0
//
// (chi in c/H0 units gives dz/dchi = H/H0). nz_cluster integrates to 1
// over z, so W_cluster integrates to 1 over chi. Zero outside the support
// of the bin's selection kernel.
//
// Parameters:
//   a       - scale factor, 0 < a < 1
//   ni      - cluster redshift bin (0 .. cluster.zdist_nbin - 1)
//   nl      - richness bin (unused by the volume kernel)
//   hoverh0 - H(a)/H0, supplied by the caller (cosmo_nodes holds it)
//
// Returns:
//   cluster density per unit comoving distance
// ---------------------------------------------------------------------------
double W_cluster(const double a, const int ni, const int nl,
  const double hoverh0)
{
  if (!(a > 0) || !(a < 1)) {
    log_fatal("a>0 and a<1 not true (a = %e)", a);
    exit(1);
  }
  if (ni < 0 || ni > cluster.zdist_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  }

  const double z = 1.0/a - 1.0;
  return nz_cluster(z, ni, nl)*hoverh0;
}


// ---------------------------------------------------------------------------
// Magnification kernel of the clusters of redshift bin ni (richness bin
// nl for the abundance kernel): the convergence kernel with the lensing
// efficiency of the cluster n(z),
//
//   W_mag_cluster(a) = 1.5 Omega_m f_K(chi(a))/a g_cluster(a, ni, nl)
//
// (distances in c/H0, so the (H0/c)^2 prefactor is 1). g_cluster is
// nonzero over the whole foreground of the bin, so this weight is valid
// (and nonzero) for every a in front of the bin's far edge, up to just
// below a = 1; it falls to 0 there only through f_K -> 0. The callers
// multiply it by cluster.magnification (C_c = -2 in eq 28).
//
// Parameters:
//   a  - scale factor, 0 < a < 1
//   fK - comoving angular diameter distance f_K(chi(a)), c/H0 units
//   ni - cluster redshift bin (0 .. cluster.zdist_nbin - 1)
//   nl - richness bin (unused by the volume kernel)
//
// Returns:
//   magnification kernel at a
// ---------------------------------------------------------------------------
double W_mag_cluster(const double a, const double fK, const int ni,
  const int nl)
{
  if (!(a > 0) || !(a < 1)) {
    log_fatal("a>0 and a<1 not true (a = %e)", a);
    exit(1);
  }
  if (ni < 0 || ni > cluster.zdist_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  }

  return (1.5*cosmology.Omega_m*fK/a)*g_cluster(a, ni, nl);
}
