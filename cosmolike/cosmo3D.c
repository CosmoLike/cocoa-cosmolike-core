#include <assert.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_sf.h>
#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "log.c/src/log.h"

#include "basics.h"
#include "baryons.h"
#include "cosmo3D.h"
#include "halo.h"
#include "structs.h"

// ---------------------------------------------------------------------------
// cosmo3D.c: background, growth and P(k) lookup functions over CAMB-fed
// tables.
//
// Data flow:
//
//   CAMB (via the C-interface setters)
//     set_distances                 -> cosmology.chi  (z, chi(z) in Mpc/h)
//     set_growth                    -> cosmology.G    (z, G(z); D = G*a)
//     set_linear_power_spectrum     -> cosmology.lnPL (ln P on log10k x z)
//     set_linear_power_spectrum_cb  -> cosmology.lnPL_cb (ln P_cb, cold
//                                      dark matter + baryons, on the
//                                      lnPL grid; optional)
//     set_non_linear_power_spectrum -> cosmology.lnP  (same layout)
//   -> the lookup functions below (chi_all, norm_growfac*, f_growth, p_lin,
//      p_lin_cb, p_nonlin, ...) interpolate those tables
//   -> radial weights and Limber integrands (cosmo2D.c) consume them.
//
// Unit conventions, once for the whole file (cosmology.coverH0 = c/H0
// in Mpc/h = 2997.92458):
//
//   distances = c/H0 units        (table Mpc/h, divided by coverH0)
//   k         = (c/H0)^-1 units   (k[h/Mpc]*coverH0)
//   P(k)      = (c/H0)^3 units    (table (Mpc/h)^3, divided by coverH0^3)
//
// COSMO3D_ASSUME_PIECEWISE_UNIFORM contract: each setter lays its z
// grid out as a few uniform segments and records them in the *_z_seg_*
// metadata (lnPL/lnP additionally use one uniform segment in log10 k).
// With the macro defined, every bracket lookup except a_chi's (the chi
// column is not uniform) becomes a direct index computed from that
// metadata instead of a binary search. The metadata is trusted, never
// checked: a table filled without it, or with a grid that is not
// piecewise-uniform, makes the lookups land in wrong brackets and
// return silently wrong interpolants. Without the macro every lookup
// binary-searches and needs no metadata.
// ---------------------------------------------------------------------------

#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// aux funtions
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Direct-index lookup for a piecewise-uniform 1D grid.
//
// Given a query value q and segment metadata (start[], len[], xmin[],
// inv_dx[]) for nseg piecewise-uniform segments of an underlying grid of
// total length n_total, returns the bracket index j such that
// grid[j] <= q < grid[j+1].
//
// The result is clamped to [0, n_total - 2] so the caller can safely
// access grid[j] and grid[j+1] without bounds checks.
//
// Parameters:
//   q       - query value on the grid axis
//   nseg    - number of piecewise-uniform segments
//   start   - first grid index of each segment (length nseg)
//   len     - number of grid points of each segment (length nseg; carried
//             with the other segment metadata, not read by the lookup)
//   xmin    - first grid value of each segment (length nseg)
//   inv_dx  - inverse grid spacing of each segment (length nseg)
//   n_total - total number of grid points
//
// Returns:
//   bracket index j with grid[j] <= q < grid[j+1], clamped to [0, n_total-2]
// ---------------------------------------------------------------------------
static inline int piecewise_index(double q,
                                  int nseg,
                                  const int *start, 
                                  const int *len,
                                  const double *xmin, 
                                  const double *inv_dx,
                                  int n_total)
{
  // Pick the segment: q is in segment s if q is in [xmin[s], xmin[s+1]).
  // Linear scan is fine for nseg <= ~10 (branch-predicted, all in L1).
  int s = 0;
  while (s < nseg - 1 && q >= xmin[s+1]) s++;

  // Direct index within segment s.
  int j = start[s] + (int)((q - xmin[s]) * inv_dx[s]);

  // Clamp to valid bilinear bracket range.
  if (j < 0)             j = 0;
  if (j > n_total - 2)   j = n_total - 2;
  return j;
}
#endif
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Background
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Comoving distance chi(a) in c/H0 units. Thin wrapper returning the chi
// field of chi_all (see chi_all for the table, units and interpolation);
// callers that also need dchi/da should call chi_all once instead.
//
// Parameters:
//   a - scale factor (a = 1/(1+z))
//
// Returns:
//   chi(a) in c/H0 units (cosmology.coverH0 = c/H0 in Mpc/h)
// ---------------------------------------------------------------------------
double chi(const double a)
{
  struct chis r = chi_all(a);
  return r.chi;
}

// ---------------------------------------------------------------------------
// Distance derivative |dchi/da| = (1/a^2) dchi/dz in c/H0 units. Thin
// wrapper returning the dchida field of chi_all; positive by the sign
// convention documented there (the minus sign of dz/da is dropped).
//
// Parameters:
//   a - scale factor
//
// Returns:
//   (1/a^2) dchi/dz at a, in c/H0 units
// ---------------------------------------------------------------------------
double dchi_da(const double a)
{
  struct chis r = chi_all(a);
  return r.dchida;
}

// ---------------------------------------------------------------------------
// Comoving distance chi(a) and its derivative in one table lookup.
//
// Reads the CAMB-fed table cosmology.chi (chi[0][j] = z_j, chi[1][j] =
// chi(z_j) in Mpc/h, loaded by set_distances in the C interface). With
// z = 1/a - 1 and dy = (z - z_j)/(z_{j+1} - z_j) in the bracket [j, j+1]:
//
//   chi(z)  = chi_j + dy*(chi_{j+1} - chi_j)
//   dchi/dz = down + dy*(up - down)
//     up   = (chi_{j+2} - chi_j)/(z_{j+2} - z_j)         (centered at j+1)
//     down = (chi_{j+1} - chi_{j-1})/(z_{j+1} - z_{j-1}) (centered at j;
//            one-sided slope over [j, j+1] when j = 0)
//
// Unit conventions: the table stores chi in Mpc/h; the result divides by
// cosmology.coverH0 = c/H0 in Mpc/h (2997.92458), so cosmolike distances
// are in c/H0 units. The derivative is returned per unit a,
//
//   result.dchida = (dchi/dz)/coverH0/a^2,
//
// i.e. |dchi/da|: only the magnitude |dz/da| = 1/a^2 of the Jacobian is
// applied, so dchida is positive although chi decreases with a.
//
// The fast variant replaces the bracket binary search with a direct-index
// lookup on the piecewise-uniform z grid (cosmology.chi_z_* metadata from
// set_distances) and clamps j so the j+2 read of the "up" slope stays in
// bounds; the fallback variant keeps the binary search.
//
// Cache invalidation:
// no static state. The cosmology.chi table and its
// grid metadata are replaced by set_distances, which also bumps
// cosmology.random so downstream caches rebuild.
//
// Parameters:
//   a - scale factor (a = 1/(1+z))
//
// Returns:
//   struct chis { chi = chi(a) in c/H0 units, dchida = (1/a^2) dchi/dz }
// ---------------------------------------------------------------------------
#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
// ---------------------------------------------------------------------------
// Fast variant: direct-index bracket lookup on the piecewise-uniform z
// grid (cosmology.chi_z_* metadata). Full contract: shared header above.
//
// Parameters:
//   a - scale factor (a = 1/(1+z))
//
// Returns:
//   struct chis { chi = chi(a) in c/H0 units, dchida = (1/a^2) dchi/dz }
// ---------------------------------------------------------------------------
struct chis chi_all(const double a)
{
  const double z = 1.0/a - 1.0;

  // Direct-index lookup on the piecewise-uniform z grid (cosmology.chi_z_*
  // metadata, populated by whichever function fills cosmology.chi).
  const int j = piecewise_index(z, 
                                cosmology.chi_z_nseg,
                                cosmology.chi_z_seg_start, 
                                cosmology.chi_z_seg_len,
                                cosmology.chi_z_seg_xmin,  
                                cosmology.chi_z_seg_inv_dx,
                                cosmology.chi_nz);

  // Pre-load the grid points and chi values used by both the chi(z)
  // interpolation and the dchi/dz finite-difference derivative.
  // The j+2 access requires j <= chi_nz - 3; piecewise_index already
  // clamps to chi_nz - 2, so we additionally clamp here for the j+2 read.
  const int jc = (j > cosmology.chi_nz - 3) ? cosmology.chi_nz - 3 : j;

  const double zjm1 = (jc > 0) ? cosmology.chi[0][jc-1] : 0.0;  // unused if jc==0
  const double zj   = cosmology.chi[0][jc  ];
  const double zj1  = cosmology.chi[0][jc+1];
  const double zj2  = cosmology.chi[0][jc+2];

  const double cjm1 = (jc > 0) ? cosmology.chi[1][jc-1] : 0.0;  // unused if jc==0
  const double cj   = cosmology.chi[1][jc  ];
  const double cj1  = cosmology.chi[1][jc+1];
  const double cj2  = cosmology.chi[1][jc+2];

  const double dy = (z - zj) / (zj1 - zj);

  // chi(z) by linear interpolation between j and j+1.
  const double chi_interp = cj + dy * (cj1 - cj);

  // dchi/dz by linear interpolation of two centered finite differences:
  //   "up"   = slope between (j, j+2)
  //   "down" = slope between (j-1, j+1)   (or (j, j+1) at the boundary j=0)
  const double up = (cj2 - cj) / (zj2 - zj);
  const double down = (jc > 0)
      ? (cj1 - cjm1) / (zj1 - zjm1)
      : (cj1 - cj  ) / (zj1 - zj  );
  const double dchidz = down + dy * (up - down);

  // convert Mpc/h -> c/H0 units (divide by coverH0 = c/H0 in Mpc/h =
  // 2997.92458), and d(chi)/dz -> d(chi)/da via |dz/da| = 1/a^2
  struct chis result;
  result.chi    = chi_interp / cosmology.coverH0;
  result.dchida = dchidz / cosmology.coverH0 / (a * a);
  return result;
}
#else
// ---------------------------------------------------------------------------
// Fallback variant: binary-search bracket lookup on the z grid. Full
// contract: shared header above.
//
// Parameters:
//   a - scale factor (a = 1/(1+z))
//
// Returns:
//   struct chis { chi = chi(a) in c/H0 units, dchida = (1/a^2) dchi/dz }
// ---------------------------------------------------------------------------
struct chis chi_all(const double a)
{
  double out[2];
  const double z = 1.0/a - 1.0;

  // bracket the query z by binary search on the z column
  int j = 0;
  {
    size_t ilo = 0;
    size_t ihi = cosmology.chi_nz - 1;
    while (ihi > ilo + 1)
    {
      size_t ll = (ihi + ilo)/2;
      if (cosmology.chi[0][ll] > z)
        ihi = ll;
      else
        ilo = ll;
    }
    j = ilo;
  }
  // the "up" slope below reads j+2; clamp as the piecewise variant does
  if (j > cosmology.chi_nz - 3) {
    j = cosmology.chi_nz - 3;
  }

  // chi(z) by linear interpolation between nodes j and j+1
  const double dy = (z                     - cosmology.chi[0][j])/
                    (cosmology.chi[0][j+1] - cosmology.chi[0][j]);
  out[0] = cosmology.chi[1][j] + dy*(cosmology.chi[1][j+1]-cosmology.chi[1][j]);

  // dchi/dz by linear interpolation of the two slopes "up" (over
  // [j, j+2]) and "down" (over [j-1, j+1]; one-sided over [j, j+1]
  // at the j = 0 boundary)
  if (j>0)
  {
    const double up = (cosmology.chi[1][j+2] - cosmology.chi[1][j])/
                      (cosmology.chi[0][j+2] - cosmology.chi[0][j]);
    
    const double down = (cosmology.chi[1][j+1] - cosmology.chi[1][j-1])/
                        (cosmology.chi[0][j+1] - cosmology.chi[0][j-1]);
    out[1] = down + dy*(up-down);
  }
  else 
  {
    const double up = (cosmology.chi[1][j+2] - cosmology.chi[1][j])/
                      (cosmology.chi[0][j+2] - cosmology.chi[0][j]);
    
    const double down = (cosmology.chi[1][j+1] - cosmology.chi[1][j])/
                        (cosmology.chi[0][j+1] - cosmology.chi[0][j]);
    out[1] = down + dy*(up-down);
  }

  // convert Mpc/h -> c/H0 units (divide by coverH0 = c/H0 in Mpc/h =
  // 2997.92458)
  out[1] = (out[1]/cosmology.coverH0);
  // convert from d\chi/dz to d\chi/da
  out[1] = out[1]/(a*a);

  struct chis result;
  result.chi = out[0]/cosmology.coverH0;
  result.dchida = out[1];
  return result;
}
#endif

// ---------------------------------------------------------------------------
// dchi/dz = a^2 * dchi_da(a) in c/H0 units. Equals c/H(z) in units of
// c/H0, i.e. H0/H(z): the inverse of hoverh0.
//
// Parameters:
//   a - scale factor
//
// Returns:
//   dchi/dz at a, in c/H0 units
// ---------------------------------------------------------------------------
double dchi_dz(const double a)
{
  return (a*a)*dchi_da(a);
}

// ---------------------------------------------------------------------------
// Dimensionless Hubble rate H(a)/H0 = 1/(dchi/dz), read off the tabulated
// distances (one chi_all lookup), so the expansion history is consistent
// with the chi table by construction.
//
// Parameters:
//   a - scale factor
//
// Returns:
//   H(a)/H0
// ---------------------------------------------------------------------------
double hoverh0(const double a)
{
  return 1.0/dchi_dz(a);
}

// ---------------------------------------------------------------------------
// H(a)/H0 = 1/(a^2 * dchida) from a dchi/da value the caller already
// holds. Node-precompute loops (see create_cosmo_nodes in cosmo2D.c)
// fetch chi and dchida in one chi_all call and use this to avoid
// hoverh0's second table lookup.
//
// Parameters:
//   a      - scale factor
//   dchida - (1/a^2) dchi/dz at a, as returned by chi_all
//
// Returns:
//   H(a)/H0
// ---------------------------------------------------------------------------
double hoverh0v2(const double a, const double dchida)
{
  return 1.0/((a*a)*dchida);
}

// ---------------------------------------------------------------------------
// Inverse distance lookup a(chi): binary search on the chi column of
// cosmology.chi, then linear inverse interpolation of z in the bracket,
//
//   z = z_j + dy*(z_{j+1} - z_j),  dy = (chi - chi_j)/(chi_{j+1} - chi_j),
//
// and a = 1/(1+z). The input is converted from c/H0 units to the table's
// Mpc/h (io_chi * coverH0). The chi column is not uniformly spaced, so
// this direction keeps the binary search even in the piecewise-uniform
// build.
//
// Cache invalidation:
// no static state; the table is maintained by
// set_distances (see chi_all).
//
// Parameters:
//   io_chi - comoving distance in c/H0 units
//
// Returns:
//   scale factor a with chi(a) = io_chi
// ---------------------------------------------------------------------------
double a_chi(const double io_chi)
{
  // convert c/H0 units -> the table's Mpc/h (multiply by coverH0 =
  // c/H0 in Mpc/h = 2997.92458)
  const double chi = io_chi*cosmology.coverH0;

  int j = 0;
  {
    size_t ilo = 0;
    size_t ihi = cosmology.chi_nz-1;
    while (ihi > ilo + 1)
    {
      size_t ll = (ihi + ilo)/2;
      if (cosmology.chi[1][ll] > chi)
        ihi = ll;
      else
        ilo = ll;
    }
    j = ilo;
  }

  const double dy = (chi                   - cosmology.chi[1][j])/
                    (cosmology.chi[1][j+1] - cosmology.chi[1][j]);
  double z  = cosmology.chi[0][j] + dy*(cosmology.chi[0][j+1]-cosmology.chi[0][j]);
  return 1.0/(1.0+z);
}
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Growth Factor
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Linear growth factor D(a) normalized to D(a=1) = 1: shorthand for
// norm_growfac(a, true) = G(z)*a/G(0) (see norm_growfac).
//
// Parameters:
//   a - scale factor
//
// Returns:
//   D(a) with D(1) = 1
// ---------------------------------------------------------------------------
double growfac(const double a)
{
  return norm_growfac(a, true);
}

// ---------------------------------------------------------------------------
// Linear growth factor D(a) = G(z)*a from the CAMB-fed table cosmology.G
// (G[0][j] = z_j, G[1][j] = G(z_j) with D = G*a, loaded by set_growth in
// the C interface). With z = 1/a - 1 and linear interpolation in the
// bracket [j, j+1]:
//
//   G(z) = G_j + dy*(G_{j+1} - G_j),  dy = (z - z_j)/(z_{j+1} - z_j)
//   D(a) = G(z)*a          (normalize_z0 = false)
//   D(a) = G(z)*a/G(0)     (normalize_z0 = true: then D(a=1) = 1,
//                           since a = 1 at z = 0)
//
// G(0) comes from the same table: the fast variant uses bracket j = 0
// directly (the grid starts at z = 0), the fallback repeats the binary
// search. The query-z bracket is a direct-index lookup on the
// piecewise-uniform z grid (cosmology.G_z_* metadata) in the fast
// variant, a binary search in the fallback.
//
// Cache invalidation:
// no static state. The cosmology.G table and its
// grid metadata are replaced by set_growth, which also bumps
// cosmology.random.
//
// Parameters:
//   a            - scale factor
//   normalize_z0 - true: divide by G(0) so that D(a=1) = 1
//
// Returns:
//   D(a), normalized per normalize_z0
// ---------------------------------------------------------------------------
#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
// ---------------------------------------------------------------------------
// Fast variant: j = 0 bracket for the G(0) normalization and direct-index
// bracket lookup at the query z. Full contract: shared header above.
//
// Parameters:
//   a            - scale factor
//   normalize_z0 - true: divide by G(0) so that D(a=1) = 1
//
// Returns:
//   D(a), normalized per normalize_z0
// ---------------------------------------------------------------------------
double norm_growfac(const double a, const bool normalize_z0)
{
  // ---------------------------------------------------------------
  // First lookup: G(z=0), used as the z=0 normalization. Skipped
  // entirely if normalize_z0 == false (the value is unused).
  // ---------------------------------------------------------------
  double growfact1 = 0.0;
  if (normalize_z0) {
    // z=0 always falls in the first bracket of a grid that starts at 0.
    const int j = 0;
    const double dy = (0.0                 - cosmology.G[0][j]) /
                      (cosmology.G[0][j+1] - cosmology.G[0][j]);
    growfact1 = cosmology.G[1][j] + dy * (cosmology.G[1][j+1] - cosmology.G[1][j]);
  }

  // ---------------------------------------------------------------
  // Second lookup: G at the query redshift. Direct-index lookup on
  // the piecewise-uniform z grid (cosmology.G_z_* metadata).
  // ---------------------------------------------------------------
  const double z = 1.0/a - 1.0;

  const int j = piecewise_index(z, 
                                cosmology.G_z_nseg,
                                cosmology.G_z_seg_start,
                                cosmology.G_z_seg_len,
                                cosmology.G_z_seg_xmin,  
                                cosmology.G_z_seg_inv_dx,
                                cosmology.G_nz);

  const double dy = (z                   - cosmology.G[0][j])/
                    (cosmology.G[0][j+1] - cosmology.G[0][j]);

  const double G = cosmology.G[1][j] + dy*(cosmology.G[1][j+1] - cosmology.G[1][j]);

  return normalize_z0 ? (G*a)/growfact1 : G*a;
}
#else
// ---------------------------------------------------------------------------
// Fallback variant: binary-search bracket lookups (both the G(0)
// normalization and the query z). Full contract: shared header above.
//
// Parameters:
//   a            - scale factor
//   normalize_z0 - true: divide by G(0) so that D(a=1) = 1
//
// Returns:
//   D(a), normalized per normalize_z0
// ---------------------------------------------------------------------------
double norm_growfac(const double a, const bool normalize_z0)
{
  // first lookup: G(0) for the z = 0 normalization (binary search)
  double growfact1;
  {
    const double z = 0.0;
    int j = 0;
    {
      size_t ilo = 0;
      size_t ihi = cosmology.G_nz-1;
      while (ihi>ilo+1) {
        size_t ll = (ihi+ilo)/2;
        if(cosmology.G[0][ll]>z)
          ihi = ll;
        else
          ilo = ll;
      }
      j = ilo;
    }
    const double dy = (z                   - cosmology.G[0][j])/
                      (cosmology.G[0][j+1] - cosmology.G[0][j]);
    
    growfact1 = cosmology.G[1][j] + dy*(cosmology.G[1][j+1] - cosmology.G[1][j]);
  }

  // second lookup: G at the query redshift (binary search)
  const double z = 1.0/a-1.0;

  int j = 0;
  {
    size_t ilo = 0;
    size_t ihi = cosmology.G_nz-1;
    while (ihi>ilo+1)
    {
      size_t ll = (ihi+ilo)/2;
      if(cosmology.G[0][ll]>z)
        ihi = ll;
      else
        ilo = ll;
    }
    j = ilo;
  }

  const double dy = (z                   - cosmology.G[0][j])/
                    (cosmology.G[0][j+1] - cosmology.G[0][j]);

  const double G = cosmology.G[1][j] + dy*(cosmology.G[1][j+1] - cosmology.G[1][j]);

  if(normalize_z0)
    return (G*a)/growfact1; // Growth D = G * a
  else
    return G*a; // Growth D = G * a
}
#endif

// ---------------------------------------------------------------------------
// Logarithmic growth rate f(z) = dlnD/dlna for D = G*a, from the same
// cosmology.G table as norm_growfac (note the argument is z, not a).
// With the bracket slope dG/dz = (G_{j+1} - G_j)/(z_{j+1} - z_j):
//
//   dlnG/dlnz = (dG/dz)*z/G(z)
//   dlnG/dlna = -dlnG/dlnz*(1+z)/z    (from ln a = -ln(1+z))
//   f         = 1 + dlnG/dlna
//
// The normalization G(0) cancels in the logarithmic derivative, so no
// z = 0 lookup is needed. Bracket selection: direct-index lookup in the
// fast variant, binary search in the fallback (see norm_growfac).
//
// Cache invalidation:
// no static state; the table is maintained by
// set_growth.
//
// Parameters:
//   z - redshift
//
// Returns:
//   f(z) = dlnD/dlna
// ---------------------------------------------------------------------------
#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
// ---------------------------------------------------------------------------
// Fast variant: direct-index bracket lookup on the piecewise-uniform z
// grid. Full contract: shared header above.
//
// Parameters:
//   z - redshift
//
// Returns:
//   f(z) = dlnD/dlna
// ---------------------------------------------------------------------------
double f_growth(const double z)
{
  // Direct-index lookup on the piecewise-uniform z grid; metadata
  // populated by whichever function fills cosmology.G.
  const int j = piecewise_index(z, cosmology.G_z_nseg,
                                cosmology.G_z_seg_start, cosmology.G_z_seg_len,
                                cosmology.G_z_seg_xmin,  cosmology.G_z_seg_inv_dx,
                                cosmology.G_nz);

  const double zj  = cosmology.G[0][j];
  const double zj1 = cosmology.G[0][j+1];
  const double Gj  = cosmology.G[1][j];
  const double Gj1 = cosmology.G[1][j+1];

  const double dy       = (z - zj) / (zj1 - zj);
  const double G        = Gj + dy * (Gj1 - Gj);
  const double dlnGdlnz = ((Gj1 - Gj) / (zj1 - zj)) * z / G;
  // z-cancelled form, finite at z = 0 (dlnGdlnz carries a factor z)
  const double dlnGdlnz_slope = ((Gj1 - Gj)/
    (zj1 - zj))*(1+z)/G;
  const double dlnGdlna = (z > 0.0) ? -dlnGdlnz*(1+z)/z
                                    : -dlnGdlnz_slope;

  return 1 + dlnGdlna; // Growth D = G * a
}
#else
// ---------------------------------------------------------------------------
// Fallback variant: binary-search bracket lookup on the z grid. Full
// contract: shared header above.
//
// Parameters:
//   z - redshift
//
// Returns:
//   f(z) = dlnD/dlna
// ---------------------------------------------------------------------------
double f_growth(const double z)
{
  // bracket the query z by binary search on the z column
  int j = 0;
  {
    size_t ilo = 0;
    size_t ihi = cosmology.G_nz-1;
    while (ihi>ilo+1)
    {
      size_t ll = (ihi+ilo)/2;
      if(cosmology.G[0][ll]>z)
        ihi = ll;
      else
        ilo = ll;
    }
    j = ilo;
  }

  const double dy = (z                   - cosmology.G[0][j])/
                    (cosmology.G[0][j+1] - cosmology.G[0][j]);
                    
  const double G = cosmology.G[1][j] + dy*(cosmology.G[1][j+1] - cosmology.G[1][j]);

  const double dlnGdlnz = ((cosmology.G[1][j+1] - cosmology.G[1][j])/
                           (cosmology.G[0][j+1] - cosmology.G[0][j]))*z/G;
  // z-cancelled form, finite at z = 0 (dlnGdlnz carries a factor z)
  const double dlnGdlnz_slope = ((cosmology.G[1][j+1] - cosmology.G[1][j])/
    (cosmology.G[0][j+1] - cosmology.G[0][j]))*(1+z)/G;
  
  const double dlnGdlna = (z > 0.0) ? -dlnGdlnz*(1+z)/z
                                    : -dlnGdlnz_slope;

  return 1 + dlnGdlna; // Growth D = G * a
}
#endif

// ---------------------------------------------------------------------------
// Growth factor D(a) and growth rate f(a) fused in one bracket lookup:
// the formulas of norm_growfac and f_growth (see those headers) evaluated
// from a single interpolation of the cosmology.G table at z = 1/a - 1,
// plus the j = 0 (fast variant) or binary-search (fallback) lookup of
// the G(0) normalization.
//
// Cache invalidation:
// no static state; the table is maintained by
// set_growth.
//
// Parameters:
//   a            - scale factor
//   normalize_z0 - true: D is divided by G(0) so that D(a=1) = 1
//
// Returns:
//   struct growths { D = D(a), f = dlnD/dlna at a }
// ---------------------------------------------------------------------------
#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
// ---------------------------------------------------------------------------
// Fast variant: j = 0 bracket for the G(0) normalization and direct-index
// bracket lookup at the query z. Full contract: shared header above.
//
// Parameters:
//   a            - scale factor
//   normalize_z0 - true: D is divided by G(0) so that D(a=1) = 1
//
// Returns:
//   struct growths { D = D(a), f = dlnD/dlna at a }
// ---------------------------------------------------------------------------
struct growths norm_growfac_all(const double a, const bool normalize_z0)
{
  // ---------------------------------------------------------------
  // First lookup: G(z=0), used as the z=0 normalization. With z=0
  // and a piecewise-uniform grid that starts at z=0, the bracket is
  // always j=0; we pick that explicitly to skip the search entirely.
  // ---------------------------------------------------------------
  double growfact1;
  {
    const double z = 0.0;
    const int j = 0;
    const double dy = (z                   - cosmology.G[0][j]) /
                      (cosmology.G[0][j+1] - cosmology.G[0][j]);
    growfact1 = cosmology.G[1][j] + dy * (cosmology.G[1][j+1] - cosmology.G[1][j]);
  }

  // ---------------------------------------------------------------
  // Second lookup: G at the query redshift. Direct-index lookup on
  // the piecewise-uniform z grid; metadata populated by set_distances
  // (or whichever function fills cosmology.G).
  // ---------------------------------------------------------------
  const double z = 1.0/a - 1.0;

  const int j = piecewise_index(z, cosmology.G_z_nseg,
                                cosmology.G_z_seg_start, cosmology.G_z_seg_len,
                                cosmology.G_z_seg_xmin,  cosmology.G_z_seg_inv_dx,
                                cosmology.G_nz);

  const double zj  = cosmology.G[0][j  ];
  const double zj1 = cosmology.G[0][j+1];
  const double Gj  = cosmology.G[1][j  ];
  const double Gj1 = cosmology.G[1][j+1];

  const double dy = (z - zj) / (zj1 - zj);
  const double G  = Gj + dy * (Gj1 - Gj);

  const double dlnGdlnz = ((Gj1 - Gj) / (zj1 - zj)) * z / G;
  // z-cancelled form, finite at z = 0 (dlnGdlnz carries a factor z)
  const double dlnGdlnz_slope = ((Gj1 - Gj)/
    (zj1 - zj))*(1+z)/G;
  const double dlnGdlna = (z > 0.0) ? -dlnGdlnz*(1+z)/z
                                    : -dlnGdlnz_slope;

  struct growths Gf;
  Gf.f = 1 + dlnGdlna;
  Gf.D = normalize_z0 ? (G*a)/growfact1 : (G*a);
  return Gf;
}
#else
// ---------------------------------------------------------------------------
// Fallback variant: binary-search bracket lookups (both the G(0)
// normalization and the query z). Full contract: shared header above.
//
// Parameters:
//   a            - scale factor
//   normalize_z0 - true: D is divided by G(0) so that D(a=1) = 1
//
// Returns:
//   struct growths { D = D(a), f = dlnD/dlna at a }
// ---------------------------------------------------------------------------
struct growths norm_growfac_all(const double a, const bool normalize_z0)
{
  // first lookup: G(0) for the z = 0 normalization (binary search)
  double growfact1;
  {
    const double z = 0.0;
    int j = 0;
    {
      size_t ilo = 0;
      size_t ihi = cosmology.G_nz-1;
      while (ihi>ilo+1) 
      {
        size_t ll = (ihi+ilo)/2;
        if(cosmology.G[0][ll]>z)
          ihi = ll;
        else
          ilo = ll;
      }
      j = ilo;
    }
    const double dy = (z                   - cosmology.G[0][j])/
                      (cosmology.G[0][j+1] - cosmology.G[0][j]);
    
    growfact1 = cosmology.G[1][j] + dy*(cosmology.G[1][j+1] - cosmology.G[1][j]);
  }

  // second lookup: G at the query redshift (binary search)
  const double z = 1.0/a-1.0;

  int j = 0;
  {
    size_t ilo = 0;
    size_t ihi = cosmology.G_nz - 1;
    while (ihi>ilo+1)
    {
      size_t ll = (ihi+ilo)/2;
      if(cosmology.G[0][ll]>z)
        ihi = ll;
      else
        ilo = ll;
    }
    j = ilo;
  }

  const double dy = (z                   - cosmology.G[0][j])/
                    (cosmology.G[0][j+1] - cosmology.G[0][j]);
                    
  const double G = cosmology.G[1][j] + dy*(cosmology.G[1][j+1] - cosmology.G[1][j]);

  const double dlnGdlnz = ((cosmology.G[1][j+1] - cosmology.G[1][j])/
                           (cosmology.G[0][j+1] - cosmology.G[0][j]))*z/G;
  // z-cancelled form, finite at z = 0 (dlnGdlnz carries a factor z)
  const double dlnGdlnz_slope = ((cosmology.G[1][j+1] - cosmology.G[1][j])/
    (cosmology.G[0][j+1] - cosmology.G[0][j]))*(1+z)/G;
  
  const double dlnGdlna = (z > 0.0) ? -dlnGdlnz*(1+z)/z
                                    : -dlnGdlnz_slope;

  struct growths Gf;
  Gf.f = 1 + dlnGdlna; // Growth D = G * a

  if(normalize_z0)
    Gf.D = (G*a)/growfact1; 
  else
    Gf.D = (G*a);

  return Gf;
}
#endif

// ---------------------------------------------------------------------------
// D(a) normalized to D(1) = 1 and f(a) in one lookup: shorthand for
// norm_growfac_all(a, true).
//
// Parameters:
//   a - scale factor
//
// Returns:
//   struct growths { D, f }
// ---------------------------------------------------------------------------
struct growths growfac_all(const double a)
{
  return norm_growfac_all(a, true);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Power Spectrum (LINEAR)
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Linear matter power spectrum P_lin(k, a) by bilinear interpolation of
// ln P on the CAMB-fed (log10 k, z) table cosmology.lnPL (loaded by
// set_linear_power_spectrum in the C interface). Table layout:
//
//   lnPL[i][j]       = ln P at (log10k_i, z_j), P in (Mpc/h)^3
//   lnPL[i][lnPL_nz] = log10k_i  (k axis, k in h/Mpc)
//   lnPL[lnPL_nk][j] = z_j       (z axis)
//
// Unit conventions: the input k is in (c/H0)^-1 units, i.e.
// k = k[h/Mpc]*coverH0, so the query point is (log10(k/coverH0),
// z = 1/a - 1); the result exp(lnP)/coverH0^3 converts (Mpc/h)^3 to
// (c/H0)^3 units. A query outside the grid keeps the nearest edge
// bracket, so ln P is linearly extrapolated from that bracket.
//
// Bracket selection: the fast variant uses direct-index lookups (single
// uniform segment in log10 k, piecewise-uniform z segments, via the
// cosmology.lnPL_* metadata); the fallback uses two binary searches.
//
// Cache invalidation:
// no static state. The cosmology.lnPL table and its
// grid metadata are replaced by set_linear_power_spectrum, which also
// bumps cosmology.random.
//
// Parameters:
//   k - wavenumber in (c/H0)^-1 units (k = k[h/Mpc]*coverH0)
//   a - scale factor
//
// Returns:
//   P_lin(k, a) in (c/H0)^3 units
// ---------------------------------------------------------------------------
#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
// ---------------------------------------------------------------------------
// Fast variant: direct-index bracket lookups (single uniform segment in
// log10 k, piecewise-uniform z). Full contract: shared header above.
//
// Parameters:
//   k - wavenumber in (c/H0)^-1 units (k = k[h/Mpc]*coverH0)
//   a - scale factor
//
// Returns:
//   P_lin(k, a) in (c/H0)^3 units
// ---------------------------------------------------------------------------
double p_lin(const double k, const double a)
{
  // convert from (x/Mpc/h - dimensioneless) to h/Mpc with x = c/H0 (Mpc)
  const double log10k = log10(k / cosmology.coverH0);
  const double z      = 1.0 / a - 1.0;

  // logk = cosmology.lnPL[0:nk, cosmology.lnPL_nz]
  // z    = cosmology.lnPL[cosmology.lnPL_nk, 0:nz]

  // -----------------------------------------------------------------
  // Direct-index lookup on log10k axis (single uniform segment) and
  // z axis (piecewise-uniform). 
  // Replaces two binary searches that were ~60% of this function's cost.
  //
  // Index is clamped so [i, i+1] and [j, j+1] are always valid;
  // out-of-range queries snap to the nearest interior bracket, which
  // matches the behavior of the previous binary-search version.
  // -----------------------------------------------------------------
  int i = (int)((log10k - cosmology.lnPL_log10k_min) * cosmology.lnPL_log10k_inv_dx);
  if (i < 0)                       i = 0;
  if (i > cosmology.lnPL_nk - 2)   i = cosmology.lnPL_nk - 2;

  int j = piecewise_index(z, cosmology.lnPL_z_nseg,
                          cosmology.lnPL_z_seg_start, cosmology.lnPL_z_seg_len,
                          cosmology.lnPL_z_seg_xmin,  cosmology.lnPL_z_seg_inv_dx,
                          cosmology.lnPL_nz);

  // Compute interpolation fractions from the grid points stored in the table
  const double xi  = cosmology.lnPL[i  ][cosmology.lnPL_nz];
  const double xi1 = cosmology.lnPL[i+1][cosmology.lnPL_nz];
  const double zj  = cosmology.lnPL[cosmology.lnPL_nk][j  ];
  const double zj1 = cosmology.lnPL[cosmology.lnPL_nk][j+1];

  const double dx = (log10k - xi) / (xi1 - xi);
  const double dy = (z      - zj) / (zj1 - zj);

  const double out_lnP =   (1-dx)*(1-dy) * cosmology.lnPL[i  ][j  ]
                         + (1-dx)*   dy  * cosmology.lnPL[i  ][j+1]
                         +    dx *(1-dy) * cosmology.lnPL[i+1][j  ]
                         +    dx *   dy  * cosmology.lnPL[i+1][j+1];

  // convert from (Mpc/h)^3 to (Mpc/h)^3/(c/H0=100)^3 (dimensioneless)
  return exp(out_lnP) / (cosmology.coverH0 * cosmology.coverH0 * cosmology.coverH0);
}
#else
// ---------------------------------------------------------------------------
// Fallback variant: binary-search bracket lookups on both axes. Full
// contract: shared header above.
//
// Parameters:
//   k - wavenumber in (c/H0)^-1 units (k = k[h/Mpc]*coverH0)
//   a - scale factor
//
// Returns:
//   P_lin(k, a) in (c/H0)^3 units
// ---------------------------------------------------------------------------
double p_lin(const double k, const double a)
{
  // convert from (x/Mpc/h - dimensioneless) to h/Mpc with x = c/H0 (Mpc)
  const double log10k = log10(k/cosmology.coverH0);
  const double z = 1.0/a-1.0;

  // logk = cosmology.lnPL[0:nk,cosmology.lnPL_nz]
  // z    = cosmology.lnPL[cosmology.lnPL_nk,0:nz]
  
  // bracket log10k by binary search on the k-axis row
  int i = 0;
  {
    size_t ilo = 0;
    size_t ihi = cosmology.lnPL_nk-1;
    while (ihi>ilo+1) 
    {
      size_t ll = (ihi+ilo)/2;
      if(cosmology.lnPL[ll][cosmology.lnPL_nz] > log10k)
        ihi = ll;
      else
        ilo = ll;
    }
    i = ilo;
  }

  // bracket z by binary search on the z-axis row
  int j = 0;
  {
    size_t ilo = 0;
    size_t ihi = cosmology.lnPL_nz-1;
    while (ihi>ilo+1) 
    {
      size_t ll = (ihi+ilo)/2;
      if(cosmology.lnPL[cosmology.lnPL_nk][ll] > z)
        ihi = ll;
      else
        ilo = ll;
    }
    j = ilo;
  }

  // bilinear ln P interpolation on the [i, i+1] x [j, j+1] cell
  double dx = (log10k                                 - cosmology.lnPL[i][cosmology.lnPL_nz])/
              (cosmology.lnPL[i+1][cosmology.lnPL_nz] - cosmology.lnPL[i][cosmology.lnPL_nz]);

  double dy = (z                                     - cosmology.lnPL[cosmology.lnPL_nk][j])/
              (cosmology.lnPL[cosmology.lnPL_nk][j+1]- cosmology.lnPL[cosmology.lnPL_nk][j]);

  const double out_lnP =    (1-dx)*(1-dy)*cosmology.lnPL[i][j]
                          + (1-dx)*dy*cosmology.lnPL[i][j+1]
                          + dx*(1-dy)*cosmology.lnPL[i+1][j]
                          + dx*dy*cosmology.lnPL[i+1][j+1];

  // convert from (Mpc/h)^3 to (Mpc/h)^3/(c/H0=100)^3 (dimensioneless)
  return exp(out_lnP)/(cosmology.coverH0*cosmology.coverH0*cosmology.coverH0);
}
#endif



// ---------------------------------------------------------------------------
// Linear power spectrum of cold dark matter plus baryons, P_cb(k, a): the
// matter field without the massive neutrinos, which free-stream out of
// halos. Read by sigma2 when like.halo_model[4] = HALO_FIELD_CB.
//
// Table: cosmology.lnPL_cb[i][j] = ln P_cb at (log10k_i, z_j), installed
// by set_linear_power_spectrum_cb on the grid of cosmology.lnPL; the axes
// and the direct-index metadata are read from lnPL. The two variants
// below are p_lin's two variants with lnPL_cb in place of lnPL in the
// four value reads: the same brackets, fractions and arithmetic, so a
// P_cb table equal to P_lin returns p_lin's values bit for bit.
//
// Precondition: cosmology.lnPL_cb installed (the caller checks; sigma2
// aborts otherwise).
//
// Cache invalidation:
// no static state. set_linear_power_spectrum_cb bumps cosmology.random
// when it installs a new table.
//
// Parameters:
//   k - wavenumber in (c/H0)^-1 units (k = k[h/Mpc]*coverH0)
//   a - scale factor
//
// Returns:
//   P_cb(k, a) in (c/H0)^3 units
// ---------------------------------------------------------------------------
#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
double p_lin_cb(const double k, const double a)
{
  // convert from (x/Mpc/h - dimensioneless) to h/Mpc with x = c/H0 (Mpc)
  const double log10k = log10(k / cosmology.coverH0);
  const double z      = 1.0 / a - 1.0;

  // brackets by direct index on lnPL's axes (see p_lin)
  int i = (int)((log10k - cosmology.lnPL_log10k_min) * cosmology.lnPL_log10k_inv_dx);
  if (i < 0)                       i = 0;
  if (i > cosmology.lnPL_nk - 2)   i = cosmology.lnPL_nk - 2;

  int j = piecewise_index(z, cosmology.lnPL_z_nseg,
                          cosmology.lnPL_z_seg_start, cosmology.lnPL_z_seg_len,
                          cosmology.lnPL_z_seg_xmin,  cosmology.lnPL_z_seg_inv_dx,
                          cosmology.lnPL_nz);

  // interpolation fractions from lnPL's grid points
  const double xi  = cosmology.lnPL[i  ][cosmology.lnPL_nz];
  const double xi1 = cosmology.lnPL[i+1][cosmology.lnPL_nz];
  const double zj  = cosmology.lnPL[cosmology.lnPL_nk][j  ];
  const double zj1 = cosmology.lnPL[cosmology.lnPL_nk][j+1];

  const double dx = (log10k - xi) / (xi1 - xi);
  const double dy = (z      - zj) / (zj1 - zj);

  // bilinear ln P_cb on the [i, i+1] x [j, j+1] cell
  const double out_lnP =   (1-dx)*(1-dy) * cosmology.lnPL_cb[i  ][j  ]
                         + (1-dx)*   dy  * cosmology.lnPL_cb[i  ][j+1]
                         +    dx *(1-dy) * cosmology.lnPL_cb[i+1][j  ]
                         +    dx *   dy  * cosmology.lnPL_cb[i+1][j+1];

  // convert from (Mpc/h)^3 to (Mpc/h)^3/(c/H0=100)^3 (dimensioneless)
  return exp(out_lnP) / (cosmology.coverH0 * cosmology.coverH0 * cosmology.coverH0);
}
#else
double p_lin_cb(const double k, const double a)
{
  // convert from (x/Mpc/h - dimensioneless) to h/Mpc with x = c/H0 (Mpc)
  const double log10k = log10(k/cosmology.coverH0);
  const double z = 1.0/a-1.0;

  // bracket log10k by binary search on lnPL's k-axis row
  int i = 0;
  {
    size_t ilo = 0;
    size_t ihi = cosmology.lnPL_nk-1;
    while (ihi>ilo+1)
    {
      size_t ll = (ihi+ilo)/2;
      if(cosmology.lnPL[ll][cosmology.lnPL_nz] > log10k)
        ihi = ll;
      else
        ilo = ll;
    }
    i = ilo;
  }

  // bracket z by binary search on lnPL's z-axis row
  int j = 0;
  {
    size_t ilo = 0;
    size_t ihi = cosmology.lnPL_nz-1;
    while (ihi>ilo+1)
    {
      size_t ll = (ihi+ilo)/2;
      if(cosmology.lnPL[cosmology.lnPL_nk][ll] > z)
        ihi = ll;
      else
        ilo = ll;
    }
    j = ilo;
  }

  // bilinear ln P_cb interpolation on the [i, i+1] x [j, j+1] cell
  double dx = (log10k                                 - cosmology.lnPL[i][cosmology.lnPL_nz])/
              (cosmology.lnPL[i+1][cosmology.lnPL_nz] - cosmology.lnPL[i][cosmology.lnPL_nz]);

  double dy = (z                                     - cosmology.lnPL[cosmology.lnPL_nk][j])/
              (cosmology.lnPL[cosmology.lnPL_nk][j+1]- cosmology.lnPL[cosmology.lnPL_nk][j]);

  const double out_lnP =    (1-dx)*(1-dy)*cosmology.lnPL_cb[i][j]
                          + (1-dx)*dy*cosmology.lnPL_cb[i][j+1]
                          + dx*(1-dy)*cosmology.lnPL_cb[i+1][j]
                          + dx*dy*cosmology.lnPL_cb[i+1][j+1];

  // convert from (Mpc/h)^3 to (Mpc/h)^3/(c/H0=100)^3 (dimensioneless)
  return exp(out_lnP)/(cosmology.coverH0*cosmology.coverH0*cosmology.coverH0);
}
#endif



// ---------------------------------------------------------------------------
// Density parameter of the field the halo model counts halos in, chosen
// by like.halo_model[4] (halo.h):
//
//   HALO_FIELD_MATTER   Omega_m             (total matter)
//   HALO_FIELD_CB       Omega_m - Omega_nu  (cold dark matter + baryons)
//
// rho_crit times this value is the mean density that ties a halo mass
// to its Lagrangian radius in sigma2 and that sets the rho/M factor of
// dn/dlnM in halo.c and halo_cluster.c. Under HALO_FIELD_MATTER it
// returns cosmology.Omega_m itself, so the products it enters are
// today's products bit for bit.
//
// Aborts: any other like.halo_model[4].
//
// Returns:
//   Omega of the halo field, dimensionless
// ---------------------------------------------------------------------------
double omega_halo_field(void)
{
  if (HALO_FIELD_MATTER == like.halo_model[4]) {
    return cosmology.Omega_m;
  }
  else if (HALO_FIELD_CB == like.halo_model[4]) {
    return cosmology.Omega_m - cosmology.Omega_nu;
  }
  else {
    log_fatal("like.halo_model[4] = %d not supported", like.halo_model[4]);
    exit(1);
  }
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Power Spectrum (NON-LINEAR)
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Non-linear matter power spectrum P_nl(k, a): the same bilinear ln P
// interpolation, unit conventions, edge-bracket extrapolation and
// direct-index/binary-search variants as p_lin (see p_lin), reading the
// cosmology.lnP table loaded by set_non_linear_power_spectrum. When
// bary.is_Pk_bary == 1 the result is multiplied by the hydro-sim
// suppression PkRatio_baryons(k, a).
//
// Cache invalidation:
// no static state. The cosmology.lnP table and its
// grid metadata are replaced by set_non_linear_power_spectrum, which
// also bumps cosmology.random.
//
// Parameters:
//   k - wavenumber in (c/H0)^-1 units (k = k[h/Mpc]*coverH0)
//   a - scale factor
//
// Returns:
//   P_nl(k, a) in (c/H0)^3 units, times the baryonic ratio when enabled
// ---------------------------------------------------------------------------
#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
// ---------------------------------------------------------------------------
// Fast variant: direct-index bracket lookups (single uniform segment in
// log10 k, piecewise-uniform z). Full contract: shared header above.
//
// Parameters:
//   k - wavenumber in (c/H0)^-1 units (k = k[h/Mpc]*coverH0)
//   a - scale factor
//
// Returns:
//   P_nl(k, a) in (c/H0)^3 units, times the baryonic ratio when enabled
// ---------------------------------------------------------------------------
double p_nonlin(const double k, const double a)
{
  const double coverH0 = cosmology.coverH0;
  // convert from (x/Mpc/h - dimensioneless) to h/Mpc with x = c/H0 (Mpc)
  const double log10k = log10(k / coverH0);
  const double z      = 1.0 / a - 1.0;

  // logk = cosmology.lnP[0:nk, cosmology.lnP_nz]
  // z    = cosmology.lnP[cosmology.lnP_nk, 0:nz]

  // -----------------------------------------------------------------
  // Direct-index lookup, as in p_lin: single uniform segment in
  // log10 k, piecewise-uniform z, with the same clamping (an
  // out-of-range query snaps to the nearest interior bracket). The
  // lnP_* metadata fields are set in set_non_linear_power_spectrum.
  // -----------------------------------------------------------------
  int i = (int)((log10k - cosmology.lnP_log10k_min) * cosmology.lnP_log10k_inv_dx);
  if (i < 0)                     i = 0;
  if (i > cosmology.lnP_nk - 2)  i = cosmology.lnP_nk - 2;

  int j = piecewise_index(z, cosmology.lnP_z_nseg,
                          cosmology.lnP_z_seg_start, cosmology.lnP_z_seg_len,
                          cosmology.lnP_z_seg_xmin,  cosmology.lnP_z_seg_inv_dx,
                          cosmology.lnP_nz);

  const double xi  = cosmology.lnP[i  ][cosmology.lnP_nz];
  const double xi1 = cosmology.lnP[i+1][cosmology.lnP_nz];
  const double zj  = cosmology.lnP[cosmology.lnP_nk][j  ];
  const double zj1 = cosmology.lnP[cosmology.lnP_nk][j+1];

  const double dx = (log10k - xi) / (xi1 - xi);
  const double dy = (z      - zj) / (zj1 - zj);

  const double out_lnP =   (1-dx)*(1-dy) * cosmology.lnP[i  ][j  ]
                         + (1-dx)*   dy  * cosmology.lnP[i  ][j+1]
                         +    dx *(1-dy) * cosmology.lnP[i+1][j  ]
                         +    dx *   dy  * cosmology.lnP[i+1][j+1];

  const double ans = exp(out_lnP) / (coverH0 * coverH0 * coverH0);

  return (bary.is_Pk_bary == 1) ? ans * PkRatio_baryons(k, a) : ans;
}
#else
// ---------------------------------------------------------------------------
// Fallback variant: binary-search bracket lookups on both axes. Full
// contract: shared header above.
//
// Parameters:
//   k - wavenumber in (c/H0)^-1 units (k = k[h/Mpc]*coverH0)
//   a - scale factor
//
// Returns:
//   P_nl(k, a) in (c/H0)^3 units, times the baryonic ratio when enabled
// ---------------------------------------------------------------------------
double p_nonlin(const double k, const double a)
{
  const double coverH0 = cosmology.coverH0;
  // convert from (x/Mpc/h - dimensioneless) to h/Mpc with x = c/H0 (Mpc)
  const double log10k = log10(k/coverH0);
  const double z = 1.0/a-1.0;

  // logk = cosmology.lnP[0:nk,cosmology.lnP_nz]
  // z    = cosmology.lnP[cosmology.lnP_nk,0:nz]
  
  // bracket log10k by binary search on the k-axis row
  int i = 0;
  {
    size_t ilo = 0;
    size_t ihi = cosmology.lnP_nk-1;
    while (ihi>ilo+1) 
    {
      size_t ll = (ihi+ilo)/2;
      if(cosmology.lnP[ll][cosmology.lnP_nz] > log10k)
        ihi = ll;
      else
        ilo = ll;
    }
    i = ilo;
  }

  // bracket z by binary search on the z-axis row
  int j = 0;
  {
    size_t ilo = 0;
    size_t ihi = cosmology.lnP_nz-1;
    while (ihi>ilo+1) 
    {
      size_t ll = (ihi+ilo)/2;
      if(cosmology.lnP[cosmology.lnP_nk][ll] > z)
        ihi = ll;
      else
        ilo = ll;
    }
    j = ilo;
  }

  // bilinear ln P interpolation on the [i, i+1] x [j, j+1] cell
  double dx = (log10k                               - cosmology.lnP[i][cosmology.lnP_nz])/
              (cosmology.lnP[i+1][cosmology.lnP_nz] - cosmology.lnP[i][cosmology.lnP_nz]);


  double dy = (z                                   - cosmology.lnP[cosmology.lnP_nk][j])/
              (cosmology.lnP[cosmology.lnP_nk][j+1]- cosmology.lnP[cosmology.lnP_nk][j]);

  const double out_lnP =  (1-dx)*(1-dy)*cosmology.lnP[i][j]
                          + (1-dx)*dy*cosmology.lnP[i][j+1]
                          + dx*(1-dy)*cosmology.lnP[i+1][j]
                          + dx*dy*cosmology.lnP[i+1][j+1];
  
  const double ans = exp(out_lnP)/(coverH0*coverH0*coverH0);
  
  return (bary.is_Pk_bary==1) ? ans*PkRatio_baryons(k,a) : ans;
}
#endif

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Matter power spectrum dispatch: p_lin when pdeltaparams.runmode is
// "linear", p_nonlin otherwise. The choice is latched in a static on the
// first call and reused for the rest of the process, so the run mode
// must be set before the first evaluation.
//
// Cache invalidation:
// none. The static P_type latch never rebuilds; only
// the tables read by p_lin/p_nonlin refresh (see those headers).
//
// Parameters:
//   io_kNL - wavenumber in (c/H0)^-1 units (k = k[h/Mpc]*coverH0)
//   io_a   - scale factor
//
// Returns:
//   P(io_kNL, io_a) in (c/H0)^3 units from the selected branch
// ---------------------------------------------------------------------------
double Pdelta(double io_kNL, double io_a)
{
  double out_PK;
  static int P_type = -1;
  if (P_type == -1) 
  {
    if (strcmp(pdeltaparams.runmode,"linear") == 0) 
    {
      P_type = 3;
    }
  }
  // P_type encoding: 3 = linear (latched above when runmode is
  // "linear"); every other value - including the -1 "unset" latch -
  // falls through to p_nonlin
  switch (P_type) 
  {
    case 3:
      out_PK = p_lin(io_kNL, io_a);
      break;
    default:
      out_PK = p_nonlin(io_kNL, io_a);
      break;
  }
  return out_PK;
}

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Comoving angular diameter distance f_K(chi). Bartelmann & Schneider
// 2001 (BS01) eqs 2.4, 2.30: f_K is a trigonometric, linear, or
// hyperbolic function of chi, depending on the curvature of the
// Universe.
//
// With K = Omega_m + Omega_v - 1 in (H0/c)^2 units (so the K_h =
// sqrt(|K|) of the code below is in H0/c units) and chi in c/H0
// units:
//
//   K >  1e-6:  f_K = sin(sqrt(K)*chi)/sqrt(K)
//   K < -1e-6:  f_K = sinh(sqrt(-K)*chi)/sqrt(-K)
//   else:       f_K = chi                          (flat)
//
// The 1e-6 threshold: at K = 0 the closed forms are 0/0, and their
// K -> 0 limit is chi. Treating |K| <= 1e-6 as flat drops the leading
// curvature term -K*chi^3/6, a fractional error |K|*chi^2/6 of at
// most a few times 1e-6 over the tabulated chi range.
//
// Parameters:
//   chi - comoving distance in c/H0 units
//
// Returns:
//   f_K(chi) in c/H0 units
// ---------------------------------------------------------------------------
double f_K(double chi)
{
  double K, K_h, f;
  K = (cosmology.Omega_m + cosmology.Omega_v - 1.);
  if (K > 1e-6) 
  { // closed
    K_h = sqrt(K); // K in units H0/c see BS eq. 2.30
    f = 1. / K_h * sin(K_h * chi);
  } else if (K < -1e-6) 
  { // open
    K_h = sqrt(-K);
    f = 1. / K_h * sinh(K_h * chi);
  } else 
  { // flat
    f = chi;
  }
  return f;
}

// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Baryons
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Baryonic feedback ratio P(k)_bary/P(k)_DMO from hydro sims.
//
// Returns 1 when bary.is_Pk_bary == 0. Otherwise evaluates the GSL 2D
// spline bary.interp2d of log10(P_bary/P_DMO) on the (log10 k, a) grid
// (bary.logk_bins, bary.a_bins; k in h/Mpc, so the input is converted
// with k_NL/coverH0), with extrapolation outside the grid, and returns
// 10^result. Aborts on GSL error.
//
// Cache invalidation:
// no static state; the bary tables are maintained by
// the baryons module (baryons.c).
//
// Parameters:
//   k_NL - wavenumber in (c/H0)^-1 units (k = k[h/Mpc]*coverH0)
//   a    - scale factor
//
// Returns:
//   P_bary/P_DMO at (k_NL, a), or 1 when baryonic feedback is off
// ---------------------------------------------------------------------------
double PkRatio_baryons(double k_NL, double a)
{
  if (bary.is_Pk_bary == 0)
  {
    return 1.;
  } else
  {
    const double kintern = k_NL/cosmology.coverH0;
    double result;
    int status = gsl_interp2d_eval_extrap_e(bary.interp2d, bary.logk_bins,
      bary.a_bins, bary.log_PkR, log10(kintern), a, NULL, NULL, &result);
    if (status)
    {
      log_fatal(gsl_strerror(status));
      exit(1);
    }
    return pow(10.0, result);
  }
}

// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// MODIFIED GRAVITY
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Modified-gravity Sigma(a) hook (lensing-amplitude modification). This
// implementation is GR-only: it returns 0 for every a.
//
// Parameters:
//   a - scale factor (unused)
//
// Returns:
//   0.0
// ---------------------------------------------------------------------------
double MG_Sigma(double a __attribute__((unused))) {
  return 0.0;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Cached sigma^2(M) at a = 1: the variance of the linear density field
// after smoothing with a top-hat sphere that holds the mass M.
//
// What the caller gets. A table of ln sigma^2 on Ntable.N_M nodes
// uniform in ln M over [ln limits.halo_m_min, ln limits.halo_m_max]
// (by default 1024 nodes over M = 1e6..1e17 M_sun/h), read back by
// linear interpolation in ln M and exponentiated. The table is built
// at a = 1 once per cosmology; halo.c rescales with the growth factor
// D(a) where it needs sigma at a < 1: nu = delta_c/(sqrt(sigma2(M)) D(a)).
//
// 1. The integral
//
// Smoothing the density field over a sphere of radius R multiplies each
// Fourier mode by the sphere's transform, the top-hat window W(kR); the
// variance of the smoothed field is the power spectrum weighted by W^2
// (Cooray & Sheth 2002, astro-ph/0206508, Sec. 3.2):
//
//   sigma^2(R) = 1/(2 pi^2) int_0^inf k^2 P_lin(k) W(kR)^2 dk,
//   W(x)       = 3 (sin x - x cos x)/x^3 = 3 j1(x)/x.
//
// j1 is the spherical Bessel function of order one,
//
//   j1(x) = sin x/x^2 - cos x/x,    j1(x) -> x/3 as x -> 0,
//
// so W(0) = 1 (W(0.01) = 0.99999): modes much longer than R pass
// through untouched, modes much shorter than R average away. Mass and
// radius are tied by the mean matter density rho_crit Omega_m (the
// density of the smoothed field: item 10),
//
//   M = (4 pi/3) R^3 rho_crit Omega_m,
//   R = (3 M/(4 pi rho_crit Omega_m))^(1/3).
//
// Substituting x = kR (so k = x/R and dk = dx/R) inside the integral,
//
//   k^2 W(kR)^2 dk = (x^2/R^2) (9 j1(x)^2/x^2) (dx/R) = 9 j1(x)^2 dx/R^3,
//
//   sigma^2(M) = 1/(2 pi^2 R^3) int_0^inf P_lin(x/R) 9 j1(x)^2 dx.
//
// The x^2 of k^2 dk cancels the 1/x^2 of W^2 exactly; that is why the
// cached weights below carry 9 j1(x)^2 and no other power of x.
//
// 2. Why the integrand is awkward, and the cure
//
// P_lin is smooth: one broad peak near k ~ 0.02 h/Mpc, a k^-3 (ln k)^2
// fall beyond it, baryon wiggles in between. The factor 9 j1(x)^2 is
// not: it oscillates forever, vanishing wherever j1 does. Between two
// consecutive zeros, however, it is a single smooth bump with no
// structure inside. The zeros of j1 are the roots of tan x = x,
//
//   z_1 = 4.4934,  z_2 = 7.7253,  z_3 = 10.9041,  z_4 = 14.0662, ...
//   z_n -> (n + 1/2) pi    (one root just below each pole of tan x),
//
// and they cut the x axis into segments:
//
//   head    [0, z_1]        the main bump (W falls from 1 to 0);
//                           holds most of sigma^2
//   lobe n  [z_n, z_{n+1}]  one bump each, shrinking as x grows
//
// Integrating every segment on its own with a small quadrature rule
// turns one hard oscillatory integral into a sum of easy ones.
//
// 3. Integrating one smooth bump with a few nodes
//
// A quadrature rule approximates an integral by a weighted sum of the
// integrand at prescribed nodes x_i with prescribed weights w_i:
//
//   int_a^b f(x) dx  ~  sum_i w_i f(x_i).
//
// The rule used here picks the n nodes (the roots of the Legendre
// polynomial P_n, mapped from [-1, 1] onto [a, b]) and the n weights
// so that the sum is exact for every polynomial of degree <= 2n - 1:
// 2n free numbers buy 2n conditions. The 2-node rule on [-1, 1] has
// nodes +-1/sqrt(3) and weights 1; it returns int x^2 dx = 2/3 and
// int x^3 dx = 0 exactly and first fails at x^4 (0.222 for 0.4). The
// error of the n-node rule is proportional to the 2n-th derivative of
// f, so on a function that is one smooth bump it falls exponentially
// with n: 8 nodes per lobe reproduce the sum of all lobes to better
// than 1e-9 of sigma^2 for every tabulated mass. This is Gauss-Legendre
// quadrature, "GL" in the comments below.
//
// GSL supplies the rule as a table. malloc_gslint_glfixed(n) wraps
// gsl_integration_glfixed_table_alloc(n): the n nodes and weights on
// [-1, 1], stored to full precision for n = 2..20, 32, 64, 96, 100,
// 128, 256, 512, 1024 and computed on the fly (less precisely) for any
// other n, which is why the ladders of item 5 use only those sizes.
// The call
//
//   gsl_integration_glfixed_point(a, b, i, &x, &w, t)
//
// writes node i of table t mapped onto [a, b] into x and its weight,
// scaled by (b - a)/2, into w.
//
// 4. The head in ln x
//
// For halo-scale R the head spans an enormous range of k. At M = 1e6
// M_sun/h with Omega_m = 0.3, R = 0.0142 Mpc/h, so x in [0, 4.49] is
// k in [0, 316] h/Mpc: the peak of P_lin (k ~ 0.02) sits at x ~ 3e-4
// and the baryon wiggles (k ~ 0.05-0.3) at x ~ 1e-3 to 4e-3. A rule
// whose 256 nodes spread over [0, 4.49] would put none of them there.
// The head is therefore integrated in s = ln x, which spreads the
// nodes evenly over decades of x:
//
//   x = e^s,  dx = x ds   ->   int f(x) dx = int f(e^s) e^s ds,
//
// so every head weight is multiplied by its node's x. A logarithmic
// variable needs a finite lower edge: XMIN = 1e-5. Below it the
// integrand behaves as P(x/R) 9 j1^2 ~ x^(n_s + 2) (P ~ k^n_s at low
// k, 9 j1^2 ~ x^2), so the dropped piece scales as XMIN^(n_s + 3):
// about 1e-10 of the head at M = 1e6 and far less for heavier halos,
// whose k = XMIN/R is smaller still.
//
// 5. Node counts: the hdi ladder
//
// hdi = abs(Ntable.high_def_integration) is the accuracy knob the
// Limber quadratures of cosmo2D.c also read (0 by default). A "ladder"
// is a chain of ?: choices mapping hdi to a size:
//
//   hdi                 0      1      2      >= 3
//   head nodes (nph)    256    512    1024   1024
//   lobe nodes (npl)    8      12     16     20
//   cached nodes        4352   6656   9216   11264   (nph + 512 npl)
//
// The head carries the error budget, so it gets the large rule; a
// lobe is one bump and 8 nodes already saturate it. Measured at
// hdi = 0 against an independent numpy integration of the same P_lin:
// 6.2e-6 maximum relative error in sigma^2 and 1.9e-6 scatter between
// neighboring masses. The scatter is what the finite difference in
// halo.c's dlognudlogm sees; that slope stays within 1.3e-4 of the
// reference.
//
// 6. Lobe count: NLOBE = 512
//
// Far out, j1(x) -> -cos x/x (at x = 50: -0.01940 against -0.01930),
// and far above the peak P_lin ~ k^-3 (ln k)^2, so the envelope of the
// integrand falls as
//
//   P(x/R) 9 j1(x)^2  ~  (R/x)^3 (ln)^2 9 cos^2(x)/x^2  ~  x^-5,
//
// and the lobe sums decay like z_n^-5. Lobe 512 ends at z_513 ~ 1613,
// where the envelope is (10/1613)^5 ~ 1e-11 of its value at x = 10.
// The stopping rule of item 8 exits long before that for every
// tabulated mass (89 segments at most), so the cache is a ceiling,
// not a cost.
//
// 7. Locating the zeros of j1
//
// j1(x) = 0 <=> sin x = x cos x <=> tan x = x. tan x has a pole at
// every (n + 1/2) pi, and one root of tan x = x sits just below each
// pole. Two steps pin it down.
//
// A starting guess from a series. Write x = q - e with q = (n + 1/2) pi
// and expand tan x = x in powers of 1/q; solving for e term by term
// gives
//
//   z_n = q - 1/q - 2/(3 q^3) - ...,    q = (n + 1/2) pi
//
// (McMahon's expansion for large Bessel zeros). The code keeps q - 1/q.
// For n = 1: q = 3 pi/2 = 4.7124 gives 4.5002 against the exact
// 4.4934, an error of 0.007.
//
// Polishing by tangent lines. From a guess z, follow the tangent line
// of f at z down to where it crosses zero and take that as the next
// guess:
//
//   z <- z - f(z)/f'(z),   f = j1,   f'(x) = j1'(x) = j0(x) - 2 j1(x)/x,
//
// with j0(x) = sin x/x; the derivative is the recurrence
// j_n'(x) = j_{n-1}(x) - (n + 1) j_n(x)/x at n = 1. This is Newton's
// method, and each step roughly squares the error: from 4.5002 the
// first step lands at 4.49340 (1e-5 off), the second at 4.4934094579
// (2e-11), the third at machine precision. The loop allows 8 steps
// and stops once |step| < 1e-14 z. The start is within 0.007 of the
// wanted root and j1 is smooth there, so the iteration cannot wander
// to a neighboring zero.
//
// 8. The per-mass sum and its stopping rule
//
// With nodes x_q and folded weights wf_q = w_q 9 j1(x_q)^2 cached (no
// Bessel function is evaluated per mass), one mass costs
//
//   s_j   = sum over q in segment j of wf_q P_lin(x_q/R),
//   total = s_0 + s_1 + s_2 + ...    (s_0 = head, s_j = lobe j).
//
// Stopping. Once two consecutive lobes decrease, r = s_j/s_{j-1} < 1,
// pretend the decay stays geometric from here on; the remaining sum
// would then be
//
//   s_j r + s_j r^2 + s_j r^3 + ... = s_j r/(1 - r)
//
// (with s_{j-1} = 4e-9 and s_j = 2e-9: r = 1/2, estimated tail 2e-9).
// The loop exits when this estimate falls below EPS = 1e-7 of the
// running total. For a power-law decay the ratios creep toward 1, so
// the true tail is somewhat larger than the estimate (1.3x for
// s_j ~ j^-5 at j = 20): the dropped tail is of order EPS, two orders
// below the head rule's own error. Segments used at hdi = 0: 20 at
// M = 1e6 (the head holds about 99.9% of sigma^2 there), about 25
// mid-range, 89 at M = 1e17 (R = 66 Mpc/h, so the head ends at
// k = 0.068 h/Mpc and the first lobes carry the peak and wiggles of
// P_lin, about 7% of the total).
//
// 9. Coarse grid and cubic upsampling
//
// When Ntable.N_M_internal is active (192 by default) the exact sums
// run only on that many coarse ln M nodes, and a natural cubic spline
// of ln sigma^2 fills the 1024-node table (the spline is explained at
// the upsampling loop). ln sigma^2 is smooth and monotone in ln M, so
// a cubic carries far more accuracy per node than the linear reads
// the consumers make, and those reads stay untouched. Cost per
// refill: ~9.7e4 p_lin reads, 0.71 ms with 4 threads.
//
// 10. Which density field
//
// like.halo_model[4] (halo.h) names the field whose variance the table
// holds:
//
//   HALO_FIELD_MATTER  P_lin (total matter), M = (4 pi/3) R^3 rho_crit
//                      Omega_m: the equations above as written
//   HALO_FIELD_CB      P_cb (cold dark matter + baryons, p_lin_cb), and
//                      Omega_m - Omega_nu in place of Omega_m in R(M)
//
// Massive neutrinos free-stream out of the potential wells, so halos
// form from the cb field (DES Y1 clusters, 2010.01138). The table is
// P_cb at a = 1; consumers rescale by the total-matter growth D(a) as
// they do for total matter (the DES reference code does the same).
// omega_halo_field returns the Omega; under HALO_FIELD_MATTER it is
// cosmology.Omega_m itself, so that path computes today's R(M).
//
// Cache invalidation:
//   allocation, ln M limits, coarse-grid map and node cache: rebuilt
//     when Ntable.random changes (hdi enters the node counts)
//   table refill: cosmology.random (cache[0]) or Ntable.random (cache[1]);
//     cosmology.random is redrawn by a new P_lin, a new P_cb table
//     (set_linear_power_spectrum_cb), a new Omega_m or Omega_nu
//     (set_cosmological_parameters) and a flip of like.halo_model[4]
//     (init_halo_matter_field)
//
// Parameters:
//   M - halo mass in M_sun/h
//
// Returns:
//   sigma^2(M) at a = 1, linearly interpolated in ln M from the table;
//   constant outside [limits.halo_m_min, limits.halo_m_max]
// ---------------------------------------------------------------------------
double sigma2(
    const double M  // halo mass in M_sun/h
  )
{
  // Static state. A static local keeps its value between calls (it
  // lives as long as the program, not as long as one call) and starts
  // zeroed: every pointer below is NULL and cache[] is all zeros on
  // the first call, which is what makes that call build everything.
  // Two blocks write the statics:
  //
  //   Ntable rebuild block (geometry; runs when Ntable.random changes)
  //     table/lnMv/lim, the coarse-grid workspace (ncoarse, dlnc,
  //     lnMc, qidx, qdel, tabc, cspl) and the node cache (nseg, off,
  //     xs, wf). Every malloc of this function lives there.
  //   Refill block (physics; runs when cosmology.random or
  //     Ntable.random changes)
  //     the values in table (and in tabc, cspl on the coarse path),
  //     then the tags cache[0] and cache[1].
  static uint64_t cache[MAX_SIZE_ARRAYS]; // [0] cosmology, [1] Ntable tag
  static double* table;    // [N_M] ln sigma^2 on the dense ln M grid
  static double* lnMv;     // [N_M] the dense ln M nodes
  static double lim[3];    // ln M_min, ln M_max, dense spacing in ln M
  static int ncoarse = 0;  // active internal coarse mass nodes (0 = off)
  static double dlnc = 0.; // coarse grid spacing in ln M
  static double* lnMc = NULL;  // coarse ln M nodes
  static int* qidx = NULL;     // fine node -> coarse interval (uniform
  static double* qdel = NULL;  //   grids: precomputed, no search)
  static double* tabc = NULL;  // coarse ln sigma^2 values
  static double* cspl = NULL;  // natural-cubic-spline c coefficients
  static int nseg = 0;         // lobe cache: head + lobes
  static int* off = NULL;      // [nseg + 1] segment offsets into xs/wf
  static double* xs = NULL;    // [off[nseg]] quadrature nodes
  static double* wf = NULL;    // [off[nseg]] GL weight x 9 j1(x)^2

  // Ntable rebuild block. fdiff2(a, b) is plain uint64 inequality (1
  // when the two tags differ). Ntable.random is a tag that changes
  // whenever any Ntable setting changes; cache[1] holds the tag of the
  // build this table comes from. On the first call table is NULL, so
  // the block runs whatever the tags say.
  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    // Dense grid: N_M nodes uniform in ln M from ln M_min to ln M_max,
    // both endpoints included, hence N_M - 1 intervals. With the
    // defaults lim[2] = ln(1e17/1e6)/1023 = 25.328/1023 = 0.02476.
    if (table != NULL) free(table);
    table = (double*) malloc(sizeof(double)*Ntable.N_M);
    lim[0] = log(limits.halo_m_min);
    lim[1] = log(limits.halo_m_max);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_M - 1.0);
    if (lnMv != NULL) free(lnMv);
    lnMv = (double*) malloc(sizeof(double)*Ntable.N_M);
    for (int i=0; i<Ntable.N_M; i++) {
      lnMv[i] = lim[0] + i*lim[2];
    }

    // Coarse-grid workspace (why a coarse grid exists: item 9 of the
    // header and the refill block below). Every allocation lives in
    // this Ntable rebuild block; the per-cosmology refill only fills.
    // The buffers of the last build are released first, so a changed
    // N_M_internal can neither leak them nor reuse them at the wrong
    // size.
    if (lnMc != NULL) { free(lnMc); lnMc = NULL; }
    if (qidx != NULL) { free(qidx); qidx = NULL; }
    if (qdel != NULL) { free(qdel); qdel = NULL; }
    if (tabc != NULL) { free(tabc); tabc = NULL; }
    if (cspl != NULL) { free(cspl); cspl = NULL; }
    // The coarse grid is active only when it is a real grid (more than
    // 3 nodes, so the cubic spline has interior nodes to solve for)
    // and really coarser than the dense one; otherwise ncoarse = 0 and
    // the refill computes every dense node exactly.
    const int nc = Ntable.N_M_internal;
    ncoarse = (nc > 3 && nc < Ntable.N_M) ? nc : 0;
    if (ncoarse > 0) {
      // ncoarse nodes uniform in ln M over the same [lim[0], lim[1]]
      // as the dense grid; default 192 nodes, dlnc = 25.328/191 =
      // 0.13261 in ln M (about 5.4 dense spacings).
      dlnc = (lim[1] - lim[0]) / ((double) ncoarse - 1.0);
      lnMc = (double*) malloc(sizeof(double)*ncoarse);
      for (int i=0; i<ncoarse; i++) {
        lnMc[i] = lim[0] + i*dlnc;
      }
      // Fine-to-coarse map. Where does dense node i sit on the coarse
      // grid? Both grids are uniform in ln M and share both endpoints,
      // so the answer is arithmetic, no search:
      //
      //   dense node i  ->  ln M = lim[0] + i lim[2]
      //                 ->  r = i lim[2]/dlnc     (in coarse spacings)
      //                 ->  j = (int) r           (left node; the cast
      //                                           truncates toward 0)
      //                 ->  qdel = (r - j) dlnc   (offset from it, ln M)
      //
      // Example with the defaults, i = 100: r = 100 x 191/1023 =
      // 18.67, j = 18, qdel = 0.67 x 0.13261 = 0.0889 in ln M.
      //
      // Why the clamp: the spline evaluates on interval [j, j+1], so
      // the largest legal j is ncoarse - 2. At the top node
      // i = N_M - 1 the ratio r is ncoarse - 1 (exactly, or one ulp
      // off, since i lim[2] and (ncoarse - 1) dlnc are two roundings
      // of the same length), and (int) r names an interval that does
      // not exist. The clamp moves that node onto the last interval,
      // at qdel = dlnc, its right endpoint.
      qidx = (int*) malloc(sizeof(int)*Ntable.N_M);
      qdel = (double*) malloc(sizeof(double)*Ntable.N_M);
      for (int i=0; i<Ntable.N_M; i++) {
        const double r = (double) i * lim[2] / dlnc;
        int j = (int) r;
        if (j > ncoarse - 2) {
          j = ncoarse - 2;
        }
        qidx[i] = j;
        qdel[i] = (r - j) * dlnc; // offset from node j, in ln M
      }
      tabc = (double*) malloc(sizeof(double)*ncoarse);
      cspl = (double*) malloc(sizeof(double)*ncoarse);
    }

    // Node cache: the head segment plus NLOBE lobes, each with its own
    // Gauss-Legendre rule, all nodes flattened into xs[] with the
    // Bessel factor folded into wf[] (header, items 2-7). It depends
    // on hdi only, never on the cosmology or the mass: the per-mass
    // loop of the refill block reads it and computes nothing beyond
    // P_lin(x_q/R) times wf_q.
    //
    // The two ladders read "if hdi is 0 take the first size, if 1 the
    // second, ...", the last size covering every larger hdi:
    //
    //   hdi            0     1     2     >= 3
    //   npl (lobe)     8     12    16    20
    //   nph (head)     256   512   1024  1024
    //
    // All are sizes GSL stores as precomputed tables (header, item 3).
    const int NLOBE = 512;
    const int hdi = abs(Ntable.high_def_integration);
    const int npl = (0 == hdi) ? 8 :
                    (1 == hdi) ? 12 :
                    (2 == hdi) ? 16 : 20;         // per lobe
    const int nph = (0 == hdi) ? 256 :
                    (1 == hdi) ? 512 : 1024;      // head, GL in ln x
                                                  // (predefined GSL tables)
    const double XMIN = 1e-5;                     // head lower edge in x
    // Layout. Segment j owns the nodes q = off[j] .. off[j+1] - 1 of
    // xs/wf, and off[nseg] is the total node count. Segment 0 is the
    // head with nph nodes, segments 1..NLOBE the lobes with npl each:
    //
    //   off[0] = 0,  off[1] = nph,  off[j] = nph + (j - 1) npl,
    //   off[nseg] = nph + NLOBE npl = 256 + 512 x 8 = 4352 at hdi = 0.
    if (xs != NULL) {
      free(xs);
      free(wf);
      free(off);
    }
    nseg = 1 + NLOBE;
    off = (int*) malloc(sizeof(int)*(nseg + 1));
    const int ntot = nph + NLOBE*npl;
    xs = (double*) malloc(sizeof(double)*ntot);
    wf = (double*) malloc(sizeof(double)*ntot);

    // Two GL tables (nodes and weights on [-1, 1]): one of size nph for
    // the head, one of size npl shared by every lobe.
    gsl_integration_glfixed_table* th = malloc_gslint_glfixed(nph);
    gsl_integration_glfixed_table* tl = malloc_gslint_glfixed(npl);

    // One pass over the segments. Each iteration finds the segment's
    // right edge (a zero of j1), maps the GL rule onto the segment,
    // folds 9 j1^2 into the weights and advances the running count.
    double zlo = 0.0; // left edge of the current segment
    int q0 = 0;       // running node count
    for (int j = 0; j < nseg; j++) {
      // Right edge: the (j+1)-th zero of j1, found as in header item
      // 7. Start from q - 1/q with q = (n + 1/2) pi and n = j + 1 (for
      // j = 0: q = 4.7124, start 4.5002, exact zero 4.4934), then
      // polish with Newton steps z <- z - j1(z)/j1'(z), where
      // j1'(z) = j0(z) - 2 j1(z)/z. gsl_sf_bessel_j0_e and _j1_e store
      // the function value in the .val field of a gsl_sf_result (the
      // struct also carries an error estimate, unused here). The step
      // shrinks quadratically (6.8e-3, 1.0e-5, 2.4e-11 for j = 0) and
      // the loop leaves as soon as it is below 1e-14 of z.
      const double qq = ((double) j + 1.5)*M_PI; // (n + 1/2) pi, n = j+1
      double z = qq - 1.0/qq; // McMahon start
      for (int it = 0; it < 8; it++) { // Newton on j1
        gsl_sf_result J0, J1;
        gsl_sf_bessel_j0_e(z, &J0);
        gsl_sf_bessel_j1_e(z, &J1);
        const double step = J1.val/(J0.val - 2.0*J1.val/z);
        z -= step;
        if (fabs(step) < 1e-14*z) {
          break;
        }
      }
      // Rule and size for this segment: the head takes the large table
      // th, every lobe the small table tl.
      const int nj = (0 == j) ? nph : npl;
      gsl_integration_glfixed_table* tt = (0 == j) ? th : tl;
      off[j] = q0;
      for (int i = 0; i < nj; i++) {
        double xi, wi;
        if (0 == j) { // head: GL in s = ln x on [ln XMIN, ln z_1], dx = x ds
          // The rule is laid out in s = ln x over [ln XMIN, ln z_1]:
          // glfixed_point returns node s_i and weight w_i for that
          // interval. The node in x is e^{s_i}, and because dx = x ds
          // the weight for an integral over x is w_i times that x
          // (header, item 4).
          double si;
          gsl_integration_glfixed_point(log(XMIN), log(z), i, &si, &wi, tt);
          xi = exp(si);
          wi *= xi;
        } else {
          // Lobe: the rule is laid out directly in x over [z_j, z_{j+1}]
          // (zlo is the right edge of segment j - 1).
          gsl_integration_glfixed_point(zlo, z, i, &xi, &wi, tt);
        }
        // Fold the window into the weight: wf = w 9 j1(x)^2, the whole
        // cosmology-independent part of the integrand of header item 1.
        gsl_sf_result J1;
        gsl_sf_bessel_j1_e(xi, &J1);
        xs[q0] = xi;
        wf[q0] = wi*9.0*J1.val*J1.val;
        q0++;
      }
      zlo = z;
    }
    off[nseg] = q0; // total node count, also the end of the last lobe
    gsl_integration_glfixed_table_free(th);
    gsl_integration_glfixed_table_free(tl);
  }
  // Refill block: runs when the cosmology tag or the Ntable tag differs
  // from the one the table holds. set_linear_power_spectrum changes
  // cosmology.random whenever it installs a new P_lin, so a new P_lin
  // always lands here.
  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random)) {
    // Which masses get an exact sum. On the coarse path (ncoarse > 0)
    // the loop runs over the ncoarse coarse nodes lnMc and writes the
    // scratch tabc; the cubic upsampling below then fills table. On the
    // exact path it runs over all N_M dense nodes lnMv and writes table
    // directly. The three selectors let one loop serve both paths.
    const double* lnm = (ncoarse > 0) ? lnMc : lnMv;
    const int nm = (ncoarse > 0) ? ncoarse : Ntable.N_M;
    double* out = (ncoarse > 0) ? tabc : table;

    // The smoothed field (header, item 10): total matter, or cold dark
    // matter + baryons when like.halo_model[4] = HALO_FIELD_CB. The
    // field fixes the spectrum of the lobe sums (p_lin or p_lin_cb) and
    // the mean density omega_field rho_crit of the Lagrangian radius.
    const int use_cb = (HALO_FIELD_CB == like.halo_model[4]);
    if (use_cb && NULL == cosmology.lnPL_cb) {
      log_fatal("sigma2: like.halo_model[4] = HALO_FIELD_CB needs the "
                "linear P_cb table (set_linear_power_spectrum_cb, after "
                "set_linear_power_spectrum)");
      exit(1);
    }
    const double omega_field = omega_halo_field();

    const double EPS = 1e-7; // relative tail tolerance of the lobe sum
    // restrict copies of the node cache. restrict is a promise to the
    // compiler that, while these pointers are in scope, the memory they
    // point to is reached only through them; the store to out[m] can
    // then not have changed xq[q], wq[q] or oq[j], and the compiler
    // need not reload them after every store. The promise is honored
    // only on accesses made through the qualified pointer: the loop
    // body must index xq/wq/oq, not the statics xs/wf/off, for it to
    // take effect.
    const double* restrict xq = xs;
    const double* restrict wq = wf;
    const int* restrict oq = off;
    // One thread per chunk of masses. schedule(static) splits the m
    // range into contiguous chunks whose bounds depend on the thread
    // count alone, and each mass's sum is a serial loop inside one
    // thread over the same nodes in the same order every time: the
    // floating-point result for a given mass is bit-identical from run
    // to run and independent of the thread count. Nothing is reduced
    // across threads.
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < nm; m++) {
      // R from M through M = (4 pi/3) R^3 rho_crit Omega (header,
      // item 1; Omega = omega_field, Omega_m for total matter);
      // 0.75/pi is 3/(4 pi). cosmology.rho_crit = 7.4775e21 is
      // the critical density in M_sun/h per (c/H0)^3, so R comes out in
      // c/H0 units and k = x/R in (c/H0)^-1 units, the units p_lin
      // expects (the unit conventions at the top of this file). At
      // M = 1e6 with Omega_m = 0.3: R = 4.74e-6 c/H0 = 0.0142 Mpc/h.
      const double Mm = exp(lnm[m]);
      const double R =
          pow(0.75*Mm/(M_PI*cosmology.rho_crit*omega_field), 1./3.);
      const double invR = 1.0/R;

      // Segment sums s_j in order, head first (header, item 8). Per
      // node one P_lin read and one multiply-add: p_lin(k, 1.0) is the
      // linear spectrum at k = x_q/R and at scale factor a = 1 (the
      // second argument); the Bessel factor already sits in wq[q].
      // p_lin_cb reads the P_cb table with p_lin's arithmetic.
      double total = 0.0;
      double sprev = 0.0;
      for (int j = 0; j < nseg; j++) {
        double s = 0.0;
        if (use_cb) {
          for (int q = oq[j]; q < oq[j+1]; q++) {
            s += wq[q]*p_lin_cb(xq[q]*invR, 1.0);
          }
        }
        else {
          for (int q = oq[j]; q < oq[j+1]; q++) {
            s += wq[q]*p_lin(xq[q]*invR, 1.0);
          }
        }
        total += s;
        // Stopping rule (header, item 8): with r = s_j/s_{j-1} the
        // geometric estimate of everything not yet summed is
        // s_j r/(1 - r); leave once it is below EPS of the total.
        // j = 0 is the head and j = 1 the first lobe: the ratio test
        // needs two consecutive lobes, so it starts at j = 2 (the head
        // is a different kind of segment and never enters a ratio).
        // The guards sprev > 0 and s < sprev keep the formula
        // meaningful: r must lie in (0, 1) for the series to converge.
        if (j > 1 && sprev > 0.0 && s < sprev) {
          const double r = s/sprev;
          if (s*r/(1.0 - r) < EPS*total) {
            break;
          }
        }
        sprev = s;
      }
      // Normalization 1/(2 pi^2 R^3) of header item 1. R^3 in (c/H0)^3
      // cancels the (c/H0)^3 of P_lin, leaving sigma^2 dimensionless.
      out[m] = total/(R*R*R*2.0*M_PI*M_PI);
    }

    if (ncoarse > 0) {
      // The spline runs through ln sigma^2, not sigma^2: sigma^2 spans
      // orders of magnitude over the mass range while its log is close
      // to a straight line in ln M, the friendliest shape for a cubic.
      // The table stores the log on both paths.
      for (int i=0; i<ncoarse; i++) {
        tabc[i] = log(tabc[i]);
      }
      // Upsampling with the house natural cubic spline. A cubic spline
      // is a chain of cubic polynomials, one per interval between
      // nodes, joined so that value, first and second derivative are
      // continuous at every node; "natural" adds S'' = 0 at both ends.
      // On interval [x_j, x_j + h] the piece is
      //
      //   S(x_j + t) = y_j + b t + c_j t^2 + d t^3,     0 <= t <= h,
      //
      // where c_j = S''(x_j)/2 is what spline_coeffs_uniform returns (a
      // tridiagonal solve, see basics.c) with c_0 = c_{n-1} = 0. The
      // other two coefficients follow from two conditions:
      //
      //   S'' runs linearly from 2 c_j to 2 c_{j+1} across the interval
      //     ->  d = (c_{j+1} - c_j)/(3 h)
      //   S hits the right node, S(x_{j+1}) = y_{j+1}
      //     ->  b = (y_{j+1} - y_j)/h - h (c_{j+1} + 2 c_j)/3
      //
      // Tiny check with three nodes y = (0, 1, 0) and h = 1: the solve
      // gives c = (0, -1.5, 0); on the first interval b = 1.5 and
      // d = -0.5, so S(1) = 0 + 1.5 - 0.5 = 1 reproduces the middle
      // node and S(0.5) = 0.6875. The polynomial is evaluated in Horner
      // form, y + t (b + t (c + t d)), at the precomputed offset qdel[i]
      // from node qidx[i] of every dense node.
      spline_coeffs_uniform(tabc, ncoarse, dlnc, cspl);
      const double hc = dlnc;
      const double inv_hc = 1.0/dlnc;
      #pragma omp parallel for schedule(static)
      for (int i=0; i<Ntable.N_M; i++) {
        const int j = qidx[i];
        const double b = (tabc[j+1] - tabc[j])*inv_hc
                         - hc*(cspl[j+1] + 2.0*cspl[j])/3.0;
        const double d = (cspl[j+1] - cspl[j])/(3.0*hc);
        table[i] = tabc[j] + qdel[i]*(b + qdel[i]*(cspl[j] + qdel[i]*d));
      }
    }
    else {
      // Exact path: every dense node holds its own sum; store the log.
      for (int i=0; i<Ntable.N_M; i++) {
        table[i] = log(table[i]);
      }
    }
    // Record the tags the table now corresponds to; the next call
    // compares against them.
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
  // Read-out. interpol1d(f, n, a, b, dx, x) is the house linear
  // interpolation on a uniform grid: with r = (x - a)/dx and
  // i = floor(r) it returns f[i] + (r - i) (f[i+1] - f[i]); below a it
  // returns f[0], and at or beyond the last node f[n-1] (constant
  // extrapolation, so a mass outside [halo_m_min, halo_m_max] gets the
  // edge value). Here f = table (ln sigma^2), a = lim[0], dx = lim[2],
  // x = ln M; b = lim[1] is accepted for symmetry and unused. Example
  // with the defaults, M = 3e10: r = ln(3e4)/0.02476 = 416.37, so the
  // value is read 37% of the way from node 416 to node 417. exp undoes
  // the stored log.
  return exp(interpol1d(table, Ntable.N_M, lim[0], lim[1], lim[2], log(M)));
}
