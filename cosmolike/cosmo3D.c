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
//     set_non_linear_power_spectrum -> cosmology.lnP  (same layout)
//   -> the lookup functions below (chi_all, norm_growfac*, f_growth, p_lin,
//      p_nonlin, ...) interpolate those tables
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
// Cached sigma^2(M) at a = 1, the variance of the linear density field in a
// top hat holding mass M: a table of ln sigma^2 on Ntable.N_M nodes uniform
// in ln M over [ln limits.halo_m_min, ln limits.halo_m_max], read back with
// linear interpol1d in ln M (exp of the stored log). The halo model rescales
// by the growth factor where it needs sigma at a < 1 (e.g.
// nu = delta_c/(sqrt(sigma2(m))*growfac(a)) in halo.c).
//
// The integral. With x = kR,
//
//   sigma^2(M) = 1/(2 pi^2 R^3) int_0^inf P_lin(x/R) 9 j1(x)^2 dx,
//   R(M)       = (3M/(4 pi rho_crit Omega_m))^(1/3),
//
// a smooth spectrum modulated by the oscillating top-hat window 3 j1(x)/x.
// The window's zeros - the roots z_n of tan x = x (4.4934, 7.7253, 10.9041,
// 14.0662, ..., approaching (n + 1/2) pi) - split the x axis into a head
// segment [0, z_1] (the main bump, W(0) = 1) and lobes [z_n, z_{n+1}], each
// ONE smooth bump. A low-order Gauss-Legendre rule per lobe is therefore
// exponentially accurate: the oscillation lives in the lobe structure,
// never inside a lobe.
//
// The head segment holds nearly all of sigma^2, and for halo-scale R it
// spans k = x/R from ~0 to z_1/R (~300 h/Mpc at M = 1e6 M_sun/h): the
// turnover and the BAO of P_lin sit at x << 1, where a rule uniform in x
// places no nodes. The head is therefore integrated in s = ln x over
// [XMIN, z_1], XMIN = 1e-5; below XMIN the integrand falls as x^(3 + n_s)
// and the truncation is below 1e-10.
//
// Node cache (built in the Ntable rebuild block, mass-independent - the
// cosmo_nodes idea of cosmo2D.c):
//
//   xs[q]   = the quadrature nodes of every segment, flattened
//   wf[q]   = GL weight x 9 j1(x_q)^2 (the Bessel function never appears
//             in the per-mass loop)
//   off[j]  = first node of segment j; off[nseg] = total node count
//
// Segment sizes ladder with hdi = abs(Ntable.high_def_integration), in
// sizes GSL tabulates: the head takes 256/512/1024 nodes at hdi =
// 0/1/>=2 (measured against a dense reference at hdi = 0: 3.8e-6 max
// error and 2.2e-6 scatter between neighboring masses, the scatter being
// what the finite difference of halo.c's dlognudlogm sees); each lobe
// takes 8 + 4 hdi nodes, capped at 20 (with 8, all lobes together are
// off by < 1e-9 of sigma^2). NLOBE = 512 lobes reach x ~ 1600, where the
// integrand envelope (P ~ k^-3 times 9 cos^2(x)/x^4 in
// the substituted variable) leaves a tail far below the stopping tolerance
// for any halo-scale R. The zeros of j1 solve tan x = x: McMahon's start
// z ~ q - 1/q with q = (n + 1/2) pi lands inside the right branch, and a
// few Newton steps on j1 itself - with j1'(x) = j0(x) - 2 j1(x)/x - polish
// it to machine precision.
//
// The lobe sum, per mass (threaded over masses in the refill block):
// segment sums s_j decay along a power-law envelope, so with
// r = s_j/s_{j-1} < 1 the remainder is bounded by the near-geometric
// estimate s_j r/(1 - r), and the loop exits when that bound falls below
// EPS = 1e-7 of the running total - an explicit, documented stopping
// condition. Per (mass, node) only one p_lin read and one multiply-add
// remain; each mass reads just the node prefix its own convergence needs.
// Deterministic by construction: fixed nodes, independent per-mass sums,
// no cross-thread reductions.
//
// Coarse grid: when Ntable.N_M_internal is active, the exact lobe sums run
// only on the coarse ln M nodes and the house natural cubic spline
// upsamples ln sigma^2 onto the unchanged N_M table (see the refill
// block).
//
// Cache invalidation:
//   allocation, ln M limits and the lobe-node cache: rebuilt when
//   Ntable.random changes (the hdi ladder enters the node counts)
//   table refill: cosmology.random (cache[0]) or Ntable.random (cache[1])
//
// Parameters:
//   M - halo mass in M_sun/h
//
// Returns:
//   sigma^2(M) at a = 1, interpolated in ln M
// ---------------------------------------------------------------------------
double sigma2(
    const double M  // halo mass in M_sun/h
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* table;
  static double* lnMv;
  static double lim[3];
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

  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
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

    // Coarse-grid workspace (the strategy note sits in the refill
    // below): every allocation lives HERE, in the Ntable rebuild
    // block; the per-cosmology refill only fills. qidx/qdel map each
    // fine ln M node onto its coarse interval - both grids are
    // uniform in ln M with shared endpoints, so the map is one
    // multiply plus a cast, no search, with the top endpoint clamped
    // onto the last interval against a 1-ulp overshoot.
    if (lnMc != NULL) { free(lnMc); lnMc = NULL; }
    if (qidx != NULL) { free(qidx); qidx = NULL; }
    if (qdel != NULL) { free(qdel); qdel = NULL; }
    if (tabc != NULL) { free(tabc); tabc = NULL; }
    if (cspl != NULL) { free(cspl); cspl = NULL; }
    const int nc = Ntable.N_M_internal;
    ncoarse = (nc > 3 && nc < Ntable.N_M) ? nc : 0;
    if (ncoarse > 0) {
      dlnc = (lim[1] - lim[0]) / ((double) ncoarse - 1.0);
      lnMc = (double*) malloc(sizeof(double)*ncoarse);
      for (int i=0; i<ncoarse; i++) {
        lnMc[i] = lim[0] + i*dlnc;
      }
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

    // Lobe-node cache (see the header): head segment + NLOBE lobes,
    // each with its own Gauss-Legendre rule, Bessel factor folded into
    // the weights.
    const int NLOBE = 512;
    const int hdi = abs(Ntable.high_def_integration);
    const int npl = 8 + 4*((hdi < 3) ? hdi : 3);  // per lobe: 8/12/16/20
    const int nph = (0 == hdi) ? 256 :
                    (1 == hdi) ? 512 : 1024;      // head, GL in ln x
                                                  // (predefined GSL tables)
    const double XMIN = 1e-5;                     // head lower edge in x
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

    gsl_integration_glfixed_table* th = malloc_gslint_glfixed(nph);
    gsl_integration_glfixed_table* tl = malloc_gslint_glfixed(npl);

    double zlo = 0.0; // left edge of the current segment
    int q0 = 0;       // running node count
    for (int j = 0; j < nseg; j++) {
      // right edge: the (j+1)-th zero of j1
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
      const int nj = (0 == j) ? nph : npl;
      gsl_integration_glfixed_table* tt = (0 == j) ? th : tl;
      off[j] = q0;
      for (int i = 0; i < nj; i++) {
        double xi, wi;
        if (0 == j) { // head: GL in s = ln x on [ln XMIN, ln z_1], dx = x ds
          double si;
          gsl_integration_glfixed_point(log(XMIN), log(z), i, &si, &wi, tt);
          xi = exp(si);
          wi *= xi;
        } else {
          gsl_integration_glfixed_point(zlo, z, i, &xi, &wi, tt);
        }
        gsl_sf_result J1;
        gsl_sf_bessel_j1_e(xi, &J1);
        xs[q0] = xi;
        wf[q0] = wi*9.0*J1.val*J1.val;
        q0++;
      }
      zlo = z;
    }
    off[nseg] = q0;
    gsl_integration_glfixed_table_free(th);
    gsl_integration_glfixed_table_free(tl);
  }
  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random)) {
    // Lobe-summed fill (see the header): Bessel factors precomputed once
    // for all masses, one threaded loop over the masses this pass needs.
    //
    // When Ntable.N_M_internal is active, the exact lobe sums run
    // only on the coarse ln M nodes and the house natural cubic
    // spline upsamples ln sigma^2 onto the unchanged N_M table:
    // ln sigma^2(ln M) is smooth and monotone, the ideal case for
    // the coarse-exact + cubic-upsample pattern of cosmo2D.c, and
    // the consumer's linear interpol1d reads stay untouched.
    const double* lnm = (ncoarse > 0) ? lnMc : lnMv;
    const int nm = (ncoarse > 0) ? ncoarse : Ntable.N_M;
    double* out = (ncoarse > 0) ? tabc : table;

    const double EPS = 1e-7; // relative tail tolerance of the lobe sum
    // local restrict copies: the loop writes out[], which the compiler
    // could not otherwise rule out as aliasing the static node arrays
    const double* restrict xq = xs;
    const double* restrict wq = wf;
    const int* restrict oq = off;
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < nm; m++) {
      const double Mm = exp(lnm[m]);
      const double R =
          pow(0.75*Mm/(M_PI*cosmology.rho_crit*cosmology.Omega_m), 1./3.);
      const double invR = 1.0/R;

      double total = 0.0;
      double sprev = 0.0;
      for (int j = 0; j < nseg; j++) {
        double s = 0.0;
        for (int q = oq[j]; q < oq[j+1]; q++) {
          s += wq[q]*p_lin(xq[q]*invR, 1.0);
        }
        total += s;
        // j = 0 is the head, j = 1 the first lobe: the ratio test
        // needs two consecutive LOBES, so it starts at j = 2
        if (j > 1 && sprev > 0.0 && s < sprev) {
          const double r = s/sprev;
          if (s*r/(1.0 - r) < EPS*total) {
            break;
          }
        }
        sprev = s;
      }
      out[m] = total/(R*R*R*2.0*M_PI*M_PI);
    }

    if (ncoarse > 0) {
      for (int i=0; i<ncoarse; i++) {
        tabc[i] = log(tabc[i]);
      }
      // Upsampling: the house cubic (spline_coeffs_uniform gives c =
      // S''/2 with natural boundaries; interpolating the right node
      // fixes b, the linear run of S'' fixes d), evaluated in Horner
      // form at the precomputed offsets qdel from node qidx.
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
      for (int i=0; i<Ntable.N_M; i++) {
        table[i] = log(table[i]);
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
  return exp(interpol1d(table, Ntable.N_M, lim[0], lim[1], lim[2], log(M)));
}
