#include <assert.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_interp2d.h>
#include <gsl/gsl_odeiv.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_sf.h>
#include <math.h>
#include <omp.h>
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
// Grid contract: redshift axes consist of at most MAX_GRID_SEGMENTS
// uniform segments; the power spectra also have a uniform log10 k axis.
// The setters check this structure and record each segment's start and
// inverse spacing. All builds use that metadata for direct indexing.
// A caller must install tables through those setters so the grid and
// its metadata remain consistent. The inverse distance reader a_chi
// uses its separate bucket index because chi itself is not uniform.
// ---------------------------------------------------------------------------

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
// A direct-index lookup uses the piecewise-uniform z grid metadata
// populated by set_distances. Clamp j so the j+2 read of the "up" slope
// stays within the table.
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
// Bucket index of the chi column for a_chi.
//
// Why it exists: a_chi runs millions of times per likelihood evaluation
// (the two-radius RSD kernel of the Limber C_gg calls it twice per lens
// bin, multipole and node), and its binary search, with branches the
// CPU cannot predict, made it ~8% of all cycles of a des_cluster 6x2pt+N
// evaluation (perf, amypond, v5.00). The chi column is not uniform, so
// the direct-index trick of the z axes does not apply; instead the chi
// range is cut into equal buckets, and each bucket stores the bracket of
// its lower edge:
//
//   chi_bucket[b] = largest j <= chi_nz-2 with chi_j <= chi_min + b*dx
//
// a_chi then starts at chi_bucket[b] and walks the few nodes to the
// bracket (see the exactness note there). Two buckets per node keep the
// walk at a step or two.
//
// A chi column that is not strictly increasing has no unique bracket (and
// a zero-width interval would divide by zero in a_chi): abort.
//
// Cache invalidation:
// set_distances calls this after every refill of cosmology.chi (always
// serially, from Python, before any parallel region reads a_chi).
//
// Returns:
//   nothing; fills the chi_bucket fields of cosmology
// ---------------------------------------------------------------------------
void set_chi_bucket_index(void)
{
  free(cosmology.chi_bucket);
  cosmology.chi_bucket = NULL;
  cosmology.chi_nbucket = 0;
  const int nz = cosmology.chi_nz;
  const double* x = cosmology.chi[1];
  for (int j=0; j<nz-1; j++) {
    if (!(x[j] < x[j+1])) {
      log_fatal("chi(z) is not strictly increasing at z = %g",
                cosmology.chi[0][j]);
      exit(1);
    }
  }
  const int nb = 2*nz;
  int* idx = (int*) malloc(nb*sizeof(int));
  if (NULL == idx) {
    log_fatal("array allocation failed"); exit(1);
  }
  const double dx = (x[nz-1] - x[0])/((double) nb);
  int j = 0;
  for (int b=0; b<nb; b++) {
    const double lo = x[0] + b*dx;
    while (j < nz-2 && x[j+1] <= lo) {
      j++;
    }
    idx[b] = j;
  }
  cosmology.chi_bucket = idx;
  cosmology.chi_bucket_min = x[0];
  cosmology.chi_bucket_inv_dx = 1.0/dx;
  cosmology.chi_nbucket = nb;
}

// ---------------------------------------------------------------------------
// Inverse distance lookup a(chi): the bracket j of chi on the chi column
// of cosmology.chi, then linear inverse interpolation of z in it,
//
//   z = z_j + dy*(z_{j+1} - z_j),  dy = (chi - chi_j)/(chi_{j+1} - chi_j),
//
// and a = 1/(1+z). The input is converted from c/H0 units to the table's
// Mpc/h (io_chi * coverH0).
//
// The bracket comes from the bucket index (set_chi_bucket_index) and is
// exactly the one of the binary search a_chi used before: inside
// [chi_0, chi_{nz-1}) the unique j with chi_j <= chi < chi_{j+1}; 0 below
// chi_0; nz-2 at or above chi_{nz-1}, and for NaN (the binary search never
// moved ihi there). The interpolation is unchanged, so a_chi is bitwise
// the binary-search version.
//
// Cache invalidation:
// no static state; the table and its bucket index are maintained by
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

  const double* restrict x = cosmology.chi[1];
  const int nz = cosmology.chi_nz;
  int j = 0;
  if (chi >= x[0] && chi < x[nz-1]) // false for NaN
  {
    int b = (int) ((chi - cosmology.chi_bucket_min)*
                   cosmology.chi_bucket_inv_dx);
    if (b > cosmology.chi_nbucket - 1) {
      b = cosmology.chi_nbucket - 1;
    }
    j = cosmology.chi_bucket[b];
    // b can be one bucket off where chi sits on a bucket edge (the
    // product above rounds), so walk both ways to the bracket
    while (j > 0 && x[j] > chi) {
      j--;
    }
    while (j < nz-2 && x[j+1] <= chi) {
      j++;
    }
  }
  else {
    j = (chi < x[0]) ? 0 : nz-2;
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
// G(0) comes from bracket j = 0 because the grid starts at z = 0.
// The query-z bracket comes from a direct-index lookup on the
// piecewise-uniform z grid (cosmology.G_z_* metadata).
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
// z = 0 lookup is needed. The bracket is found by direct indexing on
// the piecewise-uniform z grid, as in norm_growfac.
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

// ---------------------------------------------------------------------------
// Growth factor D(a) and growth rate f(a) fused in one bracket lookup:
// the formulas of norm_growfac and f_growth (see those headers) evaluated
// from a single interpolation of the cosmology.G table at z = 1/a - 1,
// plus the j = 0 lookup of the G(0) normalization.
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
// Bracket selection uses direct indexing: one uniform segment in
// log10 k and piecewise-uniform z segments, via cosmology.lnPL_* metadata.
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



// ---------------------------------------------------------------------------
// Linear power spectrum of cold dark matter plus baryons, P_cb(k, a): the
// matter field without the massive neutrinos, which free-stream out of
// halos. Its evolving variance is used by every halo statistic.
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



// ---------------------------------------------------------------------------
// The cold matter density sets the halo mass-radius relation and the
// rho/M factor in the mass function. Free-streaming neutrinos contribute
// to the background and to lensing, but not to the mass in these halos.
// ---------------------------------------------------------------------------
double omega_halo_field(void)
{
  return cosmology.Omega_m - cosmology.Omega_nu;
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
// direct-index lookups as p_lin (see p_lin), reading the
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

// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------
// ----------------------------------------------------------------------

// ---------------------------------------------------------------------------
// P_lin at ONE scale factor and n wavenumbers: out[m] = p_lin(k[m], a).
//
// Why it exists: the Limber fills evaluate P on a (node, multipole) grid,
// and a node fixes a. p_lin recomputes the z half of its bilinear read
// (z = 1/a - 1, the z bracket j, the weight dy) at every call; here that
// half runs once per node and only the k half (log10 k, its bracket i,
// dx, the four table reads, exp) runs per wavenumber.
//
// Every out[m] is bitwise p_lin(k[m], a): the k half and the bilinear
// combination are p_lin's expressions verbatim, and the hoisted z half
// computes the same values from the same a.
//
// Parameters:
//   a   - scale factor
//   k   - wavenumbers in (c/H0)^-1 units, length n
//   n   - number of wavenumbers
//   out - output, length n
//
// Returns:
//   nothing; P_lin(k[m], a) in (c/H0)^3 units into out[m]
// ---------------------------------------------------------------------------
void p_lin_at_a(const double a, const double* k, const int n, double* out)
{
  const double z = 1.0 / a - 1.0;
  const int j = piecewise_index(z, cosmology.lnPL_z_nseg,
                                cosmology.lnPL_z_seg_start, cosmology.lnPL_z_seg_len,
                                cosmology.lnPL_z_seg_xmin,  cosmology.lnPL_z_seg_inv_dx,
                                cosmology.lnPL_nz);
  const double zj  = cosmology.lnPL[cosmology.lnPL_nk][j  ];
  const double zj1 = cosmology.lnPL[cosmology.lnPL_nk][j+1];
  const double dy = (z      - zj) / (zj1 - zj);
  for (int m=0; m<n; m++) {
    const double log10k = log10(k[m] / cosmology.coverH0);
    int i = (int)((log10k - cosmology.lnPL_log10k_min) * cosmology.lnPL_log10k_inv_dx);
    if (i < 0)                       i = 0;
    if (i > cosmology.lnPL_nk - 2)   i = cosmology.lnPL_nk - 2;
    const double xi  = cosmology.lnPL[i  ][cosmology.lnPL_nz];
    const double xi1 = cosmology.lnPL[i+1][cosmology.lnPL_nz];
    const double dx = (log10k - xi) / (xi1 - xi);
    const double out_lnP =   (1-dx)*(1-dy) * cosmology.lnPL[i  ][j  ]
                           + (1-dx)*   dy  * cosmology.lnPL[i  ][j+1]
                           +    dx *(1-dy) * cosmology.lnPL[i+1][j  ]
                           +    dx *   dy  * cosmology.lnPL[i+1][j+1];
    out[m] = exp(out_lnP) / (cosmology.coverH0 * cosmology.coverH0 * cosmology.coverH0);
  }
}

// ---------------------------------------------------------------------------
// P_nl at ONE scale factor and n wavenumbers: out[m] = p_nonlin(k[m], a).
// The z half of the bilinear read runs once, the k half per wavenumber,
// with p_nonlin's expressions verbatim: every out[m] is bitwise
// p_nonlin(k[m], a) (see p_lin_at_a for the reasoning).
//
// Parameters:
//   a   - scale factor
//   k   - wavenumbers in (c/H0)^-1 units, length n
//   n   - number of wavenumbers
//   out - output, length n
//
// Returns:
//   nothing; P_nl(k[m], a) in (c/H0)^3 units (times the baryonic ratio
//   when enabled) into out[m]
// ---------------------------------------------------------------------------
void p_nonlin_at_a(const double a, const double* k, const int n, double* out)
{
  const double coverH0 = cosmology.coverH0;
  const double z      = 1.0 / a - 1.0;
  const int j = piecewise_index(z, cosmology.lnP_z_nseg,
                                cosmology.lnP_z_seg_start, cosmology.lnP_z_seg_len,
                                cosmology.lnP_z_seg_xmin,  cosmology.lnP_z_seg_inv_dx,
                                cosmology.lnP_nz);
  const double zj  = cosmology.lnP[cosmology.lnP_nk][j  ];
  const double zj1 = cosmology.lnP[cosmology.lnP_nk][j+1];
  const double dy = (z      - zj) / (zj1 - zj);
  for (int m=0; m<n; m++) {
    const double log10k = log10(k[m] / coverH0);
    int i = (int)((log10k - cosmology.lnP_log10k_min) * cosmology.lnP_log10k_inv_dx);
    if (i < 0)                     i = 0;
    if (i > cosmology.lnP_nk - 2)  i = cosmology.lnP_nk - 2;
    const double xi  = cosmology.lnP[i  ][cosmology.lnP_nz];
    const double xi1 = cosmology.lnP[i+1][cosmology.lnP_nz];
    const double dx = (log10k - xi) / (xi1 - xi);
    const double out_lnP =   (1-dx)*(1-dy) * cosmology.lnP[i  ][j  ]
                           + (1-dx)*   dy  * cosmology.lnP[i  ][j+1]
                           +    dx *(1-dy) * cosmology.lnP[i+1][j  ]
                           +    dx *   dy  * cosmology.lnP[i+1][j+1];
    const double ans = exp(out_lnP) / (coverH0 * coverH0 * coverH0);
    out[m] = (bary.is_Pk_bary == 1) ? ans * PkRatio_baryons(k[m], a) : ans;
  }
}

// ---------------------------------------------------------------------------
// The dispatch of Pdelta, shared with Pdelta_at_a so the two can never
// disagree: 3 (p_lin) is latched the first time pdeltaparams.runmode reads
// "linear"; -1 means p_nonlin and is checked again on every call (the
// latch only ever moves to 3).
// ---------------------------------------------------------------------------
static int pdelta_type = -1;

static inline int pdelta_dispatch(void)
{
  if (pdelta_type == -1)
  {
    if (strcmp(pdeltaparams.runmode,"linear") == 0)
    {
      pdelta_type = 3;
    }
  }
  return pdelta_type;
}

// ---------------------------------------------------------------------------
// Matter power spectrum dispatch: p_lin when pdeltaparams.runmode is
// "linear", p_nonlin otherwise. The linear choice is latched in a static
// (pdelta_dispatch, shared with Pdelta_at_a) the first time the run mode
// reads "linear" and reused for the rest of the process, so the run mode
// must be set before the first evaluation.
//
// Cache invalidation:
// none. The static latch never rebuilds; only
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
  const int P_type = pdelta_dispatch();
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

// ---------------------------------------------------------------------------
// Pdelta at ONE scale factor and n wavenumbers: out[m] = Pdelta(k[m], a),
// bitwise (Pdelta's dispatch, then p_lin_at_a or p_nonlin_at_a). The
// Limber fills call it once per quadrature node with the node's Limber
// wavenumbers k = (l + 1/2)/f_K of every multipole.
//
// Parameters:
//   a   - scale factor
//   k   - wavenumbers in (c/H0)^-1 units, length n
//   n   - number of wavenumbers
//   out - output, length n
//
// Returns:
//   nothing; P(k[m], a) in (c/H0)^3 units into out[m]
// ---------------------------------------------------------------------------
void Pdelta_at_a(const double a, const double* k, const int n, double* out)
{
  switch (pdelta_dispatch())
  {
    case 3:
      p_lin_at_a(a, k, n, out);
      break;
    default:
      p_nonlin_at_a(a, k, n, out);
      break;
  }
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
// ============================================================================
// [SECTION] MASS VARIANCE OF TOTAL MATTER AND COLD MATTER
// ============================================================================
//
// PHYSICAL DERIVATION & LOGIC FLOW
//
// 1. Associate a smoothing radius with the halo mass.
//
//    Imagine spreading the halo's mass uniformly at the mean cosmic
//    density of the chosen field. The radius of that sphere is R:
//
//      M = (4 pi/3) rho_field R^3,
//      R = [3 M / (4 pi rho_field)]^(1/3).
//
//    Total matter uses rho_field = rho_crit Omega_m. Cold dark matter
//    plus baryons (cb) uses rho_field = rho_crit (Omega_m - Omega_nu).
//    R is a Lagrangian smoothing radius, not the smaller radius r_200m
//    of the collapsed halo used in its NFW profile.
//
// 2. Smooth the linear density field and calculate its variance.
//
//    A uniform sphere in real space has the Fourier-space window
//
//      W(x) = 3 (sin x - x cos x) / x^3,       x = k R.
//
//    Squaring the smoothed density and averaging gives
//
//      sigma^2(R,a) = integral dlnk Delta^2(k,a) W(kR)^2,
//      Delta^2(k,a) = k^3 P(k,a) / (2 pi^2).
//
//    Delta^2 is the variance per logarithmic interval in wavenumber.
//    For cb we integrate P_cb at this a, rather than growing a z=0
//    variance with a single growth factor. Massive neutrinos make the
//    growth depend on k, hence on the mass scale being smoothed.
//
// 3. Replace a separate integral at every R by a Fourier transform.
//
//    Put u = ln k. Divide Delta^2 by a smooth power law k^b before
//    taking its Fourier transform:
//
//      g(u,a) = exp(-b u) Delta^2(exp(u),a).
//
//    This division reduces the difference between the two ends of the
//    finite input interval. The number b is called the FFTLog bias;
//    it is a numerical choice, unrelated to the physical halo bias.
//    The power k^b is restored analytically below.
//
//    On an evenly spaced u grid the FFT writes g as a sum of modes
//    exp(i eta u). Restoring k^b turns each mode into k^s, where
//
//      s = b + i eta.
//
//    The variance integral of this one mode can be done analytically.
//    Substituting x = kR gives dlnk = dx/x and k^s = x^s R^(-s):
//
//      integral dlnk k^s W(kR)^2
//        = R^(-s) integral_0^infinity dx x^(s-1) W(x)^2
//        = R^(-s) Mellin(s).
//
//    The last integral defines the Mellin transform of W^2. Its closed
//    form is a ratio of Gamma functions (valid for 0 < Re(s) < 4):
//
//                      (9 pi/2) Gamma(4-s) Gamma(s/2)
//      Mellin(s) = ------------------------------------------------- .
//                  2^(4-s) Gamma((5-s)/2)^2 Gamma(4-s/2)
//
//    Thus we need not integrate each oscillatory mode numerically.
//    Multiply each FFT coefficient by Mellin(s), then sum the modes
//    at every lnR with an inverse FFT. The factor R^(-b) restores the
//    real power removed before the forward transform.
//
// 4. Obtain the mass slope from the same Fourier coefficients.
//
//    The only R dependence of a mode is R^(-s). Therefore
//
//      d[R^(-s)]/dlnR = -s R^(-s).
//
//    A second inverse FFT, with Mellin(s) replaced by -s Mellin(s),
//    gives d sigma^2/dlnR. No finite difference in mass is required.
//    Since M is proportional to R^3 and sigma is sqrt(sigma^2),
//
//      dln sigma/dlnM = (1 / (6 sigma^2)) d sigma^2/dlnR.
//
// 5. Store the result on the grids used by the halo integrals.
//
//    FFTLog returns values uniformly spaced in lnR. A cubic spline
//    evaluates them at the desired lnM nodes. Repeat this calculation
//    for each scale factor and each field. The public readers then
//    interpolate these tables in lnM and a.
//
// Units throughout this calculation:
//   k: h/Mpc; R: Mpc/h; M: M_sun/h; P: (Mpc/h)^3.
//   The variance and its logarithmic slope are dimensionless.
//
// The work arrays below live between calls. Arrays of the same shape
// share one allocation, with a documented extra index for their role.
// All rows use the aligned allocators in basics.c, including the FFTW
// complex rows. Each thread writes only its own work rows.
// ---------------------------------------------------------------------------
static struct {
  uint64_t cosmology_tag; // cosmology generation represented by the tables
  uint64_t ntable_tag;    // numerical-settings generation used for the grids

  int nfft;              // transform length, including appended zeros
  int ninput;            // k nodes containing the continued power spectrum
  int nradial;           // radius nodes used by the spline, including end margins
  int nthreads;          // number of threads with allocated work buffers
  int nmass;             // mass nodes in each output row
  int na;                // scale-factor nodes in each output table
  int nk;                // k nodes in the supplied power-spectrum table

  double lnk0;           // natural logarithm of the first supplied k in h/Mpc
  double dlnk;           // spacing in lnk, also the FFT output spacing in lnR
  double lnm0;           // natural logarithm of the minimum halo mass in M_sun/h
  double lnm1;           // natural logarithm of the maximum halo mass in M_sun/h
  double dlnm;           // spacing between output mass nodes in lnM
  double amin;           // smallest tabulated scale factor
  double da;             // spacing between scale-factor nodes

  // [quantity][field][a][mass]: quantity 0 = ln(sigma^2/a^2),
  // quantity 1 = dln sigma/dlnM; field 0 = matter, field 1 = cb.
  double**** table;      // cached variance and mass slope for both fields

  // [thread][role][FFT node]: role 0 = input g, role 1 = variance,
  // role 2 = its lnR derivative (before normalization and splining).
  double*** fft_real;    // each thread's real input and two inverse-FFT outputs

  // [thread][role][radial node]: role 0 = cubic coefficients,
  // role 1 = temporary values for the tridiagonal spline solve.
  double*** spline_work; // each thread's coefficients and spline-solve workspace

  // [2*thread+role][frequency]: role 0 = forward FFT of g,
  // role 1 = forward coefficients multiplied by the integration kernel.
  fftw_complex** fft_complex; // each thread's forward FFT and kernel product

  double** mellin;       // [amplitude/phase][frequency], grid-only kernel
  fftw_complex** kernel; // [2*field+derivative][frequency], shifted kernel
  fftw_plan plan_forward; // cached instructions for the real-to-complex FFT
  fftw_plan plan_inverse; // cached instructions for the complex-to-real FFT
} sigma_fields_ = {0};


// ---------------------------------------------------------------------------
// Choose an even transform length that FFTW can factor efficiently.
//
// A direct discrete Fourier transform of N samples evaluates N sums,
// each containing N terms: its work grows as N^2. An FFT reduces that
// work by splitting the transform into smaller transforms and combining
// their results. If N = r*m, one such step splits the problem into r
// transforms of length m. The number r is the radix of that step.
//
// Small radices mean each combination step handles only a few values.
// FFTW has efficient routines for these short transforms, particularly
// lengths 2, 3, 5 and 7. When N is a product of those factors, FFTW can
// repeatedly split it into these small pieces. For example, 1024 = 2^10
// permits repeated radix-2 splits. A large prime length cannot be split
// this way and needs a different, potentially more expensive algorithm.
// FFTW supports such lengths too; they are not invalid.
//
// We therefore append zeros until the length has only the chosen prime
// factors, as in the non-Limber calculation. We also require an even N
// so our real-FFT indexing has a Nyquist coefficient at N/2. The even
// requirement is this implementation's convention, not an FFTW limit.
//
// Algorithm:
//   Start at the first even integer >= length. Divide out every factor
//   of 2, then 3, 5 and 7. A remainder of 1 means all prime factors were
//   in that list. Otherwise test the next even integer.
//
// Parameters:
//   length - number of physical input samples; must be at least two
//
// Returns:
//   the smallest qualifying even length >= length. The caller fills
//   the added entries with zeros after continuing the physical spectrum
//   across the chosen k range; the helper changes no sampling interval.
// ---------------------------------------------------------------------------
static int sigma2_fft_size(
    int length  // minimum transform length
  )
{
  if (length < 2) {
    log_fatal("sigma2_fft_size: at least two input samples are required");
    exit(1);
  }

  if (length % 2 != 0) {
    length++;
  }
  // Try successive even lengths without changing the spacing in lnk.
  for (;;) {
    int remaining = length;
    const int primes[4] = {2, 3, 5, 7};
    for (int prime=0; prime<4; prime++) {
      while (remaining % primes[prime] == 0) {
        remaining /= primes[prime];
      }
    }
    if (remaining == 1) {
      return length;
    }
    length += 2;
  }
}


// ---------------------------------------------------------------------------
// Natural cubic spline on the evenly spaced lnR grid.
//
// Why a spline is needed:
//   the inverse FFT gives sigma^2 at its own radii, while halo integrals
//   ask for values at a different set of mass nodes. We interpolate
//   ln(sigma^2/a^2) and dln sigma/dlnM, which vary smoothly with lnR.
//
// Write the polynomial in interval j as
//
//   S(x_j + t) = y_j + B_j t + C_j t^2 + E_j t^3,    0 <= t <= h,
//
// where x = lnR, h is the grid spacing, and y_j is the supplied value.
// Matching S and its first two derivatives at neighboring nodes gives
//
//   C_(j-1) + 4 C_j + C_(j+1) = 3 (y_(j-1) - 2 y_j + y_(j+1)) / h^2.
//
// This linear system has only three nonzero diagonals. The first loop
// eliminates the lower diagonal; the second substitutes backward to
// obtain every C_j. The caller then constructs
//
//   B_j = (y_(j+1)-y_j)/h - h (C_(j+1)+2 C_j)/3,
//   E_j = (C_(j+1)-C_j)/(3 h).
//
// The natural boundary condition sets C=0 at both ends (S''=0).
// The physical variance need not have zero curvature there, so the
// caller supplies extra radius nodes beyond both ends of the requested
// mass range. This moves that artificial boundary away from all queried
// masses; its effect decays rapidly through the tridiagonal equations.
//
// Parameters:
//   values  - y_j on the uniform lnR grid, including extra end nodes
//   count   - number of radial nodes
//   spacing - h = delta lnR
//   coeff   - output C_j = S''(x_j)/2, length count
//   scratch - temporary reciprocal diagonal entries, length count
//
// Returns:
//   nothing; fills coeff and overwrites scratch. Both arrays are
//   supplied by the caller, so this function allocates no memory.
//   Parallel callers must provide a separate pair for each thread.
// ---------------------------------------------------------------------------
static void sigma2_spline_coeffs(
    const double* restrict values, // function on a uniform grid
    const int count,               // number of radial nodes
    const double spacing,          // spacing in lnR
    double* restrict coeff,        // half the second derivative
    double* restrict scratch       // elimination multipliers
  )
{
  const double rhs_scale = 3.0/(spacing*spacing);

  // Left natural boundary and forward elimination of the lower diagonal.
  coeff[0] = 0.0;
  scratch[0] = 0.0;
  for (int node=1; node<count-1; node++) {
    const double rhs = rhs_scale*(values[node-1] - 2.0*values[node]
                                 + values[node+1]);
    scratch[node] = 1.0/(4.0 - scratch[node-1]);
    coeff[node] = (rhs - coeff[node-1])*scratch[node];
  }

  // Right natural boundary, then solve for the remaining coefficients.
  coeff[count-1] = 0.0;

  for (int node=count-2; node>0; node--) {
    coeff[node] -= scratch[node]*coeff[node+1];
  }
}


// ---------------------------------------------------------------------------
// Build both fields' variance and mass-slope tables at all scale factors.
//
// The derivation above explains the integral; this function implements
// it in three stages:
//
//   1. Set the grids and allocate work arrays. Cache two FFTW plans and
//      the Mellin transform of W^2. These depend on grid geometry, not
//      on the amplitudes of the power spectra.
//
//   2. Map mass to radius for this cosmology. The two mean densities
//      give different radius origins. Shift the cached Mellin kernel
//      to each origin and form the derivative kernel once per field.
//
//   3. For each (field, a) row, read P(k,a), take one forward FFT and
//      two inverse FFTs, then interpolate from lnR to the mass grid.
//
// Input range and padding are different operations:
//   The supplied spectrum is continued to lower and higher k with its
//   edge power laws, just as p_lin does. This retains high-k power that
//   contributes to small halos. Only after that physical continuation
//   do we append zeros, if necessary, to reach a transform length that
//   FFTW can factor efficiently. The two operations must not be swapped.
//
// Threading and plan reuse follow cfftlog_ells_p1/p2 in cosmo2D.c:
//   FFTW planning is done here, outside OpenMP, only when the workspace
//   is rebuilt. Cosmology changes reuse those plans. Parallel workers
//   execute a shared plan on their own arrays through FFTW's new-array
//   interface. Each worker completes an entire (field, a) row, so no
//   sum or temporary array is shared between workers.
//
// Cache invalidation:
//   geometry/workspace: Ntable.random, k grid, mass/a limits, threads
//   table values:      cosmology.random, or any geometry rebuild
//   Mellin factors:    only a geometry rebuild; shared by both fields
//
// Parameters:
//   none. Reads cosmology.lnPL and lnPL_cb, the mean densities, limits,
//   and Ntable. The spectrum axes are lnPL[i][nz] = log10 k_i and
//   lnPL[nk][j] = z_j; lnPL[i][j] contains ln P(k_i,z_j).
//
// Returns:
//   nothing; fills sigma_fields_.table. If P_cb is absent, only the
//   total-matter table is filled; a cb request is rejected by the reader.
//
// Call once outside any consumer's OpenMP loop to build the lazy cache.
// Subsequent reads of the unchanged cache return immediately.
// ---------------------------------------------------------------------------
static void sigma2_fields_build(void)
{
  if (sigma_fields_.table != NULL
      && !fdiff2(sigma_fields_.cosmology_tag, cosmology.random)
      && !fdiff2(sigma_fields_.ntable_tag, Ntable.random)
      && sigma_fields_.nthreads == omp_get_max_threads()) {
    return;
  }



  // --- 1a. CHOOSE THE INPUT k GRID AND OUTPUT MASS/a GRIDS ---

  const double bias = 1.5;        // power divided out of Delta^2 before FFT
  const double kmin = 1.e-7;      // h/Mpc: low-k continuation
  const double kmax = 1.e5;       // h/Mpc: retain the small-mass tail
  const int padding = 16;        // radial spline boundary margin
  const int nk = cosmology.lnPL_nk;
  const int nz = cosmology.lnPL_nz;
  const int threads = omp_get_max_threads();

  if (NULL == cosmology.lnPL || nk < 2 || nz < 2) {
    log_fatal("sigma2_field needs a linear power table with at least two "
              "k and z nodes; call set_linear_power_spectrum first");
    exit(1);
  }

  // The likelihood supplies uniform log10 k nodes. Convert the origin
  // and spacing to natural logarithms, the coordinate used by FFTLog.
  const double lnk0 = log(10.0)*cosmology.lnPL[0][nz];
  const double dlnk = log(10.0)*(cosmology.lnPL[nk-1][nz]
                                - cosmology.lnPL[0][nz])/(nk-1);

  if (!(dlnk > 0.0)) {
    log_fatal("sigma2_field: the input k grid must be increasing");
    exit(1);
  }

  // Continue on the same grid until both integration bounds are covered.
  // first/last are integer offsets relative to the first supplied k node;
  // first may be negative when the continuation reaches lower k.
  const int first = (int) floor((log(kmin)-lnk0)/dlnk);
  const int last  = (int) ceil((log(kmax)-lnk0)/dlnk);
  const int ninput = last-first+1;
  const int nfft = sigma2_fft_size(ninput);

  // The conjugate FFT grid has spacing delta lnR = delta lnk.
  // Since lnM = 3 lnR + constant, the required radius interval spans
  // one third of the mass interval in logarithmic coordinates.
  // Extra radius nodes on each side protect the cubic spline from its
  // artificial zero-curvature boundary condition.
  const int nradial = (int) ceil(log(limits.halo_m[RANGE_MAX]
                                     /limits.halo_m[RANGE_MIN])/(3.0*dlnk))
                     + 2*padding+1;
  const int nmass = Ntable.N_M[NODES_DENSE];
  const int na = Ntable.N_a;
  const double lnm0 = log(limits.halo_m[RANGE_MIN]);
  const double lnm1 = log(limits.halo_m[RANGE_MAX]);
  if (nradial > nfft || nmass < 2 || na < 2) {
    log_fatal("sigma2_field: incompatible FFT/mass/a grid; use finer "
              "logarithmic k sampling and at least two mass and a nodes");
    exit(1);
  }



  // --- 1b. REUSE THE WORKSPACE, OR REBUILD IT IF ITS SHAPE CHANGED ---

  if (NULL == sigma_fields_.table
      || fdiff2(sigma_fields_.ntable_tag, Ntable.random)
      || sigma_fields_.nk != nk
      || sigma_fields_.lnk0 != lnk0
      || sigma_fields_.dlnk != dlnk
      || sigma_fields_.nthreads != threads
      || sigma_fields_.lnm0 != lnm0
      || sigma_fields_.lnm1 != lnm1
      || sigma_fields_.amin != limits.a_min) {
    // Each multidimensional allocator returns one owned block. Free it
    // once, including its row pointers; never free the individual rows.
    if (sigma_fields_.table != NULL) {
      fftw_destroy_plan(sigma_fields_.plan_forward);
      fftw_destroy_plan(sigma_fields_.plan_inverse);
      free(sigma_fields_.table);
      free(sigma_fields_.fft_real);
      free(sigma_fields_.spline_work);
      free(sigma_fields_.fft_complex);
      free(sigma_fields_.kernel);
      free(sigma_fields_.mellin);
    }

    sigma_fields_.table = (double****) malloc4d(2, 2, na, nmass);
    sigma_fields_.fft_real = (double***) malloc3d(threads, 3, nfft);
    sigma_fields_.spline_work = (double***) malloc3d(threads, 2, nradial);
    sigma_fields_.fft_complex = (fftw_complex**) malloc2d_fftwc(2*threads, nfft/2+1);
    sigma_fields_.kernel = (fftw_complex**) malloc2d_fftwc(4, nfft/2+1);
    sigma_fields_.mellin = (double**) malloc2d(2, nfft/2+1);

    // The plans describe transforms of length nfft, not a cosmology.
    // Plan against thread zero's aligned arrays. During execution each
    // thread supplies its own equally aligned rows to the same plan.
    // FFTW_ESTIMATE chooses the transform recipe without timing trial
    // transforms; later execute calls apply that recipe to new data.
    sigma_fields_.plan_forward = fftw_plan_dft_r2c_1d(nfft,
        sigma_fields_.fft_real[0][0], sigma_fields_.fft_complex[0], FFTW_ESTIMATE);
    sigma_fields_.plan_inverse = fftw_plan_dft_c2r_1d(nfft,
        sigma_fields_.fft_complex[1], sigma_fields_.fft_real[0][1], FFTW_ESTIMATE);

    sigma_fields_.nfft = nfft;
    sigma_fields_.ninput = ninput;
    sigma_fields_.nradial = nradial;
    sigma_fields_.nthreads = threads;
    sigma_fields_.nmass = nmass;
    sigma_fields_.na = na;
    sigma_fields_.nk = nk;
    sigma_fields_.lnk0 = lnk0;
    sigma_fields_.dlnk = dlnk;
    sigma_fields_.lnm0 = lnm0;
    sigma_fields_.lnm1 = lnm1;
    sigma_fields_.dlnm = (lnm1-lnm0)/(nmass-1);
    sigma_fields_.amin = limits.a_min;
    sigma_fields_.da = (1.0-limits.a_min)/(na-1);



    // --- 1c. INTEGRATE EACH FOURIER MODE ANALYTICALLY ---

    // Like the non-Limber c-window cache, the Mellin factors depend on
    // the FFT grid, not the cosmology. Evaluate the Gamma functions
    // once here; both fields and every a row reuse them below.
    for (int mode=0; mode<=nfft/2; mode++) {
      const double eta = 2.0*M_PI*mode/(nfft*dlnk);

      // GSL returns ln|Gamma| and arg(Gamma), so products and ratios
      // of potentially enormous Gamma values become sums of logarithms.
      // With s = bias + i eta, the four entries are the Gamma factors
      // in Mellin(s), in the order written in the derivation above.
      gsl_sf_result amplitude[4]; // logarithms of the four Gamma magnitudes
      gsl_sf_result phase[4];     // complex arguments of the four Gamma factors
      gsl_sf_lngamma_complex_e(4.0-bias, -eta, &amplitude[0], &phase[0]);
      gsl_sf_lngamma_complex_e(bias/2.0, eta/2.0, &amplitude[1], &phase[1]);
      gsl_sf_lngamma_complex_e((5.0-bias)/2.0, -eta/2.0,
                               &amplitude[2], &phase[2]);
      gsl_sf_lngamma_complex_e(4.0-bias/2.0, -eta/2.0,
                               &amplitude[3], &phase[3]);

      // Numerator factors add; denominator factors subtract. Gamma
      // ((5-s)/2) is squared, hence its coefficient of two below.
      const double lnamp = log(4.5*M_PI) + amplitude[0].val + amplitude[1].val
          -(4.0-bias)*log(2.0)-2.0*amplitude[2].val-amplitude[3].val;
      const double angle = phase[0].val+phase[1].val+eta*log(2.0)
          -2.0*phase[2].val-phase[3].val;

      // A finite FFT treats its input as periodically repeating. The
      // meeting of the two distant k endpoints can create oscillations
      // in the result. Reduce the highest-frequency coefficients with
      // a smooth window, leaving the lower-frequency modes untouched.
      // The window t-sin(2 pi t)/(2 pi) joins 0 and 1 with zero slope.
      const double fraction = 2.0*mode/nfft;
      double window = 1.0;
      if (fraction > 0.75) {
        const double taper = (1.0-fraction)/0.25;
        window = taper-sin(2.0*M_PI*taper)/(2.0*M_PI);
      }

      sigma_fields_.mellin[0][mode] = exp(lnamp)*window;
      sigma_fields_.mellin[1][mode] = angle;
    }
  }



  // --- 2. THIS COSMOLOGY: MASS-RADIUS MAPS AND THE TWO KERNELS ---

  const int nfields = (cosmology.lnPL_cb == NULL) ? 1 : 2;
  double lnR0[2];
  const double lnkin = lnk0+first*dlnk;

  // rho_crit is stored per (c/H0)^3. Convert to the Mpc/h volume unit
  // before solving M = (4 pi/3) rho_field R^3 for the smoothing radius.
  const double rho_unit = cosmology.rho_crit/pow(cosmology.coverH0, 3.0);

  for (int field=0; field<nfields; field++) {
    double omega = cosmology.Omega_m;
    if (field == HALO_FIELD_CB) {
      omega -= cosmology.Omega_nu;
    }

    if (omega <= 0.0) {
      log_fatal("sigma2_field: Omega_field = %g must be positive", omega);
      exit(1);
    }

    // Radius of the minimum tabulated mass, then move left by the
    // extra spline nodes. Different mean densities shift this origin.
    lnR0[field] = (lnm0-log(4.0*M_PI*rho_unit*omega/3.0))/3.0-padding*dlnk;

    // Fourier coefficients refer to samples starting at lnkin, whereas
    // the output must start at lnR0. Account for both nonzero origins
    // with exp[-i eta (lnkin+lnR0)] multiplying Mellin(s).
    for (int mode=0; mode<=nfft/2; mode++) {
      const double eta = 2.0*M_PI*mode/(nfft*dlnk);
      const double angle = sigma_fields_.mellin[1][mode]
          -eta*(lnkin+lnR0[field]);
      const double amplitude = sigma_fields_.mellin[0][mode];

      // FFTW's inverse uses exp(+i eta lnR), while the integrated mode
      // varies as exp(-i eta lnR). Conjugate both the kernel here and
      // the forward coefficient below to obtain the required real sum.
      const double real_kernel = amplitude*cos(angle);
      const double imag_kernel = -amplitude*sin(angle);
      sigma_fields_.kernel[2*field][mode][0] = real_kernel;
      sigma_fields_.kernel[2*field][mode][1] = imag_kernel;

      // Differentiating R^(-bias-i*eta) gives -(bias+i*eta).
      // Conjugation reverses that sign of i in the inverse-FFT kernel.
      sigma_fields_.kernel[2*field+1][mode][0] = -bias*real_kernel-eta*imag_kernel;
      sigma_fields_.kernel[2*field+1][mode][1] = -bias*imag_kernel+eta*real_kernel;
    }
  }



  // --- 3. TRANSFORM EACH (FIELD, a) ROW INDEPENDENTLY ---

  // collapse(2) treats the (field,row) pairs as one list of independent
  // jobs. schedule(static) divides that list among the workers. A worker
  // finishes a row before reusing its scratch arrays for the next one.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int field=0; field<nfields; field++) {
    for (int row=0; row<na; row++) {
      const int thread = omp_get_thread_num();

      // Each work/output row occupies a separate region of memory.
      // restrict promises that stores through one local pointer cannot
      // change values read through another, so the compiler need not
      // reload them to account for possible overlap. The calculations
      // below use these local pointers to make that promise effective.
      // The thread index also gives each worker its own scratch rows;
      // workers share no writable data during the row calculation.
      double* restrict input = sigma_fields_.fft_real[thread][0];
      double* restrict radial = sigma_fields_.fft_real[thread][1];
      double* restrict derivative = sigma_fields_.fft_real[thread][2];

      double* restrict coeff = sigma_fields_.spline_work[thread][0];
      double* restrict scratch = sigma_fields_.spline_work[thread][1];

      double* restrict variance = sigma_fields_.table[0][field][row];
      double* restrict slope = sigma_fields_.table[1][field][row];

      fftw_complex* restrict forward = sigma_fields_.fft_complex[2*thread];
      fftw_complex* restrict product = sigma_fields_.fft_complex[2*thread+1];


      // --- 3a. READ THE POWER SPECTRUM AT THIS SCALE FACTOR ---

      const double a = limits.a_min+row*sigma_fields_.da;
      const double redshift = 1.0/a-1.0;

      // Locate the two input redshifts surrounding this scale factor.
      // The setter records the few uniform segments of the z grid.
      // Within one segment, (z-z_start)/dz gives the node directly;
      // piecewise_index adds the segment's starting index and keeps
      // the result in [0,nz-2], so both lower and lower+1 exist.
      // Both fields share this grid and its metadata, as in p_lin_cb.
      const int lower = piecewise_index(redshift, cosmology.lnPL_z_nseg,
          cosmology.lnPL_z_seg_start, cosmology.lnPL_z_seg_len,
          cosmology.lnPL_z_seg_xmin, cosmology.lnPL_z_seg_inv_dx, nz);

      // Keep the edge bracket for z outside the supplied range. The
      // fraction can then lie outside [0,1]: this continues lnP linearly
      // in z, matching the power-spectrum readers' edge behavior.
      const double fraction_z = (redshift-cosmology.lnPL[nk][lower])
          /(cosmology.lnPL[nk][lower+1]-cosmology.lnPL[nk][lower]);

      double** power = cosmology.lnPL;
      if (field == HALO_FIELD_CB) {
        power = cosmology.lnPL_cb;
      }

      // Each extended k node is an integer step from the supplied grid.
      // Use its two neighboring k nodes, or the first/last pair outside
      // the input range. Continuing lnP linearly in lnk is a power-law
      // continuation of P, not a constant or a zero beyond the table.
      for (int node=0; node<ninput; node++) {
        const int source = node+first;
        const int bracket = (int) fmin(fmax(source, 0), nk-2);

        // Interpolate lnP in redshift at each end of the k interval.
        const double left = power[bracket][lower]
            +fraction_z*(power[bracket][lower+1]-power[bracket][lower]);
        const double right = power[bracket+1][lower]
            +fraction_z*(power[bracket+1][lower+1]-power[bracket+1][lower]);

        // Then interpolate (or continue) along lnk, and form
        // g = k^(-bias) Delta^2 = k^(3-bias) P / (2 pi^2).
        const double lnP = left+(source-bracket)*(right-left);
        const double lnk = lnkin+node*dlnk;
        input[node] = exp(lnP+(3.0-bias)*lnk)/(2.0*M_PI*M_PI);
      }

      // ninput covers the entire chosen physical k range. nfft may be
      // slightly larger because FFTW is faster at lengths whose prime
      // factors are small. The remaining entries are numerical padding:
      // set them to zero so they add no power to the integral. They lie
      // after the high-k power-law continuation, never inside it.
      for (int node=ninput; node<nfft; node++) {
        input[node] = 0.0;
      }


      // --- 3b. ONE FORWARD FFT, TWO INVERSE FFTS ---

      fftw_execute_dft_r2c(sigma_fields_.plan_forward, input, forward);

      // Reuse the one forward FFT for sigma^2 and its mass derivative.
      // Only the inverse FFT differs, just as non-Limber p2 reuses p1.
      for (int derivative_order=0; derivative_order<2; derivative_order++) {
        const fftw_complex* restrict kernel =
            sigma_fields_.kernel[2*field+derivative_order];

        for (int mode=0; mode<=nfft/2; mode++) {
          const double real_kernel = kernel[mode][0];
          const double imag_kernel = kernel[mode][1];

          // conj(forward) times the conjugated Mellin/phase kernel.
          const double real_product = forward[mode][0]*real_kernel
                                      +forward[mode][1]*imag_kernel;
          const double imag_product = forward[mode][0]*imag_kernel
                                      -forward[mode][1]*real_kernel;

          product[mode][0] = real_product;
          product[mode][1] = imag_product;
        }

        double* output = radial;
        if (derivative_order == 1) {
          output = derivative;
        }

        // The shared plan is executed on this thread's arrays. The
        // forward coefficients remain intact for the second inverse.
        fftw_execute_dft_c2r(sigma_fields_.plan_inverse, product, output);
      }


      // --- 3c. RESTORE NORMALIZATION AND FORM THE MASS SLOPE ---

      // FFTW leaves its inverse transform unnormalized. Divide by nfft
      // and multiply by R^(-bias) to recover sigma^2. Those two factors
      // cancel in the derivative/variance ratio; the factor 1/6 converts
      // d sigma^2/dlnR into dln sigma/dlnM (derivation above).
      for (int node=0; node<nradial; node++) {
        const double lnR = lnR0[field]+node*dlnk;
        if (!(radial[node] > 0.0)) {
          log_fatal("sigma2_field: nonpositive FFTLog variance at field=%d a=%g", field, a);
          exit(1);
        }

        derivative[node] /= 6.0*radial[node];

        // Divide out a^2 before the linear a interpolation: in matter
        // domination sigma^2 is nearly proportional to a^2. Removing
        // that curvature keeps the early-time interpolation accurate.
        radial[node] = log(radial[node]/nfft)-bias*lnR-2.0*log(a);
      }


      // --- 3d. FROM FFT RADIUS NODES TO THE HALO MASS NODES ---

      for (int quantity=0; quantity<2; quantity++) {
        const double* values = radial;
        double* output = variance;
        if (quantity == 1) {
          values = derivative;
          output = slope;
        }

        sigma2_spline_coeffs(values, nradial, dlnk, coeff, scratch);

        // lnR-lnR0 = padding*dlnk + (lnM-lnMmin)/3. This directly
        // locates each requested mass in the uniform radial grid.
        for (int mass=0; mass<nmass; mass++) {
          const double position = padding+mass*sigma_fields_.dlnm/(3.0*dlnk);
          const int node = (int) floor(position);
          const double offset = (position-node)*dlnk;

          // S(t) = y + B t + C t^2 + E t^3 in this interval. The spline
          // solve supplied C; form B and E from the two end values.
          const double linear = (values[node+1]-values[node])/dlnk
              -dlnk*(coeff[node+1]+2.0*coeff[node])/3.0;
          const double cubic = (coeff[node+1]-coeff[node])/(3.0*dlnk);

          output[mass] = values[node]+offset*(linear
                          +offset*(coeff[node]+offset*cubic));
        }
      }
    }
  }

  // Publish the cache tags only after every parallel row is complete.
  sigma_fields_.cosmology_tag = cosmology.random;
  sigma_fields_.ntable_tag = Ntable.random;
}


// ---------------------------------------------------------------------------
// Read a variance or slope from the cached (lnM, a) table.
//
// First interpolate between neighboring masses in each of the two
// neighboring a rows, then interpolate between those two results in a.
// The variance table stores ln(sigma^2/a^2); sigma2_field restores a^2
// after this interpolation. The mass-slope table needs no rescaling.
//
// Parameters:
//   M          - positive halo mass in M_sun/h
//   a          - scale factor in [limits.a_min, 1]
//   field      - HALO_FIELD_MATTER or HALO_FIELD_CB
//   derivative - 0 reads ln(sigma^2/a^2); 1 reads dln sigma/dlnM
//
// Returns:
//   the requested dimensionless table value. Masses outside the halo
//   interval use the nearest mass endpoint. Invalid scale factors and
//   a missing cb spectrum are errors; neither has an implicit substitute.
// ---------------------------------------------------------------------------
static double sigma2_field_read(
    const double M,      // mass in M_sun/h
    const double a,      // scale factor
    const int field,     // HALO_FIELD_MATTER or HALO_FIELD_CB
    const int derivative // 0: ln(sigma^2/a^2), 1: dln sigma/dlnM
  )
{
  if (!isfinite(M) || M <= 0.0 || !isfinite(a) || a < limits.a_min || a > 1.0
      || (field != HALO_FIELD_MATTER && field != HALO_FIELD_CB)) {
    log_fatal("sigma2_field: M=%g a=%g field=%d; require positive M, "
              "a in [%g,1], field 0 or 1", M, a, field, limits.a_min);
    exit(1);
  }
  if (field == HALO_FIELD_CB && cosmology.lnPL_cb == NULL) {
    log_fatal("sigma2_field: cb variance needs P_cb; request delta_nonu "
              "and call set_linear_power_spectrum_cb");
    exit(1);
  }

  sigma2_fields_build();

  // --- 1. LOCATE THE MASS AND SCALE FACTOR IN THE TABLE ---

  // The table is uniform in lnM, not M: neighboring nodes have a fixed
  // mass ratio, rather than a fixed mass difference. Convert M to lnM
  // and keep out-of-range queries at the nearest tabulated endpoint.
  double lnM = log(M);
  if (lnM < sigma_fields_.lnm0) {
    lnM = sigma_fields_.lnm0;
  } else if (lnM > sigma_fields_.lnm1) {
    lnM = sigma_fields_.lnm1;
  }

  // Subtract the grid origin and divide by the spacing to measure the
  // location in units of table intervals. For example, position 12.25
  // lies one quarter of the way from node 12 to node 13. The same rule
  // applies to a, whose valid range was checked above.
  const double mass_position = (lnM-sigma_fields_.lnm0)/sigma_fields_.dlnm;
  const double a_position = (a-sigma_fields_.amin)/sigma_fields_.da;

  // Both positions are nonnegative, so conversion to int takes their
  // integer part: the node on the left of the interpolation interval.
  int mass_node = (int) mass_position;
  int a_node = (int) a_position;

  // A table with N nodes has intervals starting at 0,...,N-2. At the
  // final node, position N-1 must use interval [N-2,N-1]; there is no
  // node N to read. Moving the left index back makes the fraction below
  // equal to 1, which returns the final node's value exactly.
  if (mass_node >= sigma_fields_.nmass-1) {
    mass_node = sigma_fields_.nmass-2;
  }
  if (a_node >= sigma_fields_.na-1) {
    a_node = sigma_fields_.na-2;
  }

  // These are distances from the left nodes, measured as fractions of
  // one interval: 0 selects the left endpoint, 1 the right endpoint.
  const double mass_fraction = mass_position-mass_node;
  const double a_fraction = a_position-a_node;


  // --- 2. SELECT THE PHYSICAL QUANTITY AND DENSITY FIELD ---

  // The first two indices of table select quantity and field. Fixing
  // them leaves a two-dimensional view values[a_node][mass_node]:
  //   quantity 0: ln(sigma_field^2/a^2), for the variance reader;
  //   quantity 1: dln sigma_field/dlnM, for the mass-slope reader.
  // field 0 selects total matter; field 1 selects cold matter+baryons.
  // Only the view changes here; no table is copied or allocated.
  double** values = sigma_fields_.table[0][field];
  if (derivative) {
    values = sigma_fields_.table[1][field];
  }


  // --- 3. INTERPOLATE BETWEEN THE FOUR SURROUNDING VALUES ---

  // For one interval, linear interpolation is y = y_left + f*(y_right
  // - y_left). Apply it in mass at each of the two bounding a values:
  // lower is the mass-interpolated value at a_node; upper is the same
  // at a_node+1. Finally interpolate these two numbers to the requested
  // a. These two linear steps are called bilinear interpolation.
  const double lower = values[a_node][mass_node]+mass_fraction
      *(values[a_node][mass_node+1]-values[a_node][mass_node]);
  const double upper = values[a_node+1][mass_node]+mass_fraction
      *(values[a_node+1][mass_node+1]-values[a_node+1][mass_node]);

  return lower+a_fraction*(upper-lower);
}


// ---------------------------------------------------------------------------
// Variance of the smoothed linear matter or cb density at (M,a).
// Parameters: M in M_sun/h, a the scale factor, field as in the reader.
// Returns: sigma_field^2(M,a), dimensionless.
// ---------------------------------------------------------------------------
double sigma2_field(
    const double M, // halo mass in M_sun/h
    const double a, // scale factor
    const int field // HALO_FIELD_MATTER or HALO_FIELD_CB
  )
{
  return exp(sigma2_field_read(M, a, field, 0))*a*a;
}


// ---------------------------------------------------------------------------
// Logarithmic mass slope of the linear rms density fluctuation.
// Parameters: M in M_sun/h, a the scale factor, field as in the reader.
// Returns: dln sigma_field(M,a)/dlnM, dimensionless (usually negative).
// ---------------------------------------------------------------------------
double dlnsigma_dlnm_field(
    const double M, // halo mass in M_sun/h
    const double a, // scale factor
    const int field // HALO_FIELD_MATTER or HALO_FIELD_CB
  )
{
  return sigma2_field_read(M, a, field, 1);
}


// Halo consumers use the cold variance at the requested scale factor.
// The separate sigma2_field reader also exposes total-matter variance.
double sigma2(
    const double M, // halo mass in M_sun/h
    const double a  // scale factor
  )
{
  return sigma2_field(M, a, HALO_FIELD_CB);
}
