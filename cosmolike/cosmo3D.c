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
// ---------------------------------------------------------------------------
static inline int piecewise_index(double q, int nseg,
                                  const int *start, const int *len,
                                  const double *xmin, const double *inv_dx,
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

double chi(const double a)
{
  struct chis r = chi_all(a);
  return r.chi;
}

double dchi_da(const double a)
{
  struct chis r = chi_all(a);
  return r.dchida;
}

#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
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

  // Convert from (Mpc/h) to (Mpc/h)/(c/H0=100)^3 (dimensionless),
  // and from d(chi)/dz to d(chi)/da via z = 1/a - 1.
  struct chis result;
  result.chi    = chi_interp / cosmology.coverH0;
  result.dchida = dchidz / cosmology.coverH0 / (a * a);
  return result;
}
#else
struct chis chi_all(const double a)
{
  double out[2];
  const double z = 1.0/a - 1.0;

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

  const double dy = (z                     - cosmology.chi[0][j])/
                    (cosmology.chi[0][j+1] - cosmology.chi[0][j]);
  out[0] = cosmology.chi[1][j] + dy*(cosmology.chi[1][j+1]-cosmology.chi[1][j]);

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

  // convert from (Mpc/h) to (Mpc/h)/(c/H0=100)^3 (dimensioneless)
  out[1] = (out[1]/cosmology.coverH0);
  // convert from d\chi/dz to d\chi/da
  out[1] = out[1]/(a*a);

  struct chis result;
  result.chi = out[0]/cosmology.coverH0;
  result.dchida = out[1];
  return result;
}
#endif

double dchi_dz(const double a)
{
  return (a*a)*dchi_da(a);
}

double hoverh0(const double a)
{
  return 1.0/dchi_dz(a);
}

double hoverh0v2(const double a, const double dchida)
{
  return 1.0/((a*a)*dchida);
}

double a_chi(const double io_chi)
{
  // convert from (Mpc/h)/(c/H0=100)^3 (dimensioneless) to (Mpc/h)
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

double growfac(const double a)
{
  return norm_growfac(a, true);
}

#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
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
double norm_growfac(const double a, const bool normalize_z0)
{
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

#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
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
  const double dlnGdlna = -dlnGdlnz * (1+z) / z;

  return 1 + dlnGdlna; // Growth D = G * a
}
#else
double f_growth(const double z)
{
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
  
  const double dlnGdlna = -dlnGdlnz*(1+z)/z;

  return 1 + dlnGdlna; // Growth D = G * a
}
#endif

#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
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
  const double dlnGdlna = -dlnGdlnz * (1+z) / z;

  struct growths Gf;
  Gf.f = 1 + dlnGdlna;
  Gf.D = normalize_z0 ? (G*a)/growfact1 : (G*a);
  return Gf;
}
#else
struct growths norm_growfac_all(const double a, const bool normalize_z0)
{
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
  
  const double dlnGdlna = -dlnGdlnz*(1+z)/z;

  struct growths Gf;
  Gf.f = 1 + dlnGdlna; // Growth D = G * a

  if(normalize_z0)
    Gf.D = (G*a)/growfact1; 
  else
    Gf.D = (G*a);

  return Gf;
}
#endif

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
#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
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
double p_lin(const double k, const double a)
{
  // convert from (x/Mpc/h - dimensioneless) to h/Mpc with x = c/H0 (Mpc)
  const double log10k = log10(k/cosmology.coverH0);
  const double z = 1.0/a-1.0;

  // logk = cosmology.lnPL[0:nk,cosmology.lnPL_nz]
  // z    = cosmology.lnPL[cosmology.lnPL_nk,0:nz]
  
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
#ifdef COSMO3D_ASSUME_PIECEWISE_UNIFORM
double p_nonlin(const double k, const double a)
{
  const double coverH0 = cosmology.coverH0;
  // convert from (x/Mpc/h - dimensioneless) to h/Mpc with x = c/H0 (Mpc)
  const double log10k = log10(k / coverH0);
  const double z      = 1.0 / a - 1.0;

  // logk = cosmology.lnP[0:nk, cosmology.lnP_nz]
  // z    = cosmology.lnP[cosmology.lnP_nk, 0:nz]

  // -----------------------------------------------------------------
  // Direct-index lookup; see p_lin for the rationale and seam-correctness
  // argument. The lnP_* metadata fields are set in
  // set_non_linear_power_spectrum.
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
double p_nonlin(const double k, const double a)
{
  const double coverH0 = cosmology.coverH0;
  // convert from (x/Mpc/h - dimensioneless) to h/Mpc with x = c/H0 (Mpc)
  const double log10k = log10(k/coverH0);
  const double z = 1.0/a-1.0;

  // logk = cosmology.lnP[0:nk,cosmology.lnP_nz]
  // z    = cosmology.lnP[cosmology.lnP_nk,0:nz]
  
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

// calculating the angular diameter distance f_K
// BS01 2.4, 2.30: f_K is a radial function that, depending on the curvature of
// the Universe, is a trigonometric, linear, or hyperbolic function of chi
double f_K(double chi) 
{
  double K, K_h, f;
  K = (cosmology.Omega_m + cosmology.Omega_v - 1.);
  if (K > 1e-6) 
  { // open
    K_h = sqrt(K); // K in units H0/c see BS eq. 2.30
    f = 1. / K_h * sin(K_h * chi);
  } else if (K < -1e-6) 
  { // closed
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

// return P(k)_bary/P(k)_DMO from hydro sims ;
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

double MG_Sigma(double a __attribute__((unused))) {
  return 0.0;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_sigma2(double x, void* params) // inner integral
{
  double* ar = (double*) params;
  const double R = ar[0];
  const double a = ar[1];
  const double PK = p_lin(x/R, a);
  
  gsl_sf_result J1;
  int status = gsl_sf_bessel_j1_e(x, &J1);
  if (status) {
    log_fatal(gsl_strerror(status)); exit(1);
  }
  const double tmp = 3.0*J1.val/ar[0];
  return PK*tmp*tmp/(ar[0] * 2.0 * M_PI * M_PI);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double sigma2_nointerp(
    const double M, 
    const double a, 
    const int init
  ) 
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = 500 + 500 * (Ntable.high_def_integration);
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }
  
  double ar[1] = {pow(0.75*M/(M_PI*cosmology.rho_crit*cosmology.Omega_m),1./3.)};
  const double xmin = 0;
  const double xmax = 14.1;

  double res;
  if (1 == init) {
    res = int_for_sigma2((xmin+xmax)/2.0, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_sigma2;
    res = gsl_integration_glfixed(&F, xmin, xmax, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double sigma2(const double M) 
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* table;
  static double lim[3];

  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double*) malloc(sizeof(double)*Ntable.N_M);
    lim[0] = log(limits.halo_m_min);
    lim[1] = log(limits.halo_m_max);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_M - 1.0);
  } 
  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random)) {
    (void) sigma2_nointerp(exp(lim[0]), 1.0, 1);    
    #pragma omp parallel for schedule(static,1)
    for (int i=0; i<Ntable.N_M; i++) {
      table[i] = log(sigma2_nointerp(exp(lim[0] + i*lim[2]), 1.0, 0));
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
  return exp(interpol1d(table, Ntable.N_M, lim[0], lim[1], lim[2], log(M)));
}
