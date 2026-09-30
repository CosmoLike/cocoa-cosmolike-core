// ---------------------------------------------------------------------------
// redshift_spline.c
//
// Tomographic redshift distributions and the projection kernels built
// from them. Everything downstream of the survey n(z) file lives here:
//
//   survey n(z) file
//     -> redshift.shear_zdist_table / clustering_zdist_table histograms
//     -> per-bin normalized splines (nz_source_photoz, nz_lens_photoz)
//     -> uniform fine grids (hot-path direct-index evaluation)
//     -> kernels g_tomo / g2_tomo / g_lens / g_cmb
//     -> Limber projections in cosmo2D.c
//
// The file also owns the tomographic pair bookkeeping (Z1/Z2/N_shear,
// ZL/ZS/N_ggl, ZCL1/ZCL2/N_CL) and the integration bounds amin_* and
// amax_*.
//
// Photo-z nuisance parameters enter through the global
//
//   nuisance.photoz[sample][param][bin]
//
//   sample = 0 (source/shear)   1 (lens/clustering)
//   param  = 0 (shift dz)       1 (stretch sigma)
//
// so e.g. nuisance.photoz[1][0][3] is the shift of lens bin 3. This
// file applies the stretch only to the lens sample; the source model
// is shift-only.
//
// Caching idiom, shared by every function here with static tables:
// each global (cosmology, redshift, nuisance, tomo, Ntable) carries a
// random_* stamp that changes whenever its contents do; a table stores
// the stamps it was built with and rebuilds when fdiff2 sees a
// mismatch. Static tables start from a sentinel no valid entry can
// equal (-42 in the int maps, NULL pointers, zeroed stamps kept
// nonzero by construction), so the first call always builds. Rebuilds
// are not thread-safe: the first call after any stamp change must
// happen outside OpenMP regions, which is why builders warm the caches
// single-threaded, e.g. "(void) nz_source_photoz(0., 0)".
// ---------------------------------------------------------------------------

#include <assert.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <gsl/gsl_spline.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "basics.h"
#include "bias.h"
#include "cosmo3D.h"
#include "redshift_spline.h"
#include "structs.h"

#include "log.c/src/log.h"

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// integration boundary routines
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Highest true redshift where the SHIFTED source n(z) can be nonzero.
//
// nz_source_photoz evaluates the tabulated n(z) at z - dz_j (dz_j =
// nuisance.photoz[0][0][j], the photo-z shift of source bin j), so a
// positive shift moves n(z) mass above the table's upper edge zmax_all:
//
//   zmax = zmax_all + max(0, max_j dz_j)
//
// Every line-of-sight integral over the source sample (the lensing
// efficiencies g_tomo, g2_tomo and the Limber ranges) must start at
// a = 1/(1 + zmax); starting at 1/(1 + zmax_all) silently drops that
// mass (for the DES Y6 source n(z), which reaches z = 3, a shift of
// +0.034 drops 0.17% of bin 0 and lowers g(a) by 3% at z = 0.6 and 16%
// at z = 2). With every dz_j <= 0, zmax = zmax_all exactly.
//
// Returns:
//   the upper edge of the shifted source support, in redshift.
// ---------------------------------------------------------------------------
double zmax_source_photoz(void)
{
  double max_shift = 0.0;
  for (int j=0; j<redshift.shear_nbin; j++) {
    max_shift = fmax(max_shift, nuisance.photoz[0][0][j]);
  }
  return redshift.shear_zdist_zmax_all + max_shift;
}

// ---------------------------------------------------------------------------
// Highest true redshift where the shifted and stretched lens n(z) can be
// nonzero.
//
// nz_lens_photoz evaluates the tabulated n(z) at
// (z - dz_i - zmean_i)/sigma_i + zmean_i (shift dz_i, stretch sigma_i,
// nuisance.photoz[1][0][i] and [1][1][i]), so the table's upper edge
// zmax_all maps to
//
//   z_i = zmean_i + sigma_i (zmax_all - zmean_i) + dz_i,
//
// and the support of the whole sample ends at
//
//   zmax = max(zmax_all, max_i z_i).
//
// The lens magnification efficiency g_lens must start at 1/(1 + zmax).
// With dz_i <= 0 and sigma_i <= 1 for every bin, zmax = zmax_all exactly.
//
// Returns:
//   the upper edge of the shifted and stretched lens support, in redshift.
// ---------------------------------------------------------------------------
double zmax_lens_photoz(void)
{
  const double zmax_all = redshift.clustering_zdist_zmax_all;
  double zmax = zmax_all;
  for (int i=0; i<redshift.clustering_nbin; i++) {
    const double zmean   = redshift.clustering_zdist_zmean[i];
    const double stretch = nuisance.photoz[1][1][i];
    const double shift   = nuisance.photoz[1][0][i];
    // Only a positive shift or a stretch > 1 can move the edge above
    // zmax_all. Skipping the other bins is exact, and it keeps the result
    // bitwise equal to zmax_all at the defaults: in floating point
    // zmean + 1.0*(zmax_all - zmean) + 0.0 can land one ulp above zmax_all.
    if (shift > 0.0 || stretch > 1.0) {
      zmax = fmax(zmax, zmean + stretch*(zmax_all - zmean) + shift);
    }
  }
  return zmax;
}

// ---------------------------------------------------------------------------
// Lower scale-factor bound for line-of-sight integrals over source bin ni.
//
//   a_min = 1 / (1 + zmax_source_photoz())
//
// the upper edge of the tabulated source n(z) range moved up by the largest
// positive photo-z shift (zmax_source_photoz). The same bound applies to
// every source bin; ni is only validated.
//
// Parameters:
//   ni - source tomographic bin index (0 .. shear_nbin-1)
//
// Returns:
//   smallest scale factor with source n(z) support.
// ---------------------------------------------------------------------------
double amin_source(int ni) 
{
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  }
  return 1. / (zmax_source_photoz() + 1.);
}

// ---------------------------------------------------------------------------
// Upper scale-factor bound for line-of-sight integrals over the source
// sample.
//
//   a_max = 1 / (1 + max(zmin_all, 0.001))
//
// where zmin_all = redshift.shear_zdist_zmin_all, the lower edge of the
// tabulated source n(z) range; the z >= 0.001 floor keeps a_max strictly
// below 1. The bin argument is unused: the bound is common to all bins.
//
// Parameters:
//   i - source tomographic bin index; unused (the bound is bin-independent).
//
// Returns:
//   largest scale factor with source n(z) support.
// ---------------------------------------------------------------------------
double amax_source(int i __attribute__((unused))) 
{
  return 1. / (1. + fmax(redshift.shear_zdist_zmin_all, 0.001));
}

// ---------------------------------------------------------------------------
// Upper scale-factor bound for intrinsic-alignment integrals over source
// bin ni. Identical value to amax_source; this variant validates the bin
// index.
//
// Parameters:
//   ni - source tomographic bin index (0 .. shear_nbin-1)
//
// Returns:
//   1 / (1 + max(shear_zdist_zmin_all, 0.001)).
// ---------------------------------------------------------------------------
double amax_source_IA(int ni) 
{
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  }
  return 1. / (1. + fmax(redshift.shear_zdist_zmin_all, 0.001));
}

// ---------------------------------------------------------------------------
// Lower scale-factor bound for line-of-sight integrals over lens bin ni.
//
// The tabulated per-bin upper edge is stretched about the fiducial mean
// redshift by the photo-z stretch parameter, then padded by twice the
// absolute photo-z shift:
//
//   zmax  = (zdist_zmax[ni] - zmean[ni]) * sigma_ni + zmean[ni]
//   a_min = 1 / (1 + zmax + 2*|dz_ni|)
//
// with sigma_ni = nuisance.photoz[1][1][ni] (stretch) and
// dz_ni = nuisance.photoz[1][0][ni] (shift), so the bound covers the
// support of nz_lens_photoz after its shift-and-stretch mapping. The
// mapping moves the stretched edge by exactly dz_ni: one |dz_ni| of
// padding covers either sign of the shift, the second is margin.
//
// Parameters:
//   ni - lens tomographic bin index (0 .. clustering_nbin-1)
//
// Returns:
//   smallest scale factor with lens n(z) support for bin ni.
// ---------------------------------------------------------------------------
double amin_lens(int ni) 
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  }
  const double zmax = 
    (redshift.clustering_zdist_zmax[ni] 
      - redshift.clustering_zdist_zmean[ni])*nuisance.photoz[1][1][ni]
      + redshift.clustering_zdist_zmean[ni];
  return 1. / (1 + zmax + 2.*fabs(nuisance.photoz[1][0][ni]));
}

// ---------------------------------------------------------------------------
// Upper scale-factor bound for line-of-sight integrals over lens bin ni.
//
// Mirror of amin_lens: the tabulated per-bin lower edge is stretched about
// the fiducial mean redshift and padded by twice the absolute photo-z
// shift,
//
//   zmin  = (zdist_zmin[ni] - zmean[ni]) * sigma_ni + zmean[ni]
//   a_max = 1 / (1 + max(zmin - 2*|dz_ni|, 0.001))
//
// with sigma_ni = nuisance.photoz[1][1][ni] and dz_ni =
// nuisance.photoz[1][0][ni]; as in amin_lens, one |dz_ni| of padding
// covers either sign of the shift and the second is margin. When
// magnification bias is active for the bin (gbmag(0, ni) != 0), the
// kernel W_mag has support well in front of the lens galaxies, and
// the bound widens to the source-sample value
// 1 / (1 + max(shear_zdist_zmin_all, 0.001)) (same as amax_source).
//
// Parameters:
//   ni - lens tomographic bin index (0 .. clustering_nbin-1)
//
// Returns:
//   largest scale factor the bin's projection kernels support.
// ---------------------------------------------------------------------------
double amax_lens(int ni) 
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  }

  const double zmin = 
    (redshift.clustering_zdist_zmin[ni] 
      - redshift.clustering_zdist_zmean[ni])*nuisance.photoz[1][1][ni]
      + redshift.clustering_zdist_zmean[ni];

  if (gbmag(0.0, ni) != 0) {
    return 1. / (1. + fmax(redshift.shear_zdist_zmin_all, 0.001));
  }
  return 1. / (1 + fmax(zmin -2.*fabs(nuisance.photoz[1][0][ni]), 0.001));
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Linear-regime scale cut for galaxy clustering.
//
// Tests whether multipole l in lens bin ni maps to a wavenumber inside the
// range where the bias model is trusted:
//
//   k = (l + 0.5) / chiref[ni]  <  kmax = 2*pi / Rmin_bias * coverH0
//
// with like.Rmin_bias in Mpc/h and cosmology.coverH0 converting kmax to
// 1/(c/H0) units, matching chiref. chiref[ni] is the comoving distance to
// the midpoint of the bin's tabulated n(z) range,
// z = (zdist_zmin[ni] + zdist_zmax[ni]) / 2.
//
// Cache invalidation:
// chiref is built on the first call (sentinel
// chiref[0] = -1) and never rebuilt, so later cosmology changes do not
// move the cut. First call must happen outside OpenMP regions.
//
// Parameters:
//   l  - multipole
//   ni - lens tomographic bin index (0 .. clustering_nbin-1)
//
// Returns:
//   1 when (l, ni) passes the cut, 0 otherwise.
// ---------------------------------------------------------------------------
int test_kmax(double l, int ni)
{
  static double chiref[MAX_SIZE_ARRAYS] = {-1.};
    
  if (chiref[0] < 0) {
    for (int i=0; i<redshift.clustering_nbin; i++) {
      chiref[i] = chi(1.0/(1. + 0.5 * (redshift.clustering_zdist_zmin[i] + 
                                       redshift.clustering_zdist_zmax[i])));
    }
  }

  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni); exit(1);
  }
  
  const double R_min = like.Rmin_bias; // set minimum scale to which
                                       // we trust our bias model, in Mpc/h
  const double kmax = 2.0*M_PI / R_min * cosmology.coverH0;
  
  int res = 0;
  if ((l + 0.5) / chiref[ni] < kmax) {
    res = 1;
  }
  return res;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GGL pair admission test: is (lens bin ni, source bin nj) a usable
// galaxy-galaxy lensing pair?
//
// With no exclusion list (tomo.ggl_exclude == NULL) every pair is
// admitted. Otherwise a static lookup table N[lens][source] is built from
// the flat exclusion list: pair k is (ggl_exclude[2k], ggl_exclude[2k+1]),
// k < tomo.N_ggl_exclude. Excluded pairs store 0, all others 1.
//
// Cache invalidation:
// the table is stamped with the tomo.random_ggl key
// it was built with and rebuilds when the stamp differs or while the
// static-initializer sentinel is present (N[0][0] = -42 < -1; built
// entries are 0 or 1).
//
// init_ntomo_powerspectra draws a fresh random_ggl key and evaluates
// every pair single-threaded, so parallel callers only read; the first
// call after a key change must happen outside OpenMP regions.
//
// Parameters:
//   ni - lens (clustering) tomographic bin index
//   nj - source (shear) tomographic bin index
//
// Returns:
//   1 when the pair enters the GGL data vector, 0 when excluded.
// ---------------------------------------------------------------------------
int test_zoverlap(int ni, int nj) 
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input (ni, nj) = (%d, %d)", ni, nj);
    exit(1);
  }
  if (tomo.ggl_exclude != NULL) {
    static int N[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS] = {{-42}};
    static uint64_t cache = 0; // tomo.random_ggl the map was built with
    if (N[0][0] < -1 || fdiff2(cache, tomo.random_ggl)) {
      cache = tomo.random_ggl;
      for (int i=0; i<redshift.clustering_nbin; i++) {
        for (int j=0; j<redshift.shear_nbin; j++) {
          N[i][j] = 1;
          for (int k=0; k<tomo.N_ggl_exclude; k++) {
            const int p = k*2+0;
            const int q = k*2+1;
            if ((i == tomo.ggl_exclude[p]) && 
                (j == tomo.ggl_exclude[q])) {
              N[i][j] = 0;
              break;
            }
          }
        }
      } 
    }
    return  N[ni][nj];
  }
  else {
    return 1;
  }
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Pair bookkeeping. Every 2-point data vector runs over a flat pair
// index n; the maps below convert between n and the tomographic bin
// pair (i, j). All enumerations are row-major (i outer, j inner):
//
//   GGL, admitted (lens i, source j) pairs only:
//     (i, j) -> test_zoverlap admits? -> counted in order -> n
//     n -> (ZL(n), ZS(n))          (i, j) -> N_ggl(i, j) (-1 = excluded)
//
//   shear-shear / clustering, unordered pairs with z1 <= z2:
//     n -> (Z1(n), Z2(n))          (i, j) -> N_shear(i, j)
//     n -> (ZCL1(n), ZCL2(n))      (i, j) -> N_CL(i, j)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Lens bin of the ni-th galaxy-galaxy lensing power spectrum.
//
// Admitted (lens, source) pairs are flattened in row-major order (lens
// outer, source inner, test_zoverlap deciding admission); returns the
// lens bin of pair ni.
//
// Cache invalidation:
// static map with sentinel N[0] = -42; rebuilds while
// N[0] < -1 (never built; built entries are bin indices >= 0) or when the
// stored tomo.random_ggl stamp changes.
//
// Warmed single-threaded by init_ntomo_powerspectra; the first call
// after a key change must happen outside OpenMP regions.
//
// Parameters:
//   ni - GGL power spectrum index (0 .. ggl_Npowerspectra-1)
//
// Returns:
//   lens (clustering) tomographic bin of that pair.
// ---------------------------------------------------------------------------
int ZL(int ni) 
{
  static int N[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS] = {-42};
  static uint64_t cache = 0; // tomo.random_ggl the map was built with
  if (N[0] < -1 || fdiff2(cache, tomo.random_ggl)) {
    cache = tomo.random_ggl;
    int n = 0;
    for (int i=0; i<redshift.clustering_nbin; i++) {
      for (int j=0; j<redshift.shear_nbin; j++) {
        if (test_zoverlap(i, j)) {
          N[n] = i;
          n++;
        }
      }
    }
  }
  
  if (ni < 0 || ni > tomo.ggl_Npowerspectra - 1)
  {
    log_fatal("invalid bin input ni = %d (max %d)", ni, tomo.ggl_Npowerspectra);
    exit(1);
  }
  return N[ni];
}

// ---------------------------------------------------------------------------
// Source bin of the nj-th galaxy-galaxy lensing power spectrum.
//
// Same admitted-pair flattening as ZL (lens outer, source inner); returns
// the source bin of pair nj.
//
// Cache invalidation:
// static map with sentinel N[0] = -42; rebuilds while
// N[0] < -1 (never built; built entries are bin indices >= 0) or when the
// stored tomo.random_ggl stamp changes.
//
// Warmed single-threaded by init_ntomo_powerspectra; the first call
// after a key change must happen outside OpenMP regions.
//
// Parameters:
//   nj - GGL power spectrum index (0 .. ggl_Npowerspectra-1)
//
// Returns:
//   source (shear) tomographic bin of that pair.
// ---------------------------------------------------------------------------
int ZS(int nj) 
{
  static int N[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS] = {-42};
  static uint64_t cache = 0; // tomo.random_ggl the map was built with
  if (N[0] < -1 || fdiff2(cache, tomo.random_ggl)) {
    cache = tomo.random_ggl;
    int n = 0;
    for (int i = 0; i < redshift.clustering_nbin; i++) {
      for (int j = 0; j < redshift.shear_nbin; j++) {
        if (test_zoverlap(i, j)) {
          N[n] = j;
          n++;
        }
      }
    }
  }

  if (nj < 0 || nj > tomo.ggl_Npowerspectra - 1)
  {
    log_fatal("invalid bin input nj = %d (max %d)", nj, tomo.ggl_Npowerspectra);
    exit(1);
  }
  return N[nj];
}

// ---------------------------------------------------------------------------
// GGL power spectrum index of lens bin ni and source bin nj.
//
// Inverse of (ZL, ZS): admitted pairs are numbered in row-major order
// (lens outer, source inner) and excluded pairs store -1.
//
// Cache invalidation:
// static map stamped with tomo.random_ggl. The
// rebuild guard is N[0][0] < -1, so the static-initializer sentinel (-42)
// triggers the first build and a changed stamp triggers later ones; when
// pair (0, 0) itself is excluded, its stored -1 keeps the guard true and
// the map is rebuilt on every call.
//
// Warmed single-threaded by init_ntomo_powerspectra; the first call
// after a key change must happen outside OpenMP regions.
//
// Parameters:
//   ni - lens (clustering) tomographic bin index
//   nj - source (shear) tomographic bin index
//
// Returns:
//   pair index (0 .. ggl_Npowerspectra-1), or -1 when the pair is
//   excluded.
// ---------------------------------------------------------------------------
int N_ggl(int ni, int nj)
{
  static int N[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS] = {{-42}};
  static uint64_t cache = 0; // tomo.random_ggl the map was built with
  if (N[0][0] < -1 || fdiff2(cache, tomo.random_ggl)) {
    cache = tomo.random_ggl;
    int n = 0;
    for (int i=0; i<redshift.clustering_nbin; i++) {
      for (int j=0; j<redshift.shear_nbin; j++) {
        if (test_zoverlap(i, j)) {
          N[i][j] = n;
          n++;
        } 
        else {
          N[i][j] = -1;
        }
      }
    }
  }
  if (ni < 0 || ni > redshift.clustering_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1)
  {
    log_fatal("invalid bin input (ni, nj) = (%d, %d)", ni, nj);
    exit(1);
  }
  return N[ni][nj];
}

// ---------------------------------------------------------------------------
// First source bin of the ni-th shear-shear power spectrum.
//
// Shear pairs (z1, z2) with z1 <= z2 are numbered in row-major order:
// (0,0), (0,1), ..., (0,nbin-1), (1,1), ...; returns z1 of pair ni.
//
// Cache invalidation:
// built once, on the first call (sentinel N[0] = -42,
// rebuilt while N[0] < -1; built entries are >= 0). There is no
// invalidation key: the layout depends only on redshift.shear_nbin.
// First call must happen outside OpenMP regions.
//
// Parameters:
//   ni - shear power spectrum index (0 .. shear_Npowerspectra-1)
//
// Returns:
//   first source bin z1 of that pair.
// ---------------------------------------------------------------------------
int Z1(int ni)
{
  static int N[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS] = {-42};
  if (N[0] < -1) 
  {
    int n = 0;
    for (int i=0; i < redshift.shear_nbin; i++) 
    {
      for (int j=i; j < redshift.shear_nbin; j++) 
      {
        N[n] = i;
        n++;
      }
    }
  }
  
  if (ni < 0 || ni > tomo.shear_Npowerspectra - 1)
  {
    log_fatal("invalid bin input ni = %d (max = %d)", 
      ni, tomo.shear_Npowerspectra);
    exit(1);
  }
  return N[ni];
}

// ---------------------------------------------------------------------------
// Second source bin of the nj-th shear-shear power spectrum.
//
// Same z1 <= z2 row-major pair layout as Z1; returns z2 of pair nj.
//
// Cache invalidation:
// built once, on the first call (sentinel N[0] = -42,
// rebuilt while N[0] < -1; built entries are >= 0). There is no
// invalidation key: the layout depends only on redshift.shear_nbin.
// First call must happen outside OpenMP regions.
//
// Parameters:
//   nj - shear power spectrum index (0 .. shear_Npowerspectra-1)
//
// Returns:
//   second source bin z2 of that pair.
// ---------------------------------------------------------------------------
int Z2(int nj)
{
  static int N[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS] = {-42};
  if (N[0] < -1) 
  {
    int n = 0;
    for (int i=0; i<redshift.shear_nbin; i++) 
    {
      for (int j=i; j<redshift.shear_nbin; j++) 
      {
        N[n] = j;
        n++;
      }
    }
  }
  
  if (nj < 0 || nj > tomo.shear_Npowerspectra - 1)
  {
    log_fatal("invalid bin input nj = %d (max = %d)", 
      nj, tomo.shear_Npowerspectra);
    exit(1);
  }
  return N[nj];
}

// ---------------------------------------------------------------------------
// Shear-shear power spectrum index of source bin pair (ni, nj).
//
// Inverse of (Z1, Z2) with symmetric storage, N[i][j] = N[j][i], so the
// bin order does not matter. Indices follow the Z1/Z2 row-major layout
// over pairs with z1 <= z2.
//
// Cache invalidation:
// built once, on the first call (sentinel
// N[0][0] = -42, rebuilt while N[0][0] < -1; built entries are >= 0).
// There is no invalidation key: the layout depends only on
// redshift.shear_nbin. First call must happen outside OpenMP regions.
//
// Parameters:
//   ni, nj - source tomographic bin indices (0 .. shear_nbin-1)
//
// Returns:
//   shear power spectrum index (0 .. shear_Npowerspectra-1).
// ---------------------------------------------------------------------------
int N_shear(int ni, int nj)
{
  static int N[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS] = {{-42}};
  if (N[0][0] < -1) 
  {
    int n = 0;
    for (int i=0; i<redshift.shear_nbin; i++) 
    {
      for (int j=i; j<redshift.shear_nbin; j++) 
      {
        N[i][j] = n;
        N[j][i] = n;
        n++;
      }
    }
  }

  const int ntomo = redshift.shear_nbin;
  if (ni < 0 || ni > ntomo - 1 || nj < 0 || nj > ntomo - 1)
  {
    log_fatal("invalid bin input (ni, nj) = (%d, %d) (max = %d)", ni, nj, ntomo);
    exit(1);
  }
  return N[ni][nj];
}

// ---------------------------------------------------------------------------
// First lens bin of the ni-th clustering bin pair.
//
// Clustering pairs (zcl1, zcl2) with zcl1 <= zcl2 are numbered in
// row-major order, as in Z1; returns zcl1 of pair ni. The bound check
// uses tomo.clustering_Npowerspectra.
//
// Cache invalidation:
// built once, on the first call (sentinel N[0] = -42,
// rebuilt while N[0] < -1; built entries are >= 0). There is no
// invalidation key: the layout depends only on redshift.clustering_nbin.
// First call must happen outside OpenMP regions.
//
// Parameters:
//   ni - clustering power spectrum index (0 .. clustering_Npowerspectra-1)
//
// Returns:
//   first lens bin zcl1 of that pair.
// ---------------------------------------------------------------------------
int ZCL1(int ni)
{
  static int N[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS] = {-42};
  if (N[0] < -1) 
  {
    int n = 0;
    for (int i=0; i<redshift.clustering_nbin; i++) 
    {
      for (int j=i; j<redshift.clustering_nbin; j++) 
      {
        N[n] = i;
        n++;
      }
    }
  }

  if (ni < 0 || ni > tomo.clustering_Npowerspectra - 1)
  {
    log_fatal("invalid bin input ni = %d (max %d)", ni, 
      tomo.clustering_Npowerspectra);
    exit(1);
  }
  return N[ni];
}

// ---------------------------------------------------------------------------
// Second lens bin of the nj-th clustering bin pair.
//
// Same zcl1 <= zcl2 row-major pair layout as ZCL1; returns zcl2 of pair
// nj. The bound check uses tomo.clustering_Npowerspectra.
//
// Cache invalidation:
// built once, on the first call (sentinel N[0] = -42,
// rebuilt while N[0] < -1; built entries are >= 0). There is no
// invalidation key: the layout depends only on redshift.clustering_nbin.
// First call must happen outside OpenMP regions.
//
// Parameters:
//   nj - clustering power spectrum index (0 .. clustering_Npowerspectra-1)
//
// Returns:
//   second lens bin zcl2 of that pair.
// ---------------------------------------------------------------------------
int ZCL2(int nj)
{
  static int N[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS] = {-42};
  if (N[0] < -1) 
  {
    int n = 0;
    for (int i=0; i<redshift.clustering_nbin; i++) 
    {
      for (int j=i; j<redshift.clustering_nbin; j++) 
      {
        N[n] = j;
        n++;
      }
    }
  }

  if (nj < 0 || nj > tomo.clustering_Npowerspectra - 1)
  {
    log_fatal("invalid bin input nj = %d (max %d)", nj, 
      tomo.clustering_Npowerspectra);
    exit(1);
  }
  return N[nj];
}

// ---------------------------------------------------------------------------
// Clustering pair index of lens bin pair (ni, nj).
//
// Inverse of (ZCL1, ZCL2) with symmetric storage, N[i][j] = N[j][i], over
// the zcl1 <= zcl2 row-major pair layout.
//
// Cache invalidation:
// built once, on the first call (sentinel
// N[0][0] = -42, rebuilt while N[0][0] < -1; built entries are >= 0).
// There is no invalidation key: the layout depends only on
// redshift.clustering_nbin. First call must happen outside OpenMP
// regions.
//
// Parameters:
//   ni, nj - lens tomographic bin indices (0 .. clustering_nbin-1)
//
// Returns:
//   pair index in the zcl1 <= zcl2 enumeration.
// ---------------------------------------------------------------------------
int N_CL(int ni, int nj)
{
  static int N[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS] = {{-42}};
  if (N[0][0] < -1) 
  {
    int n = 0;
    for (int i=0; i<redshift.clustering_nbin; i++) 
    {
      for (int j=i; j<redshift.clustering_nbin; j++) 
      {
        N[i][j] = n;
        N[j][i] = n;
        n++;
      }
    }
  }

  const int ntomo = redshift.clustering_nbin;
  if (ni < 0 || ni > ntomo - 1 ||  nj < 0 || nj > ntomo - 1)
  {
    log_fatal("invalid bin input (ni, nj) = (%d, %d) (max = %d)", ni, nj, ntomo);
    exit(1);
  }
  return N[ni][nj];
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Shear routines for redshift distributions
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Raw (unnormalized) histogram lookup for the source galaxy redshift
// distribution in tomography bin ni.
//
// Locates the histogram cell containing z by direct index on the uniform
// grid stored in redshift.shear_zdist_table (row ntomo holds the z nodes)
// and returns its stored density:
//
//   nj = floor((z - z_v[0]) / dz),  dz = (z_v[nzbins-1] - z_v[0])
//                                        / (nzbins - 1)
//
// Parameters:
//   z  - redshift
//   ni - source tomographic bin index (0 .. shear_nbin-1)
//
// Returns:
//   tabulated n(z) value; 0 outside [zmin_all, zmax_all).
// ---------------------------------------------------------------------------
double zdistr_histo_n(double z, const int ni)
{
  if (redshift.shear_zdist_table == NULL) {
    log_fatal("redshift n(z) not loaded");
    exit(1);
  } 
  double res = 0.0;
  if ((z >= redshift.shear_zdist_zmin_all) && (z<redshift.shear_zdist_zmax_all)) 
  {
    const int ntomo = redshift.shear_nbin;
    const int nzbins = redshift.shear_nzbins;
    double** tab = redshift.shear_zdist_table;
    double* z_v  = redshift.shear_zdist_table[ntomo];

    const double dz_histo = (z_v[nzbins - 1] - z_v[0]) / ((double) nzbins - 1.);
    const double zhisto_min = z_v[0];
    const int nj = (int) floor((z - zhisto_min) / dz_histo);
    if (ni < 0 || ni > ntomo-1 || nj < 0 || nj > nzbins-1) {
      log_fatal("invalid bin input (zbin = ni, bin = nj) = (%d, %d)", ni, nj);
      exit(1);
    } 
    res = tab[ni][nj];
  }
  return res;
}

// ---------------------------------------------------------------------------
// Photometric redshift distribution n(z) for source tomographic bin nj.
//
// PHYSICS:
//   Returns the normalized redshift distribution of source (background)
//   galaxies in tomographic bin nj, evaluated at redshift zz. This
//   function is called inside the Limber projection integrals for every
//   probe involving source galaxies (shear-shear, galaxy-shear, CMB
//   lensing × shear), where it enters through the radial weight
//   functions W_kappa, W_source, and g_tomo.
//
// PHOTO-Z MODEL:
//   The raw histogram n_raw(z, i) from the survey (accessed via
//   zdistr_histo_n) is normalized per bin:
//     n_i(z) = n_raw(z, i) / ∫ n_raw(z', i) dz'
//
//   A combined distribution (stored in table[0]) sums over all bins
//   weighted by their fractional contribution to the total sample.
//
//   At query time, a shift photo-z model is applied:
//     z → z − Δz_i
//   where Δz_i = nuisance.photoz[0][0][nj] is the per-bin shift.
//   Unlike nz_lens_photoz, there is no stretch factor — only a shift.
//
// NUMERICAL SCHEME:
//   Two-stage interpolation for speed on the hot path:
//
//   raw histogram -> normalize per bin -> GSL spline on the node grid
//     -> resample onto a uniform fine grid
//     -> tridiagonal solve for the cubic coefficients
//     -> hot path: direct-index lookup + Horner cubic
//
//   Stage 1 (cache rebuild, runs once when redshift.random_shear
//   changes) covers the first four links. The GSL spline type is set
//   by Ntable.photoz_interpolation_type (cspline default, linear, or
//   Steffen monotone). The node positions follow
//   Ntable.photoz_zmid_convention: the file z column is read as Z_LOW
//   left bin edges (default, values at centers z + dz/2) or as Z_MID
//   sample points (values at z). The fine grid holds
//   Ntable.nz_fine_sampling_factor x the original resolution, with
//   cubic coefficients from spline_coeffs_uniform. Defining
//   DONT_NZ_FAST_SUMBSAMPLE skips the fine grid and sends the hot
//   path back to GSL evaluation.
//
//   Stage 2 (hot path, called millions of times per likelihood): one
//   multiply computes the fine-grid index (no binary search, no GSL
//   function-pointer dispatch), then a Horner-form cubic.
//
// Cache invalidation:
//   Recomputes when redshift.random_shear changes, which tracks:
//     - source n(z) histogram values
//     - number of bins, redshift range, bin edges
//   Also recomputes when Ntable.photoz_interpolation_type or
//   Ntable.photoz_zmid_convention change, via the packed settings key
//   described above the rebuild condition, and when
//   Ntable.nz_fine_sampling_factor changes (init_accuracy_boost scales
//   it; without this key a boost set after the n(z) is loaded would
//   keep the fine grid of the old factor).
//
// Parameters:
//   zz — redshift at which to evaluate (before photo-z shift)
//   nj — source tomographic bin index (0 .. shear_nbin − 1)
//
// Returns:
//   n(z − Δz_nj, nj). Returns 0 outside the tabulated range.
// ---------------------------------------------------------------------------
double nz_source_photoz(double zz, const int nj)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static gsl_interp* photoz_splines[MAX_SIZE_ARRAYS+1];
// DONT_NZ_FAST_SUMBSAMPLE (sic - SUBSAMPLE) is the escape hatch:
// defining it removes the uniform fine grid, and the hot path falls
// back to GSL spline evaluation on the node grid.
#ifndef DONT_NZ_FAST_SUMBSAMPLE
  // uniform fine grid for direct-index evaluation (no binary search)
  // fine[0] = table_fine (y values), fine[1] = c_fine (spline coefficients)
  static double*** fine = NULL;
  static double zmin_fine;
  static double inv_dz_fine;
  static int nzbins_fine;
#endif

  // The two photo-z settings are packed into ONE integer so a single
  // comparison detects a change in either of them:
  //
  //   packed = 1 + photoz_interpolation_type + 8*photoz_zmid_convention
  //
  // Read it like a two-digit number whose "ones digit" is the
  // interpolation type (0-7) and whose "eights digit" is the z-column
  // convention: type 2 with convention 0 gives 1 + 2 + 0 = 3, while
  // type 0 with convention 1 gives 1 + 0 + 8 = 9.
  //
  // Because the type can never reach 8, two DIFFERENT settings pairs
  // can never produce the SAME packed number. (With a multiplier of 2
  // they could: type 2 with convention 0 and type 0 with convention 1
  // would both give 3, and that settings change would be mistaken for
  // "nothing changed" - stale n(z) tables, silently wrong physics.)
  //
  // The 1+ offset keeps a stamped slot nonzero, so it can never equal
  // the zero that C puts in the static cache array before the first
  // build.
  //
  // cache[2] holds the fine-sampling factor the fine grid was built
  // with. It is keyed on its value, not on Ntable.random: Ntable.random
  // is redrawn by every Ntable setter (init_binning, ...), and the n(z)
  // tables must not rebuild for changes that do not touch them.
  if (table == NULL || fdiff2(cache[0], redshift.random_shear) ||
      cache[1] != (uint64_t) (1 + Ntable.photoz_interpolation_type
                                + 8*Ntable.photoz_zmid_convention) ||
      cache[2] != (uint64_t) Ntable.nz_fine_sampling_factor) {
    if (table == NULL) {
      for (int i = 0; i < MAX_SIZE_ARRAYS+1; i++)
        photoz_splines[i] = NULL;
    }
    const int ntomo  = redshift.shear_nbin;
    const int nzbins = redshift.shear_nzbins;

    if (table != NULL) free(table);
    table = (double**) malloc2d(ntomo + 2, nzbins);
    const double zmin = redshift.shear_zdist_zmin_all;
    const double zmax = redshift.shear_zdist_zmax_all;
    const double dz_histo = (zmax - zmin) / ((double) nzbins);
    // dz_histo here is the width of the nzbins cells spanning
    // [zmin_all, zmax_all). It equals the node spacing zdistr_histo_n
    // computes as range/(nzbins - 1) over the nzbins file nodes,
    // because the loader places zmax_all one node spacing past the
    // last file node.
    //
    // Z_LOW convention (default): the file z column holds left bin edges,
    // so the tabulated value belongs at the cell center z + dz/2.
    // Z_MID: the column holds the sample points themselves.
    const double off = (1 == Ntable.photoz_zmid_convention) ? 0.0 : 0.5;
    for (int k = 0; k < nzbins; k++) {
      table[ntomo+1][k] = zmin + (k + off) * dz_histo;
    }

    // Row layout of table (ntomo + 2 rows, nzbins columns):
    //
    //   table[0]        = combined n(z) of the whole sample
    //   table[1..ntomo] = per-bin n(z), each normalized to unit integral
    //   table[ntomo+1]  = the z nodes shared by every row
    //
    // NORM[i] = bin i's raw-histogram integral (cell sum times width
    // dz_histo) and norm = sum_i NORM[i], the whole sample's. Dividing
    // bin i by NORM[i] makes rows 1..ntomo unit-normalized; the
    // combined row then reweights them by the sample fractions
    // NORM[i]/norm:
    //
    //   table[0][k] = sum_i (raw_i(z_k)/NORM[i]) * (NORM[i]/norm)
    //               = sum_i raw_i(z_k) / norm
    //
    // the raw histograms stacked, normalized to unit total integral.
    double NORM[MAX_SIZE_ARRAYS];
    double norm = 0;
    #pragma omp parallel for reduction( + : norm )
    for (int i = 0; i < ntomo; i++) {
      NORM[i] = 0.0;
      for (int k = 0; k < nzbins; k++) {
        const double z = table[ntomo+1][k];
        NORM[i] += zdistr_histo_n(z, i) * dz_histo;
      }
      if (!(NORM[i] > 0.0)) {
        log_fatal("zero/negative n(z) normalization for source bin %d", i);
        exit(1);
      }
      norm += NORM[i];
    }
    if (!(norm > 0.0)) {
      log_fatal("zero/negative total n(z) normalization");
      exit(1);
    }

    #pragma omp parallel for schedule(static)
    for (int k = 0; k < nzbins; k++) {
      table[0][k] = 0;
      for (int i = 0; i < ntomo; i++) {
        const double z = table[ntomo+1][k];
        table[i + 1][k] = zdistr_histo_n(z, i) / NORM[i];
        table[0][k] += table[i+1][k] * NORM[i] / norm;
      }
    }

    for (int i = 0; i < ntomo+1; i++) {
      if (photoz_splines[i] != NULL) gsl_interp_free(photoz_splines[i]);
      photoz_splines[i] = malloc_gsl_interp(nzbins);
    }
#ifndef DONT_NZ_FAST_SUMBSAMPLE
    // -----------------------------------------------------------------
    // Resample each spline onto a uniform fine grid. The GSL spline on
    // the node grid is evaluated once here; the hot path below uses
    // direct-index lookup with no binary search.
    // -----------------------------------------------------------------
    nzbins_fine = Ntable.nz_fine_sampling_factor*nzbins + 1;
    // eps nudges both fine-grid endpoints just inside the spline's
    // native domain [z_0, z_{nzbins-1}], so rounding in the node
    // formula below can never hand gsl_interp_eval_e a point outside
    // it (outside the domain, GSL fails with GSL_EDOM).
    const double eps = 1e-15;
    zmin_fine = table[ntomo+1][0] + eps;
    const double zmax_fine = table[ntomo+1][nzbins - 1] - eps;
    const double dz_fine = (zmax_fine - zmin_fine) / ((double) nzbins_fine - 1);
    inv_dz_fine = 1.0 / dz_fine;

    if (fine != NULL) free(fine);
    fine = (double***) malloc3d(2, ntomo + 1, nzbins_fine);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ntomo+1; i++) {
      int status = gsl_interp_init(photoz_splines[i],
                                   table[ntomo+1],
                                   table[i],
                                   nzbins);
      if (status) {
        log_fatal(gsl_strerror(status));
        exit(1);
      }
      for (int k = 0; k < nzbins_fine; k++) {
        const double z = (k < nzbins_fine - 1)
          ? zmin_fine + k * dz_fine : zmax_fine;
        int status = gsl_interp_eval_e(photoz_splines[i],
                                       table[ntomo+1], table[i],
                                       z, NULL, &fine[0][i][k]);
        if (status) {
          printf("gsl_interp_eval_e failed: bin=%d k=%d z=%.10e status=%s\n",
                 i, k, z, gsl_strerror(status));
          exit(1);
        }
      }
      spline_coeffs_uniform(fine[0][i], nzbins_fine, dz_fine, fine[1][i]);
    }
#else
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ntomo+1; i++) {
      int status = gsl_interp_init(photoz_splines[i],
                                   table[ntomo+1],
                                   table[i],
                                   nzbins);
      if (status) {
        log_fatal(gsl_strerror(status));
        exit(1);
      }
    }
#endif
    cache[0] = redshift.random_shear;
    // same packed encoding as the rebuild condition above
    cache[1] = (uint64_t) (1 + Ntable.photoz_interpolation_type
                             + 8*Ntable.photoz_zmid_convention);
    cache[2] = (uint64_t) Ntable.nz_fine_sampling_factor;
  }

  const int ntomo = redshift.shear_nbin;
  if (nj < 0 || nj > ntomo - 1) {
    log_fatal("nj = %d bin outside range (max = %d)", nj, ntomo);
    exit(1);
  }

  zz = zz - nuisance.photoz[0][0][nj];

#ifdef DONT_NZ_FAST_SUMBSAMPLE
  const int nzbins = redshift.shear_nzbins;
  double res;
  if (zz <= table[ntomo+1][0] || zz >= table[ntomo+1][nzbins - 1]) {
    res = 0.0;
  }
  else {
    int status = gsl_interp_eval_e(photoz_splines[nj+1],
                                   table[ntomo+1],
                                   table[nj+1],
                                   zz, NULL, &res);
    if (status) {
      log_fatal(gsl_strerror(status));
      exit(1);
    }
  }
  return res;
#else
  // out-of-support test; the upper bound rebuilds the grid edge:
  // zmin_fine + (nzbins_fine - 1)*dz_fine = zmax_fine
  if (zz <= zmin_fine || zz >= zmin_fine + (nzbins_fine - 1) / inv_dz_fine) {
    return 0.0;
  }
  // -----------------------------------------------------------------------
  // Hot-path cubic spline evaluation on the uniform fine grid.
  //
  // Direct-index lookup: the uniform spacing turns the interval search
  // into one multiply (no binary search, no accelerator):
  //
  //   r     = fractional grid index (floating point)
  //   index = integer part -> left node of the bracketing interval
  //   delx  = (r - index) * dx -> offset from that node, in z
  //
  // On interval [z_i, z_i + dx] the house spline (spline_coeffs_uniform)
  // is the cubic
  //
  //   S(z_i + delx) = y_i + b delx + c_i delx^2 + d delx^3
  //
  // where c = fine[1][..] is the coefficient array the tridiagonal
  // solve in the rebuild produced: the spline's second derivative / 2,
  // with natural boundaries c_0 = c_{n-1} = 0.
  //
  // The other two coefficients follow from two conditions:
  //
  //   S'' runs linearly from 2 c_i to 2 c_{i+1}
  //     -> d = (c_{i+1} - c_i) / (3 dx)
  //
  //   S(z_{i+1}) = y_{i+1}, interpolate the right node
  //     -> b = (y_{i+1} - y_i)/dx - dx (c_{i+1} + 2 c_i)/3
  //
  // The polynomial is evaluated in Horner form.
  // -----------------------------------------------------------------------
  const double r = (zz - zmin_fine) * inv_dz_fine;
  const int index = (int) r;
  const double dx = 1.0 / inv_dz_fine;
  const double delx = (r - index) * dx;
  const double dy = fine[0][nj+1][index+1] - fine[0][nj+1][index];
  const double c_i  = fine[1][nj+1][index];
  const double c_i1 = fine[1][nj+1][index+1];
  const double b = (dy * inv_dz_fine) - dx * (c_i1 + 2.0 * c_i) / 3.0;
  const double d = (c_i1 - c_i) / (3.0 * dx);
  return fine[0][nj+1][index] + delx * (b + delx * (c_i + delx * d));
#endif
}

// ---------------------------------------------------------------------------
// Integrand for the mean source redshift: z * n_j(z).
//
// GSL quadrature callback used by zmean_source, with n_j =
// nz_source_photoz.
//
// Parameters:
//   z      - redshift
//   params - double[1]; params[0] = source bin index (cast to double)
//
// Returns:
//   z * nz_source_photoz(z, bin).
// ---------------------------------------------------------------------------
double int_for_zmean_source(double z, void* params)
{
  double* ar = (double*) params;
  const int ni = (int) ar[0];
  
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni); exit(1);
  } 
  return z * nz_source_photoz(z, ni);
}

// ---------------------------------------------------------------------------
// Mean true redshift of source galaxies in tomography bin ni.
//
//   zmean_source(ni) = int_{zmin[ni]}^{zmax[ni]} z * n_i(z) dz
//
// over the bin's tabulated range, with n_i = nz_source_photoz (already
// normalized to unit integral, so no norm division is needed). All bins
// are tabulated at once with fixed-order Gauss-Legendre quadrature
// (256/512/1024 nodes, chosen by |Ntable.high_def_integration|).
//
// Cache invalidation:
// the table rebuilds when Ntable.random or
// redshift.random_shear change. nz_source_photoz is warmed
// single-threaded before the parallel loop over bins.
//
// Parameters:
//   ni - source tomographic bin index (0 .. shear_nbin-1)
//
// Returns:
//   tabulated mean redshift of bin ni.
// ---------------------------------------------------------------------------
double zmean_source(int ni)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* table = NULL;
  static gsl_integration_glfixed_table* w = NULL;

  if (table == NULL || 
      fdiff2(cache[0], Ntable.random) ||
      fdiff2(cache[1], redshift.random_shear))
  {    
    if (table != NULL) free(table);
    table = (double*) malloc1d(redshift.shear_nbin);
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 256 : 
                         (1 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);

    (void) nz_source_photoz(0., 0); // init static variables
    #pragma omp parallel for schedule(static)
    for (int i=0; i<redshift.shear_nbin; i++) {
      double ar[1] = {(double) i};
      gsl_function F;
      F.params = ar;
      F.function = int_for_zmean_source;
      table[i] = gsl_integration_glfixed(&F, redshift.shear_zdist_zmin[i], 
                                             redshift.shear_zdist_zmax[i], w);
    }
    cache[0] = Ntable.random;
    cache[1] = redshift.random_shear;
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni); exit(1);
  }
  return table[ni];
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Lenses routines for redshift distributions
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Raw (unnormalized) histogram lookup for the lens galaxy redshift
// distribution in tomography bin ni.
//
// Locates the histogram cell containing z by direct index on the uniform
// grid stored in redshift.clustering_zdist_table (row ntomo holds the z
// nodes) and returns its stored density:
//
//   nj = floor((z - z_v[0]) / dz),  dz = (z_v[nzbins-1] - z_v[0])
//                                        / (nzbins - 1)
//
// Parameters:
//   z  - redshift
//   ni - lens tomographic bin index (0 .. clustering_nbin-1)
//
// Returns:
//   tabulated n(z) value; 0 outside [zmin_all, zmax_all).
// ---------------------------------------------------------------------------
double pf_histo_n(double z, const int ni)
{

  if (redshift.clustering_zdist_table == NULL) 
  {
    log_fatal("redshift n(z) not loaded");
    exit(1);
  } 
  
  double res = 0.0;
  if ((z >= redshift.clustering_zdist_zmin_all) && 
      (z < redshift.clustering_zdist_zmax_all)) 
  {
    
    const int ntomo = redshift.clustering_nbin;             // alias
    const int nzbins = redshift.clustering_nzbins;          // alias
    double** tab = redshift.clustering_zdist_table;         // alias
    double* z_v = redshift.clustering_zdist_table[ntomo];   // alias
    
    const double dz_histo = (z_v[nzbins - 1] - z_v[0]) / ((double) nzbins - 1.);
    const double zhisto_min = z_v[0];
    const int nj = (int) floor((z - zhisto_min) / dz_histo);
    
    if (ni < 0 || ni > ntomo - 1 || nj < 0 || nj > nzbins - 1) {
      log_fatal("invalid bin input (zbin = ni, bin = nj) = (%d, %d)", ni, nj);
      exit(1);
    } 
    res = tab[ni][nj];
  }
  return res;
}

// ---------------------------------------------------------------------------
// Photometric redshift distribution n(z) for lens tomographic bin nj.
//
// PHYSICS:
//   Returns the normalized redshift distribution of lens (foreground)
//   galaxies in tomographic bin nj, evaluated at redshift zz. This
//   function is called inside the Limber projection integrals for every
//   probe involving lens galaxies (galaxy-galaxy lensing, galaxy
//   clustering, galaxy–CMB lensing), where it enters through the
//   radial weight functions W_gal and g_lens.
//
// PHOTO-Z MODEL:
//   The raw histogram n_raw(z, i) from the survey (accessed via
//   pf_histo_n) is normalized per bin:
//     n_i(z) = n_raw(z, i) / ∫ n_raw(z', i) dz'
//
//   A combined distribution (stored in table[0]) sums over all bins
//   weighted by their fractional contribution to the total sample.
//
//   At query time, a shift-and-stretch photo-z model is applied:
//     z → (z − Δz_i − z̄_i) / σ_i + z̄_i
//   where Δz_i = nuisance.photoz[1][0][nj] is the per-bin shift,
//   σ_i = nuisance.photoz[1][1][nj] is the per-bin stretch, and
//   z̄_i = clustering_zdist_zmean[nj] is the fiducial mean redshift.
//   The returned value is divided by σ_i to preserve normalization
//   under the stretch (∫ n(z) dz = 1).
//
// NUMERICAL SCHEME:
//   Two-stage interpolation for speed on the hot path:
//
//   raw histogram -> normalize per bin -> GSL spline on the node grid
//     -> resample onto a uniform fine grid
//     -> tridiagonal solve for the cubic coefficients
//     -> hot path: direct-index lookup + Horner cubic
//
//   Stage 1 (cache rebuild, runs once when redshift.random_clustering
//   changes) covers the first four links. The GSL spline type is set
//   by Ntable.photoz_interpolation_type (cspline default, linear, or
//   Steffen monotone). The node positions follow
//   Ntable.photoz_zmid_convention: the file z column is read as Z_LOW
//   left bin edges (default, values at centers z + dz/2) or as Z_MID
//   sample points (values at z). The fine grid holds
//   Ntable.nz_fine_sampling_factor x the original resolution, with
//   cubic coefficients from spline_coeffs_uniform. Defining
//   DONT_NZ_FAST_SUMBSAMPLE skips the fine grid and sends the hot
//   path back to GSL evaluation.
//
//   Stage 2 (hot path, called millions of times per likelihood): one
//   multiply computes the fine-grid index (no binary search, no GSL
//   function-pointer dispatch), then a Horner-form cubic.
//
// Cache invalidation:
//   Recomputes when redshift.random_clustering changes, which tracks:
//     - lens n(z) histogram values
//     - number of bins, redshift range, bin edges
//   Also recomputes when Ntable.photoz_interpolation_type or
//   Ntable.photoz_zmid_convention change, via the packed settings key
//   described above the rebuild condition, and when
//   Ntable.nz_fine_sampling_factor changes (as in nz_source_photoz).
//
// Parameters:
//   zz — redshift at which to evaluate (before photo-z transformation)
//   nj — lens tomographic bin index (0 .. clustering_nbin − 1)
//
// Returns:
//   n(z_transformed, nj) / σ_nj, the stretch-corrected normalized
//   redshift distribution. Returns 0 outside the tabulated range.
// ---------------------------------------------------------------------------
double nz_lens_photoz(double zz, int nj)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static gsl_interp* photoz_splines[MAX_SIZE_ARRAYS+1];
#ifndef DONT_NZ_FAST_SUMBSAMPLE
  // uniform fine grid for direct-index evaluation (no binary search)
  // fine[0] = table_fine (y values), fine[1] = c_fine (spline coefficients)
  static double*** fine = NULL;
  static double zmin_fine;
  static double inv_dz_fine;
  static int nzbins_fine;
#endif

  // The two photo-z settings are packed into ONE integer so a single
  // comparison detects a change in either of them:
  //
  //   packed = 1 + photoz_interpolation_type + 8*photoz_zmid_convention
  //
  // Read it like a two-digit number whose "ones digit" is the
  // interpolation type (0-7) and whose "eights digit" is the z-column
  // convention: type 2 with convention 0 gives 1 + 2 + 0 = 3, while
  // type 0 with convention 1 gives 1 + 0 + 8 = 9.
  //
  // Because the type can never reach 8, two DIFFERENT settings pairs
  // can never produce the SAME packed number. (With a multiplier of 2
  // they could: type 2 with convention 0 and type 0 with convention 1
  // would both give 3, and that settings change would be mistaken for
  // "nothing changed" - stale n(z) tables, silently wrong physics.)
  //
  // The 1+ offset keeps a stamped slot nonzero, so it can never equal
  // the zero that C puts in the static cache array before the first
  // build.
  //
  // cache[2]: the fine-sampling factor of the fine grid, keyed on its
  // value (see nz_source_photoz).
  if (NULL == table || fdiff2(cache[0], redshift.random_clustering) ||
      cache[1] != (uint64_t) (1 + Ntable.photoz_interpolation_type
                                + 8*Ntable.photoz_zmid_convention) ||
      cache[2] != (uint64_t) Ntable.nz_fine_sampling_factor)
  {
    if (table == NULL) {
      for (int i = 0; i < MAX_SIZE_ARRAYS+1; i++) {
        photoz_splines[i] = NULL;
      }
    }
    const int ntomo  = redshift.clustering_nbin;
    const int nzbins = redshift.clustering_nzbins;

    if (table != NULL) free(table);
    table = (double**) malloc2d(ntomo + 2, nzbins);
    const double zmin = redshift.clustering_zdist_zmin_all;
    const double zmax = redshift.clustering_zdist_zmax_all;
    const double dz_histo = (zmax - zmin) / ((double) nzbins);
    // dz_histo here is the width of the nzbins cells spanning
    // [zmin_all, zmax_all). It equals the node spacing pf_histo_n
    // computes as range/(nzbins - 1) over the nzbins file nodes,
    // because the loader places zmax_all one node spacing past the
    // last file node.
    //
    // Z_LOW convention (default): the file z column holds left bin edges,
    // so the tabulated value belongs at the cell center z + dz/2.
    // Z_MID: the column holds the sample points themselves.
    const double off = (1 == Ntable.photoz_zmid_convention) ? 0.0 : 0.5;
    for (int k = 0; k < nzbins; k++) {
      table[ntomo+1][k] = zmin + (k + off) * dz_histo;
    }

    // Row layout of table (ntomo + 2 rows, nzbins columns):
    //
    //   table[0]        = combined n(z) of the whole sample
    //   table[1..ntomo] = per-bin n(z), each normalized to unit integral
    //   table[ntomo+1]  = the z nodes shared by every row
    //
    // NORM[i] = bin i's raw-histogram integral (cell sum times width
    // dz_histo) and norm = sum_i NORM[i], the whole sample's. Dividing
    // bin i by NORM[i] makes rows 1..ntomo unit-normalized; the
    // combined row then reweights them by the sample fractions
    // NORM[i]/norm:
    //
    //   table[0][k] = sum_i (raw_i(z_k)/NORM[i]) * (NORM[i]/norm)
    //               = sum_i raw_i(z_k) / norm
    //
    // the raw histograms stacked, normalized to unit total integral.
    double NORM[MAX_SIZE_ARRAYS];
    double norm = 0;
    #pragma omp parallel for reduction( + : norm )
    for (int i = 0; i < ntomo; i++) {
      NORM[i] = 0.0;
      for (int k = 0; k < nzbins; k++) {
        const double z = table[ntomo+1][k];
        NORM[i] += pf_histo_n(z, i) * dz_histo;
      }
      if (!(NORM[i] > 0.0)) {
        log_fatal("zero/negative n(z) normalization for source bin %d", i);
        exit(1);
      }
      norm += NORM[i];
    }

    if (!(norm > 0.0)) {
      log_fatal("zero/negative total n(z) normalization"); exit(1);
    }

    #pragma omp parallel for schedule(static)
    for (int k=0; k<nzbins; k++) {
      table[0][k] = 0;
      for (int i=0; i<ntomo; i++) {
        const double z = table[ntomo+1][k];
        table[i + 1][k] = pf_histo_n(z, i) / NORM[i];
        table[0][k] += table[i+1][k] * NORM[i] / norm;
      }
    }
    
    for (int i = 0; i < ntomo+1; i++) {
      if (photoz_splines[i] != NULL) {
        gsl_interp_free(photoz_splines[i]);
      }
      photoz_splines[i] = malloc_gsl_interp(nzbins);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ntomo+1; i++) {
      int status = gsl_interp_init(photoz_splines[i],
                                   table[ntomo+1],
                                   table[i],
                                   nzbins);
      if (status) {
        log_fatal(gsl_strerror(status)); exit(1);
      }
    }
#ifndef DONT_NZ_FAST_SUMBSAMPLE
    // -----------------------------------------------------------------
    // Resample each spline onto a uniform fine grid. The GSL spline on
    // the node grid is evaluated once here; the hot path below uses
    // direct-index lookup with no binary search.
    // -----------------------------------------------------------------
    nzbins_fine = Ntable.nz_fine_sampling_factor*nzbins + 1;
    
    // eps nudges both fine-grid endpoints just inside the spline's
    // native domain [z_0, z_{nzbins-1}], so rounding in the node
    // formula below can never hand gsl_interp_eval_e a point outside
    // it (outside the domain, GSL fails with GSL_EDOM).
    const double eps = 1e-15;
    zmin_fine = table[ntomo+1][0] + eps;
    const double zmax_fine = table[ntomo+1][nzbins - 1] - eps;
    const double dz_fine = (zmax_fine - zmin_fine) / ((double) nzbins_fine - 1);
    inv_dz_fine = 1.0 / dz_fine;

    if (fine != NULL) {
      free(fine);
    }
    fine = (double***) malloc3d(2, ntomo + 1, nzbins_fine);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < ntomo+1; i++) {
      for (int k = 0; k < nzbins_fine; k++) {
        const double z = (k < nzbins_fine - 1)
          ? zmin_fine + k * dz_fine : zmax_fine; // clamp last point to exact endpoint
        int status = gsl_interp_eval_e(photoz_splines[i],
                          table[ntomo+1], table[i],
                          z, NULL, &fine[0][i][k]);
        if (status) {
          log_fatal(gsl_strerror(status)); exit(1);
        }
      }
      spline_coeffs_uniform(fine[0][i], nzbins_fine, dz_fine, fine[1][i]);
    }
#endif
    cache[0] = redshift.random_clustering;
    // same packed encoding as the rebuild condition above
    cache[1] = (uint64_t) (1 + Ntable.photoz_interpolation_type
                             + 8*Ntable.photoz_zmid_convention);
    cache[2] = (uint64_t) Ntable.nz_fine_sampling_factor;
  }

  const int ntomo = redshift.clustering_nbin;
  if (nj < 0 || nj > ntomo - 1) {
    log_fatal("nj = %d bin outside range (max = %d)", nj, ntomo);
    exit(1);
  }
  zz = (zz - nuisance.photoz[1][0][nj]
           - redshift.clustering_zdist_zmean[nj]) / nuisance.photoz[1][1][nj]
       + redshift.clustering_zdist_zmean[nj];

#ifdef DONT_NZ_FAST_SUMBSAMPLE
  const int nzbins = redshift.clustering_nzbins;
  double res; 
  if (zz <= table[ntomo+1][0] || zz >= table[ntomo+1][nzbins - 1]) { // z_v = table[ntomo+1]
    res = 0.0;
  }
  else {
    int status = gsl_interp_eval_e(photoz_splines[nj+1], 
                                   table[ntomo+1],
                                   table[nj+1],
                                   zz, 
                                   NULL, 
                                   &res);
    if (status) {
      log_fatal(gsl_strerror(status));
      exit(1);
    }
    res = res / nuisance.photoz[1][1][nj];
  }
  return res;
#else
  // -----------------------------------------------------------------------
  // Hot-path cubic spline evaluation on the uniform fine grid.
  //
  // Direct-index lookup: the uniform spacing turns the interval search
  // into one multiply (no binary search, no accelerator):
  //
  //   r     = fractional grid index (floating point)
  //   index = integer part -> left node of the bracketing interval
  //   delx  = (r - index) * dx -> offset from that node, in z
  //
  // On interval [z_i, z_i + dx] the house spline (spline_coeffs_uniform)
  // is the cubic
  //
  //   S(z_i + delx) = y_i + b delx + c_i delx^2 + d delx^3
  //
  // where c = fine[1][..] is the coefficient array the tridiagonal
  // solve in the rebuild produced: the spline's second derivative / 2,
  // with natural boundaries c_0 = c_{n-1} = 0.
  //
  // The other two coefficients follow from two conditions:
  //
  //   S'' runs linearly from 2 c_i to 2 c_{i+1}
  //     -> d = (c_{i+1} - c_i) / (3 dx)
  //
  //   S(z_{i+1}) = y_{i+1}, interpolate the right node
  //     -> b = (y_{i+1} - y_i)/dx - dx (c_{i+1} + 2 c_i)/3
  //
  // The polynomial is evaluated in Horner form. The final division by
  // nuisance.photoz[1][1][nj] applies the photo-z stretch factor to
  // the interpolated n(z) value.
  // -----------------------------------------------------------------------
  // out-of-support test; the upper bound rebuilds the grid edge:
  // zmin_fine + (nzbins_fine - 1)*dz_fine = zmax_fine
  if (zz <= zmin_fine || zz >= zmin_fine + (nzbins_fine - 1) / inv_dz_fine) {
    return 0.0;
  }
  const double r = (zz - zmin_fine) * inv_dz_fine;
  const int index = (int) r;
  const double dx = 1.0 / inv_dz_fine;
  const double delx = (r - index) * dx;
  const double dy = fine[0][nj+1][index+1] - fine[0][nj+1][index];
  const double c_i = fine[1][nj+1][index];
  const double c_i1 = fine[1][nj+1][index+1];
  const double b = (dy * inv_dz_fine) - dx * (c_i1 + 2.0 * c_i) / 3.0;
  const double d = (c_i1 - c_i) / (3.0 * dx);
  double res = fine[0][nj+1][index] + delx * (b + delx * (c_i + delx * d));
  res = res / nuisance.photoz[1][1][nj];
  return res;
#endif
}

// ---------------------------------------------------------------------------
// Integrand for the mean lens redshift: z * n_j(z).
//
// GSL quadrature callback used by zmean, with n_j = nz_lens_photoz.
//
// Parameters:
//   z      - redshift
//   params - double[1]; params[0] = lens bin index (cast to double)
//
// Returns:
//   z * nz_lens_photoz(z, bin).
// ---------------------------------------------------------------------------
double int_for_zmean(double z, void* params)
{
  double* ar = (double*) params;
  const int ni = (int) ar[0];
  
  if (ni < 0 || ni > redshift.clustering_nbin - 1)
  {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  } 
  return z * nz_lens_photoz(z, ni);
}

// ---------------------------------------------------------------------------
// Normalization integrand for the mean lens redshift: n_j(z).
//
// GSL quadrature callback used by zmean to compute the denominator
// int n_j(z) dz, with n_j = nz_lens_photoz.
//
// Parameters:
//   z      - redshift
//   params - double[1]; params[0] = lens bin index (cast to double)
//
// Returns:
//   nz_lens_photoz(z, bin).
// ---------------------------------------------------------------------------
double norm_for_zmean(double z, void* params)
{
  double* ar = (double*) params;
  const int ni = (int) ar[0];
  
  if (ni < 0 || ni > redshift.clustering_nbin - 1)
  {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  } 
  return nz_lens_photoz(z, ni);
}

// ---------------------------------------------------------------------------
// Mean true redshift of lens galaxies in tomography bin ni.
//
//   zmean(ni) = int z * n_i(z) dz / int n_i(z) dz
//
// over the bin's tabulated range [zdist_zmin[ni], zdist_zmax[ni]], with
// n_i = nz_lens_photoz. Unlike zmean_source, this explicitly divides by
// the norm because nz_lens_photoz includes a stretch factor that breaks
// unit normalization. All bins are tabulated at once with fixed-order
// Gauss-Legendre quadrature (256/512/1024 nodes, chosen by
// |Ntable.high_def_integration|).
//
// Cache invalidation:
// the table rebuilds when Ntable.random or
// redshift.random_clustering change. nz_lens_photoz is warmed
// single-threaded before the parallel loop over bins.
//
// Parameters:
//   ni - lens tomographic bin index (0 .. clustering_nbin-1)
//
// Returns:
//   tabulated mean redshift of bin ni.
// ---------------------------------------------------------------------------
double zmean(const int ni)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* table = NULL;
  static gsl_integration_glfixed_table* w = NULL;

  if (table == NULL || 
      fdiff2(cache[0], Ntable.random) ||
      fdiff2(cache[1], redshift.random_clustering))
  {
    if (table != NULL) free(table);
    // one slot of slack: only the first clustering_nbin entries are
    // ever written or read
    table = (double*) malloc1d(redshift.clustering_nbin+1);
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 256 : 
                         (1 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);

    (void) nz_lens_photoz(0., 0); // init static vars
    #pragma omp parallel for schedule(static)
    for (int i=0; i<redshift.clustering_nbin; i++) {
      double ar[1] = {(double) i};
      gsl_function F;
      F.params = ar;
      
      F.function = int_for_zmean;
      const double num = gsl_integration_glfixed(&F, 
                                          redshift.clustering_zdist_zmin[i], 
                                          redshift.clustering_zdist_zmax[i], w);
      F.function = norm_for_zmean;
      const double den = gsl_integration_glfixed(&F, 
                                          redshift.clustering_zdist_zmin[i], 
                                          redshift.clustering_zdist_zmax[i], w);
      if (!(den > 0.0)) {
        log_fatal("zmean denominator is non-positive (lens bin %d)", i);
        exit(1);
      }
      table[i] = num/den;
    }
    cache[0] = Ntable.random;
    cache[1] = redshift.random_clustering;
  }

  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni); exit(1);
  }  
  return table[ni];
}

// ---------------------------------------------------------------------------
// Lensing efficiency g(a) for source tomographic bin ni.
//
// PHYSICS:
//   The lensing convergence kernel W_κ(a) for source bin j involves the
//   cumulative lensing efficiency — the integrated geometric weight of all
//   sources behind scale factor a. For a flat cosmology (f_K = χ) it
//   reads:
//
//     g(a) = ∫_{a_min}^{a} [n_j(z(a')) / a'^2]
//                           × [1 − χ(a)/χ(a')]  da'
//
//   where n_j = nz_source_photoz is the (normalized) source photometric
//   redshift distribution for bin j and the 1/a'^2 Jacobian converts
//   dz → da.  This is the same functional form as
//   g_lens, but integrated against the source distribution rather than the
//   lens distribution.  It appears in W_κ(a, j) = g(a, j) / χ(a), which
//   enters every shear-related angular power spectrum (shear-shear,
//   galaxy-shear, CMB lensing × shear).
//
//   Splitting the geometric factor [1 − χ(a)/χ(a')] gives two cumulative
//   integrals:
//
//     P(a) = ∫_{a_min}^{a} n_j(z') / a'^2              da'
//     Q(a) = ∫_{a_min}^{a} n_j(z') / [χ(a') · a'^2]    da'
//
//   so that g(a) = P(a) − χ(a) · Q(a).
//
// NUMERICAL SCHEME:
//   Same fine/coarse two-grid scheme as g_lens:
//
//     Fine grid:   Na = x·(N_a − 1) + 1 points on [a_min, a_max]
//     Coarse grid: N_a points (every x-th fine point)
//     x = 60·(1 + |high_def_integration|)
//
//   Step 1 — Sample integrands on the fine grid (parallel over bins × points):
//     Pint[j][i] = n_j(z(a_i)) / a_i^2
//     Qint[j][i] = Pint[j][i] / χ(a_i)
//
//   Step 2 — Cumulative trapezoidal integration on the fine grid (serial
//   within each bin). Every x-th fine step, subsample onto the coarse grid:
//     table[j][k] = P − χ(a_k) · Q
//
//   Step 3 — At query time, linearly interpolate table[ni] at a.
//
// Cache invalidation:
//   Recomputes when any of these change:
//     - Ntable.random             (grid parameters: N_a, high_def_integration)
//     - cosmology.random          (χ(a) depends on cosmological parameters)
//     - nuisance.random_photoz_shear  (source photo-z nuisance shifts)
//     - redshift.random_shear         (source n(z) distribution)
//
// Parameters:
//   ainput — scale factor at which to evaluate the lensing efficiency
//   ni     — source tomographic bin index (0 .. shear_nbin − 1)
//
// Returns:
//   g(a, ni), linearly interpolated from the precomputed table.
//   Returns 0 if a ≤ a_min or a > 1 − dac (outside the tabulated range).
// ---------------------------------------------------------------------------
double g_tomo(double ainput, const int ni) {
  static uint64_t cache[MAX_SIZE_ARRAYS]; 
  static double** table = NULL;
  static double** Pint = NULL; // P integrand samples on fine grid
  static double** Qint = NULL; // Q integrand samples on fine grid

  // x = fine trapezoid steps per coarse table cell (Na - 1 = x*(N_a - 1))
  const int x  = 60*(1 + abs(Ntable.high_def_integration));
  const int Na = x * (Ntable.N_a - 1) + 1;
  // start of the SHIFTED source support (zmax_source_photoz)
  const double amin = 1.0/(zmax_source_photoz() + 1.0);
  // amax stops just short of a = 1 (today): chi(1) = 0 and the Q
  // integrand divides by chi(a')
  const double amax = 0.999999;
  
  if (NULL == table || fdiff2(cache[0], Ntable.random))  { 
    if (table != NULL) free(table);
    table = (double**) malloc2d(redshift.shear_nbin, Ntable.N_a);
    if (Pint != NULL) free(Pint);
    Pint = (double**) malloc2d(redshift.shear_nbin, Na);
    if (Qint != NULL) free(Qint);
    Qint = (double**) malloc2d(redshift.shear_nbin, Na);
  }
  if (fdiff2(cache[0], Ntable.random) ||
      fdiff2(cache[1], cosmology.random) ||
      fdiff2(cache[2], nuisance.random_photoz_shear) ||
      fdiff2(cache[3], redshift.random_shear)) 
  {
    (void) nz_source_photoz(0.0, 0); // init static variables
    const double da = (amax - amin) / ((double) Na - 1.0); // fine

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j=0; j<redshift.shear_nbin; j++) {
      for (int i=0; i<Na; i++) {
        const double a = amin + i*da;
        Pint[j][i] = nz_source_photoz(1./a-1., j) / (a * a);
        const double c = chi(amin + i*da);
        if (!(c > 0.0)) {
          log_fatal("division by zero (chi = 0)"); exit(1);
        }
        Qint[j][i] = Pint[j][i] / c;
      }
    }
    #pragma omp parallel for schedule(static)
    for (int j=0; j<redshift.shear_nbin; j++) {
      double P = 0.0;
      double Q = 0.0;
      // 1st point: 0 (P = Q = 0; the source n(z) support starts at amin)
      table[j][0] = 0.0;
      for (int i=1; i<Na; i++) {
        P += 0.5 * da * (Pint[j][i-1] + Pint[j][i]);
        Q += 0.5 * da * (Qint[j][i-1] + Qint[j][i]);
        // every x-th fine step lands on coarse node k; assemble the
        // kernel there from the running sums: g = P - chi * Q
        if (i % x == 0) {
          const int k = i / x;
          table[j][k] = P - chi(amin+i*da) * Q; // that is glens
        }
      }
    }
    cache[0] = Ntable.random;
    cache[1] = cosmology.random;
    cache[2] = nuisance.random_photoz_shear;
    cache[3] = redshift.random_shear;
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni); exit(1);
  } 
  const double dac = (amax - amin) / ((double) Ntable.N_a - 1.0); //coarse
  return (ainput <= amin || ainput > 1.0 - dac) ? 0.0 :
    interpol1d(table[ni], Ntable.N_a, amin, amax, dac, ainput);
}

// ---------------------------------------------------------------------------
// Integral of the squared lensing kernel for source tomographic bin ni.
//
// PHYSICS:
//   Several terms in the angular power spectrum of source galaxy clustering
//   (and magnification–magnification correlations) involve the integral of
//   the *squared* lensing efficiency kernel over the source distribution,
//   here in its flat-cosmology (f_K = χ) form:
//
//     g2(a) = ∫_{a_min}^{a} [n_j(z(a')) / a'^2]
//                            × [1 − χ(a)/χ(a')]^2  da'
//
//   This is distinct from [g(a)]^2: g2 is the integral of the square,
//   not the square of the integral.  Physically, g(a) gives the mean
//   lensing weight (used in galaxy-shear cross-correlations), while g2(a)
//   gives the second moment of the lensing weight along the line of sight
//   (used when computing magnification auto-correlations or source
//   clustering terms where two lensing factors share the same radial
//   integration variable).
//
//   Expanding the squared kernel:
//
//     [1 − χ(a)/χ(a')]^2 = 1 − 2χ(a)/χ(a') + χ(a)^2/χ(a')^2
//
//   the integral splits into three cumulative pieces:
//
//     P(a) = ∫_{a_min}^{a} n_j(z') / a'^2                da'
//     Q(a) = ∫_{a_min}^{a} n_j(z') / [χ(a') · a'^2]      da'
//     R(a) = ∫_{a_min}^{a} n_j(z') / [χ(a')^2 · a'^2]    da'
//
//   so that g2(a) = P(a) − 2χ(a)·Q(a) + χ(a)^2·R(a), with each integral
//   computable as a single running sum rather than a nested quadrature.
//
// NUMERICAL SCHEME:
//   Same fine/coarse two-grid scheme as g_lens:
//
//     Fine grid:   Na = x·(N_a − 1) + 1 points on [a_min, a_max]
//     Coarse grid: N_a points (every x-th fine point)
//
//   Step 1 — Sample the three integrands on the fine grid:
//     Pint[j][i] = n_j(z(a_i)) / a_i^2
//     Qint[j][i] = Pint[j][i] / χ(a_i)
//     Rint[j][i] = Qint[j][i] / χ(a_i)
//
//   Step 2 — Cumulative trapezoidal integration. Every x-th fine step, 
//   subsample onto the coarse grid: table[j][k] = P − 2χ(a_k)·Q + χ(a_k)^2·R.
//
//   Step 3 — At query time, linearly interpolate table[ni] at a.
//
// Cache invalidation:
//   Recomputes when any of these change:
//     - Ntable.random             (grid parameters)
//     - cosmology.random          (χ(a) depends on cosmology)
//     - nuisance.random_photoz_shear  (source photo-z shifts)
//     - redshift.random_shear         (source n(z) distribution)
//
// Parameters:
//   a  — scale factor at which to evaluate
//   ni — source tomographic bin index (0 .. shear_nbin − 1)
//
// Returns:
//   g2(a, ni), linearly interpolated from the precomputed table.
//   Returns 0 if a is outside the tabulated range.
// ---------------------------------------------------------------------------
double g2_tomo(double a, int ni)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double** Pint  = NULL;  // P integrand on fine grid
  static double** Qint  = NULL;  // Q integrand on fine grid
  static double** Rint  = NULL;  // R integrand on fine grid

  // x = fine trapezoid steps per coarse table cell (Na - 1 = x*(N_a - 1))
  const int x = 60*(1 + abs(Ntable.high_def_integration));
  const int Na = x * (Ntable.N_a - 1) + 1;
  // start of the SHIFTED source support (zmax_source_photoz)
  const double amin = 1.0 / (zmax_source_photoz() + 1.0);
  // amax stops just short of a = 1 (today): chi(1) = 0 and the Q and R
  // integrands divide by chi(a')
  const double amax = 0.999999;

  if (NULL == table || fdiff2(cache[0], Ntable.random)) {
    if (table != NULL) free(table);
    if (Pint  != NULL) free(Pint);
    if (Qint  != NULL) free(Qint);
    if (Rint  != NULL) free(Rint);
    table = (double**) malloc2d(redshift.shear_nbin, Ntable.N_a);
    Pint  = (double**) malloc2d(redshift.shear_nbin, Na);
    Qint  = (double**) malloc2d(redshift.shear_nbin, Na);
    Rint  = (double**) malloc2d(redshift.shear_nbin, Na);
  }

  if (fdiff2(cache[0], Ntable.random) ||
      fdiff2(cache[1], cosmology.random) ||
      fdiff2(cache[2], nuisance.random_photoz_shear) ||
      fdiff2(cache[3], redshift.random_shear))
  {
    const double da = (amax - amin) / ((double) Na - 1.0); // fine

    (void) nz_source_photoz(0.0, 0); // warm cache before threading

    double chia[Na];
    for (int i = 0; i < Na; i++) {
      const double c = chi(amin + i * da);
      if (!(c > 0.0)) {
        log_fatal("division by zero (chi = 0)"); exit(1);
      }
      chia[i] = c;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < redshift.shear_nbin; j++) {
      for (int i = 0; i < Na; i++) {
        const double ap = amin + i * da;
        const double z  = 1.0/ap - 1.0;
        Pint[j][i] = nz_source_photoz(z, j) / (ap * ap);
        Qint[j][i] = Pint[j][i] / chia[i];
        Rint[j][i] = Qint[j][i] / chia[i];
      }
    }
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < redshift.shear_nbin; j++) {
      double P = 0.0, Q = 0.0, R = 0.0;
      table[j][0] = 0.0;
      for (int i = 1; i < Na; i++) {
        P += 0.5 * da * (Pint[j][i-1] + Pint[j][i]);
        Q += 0.5 * da * (Qint[j][i-1] + Qint[j][i]);
        R += 0.5 * da * (Rint[j][i-1] + Rint[j][i]);
        if (i % x == 0) {
          const int k = i / x;
          const double c = chia[i];
          table[j][k] = P - 2.0*c*Q + c*c*R;
        }
      }
    }

    cache[0] = Ntable.random;
    cache[1] = cosmology.random;
    cache[2] = nuisance.random_photoz_shear;
    cache[3] = redshift.random_shear;
  }

  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni); exit(1);
  }
  const double dac = (amax - amin) / ((double) Ntable.N_a - 1.0); // coarse
  return (a <= amin || a > 1.0 - dac) ? 0.0 :
    interpol1d(table[ni], Ntable.N_a, amin, amax, dac, a);
}

// ---------------------------------------------------------------------------
// Bin-averaged lensing efficiency g(a) for lens tomographic bin ni.
//
// PHYSICS:
//   In the flat-sky Limber approximation for galaxy-galaxy lensing (and
//   magnification), the angular power spectrum C_l^{gκ} involves a radial
//   projection kernel that weights lens galaxies by how efficiently they
//   lens background sources. For a flat cosmology (f_K = χ), this kernel
//   factors as:
//
//     g(a) = ∫_{a_min}^{a} [n_j(z(a')) / a'^2]
//                           × [1 - χ(a)/χ(a')] da'
//
//   where n_j = nz_lens_photoz is the (normalized) photometric redshift
//   distribution for lens bin j, and the 1/a'^2 Jacobian converts dz → da.
//   The geometric factor [1 − χ(a)/χ(a')] = [χ(a') − χ(a)] / χ(a')
//   is the standard lensing efficiency: it vanishes when the lens sits
//   at the same distance as the source (χ = χ') and grows as the lens
//   moves closer to the observer relative to the source.
//
//   Rather than evaluating the full expression directly, the code splits
//   the kernel into two simpler cumulative integrals:
//
//     P(a) = ∫_{a_min}^{a} n_j(z') / a'^2              da'
//     Q(a) = ∫_{a_min}^{a} n_j(z') / [χ(a') · a'^2]    da'
//
//   so that g(a) = P(a) − χ(a) · Q(a).  This avoids recomputing the full
//   double integral for each query point a.
//
// NUMERICAL SCHEME:
//   Two nested grids are used:
//
//     Fine grid:   Na = x·(N_a − 1) + 1 points on [a_min, a_max]
//                  spacing da = (a_max − a_min) / (Na − 1)
//                  x = 60·(1 + |high_def_integration|), so x ∈ {60, 120, ...}
//
//     Coarse grid: N_a points on [a_min, a_max]
//                  spacing dac = (a_max − a_min) / (N_a − 1)
//                  every x-th fine point maps to a coarse grid point
//
//   Step 1 — Sample integrands on the fine grid (parallelized over bins × points):
//     Pint[j][i] = n_j(z(a_i)) / a_i^2
//     Qint[j][i] = Pint[j][i] / χ(a_i)
//
//   Step 2 — Cumulative trapezoidal integration on the fine grid:
//     P and Q start at zero. Every x-th fine step, the running sums are
//     subsampled onto the coarse grid: table[j][k] = P − χ(a_k) · Q.
//     This gives the accuracy of fine-grid trapezoidal integration with
//     the memory footprint and lookup speed of the coarse grid.
//
//   Step 3 — At query time, linearly interpolate table[ni] at the requested a.
//
// Cache invalidation:
//   Recomputes when any of these change:
//     - Ntable.random          (grid parameters: N_a, high_def_integration)
//     - cosmology.random       (χ(a) depends on cosmological parameters)
//     - nuisance.random_photoz_clustering  (photo-z nuisance shifts)
//     - redshift.random_clustering         (lens n(z) distribution)
//
// Parameters:
//   a  — scale factor at which to evaluate the lensing efficiency
//   ni — lens tomographic bin index (0 .. clustering_nbin − 1)
//
// Returns:
//   g(a, ni), linearly interpolated from the precomputed table.
//   Returns 0 if a < a_min or a > 1 − dac (outside the tabulated range).
// ---------------------------------------------------------------------------
double g_lens(double a, int ni)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double** Pint  = NULL; // P integrand samples on fine grid
  static double** Qint  = NULL; // Q integrand samples on fine grid

  // x = fine trapezoid steps per coarse table cell (Na - 1 = x*(N_a - 1))
  const int x = 60*(1 + abs(Ntable.high_def_integration));
  const int Na = x * (Ntable.N_a - 1) + 1;
  // start of the shifted and stretched lens support (zmax_lens_photoz)
  const double amin = 1.0 / (zmax_lens_photoz() + 1.0);
  // amax stops just short of a = 1 (today): chi(1) = 0 and the Q
  // integrand divides by chi(a')
  const double amax = 0.999999;

  if (table == NULL || fdiff2(cache[0], Ntable.random)) {
    if (table != NULL) free(table);
    if (Pint  != NULL) free(Pint);
    if (Qint  != NULL) free(Qint);
    table = (double**) malloc2d(redshift.clustering_nbin, Ntable.N_a);
    Pint  = (double**) malloc2d(redshift.clustering_nbin, Na);
    Qint  = (double**) malloc2d(redshift.clustering_nbin, Na);
  }

  if (fdiff2(cache[0], Ntable.random) ||
      fdiff2(cache[1], cosmology.random) ||
      fdiff2(cache[2], nuisance.random_photoz_clustering) ||
      fdiff2(cache[3], redshift.random_clustering))
  {
    (void) nz_lens_photoz(0.0, 0);
    const double da = (amax - amin) / ((double) Na - 1.0);

    #pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < redshift.clustering_nbin; j++) {
      for (int i = 0; i < Na; i++) {
        const double ap = amin + i * da;
        const double z  = 1.0/ap - 1.0;
        Pint[j][i] = nz_lens_photoz(z, j) / (ap * ap);
        Qint[j][i] = Pint[j][i] / chi(amin + i * da);
      }
    }
    #pragma omp parallel for schedule(static)
    for (int j = 0; j < redshift.clustering_nbin; j++) {
      double P = 0.0;
      double Q = 0.0; 
      table[j][0] = P - chi(amin) * Q; // 1st point: 0 (P = Q = 0; the lens
                                       // n(z) support starts at amin)
      for (int i = 1; i < Na; i++) {
        P += 0.5 * da * (Pint[j][i-1] + Pint[j][i]);
        Q += 0.5 * da * (Qint[j][i-1] + Qint[j][i]);
        if (i % x == 0) {
          const int k = i / x;
          table[j][k] = P - chi(amin + i * da) * Q;
        }
      }
    }
    cache[0] = Ntable.random;
    cache[1] = cosmology.random;
    cache[2] = nuisance.random_photoz_clustering;
    cache[3] = redshift.random_clustering;
  }
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni); exit(1);
  }
  const double dac = (amax - amin) / ((double) Ntable.N_a - 1.0); // coarse
  return (a < amin || a > 1.0 - dac) ? 0.0 :
    interpol1d(table[ni], Ntable.N_a, amin, amax, dac, a);
}

// ---------------------------------------------------------------------------
// Lensing efficiency of the CMB source plane.
//
//   g_cmb(a) = f_K(chi_cmb - chi(a)) / f_K(chi_cmb)
//
// with chi_cmb = chi(a = 1/1091), the comoving distance to z = 1090, and
// f_K the curvature-dependent comoving angular diameter distance. The
// single-source analog of g_tomo; it enters the CMB lensing convergence
// kernel W_k.
//
// Cache invalidation:
// chi_cmb and f_K(chi_cmb) are recomputed when
// cosmology.random changes. The first call after a cosmology change must
// happen outside OpenMP regions.
//
// Parameters:
//   a - scale factor
//
// Returns:
//   g_cmb(a) (dimensionless).
// ---------------------------------------------------------------------------
double g_cmb(double a) 
{
  static uint64_t cache_cosmo_params;
  static double chi_cmb = 0.;
  static double fchi_cmb = 0.;
  
  if (fdiff2(cache_cosmo_params, cosmology.random)) 
  {
    chi_cmb = chi(1./1091.);
    fchi_cmb = f_K(chi_cmb);
    cache_cosmo_params = cosmology.random;
  }
  
  return f_K(chi_cmb - chi(a))/fchi_cmb;
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------