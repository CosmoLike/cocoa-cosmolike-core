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

double amin_source(int ni) 
{
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  }
  return 1. / (redshift.shear_zdist_zmax_all + 1.);
}

double amax_source(int i __attribute__((unused))) 
{
  return 1. / (1. + fmax(redshift.shear_zdist_zmin_all, 0.001));
}

double amax_source_IA(int ni) 
{
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  }
  return 1. / (1. + fmax(redshift.shear_zdist_zmin_all, 0.001));
}

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

int test_kmax(double l, int ni) // return 1 if true, 0 otherwise
{ // test whether the (l, ni) bin is in the linear clustering regime
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

int test_zoverlap(int ni, int nj) 
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input (ni, nj) = (%d, %d)", ni, nj);
    exit(1);
  }
  if (tomo.ggl_exclude != NULL) {
    static int N[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS] = {{-42}};
    if (N[0][0] < -1) {
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

int ZL(int ni) 
{
  static int N[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS] = {-42};
  if (N[0] < -1) {
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

int ZS(int nj) 
{
  static int N[MAX_SIZE_ARRAYS*MAX_SIZE_ARRAYS] = {-42};
  if (N[0] < -1) {
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

int N_ggl(int ni, int nj) 
{ // ni = redshift bin of the lens, nj = redshift bin of the source
  static int N[MAX_SIZE_ARRAYS][MAX_SIZE_ARRAYS] = {{-42}};
  if (N[0][0] < 0) {
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

int Z1(int ni) 
{ // find z1 of tomography combination (z1, z2) constituting shear tomo bin Nbin
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

int Z2(int nj) 
{ // find z2 of tomography combination (z1,z2) constituting shear tomo bin Nbin
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

int N_shear(int ni, int nj) 
{ // find shear tomography bin number N_shear of tomography combination (z1, z2)
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

int ZCL1(int ni) 
{ // find ZCL1 of tomography combination (zcl1, zcl2) constituting tomo bin Nbin
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

int ZCL2(int nj) 
{ // find ZCL2 of tomography combination (zcl1, zcl2) constituting tomo bin Nbin
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

double zdistr_histo_n(double z, const int ni)
{ // Raw (unnormalized) histogram lookup for the source galaxy redshift
  // distribution in tomography bin ni. Returns the tabulated n(z) value
  // at redshift z by finding the histogram bin that contains z and
  // returning its stored density. Returns 0 outside the tabulated range.
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
//   Stage 1 (cache rebuild, runs once when redshift.random_shear changes):
//     - Normalize the raw histograms and fit cubic splines (via GSL) on
//       the original (possibly non-uniform) bin-center grid.
//     - Resample each spline onto a uniform fine grid (20× the original
//       resolution) and precompute cubic spline coefficients via a
//       tridiagonal solve (spline_coeffs_uniform).
//     - Controlled by DONT_NZ_FAST_SUMBSAMPLE: when defined, the fine
//       grid is skipped and the hot path falls back to GSL evaluation.
//
//   Stage 2 (hot path, called millions of times per likelihood):
//     - Direct-index lookup on the uniform fine grid: one multiply to
//       compute the grid index (no binary search, no GSL function pointer
//       dispatch), then Horner-form cubic polynomial evaluation.
//
// CACHE INVALIDATION:
//   Recomputes when redshift.random_shear changes, which tracks:
//     - source n(z) histogram values
//     - number of bins, redshift range, bin edges
//
// PARAMETERS:
//   zz — redshift at which to evaluate (before photo-z shift)
//   nj — source tomographic bin index (0 .. shear_nbin − 1)
//
// RETURNS:
//   n(z − Δz_nj, nj). Returns 0 outside the tabulated range.
// ---------------------------------------------------------------------------
double nz_source_photoz(double zz, const int nj)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static gsl_interp* photoz_splines[MAX_SIZE_ARRAYS+1];
#ifndef DONT_NZ_FAST_SUMBSAMPLE
  static double*** fine = NULL;
  static double zmin_fine;
  static double inv_dz_fine;
  static int nzbins_fine;
#endif

  if (table == NULL || fdiff2(cache[0], redshift.random_shear)) {
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
    for (int k = 0; k < nzbins; k++) {
      table[ntomo+1][k] = zmin + (k + 0.5) * dz_histo;
    }

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
    nzbins_fine = 20 * (1 + abs(Ntable.high_def_integration)) * nzbins + 1;
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
  if (zz <= zmin_fine || zz >= zmin_fine + (nzbins_fine - 1) / inv_dz_fine) {
    return 0.0;
  }
  // -----------------------------------------------------------------------
  // Hot-path cubic spline evaluation on the uniform fine grid.
  //
  // Direct-index lookup: the uniform spacing allows computing the grid
  // index from a single multiply (no binary search, no accelerator).
  //   r     = fractional grid index (floating point)
  //   index = integer part → left bracket
  //   delx  = (r - index) * dx → distance from left grid point
  //
  // The cubic polynomial on interval [z_index, z_{index+1}] is:
  //   S(z) = y_i + delx * (b + delx * (c_i + delx * d))
  // with coefficients b, c_i, d derived from the precomputed spline
  // coefficients c_i, c_{i+1} and the local slope dy/dx, following
  // the same formulation as GSL's cspline_eval (Horner form).
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

double int_for_zmean_source(double z, void* params) 
{ // Integrand for computing the mean source redshift: z * n_j(z).
  // params[0] = bin index j (cast to double). Used with GSL-quad in zmean_source.
  double* ar = (double*) params;
  const int ni = (int) ar[0];
  
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni); exit(1);
  } 
  return z * nz_source_photoz(z, ni);
}

double zmean_source(int ni) 
{ // Mean true redshift of source galaxies in tomography bin ni,
  // computed as ∫ z * n_i(z) dz over the bin's redshift range,
  // where n_i = nz_source_photoz (already normalized to unit integral).
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
    cache[0] = redshift.random_shear;
    cache[1] = Ntable.random;
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
double pf_histo_n(double z, const int ni) 
{ // Raw (unnormalized) histogram lookup for the lens galaxy redshift
  // distribution in tomography bin ni. Returns the tabulated n(z) value
  // at redshift z by finding the histogram bin that contains z and
  // returning its stored density. Returns 0 outside the tabulated range.

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
//   Stage 1 (cache rebuild, runs once per parameter change):
//     - Fit cubic splines (via GSL) to the normalized histograms on
//       the original (possibly non-uniform) bin-center grid.
//     - Resample each spline onto a uniform fine grid (20× the
//       original resolution) and precompute cubic spline coefficients
//       on that grid via a tridiagonal solve.
//
//   Stage 2 (hot path, called millions of times per likelihood):
//     - Direct-index lookup on the uniform fine grid: one multiply
//       to compute the grid index (no binary search, no GSL function
//       pointer dispatch), then Horner-form cubic polynomial evaluation.
//
//   This replaces the original GSL gsl_interp_eval_e call, which
//   accounted for ~6% of wall time due to binary search overhead and
//   function pointer indirection in the inner Limber loops.
//
// CACHE INVALIDATION:
//   Recomputes when redshift.random_clustering changes, which tracks:
//     - lens n(z) histogram values
//     - number of bins, redshift range, bin edges
//
// PARAMETERS:
//   zz — redshift at which to evaluate (before photo-z transformation)
//   nj — lens tomographic bin index (0 .. clustering_nbin − 1)
//
// RETURNS:
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

  if (NULL == table || fdiff2(cache[0], redshift.random_clustering))
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
    for (int k = 0; k < nzbins; k++) {
      table[ntomo+1][k] = zmin + (k + 0.5) * dz_histo;
    }

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
    // the original (possibly non-uniform) histogram grid is evaluated
    // once here; the hot path below uses direct-index lookup with no
    // binary search.
    // -----------------------------------------------------------------
    nzbins_fine = Ntable.nz_fine_sampling_factor*nzbins + 1;
    
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
  // Direct-index lookup: the uniform spacing allows computing the grid
  // index from a single multiply (no binary search, no accelerator).
  //   r     = fractional grid index (floating point)
  //   index = integer part → left bracket
  //   delx  = (r - index) * dx → distance from left grid point
  //
  // The cubic polynomial on interval [z_index, z_{index+1}] is:
  //   S(z) = y_i + delx * (b + delx * (c_i + delx * d))
  // with coefficients b, c_i, d derived from the precomputed spline
  // coefficients c_i, c_{i+1} and the local slope dy/dx, following
  // the same formulation as GSL's cspline_eval (Horner form).
  //
  // The final division by nuisance.photoz[1][1][nj] applies the
  // photo-z stretch factor to the interpolated n(z) value.
  // -----------------------------------------------------------------------
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


double int_for_zmean(double z, void* params) 
{ 
  // Integrand for computing the mean lens redshift: z * n_j(z).
  // params[0] = bin index j (cast to double). Used with GSL quadrature in zmean.
  double* ar = (double*) params;
  const int ni = (int) ar[0];
  
  if (ni < 0 || ni > redshift.clustering_nbin - 1)
  {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  } 
  return z * nz_lens_photoz(z, ni);
}

double norm_for_zmean(double z, void* params) 
{ 
  // Integrand for the norm of the mean lens redshift: n_j(z).
  // params[0] = bin index j. Used with GSL-quad in zmean to compute ∫ n_j(z) dz.
  double* ar = (double*) params;
  const int ni = (int) ar[0];
  
  if (ni < 0 || ni > redshift.clustering_nbin - 1)
  {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  } 
  return nz_lens_photoz(z, ni);
}

double zmean(const int ni)
{ // Mean true redshift of lens galaxies in tomography bin ni,
  // computed as ∫ z * n_i(z) dz / ∫ n_i(z) dz, where n_i = pf_photoz.
  // Unlike zmean_source, this explicitly divides by the norm because
  // pf_photoz includes a stretch factor that breaks unit normalization.
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* table = NULL;
  static gsl_integration_glfixed_table* w = NULL;

  if (table == NULL || 
      fdiff2(cache[0], Ntable.random) ||
      fdiff2(cache[1], redshift.random_clustering))
  {
    if (table != NULL) free(table);
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
//   sources behind scale factor a:
//
//     g(a) = ∫_{a_min}^{a} [n_j(z(a')) / a'^2]
//                           × [1 − χ(a)/χ(a')]  da'
//
//   where n_j = nz_source_photoz is the (normalized) source photometric
//   redshift distribution for bin j.  This is the same functional form as
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
//     x = 25·(1 + |high_def_integration|)
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
// NOTE:
//   The Step 2 loop is not parallelized over bins (unlike g_lens). Adding
//   #pragma omp parallel for schedule(static) over j would be safe here
//   since each bin's running sum is independent.
//
// CACHE INVALIDATION:
//   Recomputes when any of these change:
//     - Ntable.random             (grid parameters: N_a, high_def_integration)
//     - cosmology.random          (χ(a) depends on cosmological parameters)
//     - nuisance.random_photoz_shear  (source photo-z nuisance shifts)
//     - redshift.random_shear         (source n(z) distribution)
//
// PARAMETERS:
//   ainput — scale factor at which to evaluate the lensing efficiency
//   ni     — source tomographic bin index (0 .. shear_nbin − 1)
//
// RETURNS:
//   g(a, ni), linearly interpolated from the precomputed table.
//   Returns 0 if a ≤ a_min or a > 1 − dac (outside the tabulated range).
// ---------------------------------------------------------------------------
double g_tomo(double ainput, const int ni) {
  // Bin-averaged lensing efficiency for *source* tomography bin ni.
  // Assumes flat cosmology: f_K(x) = x, so the lensing kernel factors as
  //   g(a_i) = P(a_i) - chi(a_i) * Q(a_i)
  // where
  //   P(a) = ∫_{amin}^{a} n_j(z') / a'^2           da'
  //   Q(a) = ∫_{amin}^{a} n_j(z') / (chi(a') a'^2) da'
  // and n_j = nz_source_photoz (the source photo-z distribution for bin j).
  static uint64_t cache[MAX_SIZE_ARRAYS]; 
  static double** table = NULL;
  static double** Pint = NULL; // P integrand samples on fine grid
  static double** Qint = NULL; // Q integrand samples on fine grid

  const int x  = 25*(1 + abs(Ntable.high_def_integration));
  const int Na = x * (Ntable.N_a - 1) + 1;
  const double amin = 1.0/(redshift.shear_zdist_zmax_all + 1.0);
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
      table[j][0] = 0.0;
      for (int i=1; i<Na; i++) {
        P += 0.5 * da * (Pint[j][i-1] + Pint[j][i]);
        Q += 0.5 * da * (Qint[j][i-1] + Qint[j][i]);
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
//   the *squared* lensing efficiency kernel over the source distribution:
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
// CACHE INVALIDATION:
//   Recomputes when any of these change:
//     - Ntable.random             (grid parameters)
//     - cosmology.random          (χ(a) depends on cosmology)
//     - nuisance.random_photoz_shear  (source photo-z shifts)
//     - redshift.random_shear         (source n(z) distribution)
//
// PARAMETERS:
//   a  — scale factor at which to evaluate
//   ni — source tomographic bin index (0 .. shear_nbin − 1)
//
// RETURNS:
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

  const int x = 25*(1 + abs(Ntable.high_def_integration));
  const int Na = x * (Ntable.N_a - 1) + 1;
  const double amin = 1.0 / (redshift.shear_zdist_zmax_all + 1.0);
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
//                  x = 25·(1 + |high_def_integration|), so x ∈ {25, 50, ...}
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
// CACHE INVALIDATION:
//   Recomputes when any of these change:
//     - Ntable.random          (grid parameters: N_a, high_def_integration)
//     - cosmology.random       (χ(a) depends on cosmological parameters)
//     - nuisance.random_photoz_clustering  (photo-z nuisance shifts)
//     - redshift.random_clustering         (lens n(z) distribution)
//
// PARAMETERS:
//   a  — scale factor at which to evaluate the lensing efficiency
//   ni — lens tomographic bin index (0 .. clustering_nbin − 1)
//
// RETURNS:
//   g(a, ni), linearly interpolated from the precomputed table.
//   Returns 0 if a < a_min or a > 1 − dac (outside the tabulated range).
// ---------------------------------------------------------------------------
double g_lens(double a, int ni)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double** Pint  = NULL; // P integrand samples on fine grid
  static double** Qint  = NULL; // Q integrand samples on fine grid

  const int x = 25*(1 + abs(Ntable.high_def_integration));
  const int Na = x * (Ntable.N_a - 1) + 1;
  const double amin = 1.0 / (redshift.clustering_zdist_zmax_all + 1.0);
  const double amax = 0.999999;
  const double amin_shear = 1.0 / (redshift.shear_zdist_zmax_all + 1.0);

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
      table[j][0] = P - chi(amin) * Q; // 1st point: integral_amin_shear^amin
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