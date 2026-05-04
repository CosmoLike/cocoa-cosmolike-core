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

double nz_source_photoz(double zz, const int nj) 
{ 
  // Normalized source galaxy redshift distribution with photo-z bias,
  // evaluated at redshift zz for tomography bin nj.
  //
  // On first call, builds a table of normalized n(z) from the raw histogram
  // values in zdistr_histo_n. The normalization works as follows:
  //
  //   1. Compute per-bin normalization NORM[i] = ∫ n_raw_i(z) dz
  //      by summing the raw histogram over all redshift bins.
  //
  //   2. Compute total normalization: norm = Σ_i NORM[i].
  //
  //   3. Build two tables:
  //      - table[i+1][k] = n_raw_i(z_k) / NORM[i]
  //        The per-bin distribution, normalized so each bin integrates
  //        to unity independently.
  //      - table[0][k] = Σ_i table[i+1][k] * NORM[i] / norm
  //        The combined (all-bin) distribution, where each bin's
  //        contribution is weighted by its fraction of the total
  //        galaxy count.
  //
  //   4. Fit cubic splines through the bin-center values for smooth
  //      interpolation (the raw data is histogram-binned, but the
  //      underlying n(z) is smooth).
  //
  // At evaluation time, applies the photo-z bias shift:
  //    zz_corrected = zz - nuisance.photoz[0][0][nj]
  // then eval the spline for bin nj+1. Returns 0 outside the tabulated range.
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static gsl_interp* photoz_splines[MAX_SIZE_ARRAYS+1];
  
  if (table == NULL || fdiff2(cache[0], redshift.random_shear)) { 
    if (table == NULL) {
      for (int i=0; i<MAX_SIZE_ARRAYS+1; i++) {
        photoz_splines[i] = NULL;
      }
    }

    const int ntomo  = redshift.shear_nbin;
    const int nzbins = redshift.shear_nzbins;

    if (table != NULL) free(table);
    table = (double**) malloc2d(ntomo + 2, nzbins);
    
    const double zmin = redshift.shear_zdist_zmin_all;
    const double zmax = redshift.shear_zdist_zmax_all;
    const double dz_histo = (zmax - zmin) / ((double) nzbins);  
    for (int k=0; k<nzbins; k++) { // redshift stored at zv = table[ntomo+1]
      table[ntomo+1][k] = zmin + (k + 0.5) * dz_histo;
    }
    
    double NORM[MAX_SIZE_ARRAYS];    
    double norm = 0; 
    #pragma omp parallel for reduction( + : norm )
    for (int i=0; i<ntomo; i++) {
      NORM[i] = 0.0;
      for (int k=0; k<nzbins; k++) {    
        const double z = table[ntomo+1][k];  
        NORM[i] += zdistr_histo_n(z, i) * dz_histo;
      }
      norm += NORM[i];
    }  
    #pragma omp parallel for
    for (int k=0; k<nzbins; k++) { 
      table[0][k] = 0; // store normalization in table[0][:]
      for (int i=0; i<ntomo; i++) {
        const double z = table[ntomo+1][k];
        table[i + 1][k] = zdistr_histo_n(z, i)/NORM[i];
        table[0][k] += table[i+1][k] * NORM[i] / norm;
      }
    }
    for (int i=0; i<ntomo+1; i++) {
      if (photoz_splines[i] != NULL) gsl_interp_free(photoz_splines[i]);
      photoz_splines[i] = malloc_gsl_interp(nzbins);
    }
    #pragma omp parallel for
    for (int i=0; i<ntomo+1; i++) {
      int status = gsl_interp_init(photoz_splines[i], 
                                   table[ntomo+1], // z_v = table[ntomo+1]
                                   table[i], 
                                   nzbins);
      if (status) {
        log_fatal(gsl_strerror(status));
        exit(1);
      }
    }
    cache[0] = redshift.random_shear;
  }
  
  const int ntomo  = redshift.shear_nbin;
  const int nzbins = redshift.shear_nzbins;

  if (nj < 0 || nj > ntomo - 1) {
    log_fatal("nj = %d bin outside range (max = %d)", nj, ntomo); exit(1);
  }

  zz = zz - nuisance.photoz[0][0][nj];
  
  double res; 
  if (zz < table[ntomo+1][0] || zz > table[ntomo+1][nzbins-1]) { // z_v = table[ntomo+1]
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
      log_fatal(gsl_strerror(status)); exit(1);
    }
  }
  return res;
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
  static gsl_integration_glfixed_table* w;

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
    #pragma omp parallel for
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

double nz_lens_photoz(double zz, int nj) 
{ 
  // Normalized lens galaxy redshift distribution with photo-z bias  and stretch,
  // evaluated at redshift zz for tomography bin nj.
  //
  // On first call, builds a table of normalized n(z) from the raw histogram 
  // values in pf_histo_n. The normalization follows the same scheme as nz_source_photoz:
  //
  //   1. Per-bin normalization: NORM[i] = ∫ n_raw_i(z) dz
  //
  //   2. Total normalization: norm = Σ_i NORM[i]
  //
  //   3. table[i+1][k] = n_raw_i(z_k) / NORM[i]       (per-bin, unit integral)
  //      table[0][k]   = Σ_i table[i+1][k] * NORM[i] / norm  (combined)
  //
  //   4. Cubic splines through bin-center values for smooth evaluation.
  //
  // At evaluation time, applies the photo-z bias and stretch:
  //   zz_corrected = (zz - bias - zmean) / stretch + zmean
  // where
  //   bias    = nuisance.photoz[1][0][nj]
  //   stretch = nuisance.photoz[1][1][nj]
  //   zmean   = redshift.clustering_zdist_zmean[nj]
  //
  // The result is divided by the stretch factor to conserve the integral under
  // the change of variable. Returns 0 outside the tabulated redshift range.
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static gsl_interp* photoz_splines[MAX_SIZE_ARRAYS+1];

  if (NULL == table || fdiff2(cache[0], redshift.random_clustering)) 
  {  
    if (table == NULL) {
      for (int i=0; i<MAX_SIZE_ARRAYS+1; i++) 
        photoz_splines[i] = NULL;
    }

    const int ntomo  = redshift.clustering_nbin;      // alias
    const int nzbins = redshift.clustering_nzbins;    // alias
    
    if (table != NULL) free(table);
    table = (double**) malloc2d(ntomo + 2, nzbins);

    const double zmin = redshift.clustering_zdist_zmin_all;
    const double zmax = redshift.clustering_zdist_zmax_all;
    const double dz_histo = (zmax - zmin) / ((double) nzbins);  
    for (int k=0; k<nzbins; k++) 
    { // redshift stored at zv = table[ntomo+1]
      table[ntomo+1][k] = zmin + (k + 0.5) * dz_histo;
    }
        
    double NORM[MAX_SIZE_ARRAYS];
    double norm = 0;
    #pragma omp parallel for reduction( + : norm )
    for (int i=0; i<ntomo; i++) {
      NORM[i] = 0.0;
      for (int k=0; k<nzbins; k++) 
      {    
        const double z = table[ntomo+1][k];  
        NORM[i] += pf_histo_n(z, i) * dz_histo;
      }
      norm += NORM[i];
    }

    #pragma omp parallel for
    for (int k=0; k<nzbins; k++) { 
      table[0][k] = 0; // store normalization in table[0][:]
      for (int i=0; i<ntomo; i++) 
      {
        const double z = table[ntomo+1][k];
        table[i + 1][k] = pf_histo_n(z, i)/NORM[i];
        table[0][k] += table[i+1][k] * NORM[i] / norm;
      }
    }

    for (int i=0; i<ntomo+1; i++)  {
      if (photoz_splines[i] != NULL) gsl_interp_free(photoz_splines[i]);
      photoz_splines[i] = malloc_gsl_interp(nzbins);
    }

    #pragma omp parallel for
    for (int i=0; i<ntomo+1; i++) {
      int status = gsl_interp_init(photoz_splines[i], 
                                   table[ntomo+1], // z_v = table[ntomo+1]
                                   table[i], 
                                   nzbins);
      if (status) {
        log_fatal(gsl_strerror(status));
        exit(1);
      }
    }

    cache[0] = redshift.random_clustering;
  }
  
  const int ntomo  = redshift.clustering_nbin;
  const int nzbins = redshift.clustering_nzbins;

  if (nj < 0 || nj > ntomo - 1) {
    log_fatal("nj = %d bin outside range (max = %d)", nj, ntomo); exit(1);
  }
  
  zz  = (zz - nuisance.photoz[1][0][nj]
            - redshift.clustering_zdist_zmean[nj])/nuisance.photoz[1][1][nj] 
        + redshift.clustering_zdist_zmean[nj];
  
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
  static gsl_integration_glfixed_table* w;

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
    #pragma omp parallel for
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

// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Bin-averaged lens efficiencies
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------

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

    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int j=0; j<redshift.shear_nbin; j++) {
      for (int i=0; i<Na; i++) {
        const double a = amin + i*da;
        Pint[j][i] = nz_source_photoz(1./a-1., j) / (a * a);
        Qint[j][i] = Pint[j][i] / chi(amin+i*da);
      }
    }
    #pragma omp parallel for schedule(static,1)
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

double g2_tomo(double a, int ni)
{ // Squared lensing efficiency for source tomography bin ni
  // (used in source clustering). Assumes flat cosmology, so
  //   g2(a_i) = P(a_i) - 2 chi(a_i) Q(a_i) + chi(a_i)^2 R(a_i)
  // where
  //   P(a) = ∫_{amin}^{a} n_j(z') / a'^2              da'
  //   Q(a) = ∫_{amin}^{a} n_j(z') / (chi(a') a'^2)    da'
  //   R(a) = ∫_{amin}^{a} n_j(z') / (chi(a')^2 a'^2)  da'
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
    for (int i = 0; i < Na; i++) chia[i] = chi(amin + i * da);

    #pragma omp parallel for collapse(2)
    for (int j = 0; j < redshift.shear_nbin; j++) {
      for (int i = 0; i < Na; i++) {
        const double ap = amin + i * da;
        const double z  = 1.0/ap - 1.0;
        Pint[j][i] = nz_source_photoz(z, j) / (ap * ap);
        Qint[j][i] = Pint[j][i] / chia[i];
        Rint[j][i] = Qint[j][i] / chia[i];
      }
    }

    #pragma omp parallel for schedule(static,1)
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

double g_lens(double a, int ni)
{ // Bin-averaged lens efficiency for lens tomography bin ni.
  // Assumes flat cosmology: f_K(x) = x, so the lensing kernel factors as
  //   g(a_i) = P(a_i) - chi(a_i) * Q(a_i)
  // where
  //   P(a) = ∫_{amin_shear}^{a} n_j(z') / a'^2           da'
  //   Q(a) = ∫_{amin_shear}^{a} n_j(z') / (chi(a') a'^2) da'
  // and n_j is nz_lens_photoz (the lens photo-z distribution for bin j).
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

    double P0[redshift.clustering_nbin];
    double Q0[redshift.clustering_nbin];
    const double da = (amax - amin) / ((double) Na - 1.0); // fine grid spacing
    {
      const int Na0 = (amin_shear<amin) ? (int) ceil((amin-amin_shear)/da)+1 : 1;
      const double da0 = (Na0>1) ? (amin-amin_shear)/((double) Na0-1.) : 0.0;
      for (int j = 0; j<redshift.clustering_nbin; j++) {
        double P = 0.0; 
        double Q = 0.0;
        for (int i=1; i<Na0; i++) {
          const double apl = amin_shear + (i-1) * da0;
          const double aph = amin_shear +  i    * da0;
          const double fl = nz_lens_photoz(1.0/apl - 1.0, j) / (apl * apl);
          const double fh = nz_lens_photoz(1.0/aph - 1.0, j) / (aph * aph);
          // P integrand = n_j(z(a')) / a'^2
          // Q integrand = P integrand / chi(a')
          P += 0.5 * da0 * (fl + fh); // trapezoidal rule
          Q += 0.5 * da0 * (fl / chi(apl) + fh / chi(aph));
        }
        P0[j] = P;
        Q0[j] = Q;
      }
    }

    #pragma omp parallel for collapse(2)
    for (int j = 0; j < redshift.clustering_nbin; j++) {
      for (int i = 0; i < Na; i++) {
        const double ap = amin + i * da;
        const double z  = 1.0/ap - 1.0;
        Pint[j][i] = nz_lens_photoz(z, j) / (ap * ap);
        Qint[j][i] = Pint[j][i] / chi(amin + i * da);
      }
    }
    
    #pragma omp parallel for
    for (int j = 0; j < redshift.clustering_nbin; j++) {
      double P = P0[j];
      double Q = Q0[j]; 
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