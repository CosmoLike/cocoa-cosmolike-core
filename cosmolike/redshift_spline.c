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
  static double chiref[10] = {-1.};
    
  if (chiref[0] < 0) {
    for (int i=0; i<redshift.clustering_nbin; i++) {
      chiref[i] = chi(1.0/(1. + 0.5 * (redshift.clustering_zdist_zmin[i] + 
                                       redshift.clustering_zdist_zmax[i])));
    }
  }

  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  }
  
  const double R_min = like.Rmin_bias; // set minimum scale to which
                                       // we trust our bias model, in Mpc/h
  const double kmax = 2.0*M_PI / R_min * cosmology.coverH0;
  
  int res = 0.0;
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
    const double zhisto_max = z_v[nzbins - 1] + dz_histo;
    const int nj = (int) floor((z - zhisto_min) / dz_histo);
    if (ni < 0 || ni > ntomo-1 || nj < 0 || nj > nzbins-1) {
      log_fatal("invalid bin input (zbin = ni, bin = nj) = (%d, %d)", ni, nj);
      exit(1);
    } 
    res = tab[ni][nj];
  }
  return res;
}

double zdistr_photoz(double zz, const int nj) 
{
  static double cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static gsl_interp* photoz_splines[MAX_SIZE_ARRAYS+1];
  
  if (table == NULL || fdiff(cache[0], redshift.random_shear)) { 
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
{
  double* ar = (double*) params;
  const int ni = (int) ar[0];
  
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni); exit(1);
  } 
  return z * zdistr_photoz(z, ni);
}

double zmean_source(int ni) 
{ // mean true redshift of source galaxies in tomography bin j
  static double cache[MAX_SIZE_ARRAYS];
  static double* table = NULL;
  static gsl_integration_glfixed_table* w;

  if (table == NULL || 
      fdiff(cache[0], Ntable.random) ||
      fdiff(cache[1], redshift.random_shear))
  {    
    if (table != NULL) free(table);
    table = (double*) malloc1d(redshift.shear_nbin);
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 256 : 
                         (1 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);

    (void) zdistr_photoz(0., 0); // init static variables
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
{ // based file with structure z[i] nz[0][i] .. nz[redshift.clustering_nbin-1][i]
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
    const double zhisto_max = z_v[nzbins - 1] + dz_histo;

    const int nj = (int) floor((z - zhisto_min) / dz_histo);
    
    if (ni < 0 || ni > ntomo - 1 || nj < 0 || nj > nzbins - 1) {
      log_fatal("invalid bin input (zbin = ni, bin = nj) = (%d, %d)", ni, nj);
      exit(1);
    } 
    res = tab[ni][nj];
  }
  return res;
}

double pf_photoz(double zz, int nj) 
{
  static double cache_redshift_nz_params_clustering;
  static double** table = NULL;
  static gsl_interp* photoz_splines[MAX_SIZE_ARRAYS+1];

  if (table == NULL || 
      fdiff(cache_redshift_nz_params_clustering, redshift.random_clustering)) 
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

    cache_redshift_nz_params_clustering = redshift.random_clustering;
  }
  
  const int ntomo  = redshift.clustering_nbin;
  const int nzbins = redshift.clustering_nzbins;

  if (nj < 0 || nj > ntomo - 1) {
    log_fatal("nj = %d bin outside range (max = %d)", nj, ntomo);
    exit(1);
  }
  
  zz  = (zz - nuisance.photoz[1][0][nj]
            - redshift.clustering_zdist_zmean[nj])/nuisance.photoz[1][1][nj] 
        + redshift.clustering_zdist_zmean[nj];

  //zz  = zz - nuisance.photoz[1][0][nj];
  
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
  double* ar = (double*) params;
  const int ni = (int) ar[0];
  
  if (ni < 0 || ni > redshift.clustering_nbin - 1)
  {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  } 
  return z * pf_photoz(z, ni);
}

double norm_for_zmean(double z, void* params) 
{
  double* ar = (double*) params;
  const int ni = (int) ar[0];
  
  if (ni < 0 || ni > redshift.clustering_nbin - 1)
  {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  } 
  return pf_photoz(z, ni);
}

double zmean(const int ni)
{ // mean true redshift of galaxies in tomography bin j
  static double cache[MAX_SIZE_ARRAYS];
  static double* table = NULL;
  static gsl_integration_glfixed_table* w;

  if (table == NULL || 
      fdiff(cache[0], Ntable.random) ||
      fdiff(cache[1], redshift.random_clustering))
  {
    if (table != NULL) free(table);
    table = (double*) malloc1d(redshift.clustering_nbin+1);
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 256 : 
                         (1 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);

    (void) pf_photoz(0., 0); // init static vars
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
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  }  
  return table[ni];
}

// ------------------------------------------------------------------------
// ------------------------------------------------------------------------
// Bin-averaged lens efficiencies
// ------------------------------------------------------------------------
// ------------------------------------------------------------------------

double int_for_g_tomo(double aprime, void* params) 
{
  if (!(aprime>0) || !(aprime<1)) {
    log_fatal("a>0 and a<1 not true");
    exit(1);
  }
  double *ar = (double*)params;
  
  const int ni = (int) ar[0];
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  } 
  
  const double chi1 = ar[1];
  const double chi_prime = chi(aprime);
  return zdistr_photoz(1. / aprime - 1., ni) * f_K(chi_prime - chi1) /
    f_K(chi_prime) / (aprime * aprime);
}

double g_tomo(double ainput, const int ni) 
{ 
  static double cache[MAX_SIZE_ARRAYS]; 
  static double** table = NULL;
  static gsl_integration_glfixed_table* w = NULL;

  const double amin = 1.0/(redshift.shear_zdist_zmax_all + 1.0);
  const double amax = 0.999999;
  const double da = (amax - amin)/((double) Ntable.N_a - 1.0);

  if (NULL == table || fdiff(cache[0], Ntable.random))  {
    if (table != NULL) free(table);
    table = (double**) malloc2d(redshift.shear_nbin, Ntable.N_a);
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 256 : 
                         (1 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
  }
  if (fdiff(cache[0], Ntable.random) ||
      fdiff(cache[1], cosmology.random) ||
      fdiff(cache[2], nuisance.random_photoz_shear) ||
      fdiff(cache[3], redshift.random_shear)) 
  {
    { // init static variables
      double ar[2] = {(double) 0, chi(amin)};
      (void) int_for_g_tomo(amin, (void*) ar);
    }
    #pragma omp parallel for collapse(2)
    for (int j=0; j<redshift.shear_nbin; j++) {
      for (int i=0; i<Ntable.N_a; i++) {
        const double a = amin + i*da;       
        double ar[2] = {(double) j, chi(a)};
        gsl_function F;
        F.params = ar;
        F.function = int_for_g_tomo;       
        table[j][i] = gsl_integration_glfixed(&F, amin, a, w);
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
  return (ainput <= amin || ainput > 1.0 - da) ? 0.0 :
    interpol1d(table[ni], Ntable.N_a, amin, amax, da, ainput);
}

double int_for_g2_tomo(double aprime, void* params) 
{ // \int n(z') W(z,z')^2 routines for source clustering
  if (!(aprime>0) || !(aprime<1))
  {
    log_fatal("a>0 and a<1 not true");
    exit(1);
  }
  double *ar = (double*) params;
  
  const int ni = (int) ar[0];
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni); exit(1);
  } 
  
  const double chi1 = ar[1];
  const double chi_prime = chi(aprime);
  
  return zdistr_photoz(1./aprime-1., ni) * f_K(chi_prime - chi1)/
    f_K(chi_prime - chi1)/(f_K(chi_prime)*f_K(chi_prime))/(aprime*aprime);
}

double g2_tomo(double a, int ni) 
{ // for tomography bin ni
  static double cache_cosmo_params;
  static double cache_table_params;
  static double cache_photoz_nuisance_params_shear;
  static double cache_redshift_nz_params_shear;
  static double** table = NULL;

  const double amin = 1.0/(redshift.shear_zdist_zmax_all + 1.0);
  const double amax = 0.999999;
  const double da = (amax - amin)/((double) Ntable.N_a - 1.0);

  if (table == NULL || 
      fdiff(cache_table_params, Ntable.random)) 
  {
    if (table != NULL) {
      free(table);
    }
    table = (double**) malloc2d(redshift.shear_nbin, Ntable.N_a);
  }

  if (fdiff(cache_cosmo_params, cosmology.random) ||
      fdiff(cache_photoz_nuisance_params_shear, nuisance.random_photoz_shear) || 
      fdiff(cache_redshift_nz_params_shear, redshift.random_shear)  || 
      fdiff(cache_table_params, Ntable.random)) 
  {
    { // init static variables
      double ar[2] = {(double) 0, chi(amin)};
      (void) int_for_g2_tomo(amin, (void*) ar);
    }
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 256 : 
                         (1 == hdi) ? 512 : 1024; // predefined GSL tables
    gsl_integration_glfixed_table* w = malloc_gslint_glfixed(szint);

    #pragma omp parallel for collapse(2)
    for (int j=0; j<redshift.shear_nbin; j++) 
    {
      for (int i=0; i<Ntable.N_a; i++) 
      {
        const double a = amin + i*da;    
        double ar[2] = {(double) j, chi(a)};
    
        gsl_function F;
        F.params = ar;
        F.function = int_for_g2_tomo;

        table[j][i] = gsl_integration_glfixed(&F, amin, a, w);
      } 
    }

    gsl_integration_glfixed_table_free(w);
    cache_cosmo_params = cosmology.random;
    cache_table_params = Ntable.random;
    cache_redshift_nz_params_shear = redshift.random_shear;
    cache_photoz_nuisance_params_shear = nuisance.random_photoz_shear;
  }

  if (ni < 0 || ni > redshift.shear_nbin - 1)
  {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  } 
 
  return (a <= amin || a > 1.0 - da) ? 0.0 : 
    interpol1d(table[ni], Ntable.N_a, amin, amax, da, a);
}

double int_for_g_lens(double aprime, void* params) 
{
  double *ar = (double*) params;

  const int ni = (int) ar[0];
  if (ni < 0 || ni > redshift.clustering_nbin - 1)
  {
    log_fatal("invalid bin input ni = %d", ni);
    exit(1);
  }

  const double chi1 = ar[1];
  const double chi_prime = chi(aprime);
  
  return pf_photoz(1. / aprime - 1., ni) * f_K(chi_prime - chi1) /
        f_K(chi_prime) / (aprime * aprime);
}

double g_lens(double a, int ni) 
{ // for *lens* tomography bin ni
  static double cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static gsl_integration_glfixed_table* w = NULL;

  const double amin = 1.0/(redshift.clustering_zdist_zmax_all + 1.0);
  const double amax = 0.999999;
  const double da = (amax - amin)/((double) Ntable.N_a - 1.0);
  const double amin_shear = 1. / (redshift.shear_zdist_zmax_all + 1.);

  if (table == NULL || fdiff(cache[0], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double**) malloc2d(redshift.clustering_nbin , Ntable.N_a);
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 128 : 
                         (1 == hdi) ? 256 : 
                         (2 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
  }
  if (fdiff(cache[0], Ntable.random) ||
      fdiff(cache[1], cosmology.random) ||
      fdiff(cache[2], nuisance.random_photoz_clustering) ||
      fdiff(cache[3], redshift.random_clustering)) 
  {
    { // init static variables
      double ar[2] = {(double) 0, chi(amin)};
     (void) int_for_g_lens(amin_shear, (void*) ar);
    }
    #pragma omp parallel for collapse(2)
    for (int j=0; j<redshift.clustering_nbin; j++) {
      for (int i=0; i<Ntable.N_a; i++) {
        const double a =  amin + i*da;
        double ar[2] = {(double) j, chi(a)};
        gsl_function F;
        F.params = ar;
        F.function = int_for_g_lens;
        table[j][i] = gsl_integration_glfixed(&F, amin_shear, a, w);
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
  return (a < amin || a > 1.0 - da) ? 0.0 :
    interpol1d(table[ni], Ntable.N_a, amin, amax, da, a);
}

double g_cmb(double a) 
{
  static double cache_cosmo_params;
  static double chi_cmb = 0.;
  static double fchi_cmb = 0.;
  
  if (fdiff(cache_cosmo_params, cosmology.random)) 
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