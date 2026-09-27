#include <assert.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gsl/gsl_sum.h>
#include <gsl/gsl_integration.h>
#include <fftw3.h>

#include "bias.h"
#include "basics.h"
#include "cfastpt/cfastpt.h"
#include "cosmo3D.h"
#include "cosmo2D.h"
#include "cosmo2D_scuts.h"
#include "halo.h"
#include "IA.h"
#include "pt_cfastpt.h"
#include "radial_weights.h"
#include "redshift_spline.h"
#include "structs.h"
#include "log.c/src/log.h"

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// DERIVATIVE: dlnX/dlnk: FOURIER SPACE
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cached dC_ss/dlnk (2011.06469 eq 17): interpolates a (ln k, ln l) table
// filled by dC_ss_dlnk_tomo_limber_work. The table spans the
// Ntable.dCX_dlnk k range and every multipole l >= 1, rebuilt when the
// cosmology, the shear nuisances or Ntable change; a (k, l) outside the
// table returns 0.
// ---------------------------------------------------------------------------
double dC_ss_dlnk_tomo_limber(
    const double k,   // wavenumber in (Mpc/h)^-1
    const double l,   // multipole
    const int ni,     // first source redshift bin
    const int nj,     // second source redshift bin
    const int EE      // 1 = E-mode, 0 = B-mode
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double**** table;
  static double lim[6];
  static int nell;
  static int nlnk;
  
  if (NULL == table || fdiff2(cache[4], Ntable.random)) {
    nell = Ntable.N_ell;
    lim[0] = 0.0; // ln(l = 1): the grid covers every multipole l >= 1
    lim[1] = log(Ntable.LMAX + 1.);
    lim[2] = (lim[1] - lim[0]) / ((double) nell - 1.);

    nlnk = Ntable.dCX_dlnk_nlnk;
    lim[3] = log(Ntable.dCX_dlnk_kmin);
    lim[4] = log(Ntable.dCX_dlnk_kmax);
    lim[5] = (lim[4] - lim[3]) / ((double) nlnk - 1.);

    if (table != NULL) free(table);
    table = (double****) malloc4d(2, tomo.shear_Npowerspectra, nlnk, nell);
  }
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    double* lnkx = (double*) malloc1d(nlnk);
    double* lx = (double*) malloc1d(nell);
    for (int f=0; f<nlnk; f++) {
      lnkx[f] = lim[3] + f*lim[5];
    }
    for (int i=0; i<nell; i++) {
      lx[i] = exp(lim[0] + i*lim[2]);
    }
    dC_ss_dlnk_tomo_limber_work(lnkx, nlnk, lx, nell,
                                tomo.shear_Npowerspectra, 0, table);
    free(lnkx);
    free(lx);
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear;
    cache[4] = Ntable.random;
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (ni,nj) = [%d,%d]",ni,nj); exit(1);
  }
  const int q = N_shear(ni, nj);
  if (q < 0 || q > tomo.shear_Npowerspectra - 1) {
    log_fatal("internal logic error in selecting bin number"); exit(1);
  }
  const double lnl = log(l);
  const double lnk = log(k);
  return (lnk < lim[3] || lnk > lim[4]) ? 0.0 :
         (lnl < lim[0] || lnl > lim[1]) ? 0.0 :
         interpol2d((1==EE) ? table[0][q] : table[1][q],
                    nlnk, lim[3], lim[4], lim[5], lnk,
                    nell, lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cached dlnC_ss/dlnk = (dC_ss/dlnk)/C_ss (2011.06469 eq 17): interpolates
// a (ln k, ln l) table filled by the normalized mode of
// dC_ss_dlnk_tomo_limber_work, on the same grid and with the same rebuild
// keys as the dC table above; a (k, l) outside the table returns 0.
// ---------------------------------------------------------------------------
double dlnC_ss_dlnk_tomo_limber(
    const double k,   // wavenumber in (Mpc/h)^-1
    const double l,   // multipole
    const int ni,     // first source redshift bin
    const int nj,     // second source redshift bin
    const int EE      // 1 = E-mode, 0 = B-mode
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double**** table;
  static double lim[6];
  static int nell;
  static int nlnk;
  
  if (NULL == table || fdiff2(cache[4], Ntable.random)) {
    nell = Ntable.N_ell;
    lim[0] = 0.0; // ln(l = 1): the grid covers every multipole l >= 1
    lim[1] = log(Ntable.LMAX + 1.);
    lim[2] = (lim[1] - lim[0]) / ((double) nell - 1.);

    nlnk = Ntable.dCX_dlnk_nlnk;
    lim[3] = log(Ntable.dCX_dlnk_kmin);
    lim[4] = log(Ntable.dCX_dlnk_kmax);
    lim[5] = (lim[4] - lim[3]) / ((double) nlnk - 1.);

    if (table != NULL) free(table);
    table = (double****) malloc4d(2, tomo.shear_Npowerspectra, nlnk, nell);
  }
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    double* lnkx = (double*) malloc1d(nlnk);
    double* lx = (double*) malloc1d(nell);
    for (int f=0; f<nlnk; f++) {
      lnkx[f] = lim[3] + f*lim[5];
    }
    for (int i=0; i<nell; i++) {
      lx[i] = exp(lim[0] + i*lim[2]);
    }
    dC_ss_dlnk_tomo_limber_work(lnkx, nlnk, lx, nell,
                                tomo.shear_Npowerspectra, 1, table);
    free(lnkx);
    free(lx);
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear;
    cache[4] = Ntable.random;
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (ni,nj) = [%d,%d]",ni,nj); exit(1);
  }
  const int q = N_shear(ni, nj);
  if (q < 0 || q > tomo.shear_Npowerspectra - 1) {
    log_fatal("internal logic error in selecting bin number"); exit(1);
  }
  const double lnl = log(l);
  const double lnk = log(k);
  return (lnk < lim[3] || lnk > lim[4]) ? 0.0 :
         (lnl < lim[0] || lnl > lim[1]) ? 0.0 :
         interpol2d((1==EE) ? table[0][q] : table[1][q],
                    nlnk, lim[3], lim[4], lim[5], lnk,
                    nell, lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cached dC_ks/dlnk (2011.06469 eq 17): interpolates a (ln k, ln l) table
// filled by dC_ks_dlnk_tomo_limber_work. The table spans the
// Ntable.dCX_dlnk k range and every multipole l >= 1, rebuilt when the
// cosmology, the shear nuisances or Ntable change (the same keys as the
// C_ks C_ell cache; the CMB beam enters only downstream, in the w_ks
// projection, so cmb.random is not a key here); a (k, l) outside the
// table returns 0.
// ---------------------------------------------------------------------------
double dC_ks_dlnk_tomo_limber(
    const double k,   // wavenumber in (Mpc/h)^-1
    const double l,   // multipole
    const int ni      // source redshift bin
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double*** table;
  static double lim[6];
  static int nell;
  static int nlnk;

  if (NULL == table || fdiff2(cache[4], Ntable.random)) {
    nell = Ntable.N_ell;
    lim[0] = 0.0; // ln(l = 1): the grid covers every multipole l >= 1
    lim[1] = log(Ntable.LMAX + 1.);
    lim[2] = (lim[1] - lim[0]) / ((double) nell - 1.);

    nlnk = Ntable.dCX_dlnk_nlnk;
    lim[3] = log(Ntable.dCX_dlnk_kmin);
    lim[4] = log(Ntable.dCX_dlnk_kmax);
    lim[5] = (lim[4] - lim[3]) / ((double) nlnk - 1.);

    if (table != NULL) free(table);
    table = (double***) malloc3d(redshift.shear_nbin, nlnk, nell);
  }
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    double* lnkx = (double*) malloc1d(nlnk);
    double* lx = (double*) malloc1d(nell);
    for (int f=0; f<nlnk; f++) {
      lnkx[f] = lim[3] + f*lim[5];
    }
    for (int i=0; i<nell; i++) {
      lx[i] = exp(lim[0] + i*lim[2]);
    }
    dC_ks_dlnk_tomo_limber_work(lnkx, nlnk, lx, nell,
                                redshift.shear_nbin, 0, table);
    free(lnkx);
    free(lx);
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear;
    cache[4] = Ntable.random;
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  const double lnl = log(l);
  const double lnk = log(k);
  return (lnk < lim[3] || lnk > lim[4]) ? 0.0 :
         (lnl < lim[0] || lnl > lim[1]) ? 0.0 :
         interpol2d(table[ni],
                    nlnk, lim[3], lim[4], lim[5], lnk,
                    nell, lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cached dlnC_ks/dlnk = (dC_ks/dlnk)/C_ks (2011.06469 eq 17): interpolates
// a (ln k, ln l) table filled by the normalized mode of
// dC_ks_dlnk_tomo_limber_work, on the same grid and with the same rebuild
// keys as the dC table above; a (k, l) outside the table returns 0.
// ---------------------------------------------------------------------------
double dlnC_ks_dlnk_tomo_limber(
    const double k,   // wavenumber in (Mpc/h)^-1
    const double l,   // multipole
    const int ni      // source redshift bin
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double*** table;
  static double lim[6];
  static int nell;
  static int nlnk;

  if (NULL == table || fdiff2(cache[4], Ntable.random)) {
    nell = Ntable.N_ell;
    lim[0] = 0.0; // ln(l = 1): the grid covers every multipole l >= 1
    lim[1] = log(Ntable.LMAX + 1.);
    lim[2] = (lim[1] - lim[0]) / ((double) nell - 1.);

    nlnk = Ntable.dCX_dlnk_nlnk;
    lim[3] = log(Ntable.dCX_dlnk_kmin);
    lim[4] = log(Ntable.dCX_dlnk_kmax);
    lim[5] = (lim[4] - lim[3]) / ((double) nlnk - 1.);

    if (table != NULL) free(table);
    table = (double***) malloc3d(redshift.shear_nbin, nlnk, nell);
  }
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    double* lnkx = (double*) malloc1d(nlnk);
    double* lx = (double*) malloc1d(nell);
    for (int f=0; f<nlnk; f++) {
      lnkx[f] = lim[3] + f*lim[5];
    }
    for (int i=0; i<nell; i++) {
      lx[i] = exp(lim[0] + i*lim[2]);
    }
    dC_ks_dlnk_tomo_limber_work(lnkx, nlnk, lx, nell,
                                redshift.shear_nbin, 1, table);
    free(lnkx);
    free(lx);
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear;
    cache[4] = Ntable.random;
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  const double lnl = log(l);
  const double lnk = log(k);
  return (lnk < lim[3] || lnk > lim[4]) ? 0.0 :
         (lnl < lim[0] || lnl > lim[1]) ? 0.0 :
         interpol2d(table[ni],
                    nlnk, lim[3], lim[4], lim[5], lnk,
                    nell, lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// physical mode cut-off (R = response function): FOURIER SPACE
// find kk such that RF \equiv \int_{-\infty}^{ln(kk)} dlnk |dlnXdlnk| = alpha
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Batch computation of the response function (2011.06469 eq 17)
//
//   RF(kmax, l) = int_{-infty}^{ln kmax} dlnk |dlnC_ss/dlnk|
//                 / int_{-infty}^{+infty} dlnk |dlnC_ss/dlnk|
//
// on a (ln kmax, ell) grid, for every tomographic pair. Both integrals
// map onto t in (0, 1] — the numerator through lnk = ln kmax - (1-t)/t,
// the denominator through lnk = +-(1-t)/t — and use the same fixed
// Gauss-Legendre rule the retired scalar version used, with the nodes
// and weights precomputed into plain arrays instead of driving a GSL
// integrand callback per point.
//
// The denominator does not depend on kmax, so one thread team first
// fills one denominator value per (tomo pair, ell) and then, after the
// loop's implicit barrier, accumulates the numerator for every
// (pair, kmax, ell) output and divides. Every integrand evaluation reads
// the cached dlnC_ss_dlnk_tomo_limber table, built once, single-threaded,
// before the parallel region.
// ---------------------------------------------------------------------------
void RF_C_ss_tomo_limber_work(
    const double* lnkmaxx, // ln kmax values (length nkmax), k in (Mpc/h)^-1
    const int nkmax,       // number of ln kmax values
    const double* lx,      // multipole values (length nl)
    const int nl,          // number of multipole values
    const int NSIZE,       // number of tomo shear power spectra
    double**** table       // output [2][NSIZE][nkmax][nl]: EE and BB
  )
{
  if (nkmax <= 0 || nl <= 0) {
    log_fatal("nkmax = %d and nl = %d must be positive", nkmax, nl);
    exit(1);
  }
  // Gauss-Legendre nodes and weights on t in [1e-5, 1] as plain arrays
  const int hdi = abs(Ntable.high_def_integration);
  const size_t szint = (0 == hdi) ? 256 :
                       (1 == hdi) ? 512 : 1024; // predefined GSL tables
  gsl_integration_glfixed_table* w = malloc_gslint_glfixed(szint);
  const int npts = (int) w->n;
  double* tq = (double*) malloc1d(npts);
  double* wq = (double*) malloc1d(npts);
  for (int p = 0; p < npts; p++) {
    gsl_integration_glfixed_point(1e-5, 1.0, p, &tq[p], &wq[p], w);
  }
  gsl_integration_glfixed_table_free(w);
  // the denominator's k nodes depend on nothing: precompute them once
  double* kd1 = (double*) malloc1d(npts);
  double* kd2 = (double*) malloc1d(npts);
  for (int p = 0; p < npts; p++) {
    kd1[p] = exp((1. - tq[p])/tq[p]);
    kd2[p] = exp(-(1. - tq[p])/tq[p]);
  }
  double*** den = (double***) malloc3d(2, NSIZE, nl);
  // build the cached dlnC table (and the statics it warms) single-threaded
  (void) dlnC_ss_dlnk_tomo_limber(1.0, lx[0], Z1(0), Z2(0), 1);
  #pragma omp parallel
  {
  #pragma omp for collapse(2) schedule(static)
  for (int nz = 0; nz < NSIZE; nz++) {
    for (int i = 0; i < nl; i++) {
      const int Z1NZ = Z1(nz);
      const int Z2NZ = Z2(nz);
      const double l = lx[i];
      double sEE = 0.0;
      double sBB = 0.0;
      for (int p = 0; p < npts; p++) {
        const double wt = wq[p]/(tq[p]*tq[p]);
        sEE += (fabs(dlnC_ss_dlnk_tomo_limber(kd1[p], l, Z1NZ, Z2NZ, 1)) +
                fabs(dlnC_ss_dlnk_tomo_limber(kd2[p], l, Z1NZ, Z2NZ, 1)))*wt;
        sBB += (fabs(dlnC_ss_dlnk_tomo_limber(kd1[p], l, Z1NZ, Z2NZ, 0)) +
                fabs(dlnC_ss_dlnk_tomo_limber(kd2[p], l, Z1NZ, Z2NZ, 0)))*wt;
      }
      den[0][nz][i] = sEE;
      den[1][nz][i] = sBB;
    }
  } // implicit barrier: denominators complete before the division below
  #pragma omp for collapse(3) schedule(static)
  for (int nz = 0; nz < NSIZE; nz++) {
    for (int m = 0; m < nkmax; m++) {
      for (int i = 0; i < nl; i++) {
        const int Z1NZ = Z1(nz);
        const int Z2NZ = Z2(nz);
        const double l = lx[i];
        double sEE = 0.0;
        double sBB = 0.0;
        for (int p = 0; p < npts; p++) {
          const double k = exp(lnkmaxx[m] - (1. - tq[p])/tq[p]);
          const double wt = wq[p]/(tq[p]*tq[p]);
          sEE += fabs(dlnC_ss_dlnk_tomo_limber(k, l, Z1NZ, Z2NZ, 1))*wt;
          sBB += fabs(dlnC_ss_dlnk_tomo_limber(k, l, Z1NZ, Z2NZ, 0))*wt;
        }
        table[0][nz][m][i] = sEE/den[0][nz][i];
        table[1][nz][m][i] = sBB/den[1][nz][i];
      }
    }
  }
  } // end of the parallel region
  free(den);
  free(kd1);
  free(kd2);
  free(tq);
  free(wq);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Batch computation of the response function (2011.06469 eq 17)
//
//   RF(kmax, l) = int_{-infty}^{ln kmax} dlnk |dlnC_ks/dlnk|
//                 / int_{-infty}^{+infty} dlnk |dlnC_ks/dlnk|
//
// on a (ln kmax, ell) grid, for every source bin. Same design as
// RF_C_ss_tomo_limber_work (the ks cross has one component per source
// bin, so there is no EE/BB leading dimension): both integrals map onto
// t in (0, 1] — the numerator through lnk = ln kmax - (1-t)/t, the
// denominator through lnk = +-(1-t)/t — with the fixed Gauss-Legendre
// nodes and weights precomputed into plain arrays.
//
// The denominator does not depend on kmax, so one thread team first
// fills one denominator value per (source bin, ell) and then, after the
// loop's implicit barrier, accumulates the numerator for every
// (bin, kmax, ell) output and divides. Every integrand evaluation reads
// the cached dlnC_ks_dlnk_tomo_limber table, built once, single-threaded,
// before the parallel region.
// ---------------------------------------------------------------------------
void RF_C_ks_tomo_limber_work(
    const double* lnkmaxx, // ln kmax values (length nkmax), k in (Mpc/h)^-1
    const int nkmax,       // number of ln kmax values
    const double* lx,      // multipole values (length nl)
    const int nl,          // number of multipole values
    const int NSIZE,       // number of source tomographic bins (= shear_nbin)
    double*** table        // output [NSIZE][nkmax][nl]
  )
{
  if (nkmax <= 0 || nl <= 0) {
    log_fatal("nkmax = %d and nl = %d must be positive", nkmax, nl);
    exit(1);
  }
  // Gauss-Legendre nodes and weights on t in [1e-5, 1] as plain arrays
  const int hdi = abs(Ntable.high_def_integration);
  const size_t szint = (0 == hdi) ? 256 :
                       (1 == hdi) ? 512 : 1024; // predefined GSL tables
  gsl_integration_glfixed_table* w = malloc_gslint_glfixed(szint);
  const int npts = (int) w->n;
  double* tq = (double*) malloc1d(npts);
  double* wq = (double*) malloc1d(npts);
  for (int p = 0; p < npts; p++) {
    gsl_integration_glfixed_point(1e-5, 1.0, p, &tq[p], &wq[p], w);
  }
  gsl_integration_glfixed_table_free(w);
  // the denominator's k nodes depend on nothing: precompute them once
  double* kd1 = (double*) malloc1d(npts);
  double* kd2 = (double*) malloc1d(npts);
  for (int p = 0; p < npts; p++) {
    kd1[p] = exp((1. - tq[p])/tq[p]);
    kd2[p] = exp(-(1. - tq[p])/tq[p]);
  }
  double** den = (double**) malloc2d(NSIZE, nl);
  // build the cached dlnC table (and the statics it warms) single-threaded
  (void) dlnC_ks_dlnk_tomo_limber(1.0, lx[0], 0);
  #pragma omp parallel
  {
  #pragma omp for collapse(2) schedule(static)
  for (int nz = 0; nz < NSIZE; nz++) {
    for (int i = 0; i < nl; i++) {
      const double l = lx[i];
      double sKS = 0.0;
      for (int p = 0; p < npts; p++) {
        const double wt = wq[p]/(tq[p]*tq[p]);
        sKS += (fabs(dlnC_ks_dlnk_tomo_limber(kd1[p], l, nz)) +
                fabs(dlnC_ks_dlnk_tomo_limber(kd2[p], l, nz)))*wt;
      }
      den[nz][i] = sKS;
    }
  } // implicit barrier: denominators complete before the division below
  #pragma omp for collapse(3) schedule(static)
  for (int nz = 0; nz < NSIZE; nz++) {
    for (int m = 0; m < nkmax; m++) {
      for (int i = 0; i < nl; i++) {
        const double l = lx[i];
        double sKS = 0.0;
        for (int p = 0; p < npts; p++) {
          const double k = exp(lnkmaxx[m] - (1. - tq[p])/tq[p]);
          const double wt = wq[p]/(tq[p]*tq[p]);
          sKS += fabs(dlnC_ks_dlnk_tomo_limber(k, l, nz))*wt;
        }
        table[nz][m][i] = sKS/den[nz][i];
      }
    }
  }
  } // end of the parallel region
  free(den);
  free(kd1);
  free(kd2);
  free(tq);
  free(wq);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// DERIVATIVE: dlnX/dlnk: REAL SPACE
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnxi_pm/dlnk at one wavenumber, for every tomographic pair and angular
// bin (2011.06469 eq 17):
//
//   dxi_pm/dlnk(theta)  = sum_l Glpm(theta, l) * (dC_EE +- dC_BB)(k, l)
//   dlnxi_pm/dlnk       = (dxi_pm/dlnk) / xi_pm(theta)
//
// Pipeline (the same design as the C_ell -> xi_pm real-space pipeline):
//   1. read dC_ss/dlnk at this k on the cached dC table's own multipole
//      log-grid — an exact-node read in ln l, so only the ln k direction
//      is interpolated;
//   2. gather those rows onto every integer multipole with the vectorized
//      linear interpolation of limber_fill_interp;
//   3. Legendre-sum against the bin-averaged Glpm kernels (hoisted
//      restrict pointers and a SIMD reduction, as in xi_pm_tomo);
//   4. normalize by xi_pm(theta).
// Steps 1 and 2 together are bilinear in (ln k, ln l) on the same knots
// as reading the 2D table once per integer multipole, so nothing is
// cached per integer multipole.
//
// Static state, rebuilt when Ntable or the tomography change:
//   Glpm[2][Ntheta][LMAX] - bin-averaged Legendre kernels (Gl+ and Gl-)
//   ln_ell[LMAX]          - log(l) at every integer multipole
//   dCgrid[2][NSIZE][N_ell], cx[2][NSIZE][LMAX] - work arrays the
//                           pipeline overwrites on every call
// ---------------------------------------------------------------------------
double** dlnxi_dlnk_pm_tomo_nointerp(
    const double k    // wavenumber in (Mpc/h)^-1
  ) // returns [2][NSIZE*Ntheta] (caller frees): [0] = xi+, [1] = xi-
{
  static double*** Glpm = NULL; //Glpm[0] = Gl+, Glpm[1] = Gl-
  static double* ln_ell = NULL;
  static double*** dCgrid = NULL; // dC at one k on the table's ell grid
  static double*** cx = NULL;     // dC at one k at every integer multipole
  static int NSIZE_alloc = 0;
  static uint64_t cache[MAX_SIZE_ARRAYS];
  const int lmin = 1;
  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized"); exit(1);
  }
  const int NSIZE = tomo.shear_Npowerspectra;
  if (NULL == Glpm || NSIZE != NSIZE_alloc || fdiff2(cache[4], Ntable.random))
  {
    if (Glpm != NULL) free(Glpm);
    Glpm = (double***) malloc3d(2, Ntable.Ntheta, Ntable.LMAX);
    
    double*** P = (double***) malloc3d(4, Ntable.Ntheta, Ntable.LMAX + 1);
    double** Pmin  = P[0]; double** Pmax  = P[1];
    double** dPmin = P[2]; double** dPmax = P[3];

    double xmin[Ntable.Ntheta];
    double xmax[Ntable.Ntheta];
    for (int i=0; i<Ntable.Ntheta; i++)
    { // Cocoa: dont thread (init of static variables inside set_bin_average)
      bin_avg r = set_bin_average(i, 0);
      xmin[i] = r.xmin;
      xmax[i] = r.xmax;
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<(Ntable.LMAX+1); l++) {
        bin_avg r   = set_bin_average(i, l);
        Pmin[i][l]  = r.Pmin;
        Pmax[i][l]  = r.Pmax;
        dPmin[i][l] = r.dPmin;
        dPmax[i][l] = r.dPmax;
      }
    }
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<lmin; l++) {
        Glpm[0][i][l] = 0.0;
        Glpm[1][i][l] = 0.0;
      }
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=lmin; l<Ntable.LMAX; l++) {
        Glpm[0][i][l] = (2.*l+1)/(2.*M_PI*l*l*(l+1)*(l+1))*(
          -l*(l-1.)/2*(l+2./(2*l+1)) * (Pmin[i][l-1]-Pmax[i][l-1])
          -l*(l-1.)*(2.-l)/2 * (xmin[i]*Pmin[i][l]-xmax[i]*Pmax[i][l])
          +l*(l-1.)/(2.*l+1) * (Pmin[i][l+1]-Pmax[i][l+1])
          +(4-l)*(dPmin[i][l]-dPmax[i][l])
          +(l+2)*(xmin[i]*dPmin[i][l-1] - xmax[i]*dPmax[i][l-1] - Pmin[i][l-1] + Pmax[i][l-1])
          +2*(l-1)*(xmin[i]*dPmin[i][l] - xmax[i]*dPmax[i][l] - Pmin[i][l] + Pmax[i][l])
          -2*(l+2)*(dPmin[i][l-1]-dPmax[i][l-1])
        )/(xmin[i]-xmax[i]);

        Glpm[1][i][l] = (2.*l+1)/(2.*M_PI*l*l*(l+1)*(l+1))*(
          -l*(l-1.)/2*(l+2./(2*l+1)) * (Pmin[i][l-1]-Pmax[i][l-1])
          -l*(l-1.)*(2.-l)/2 * (xmin[i]*Pmin[i][l]-xmax[i]*Pmax[i][l])
          +l*(l-1.)/(2.*l+1)* (Pmin[i][l+1]-Pmax[i][l+1])
          +(4-l)*(dPmin[i][l]-dPmax[i][l])
          +(l+2)*(xmin[i]*dPmin[i][l-1] - xmax[i]*dPmax[i][l-1] - Pmin[i][l-1] + Pmax[i][l-1])
          -2*(l-1)*(xmin[i]*dPmin[i][l] - xmax[i]*dPmax[i][l] - Pmin[i][l] + Pmax[i][l])
          +2*(l+2)*(dPmin[i][l-1]-dPmax[i][l-1])
          )/(xmin[i]-xmax[i]);
      }
    }
    free(P);

    if (ln_ell != NULL) free(ln_ell);
    ln_ell = (double*) malloc1d(Ntable.LMAX);
    ln_ell[0] = 0.0; // unused (the sums start at lmin = 1)
    for (int l=1; l<Ntable.LMAX; l++) {
      ln_ell[l] = log((double) l);
    }
    // per-call work arrays whose sizes only change with Ntable or the
    // tomography: allocated once here, overwritten on every call
    if (dCgrid != NULL) free(dCgrid);
    dCgrid = (double***) malloc3d(2, NSIZE, Ntable.N_ell);
    zero3d(dCgrid, 2, NSIZE, Ntable.N_ell);
    if (cx != NULL) free(cx);
    cx = (double***) malloc3d(2, NSIZE, Ntable.LMAX);
    zero3d(cx, 2, NSIZE, Ntable.LMAX);
    NSIZE_alloc = NSIZE;
    cache[4] = Ntable.random;
  }
  double** ans = (double**) malloc2d(2, NSIZE*Ntable.Ntheta);
  #pragma omp parallel for collapse(3) schedule(static)
  for (int p=0; p<2; p++) {
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        const int q = nz * Ntable.Ntheta + i;
        ans[p][q] = 0.0;
      }
    }
  }
  const double lnk = log(k);
  if (lnk > log(Ntable.dCX_dlnk_kmin) && lnk < log(Ntable.dCX_dlnk_kmax)) {
    // build (or reuse) the cached dC table single-threaded before the
    // parallel loops below read it
    (void) dC_ss_dlnk_tomo_limber(k, (double) limits.LMIN_tab,
                                  Z1(0), Z2(0), 1);
    // dC_ss/dlnk at this k on the dC table's own multipole log-grid
    const int nell = Ntable.N_ell;
    const double la = 0.0; // ln(l = 1): the dC table's multipole grid start
    const double ldx = (log(Ntable.LMAX + 1.) - la)/((double) nell - 1.);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<nell; i++) {
        const double lg = exp(la + i*ldx);
        dCgrid[0][nz][i] = dC_ss_dlnk_tomo_limber(k, lg, Z1(nz), Z2(nz), 1);
        dCgrid[1][nz][i] = dC_ss_dlnk_tomo_limber(k, lg, Z1(nz), Z2(nz), 0);
      }
    }
    // gather onto every integer multipole (vectorized linear interpolation)
    #pragma omp parallel for schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      const double* tab2[2] = {dCgrid[0][nz], dCgrid[1][nz]};
      double* out2[2] = {cx[0][nz], cx[1][nz]};
      limber_fill_interp(2, tab2, out2, lmin, Ntable.LMAX, ln_ell,
                         la, 1.0/ldx, nell);
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        // Local restrict pointers: without these, GCC cannot prove the
        // Glpm and cx rows don't alias (pointer-to-pointer indirection
        // inside a collapse(2) OpenMP region) and gives up on the SIMD
        // reduction below
        const double* restrict c0 = cx[0][nz];
        const double* restrict c1 = cx[1][nz];
        const double* restrict g0 = Glpm[0][i];
        const double* restrict g1 = Glpm[1][i];
        double sum0 = 0.0;
        double sum1 = 0.0;
        #pragma omp simd reduction(+:sum0, sum1)
        for (int l=lmin; l<Ntable.LMAX; l++) {
          sum0 += g0[l] * (c0[l] + c1[l]);
          sum1 += g1[l] * (c0[l] - c1[l]);
        }
        const int q = nz * Ntable.Ntheta + i;
        ans[0][q] = sum0;
        ans[1][q] = sum1;
      }
    }
    // warm xi_pm_tomo's cached tables single-threaded (one call builds
    // both xi+ and xi-); the parallel loop below then only reads them
    (void) xi_pm_tomo(0, 0, Z1(0), Z2(0), 1);
    #pragma omp parallel for collapse(3) schedule(static)
    for (int p=0; p<2; p++) {
      for (int nz=0; nz<NSIZE; nz++) {
        for (int i=0; i<Ntable.Ntheta; i++) {
          const int q = nz * Ntable.Ntheta + i;
          const double dxipmdlnk = ans[p][q];
          if (fabs(ans[p][q])>1.e-50) {
            const double xipm = xi_pm_tomo(p, i, Z1(nz), Z2(nz), 1);
            ans[p][q] = (fabs(xipm) > 1.e-50) ? dxipmdlnk/xipm : 0.0;
          }
        }
      }
    }
  }
  return ans;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cached dlnxi_pm/dlnk: interpolates in ln k a table of
// dlnxi_dlnk_pm_tomo_nointerp results on the Ntable.dCX_dlnk k grid,
// rebuilt when the cosmology, the shear nuisances or Ntable change; a k
// outside the grid returns 0. This is what the RF_xi integrals read.
// ---------------------------------------------------------------------------
double dlnxi_dlnk_pm_tomo(
    const double k,   // wavenumber in (Mpc/h)^-1
    const int pm,     // 1 = xi_+, 0 = xi_-
    const int nt,     // angular bin index (0..Ntheta-1)
    const int ni,     // first source redshift bin
    const int nj      // second source redshift bin
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double*** table = NULL; 
  static double lim[6];
  static int nlnk;
  const int NSIZE = tomo.shear_Npowerspectra;
  if (NULL == table || fdiff2(cache[4], Ntable.random)) {
    nlnk = Ntable.dCX_dlnk_nlnk;
    lim[0] = log(Ntable.dCX_dlnk_kmin);
    lim[1] = log(Ntable.dCX_dlnk_kmax);
    lim[2] = (lim[1] - lim[0]) / ((double) nlnk - 1.);
    if (table != NULL) free(table);
    table = (double***) malloc3d(2, NSIZE*Ntable.Ntheta, nlnk);
  }
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    for (int f=0; f<nlnk; f++) {  
      double** tmp = dlnxi_dlnk_pm_tomo_nointerp(exp(lim[0] + f*lim[2]));
      for (int p=0; p<2; p++) {
        for (int nz=0; nz<NSIZE; nz++) {
          for (int i=0; i<Ntable.Ntheta; i++) {
            const int q = nz * Ntable.Ntheta + i;
            table[p][q][f] =  tmp[p][q];
          }
        }
      }
      free(tmp);
    }
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear;
    cache[4] = Ntable.random; 
  }
  if (nt < 0 || nt > Ntable.Ntheta - 1) {
    log_fatal("error in selecting bin number nt = %d", nt); exit(1); 
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (ni,nj) = [%d,%d]",ni,nj); exit(1);
  }
  const int ntomo = N_shear(ni, nj);
  const int q = ntomo*Ntable.Ntheta + nt;
  if (q < 0 || q > NSIZE*Ntable.Ntheta - 1) {
    log_fatal("internal logic error in selecting bin number"); exit(1);
  }
  const double lnk = log(k);
  return (lnk < lim[0] || lnk > lim[1]) ? 0.0 :
         interpol1d((pm>0)?table[0][q]:table[1][q],nlnk,lim[0],lim[1],lim[2],lnk);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// dlnw_ks/dlnk at one wavenumber, for every source bin and angular bin
// (2011.06469 eq 17):
//
//   dw_ks/dlnk(theta) = sum_l Pl(theta, l) * cmbf(l) * dC_ks(k, l)
//   dlnw_ks/dlnk      = (dw_ks/dlnk) / w_ks(theta)
//
// Pipeline (the same design as dlnxi_dlnk_pm_tomo_nointerp, with the
// w_ks projection in place of xi_pm's):
//   1. read dC_ks/dlnk at this k on the cached dC table's own multipole
//      log-grid — an exact-node read in ln l, so only the ln k direction
//      is interpolated;
//   2. gather those rows onto every integer multipole with the vectorized
//      linear interpolation of limber_fill_interp;
//   3. Legendre-sum against the bin-averaged Pl kernel — the same spin-0
//      x spin-2 (gamma_t-type) kernel as w_ks_tomo — with each multipole
//      weighted by the CMB filter cmbf(l) = beam_cmb(l) (times the
//      HEALPix pixel window when one is loaded), exactly as w_ks_tomo
//      filters its C_l before summing;
//   4. normalize by w_ks(theta).
// Steps 1 and 2 together are bilinear in (ln k, ln l) on the same knots
// as reading the 2D table once per integer multipole, so nothing is
// cached per integer multipole.
//
// Static state, rebuilt when Ntable or the tomography change:
//   Pl[Ntheta][LMAX] - bin-averaged Legendre kernel (as in w_ks_tomo)
//   ln_ell[LMAX]     - log(l) at every integer multipole
//   dCgrid[NSIZE][N_ell], cx[NSIZE][LMAX] - work arrays the pipeline
//                      overwrites on every call
//   cmbf[LMAX]       - CMB filter, refilled when cmb or Ntable change
// ---------------------------------------------------------------------------
double* dlnw_ks_dlnk_tomo_nointerp(
    const double k    // wavenumber in (Mpc/h)^-1
  ) // returns [NSIZE*Ntheta] (caller frees), NSIZE = shear_nbin
{
  static double** Pl = NULL;
  static double* cmbf = NULL;   // CMB filter
  static double* ln_ell = NULL;
  static double** dCgrid = NULL; // dC at one k on the table's ell grid
  static double** cx = NULL;     // dC at one k at every integer multipole
  static int NSIZE_alloc = 0;
  static uint64_t cache[MAX_SIZE_ARRAYS];
  const int lmin = 1;
  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized"); exit(1);
  }
  const int NSIZE = redshift.shear_nbin;
  if (NSIZE <= 0) {
    log_fatal("dlnw_ks requested but redshift.shear_nbin = %d", NSIZE);
    exit(1);
  }
  if (NULL == Pl || NSIZE != NSIZE_alloc || fdiff2(cache[4], Ntable.random))
  {
    if (Pl != NULL) free(Pl);
    Pl = (double**) malloc2d(Ntable.Ntheta, Ntable.LMAX);

    double*** P = (double***) malloc3d(2, Ntable.Ntheta, Ntable.LMAX + 1);
    double** Pmin  = P[0]; double** Pmax  = P[1];

    double xmin[Ntable.Ntheta];
    double xmax[Ntable.Ntheta];
    for (int i=0; i<Ntable.Ntheta; i++)
    { // Cocoa: dont thread (init of static variables inside set_bin_average)
      bin_avg r = set_bin_average(i, 0);
      xmin[i] = r.xmin;
      xmax[i] = r.xmax;
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<(Ntable.LMAX+1); l++) {
        bin_avg r  = set_bin_average(i, l);
        Pmin[i][l] = r.Pmin;
        Pmax[i][l] = r.Pmax;
      }
    }
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<lmin; l++) {
        Pl[i][l] = 0.0;
      }
    }
    // the same bin-averaged spin-0 x spin-2 kernel as w_ks_tomo (see the
    // w_gammat kernel documentation for the derivation)
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=lmin; l<Ntable.LMAX; l++) {
        Pl[i][l] = (2.*l+1)/(4.*M_PI*l*(l+1)*(xmin[i]-xmax[i]))
          *((l+2./(2*l+1.))*(Pmin[i][l-1]-Pmax[i][l-1])
          +(2-l)*(xmin[i]*Pmin[i][l]-xmax[i]*Pmax[i][l])
          -2./(2*l+1.)*(Pmin[i][l+1]-Pmax[i][l+1]));
      }
    }
    free(P);

    if (cmbf != NULL) free(cmbf);
    cmbf = (double*) malloc1d(Ntable.LMAX);

    if (ln_ell != NULL) free(ln_ell);
    ln_ell = (double*) malloc1d(Ntable.LMAX);
    ln_ell[0] = 0.0; // unused (the sums start at lmin = 1)
    for (int l=1; l<Ntable.LMAX; l++) {
      ln_ell[l] = log((double) l);
    }
    // per-call work arrays whose sizes only change with Ntable or the
    // tomography: allocated once here, overwritten on every call
    if (dCgrid != NULL) free(dCgrid);
    dCgrid = (double**) malloc2d(NSIZE, Ntable.N_ell);
    zero2d(dCgrid, NSIZE, Ntable.N_ell);
    if (cx != NULL) free(cx);
    cx = (double**) malloc2d(NSIZE, Ntable.LMAX);
    zero2d(cx, NSIZE, Ntable.LMAX);
    NSIZE_alloc = NSIZE;
    cache[4] = Ntable.random;
  }
  // CMB filter, exactly as w_ks_tomo builds it (cache[6] tracks Ntable
  // separately from the geometry block above, so a reallocation always
  // refills the filter)
  if (fdiff2(cache[5], cmb.random) || fdiff2(cache[6], Ntable.random)) {
    #pragma omp parallel for schedule(static)
    for (int l=0; l<Ntable.LMAX; l++) {
      double f = beam_cmb(l);
      if (cmb.healpixwin_ncls > 0) {
        f *= w_pixel(l);
      }
      cmbf[l] = f;
    }
    cache[5] = cmb.random;
    cache[6] = Ntable.random;
  }
  double* ans = (double*) calloc1d(NSIZE*Ntable.Ntheta);
  const double lnk = log(k);
  if (lnk > log(Ntable.dCX_dlnk_kmin) && lnk < log(Ntable.dCX_dlnk_kmax)) {
    // build (or reuse) the cached dC table single-threaded before the
    // parallel loops below read it
    (void) dC_ks_dlnk_tomo_limber(k, (double) limits.LMIN_tab, 0);
    // dC_ks/dlnk at this k on the dC table's own multipole log-grid
    const int nell = Ntable.N_ell;
    const double la = 0.0; // ln(l = 1): the dC table's multipole grid start
    const double ldx = (log(Ntable.LMAX + 1.) - la)/((double) nell - 1.);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<nell; i++) {
        const double lg = exp(la + i*ldx);
        dCgrid[nz][i] = dC_ks_dlnk_tomo_limber(k, lg, nz);
      }
    }
    // gather onto every integer multipole (vectorized linear interpolation)
    #pragma omp parallel for schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      const double* tab1[1] = {dCgrid[nz]};
      double* out1[1] = {cx[nz]};
      limber_fill_interp(1, tab1, out1, lmin, Ntable.LMAX, ln_ell,
                         la, 1.0/ldx, nell);
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        // Local restrict pointers: without these, GCC cannot prove the
        // Pl, cmbf and cx rows don't alias (pointer-to-pointer
        // indirection inside a collapse(2) OpenMP region) and gives up
        // on the SIMD reduction below
        const double* restrict c0 = cx[nz];
        const double* restrict cf = cmbf;
        const double* restrict g0 = Pl[i];
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int l=lmin; l<Ntable.LMAX; l++) {
          sum += g0[l] * cf[l] * c0[l];
        }
        ans[nz*Ntable.Ntheta + i] = sum;
      }
    }
    // warm w_ks_tomo's cached table single-threaded (one call builds
    // every bin); the parallel loop below then only reads it
    (void) w_ks_tomo(0, 0, 1);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        const int q = nz * Ntable.Ntheta + i;
        const double dwksdlnk = ans[q];
        if (fabs(ans[q]) > 1.e-50) {
          const double wks = w_ks_tomo(i, nz, 1);
          ans[q] = (fabs(wks) > 1.e-50) ? dwksdlnk/wks : 0.0;
        }
      }
    }
  }
  return ans;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Cached dlnw_ks/dlnk: interpolates in ln k a table of
// dlnw_ks_dlnk_tomo_nointerp results on the Ntable.dCX_dlnk k grid,
// rebuilt when the cosmology, the shear nuisances, the CMB filter or
// Ntable change (cmb.random is a key here — unlike the Fourier-space dC
// tables — because the CMB beam enters the w_ks projection); a k outside
// the grid returns 0. This is what the RF_w_ks integrals read.
// ---------------------------------------------------------------------------
double dlnw_ks_dlnk_tomo(
    const double k,   // wavenumber in (Mpc/h)^-1
    const int nt,     // angular bin index (0..Ntheta-1)
    const int ni      // source redshift bin
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double lim[3];
  static int nlnk;
  const int NSIZE = redshift.shear_nbin;
  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized"); exit(1);
  }
  if (NULL == table || fdiff2(cache[4], Ntable.random)) {
    nlnk = Ntable.dCX_dlnk_nlnk;
    lim[0] = log(Ntable.dCX_dlnk_kmin);
    lim[1] = log(Ntable.dCX_dlnk_kmax);
    lim[2] = (lim[1] - lim[0]) / ((double) nlnk - 1.);
    if (table != NULL) free(table);
    table = (double**) malloc2d(NSIZE*Ntable.Ntheta, nlnk);
  }
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random) ||
      fdiff2(cache[5], cmb.random))
  {
    for (int f=0; f<nlnk; f++) {
      double* tmp = dlnw_ks_dlnk_tomo_nointerp(exp(lim[0] + f*lim[2]));
      for (int nz=0; nz<NSIZE; nz++) {
        for (int i=0; i<Ntable.Ntheta; i++) {
          const int q = nz * Ntable.Ntheta + i;
          table[q][f] = tmp[q];
        }
      }
      free(tmp);
    }
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear;
    cache[4] = Ntable.random;
    cache[5] = cmb.random;
  }
  if (nt < 0 || nt > Ntable.Ntheta - 1) {
    log_fatal("error in selecting bin number nt = %d", nt); exit(1);
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  const int q = ni*Ntable.Ntheta + nt;
  if (q < 0 || q > NSIZE*Ntable.Ntheta - 1) {
    log_fatal("internal logic error in selecting bin number"); exit(1);
  }
  const double lnk = log(k);
  return (lnk < lim[0] || lnk > lim[1]) ? 0.0 :
         interpol1d(table[q], nlnk, lim[0], lim[1], lim[2], lnk);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// physical mode cut-off (R = response function): REAL SPACE
// find kk such that RF \equiv \int_{-\infty}^{ln(kk)} dlnk |dlnXdlnk| = alpha
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Batch computation of the real-space response function (2011.06469 eq 17)
//
//   RF(kmax, theta) = int_{-infty}^{ln kmax} dlnk |dlnxi_pm/dlnk|
//                     / int_{-infty}^{+infty} dlnk |dlnxi_pm/dlnk|
//
// on a ln kmax grid, for every tomographic pair and angular bin. Both
// integrals map onto t in (0, 1] — the numerator through
// lnk = ln kmax - (1-t)/t, the denominator through lnk = +-(1-t)/t — and
// use the same fixed Gauss-Legendre rule the retired scalar version used,
// with the nodes and weights precomputed into plain arrays instead of
// driving a GSL integrand callback per point.
//
// The denominator does not depend on kmax, so one thread team first fills
// one denominator value per (tomo pair, angular bin) and then, after the
// loop's implicit barrier, accumulates the numerator for every
// (pair, kmax, angular bin) output and divides. Every integrand
// evaluation reads the k-cached dlnxi_dlnk_pm_tomo table, built once,
// single-threaded, before the parallel region.
// ---------------------------------------------------------------------------
void RF_xi_tomo_limber_work(
    const double* lnkmaxx, // ln kmax values (length nkmax), k in (Mpc/h)^-1
    const int nkmax,       // number of ln kmax values
    const int NSIZE,       // number of tomo shear power spectra
    double**** table       // output [2][NSIZE][nkmax][Ntheta]: xi+ and xi-
  )
{
  if (nkmax <= 0) {
    log_fatal("nkmax = %d must be positive", nkmax);
    exit(1);
  }
  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized"); exit(1);
  }
  // Gauss-Legendre nodes and weights on t in [1e-5, 1] as plain arrays
  const int hdi = abs(Ntable.high_def_integration);
  const size_t szint = (0 == hdi) ? 256 :
                       (1 == hdi) ? 512 : 1024; // predefined GSL tables
  gsl_integration_glfixed_table* w = malloc_gslint_glfixed(szint);
  const int npts = (int) w->n;
  double* tq = (double*) malloc1d(npts);
  double* wq = (double*) malloc1d(npts);
  for (int p = 0; p < npts; p++) {
    gsl_integration_glfixed_point(1e-5, 1.0, p, &tq[p], &wq[p], w);
  }
  gsl_integration_glfixed_table_free(w);
  // the denominator's k nodes depend on nothing: precompute them once
  double* kd1 = (double*) malloc1d(npts);
  double* kd2 = (double*) malloc1d(npts);
  for (int p = 0; p < npts; p++) {
    kd1[p] = exp((1. - tq[p])/tq[p]);
    kd2[p] = exp(-(1. - tq[p])/tq[p]);
  }
  double*** den = (double***) malloc3d(2, NSIZE, Ntable.Ntheta);
  // build the k-cached dlnxi table single-threaded before the parallel
  // region below reads it
  (void) dlnxi_dlnk_pm_tomo(1.0, 1, 0, Z1(0), Z2(0));
  #pragma omp parallel
  {
  #pragma omp for collapse(2) schedule(static)
  for (int nz = 0; nz < NSIZE; nz++) {
    for (int nt = 0; nt < Ntable.Ntheta; nt++) {
      const int Z1NZ = Z1(nz);
      const int Z2NZ = Z2(nz);
      double sXP = 0.0;
      double sXM = 0.0;
      for (int p = 0; p < npts; p++) {
        const double wt = wq[p]/(tq[p]*tq[p]);
        sXP += (fabs(dlnxi_dlnk_pm_tomo(kd1[p], 1, nt, Z1NZ, Z2NZ)) +
                fabs(dlnxi_dlnk_pm_tomo(kd2[p], 1, nt, Z1NZ, Z2NZ)))*wt;
        sXM += (fabs(dlnxi_dlnk_pm_tomo(kd1[p], 0, nt, Z1NZ, Z2NZ)) +
                fabs(dlnxi_dlnk_pm_tomo(kd2[p], 0, nt, Z1NZ, Z2NZ)))*wt;
      }
      den[0][nz][nt] = sXP;
      den[1][nz][nt] = sXM;
    }
  } // implicit barrier: denominators complete before the division below
  #pragma omp for collapse(3) schedule(static)
  for (int nz = 0; nz < NSIZE; nz++) {
    for (int m = 0; m < nkmax; m++) {
      for (int nt = 0; nt < Ntable.Ntheta; nt++) {
        const int Z1NZ = Z1(nz);
        const int Z2NZ = Z2(nz);
        double sXP = 0.0;
        double sXM = 0.0;
        for (int p = 0; p < npts; p++) {
          const double k = exp(lnkmaxx[m] - (1. - tq[p])/tq[p]);
          const double wt = wq[p]/(tq[p]*tq[p]);
          sXP += fabs(dlnxi_dlnk_pm_tomo(k, 1, nt, Z1NZ, Z2NZ))*wt;
          sXM += fabs(dlnxi_dlnk_pm_tomo(k, 0, nt, Z1NZ, Z2NZ))*wt;
        }
        table[0][nz][m][nt] = sXP/den[0][nz][nt];
        table[1][nz][m][nt] = sXM/den[1][nz][nt];
      }
    }
  }
  } // end of the parallel region
  free(den);
  free(kd1);
  free(kd2);
  free(tq);
  free(wq);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Batch computation of the real-space response function (2011.06469 eq 17)
//
//   RF(kmax, theta) = int_{-infty}^{ln kmax} dlnk |dlnw_ks/dlnk|
//                     / int_{-infty}^{+infty} dlnk |dlnw_ks/dlnk|
//
// on a ln kmax grid, for every source bin and angular bin. Same design as
// RF_xi_tomo_limber_work (the ks cross has one component per source bin,
// so there is no xi+/xi- leading dimension): both integrals map onto t in
// (0, 1] — the numerator through lnk = ln kmax - (1-t)/t, the denominator
// through lnk = +-(1-t)/t — with the fixed Gauss-Legendre nodes and
// weights precomputed into plain arrays.
//
// The denominator does not depend on kmax, so one thread team first fills
// one denominator value per (source bin, angular bin) and then, after the
// loop's implicit barrier, accumulates the numerator for every
// (bin, kmax, angular bin) output and divides. Every integrand evaluation
// reads the k-cached dlnw_ks_dlnk_tomo table, built once, single-threaded,
// before the parallel region.
// ---------------------------------------------------------------------------
void RF_w_ks_tomo_limber_work(
    const double* lnkmaxx, // ln kmax values (length nkmax), k in (Mpc/h)^-1
    const int nkmax,       // number of ln kmax values
    const int NSIZE,       // number of source tomographic bins (= shear_nbin)
    double*** table        // output [NSIZE][nkmax][Ntheta]
  )
{
  if (nkmax <= 0) {
    log_fatal("nkmax = %d must be positive", nkmax);
    exit(1);
  }
  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized"); exit(1);
  }
  // Gauss-Legendre nodes and weights on t in [1e-5, 1] as plain arrays
  const int hdi = abs(Ntable.high_def_integration);
  const size_t szint = (0 == hdi) ? 256 :
                       (1 == hdi) ? 512 : 1024; // predefined GSL tables
  gsl_integration_glfixed_table* w = malloc_gslint_glfixed(szint);
  const int npts = (int) w->n;
  double* tq = (double*) malloc1d(npts);
  double* wq = (double*) malloc1d(npts);
  for (int p = 0; p < npts; p++) {
    gsl_integration_glfixed_point(1e-5, 1.0, p, &tq[p], &wq[p], w);
  }
  gsl_integration_glfixed_table_free(w);
  // the denominator's k nodes depend on nothing: precompute them once
  double* kd1 = (double*) malloc1d(npts);
  double* kd2 = (double*) malloc1d(npts);
  for (int p = 0; p < npts; p++) {
    kd1[p] = exp((1. - tq[p])/tq[p]);
    kd2[p] = exp(-(1. - tq[p])/tq[p]);
  }
  double** den = (double**) malloc2d(NSIZE, Ntable.Ntheta);
  // build the k-cached dlnw_ks table single-threaded before the parallel
  // region below reads it
  (void) dlnw_ks_dlnk_tomo(1.0, 0, 0);
  #pragma omp parallel
  {
  #pragma omp for collapse(2) schedule(static)
  for (int nz = 0; nz < NSIZE; nz++) {
    for (int nt = 0; nt < Ntable.Ntheta; nt++) {
      double sKS = 0.0;
      for (int p = 0; p < npts; p++) {
        const double wt = wq[p]/(tq[p]*tq[p]);
        sKS += (fabs(dlnw_ks_dlnk_tomo(kd1[p], nt, nz)) +
                fabs(dlnw_ks_dlnk_tomo(kd2[p], nt, nz)))*wt;
      }
      den[nz][nt] = sKS;
    }
  } // implicit barrier: denominators complete before the division below
  #pragma omp for collapse(3) schedule(static)
  for (int nz = 0; nz < NSIZE; nz++) {
    for (int m = 0; m < nkmax; m++) {
      for (int nt = 0; nt < Ntable.Ntheta; nt++) {
        double sKS = 0.0;
        for (int p = 0; p < npts; p++) {
          const double k = exp(lnkmaxx[m] - (1. - tq[p])/tq[p]);
          const double wt = wq[p]/(tq[p]*tq[p]);
          sKS += fabs(dlnw_ks_dlnk_tomo(k, nt, nz))*wt;
        }
        table[nz][m][nt] = sKS/den[nz][nt];
      }
    }
  }
  } // end of the parallel region
  free(den);
  free(kd1);
  free(kd2);
  free(tq);
  free(wq);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
