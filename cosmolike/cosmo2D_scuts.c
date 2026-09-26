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

double dC_ss_dlnk_tomo_limber(
    const double k,
    const double l, 
    const int ni, 
    const int nj, 
    const int EE
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

double dlnC_ss_dlnk_tomo_limber(
    const double k,
    const double l, 
    const int ni, 
    const int nj, 
    const int EE
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
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// DERIVATIVE: dlnX/dlnk: REAL SPACE
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double** dlnxi_dlnk_pm_tomo_nointerp(const double k)
{  
  static double*** Glpm = NULL; //Glpm[0] = Gl+, Glpm[1] = Gl-
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double**** dCldlnk = NULL;
  static double lim[3];
  static int nlnk;
  const int lmin = 1;
  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized"); exit(1);
  }
  const int NSIZE = tomo.shear_Npowerspectra;
  if (NULL == Glpm ||  NULL == dCldlnk || fdiff2(cache[4], Ntable.random))
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
    #pragma omp parallel for collapse(2) schedule(static,1)
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
    #pragma omp parallel for collapse(2) schedule(static,1)
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
    
    nlnk = Ntable.dCX_dlnk_nlnk;
    lim[0] = log(Ntable.dCX_dlnk_kmin);
    lim[1] = log(Ntable.dCX_dlnk_kmax);
    lim[2] = (lim[1] - lim[0]) / ((double) nlnk - 1.);
    if (dCldlnk != NULL) free(dCldlnk);
    dCldlnk = (double****) malloc4d(2, NSIZE, Ntable.LMAX, nlnk);
  }
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    for (int q=0; q<nlnk; q++) {
      for (int i=0; i<NSIZE; i++) {
        for (int l=0; l<lmin; l++) {
          dCldlnk[0][i][l][q] = 0.0;
          dCldlnk[1][i][l][q] = 0.0;
        }
      }
    }
    // init static vars (also builds the cached dC table before the loop)
    (void) dC_ss_dlnk_tomo_limber(Ntable.dCX_dlnk_kmin, (double) limits.LMIN_tab,
                                  Z1(0), Z2(0), 1);
    #pragma omp parallel for collapse(4) schedule(static,1)
    for (int p=0; p<2; p++) {
      for (int q=0; q<nlnk; q++)  {
        for (int nz=0; nz<NSIZE; nz++) {
          for (int l=lmin; l<Ntable.LMAX; l++) {
            const double kin = exp(lim[0] + q * lim[2]);
            dCldlnk[p][nz][l][q] =
                   dC_ss_dlnk_tomo_limber(kin, (double) l, Z1(nz), Z2(nz), 1-p);
          }
        }
      }
    }
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear;
    cache[4] = Ntable.random;
  }
  double** ans = (double**) malloc2d(2, NSIZE*Ntable.Ntheta);
  #pragma omp parallel for collapse(3) schedule(static,1)
  for (int p=0; p<2; p++) {
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        const int q = nz * Ntable.Ntheta + i;
        ans[p][q] = 0.0;
      }
    }
  }
  const double lnk = log(k);
  if (lnk > lim[0] && lnk < lim[1]) {
    double*** cx = (double***) malloc3d(2, NSIZE, Ntable.LMAX);
    for (int nz=0; nz<NSIZE; nz++) {
      for (int l=0; l<lmin; l++) {
        cx[0][nz][l] = 0.0;
        cx[1][nz][l] = 0.0;
      }
    }
    #pragma omp parallel for collapse(3) schedule(static,1)
    for (int p=0; p<2; p++) {
      for (int nz=0; nz<NSIZE; nz++) {
        for (int l=lmin; l<Ntable.LMAX; l++) {
          cx[p][nz][l] = interpol1d(dCldlnk[p][nz][l], nlnk, 
                                    lim[0], lim[1], lim[2], lnk);
        } 
      } 
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        const int q = nz * Ntable.Ntheta + i;
        double sum0 = 0.0;
        double sum1 = 0.0;   
        for (int l=lmin; l<Ntable.LMAX; l++) {
          const double c0 = cx[0][nz][l];
          const double c1 = cx[1][nz][l];
          sum0 += Glpm[0][i][l] * (c0 + c1); 
          sum1 += Glpm[1][i][l] * (c0 - c1);
        }
        ans[0][q] = sum0;
        ans[1][q] = sum1;
      } 
    }
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
    free(cx);
  }
  return ans;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double dlnxi_dlnk_pm_tomo(
    const double k,
    const int pm, 
    const int nt, 
    const int ni, 
    const int nj
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

double int_RF_xi(double t, void* params) 
{ // \int_{-infty}^{b} dlnk f(lnk) = \int_0^1 f(b-(1-t)/t)/t^2 
  double* ar = (double*) params;
  const int nt = (int) ar[0];
  if (nt < 0 || nt > Ntable.Ntheta - 1) {
    log_fatal("error in selecting bin number nt = %d", nt); exit(1); 
  }
  const int ni = (int) ar[1];
  const int nj = (int) ar[2];
  if (ni < 0 || ni > redshift.shear_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (ni,nj) = [%d,%d]",ni,nj); exit(1);
  }
  const int pm = (int) ar[3];
  const double lnkmax = ar[4];
  const double k = exp(lnkmax - (1. - t)/t);
  const double f1 = dlnxi_dlnk_pm_tomo(k, pm, nt, ni, nj);
  return fabs(f1)/(t*t);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_norm_RF_xi(double t, void* params) 
{ // \int_{-infty}^{infty} dlnk f(lnk) = \int_0^1 (f((1-t)/t)+f(-(1-t)/t))/t^2 
  double* ar = (double*) params;
  const int nt = (int) ar[0];
  if (nt < 0 || nt > Ntable.Ntheta - 1) {
    log_fatal("error in selecting bin number nt = %d", nt); exit(1); 
  }
  const int ni = (int) ar[1];
  const int nj = (int) ar[2];
  if (ni < 0 || ni > redshift.shear_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (ni,nj) = [%d,%d]",ni,nj); exit(1);
  }
  const int pm = (int) ar[3];
  const double lnkmax = ar[4];
  const double k1 = exp((1.-t)/t);
  const double f1 = dlnxi_dlnk_pm_tomo(k1, pm, nt, ni, nj);
  const double k2 = exp(-(1.-t)/t);
  const double f2 = dlnxi_dlnk_pm_tomo(k2, pm, nt, ni, nj);
  return (fabs(f1) + fabs(f2))/(t*t);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double RF_xi_tomo_limber_nointerp(
    const double kmax,
    const int pm, 
    const int nt, 
    const int ni, 
    const int nj, 
    const int init
  ) // compute RF_X = \int_{-infty}^{kmax} dlnk |dlnX_dlnk|
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL; 
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 256 : 
                         (1 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }
  double ar[5] = {(double) nt, (double) ni, 
                  (double) nj, (double) pm, log(kmax)};
  double res = 0.0;
  if (1 == init) {
    (void) int_RF_xi(1e-1, (void*) ar);
    (void) int_norm_RF_xi(1e-1, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_RF_xi;
    const double num = gsl_integration_glfixed(&F, 1e-5, 1.0, w);
    F.function = int_norm_RF_xi;
    const double den = gsl_integration_glfixed(&F, 1e-5, 1.0, w);
    res = num/den;
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
