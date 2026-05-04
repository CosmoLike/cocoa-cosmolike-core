#include <assert.h>
#include <complex.h>
#include <fftw3.h>
#include <gsl/gsl_sum.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_sf.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "bias.h"
#include "basics.h"
#include "cfastpt/cfastpt.h"
#include "cosmo3D.h"
#include "cosmo2D.h"
#include "halo.h"
#include "IA.h"
#include "pt_cfastpt.h"
#include "radial_weights.h"
#include "redshift_spline.h"
#include "structs.h"
#include "log.c/src/log.h"

//#define COSMO2D_NOT_USE_SIMD

#ifndef COSMO2D_NOT_USE_SIMD
#include "simde/x86/avx2.h"
#include "simde/x86/fma.h"
#endif

static int include_HOD_GX = 0; // 0 or 1
static int include_RSD_GS = 0; // 0 or 1 
static int include_RSD_GG = 1; // 0 or 1 
static int include_RSD_GK = 0; // 0 or 1
static int include_RSD_GY = 0; // 0 or 1

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// SIMD FUNCTIONS
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
#ifndef COSMO2D_NOT_USE_SIMD
// -----------------------------------------------------------------------------
// What is SIMD? How is the basic building block of SIMD?
// A normal double variable holds 1 number (64 bits).
// A simde__m256d holds 4 doubles side-by-side (256 bits = 4 x 64).
//
// Think of it as a box with 4 slots ("lanes"):
//
//   simde__m256d box = [ slot0 | slot1 | slot2 | slot3 ]
//                        64 bit  64 bit  64 bit  64 bit
//                      <----------- 256 bits ---------->
//
// When you add two such boxes, all 4 slots are added in parallel:
//
//   box_a = [ 1.0 | 2.0 | 3.0 | 4.0 ]
//   box_b = [ 5.0 | 6.0 | 7.0 | 8.0 ]
//   result = [ 6.0 | 8.0 | 10.0 | 12.0 ]   (one instruction!)
// -----------------------------------------------------------------------------

static inline double simd_horizontal_sum(simde__m256d four_lanes)
{ // Takes a 4-lane register and sums all 4 values into a single double
  double tmp[4]; // Store the 4 lanes into a regular C array
  simde_mm256_storeu_pd(tmp, four_lanes);
  return tmp[0] + tmp[1] + tmp[2] + tmp[3];
}

// -----------------------------------------------------------------------------
// simd_dot_product
// -----------------------------------------------------------------------------
//
// Computes:   result = a[0]*b[0] + a[1]*b[1] + ... + a[n-1]*b[n-1]
//
//   We use two 4-lane accumulator registers, A and B:
//
//     accum_A = [0, 0, 0, 0]      accum_B = [0, 0, 0, 0]
//
//   Iteration 1 processing 8 elements:
//     Load a[0..3] and b[0..3] into 4-lane registers.
//     accum_A += a[0..3] * b[0..3]    (fused multiply-add)
//
//     Load a[4..7] and b[4..7] into 4-lane registers.
//     accum_B += a[4..7] * b[4..7]    (into the SECOND accumulator)
//
//   No more groups of 8 left, so go to scalar tail
// -----------------------------------------------------------------------------
static inline double simd_dot_product(
    const double* restrict a,
    const double* restrict b,
    const int n
  )
{
  // Two independent accumulators, each holding 4 partial sums, init to [0,0,0,0]
  simde__m256d accum_A = simde_mm256_setzero_pd();
  simde__m256d accum_B = simde_mm256_setzero_pd();
 
  int l = 0; // l must survive past the loop (for scalar tail)

  for (; l<= n-8; l+=8) { // Main loop: process 8 doubles per iteration
    // Load 4 consecutive doubles from a[] and b[], multiply and acc. into accum_A:
    accum_A = simde_mm256_fmadd_pd(
        simde_mm256_loadu_pd(a + l),       // load a[l..l+3]
        simde_mm256_loadu_pd(b + l),       // load b[l..l+3]
        accum_A
      ); // add to running sum
    
    // Same for the next 4 doubles, into the independent accumulator accum_B:
    accum_B = simde_mm256_fmadd_pd(
        simde_mm256_loadu_pd(a + l + 4),   // load a[l+4..l+7]
        simde_mm256_loadu_pd(b + l + 4),   // load b[l+4..l+7]
        accum_B
      ); // add to running sum
  }

  // Reduce partial sums to a scalar
  double result = simd_horizontal_sum(accum_A) + simd_horizontal_sum(accum_B);
  
  // Scalar tail: remaining 0-7 elements one at a time
  for (; l < n; l++) { 
    result += a[l] * b[l];
  }
  return result;
}

// -----------------------------------------------------------------------------
// simd_xipm_dot_product
// -----------------------------------------------------------------------------
// Computes 2 dot products at the same time: the xi+/xi- correlation functions
//
//   xi_plus  = sum_l  Gplus[l] * (Cl_EE[l] + Cl_BB[l])
//   xi_minus = sum_l  Gminus[l] * (Cl_EE[l] - Cl_BB[l])
//
//   We maintain 4 accumulator registers:
//     plus_accum_A  = [0, 0, 0, 0]    \  two for xi+ (latency hiding)
//     plus_accum_B  = [0, 0, 0, 0]    /
//     minus_accum_A = [0, 0, 0, 0]    \  two for xi- (latency hiding)
//     minus_accum_B = [0, 0, 0, 0]    /
// -----------------------------------------------------------------------------
static inline void simd_xipm_dot_product(
    const double* restrict cl_ee,    // EE power spectrum (offset to lmin)
    const double* restrict cl_bb,    // BB power spectrum (offset to lmin)
    const double* restrict gl_plus,  // G_l^+ filter (offset to lmin)
    const double* restrict gl_minus, // G_l^- filter (offset to lmin)
    const int n,                     // number of multipoles = LMAX - lmin
    double* restrict out_xi_plus,    // output: xi+
    double* restrict out_xi_minus    // output: xi-
  )     
{
  // Two accumulators: A and B for xi+, A and B for xi-.
  simde__m256d plus_accum_A  = simde_mm256_setzero_pd();  // [0, 0, 0, 0]
  simde__m256d plus_accum_B  = simde_mm256_setzero_pd();
  
  // Two accumulators: A and B for xi-.
  simde__m256d minus_accum_A = simde_mm256_setzero_pd();
  simde__m256d minus_accum_B = simde_mm256_setzero_pd();
 
  int l = 0;
 
  for (; l <= n - 8; l += 8) { // Main loop: 8 multipoles per iteration
    // Load 4 values from each input array into 4-lane registers.
    simde__m256d ee_lanes = simde_mm256_loadu_pd(cl_ee + l);
    simde__m256d bb_lanes = simde_mm256_loadu_pd(cl_bb + l);
    
    simde__m256d gp_lanes = simde_mm256_loadu_pd(gl_plus + l);
    simde__m256d gm_lanes = simde_mm256_loadu_pd(gl_minus + l);
 
    
    simde__m256d ee_plus_bb  = simde_mm256_add_pd(ee_lanes, bb_lanes);
    simde__m256d ee_minus_bb = simde_mm256_sub_pd(ee_lanes, bb_lanes);
    
    // Accumulate:  plus_accum_A[i] += G+[l+i] * (EE[l+i] + BB[l+i])
    plus_accum_A = simde_mm256_fmadd_pd(gp_lanes, ee_plus_bb, plus_accum_A);
    
    // Accumulate:  minus_accum_A[i] += G-[l+i] * (EE[l+i] - BB[l+i])
    minus_accum_A = simde_mm256_fmadd_pd(gm_lanes, ee_minus_bb, minus_accum_A);
 
    // Same operations, but accumulating into the B registers. 
    ee_lanes = simde_mm256_loadu_pd(cl_ee + l + 4);
    bb_lanes = simde_mm256_loadu_pd(cl_bb + l + 4);
    
    gp_lanes = simde_mm256_loadu_pd(gl_plus + l + 4);
    gm_lanes = simde_mm256_loadu_pd(gl_minus + l + 4);
 
    ee_plus_bb  = simde_mm256_add_pd(ee_lanes, bb_lanes);
    ee_minus_bb = simde_mm256_sub_pd(ee_lanes, bb_lanes);

    // Accumulate:  plus_accum_B[i] += G+[l+i] * (EE[l+i] + BB[l+i])
    plus_accum_B  = simde_mm256_fmadd_pd(gp_lanes, ee_plus_bb, plus_accum_B);

    // Accumulate:  minus_accum_B[i] += G-[l+i] * (EE[l+i] - BB[l+i])
    minus_accum_B = simde_mm256_fmadd_pd(gm_lanes, ee_minus_bb, minus_accum_B);
  }
 
  // Reduce each 4-lane accumulator to a single scalar
  double xip = simd_horizontal_sum(plus_accum_A) + simd_horizontal_sum(plus_accum_B);
  double xim = simd_horizontal_sum(minus_accum_A) + simd_horizontal_sum(minus_accum_B);
 
  for (; l < n; l++) { // Scalar tail: remaining 0-7 multipoles, one at a time
    xip += gl_plus[l]  * (cl_ee[l] + cl_bb[l]);
    xim += gl_minus[l] * (cl_ee[l] - cl_bb[l]);
  }
  *out_xi_plus = xip;
  *out_xi_minus = xim;
}

// -----------------------------------------------------------------------------
// simd_array_sum
// -----------------------------------------------------------------------------
static inline double simd_array_sum(const double* restrict a, const int n)
{ // Computes:   result = a[0] + a[1] + ... + a[n-1]
  
  // Two independent accumulators, each holding 4 partial sums, init to [0,0,0,0]
  simde__m256d accum_A = simde_mm256_setzero_pd();
  simde__m256d accum_B = simde_mm256_setzero_pd();
 
  int q = 0;
 
  for (; q <= n - 8; q += 8) { // Main loop: process 8 doubles per iteration
    accum_A = simde_mm256_add_pd(accum_A, simde_mm256_loadu_pd(a + q));
    accum_B = simde_mm256_add_pd(accum_B, simde_mm256_loadu_pd(a + q + 4));
  }
 
  double result = simd_horizontal_sum(accum_A) + simd_horizontal_sum(accum_B);
 
  for (; q < n; q++) { // Scalar tail: remaining 0-7 elements, one at a time
    result += a[q];
  }
  return result;
}
#endif
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// BASIC DEFINITIONS & DECLARATIONS
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

#ifdef __APPLE__
  #define cosmo_sincos(x, s, c) __sincos((x), (s), (c))
#else
  #define cosmo_sincos(x, s, c) sincos((x), (s), (c))
#endif

double beam_cmb(const int l) {
  const double s = cmb.fwhm/sqrt(16.0*log(2.0));
  return ((l<cmb.lmink_wxk) || (l>cmb.lmaxk_wxk)) ? 0.0 : exp(-l*(l+1.0)*s*s);
}

double w_pixel(const int l) {
  if (0 == cmb.healpixwin_ncls) {
    log_fatal("cmb.healpixwin_ncls not initialized");
    exit(1);
  }
  return (l < cmb.healpixwin_ncls) ? cmb.healpixwin[l] : 0.0;
}

static int has_b2_galaxies(void) {
  int res = 0;
  for (int i=0; i<redshift.clustering_nbin; i++) 
    if (nuisance.gb[1][i])
      res = 1;
  return res;
}

static inline double wtime(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

// -------------------------------------------------------------------------
// optimization: real 2pt computes C_xy_tomo_limber so many times that the  
//               overhead to calls to logl/N_shear/interpol1d is quite expensive
// -------------------------------------------------------------------------
void C_ss_tomo_limber_fill(
    const int nz, 
    const int lmin, 
    const int lmax,
    const double* RESTRICT ln_ell,
    double* RESTRICT out_EE,
    double* RESTRICT out_BB
  );

// -------------------------------------------------------------------------
// optimization: real 2pt computes C_xy_tomo_limber so many times that the  
//               overhead to calls to logl/N_shear/interpol1d is quite expensive
// -------------------------------------------------------------------------
void C_gs_tomo_limber_fill(
    const int nz,
    const int lmin,
    const int lmax,
    const double* RESTRICT ln_ell,
    double* RESTRICT out
  );

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Correlation Functions (real Space) - Full Sky - bin average
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

double xi_pm_tomo(
    const int pm, 
    const int nt, 
    const int ni, 
    const int nj, 
    const int limber
  )
{  
  static double*** Glpm = NULL; //Glpm[0] = Gl+, Glpm[1] = Gl-
  static double** xipm = NULL;  //xipm[0] = xi+, xipm[1] = xi-
  static double*** Cl = NULL;
  static double* lnell = NULL;
  static uint64_t cache[MAX_SIZE_ARRAYS];

  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized"); exit(1);
  }

  const int NSIZE = tomo.shear_Npowerspectra;

  if (NULL == Glpm || 
      NULL == xipm || 
      NULL == Cl || 
      fdiff2(cache[4], Ntable.random))
  {
    if (lnell != NULL) {
      free(lnell);
    }
    lnell = (double*) malloc1d(Ntable.LMAX + 1);
    for (int l =1; l <= Ntable.LMAX; l++) {
      lnell[l] = log((double) l);
    }

    if (Glpm != NULL) {
      free(Glpm);
    }
    Glpm = (double***) malloc3d(2, Ntable.Ntheta, Ntable.LMAX);
    if (xipm != NULL) {
      free(xipm);
    }
    xipm = (double**) malloc2d(2, NSIZE*Ntable.Ntheta);
    
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

    const int lmin = 1;
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

    if (Cl != NULL) {
      free(Cl);
    }
    Cl = (double***) malloc3d(2, NSIZE, Ntable.LMAX); // Cl_EE=Cl[0], Cl_BB=Cl[1]
  }

  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    const int lmin = 1;
    for (int i=0; i<NSIZE; i++) {
      for (int l=0; l<lmin; l++) {
        Cl[0][i][l] = 0.0;
        Cl[1][i][l] = 0.0;
      }
    }
    if (1 == limber) {
      // init static vars
      (void) C_ss_tomo_limber((double) limits.LMIN_tab + 1, Z1(0), Z2(0), 1);
      #pragma omp parallel for collapse(2) schedule(static,1)
      for (int nz=0; nz<NSIZE; nz++)  {
        for (int l=lmin; l<limits.LMIN_tab; l++) {
          const int Z1NZ = Z1(nz);
          const int Z2NZ = Z2(nz);
          Cl[0][nz][l] = C_ss_tomo_limber_nointerp((double) l, Z1NZ, Z2NZ, 1, 0);
          Cl[1][nz][l] = C_ss_tomo_limber_nointerp((double) l, Z1NZ, Z2NZ, 0, 0);
        }
      }
      #pragma omp parallel for schedule(static)
      for (int nz = 0; nz < NSIZE; nz++) {
        C_ss_tomo_limber_fill(nz,
                              limits.LMIN_tab, 
                              Ntable.LMAX, 
                              lnell,
                              Cl[0][nz], 
                              Cl[1][nz]);
      }
    }
    else {
      log_fatal("NonLimber not implemented"); exit(1);
    }
#ifdef COSMO2D_NOT_USE_SIMD
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        const int q = nz * Ntable.Ntheta + i;
        double sum0 = 0.0;
        double sum1 = 0.0;
        for (int l=lmin; l<Ntable.LMAX; l++) {
          const double c0 = Cl[0][nz][l];
          const double c1 = Cl[1][nz][l];
          sum0 += Glpm[0][i][l] * (c0 + c1);
          sum1 += Glpm[1][i][l] * (c0 - c1);
        }
        xipm[0][q] = sum0;
        xipm[1][q] = sum1;
      }
    }
#else
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        const int q = nz * Ntable.Ntheta + i;
        double sum0, sum1;
        simd_xipm_dot_product(
          Cl[0][nz]  + lmin,   // EE spectrum, starting at lmin
          Cl[1][nz]  + lmin,   // BB spectrum, starting at lmin
          Glpm[0][i] + lmin,   // G_l^+ filter
          Glpm[1][i] + lmin,   // G_l^- filter
          Ntable.LMAX - lmin,  // number of multipoles
          &sum0, &sum1);
        xipm[0][q] = sum0;
        xipm[1][q] = sum1;
      }
    }
#endif

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
  return (pm > 0) ? xipm[0][q] : xipm[1][q];
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double w_gammat_tomo(const int nt, const int ni, const int nj, const int limber)
{
  static double** Pl = NULL;
  static double* w_vec = NULL;
  static double** Cl = NULL;
  static double* lnell = NULL;
  static uint64_t cache[MAX_SIZE_ARRAYS];

  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized");
    exit(1);
  }

  const int NSIZE = tomo.ggl_Npowerspectra;

  if (NULL == Pl || 
      NULL == w_vec || 
      NULL == Cl || 
      fdiff2(cache[6], Ntable.random))
  {
    const int lmin = 1;

    if (lnell != NULL) {
      free(lnell);
    }
    lnell = (double*) malloc1d(Ntable.LMAX + 1);
    for (int l = 1; l <= Ntable.LMAX; l++) {
      lnell[l] = log((double) l);
    }

    if (Pl != NULL) {
      free(Pl);
    }
    Pl = (double**) malloc2d(Ntable.Ntheta, Ntable.LMAX);;
    
    if (w_vec != NULL) {
      free(w_vec);
    }
    w_vec = (double*) calloc1d(NSIZE*Ntable.Ntheta);

    double*** P = (double***) malloc3d(2, Ntable.Ntheta, Ntable.LMAX + 1);
    double** Pmin  = P[0]; double** Pmax  = P[1];

    double xmin[Ntable.Ntheta];
    double xmax[Ntable.Ntheta];
    for (int i=0; i<Ntable.Ntheta; i++)
    { // Cocoa: dont thread (init of static variables inside set_bin_average)
      bin_avg r = set_bin_average(i,0);
      xmin[i] = r.xmin;
      xmax[i] = r.xmax;
    }

    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.Ntheta; i ++) {
      for (int l=0; l<(Ntable.LMAX+1); l++) {
        bin_avg r = set_bin_average(i, l);
        Pmin[i][l] = r.Pmin;
        Pmax[i][l] = r.Pmax;
      }
    }
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<lmin; l++) {
        Pl[i][l] = 0.0;
      }
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=lmin; l<Ntable.LMAX; l++) {
        Pl[i][l] = (2.*l+1)/(4.*M_PI*l*(l+1)*(xmin[i]-xmax[i]))
          *((l+2./(2*l+1.))*(Pmin[i][l-1]-Pmax[i][l-1])
          +(2-l)*(xmin[i]*Pmin[i][l]-xmax[i]*Pmax[i][l])
          -2./(2*l+1.)*(Pmin[i][l+1]-Pmax[i][l+1]));
      }
    }

    free(P);
    if (Cl != NULL) free(Cl);
    Cl = (double**) malloc2d(NSIZE, Ntable.LMAX);
  }

  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_photoz_clustering) ||
      fdiff2(cache[3], nuisance.random_ia) ||
      fdiff2(cache[4], redshift.random_shear) ||
      fdiff2(cache[5], redshift.random_clustering) ||
      fdiff2(cache[6], Ntable.random) ||
      fdiff2(cache[7], nuisance.random_galaxy_bias))
  {
    const int lmin = 1;
    for (int i=0; i<NSIZE; i++) {
      for (int l=0; l<lmin; l++) {
        Cl[i][l] = 0.0;
      }
    }
    (void) C_gs_tomo_limber((double) limits.LMIN_tab + 1, ZL(0), ZS(0));
    
    if (1 == limber) {
      #pragma omp parallel for collapse(2) schedule(static,1)
      for (int nz=0; nz<NSIZE; nz++) {
        for (int l=lmin; l<limits.LMIN_tab; l++) {
          Cl[nz][l] = C_gs_tomo_limber_nointerp((double) l, ZL(nz), ZS(nz), 0);
        }
      }
      #pragma omp parallel for collapse(2) schedule(static,1)
      for (int nz=0; nz<NSIZE; nz++) {
        for (int l=limits.LMIN_tab; l<Ntable.LMAX; l++) {  
          Cl[nz][l] = C_gs_tomo_limber((double) l, ZL(nz), ZS(nz));
        }
      }
      #pragma omp parallel for schedule(static)
      for (int nz = 0; nz < NSIZE; nz++) {
        C_gs_tomo_limber_fill(nz, limits.LMIN_tab, Ntable.LMAX, lnell, Cl[nz]);
      }
    }
    else {
      log_fatal("NonLimber not implemented");
      exit(1);
    }

#ifdef COSMO2D_NOT_USE_SIMD
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        double sum = 0.0;
        for (int l=lmin; l<Ntable.LMAX; l++) {
          sum += Pl[i][l] * Cl[nz][l];
        }
        w_vec[nz * Ntable.Ntheta + i] = sum;
      }
    }
#else
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        w_vec[nz * Ntable.Ntheta + i] =
          simd_dot_product(Pl[i] + lmin, Cl[nz] + lmin, Ntable.LMAX - lmin);
      }
    }  
#endif

    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_photoz_clustering;
    cache[3] = nuisance.random_ia;
    cache[4] = redshift.random_shear;
    cache[5] = redshift.random_clustering;
    cache[6] = Ntable.random;
    cache[7] = nuisance.random_galaxy_bias;
  }
  // ---------------------------------------------------------------------------
  if (nt < 0 || nt > Ntable.Ntheta - 1) {
    log_fatal("error in selecting bin number nt = %d (max %d)", nt, Ntable.Ntheta);
    exit(1); 
  }
  if (ni < -1 || 
      ni > redshift.clustering_nbin - 1 || 
      nj < -1 || 
      nj > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (ni, nj) = [%d,%d]", ni, nj);
    exit(1);
  }
  if (test_zoverlap(ni,nj)) {
    const int q = N_ggl(ni,nj)*Ntable.Ntheta + nt;
    if (q < 0 || q > NSIZE*Ntable.Ntheta - 1) {
      log_fatal("internal logic error in selecting bin number");
      exit(1);
    }
    return w_vec[q];
  }
  else {
    return 0.0;
  }
}
//
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double w_gg_tomo(const int nt, const int ni, const int nj, const int limber)
{
  static double** Pl = NULL;
  static double* w_vec = NULL;
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** Cl = NULL; 
  static double* lnell = NULL;

  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized");
    exit(1);
  }

  const int NSIZE = tomo.clustering_Npowerspectra;

  if (NULL == Pl || 
      NULL == w_vec || 
      NULL == Cl || 
      fdiff2(cache[3], Ntable.random))
  {
    const int lmin = 1;

    if (lnell != NULL) {
      free(lnell);
    }
    lnell = (double*) malloc1d(Ntable.LMAX + 1);
    for (int l = 1; l <= Ntable.LMAX; l++) {
      lnell[l] = log((double) l);
    }

    if (Pl != NULL) {
      free(Pl);
    }
    Pl = (double**) malloc2d(Ntable.Ntheta, Ntable.LMAX);
    if (w_vec != NULL) {
      free(w_vec);
    }
    w_vec = (double*) calloc1d(NSIZE*Ntable.Ntheta);

    double*** P = (double***) malloc3d(2, Ntable.Ntheta, Ntable.LMAX + 1);
    double** Pmin  = P[0]; double** Pmax  = P[1];

    double xmin[Ntable.Ntheta];
    double xmax[Ntable.Ntheta];
    for (int i=0; i<Ntable.Ntheta; i ++)
    { // Cocoa: dont thread (init of static variables inside set_bin_average)
      bin_avg r = set_bin_average(i,0);
      xmin[i] = r.xmin;
      xmax[i] = r.xmax;
    }

    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<(Ntable.LMAX+1); l++) {
        bin_avg r = set_bin_average(i,l);
        Pmin[i][l] = r.Pmin;
        Pmax[i][l] = r.Pmax;
      }
    }

    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<lmin; l++) {
        Pl[i][l] = 0.0;
      }
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=lmin; l<Ntable.LMAX; l++) { 
        const double tmp = (1.0/(xmin[i] - xmax[i]))*(1. / (4.0 * M_PI));
        Pl[i][l] = tmp*(Pmin[i][l + 1] - Pmax[i][l + 1] 
                        - Pmin[i][l - 1] + Pmax[i][l - 1]);
      }
    }

    free(P);

    if (Cl != NULL) {
      free(Cl);
    }
    Cl = (double**) malloc2d(NSIZE, Ntable.LMAX);
  }

  if (fdiff2(cache[0], cosmology.random) || 
      fdiff2(cache[1], nuisance.random_photoz_clustering) ||
      fdiff2(cache[2], redshift.random_clustering) ||
      fdiff2(cache[3], Ntable.random) ||
      fdiff2(cache[4], nuisance.random_galaxy_bias))
  {
    const int lmin = 1;
    for (int i=0; i<NSIZE; i++) {
      for (int l=0; l<lmin; l++) {
        Cl[i][l] = 0.0;
      }
    }               
    (void) C_gg_tomo_limber((double) limits.LMIN_tab + 1, 0, 0); // init static vars
    if (1 == limber) {
      #pragma omp parallel for collapse(2) schedule(static,1)
      for (int nz=0; nz<NSIZE; nz++) {
        for (int l=lmin; l<limits.LMIN_tab; l++) {
          Cl[nz][l] = C_gg_tomo_limber_nointerp((double) l, nz, nz, 0);
        }
      }
      
      #pragma omp parallel for collapse(2) schedule(static,1)
      for (int nz=0; nz<NSIZE; nz++) {
        for (int l=limits.LMIN_tab; l<Ntable.LMAX; l++) {
          Cl[nz][l] = C_gg_tomo_limber((double) l, nz, nz);
        }
      }
    }
    else {
      const double tolerance = 0.01;
      C_cl_tomo(Cl, tolerance);
      #pragma omp parallel for collapse(2) schedule(static,1)
      for (int nz=0; nz<NSIZE; nz++) { // LIMBER PART
        for (int l=limits.LMAX_NOLIMBER; l<Ntable.LMAX; l++) {
          Cl[nz][l] = C_gg_tomo_limber(l, nz, nz);
        }
      }
    }

#ifdef COSMO2D_NOT_USE_SIMD
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        double sum = 0.0;
        for (int l=lmin; l<Ntable.LMAX; l++) {
          sum += Pl[i][l]*Cl[nz][l];
        }
        w_vec[nz*Ntable.Ntheta + i] = sum;
      }
    }
#else
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        w_vec[nz*Ntable.Ntheta + i] =
          simd_dot_product(Pl[i] + lmin, Cl[nz] + lmin, Ntable.LMAX - lmin);
      }
    }
#endif

    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_clustering;
    cache[2] = redshift.random_clustering;
    cache[3] = Ntable.random;
    cache[4] = nuisance.random_galaxy_bias;
  }

  if (nt < 0 || nt > Ntable.Ntheta - 1) {
    log_fatal("error in selecting bin number nt = %d (max %d)", nt, Ntable.Ntheta);
    exit(1); 
  }
  if (ni < -1 || 
      ni > redshift.clustering_nbin - 1 || 
      nj < -1 || 
      nj > redshift.clustering_nbin - 1)
  {
    log_fatal("error in selecting bin number (ni,nj) = [%d,%d]",ni,nj); exit(1);
  }
  if (ni != nj) {
    log_fatal("ni != nj tomography not supported"); exit(1);
  }
  const int q = ni * Ntable.Ntheta + nt;
  if (q  < 0 || q > NSIZE*Ntable.Ntheta - 1) {
    log_fatal("internal logic error in selecting bin number");
    exit(1);
  }  
  return w_vec[q];
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double w_gk_tomo(const int nt, const int ni, const int limber)
{
  static double** Pl = NULL;
  static double* w_vec = NULL;
  static double** Cl = NULL; 
  static uint64_t cache[MAX_SIZE_ARRAYS];

  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized");
    exit(1);
  }

  const int NSIZE = redshift.clustering_nbin;
  
  if (NULL == Pl ||
      NULL == w_vec || 
      NULL == Cl || 
      fdiff2(cache[3], Ntable.random))
  {
    if (Pl != NULL) free(Pl);
    Pl = (double**) malloc2d(Ntable.Ntheta, Ntable.LMAX);;

    if (w_vec != NULL) free(w_vec);
    w_vec = calloc1d(NSIZE*Ntable.Ntheta);

    double*** P = (double***) malloc3d(2, Ntable.Ntheta, Ntable.LMAX + 1);
    double** Pmin  = P[0]; double** Pmax  = P[1];

    double xmin[Ntable.Ntheta];
    double xmax[Ntable.Ntheta];
    for (int i=0; i<Ntable.Ntheta; i++)
    { // Cocoa: dont thread (init of static variables inside set_bin_average)
      bin_avg r = set_bin_average(i,0);
      xmin[i] = r.xmin;
      xmax[i] = r.xmax;
    }

    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<Ntable.LMAX; l++) {
        bin_avg r = set_bin_average(i,l);
        Pmin[i][l] = r.Pmin;
        Pmax[i][l] = r.Pmax;
      }
    }

    const int lmin = 1;
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<lmin; l++) {
        Pl[i][l] = 0.0;
      }
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=lmin; l<Ntable.LMAX; l++) {
        const double tmp = (1.0/(xmin[i] - xmax[i]))*(1.0 / (4.0 * M_PI));
        Pl[i][l] = tmp*(Pmin[i][l + 1] - Pmax[i][l + 1] - Pmin[i][l - 1] + Pmax[i][l - 1]);
      }
    }
    free(P);

    if (Cl != NULL) {
      free(Cl);
    }
    Cl = (double**) malloc2d(NSIZE, Ntable.LMAX);
  }

  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_clustering) ||
      fdiff2(cache[2], redshift.random_clustering) ||
      fdiff2(cache[3], Ntable.random) ||
      fdiff2(cache[4], nuisance.random_galaxy_bias) ||
      fdiff2(cache[5], cmb.random))
  { 
    const int lmin = 1;
    for (int i=0; i<NSIZE; i++) {
      for (int l=0; l<lmin; l++) {
        Cl[i][l] = 0.0;
      }
    } 
    if (1 == limber) {
      (void) C_gk_tomo_limber((double) limits.LMIN_tab + 1, 0); // init static vars
      #pragma omp parallel for collapse(2) schedule(static,1)
      for (int nz=0; nz<NSIZE; nz++) {
        for (int l=lmin; l<limits.LMIN_tab; l++) {
          Cl[nz][l] = 
              C_gk_tomo_limber_nointerp((double) l, nz, 0)*beam_cmb((double) l);
          if (cmb.healpixwin_ncls > 0) {
            Cl[nz][l] *= w_pixel((double) l);
          }
        }
      }
      #pragma omp parallel for collapse(2) schedule(static,1)
      for (int nz=0; nz<NSIZE; nz++) {
        for (int l=limits.LMIN_tab; l<Ntable.LMAX; l++) {
          Cl[nz][l] = C_gk_tomo_limber((double) l, nz)*beam_cmb((double) l);
          if (cmb.healpixwin_ncls > 0) {
            Cl[nz][l] *= w_pixel((double) l);
          }
        }
      }
    }
    else {
      log_fatal("NonLimber not implemented");
      exit(1);
    }
#ifdef COSMO2D_NOT_USE_SIMD
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        double sum = 0;
        for (int l=lmin; l<Ntable.LMAX; l++) {
          sum += Pl[i][l]*Cl[nz][l];
        }
        w_vec[nz*Ntable.Ntheta+i] = sum;
      }
    }
#else
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        w_vec[nz*Ntable.Ntheta+i] =
          simd_dot_product(Pl[i] + lmin, Cl[nz] + lmin, Ntable.LMAX - lmin);
      }
    }
#endif

    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_clustering;
    cache[2] = redshift.random_clustering;
    cache[3] = Ntable.random;
    cache[4] = nuisance.random_galaxy_bias;
    cache[5] = cmb.random;
  }
  if (ni < -1 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d (max %d)", ni, redshift.clustering_nbin);
    exit(1); 
  }
  if (nt < 0 || nt > Ntable.Ntheta - 1) {
    log_fatal("error in selecting bin number nt = %d (max %d)", nt, Ntable.Ntheta);
    exit(1); 
  }
  const int q = ni * Ntable.Ntheta + nt;
  if (q < 0 || q > NSIZE*Ntable.Ntheta - 1) {
    log_fatal("internal logic error in selecting bin number");
    exit(1);
  }
  return w_vec[q];
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double w_ks_tomo(const int nt, const int ni, const int limber)
{
  static double** Pl = NULL;
  static double* w_vec = NULL;
  static double** Cl = NULL; 
  static uint64_t cache[MAX_SIZE_ARRAYS];
  
  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized"); exit(1);
  }

  const int NSIZE = redshift.shear_nbin;

  if (Pl == NULL || 
      w_vec == NULL || 
      NULL == Cl || 
      fdiff2(cache[4], Ntable.random))
  {
    if (Pl != NULL) free(Pl);
    Pl = (double**) malloc2d(Ntable.Ntheta, Ntable.LMAX);;

    if (w_vec != NULL) free(w_vec);
    w_vec = calloc1d(NSIZE*Ntable.Ntheta);
    
    double*** P = (double***) malloc3d(2, Ntable.Ntheta, Ntable.LMAX + 1);
    double** Pmin  = P[0]; double** Pmax  = P[1];

    double xmin[Ntable.Ntheta];
    double xmax[Ntable.Ntheta];
    for (int i=0; i<Ntable.Ntheta; i++)
    { // Cocoa: dont thread (init of static variables inside set_bin_average)
      bin_avg r = set_bin_average(i,0);
      xmin[i] = r.xmin;
      xmax[i] = r.xmax;
    }

    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<Ntable.LMAX; l++) {
        bin_avg r = set_bin_average(i,l);
        Pmin[i][l] = r.Pmin;
        Pmax[i][l] = r.Pmax;
      }
    }

    const int lmin = 1;
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<lmin; l++) {
        Pl[i][0] = 0.0;
      }
    }

    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=lmin; l<Ntable.LMAX; l++) {
        Pl[i][l] = (2.*l+1)/(4.*M_PI*l*(l+1)*(xmin[i]-xmax[i]))
          *((l+2./(2*l+1.))*(Pmin[i][l-1]-Pmax[i][l-1])
          +(2-l)*(xmin[i]*Pmin[i][l]-xmax[i]*Pmax[i][l])
          -2./(2*l+1.)*(Pmin[i][l+1]-Pmax[i][l+1]));
      }
    }
    free(P);
    if (Cl != NULL) {
      free(Cl);
    }
    Cl = (double**) malloc2d(NSIZE, Ntable.LMAX);
  }

  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) || 
      fdiff2(cache[4], Ntable.random) ||
      fdiff2(cache[5], cmb.random))
  {
    const int lmin = 1;
    for (int i=0; i<NSIZE; i++) {
      for (int l=0; l<lmin; l++) {
        Cl[i][l] = 0.0;
      }
    } 
    if (1 == limber) {      
      (void) C_ks_tomo_limber((double) limits.LMIN_tab + 1, 0); // init static vars
      #pragma omp parallel for collapse(2) schedule(static,1)
      for (int nz=0; nz<redshift.shear_nbin; nz++) {
        for (int l=lmin; l<limits.LMIN_tab; l++) {
          Cl[nz][l] = 
              C_ks_tomo_limber_nointerp((double) l, nz, 0)*beam_cmb((double) l);
          if (cmb.healpixwin_ncls > 0) {
            Cl[nz][l] *= w_pixel((double) l);
          }
        }
      }
      #pragma omp parallel for collapse(2) schedule(static,1)
      for (int nz=0; nz<NSIZE; nz++) {
        for (int l=limits.LMIN_tab; l<Ntable.LMAX; l++) {
          Cl[nz][l] = C_ks_tomo_limber((double) l, nz)*beam_cmb((double) l);
          if (cmb.healpixwin_ncls > 0) {
            Cl[nz][l] *= w_pixel((double) l);
          }
        }
      }
    } 
    else {
      log_fatal("NonLimber not implemented");
      exit(1);
    }

#ifdef COSMO2D_NOT_USE_SIMD
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        double sum = 0;
        for (int l=lmin; l<Ntable.LMAX; l++) {
          sum += Pl[i][l]*Cl[nz][l];
        }
        w_vec[nz*Ntable.Ntheta+i] = sum;
      }
    }
#else
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        w_vec[nz*Ntable.Ntheta+i] =
          simd_dot_product(Pl[i] + lmin, Cl[nz] + lmin, Ntable.LMAX - lmin);
      }
    }
#endif

    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear; 
    cache[4] = Ntable.random;
    cache[5] = cmb.random;
  }
  if (nt < 0 || nt > Ntable.Ntheta - 1) {
    log_fatal("error in selecting bin number nt = %d (max %d)", nt, Ntable.Ntheta);
    exit(1); 
  }
  if (ni < -1 || ni > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d (max %d)", ni, redshift.shear_nbin);
    exit(1);
  }
  const int q = ni * Ntable.Ntheta + nt;
  if (q  < 0 || q > NSIZE*Ntable.Ntheta - 1) {
    log_fatal("internal logic error in selecting bin number");
    exit(1);
  }  
  return w_vec[q];
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Limber Approximation (Angular Power Spectrum)
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
//  create_cosmo_nodes once: chi, G, f_K, hoverh0 at all quadrature nodes
//                           These are independent of ell and tomo bins.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

typedef struct {
  int npts;
  double** data;
} cosmo_nodes;

enum {
  CN_A = 0,
  CN_WT,
  CN_FK,        // f_K
  CN_GROWFAC,   // growfac;
  CN_HOVERH0,   // hoverh0v2
  CN_DCHIDA,    // chidchi.dchida
  CN_NPARAMS    // trick to set automatically the number of parameters
};

cosmo_nodes create_cosmo_nodes(
    const double amin,
    const double amax,
    const gsl_integration_glfixed_table* w)
{
  cosmo_nodes cn;
  cn.npts = (int) w->n;
  cn.data = (double**) malloc2d(CN_NPARAMS, cn.npts);

  #pragma omp parallel for schedule(static,1)
  for (int p = 0; p < cn.npts; p++) {
    gsl_integration_glfixed_point(amin, 
                                  amax, 
                                  p, 
                                  &cn.data[CN_A][p], 
                                  &cn.data[CN_WT][p], 
                                  w);
    const double a      = cn.data[CN_A][p];
    struct chis chidchi = chi_all(a);
    cn.data[CN_FK][p]   = chidchi.chi;
    cn.data[CN_GROWFAC][p] = growfac(a);
    cn.data[CN_HOVERH0][p] = hoverh0v2(a, chidchi.dchida);
    cn.data[CN_DCHIDA][p]  = chidchi.dchida;
  }
  return cn;
}

void free_cosmo_nodes(cosmo_nodes* cn) {
  free(cn->data);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// SS = SHEAR SHEAR
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

static double int_for_C_ss_tomo_limber_core(
    const double a,
    const double fK,
    const double PK,
    const double growfac_a,
    const double hoverh0,
    const double dchida,
    const double ell_prefactor,
    const double l,
    const int n1,
    const int n2,
    const double WK1, 
    const double WK2,
    const double WS1, 
    const double WS2,
    const int EE,
    const int deriv
  )
{
  const double ell = l + 0.5;
  const double k = ell/fK;
  
  double ans = 1.0;
  switch(nuisance.IA_MODEL) 
  {
    case IA_MODEL_TATT:
    {
      if (0 == nuisance.IA_code) { // call C-FAST-PT to compute IA terms
        get_FPT_IA();
      }

      const double lnk = log(k);
      const double g4 = growfac_a*growfac_a*growfac_a*growfac_a;
      
      double IA_AX[2];
      IA_A1_Z1Z2(a, growfac_a, n1, n2, IA_AX);
      const double C11 = IA_AX[0];
      const double C12 = IA_AX[1];
      IA_A2_Z1Z2(a, growfac_a, n1, n2, IA_AX);
      const double C21 = IA_AX[0];
      const double C22 = IA_AX[1];
      IA_BTA_Z1Z2(a, growfac_a, n1, n2, IA_AX);
      const double bta1 = IA_AX[0];
      const double bta2 = IA_AX[1];
      
      double lim[3];
      lim[0] = log(FPTIA.k_min);
      lim[1] = log(FPTIA.k_max);
      lim[2] = (lim[1] - lim[0])/FPTIA.N;

      if (1 == EE) {
        double tt, ta_dE1, ta_dE2, ta, mixA, mixB, mixEE;
        if (lnk < lim[0] || lnk > lim[1]) {
          tt = 0.0; ta_dE1 = 0.0; ta_dE2 = 0.0; ta = 0.0;
          mixA = 0.0; mixB = 0.0; mixEE = 0.0;
        }
        else {
          const double r = (lnk - lim[0]) / lim[2];
          const int i = (int)floor(r);
          if (i + 1 >= FPTIA.N) {
            tt     = g4 * FPTIA.tab[0][FPTIA.N - 1];
            ta_dE1 = g4 * FPTIA.tab[2][FPTIA.N - 1];
            ta_dE2 = g4 * FPTIA.tab[3][FPTIA.N - 1];
            ta     = g4 * FPTIA.tab[4][FPTIA.N - 1];
            mixA   = g4 * FPTIA.tab[6][FPTIA.N - 1];
            mixB   = g4 * FPTIA.tab[7][FPTIA.N - 1];
            mixEE  = g4 * FPTIA.tab[8][FPTIA.N - 1];
          }
          else {
            const double w = r - i;
            tt = g4 *(w*(FPTIA.tab[0][i+1] - FPTIA.tab[0][i]) + FPTIA.tab[0][i]);
            ta_dE1 = g4*(w*(FPTIA.tab[2][i+1] - FPTIA.tab[2][i]) + FPTIA.tab[2][i]);
            ta_dE2 = g4*(w*(FPTIA.tab[3][i+1] - FPTIA.tab[3][i]) + FPTIA.tab[3][i]);
            ta = g4*(w*(FPTIA.tab[4][i+1] - FPTIA.tab[4][i]) + FPTIA.tab[4][i]);
            mixA = g4*(w*(FPTIA.tab[6][i+1] - FPTIA.tab[6][i]) + FPTIA.tab[6][i]);
            mixB = g4*(w*(FPTIA.tab[7][i+1] - FPTIA.tab[7][i]) + FPTIA.tab[7][i]);
            mixEE  = g4*(w*(FPTIA.tab[8][i+1] - FPTIA.tab[8][i]) + FPTIA.tab[8][i]);
          }
        }

        ans = WK1*WK2*PK 
              - WS1*WK2*(C11*PK + C11*bta1*(ta_dE1+ta_dE2) - 5*C21*(mixA+mixB))
              - WS2*WK1*(C12*PK + C12*bta2*(ta_dE1+ta_dE2) - 5*C22*(mixA+mixB))
              + WS1*WS2*(C11*C12*PK 
                         + C11*C12*(bta1*bta2*ta + (bta1+bta2)*(ta_dE1+ta_dE2))
                         - 5.*(C11*C22 + C12*C21)*(mixA+mixB)
                         - 5.*(C11*bta1*C22+C12*bta2*C21)*mixEE
                         + 25.*C21*C22*tt);
      }
      else  {
        double tt, ta, mix;
        if (lnk < lim[0] || lnk > lim[1]) {
          tt  = 0.0;
          ta  = 0.0;
          mix = 0.0;
        } else {
          const double r = (lnk - lim[0]) / lim[2];
          const int i = (int) floor(r);
          const double w = r - i;
          const int i1 = (i + 1 >= FPTIA.N) ? FPTIA.N - 1 : i + 1;
          tt  = g4 * (w * (FPTIA.tab[1][i1] - FPTIA.tab[1][i]) + FPTIA.tab[1][i]);
          ta  = g4 * (w * (FPTIA.tab[5][i1] - FPTIA.tab[5][i]) + FPTIA.tab[5][i]);
          mix = g4 * (w * (FPTIA.tab[9][i1] - FPTIA.tab[9][i]) + FPTIA.tab[9][i]);
        }        
        ans = WS1*WS2*(C11*C12*bta1*bta2*ta 
                       - 5.*(C11*bta1*C22+C12*bta2*C21)*mix 
                       + 25.*C21*C22*tt);
      }
      break;
    }
    case IA_MODEL_NLA:
    {
      if (1 == EE) { 
        double IA_A1[2];
        IA_A1_Z1Z2(a, growfac_a, n1, n2, IA_A1);
        const double C11 = IA_A1[0];
        const double C12 = IA_A1[1];
        ans =   WK1*WK2*PK 
              - WS1*WK2*C11*PK 
              - WS2*WK1*C12*PK
              + WS1*WS2*C11*C12*PK;
      }
      else {
        ans = 0.0;
      }
      break;
    }
    default: {
      log_fatal("nuisance.IA_MODEL = %d not supported", nuisance.IA_MODEL); exit(1);
    }
  }
  if (0 == deriv) {
    return ans*(dchida/(fK*fK))*ell_prefactor;
  } 
  else { // dCXY/dlnk: important to determine scale cuts (2011.06469 eq 17)
    return ans*(dchida/fK)*ell_prefactor;
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_C_ss_tomo_limber(double a, void* params)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  double* ar = (double*) params;
  const int n1 = (int) ar[0]; // first source bin 
  const int n2 = (int) ar[1]; // second source bin 
  if (n1 < 0 || n1 > redshift.shear_nbin - 1 || 
      n2 < 0 || n2 > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (ni,nj) = [%d,%d]", n1,n2); exit(1);
  }
  const double l = ar[2];
  const int EE = (int) ar[3];
  const int deriv = (int) ar[4];

  const double ell = l + 0.5;
  struct chis chidchi = chi_all(a);
  const double growfac_a = growfac(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const double fK = chidchi.chi; // (Mpc/h)/(c/H0=100) (dimensionless)
  const double k = ell/fK;       // (c/H0)/(Mpc/h)
  const double PK  = Pdelta(k, a);
  const double ell4 = ell*ell*ell*ell; // correction (1812.05995 eqs 74-79)
  const double ell_prefactor = l*(l - 1.)*(l + 1.)*(l + 2.)/ell4; 

  const double WK1 = W_kappa(a, fK, n1);
  const double WK2 = W_kappa(a, fK, n2);
  const double WS1 = W_source(a, n1, hoverh0);
  const double WS2 = W_source(a, n2, hoverh0);

  return int_for_C_ss_tomo_limber_core(
      a, fK, PK, growfac_a, hoverh0, chidchi.dchida,
      ell_prefactor, l, n1, n2, WK1, WK2, WS1, WS2, EE, deriv
    );
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double C_ss_tomo_limber_nointerp(
    const double l, 
    const int ni, 
    const int nj, 
    const int EE, 
    const int init
  ) // still used for l < limits.LMIN_tab
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL; 
  if (ni < -1 || ni > redshift.shear_nbin -1 || 
      nj < -1 || nj > redshift.shear_nbin -1) {
    log_fatal("invalid bin input (ni, nj) = (%d, %d)", ni, nj); exit(1);
  }
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 64 : 
                         (1 == hdi) ? 96 : 
                         (2 == hdi) ? 128 : 
                         (3 == hdi) ? 256 : 512; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }
  double ar[5] = {(double) ni, 
                  (double) nj, 
                  l, 
                  (double) EE, 
                  (double) 0};
  double res = 0.0;
  const double amin = 1./(redshift.shear_zdist_zmax_all+1.);
  const double amax = 1./(1.+fmax(redshift.shear_zdist_zmin_all,1e-6));
  if (1 == init) {
    res = int_for_C_ss_tomo_limber(amin, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_ss_tomo_limber;
    res = gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;   
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// so C_ss_tomo_limber_fill can see C_ss_tomo_limber data
static struct { double*** tab; double lim[3]; int nell; } ss_ = {0};

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double C_ss_tomo_limber(
    const double l, 
    const int ni, 
    const int nj, 
    const int EE
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double*** table;
  static double lim[3];
  static int nell;
  static gsl_integration_glfixed_table* w = NULL;

  if (NULL == table || fdiff2(cache[4], Ntable.random)) {
    nell = Ntable.N_ell;
    lim[0] = log(fmax(limits.LMIN_tab - 1., 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0]) / ((double) nell - 1.);
    
    if (table != NULL) free(table);
    table = (double***) malloc3d(2, tomo.shear_Npowerspectra, nell);
  
    ss_.tab = table; 
    ss_.lim[0] = lim[0]; 
    ss_.lim[1] = lim[1]; 
    ss_.lim[2] = lim[2]; 
    ss_.nell = nell;  

    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 64 :
                         (1 == hdi) ? 96 :
                         (2 == hdi) ? 128 :
                         (3 == hdi) ? 256 : 512;
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
  }

  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    for (int k=0; k<tomo.shear_Npowerspectra; k++) { // init static vars
      const double Z1NZ = Z1(k);
      const double Z2NZ = Z2(k);
      (void) C_ss_tomo_limber_nointerp(exp(lim[0]), Z1NZ, Z2NZ, 1, 1); // EE
      (void) C_ss_tomo_limber_nointerp(exp(lim[0]), Z1NZ, Z2NZ, 0, 1); // BB  
    }

    // ------------------------------------------------------------------
    // optimization: - compute cosmo quantities and prefactor only once.
    //                 (why once? amin and amax are nl and ns independent)
    //               - compute WS only nsource times (not ns(ns-1)/2 !) 
    // ------------------------------------------------------------------
    const double amin = 1./(redshift.shear_zdist_zmax_all+1.);
    const double amax = 1./(1.+fmax(redshift.shear_zdist_zmin_all,1e-6));
    
    cosmo_nodes cn = create_cosmo_nodes(amin, amax, w);
    
    double lx[nell];
    double ell_prefactor[nell];
    for (int i = 0; i < nell; i++) {
      lx[i] = exp(lim[0] + i * lim[2]);
      const double ell = lx[i] + 0.5;
      const double ell4 = ell * ell * ell * ell;
      ell_prefactor[i] = lx[i]*(lx[i]-1.)*(lx[i]+1.)*(lx[i]+2.)/ell4;
    }

    // precompute P(k,z) (matter power spectrum)
    double** PK = (double**) malloc2d(nell, cn.npts);
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i = 0; i < nell; i++)  {
      for (int p = 0; p < cn.npts; p++) {
        const double ell = lx[i] + 0.5;
        const double fK = cn.data[CN_FK][p];
        const double k = ell / fK;
        PK[i][p] = Pdelta(k, cn.data[CN_A][p]);
      }
    }
    
    // precompute: WS (only  need to be computed ns times, not ns (ns -1)/2
    double*** W = (double***) malloc3d(2, redshift.shear_nbin, cn.npts);
    for (int b = 0; b < redshift.shear_nbin; b++) {
      for (int p = 0; p < cn.npts; p++) {
        W[0][b][p] = W_kappa(cn.data[CN_A][p], cn.data[CN_FK][p], b);
        W[1][b][p] = W_source(cn.data[CN_A][p], b, cn.data[CN_HOVERH0][p]);
      }
    }

    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i = 0; i < nell; i++) {
      for (int k = 0; k < tomo.shear_Npowerspectra; k++) {
        const int Z1NZ = Z1(k);
        const int Z2NZ = Z2(k);
        if (Z1NZ < 0 || Z1NZ > redshift.shear_nbin - 1 || 
            Z2NZ < 0 || Z2NZ > redshift.shear_nbin - 1) {
          log_fatal("error in selecting bin number (ni,nj) = [%d,%d]",Z1NZ, Z2NZ); 
          exit(1);
        }
        double sum_EE = 0.0;
        double sum_BB = 0.0;
        for (int p = 0; p < cn.npts; p++) {
          const double a = cn.data[CN_A][p];
          const double fK = cn.data[CN_FK][p];
          const double growfac_a = cn.data[CN_GROWFAC][p];
          const double hoverh0 = cn.data[CN_HOVERH0][p];
          const double dchida = cn.data[CN_DCHIDA][p];
          const double wt = cn.data[CN_WT][p];
          
          const double WK1 = W[0][Z1NZ][p];
          const double WK2 = W[0][Z2NZ][p];
          const double WS1 = W[1][Z1NZ][p];
          const double WS2 = W[1][Z2NZ][p];

          sum_EE += int_for_C_ss_tomo_limber_core(a, fK, PK[i][p], growfac_a, 
            hoverh0, dchida, ell_prefactor[i], lx[i], Z1NZ, Z2NZ, 
            WK1, WK2, WS1, WS2, 1, 0) * wt;
          if (nuisance.IA_MODEL == IA_MODEL_TATT) {
            sum_BB += int_for_C_ss_tomo_limber_core(a, fK, PK[i][p], growfac_a, 
              hoverh0, dchida, ell_prefactor[i], lx[i], Z1NZ, Z2NZ, 
              WK1, WK2, WS1, WS2, 0, 0) * wt;
          }
        }
        table[0][k][i] = sum_EE;
        table[1][k][i] = sum_BB;
      }
    }
    
    free(W);
    free(PK);
    free_cosmo_nodes(&cn);

    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear;
    cache[4] = Ntable.random;
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (ni,nj) = [%d,%d]", ni,nj); exit(1);
  }
  const double lnl = log(l);
  if (lnl < lim[0]) {
    log_warn("l = %e < lmin = %e. Extrapolation adopted", l, exp(lim[0]));
  }
  if (lnl > lim[1]) {
    log_warn("l = %e > lmax = %e. Extrapolation adopted", l, exp(lim[1]));
  }
  const int q = N_shear(ni, nj);
  if (q < 0 || q > tomo.shear_Npowerspectra - 1) {
    log_fatal("internal logic error in selecting bin number");
    exit(1);
  }
  return interpol1d((1==EE)?table[0][q]:table[1][q],nell,lim[0],lim[1],lim[2],lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void C_ss_tomo_limber_fill(
    const int nz, 
    const int lmin, 
    const int lmax,
    const double* restrict ln_ell,
    double* restrict out_EE,
    double* restrict out_BB
  )
{
  const double* restrict tab_EE = ss_.tab[0][nz];
  const double* restrict tab_BB = ss_.tab[1][nz];
  const double inv_dx = 1.0 / ss_.lim[2];
  const double a = ss_.lim[0];
  const int n = ss_.nell;

  for (int l = lmin; l < lmax; l++) { // inline interpol 1D
    const double r = (ln_ell[l] - a) * inv_dx;
    const int i = (int) floor(r);
    const int ic = i < 0 ? 0 : (i >= n - 1 ? n - 2 : i);
    const double t = r - ic;
    out_EE[l] = tab_EE[ic] + t * (tab_EE[ic + 1] - tab_EE[ic]);
    out_BB[l] = tab_BB[ic] + t * (tab_BB[ic + 1] - tab_BB[ic]);
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GS = GALAXY-SHEAR
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

static double int_for_C_gs_tomo_limber_core(
    const double a,
    const double fK,
    const double PK,
    const double growfac_a,
    const double hoverh0,
    const double dchida,
    const double ell_prefactor,
    const double ell_prefactor2,
    const double l,
    const int nl, 
    const int ns,
    const double WK, 
    const double WS,
    const double WGAL, 
    const double WMAG,
    const int nonlinear_bias
  ) 
{
  const double ell = l + 0.5;
  const double z = 1.0/a - 1.0;
  const double b1 = gb1(z, nl);
  const double bmag = gbmag(z, nl);

  double ans;

  switch(nuisance.IA_MODEL)
  {
    case IA_MODEL_TATT:
    {
      if (1 == include_HOD_GX) {
        log_fatal("HOD NOT IMPLEMENTED");
        exit(1);
      }

      if (0 == nuisance.IA_code){
        get_FPT_IA();
      }
      
      const double k = ell/fK;
      const double lnk = log(k);
      double lim[3];
      lim[0] = log(FPTIA.k_min);
      lim[1] = log(FPTIA.k_max);
      lim[2] = (lim[1] - lim[0])/FPTIA.N;
      
      const double g4 = growfac_a*growfac_a*growfac_a*growfac_a;

      double mixA, mixB, ta_dE1, ta_dE2;
      if (lnk < lim[0] || lnk > lim[1]) {
        mixA = 0.0; mixB = 0.0; ta_dE1 = 0.0; ta_dE2 = 0.0;
      }
      else {
        const double r = (lnk - lim[0]) / lim[2];
        const int i = (int) floor(r);
        if (i + 1 >= FPTIA.N)
        {
          mixA   = g4 * FPTIA.tab[6][FPTIA.N - 1];
          mixB   = g4 * FPTIA.tab[7][FPTIA.N - 1];
          ta_dE1 = g4 * FPTIA.tab[2][FPTIA.N - 1];
          ta_dE2 = g4 * FPTIA.tab[3][FPTIA.N - 1];
        }
        else {
          const double t = r - i;
          mixA   = g4 * (t * (FPTIA.tab[6][i+1] - FPTIA.tab[6][i]) + FPTIA.tab[6][i]);
          mixB   = g4 * (t * (FPTIA.tab[7][i+1] - FPTIA.tab[7][i]) + FPTIA.tab[7][i]);
          ta_dE1 = g4 * (t * (FPTIA.tab[2][i+1] - FPTIA.tab[2][i]) + FPTIA.tab[2][i]);
          ta_dE2 = g4 * (t * (FPTIA.tab[3][i+1] - FPTIA.tab[3][i]) + FPTIA.tab[3][i]);
        }
      }

      double WRSD = 0.0;
      if (1 == include_RSD_GS) {
        const double chi_0 = ell/k;
        const double chi_1 = (ell+1.)/k;
        const double a_0 = a_chi(chi_0);
        const double a_1 = a_chi(chi_1);
        WRSD = W_RSD(ell, a_0, a_1, nl);
      }

      double oneloop = 0.0;
      if (1 == nonlinear_bias)
      { 
        if (0 == nuisance.IA_code){
          get_FPT_bias();
        }
        
        lim[0] = log(FPTbias.k_min);
        lim[1] = log(FPTbias.k_max);
        lim[2] = (lim[1] - lim[0])/FPTbias.N;

        const double d1d2 = (lnk<lim[0] || lnk>lim[1]) ? 0.0 :
          interpol1d(FPTbias.tab[0], FPTbias.N, lim[0], lim[1], lim[2], lnk);
        
        const double d1s2 = (lnk<lim[0] || lnk>lim[1]) ? 0.0 :
          interpol1d(FPTbias.tab[2], FPTbias.N, lim[0], lim[1], lim[2], lnk);
        
        const double d1p3 = (lnk<lim[0] || lnk>lim[1]) ? 0.0 :
          interpol1d(FPTbias.tab[5], FPTbias.N, lim[0], lim[1], lim[2], lnk);

        const double b2 = gb2(z, nl);
        const double bs2 = gbs2(z, nl);
        const double b3 = gb3(z, nl);
        const double bk = gbK(z, nl);

        oneloop = 0.5*g4*(b2 * d1d2 + bs2 * d1s2 + b3 * d1p3) + (bk * k * k * PK);
      }
      

      const double C1ZS  = IA_A1_Z1(a, growfac_a, ns);
      const double btazs = IA_BTA_Z1(a, growfac_a, ns);
      const double C2ZS  = IA_A2_Z1(a, growfac_a, ns);

      // TODO: IS THIS CONSISTENT (WRSD, ONELOOP AND IA CROSS TERMS)?
      ans =  WK*((WGAL*b1+WMAG*ell_prefactor*bmag+WRSD)*PK+WGAL*oneloop) 
            -WS*(WGAL*b1+WMAG*ell_prefactor*bmag)*( C1ZS*PK
                                                    + C1ZS*btazs*(ta_dE1+ta_dE2) 
                                                    - 5*C2ZS*(mixA+mixB));
      break;
    }
    case IA_MODEL_NLA:
    {
      if (include_HOD_GX == 1) {
        log_fatal("HOD NOT IMPLEMENTED");
        exit(1);
      }

      double WRSD = 0.0;
      if (1 == include_RSD_GS) {
        const double k = ell/fK;
        const double chi_0 = ell/k;
        const double chi_1 = (ell+1.)/k;
        const double a_0 = a_chi(chi_0);
        const double a_1 = a_chi(chi_1);
        WRSD = W_RSD(ell, a_0, a_1, nl);
      }

      double oneloop = 0.0;
      
      if (1 == nonlinear_bias) {
        if (0 == nuisance.IA_code){
          get_FPT_bias();
        }
        
        const double k = ell/fK;
        const double lnk = log(k);
        double lim[3];
        lim[0] = log(FPTbias.k_min);
        lim[1] = log(FPTbias.k_max);
        lim[2] = (lim[1] - lim[0])/FPTbias.N;

        const double d1d2 = (lnk<lim[0] || lnk>lim[1]) ? 0.0 :
          interpol1d(FPTbias.tab[0], FPTbias.N, lim[0], lim[1], lim[2], lnk);
        
        const double d1s2 = (lnk<lim[0] || lnk>lim[1]) ? 0.0 :
          interpol1d(FPTbias.tab[2], FPTbias.N, lim[0], lim[1], lim[2], lnk);
        
        const double d1p3 = (lnk<lim[0] || lnk>lim[1]) ? 0.0 :
          interpol1d(FPTbias.tab[5], FPTbias.N, lim[0], lim[1], lim[2], lnk);

        const double b2 = gb2(z, nl);
        const double bs2 = gbs2(z, nl);
        const double b3 = gb3(z, nl);
        const double bk = gbK(z, nl);
        
        const double g4 = growfac_a*growfac_a*growfac_a*growfac_a;

        oneloop = 0.5*g4*(b2*d1d2 + bs2*d1s2 + b3*d1p3) + (bk * k * k * PK);
      }
          
      const double C1ZS = IA_A1_Z1(a, growfac_a, ns);

      ans = (WK - WS*C1ZS)*((WGAL*b1 + WMAG*ell_prefactor*bmag + WRSD)*PK 
                            + WGAL*oneloop);
      break;
    }
    default:
    {
      log_fatal("nuisance.IA_MODEL = %d not supported", nuisance.IA_MODEL);
      exit(1);
    }
  }
  return ans*(dchida/(fK*fK))*ell_prefactor2;
}

double int_for_C_gs_tomo_limber(double a, void* params)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true");
    exit(1);
  }
  double* ar = (double*) params;
  const int nl = (int) ar[0];
  const int ns = (int) ar[1];
  if (nl < 0 || nl > redshift.clustering_nbin - 1 || 
      ns < 0 || ns > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (nl, ns) = [%d,%d]", nl, ns);
    exit(1);
  }
  const double l = ar[2];
  const int nonlinear_bias = ar[3];
  
  const double growfac_a = growfac(a);
  struct chis chidchi = chi_all(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const double g4 = growfac_a*growfac_a*growfac_a*growfac_a;
  const double ell = l + 0.5;
  const double fK = chidchi.chi;
  const double k = ell/fK;
  const double z = 1.0/a - 1.0;
  const double PK = Pdelta(k,a);

  const double ell_prefactor = l*(l + 1.)/(ell*ell); // correction (1812.05995 eqs 74-79)
  const double tmp = (l - 1.)*l*(l + 1.)*(l + 2.);   // correction (1812.05995 eqs 74-79)
  const double ell_prefactor2 = (tmp > 0) ? sqrt(tmp)/(ell*ell) : 0.0;

  const double WK = W_kappa(a, fK, ns);
  const double WS   = W_source(a, ns, hoverh0);
  const double WGAL = W_gal(a, nl, hoverh0);
  const double WMAG = W_mag(a, fK, nl);

  return int_for_C_gs_tomo_limber_core(a, fK, PK, growfac_a, hoverh0,
                                       chidchi.dchida, ell_prefactor, 
                                       ell_prefactor2, l,
                                       nl, ns, WK, WS, WGAL, WMAG, 
                                       nonlinear_bias);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double C_gs_tomo_limber_nointerp(
    const double l, 
    const int nl, 
    const int ns,
    const int init
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  
  if (nl < -1 || 
      nl > redshift.clustering_nbin -1 || 
      ns < -1 || 
      ns > redshift.shear_nbin -1)
  {
    log_fatal("invalid bin input (ni, nj) = (%d, %d)", nl, ns);
    exit(1);
  }

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 64 : 
                         (1 == hdi) ? 96 : 
                         (2 == hdi) ? 128 : 
                         (3 == hdi) ? 256 : 512; // predefined GSL tables
    if (w != NULL)  {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[4] = {(double) nl, (double) ns, l, has_b2_galaxies()};
  
  const double amin = amin_lens(nl);
  const double amax = amax_lens(nl);
  
  if (!(amin>0) || !(amin<1) || !(amax>0) || !(amax<1)) {
    log_fatal("0 < amin/amax < 1 not true");
    exit(1);
  }
  if (!(amin < amax)) {
    log_fatal("amin < amax not true");
    exit(1);
  }

  double res;
  if (init == 1) {
    res = int_for_C_gs_tomo_limber(amin, (void*) ar);
  }
  else {    
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_gs_tomo_limber;
    res = gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// so C_gs_tomo_limber_fill can see C_gs_tomo_limber data
static struct { double** tab; double lim[3]; int nell; } gs_ = {0};

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double C_gs_tomo_limber(const double l, const int ni, const int nj)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static int nell;
  static double lim[3];
  static gsl_integration_glfixed_table* w = NULL;

  if (NULL == table || fdiff2(cache[6], Ntable.random)) {
    nell   = Ntable.N_ell;
    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0]) / ((double) nell - 1.0);

    if (table != NULL) {
      free(table);
    }
    table = (double**) malloc2d(tomo.ggl_Npowerspectra, nell);

    gs_.tab    = table;
    gs_.lim[0] = lim[0];
    gs_.lim[1] = lim[1];
    gs_.lim[2] = lim[2];
    gs_.nell   = nell;

    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 64 :
                         (1 == hdi) ? 96 :
                         (2 == hdi) ? 128 :
                         (3 == hdi) ? 256 : 512;
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
  }

  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_photoz_clustering) ||
      fdiff2(cache[3], nuisance.random_ia) ||
      fdiff2(cache[4], redshift.random_shear) ||
      fdiff2(cache[5], redshift.random_clustering) ||
      fdiff2(cache[6], Ntable.random) ||
      fdiff2(cache[7], nuisance.random_galaxy_bias))
  {
    for (int k=0; k<tomo.ggl_Npowerspectra; k++) { // init static vars     
      (void) C_gs_tomo_limber_nointerp(exp(lim[0]), ZL(k), ZS(k), 1);
    }
    
    const int nonlinear_bias = has_b2_galaxies();

    // ------------------------------------------------------------------
    // optimization: amin and amax are dependent of redshift bin
    //               we can compute cosmo quantities and nlens times. 
    // ------------------------------------------------------------------
    const int nbin = redshift.clustering_nbin;
    cosmo_nodes cn_all[redshift.clustering_nbin];
    for (int ZLNZ = 0; ZLNZ < redshift.clustering_nbin; ZLNZ++) {
      const double amin = amin_lens(ZLNZ);
      const double amax = amax_lens(ZLNZ);
      cn_all[ZLNZ] = create_cosmo_nodes(amin, amax, w);
    }

    double lx[nell];
    double ell_prefactor[nell];
    double ell_prefactor2[nell];
    for (int i = 0; i < nell; i++) {
      lx[i] = exp(lim[0] + i * lim[2]);
      const double ell = lx[i] + 0.5;
      ell_prefactor[i] = lx[i]*(lx[i]+1.)/(ell*ell); // correction (1812.05995 eqs 74-79)
      const double tmp = (lx[i]-1.)*lx[i]*(lx[i]+1.)*(lx[i]+2.);
      ell_prefactor2[i] = (tmp > 0) ? sqrt(tmp)/(ell*ell) : 0.0;  
    }

    // precompute P(k,z)
    double*** PK = (double***) malloc3d(redshift.clustering_nbin, nell, cn_all[0].npts);
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int q = 0; q < redshift.clustering_nbin; q++) {
      for (int i = 0; i < nell; i++) {
        for (int p = 0; p < cn_all[0].npts; p++) {
          const double ell = lx[i] + 0.5;
          const double k = ell / cn_all[q].data[CN_FK][p];
          PK[q][i][p] = Pdelta(k, cn_all[q].data[CN_A][p]);
        }
      }
    }

    // precompute lens weights 
    double*** WXL = (double***) malloc3d(2, redshift.clustering_nbin, cn_all[0].npts);
    #pragma omp parallel for schedule(static,1)
    for (int zl = 0; zl < redshift.clustering_nbin; zl++) {
      for (int p = 0; p < cn_all[0].npts; p++) {
        const cosmo_nodes* cn = &cn_all[zl];
        WXL[0][zl][p] = W_gal(cn->data[CN_A][p], zl, cn->data[CN_HOVERH0][p]);
        WXL[1][zl][p] = W_mag(cn->data[CN_A][p], cn->data[CN_FK][p], zl);
      }
    }

    // precompute source weights
    double**** WXS = (double****) malloc4d(2, redshift.clustering_nbin, 
                                          redshift.shear_nbin, cn_all[0].npts);
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int zl = 0; zl < redshift.clustering_nbin; zl++) {
      for (int zs = 0; zs < redshift.shear_nbin; zs++) {
        for (int p = 0; p < cn_all[0].npts; p++) {
          const cosmo_nodes* cn = &cn_all[zl];
          WXS[0][zl][zs][p] = W_kappa(cn->data[CN_A][p], cn->data[CN_FK][p], zs);
          WXS[1][zl][zs][p] = W_source(cn->data[CN_A][p], zs, cn->data[CN_HOVERH0][p]);
        }
      }
    }

    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int k = 0; k<tomo.ggl_Npowerspectra; k++) {
      for (int i = 0; i<nell; i++) {
        const int ZLNZ = ZL(k);
        const int ZSNZ = ZS(k);
        const cosmo_nodes* cn = &cn_all[ZLNZ];
        
        double sum = 0.0;
        for (int p = 0; p<cn_all[0].npts; p++) {
          const double a = cn->data[CN_A][p];
          const double fK = cn->data[CN_FK][p];
          const double growfac_a = cn->data[CN_GROWFAC][p];
          const double hoverh0 = cn->data[CN_HOVERH0][p];
          const double dchida = cn->data[CN_DCHIDA][p];
          const double wt = cn->data[CN_WT][p];

          const double WK   = WXS[0][ZLNZ][ZSNZ][p];
          const double WS   = WXS[1][ZLNZ][ZSNZ][p];
          const double WGAL = WXL[0][ZLNZ][p];
          const double WMAG = WXL[1][ZLNZ][p];

          sum += int_for_C_gs_tomo_limber_core(a, fK, PK[ZLNZ][i][p], growfac_a, 
                                               hoverh0, dchida, ell_prefactor[i], 
                                               ell_prefactor2[i], lx[i], ZLNZ,
                                               ZSNZ, WK, WS, WGAL, WMAG, 
                                               nonlinear_bias) * wt;
        }
        table[k][i] = sum;
      }
    }

    free(PK);
    for (int ZLNZ = 0; ZLNZ < nbin; ZLNZ++) {
      free_cosmo_nodes(&cn_all[ZLNZ]);
    }
    
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_photoz_clustering;
    cache[3] = nuisance.random_ia;
    cache[4] = redshift.random_shear;
    cache[5] = redshift.random_clustering;
    cache[6] = Ntable.random;
    cache[7] = nuisance.random_galaxy_bias;
  }

  if (ni < 0 || ni > redshift.clustering_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (ni, nj) = [%d,%d]", ni, nj);
    exit(1);
  }
  double res = 0.0;
  if (test_zoverlap(ni,nj)) {
    const double lnl = log(l);
    if (lnl < lim[0]) {
      log_warn("l = %e < lmin = %e. Extrapolation adopted", l, exp(lim[0]));
    }
    if (lnl > lim[1]) {
      log_warn("l = %e > lmax = %e. Extrapolation adopted", l, exp(lim[1]));
    }
    const int q = N_ggl(ni, nj);
    if (q < 0 || q > tomo.ggl_Npowerspectra - 1) {
      log_fatal("internal logic error in selecting bin number");
      exit(1);
    }
    res = interpol1d(table[q], nell, lim[0], lim[1], lim[2], lnl);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

void C_gs_tomo_limber_fill(
    const int nz,
    const int lmin,
    const int lmax,
    const double* RESTRICT ln_ell,
    double* RESTRICT out
  )
{ 
  const double* RESTRICT tab = gs_.tab[nz];
  const double inv_dx = 1.0 / gs_.lim[2];
  const double a = gs_.lim[0];
  const int n = gs_.nell;
 
  for (int l = lmin; l < lmax; l++) { // inline interpol1D
    const double r = (ln_ell[l] - a) * inv_dx;
    const int i = (int) floor(r);
    const int ic = i < 0 ? 0 : (i >= n - 1 ? n - 2 : i);
    const double t = r - ic;
    out[l] = tab[ic] + t * (tab[ic + 1] - tab[ic]);
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_C_gg_tomo_limber(double a, void* params)
{
  if(!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  double* ar = (double*) params;

  const int ni = (int) ar[0];
  const int nj = (int) ar[1];
  const double l  = ar[2];
  const int use_linear_ps = (int) ar[3];
  const int nonlinear_bias = ar[4];


  struct chis chidchi = chi_all(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const double ell = l + 0.5;
  const double fK = chidchi.chi;
  const double k = ell / fK;
  const double z = 1.0/a - 1.0;

  const double b1i   = gb1(z, ni);
  const double bmagi = gbmag(z, ni);
  const double WGALi = W_gal(a, ni, hoverh0);
  const double WMAGi = W_mag(a, fK, ni);
  
  const double b1j   = b1i;    // We assume ni = nj;
  const double bmagj = bmagi;
  const double WGALj = WGALi;
  const double WMAGj = WMAGi;

  const double ell_prefactor = l*(l+1.)/(ell*ell); // prefactor correction (1812.05995 eqs 74-79)
  
  double res = 1.0;
  
  if (include_HOD_GX == 1)
  {
    if (include_RSD_GG == 1)
    {
      log_fatal("RSD not implemented with (HOD = TRUE)");
      exit(1);
    }
    else
      res *= WGALi*WGALj;
    res *= p_gg(k, a, ni, nj);
  }
  else
  {
    if(include_RSD_GG == 1)
    {
      const double chi_a_min = chi(limits.a_min);
      const double chi_0 = ell/k;
      const double chi_1 = (ell + 1.0)/k;
      if (chi_1 > chi_a_min)  return 0;
      
      const double a_0 = a_chi(chi_0);
      const double a_1 = a_chi(chi_1);
      const double WRSDi =  W_RSD(ell, a_0, a_1, ni);

      res *= (WGALi*b1i + WMAGi*ell_prefactor*bmagi + WRSDi);
      res *= (WGALi*b1i + WMAGi*ell_prefactor*bmagi + WRSDi);
    }
    else
    {
      res *= (WGALi*b1i + WMAGi*ell_prefactor*bmagi);
      res *= (WGALi*b1i + WMAGi*ell_prefactor*bmagi);
    }
    res *= (use_linear_ps ? p_lin(k,a) : Pdelta(k,a));
  }

  double oneloop = 0.0;
  if (1 == nonlinear_bias && 0 == use_linear_ps)
  {
    if (0 == nuisance.IA_code){
      get_FPT_bias();
    }
    const double lnk = log(k);
    double lim[3];
    lim[0] = log(FPTbias.k_min);
    lim[1] = log(FPTbias.k_max);
    lim[2] = (lim[1] - lim[0])/FPTbias.N;

    const double s4 = FPTbias.sigma4; // PT_sigma4(k);

    const double d1d2 = (lnk<lim[0] || lnk>lim[1]) ? 0.0 :
      interpol1d(FPTbias.tab[0], FPTbias.N, lim[0], lim[1], lim[2], lnk);
    
    const double d2d2 = (lnk<lim[0] || lnk>lim[1]) ? 0.0 :
      interpol1d(FPTbias.tab[1], FPTbias.N, lim[0], lim[1], lim[2], lnk) - 2.*s4;
    
    const double d1s2 = (lnk<lim[0] || lnk>lim[1]) ? 0.0 :
      interpol1d(FPTbias.tab[2], FPTbias.N, lim[0], lim[1], lim[2], lnk);
    
    const double d2s2 = (lnk<lim[0] || lnk>lim[1]) ? 0.0 :
      interpol1d(FPTbias.tab[3], FPTbias.N, lim[0], lim[1], lim[2], lnk) - 4. / 3.*s4;
    
    const double s2s2 = (lnk<lim[0] || lnk>lim[1]) ? 0.0 :
      interpol1d(FPTbias.tab[4], FPTbias.N, lim[0], lim[1], lim[2], lnk) - 8. / 9. * s4;

    const double d1p3 = (lnk<lim[0] || lnk>lim[1]) ? 0.0 :
      interpol1d(FPTbias.tab[5], FPTbias.N, lim[0], lim[1], lim[2], lnk);

    const double PK = (use_linear_ps ? p_lin(k,a) : Pdelta(k,a));

    const double growfac_a = growfac(a);
    const double g4 = growfac_a*growfac_a*growfac_a*growfac_a;
    const double b2 = gb2(z, ni);
    const double bs2 = gbs2(z, ni);
    const double b3 = gb3(z, ni);
    const double bk = gbK(z, ni);
    
    oneloop = 1.0;
    oneloop *= WGALi*WGALi;
    oneloop *= g4*(b1i*b2*d1d2 + 0.25*b2*b2 * d2d2 +
      			   b1i*bs2*d1s2 + 0.5*b2*bs2 * d2s2 +
      			   0.25*bs2*bs2*s2s2 + b1i*b3*d1p3) + (2*b1i*bk * k*k * PK);
  }
  return (res +  oneloop)*chidchi.dchida/(fK*fK);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double C_gg_tomo_limber_linpsopt_nointerp(
    const double l, 
    const int ni, 
    const int nj,
    const int use_linear_ps,
    const int init
  ) // We need that for limber calculation
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  
  if (ni < 0 || ni > redshift.clustering_nbin - 1 || 
      nj < 0 || nj > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number (ni, nj) = [%d,%d]", ni, nj);
    exit(1);
  }
  if (ni != nj) {
    log_fatal("cross-tomography (ni,nj) = (%d,%d) bins not supported", ni, nj);
    exit(1);
  }
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 96 : 
                         (1 == hdi) ? 128 : 
                         (2 == hdi) ? 256 : 
                         (3 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[5] = {ni, nj, l, use_linear_ps, has_b2_galaxies()};
  
  const double amin = amin_lens(ni);
  const double amax = amax_lens(ni);
  if (!(amin>0) || !(amin<1) || !(amax>0) || !(amax<1)) {
    log_fatal("0 < amin/amax < 1 not true"); exit(1);
  }
  if (!(amin < amax)) {
    log_fatal("amin < amax not true"); exit(1);
  }
  double res = 0.0;
  if (1 == init) {
    int_for_C_gg_tomo_limber(amin, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_gg_tomo_limber;
    res = gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double C_gg_tomo_limber_nointerp(
    const double l, 
    const int ni, 
    const int nj,
    const int init
  )
{
  return  C_gg_tomo_limber_linpsopt_nointerp(l, ni, nj, 0, init);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double C_gg_tomo_limber(const double l, const int ni, const int nj)
{ // cross redshift bin not supported
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static int nell;
  static int NSIZE;
  static double lim[3];

  if (NULL == table || fdiff2(cache[3], Ntable.random)) {
    nell   = Ntable.N_ell;
    NSIZE  = redshift.clustering_nbin;
    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0]) / ((double) nell - 1.0);
    if (table != NULL) free(table);
    table = (double**) malloc2d(NSIZE, nell);
  }

  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_clustering) ||
      fdiff2(cache[2], redshift.random_clustering) ||
      fdiff2(cache[3], Ntable.random) ||
      fdiff2(cache[4], nuisance.random_galaxy_bias))
  {
    for (int k=0; k<NSIZE; k++)  { // init static variables
      (void) C_gg_tomo_limber_nointerp(exp(lim[0]), k, k, 1);
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int k=0; k<NSIZE; k++) {
      for (int i=0; i<nell; i++) {
        const double lx = exp(lim[0] + i*lim[2]);
        table[k][i] = C_gg_tomo_limber_nointerp(lx, k, k, 0);
      }
    }
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_clustering;
    cache[2] = redshift.random_clustering;
    cache[3] = Ntable.random;
    cache[4] = nuisance.random_galaxy_bias;
  }

  if (ni < 0 || ni > redshift.clustering_nbin - 1 || 
      nj < 0 || nj > redshift.clustering_nbin - 1)
  {
    log_fatal("error in selecting bin number (ni,nj) = [%d,%d]",ni,nj); exit(1);
  }
  if (ni != nj) {
    log_fatal("cross-tomography not supported"); exit(1);
  }
  const double lnl = log(l);
  if (lnl < lim[0]) {
    log_warn("l = %e < lmin = %e. Extrapolation adopted", l, exp(lim[0]));
  }
  if (lnl > lim[1]) {
    log_warn("l = %e > lmax = %e. Extrapolation adopted", l, exp(lim[1]));
  }
  const int q = ni; // cross redshift bin not supported; not using N_CL(ni, nj)
  if (q < 0 || q > NSIZE - 1) {
    log_fatal("internal logic error in selecting bin number");
    exit(1);
  }  
  return interpol1d(table[q], nell, lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_C_gk_tomo_limber(double a, void* params)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  double* ar = (double*) params;

  const int nl = (int) ar[0];
  if (nl < 0 || nl > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", nl); exit(1);
  }
  const double l = ar[1];
  const int nonlinear_bias = ar[2];

  const double ell = l + 0.5;
  struct chis chidchi = chi_all(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const double fK = chidchi.chi;
  const double k = ell/fK;
  const double z = 1./a - 1.;

  const double b1   = gb1(z, nl);
  const double bmag = gbmag(z, nl);

  const double WK = W_k(a, fK);
  const double WGAL = W_gal(a, nl, hoverh0);
  const double WMAG = W_mag(a, fK, nl);


  const double ell_prefactor = l*(l + 1.)/(ell*ell); // prefactor correction (1812.05995 eqs 74-79)

  double res = WK; 

  if (include_HOD_GX == 1)
  {
    if (include_RSD_GK == 1) {
      log_fatal("RSD not implemented with (HOD = TRUE)");
      exit(1);
    }
    else { 
      res *= WGAL;
    }
    res *= p_gm(k, a, nl);
  }
  else
  {
    if (include_RSD_GK == 1) {
      const double chi_0 = ell/k;
      const double chi_1 = (ell+1.)/k;
      const double a_0 = a_chi(chi_0);
      const double a_1 = a_chi(chi_1);
      const double WRSD = W_RSD(ell, a_0, a_1, nl);

      res *= WGAL*b1 + WMAG*ell_prefactor*bmag + WRSD;
    }
    else {
      res *= WGAL*b1 + WMAG*ell_prefactor*bmag;
    }
    const double PK = Pdelta(k,a);
    res *= PK;
  }
  double oneloop = 0.0;
  if (1 == nonlinear_bias) {
    if (0 == nuisance.IA_code){
      get_FPT_bias();
    }
    const double growfac_a = growfac(a);
    const double g4 = growfac_a*growfac_a*growfac_a*growfac_a;

    const double lnk = log(k);
    double lim[3];
    lim[0] = log(FPTbias.k_min);
    lim[1] = log(FPTbias.k_max);
    lim[2] = (lim[1] - lim[0])/FPTbias.N;

    const double d1d2 = (lnk<lim[0] || lnk>lim[1]) ? 0.0 :
      interpol1d(FPTbias.tab[0], FPTbias.N, lim[0], lim[1], lim[2], lnk);
    
    const double d1s2 = (lnk<lim[0] || lnk>lim[1]) ? 0.0 :
      interpol1d(FPTbias.tab[2], FPTbias.N, lim[0], lim[1], lim[2], lnk);

    const double d1p3 = (lnk<lim[0] || lnk>lim[1]) ? 0.0 :
      interpol1d(FPTbias.tab[5], FPTbias.N, lim[0], lim[1], lim[2], lnk);

    const double PK = Pdelta(k,a);

    const double b2 = gb2(z, nl);
    const double bs2 = gbs2(z, nl);
    const double b3 = gb3(z, nl);
    const double bk = gbK(z, nl);
    
    oneloop = WK;
    oneloop *= WGAL;
    oneloop *= g4*(0.5*b2*d1d2 + 0.5*bs2*d1s2 + 0.5*b3*d1p3) + (bk * k * k * PK);
  }
  return ((res + oneloop)*chidchi.dchida/(fK*fK))*ell_prefactor;
}

double C_gk_tomo_limber_nointerp(const double l, const int ni, const int init)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 96 : 
                         (1 == hdi) ? 128 : 
                         (2 == hdi) ? 256 : 
                         (3 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[3] = {(double) ni, l, has_b2_galaxies()};
  
  const double amin = amin_lens(ni);
  const double amax = amax_lens(ni);
  if (!(amin>0) || !(amin<1) || !(amax>0) || !(amax<1)) {
    log_fatal("0 < amin/amax < 1 not true");
    exit(1);
  }
  if (!(amin < amax)) {
    log_fatal("amin < amax not true");
    exit(1);
  }

  double res = 0.0;
  if (1 == init) {
    res = int_for_C_gk_tomo_limber(amin, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_gk_tomo_limber;
    res = gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;
}

double C_gk_tomo_limber(const double l, const int ni)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static int nell;
  static double lim[3];

  if (NULL == table || fdiff2(cache[3], Ntable.random)) {
    nell = Ntable.N_ell;
    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_ell - 1.0);
    if (table != NULL) free(table);
    table = (double**) malloc2d(redshift.clustering_nbin, Ntable.N_ell);
  }

  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_clustering) ||
      fdiff2(cache[2], redshift.random_clustering) ||
      fdiff2(cache[3], Ntable.random) ||
      fdiff2(cache[4], nuisance.random_galaxy_bias))
  {
    for (int k=0; k<redshift.clustering_nbin; k++) { // init static variables
      (void) C_gk_tomo_limber_nointerp(exp(lim[0]), k, 1);
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int k=0; k<redshift.clustering_nbin; k++) {
      for (int i=0; i<nell; i++)  {
        const double lx = exp(lim[0] + i*lim[2]);
        table[k][i] = C_gk_tomo_limber_nointerp(lx, k, 0);
      }
    }
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_clustering;
    cache[2] = redshift.random_clustering;
    cache[3] = Ntable.random;
    cache[4] = nuisance.random_galaxy_bias;
  }
  
  if (ni < -1 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }
  const double lnl = log(l);
  if (lnl < lim[0]) {
    log_warn("l = %e < lmin = %e. Extrapolation adopted", l, exp(lim[0]));
  }
  if (lnl > lim[1]) {
    log_warn("l = %e > lmax = %e. Extrapolation adopted", l, exp(lim[1]));
  }
  const int q =  ni; 
  if (q < 0 || q > redshift.clustering_nbin - 1) {
    log_fatal("internal logic error in selecting bin number");
    exit(1);
  }
  return interpol1d(table[q], nell, lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_C_ks_tomo_limber(double a, void* params)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }

  double* ar = (double*) params;
  const int ni = (int) ar[0];
  if (ni < -1 || ni > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  const double l = ar[1];  
  const double ell = l + 0.5;  
  const double growfac_a = growfac(a);
  struct chis chidchi = chi_all(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const double fK = chidchi.chi;
  const double k = ell/fK;
  const double PK = Pdelta(k,a);

  const double WK1 = W_kappa(a, fK, ni);
  const double WK2 = W_k(a, fK);

  const double ell_prefactor1 = l*(l + 1.)/(ell*ell); // prefactor correction (1812.05995 eqs 74-79)
  const double tmp = (l - 1.)*l*(l + 1.)*(l + 2.);    // prefactor correction (1812.05995 eqs 74-79)
  const double ell_prefactor2 = (tmp > 0) ? sqrt(tmp)/(ell*ell) : 0.0; 

  const double A_Z1 = IA_A1_Z1(a, growfac_a, ni);
  const double WS1  = W_source(a, ni, hoverh0) * A_Z1;

  const double res = (WK1 - WS1)*WK2;

  return (res*PK*chidchi.dchida/(fK*fK))*ell_prefactor1*ell_prefactor2;
}

double C_ks_tomo_limber_nointerp(const double l, const int ni, const int init)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;  

  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  if (NULL ==  w || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 96 : 
                         (1 == hdi) ? 128 : 
                         (2 == hdi) ? 256 : 
                         (3 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }
  
  double ar[3] = {(double) ni, l};
  const double amin = amin_source(ni);
  const double amax = amax_source(ni);
  if (!(amin>0) || !(amin<1) || !(amax>0) || !(amax<1)) {
    log_fatal("0 < amin/amax < 1 not true");
    exit(1);
  }
 
  double res = 0.0;
  if (init == 1) {
    res = int_for_C_ks_tomo_limber(amin, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_ks_tomo_limber;
    res = gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;  
}

double C_ks_tomo_limber(double l, int ni)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static int nell;
  static double lim[3];

  if (NULL == table || fdiff2(cache[4], Ntable.random)) {
    nell = Ntable.N_ell;
    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_ell - 1.0);

    if (table != NULL) free(table);
    table = (double**) malloc2d(redshift.shear_nbin, Ntable.N_ell);
  }

  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    for (int k=0; k<redshift.shear_nbin; k++) {  // init static vars
      (void) C_ks_tomo_limber_nointerp(exp(lim[0]), k, 1);
    } 
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int k=0; k<redshift.shear_nbin; k++) {
      for (int i=0; i<Ntable.N_ell; i++) {
        const double lx = exp(lim[0] + i*lim[2]);
        table[k][i] = C_ks_tomo_limber_nointerp(lx, k, 0);
      }
    }
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
  if (lnl < lim[0]) {
    log_warn("l = %e < lmin = %e. Extrapolation adopted", l, exp(lim[0]));
  }
  if (lnl > lim[1]) {
    log_warn("l = %e > lmax = %e. Extrapolation adopted", l, exp(lim[1]));
  }
  const int q =  ni; 
  if (q < 0 || q > redshift.shear_nbin - 1) {
    log_fatal("internal logic error in selecting bin number");
    exit(1);
  }
  return interpol1d(table[q], Ntable.N_ell, lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_C_kk_limber(double a, void* params)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  
  double* ar = (double*) params;
  const double l = ar[0];
  
  struct chis chidchi = chi_all(a);
  const double ell = l + 0.5;
  const double fK = chidchi.chi;
  const double k = ell/fK;
  const double WK = W_k(a, fK);
  const double PK = Pdelta(k,a);
  
  const double ell_prefactor = l*(l + 1.0)/(ell*ell);  // prefac correction (1812.05995 eqs 74-79)

  return WK*WK*PK*(chidchi.dchida/(fK*fK))*ell_prefactor*ell_prefactor;
}

double C_kk_limber_nointerp(const double l, const int init)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 96 : 
                         (1 == hdi) ? 128 : 
                         (2 == hdi) ? 256 : 
                         (3 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[1] = {l};
  const double amin = limits.a_min*(1. + 1.e-5);
  const double amax = 0.99999;
  
  double res = 0.0;
  if (init == 1) {
    res = int_for_C_kk_limber(amin, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_kk_limber;
    res = gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;
}

double C_kk_limber(const double l)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* table = NULL;
  static double lim[3];

  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_ell - 1.0);
    if (table != NULL) free(table);
    table = (double*) malloc1d(Ntable.N_ell);
  }
  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random)) {
    (void) C_kk_limber_nointerp(exp(lim[0]), 1); // init static vars    
    #pragma omp parallel for schedule(static,1)
    for (int i=0; i<Ntable.N_ell; i++) {
      const double lx = exp(lim[0] + i*lim[2]);
      table[i] = C_kk_limber_nointerp(lx, 0);
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }

  const double lnl = log(l);
  if (lnl < lim[0]) {
    log_warn("l = %e < lmin = %e. Extrapolation adopted", l, exp(lim[0]));
  }
  if (lnl > lim[1]) {
    log_warn("l = %e > lmax = %e. Extrapolation adopted", l, exp(lim[1]));
  }
  return interpol1d(table, Ntable.N_ell, lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_C_gy_tomo_limber(double a, void* params)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  double* ar = (double*) params;
  const int nl = (int) ar[0];
  if (nl < 0 || nl > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", nl); exit(1);
  }
  const double l = ar[1];

  const double ell = l + 0.5;
  struct chis chidchi = chi_all(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const double fK = chidchi.chi;
  const double k = ell/fK;
  const double z = 1./a - 1.;

  const double b1   = gb1(z, nl);
  const double bmag = gbmag(z, nl);

  const double WY = W_y(a);
  const double WGAL = W_gal(a, nl, hoverh0);
  const double WMAG = W_mag(a, fK, nl);

  const double ell_prefactor = l*(l + 1.)/(ell*ell); // prefactor correction (1812.05995 eqs 74-79)

  double res = WY;

  if (include_HOD_GX == 1)
  {
    if (include_RSD_GY == 1) {
      log_fatal("RSD not implemented with (HOD = TRUE)"); exit(1);
    }
    else { 
      log_fatal("(HOD = TRUE) not implemented"); exit(1);
    }
  }
  else
  {
    if (include_RSD_GY == 1)
    {
      log_fatal("RSD not implemented");
      exit(1);
    }
    else
      res *= WGAL*b1 + WMAG*ell_prefactor*bmag;

    const double PK = p_my(k, a);
    res *= PK;
  }
  return res*chidchi.dchida/(fK*fK);
}

double C_gy_tomo_limber_nointerp(const double l, const int ni, const int init)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  if (w == NULL || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 96 : 
                         (1 == hdi) ? 128 : 
                         (2 == hdi) ? 256 : 
                         (3 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[2] = {(double) ni, l};
  const double amin = amin_lens(ni);
  const double amax = 0.99999;

  double res = 0.0;
  if (init == 1)
    res = int_for_C_gy_tomo_limber(amin, (void*) ar);
  else
  {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_gy_tomo_limber;
    res =  gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;
}

double C_gy_tomo_limber(double l, int ni)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double lim[3];

  if (table == NULL || fdiff2(cache[4], Ntable.random))
  {
    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2]   = (lim[1] - lim[0])/((double) Ntable.N_ell - 1.0);

    if (table != NULL) free(table);
    table = (double**) malloc2d(redshift.clustering_nbin, Ntable.N_ell);
  }

  if (fdiff2(cache[1], cosmology.random) || 
      fdiff2(cache[2], nuisance.random_photoz_clustering) ||
      fdiff2(cache[3], redshift.random_clustering) ||
      fdiff2(cache[4], Ntable.random) ||
      fdiff2(cache[5], nuisance.random_galaxy_bias))
  {
    { // init static variables inside the C_XY_limber_nointerp function
      (void) C_gy_tomo_limber_nointerp(exp(lim[0]), 0, 1);
    }    
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int k=0; k<redshift.clustering_nbin; k++) {
      for (int i=0; i<Ntable.N_ell; i++) {
        table[k][i]= C_gy_tomo_limber_nointerp(exp(lim[0] + i*lim[2]), k, 0);
      }
    }
    cache[1] = cosmology.random;
    cache[2] = nuisance.random_photoz_clustering;
    cache[3] = redshift.random_clustering;
    cache[4] = Ntable.random;
    cache[5] = nuisance.random_galaxy_bias;
  }

  const int q =  ni; 
  if (q < 0 || q > redshift.clustering_nbin - 1)
  {
    log_fatal("internal logic error in selecting bin number");
    exit(1);
  } 
  
  const double lnl = log(l);
  if (lnl < lim[0])
    log_warn("l = %e < lmin = %e. Extrapolation adopted", l, exp(lim[0]));
  if (lnl > lim[1])
    log_warn("l = %e > lmax = %e. Extrapolation adopted", l, exp(lim[1]));

  return interpol1d(table[q], Ntable.N_ell, lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_C_ys_tomo_limber(double a, void* params)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }

  double* ar = (double*) params;
  const int ni = (int) ar[0];
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  const double l = ar[1];
  
  const double ell = l + 0.5;
  const double growfac_a = growfac(a);
  struct chis chidchi = chi_all(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
  const double fK = chidchi.chi;
  const double k  = ell/fK;
  
  const double PK = p_my(k, a);

  const double WK1 = W_kappa(a, fK, ni);
  const double WY  = W_y(a);

  const double tmp = (l - 1.0)*l*(l + 1.0)*(l + 2.0); // prefactor correction (1812.05995 eqs 74-79)
  const double ell_prefactor2 = (tmp > 0) ? sqrt(tmp)/(ell*ell) : 0.0;

  const double A_Z1 = IA_A1_Z1(a, growfac_a, ni);
  const double WS1  = W_source(a, ni, hoverh0) * A_Z1;

  const double res = (WK1 - WS1)*WY;

  return res*PK*(chidchi.dchida/(fK*fK))*ell_prefactor2;
}

double C_ys_tomo_limber_nointerp(const double l, const int ni, const int init)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (ni < 0 || ni > redshift.shear_nbin - 1)
  {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  } 

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = 80 + 50 * abs(Ntable.high_def_integration);
    if (w != NULL) {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[2] = {(double) ni, l};
  const double amin = amin_source(ni);
  const double amax = 0.99999;

  double res = 0.0;
  if (init == 1)
    res = int_for_C_ys_tomo_limber(amin, (void*) ar);
  else
  {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_ys_tomo_limber;
    res =  gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;
}

double C_ys_tomo_limber(double l, int ni)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double lim[3];

  if (table == NULL || fdiff2(cache[4], Ntable.random))
  {
    if (table != NULL) free(table);
    table = (double**) malloc2d(redshift.shear_nbin, Ntable.N_ell);

    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_ell - 1.0);
  }

  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    { // init static variables inside the C_XY_limber_nointerp function
      (void) C_ys_tomo_limber_nointerp(exp(lim[0]), 0, 1);
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int k=0; k<redshift.shear_nbin; k++) {
      for (int i=0; i<Ntable.N_ell; i++) {
        table[k][i] = C_ys_tomo_limber_nointerp(exp(lim[0] + i*lim[2]), k, 0);
      }
    } 
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear;
    cache[4] = Ntable.random;
  }
  
  if (ni < 0 || ni > redshift.shear_nbin - 1)
  {
    log_fatal("error in selecting bin number ni = %d (max %d)", ni, 
      redshift.shear_nbin);
    exit(1);
  }

  const int q =  ni; 
  if (q < 0 || q > redshift.shear_nbin - 1)
  {
    log_fatal("internal logic error in selecting bin number");
    exit(1);
  } 
  
  const double lnl = log(l);
  if (lnl < lim[0])
    log_warn("l = %e < lmin = %e. Extrapolation adopted", l, exp(lim[0]));
  if (lnl > lim[1])
    log_warn("l = %e > lmax = %e. Extrapolation adopted", l, exp(lim[1]));

  return interpol1d(table[q], Ntable.N_ell, lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_C_ky_limber(double a, void* params)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }

  double *ar = (double*) params;
  const double l = ar[0];

  const double ell = l + 0.5;
  struct chis chidchi = chi_all(a);
  const double fK = chidchi.chi;
  const double k = ell/fK;
  
  const double PK = p_my(k, a);
  const double WK = W_k(a, fK);
  const double WY = W_y(a);

  const double ell_prefactor = l*(l + 1.0)/(ell*ell); // prefactor correction (1812.05995 eqs 74-79)

  return (WK*WY*PK*chidchi.dchida/(fK*fK))*ell_prefactor;
}

double C_ky_limber_nointerp(const double l, const int init)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  
  if (w == NULL || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = 80 + 50 * abs(Ntable.high_def_integration);
    if (w != NULL)  {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[1] = {l};
  const double amin = limits.a_min_hm;
  const double amax = 1.0 - 1.e-5;

  double res = 0.0;
  if (init == 1)
    res = int_for_C_ky_limber(amin, (void*) ar);
  else
  {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_ky_limber;
    res =  gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;
}

double C_ky_limber(double l)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* table = NULL;
  static double lim[3];

  if (table == NULL || fdiff2(cache[1], Ntable.random))
  {
    if (table != NULL) free(table);
    table = (double*) malloc1d(Ntable.N_ell);

    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_ell - 1.0);
  }

  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random))
  {
    { // init static variables inside the C_XY_limber_nointerp function
      (void) C_ky_limber_nointerp(exp(lim[0]), 1);
    }
    #pragma omp parallel for schedule(static,1)
    for (int i=0; i<Ntable.N_ell; i++) {
      table[i] = log(C_ky_limber_nointerp(exp(lim[0] + i*lim[2]), 0));
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
  
  const double lnl = log(l);
  if (lnl < lim[0])
    log_warn("l = %e < lmin = %e. Extrapolation adopted", l, exp(lim[0]));
  if (lnl > lim[1])
    log_warn("l = %e > l_max = %e. Extrapolation adopted", l, exp(lim[1]));

  return exp(interpol1d(table, Ntable.N_ell, lim[0], lim[1], lim[2], lnl));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_C_yy_limber(double a, void *params)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  double* ar = (double*) params;
  const double l = ar[0];

  const double ell = l + 0.5;
  struct chis chidchi = chi_all(a);
  const double fK = chidchi.chi;
  const double k  = ell/fK;

  const double PK = p_yy(k, a);
  
  const double WY = W_y(a);

  return WY*WY*PK*chidchi.dchida/(fK*fK);
}

double C_yy_limber_nointerp(const double l, const int init)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = 80 + 50 * abs(Ntable.high_def_integration);
    if (w != NULL) {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[2] = {l};
  const double amin = limits.a_min;
  const double amax = 1.0 - 1.e-5;

  double res = 0.0;
  if (init == 1) {
    res = int_for_C_yy_limber(amin, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_C_yy_limber;
    res = gsl_integration_glfixed(&F, amin, amax, w);
  }
  return res;
}

double C_yy_limber(double l)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* table = NULL;
  static double lim[3];

  if (table == NULL || fdiff2(cache[1], Ntable.random))
  {
    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1.0);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_ell - 1.0);

    if (table != NULL) free(table);
    table = (double*) malloc1d(Ntable.N_ell);
  }

  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random))
  {
    { // init static variables inside the C_XY_limber_nointerp function
      (void) C_yy_limber_nointerp(exp(lim[0]), 1);
    }
    #pragma omp parallel for schedule(static,1)
    for (int i=0; i<Ntable.N_ell; i++) {
      table[i] = C_yy_limber_nointerp(exp(lim[0] + i*lim[2]), 0);
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }

  const double lnl = log(l);
  if (lnl < lim[0]) {
    log_warn("l = %e < lmin = %e. Extrapolation adopted", l, exp(lim[0]));
  }
  if (lnl > lim[1]) {
    log_warn("l = %e > lmax = %e. Extrapolation adopted", l, exp(lim[1]));
  }
  return interpol1d(table, Ntable.N_ell, lim[0], lim[0], lim[2], lnl);
}

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Non Limber (Angular Power Spectrum)
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// next_fft_size: Round up n to the next "FFT-friendly" number whose only
// prime factors are 2, 3, 5, or 7.
//
// The FFT algorithm works by recursively splitting a size-N transform into
// smaller sub-transforms based on N's prime factorization. 
// FFTW has highly optimized, SIMD-vectorized "codelets" for small prime factors 
// (2, 3, 5, 7), making these splits very fast.
//
// When N has a large prime factor p, FFTW cannot split it efficiently and
// must fall back to generic algorithms, which are slower and cannot be vectorized. 

// For example:
//   N = 10240 = 2^11 × 5  → 11 radix-2 stages + 1 radix-5 stage, all fast
//   N = 10201 = 101 × 101 → two levels of prime-101 sub-transforms, slow
//
// Padding to a slightly larger FFT-friendly size does not affect the convolution
// result: the extra elements are zeros, and we read the same output indices 
// regardless of the padded size.
// ---------------------------------------------------------------------------
static long next_fft_size(long n) {
  while (1) {
    long m = n;
    while (m % 2 == 0) m /= 2;
    while (m % 3 == 0) m /= 3;
    while (m % 5 == 0) m /= 5;
    while (m % 7 == 0) m /= 7;
    if (m == 1) return n;
    n++;
  }
  return n;
}


typedef struct config 
{
  double nu;
  double c_window_width;
  int derivative;
  long N_pad;
  long N_extrap_low;
  long N_extrap_high;
} config;

void cfftlog_ells_p1(
  double* const x,
  double* const* const* const fx,
  int const Nx,
  config* const cfg,
  fftw_complex* const* const toutfwd,
  double* const* const eta_m,
  int const N[][3],
  int const Nmax,
  int const SIZE1,
  int const SIZE2
)
{
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------  
  double*** fb = (double***) malloc3d(SIZE1, SIZE2, Nmax); // biased input func
  #pragma omp parallel for collapse(2) schedule(static,1)
  for(int i=0; i<SIZE1; i++) {
    for(int j=0; j<SIZE2; j++) {
      for(int k=0; k<N[j][0]; k++) {
        fb[i][j][k] = 0.; // padding
      }
      for(int k=N[j][0]; k<N[j][0]+N[j][1]; k++) {
        const int q = k - N[j][0];
        if (q < 0 || q > Nx-1) {
          log_fatal("logical error on the array indexes"); exit(1);
        }
        fb[i][j][k] = fx[i][j][q] / pow(x[q], cfg[j].nu) ;
      }
      for(int k=N[j][0]+N[j][1]; k<N[j][2]; k++) {
        fb[i][j][k] = 0.; // padding
      }
    }
  }
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------  
  fftw_plan planf[SIZE2];
  for (int j=0; j<SIZE2; j++) {
    planf[j] = fftw_plan_dft_r2c_1d(N[j][2], 
                                    fb[0][j], 
                                    toutfwd[0*SIZE2+j],
                                    FFTW_ESTIMATE);
  }
  #pragma omp parallel for collapse(2) schedule(static,1)
  for(int i=0; i<SIZE1; i++) {
    for(int j=0; j<SIZE2; j++) {
      fftw_execute_dft_r2c(planf[j], fb[i][j], toutfwd[i*SIZE2+j]);
      // c_window_cfft function begins -----------------------------------------
      const double cww = cfg[j].c_window_width;
      if( !(cww > 0) || !(cww < 1)) {
        log_fatal("improper window width"); exit(1);
      }
      const int halfN = N[j][2]/2;
      const int kmax = (int) (halfN * cww);
      for(int k=0; k<(kmax+1); k++) { // window for right-side
        const double W = (double)(k)/kmax - sin(2.*M_PI*k/kmax)/(2.*M_PI);
        toutfwd[i*SIZE2+j][N[j][2]/2-k] *= W;
      }
    }
  }
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  const double dlnx = log(x[1]/x[0]);
  for(int j=0; j<SIZE2; j++) {
    #pragma omp parallel for schedule(static,1)
    for(int q=0; q<N[j][2]/2+1; q++) {
      eta_m[j][q] = (2.0*M_PI/(dlnx * N[j][2])) * q;  
    }
  }
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  for (int j=0; j<SIZE2; j++) {
    fftw_destroy_plan(planf[j]);
  }
  free((void*) fb);
}

void cfftlog_ells_p2(
  double* const x,
  int const Nx,
  config* const cfg,
  int const LMAX,
  double* const* const* const y,
  double* const* const* const* const Fy,
  fftw_complex* const* const toutfwd,
  double* const* const eta_m,
  int const N[][3],
  int const Nmax,
  int const ks, // k start
  int const ke, // k end
  const int* const converged,
  int const SIZE1,
  int const SIZE2
) 
{
  static int cache[MAX_SIZE_ARRAYS];
  static fftw_complex** outfwd = NULL;
  static double** outbcw = NULL;
  static double complex*** gl = NULL;
  static fftw_plan* planb = NULL;
  static double* base_j = NULL;
  static double pfac[] = {0.99999999999980993227684700473478,
                           676.520368121885098567009190444019,
                          -1259.13921672240287047156078755283,
                           771.3234287776530788486528258894,
                          -176.61502916214059906584551354,
                           12.507343278686904814458936853,
                          -0.13857109526572011689554707,
                           9.984369578019570859563e-6,
                           1.50563273514931155834e-7};

  if (SIZE1 < 1 || SIZE2 < 1) {
    log_fatal("SIZE1 and SIZE2 must be >= 1"); exit(1);
  }

  const int kmax = (ke < LMAX) ? ke : LMAX;
  const int BLOCK = ke - ks;
  const int NTHREADS = omp_get_max_threads();

  const double sqrtpi = sqrt(M_PI);
  const double ln2 = log(2.);
  const double x0   = x[0];
  const double dlnx = log(x[1]/x[0]);
  const double complex clogpi = clog(M_PI);
  const double ln2pio2 = 0.5*log(2*M_PI);

  if (outfwd == NULL   || 
      NTHREADS != cache[0] ||
      Nmax != cache[1] || 
      BLOCK > cache[2] || 
      SIZE2 != cache[3]) 
  {
    if (gl != NULL) free((void*) gl);
    gl = (double complex***) malloc3d_complex(SIZE2, BLOCK, Nmax/2+1);
    
    if (outfwd != NULL) free((void*) outfwd);
    outfwd = (fftw_complex**) malloc2d_fftwc(NTHREADS, Nmax/2+1);
    
    if (outbcw != NULL) free((void*) outbcw);
    outbcw = (double**) malloc2d(NTHREADS, Nmax);

    if (planb != NULL) {
      for (int j=0; j<cache[3]; j++) fftw_destroy_plan(planb[j]);
      free(planb);
    }
    planb = (fftw_plan*) malloc(sizeof(fftw_plan)*SIZE2);
    for(int j=0; j<SIZE2; j++) {
      planb[j] = fftw_plan_dft_c2r_1d(N[j][2],
                                      outfwd[0], 
                                      outbcw[0], 
                                      FFTW_ESTIMATE);
    }

    if (base_j != NULL) free((void*) base_j);
    base_j = (double*) malloc(sizeof(double) * SIZE2);
    for(int j=0; j<SIZE2; j++) {
      base_j[j] = x0 / exp(2 * N[j][0] * dlnx);
    }

    cache[0] = NTHREADS;
    cache[1] = Nmax;
    cache[2] = BLOCK; 
    cache[3] = SIZE2;
  }

  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------  
  
  #pragma omp parallel for collapse(2) schedule(static,1)
  for(int i=0; i<SIZE1; i++) {
    for(int q=0; q<Nx; q++) { // q < Nx
      for (int k=ks; k<kmax; k++) {
        y[i][k][q] = (k + 1.) / x[Nx -1 -q];
      }
    }
  } 

  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  /* Gamma function recurrence Γ(a+1) = a·Γ(a)
    For case 0:
      gl[k] = exp(z·ln2) · Γ(½(k+z)) / Γ(½(3+k-z))
      When k → k+2, both arguments shift by 1:
      Γ(½(k+2+z)) = Γ(½(k+z) + 1) = ½(k+z) · Γ(½(k+z))
      Γ(½(5+k-z)) = Γ(½(3+k-z) + 1) = ½(3+k-z) · Γ(½(3+k-z))
    So:
      gl[k+2] / gl[k] = ½(k+z) / ½(3+k-z) = (k+z) / (k+3-z)
  */
  for(int j=0; j<SIZE2; j++) {
    const double nu = cfg[j].nu;
    switch(cfg[j].derivative) 
    { 
      case 0: 
      {
        const int ks2 = (ks + 2 < ke) ? ks + 2 : ke;

        #pragma omp parallel for collapse(2) schedule(static,1)
        for (int k=ks; k<ks2; k++) {
          for(int q=0; q<N[j][2]/2+1; q++) 
          {
            const double complex z = nu + I*eta_m[j][q];
            double complex part1;
            {
              const double complex a = 0.5*(k + z);
              if(creal(a) < 0.5) {
                double complex tmp = pfac[0];
                for(int w=1; w<9; w++) tmp += pfac[w] / ((-a) + w);
                const double complex t = (-a) + 7.5;
                part1 = clogpi - clog(csin(M_PI*a)) - 
                        (ln2pio2 + ((-a) + 0.5)*clog(t) - t + clog(tmp));
              }
              else {
                double complex tmp = pfac[0];
                for(int w=1; w<9; w++) tmp += pfac[w] / ((a-1) + w);
                const double complex t = (a-1) + 7.5;
                part1 = ln2pio2 + ((a-1) + 0.5)*clog(t) - t + clog(tmp);
              }
            }
            double complex part2;
            {
              const double complex a = 0.5*(3 + k - z);
              if(creal(a) < 0.5) {
                double complex tmp = pfac[0];
                for(int w=1; w<9; w++) tmp += pfac[w] / ((-a) + w);
                const double complex t = (-a) + 7.5;
                part2 = clogpi - clog(csin(M_PI*a)) - 
                        (ln2pio2 + ((-a) + 0.5)*clog(t) - t + clog(tmp));
              }
              else {
                double complex tmp = pfac[0];
                for(int w=1; w<9; w++) tmp += pfac[w] / ((a-1) + w);
                const double complex t = (a-1) + 7.5;
                part2 = ln2pio2 + ((a-1) + 0.5)*clog(t) - t + clog(tmp);
              }
            }
            gl[j][k-ks][q] = cexp(z*ln2 + part1 - part2); 
          }
        }
        #pragma omp parallel for schedule(static,1)
        for (int q = 0; q < N[j][2]/2+1; q++) {
          const double complex z = nu + I * eta_m[j][q];
          for (int k=ks+2; k < ke; k++) {
            gl[j][k-ks][q] = gl[j][k-ks-2][q] * (k-2+z) / (k + 1 - z);
          }
        }
        break;
      }
      case 1: 
      {
        const int ks2 = (ks + 2 < ke) ? ks + 2 : ke;
        #pragma omp parallel for collapse(2) schedule(static,1)
        for (int k = ks; k < ks2; k++) {
          for(int q=0; q<N[j][2]/2+1; q++) {
            const double complex z = nu + I*eta_m[j][q];
            double complex part1;
            {
              const double complex a = 0.5*(k + z - 1.);
              if(creal(a) < 0.5) {
                double complex tmp = pfac[0];
                for(int w=1; w<9; w++) tmp += pfac[w] / ((-a) + w);
                const double complex t = (-a) + 7.5;
                part1 = clogpi - clog(csin(M_PI*a)) - 
                        (ln2pio2 + ((-a) + 0.5)*clog(t) - t + clog(tmp));
              }
              else {
                double complex tmp = pfac[0];
                for(int w=1; w<9; w++) tmp += pfac[w] / ((a-1) + w);
                const double complex t = (a-1) + 7.5;
                part1 = ln2pio2 + ((a-1) + 0.5)*clog(t) - t + clog(tmp);
              }
            }
            double complex part2;
            {
              const double complex a = 0.5*(4 + k - z);
              if(creal(a) < 0.5) {
                double complex tmp = pfac[0];
                for(int w=1; w<9; w++) tmp += pfac[w] / ((-a) + w);
                const double complex t = (-a) + 7.5;
                part2 = clogpi - clog(csin(M_PI*a)) - 
                        (ln2pio2 + ((-a) + 0.5)*clog(t) - t + clog(tmp));
              }
              else {
                double complex tmp = pfac[0];
                for(int w=1; w<9; w++) tmp += pfac[w] / ((a-1) + w);
                const double complex t = (a-1) + 7.5;
                part2 = ln2pio2 + ((a-1) + 0.5)*clog(t) - t + clog(tmp);
              }
            }
            gl[j][k-ks][q] = -(z-1)*cexp((z-1)*ln2 + part1 - part2);
          }
        }
        #pragma omp parallel for schedule(static,1)
        for (int q = 0; q < N[j][2]/2+1; q++) {
          for (int k = ks + 2; k < ke; k++) {
            const double complex z = nu + I * eta_m[j][q];
            gl[j][k-ks][q] = gl[j][k-ks-2][q] * (k - 3 + z) / (k + 2 - z);
          }
        }
        break;
      }
      case 2: 
      {
        const int ks2 = (ks + 2 < ke) ? ks + 2 : ke;
        #pragma omp parallel for collapse(2) schedule(static,1)
        for (int k=ks; k<ks2; k++) {
          for(int q=0; q<N[j][2]/2+1; q++) {
            const double complex z = nu + I*eta_m[j][q];
            double complex part1;
            {
              const double complex a = 0.5*(k + z - 2);
              if(creal(a) < 0.5) {
                double complex tmp = pfac[0];
                for(int w=1; w<9; w++) tmp += pfac[w] / ((-a) + w);
                const double complex t = (-a) + 7.5;
                part1 = clogpi - clog(csin(M_PI*a)) - 
                        (ln2pio2 + ((-a) + 0.5)*clog(t) - t + clog(tmp));
              }
              else {
                double complex tmp = pfac[0];
                for(int w=1; w<9; w++) tmp += pfac[w] / ((a-1) + w);
                const double complex t = (a-1) + 7.5;
                part1 = ln2pio2 + ((a-1) + 0.5)*clog(t) - t + clog(tmp);
              }
            }
            double complex part2;
            {
              const double complex a = 0.5*(5 + k - z);
              if(creal(a) < 0.5) {
                double complex tmp = pfac[0];
                for(int w=1; w<9; w++) tmp += pfac[w] / ((-a) + w);
                const double complex t = (-a) + 7.5;
                part2 = clogpi - clog(csin(M_PI*a)) - 
                        (ln2pio2 + ((-a) + 0.5)*clog(t) - t + clog(tmp));
              }
              else {
                double complex tmp = pfac[0];
                for(int w=1; w<9; w++) tmp += pfac[w] / ((a-1) + w);
                const double complex t = (a-1) + 7.5;
                part2 = ln2pio2 + ((a-1) + 0.5)*clog(t) - t + clog(tmp);
              }
            }
            gl[j][k-ks][q] = (z-1)*(z-2)*cexp((z-2)*ln2+part1-part2);
          }
        }
        #pragma omp parallel for schedule(static,1)
        for (int q = 0; q < N[j][2]/2+1; q++) {
          for (int k = ks + 2; k < ke; k++) {  
            const double complex z = nu + I * eta_m[j][q];
            gl[j][k-ks][q] = gl[j][k-ks-2][q] * (k - 4 + z) / (k + 3 - z);
          }
        }
        break;
      }
    }
  } 
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
#ifndef COSMO2D_NOT_USE_SIMD 
  double** x_pow_nu = (double**) malloc2d(SIZE2, Nx);
  for (int j=0; j<SIZE2; j++) {
    for (int q=0; q<Nx; q++) {
      x_pow_nu[j][q] = pow(x[Nx - 1 - q], cfg[j].nu);
    }
  }
#endif

  for(int i=0; i<SIZE1; i++) {
    if (converged[i]) continue;
    #pragma omp parallel for collapse(2) schedule(static,1)
    for(int j=0; j<SIZE2; j++) {
      for (int k=ks; k<kmax; k++) { 
        const int id = omp_get_thread_num();  
        const double lnbase = log(base_j[j] * y[i][k][0]);    
#ifndef COSMO2D_NOT_USE_SIMD 
        // Explore the fact that the phase eta_m[j][q] is linear in q
        const double delta_phase = -eta_m[j][1] * lnbase;
        double step_re, step_im;
        cosmo_sincos(delta_phase, &step_im, &step_re);
        double phasor_re = 1.0; // Phasor starts at exp(i * 0) = 1 + 0i.
        double phasor_im = 0.0; // Phasor starts at exp(i * 0) = 1 + 0i.
#endif  
        for(int q=0; q<(N[j][2]/2+1); q++) {
          fftw_complex val = toutfwd[i*SIZE2+j][q];
#ifdef COSMO2D_NOT_USE_SIMD 
          const double phase = -eta_m[j][q] * lnbase;
          val *= cos(phase) + I * sin(phase);
#else
          if (q > 0 && (q % 64) == 0) {
            // recompute phasor exactly to prevent drift (numerical error)
            const double exact_phase = -eta_m[j][q] * lnbase;
            cosmo_sincos(exact_phase, &phasor_im, &phasor_re);
          }
          val *= phasor_re + I * phasor_im;
          // Rotate phasor by one step: phasor *= step.
          const double new_re = phasor_re*step_re - phasor_im*step_im;
          const double new_im = phasor_re*step_im + phasor_im*step_re;
          phasor_re = new_re;
          phasor_im = new_im;
#endif
          val *= gl[j][k-ks][q];
          outfwd[id][q] = conj(val);
        }

        fftw_execute_dft_c2r(planb[j], outfwd[id], outbcw[id]);

#ifdef COSMO2D_NOT_USE_SIMD        
        for(int q=0; q<Nx; q++) {
          Fy[i][j][k][q] = outbcw[id][N[j][0]+q] * sqrtpi / 
                           (4.*N[j][2] * pow(y[i][k][q], cfg[j].nu));
        }
#else
        // Using the fact that y[i][k][q] = (k + 1.) / x[Nx -1 -q];
        const double prefactor = 
                    sqrtpi / (4.0 * N[j][2] * pow((double)(k + 1), cfg[j].nu));
        for (int q = 0; q<Nx; q++) {
          Fy[i][j][k][q] = outbcw[id][N[j][0] + q] * prefactor * x_pow_nu[j][q];
        }
#endif
        
      }
    }
  }
#ifndef COSMO2D_NOT_USE_SIMD
  free((void*) x_pow_nu);
#endif
  return;
}

void C_cl_tomo(
    double* const* const Cl,
    double tol
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static int* LMAX = NULL;
  static double* x = NULL;
  static double*** fx= NULL;
  static double*** y = NULL;
  static double**** Fy = NULL;
  static double*** vres = NULL;
  static config cfg[3];
  
  const int nbins = redshift.clustering_nbin; 
  const int nchi  = Ntable.NL_Nchi;
  
  if (NULL == LMAX || 
      NULL == x || 
      NULL == y || 
      NULL == Fy || 
      NULL == fx ||
      fdiff2(cache[0], Ntable.random))
  {
    if (LMAX != NULL) free((void*) LMAX);
    LMAX = (int*) malloc1d_int(nbins);
    if (x != NULL) free((void*) x);
    x =  (double*) malloc1d(nchi);
    if (y != NULL) free((void*) y);
    y  = (double***) malloc3d(nbins, limits.LMAX_NOLIMBER, nchi);
    if (Fy != NULL) free((void*) Fy);
    Fy = (double****) malloc4d(nbins, 3, limits.LMAX_NOLIMBER, nchi);
    if (fx != NULL) free((void*) fx);
    fx = (double***) malloc3d(nbins,3,nchi); 
    if (vres != NULL) free((void*) vres);
    vres = (double***) malloc3d(nbins, limits.LMAX_NOLIMBER, nchi); 
    cfg[0].nu = 1.;
    cfg[0].c_window_width = 0.25;
    cfg[0].derivative = 0;
    cfg[0].N_pad = 200;
    // RSD
    cfg[1].nu = 1.01;
    cfg[1].c_window_width = 0.25;
    cfg[1].derivative = 2;
    cfg[1].N_pad = 500;
    // MAG
    cfg[2].nu = 1.;
    cfg[2].c_window_width = 0.25;
    cfg[2].derivative = 0;
    cfg[2].N_pad = 500;
    cache[0] = Ntable.random;
  }
  const double real_coverH0 = cosmology.coverH0/cosmology.h0;
  const double chi_min = chi(1./(1.0 + 0.002))*real_coverH0; // DIMENSIONELESS
  const double chi_max = chi(1./(1.0 + 4.0))*real_coverH0;   // DIMENSIONELESS
  const double dlnchi  = log(chi_max/chi_min) / ((double) nchi - 1.0);
  const double dlnk    = dlnchi;
  { // INIT: make sure static init variables inside the funcs are defined
    const int i = 0;
    const double chi = chi_min/real_coverH0;
    const double a   = a_chi(chi);
    const double z   = 1. / a - 1.;
    const double fK = chi;
    (void) growfac_all(a);
    (void) hoverh0(a);
    for (int i=0; i<nbins; i++) {
      (void) nz_lens_photoz(z,i);
      (void) W_mag(a,fK,i);
      (void) gb1(z,i);
    }
    (void) gbmag(0.,0); (void) p_lin(0.1, 1.0);
    (void) C_gg_tomo_limber_nointerp((double) 100, 0, 0, 1);
  }
  #pragma omp parallel for schedule(static,1)
  for (int j=0; j<nchi; j++) {
    x[j] = chi_min * exp(dlnchi * j);
    const double chi = x[j]/real_coverH0;
    const double a   = a_chi(chi);
    const double z   = 1. / a - 1.;
    const double hoverh0_a = hoverh0(a);
    const double fK = chi;
    struct growths growfac_a = growfac_all(a);
    const double D = growfac_a.D;
    const double f = growfac_a.f;
    for (int i=0; i<nbins; i++) {  
      if (z < redshift.clustering_zdist_zmin[i] || 
          z > redshift.clustering_zdist_zmax[i]) { 
        fx[i][0][j] = 0.;
        fx[i][1][j] = 0.;
        fx[i][2][j] = 0.;
      }
      else {
        const double pf = nz_lens_photoz(z,i);
        const double WM = W_mag(a, fK, i);
        fx[i][0][j] =  chi*pf*D*hoverh0_a*gb1(z,i);
        fx[i][1][j] = -chi*pf*D*hoverh0_a*f;
        fx[i][2][j] = (WM/fK/(real_coverH0*real_coverH0))*D; // [Mpc^-2] 
      }
    }
  }

  int is_bmag_zero = 1;
  for (int i=0; i<nbins; i++) {
    if (fabs(gbmag(0.,i)) > 1.e-12) {
      is_bmag_zero = 0;
    }    
  }

  const int SIZE2 = (is_bmag_zero == 0) ? 3 : 2;
  int Nmax = 0;
  int N[SIZE2][3];
  for(int j=0; j<SIZE2; j++) {
    N[j][0] = cfg[j].N_pad;
    N[j][1] = nchi;
    // -------------------------------------------------------------------------
    // Round up to the next FFT-friendly even size whose only prime factors
    // are 2, 3, 5, or 7.  The extra elements are zero-padded in cfftlog_ells_p1
    // and do not affect the convolution result.
    // -------------------------------------------------------------------------
    long raw = 2*N[j][0] + N[j][1];
    if (raw % 2) raw++;
    N[j][2] = (int) next_fft_size(raw);
    if (N[j][2] % 2) {
      N[j][2] = (int) next_fft_size(N[j][2] + 1);
    }
    if (N[j][2] > Nmax)
      Nmax = N[j][2];
  }

  fftw_complex** toutfwd = (fftw_complex**) malloc2d_fftwc(nbins*SIZE2, Nmax/2+1);
  
  double** eta_m = (double**) malloc2d(SIZE2, Nmax/2+1);

  cfftlog_ells_p1((double* const) x, 
                  (double* const* const* const) fx, 
                  nchi, 
                  cfg, 
                  (fftw_complex* const* const) toutfwd,
                  (double* const* const) eta_m, 
                  N, 
                  Nmax, 
                  nbins, 
                  SIZE2);

  const int BLOCK = 16;
  int converged[nbins];
  for (int i=0; i<nbins; i++) {
    converged[i] = 0;
    LMAX[i] = limits.LMAX_NOLIMBER;
  } 
  int all_done = 0;
  int ks = 0;

  while (!all_done && ks < limits.LMAX_NOLIMBER)
  {
    if (ks >= limits.LMAX_NOLIMBER - BLOCK) break;
    
    const int ke = (ks + BLOCK < limits.LMAX_NOLIMBER) ? ks + BLOCK : limits.LMAX_NOLIMBER;

    cfftlog_ells_p2((double* const) x,
                     nchi, 
                     cfg, 
                     limits.LMAX_NOLIMBER, 
                     (double* const* const* const) y, 
                     (double* const* const* const* const) Fy, 
                     (fftw_complex* const* const) toutfwd,
                     (double* const* const) eta_m,
                     N,
                     Nmax,
                     ks, 
                     ke,
                     converged,
                     nbins, 
                     SIZE2);
    if (0 != is_bmag_zero) { // this is the case where gbmag = 0 (avoid garbage)
      for (int i=0; i<nbins; i++) { 
        const int kk = (ke < LMAX[i]) ? ke : LMAX[i];
        for (int k=ks; k<kk; k++) {
          for (int q=0; q<nchi; q++) {
            Fy[i][2][k][q] = 0.0;
          }
        }
      }
    }
    
    for (int i=0; i<nbins; i++) {
      if (converged[i] || ks >= LMAX[i]) continue;
      const int kk = (ke < LMAX[i]) ? ke : LMAX[i];

      #pragma omp parallel for collapse(2) schedule(static,1)
      for (int k=ks; k<kk; k++) {
        for (int q=0; q<nchi; q++) {
          const double ell_prefactor = k * (k + 1.);
          const double ty    = y[i][k][q];
          const double k1cH0 = ty*real_coverH0;
          const double F = Fy[i][0][k][q] + Fy[i][1][k][q] + 
                           gbmag(0.,i)*ell_prefactor*Fy[i][2][k][q]/(ty*ty);
          vres[i][k][q] = F*F*(k1cH0*k1cH0*k1cH0)*p_lin(k1cH0,1);
        }
      }
      #pragma omp parallel for
      for (int k=ks; k<kk; k++) {  
#ifdef COSMO2D_NOT_USE_SIMD
        double tcl = 0.0;
        for (int q=0; q<nchi; q++) {
          tcl += vres[i][k][q];
        }
#else
        const double tcl = simd_array_sum(vres[i][k], nchi);
#endif   
        Cl[i][k] = tcl * dlnk * 2. / M_PI + 
                       C_gg_tomo_limber_linpsopt_nointerp((double) k, i, i, 0, 0)
                      -C_gg_tomo_limber_linpsopt_nointerp((double) k, i, i, 1, 0);
      }
      const int L = kk - 1; // check convergeence
      const double dev = Cl[i][L] / C_gg_tomo_limber_nointerp(L, i, i, 0) - 1.0;
      if (fabs(dev) < tol) {
        converged[i] = 1;
        LMAX[i] = kk;
      }
    }

    all_done = 1;
    for (int i=0; i <nbins; i++) {
      if (!converged[i]) all_done = 0;
    }
    ks = ke;
  }

  for (int i=0; i<nbins; i++) {
    for (int k=LMAX[i]; k<limits.LMAX_NOLIMBER+1; k++) {
      Cl[i][k] = (k > limits.LMIN_tab) ? C_gg_tomo_limber(k, i, i) :
                                         C_gg_tomo_limber_nointerp(k, i, i, 0);
    }
  }

  free((void*) toutfwd);
  free((void*) eta_m);
}
