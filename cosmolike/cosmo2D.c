#define _GNU_SOURCE
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
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// BASIC DEFINITIONS & DECLARATIONS
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CMB beam transfer function (Gaussian approximation).
//
// Models the smoothing of the CMB convergence map by the instrument beam
// as a Gaussian in harmonic space:
//   B_l = exp(-l*(l+1)*sigma^2)
// where sigma = FWHM / sqrt(16*ln(2)) converts the beam full-width at
// half-maximum to the Gaussian width parameter.
//
// Returns 0 outside the multipole range [lmink_wxk, lmaxk_wxk] used for
// the CMB lensing cross-correlations (gk, ks, kk).
// ---------------------------------------------------------------------------
double beam_cmb(
    const int l  // multipole moment
  )
{
  const double s = cmb.fwhm/sqrt(16.0*log(2.0));
  return ((l<cmb.lmink_wxk) || (l>cmb.lmaxk_wxk)) ? 0.0 : exp(-l*(l+1.0)*s*s);
}

// ---------------------------------------------------------------------------
// HEALPix pixel window function.
//
// Returns the precomputed HEALPix pixel window function at multipole l,
// which accounts for the finite pixel size of the CMB convergence map.
// The window function is loaded at initialization into cmb.healpixwin[]
// with cmb.healpixwin_ncls entries (typically lmax+1).
//
// Returns 0 for l >= healpixwin_ncls (beyond the precomputed range).
// ---------------------------------------------------------------------------
double w_pixel(
    const int l  // multipole moment
  )
{
  if (0 == cmb.healpixwin_ncls) {
    log_fatal("cmb.healpixwin_ncls not initialized"); exit(1);
  }
  return (l < cmb.healpixwin_ncls) ? cmb.healpixwin[l] : 0.0;
}

// ---------------------------------------------------------------------------
// Check whether any lens bin has nonzero second-order galaxy bias (b2).
//
// Returns 1 if one-loop galaxy bias corrections should be computed
// (at least one bin has b2 != 0), 0 otherwise. Used to guard the
// allocation and computation of FPTbias one-loop kernels (d1d2, d1s2, d1p3)
// and the higher-derivative counterterm (bk*k^2*PK) in the GS and GG probes.
// ---------------------------------------------------------------------------
static int has_b2_galaxies(void) {
  int res = 0;
  for (int i=0; i<redshift.clustering_nbin; i++) 
    if (nuisance.gb[1][i])
      res = 1;
  return res;
}

// ---------------------------------------------------------------------------
// SIMD type aliases for readability.
//
// All SIMD code goes through SIMDe (SIMD Everywhere), which provides
// portable intrinsics that compile to native AVX2/SSE2 on x86 and fall
// back to scalar emulation on other architectures (ARM, POWER, etc.).
//
// The short aliases keep the vectorized fill/dot-product code readable
// without repeating the simde__ prefix on every variable declaration.
//
//   v4d  = 256-bit register holding 4 doubles (AVX2)
//          used for the main arithmetic in limber_fill_interp, xipm dot products
//   v2d  = 128-bit register holding 2 doubles (SSE2)
//          used for horizontal reduction (sum the 4 lanes of a v4d down to scalar)
//   v4i  = 128-bit register holding 4 int32s (SSE2)
//          used as index registers for AVX2 gather instructions (i32gather_pd)
//          which load 4 non-contiguous doubles from a table in one instruction
// ---------------------------------------------------------------------------
#ifndef COSMO2D_NOT_USE_SIMD
typedef simde__m256d v4d;   // 4 doubles, AVX2-width
typedef simde__m128d v2d;   // 2 doubles (SSE2)
typedef simde__m128i v4i;   // 4 int32s (SSE2) — used for SIMD gather indices
#endif

// ---------------------------------------------------------------------------
// Generic Limber table interpolation with optional SIMD (AVX2) acceleration.
//
// Interpolates ntab precomputed C_l tables simultaneously at multipoles
// l = lmin..lmax-1, sharing the index arithmetic (log-space position,
// clamping, fractional offset) across all tables.
//
// The tables are log-spaced grids: tab[q][i] = C_l at l_i = exp(a + i*dx),
// where a = lim[0], dx = lim[2], and n = nell grid points. Given ln(l),
// the interpolation finds the enclosing grid cell and does linear interp:
//   r = (ln_ell[l] - a) * inv_dx       (fractional grid position)
//   ic = clamp(floor(r), 0, n-2)       (grid index, clamped to valid range)
//   t = r - ic                          (fractional offset within cell)
//   out[q][l] = tab[q][ic] + t * (tab[q][ic+1] - tab[q][ic])
//
// SIMD path (AVX2):
//   Processes 4 ells per iteration using 256-bit vector arithmetic.
//   The table access uses i32gather_pd (AVX2 gather instruction) because
//   the grid indices ic are data-dependent — different ells map to different
//   table positions, so contiguous vector loads are not possible. GCC cannot
//   auto-vectorize this pattern, which is why we use explicit intrinsics.
//   A scalar tail handles the remaining lmax % 4 elements.
//
// The inner loop over q (number of tables) is unrolled by the compiler
// when ntab is a compile-time constant at the call site:
//   ntab = 1: GGL (C_gs), GG (C_gg), GK (C_gk), KS (C_ks)
//   ntab = 2: SS (C_ss, EE + BB simultaneously)
//
// Parameters:
//   ntab   - number of tables to interpolate simultaneously
//   tab    - input tables tab[ntab][n], precomputed C_l on log-spaced grid
//   out    - output arrays out[ntab][>=lmax], written at indices lmin..lmax-1
//   lmin   - first multipole to fill (inclusive)
//   lmax   - last multipole to fill (exclusive)
//   ln_ell - precomputed log(l) array, indexed by l (ln_ell[l] = log(l))
//   a      - log(l_min) of the interpolation grid (= lim[0])
//   inv_dx - reciprocal of grid spacing (= 1/lim[2])
//   n      - number of grid points in the interpolation table (= nell)
// ---------------------------------------------------------------------------
static inline void limber_fill_interp(
    const int ntab,                    // number of tables (1 or 2)
    const double** restrict tab,       // input tables [ntab][n]
    double** restrict out,             // output arrays [ntab][>=lmax]
    const int lmin,                    // first multipole (inclusive)
    const int lmax,                    // last multipole (exclusive)
    const double* restrict ln_ell,     // log(l) array, indexed by l
    const double a,                    // log(l_min) of the grid
    const double inv_dx,               // 1 / grid spacing in log(l)
    const int n                        // number of grid points
  )
{
#ifdef COSMO2D_NOT_USE_SIMD
  for (int l = lmin; l < lmax; l++) {
    const double r = (ln_ell[l] - a) * inv_dx;
    const double i = floor(r);
    const double rc = fmin(fmax(i, 0.0), (double)(n - 2));
    const int ic = (int) rc;
    const double t = r - rc;
    for (int q = 0; q < ntab; q++) {
      out[q][l] = tab[q][ic] + t * (tab[q][ic + 1] - tab[q][ic]);
    }
  }
#else
  const v4d va       = simde_mm256_set1_pd(a);
  const v4d vinv_dx  = simde_mm256_set1_pd(inv_dx);
  const v4d vzero    = simde_mm256_setzero_pd();
  const v4d vmax_idx = simde_mm256_set1_pd((double)(n - 2));
  const v4i vone     = simde_mm_set1_epi32(1);
  int l = lmin;
  for (; l <= lmax - 4; l += 4) {
    v4d vlnell = simde_mm256_loadu_pd(ln_ell + l);
    v4d vr = simde_mm256_mul_pd(simde_mm256_sub_pd(vlnell, va), vinv_dx);
    v4d vi = simde_mm256_floor_pd(vr);
    v4d vicdb = simde_mm256_min_pd(simde_mm256_max_pd(vi, vzero), vmax_idx);
    v4d vt = simde_mm256_sub_pd(vr, vicdb);
    v4i vic = simde_mm256_cvttpd_epi32(vicdb);
    v4i vicp1 = simde_mm_add_epi32(vic, vone);
    for (int q = 0; q < ntab; q++) {
      v4d v0 = simde_mm256_i32gather_pd(tab[q], vic, 8);
      v4d v1 = simde_mm256_i32gather_pd(tab[q], vicp1, 8);
      simde_mm256_storeu_pd(out[q] + l,
        simde_mm256_fmadd_pd(vt, simde_mm256_sub_pd(v1, v0), v0));
    }
  }
  for (; l < lmax; l++) {
    const double r = (ln_ell[l] - a) * inv_dx;
    const int i = (int) floor(r);
    const int ic = i < 0 ? 0 : (i >= n - 1 ? n - 2 : i);
    const double t = r - ic;
    for (int q = 0; q < ntab; q++) {
      out[q][l] = tab[q][ic] + t * (tab[q][ic + 1] - tab[q][ic]);
    }
  }
#endif
}

// ---------------------------------------------------------------------------
// Index mappings from FPTIA/FPTbias internal table ordering to the KIA
// precomputed array ordering used by the vectorized inner loops.
//
// FPTIA.tab stores 10 one-loop IA kernels computed by C-FAST-PT in a fixed
// order that reflects the mathematical structure of the perturbative expansion.
// The KIA arrays reorder these for cache-friendly access per probe.
//
// SS (shear-shear): KIA[0..9] maps all 10 FPTIA kernels
//   KIA[0]  ← FPTIA.tab[0]  tt      (tidal-tidal, EE)
//   KIA[1]  ← FPTIA.tab[2]  ta_dE1  (tidal-density E-mode 1, EE)
//   KIA[2]  ← FPTIA.tab[3]  ta_dE2  (tidal-density E-mode 2, EE)
//   KIA[3]  ← FPTIA.tab[4]  ta      (tidal-alignment, EE)
//   KIA[4]  ← FPTIA.tab[6]  mixA    (mixed A, EE)
//   KIA[5]  ← FPTIA.tab[7]  mixB    (mixed B, EE)
//   KIA[6]  ← FPTIA.tab[8]  mixEE   (mixed EE)
//   KIA[7]  ← FPTIA.tab[1]  tt      (tidal-tidal, BB)
//   KIA[8]  ← FPTIA.tab[5]  ta      (tidal-alignment, BB)
//   KIA[9]  ← FPTIA.tab[9]  mix     (mixed, BB)
//
// GS (galaxy-shear): KIA[2..5] maps the 4 FPTIA kernels needed for GGL
//   KIA[2]  ← FPTIA.tab[6]  mixA    (mixed A)
//   KIA[3]  ← FPTIA.tab[7]  mixB    (mixed B)
//   KIA[4]  ← FPTIA.tab[2]  ta_dE1  (tidal-density 1)
//   KIA[5]  ← FPTIA.tab[3]  ta_dE2  (tidal-density 2)
//   (GS has no BB mode — galaxy density is spin-0)
//
// GS_BIAS (galaxy-shear one-loop bias): KIA[6..8] maps FPTbias correlators
//   KIA[6]  ← FPTbias.tab[0]  d1d2  (delta × delta_2 correlator)
//   KIA[7]  ← FPTbias.tab[2]  d1s2  (delta × s_2 correlator)
//   KIA[8]  ← FPTbias.tab[5]  d1p3  (delta × psi_3 correlator)
//
// These mappings are used with the LERP macro in the precompute loops:
//   for (int m = 0; m < N; m++)
//     KIA[offset+m][i][p] = g4 * LERP(FPTIA.tab[SRC[m]], idx, dr);
// ---------------------------------------------------------------------------
static const int SS_IA_SRC[] = {0, 2, 3, 4, 6, 7, 8, 1, 5, 9}; 
static const int GS_IA_SRC[] = {6, 7, 2, 3}; // KIA[2..5] ← FPTIA.tab
static const int GS_BIAS_SRC[] = {0, 2, 5};  // KIA[6..8] ← FPTbias.tab

// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// optimization: real 2pt computes C_xy_tomo_limber so many times that the  
//               overhead to calls to logl/N_shear/interpol1d is quite expensive
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------
// -------------------------------------------------------------------------

void C_ss_tomo_limber_fill(
    const int nz,                       // tomographic pair index (0..shear_Npowerspectra-1)
    const int lmin,                     // first multipole to fill (inclusive)
    const int lmax,                     // last multipole to fill (exclusive)
    const double* restrict ln_ell,      // precomputed log(l) array, indexed by l
    double* restrict out_EE,            // output EE C_l array, indexed by l
    double* restrict out_BB             // output BB C_l array, indexed by l
  );

void C_gs_tomo_limber_fill(
    const int nz,
    const int lmin,
    const int lmax,
    const double* RESTRICT ln_ell,
    double* RESTRICT out
  );

void C_gg_tomo_limber_fill(
    const int nz,
    const int lmin,
    const int lmax,
    const double* RESTRICT ln_ell,
    double* RESTRICT out
  );

void C_gk_tomo_limber_fill(
    const int nz,
    const int lmin,
    const int lmax,
    const double* RESTRICT ln_ell,
    double* RESTRICT out
  );

void C_ks_tomo_limber_fill(
    const int nz,
    const int lmin,
    const int lmax,
    const double* RESTRICT ln_ell,
    double* RESTRICT out
  );

// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Correlation Functions (real Space) - Full Sky - bin average
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shear-shear real-space two-point correlation functions xi_+(theta) and
// xi_-(theta) with bin-averaged Hankel transform.
//
// Computes xi_pm by summing the angular power spectrum C_l against
// bin-averaged Legendre polynomial kernels Gl_pm:
//
//   xi_+(theta_i) = sum_l Gl_+(i,l) * [C_l^EE + C_l^BB]
//   xi_-(theta_i) = sum_l Gl_-(i,l) * [C_l^EE - C_l^BB]
//
// The Gl_pm kernels are precomputed from associated Legendre polynomials
// and their derivatives (Pmin, Pmax, dPmin, dPmax) evaluated at the angular
// bin edges (xmin = cos(theta_max), xmax = cos(theta_min)). This replaces 
// the naive point-evaluation J_0/J_4 Hankel transform with an exact bin average.
//
// The C_l array is filled in two stages:
//   1. Low-ell (l = 1..LMIN_tab): direct quadrature via _nointerp (or batch)
//   2. High-ell (l = LMIN_tab..LMAX): fast interpolation from the cached
//      log-spaced table via C_ss_tomo_limber_fill with AVX2 gather
//
// The final Hankel sum over ~100k multipoles is SIMD-vectorized via
// #pragma omp simd with dual reduction accumulators for xi_+ and xi_-.
//
// Cache invalidation: recomputes when cosmology, shear photo-z, IA, shear
// redshift distribution, or Ntable settings change. The Gl_pm kernels only
// depend on angular binning (Ntable.Ntheta, Ntable.LMAX) and are rebuilt
// when Ntable.random changes.
//
// Parameters:
//   pm     - 1 for xi_+, 0 for xi_-
//   nt     - angular bin index (0..Ntable.Ntheta-1)
//   ni     - first source redshift bin
//   nj     - second source redshift bin
//   limber - 1 for full Limber (only supported option; 0 exits with error)
//
// Returns:
//   xi_+(theta_nt) or xi_-(theta_nt) for the (ni, nj) tomographic pair
// ---------------------------------------------------------------------------
double xi_pm_tomo(
    const int pm,     // 1 = xi_+, 0 = xi_-
    const int nt,     // angular bin index (0..Ntheta-1)
    const int ni,     // first source redshift bin
    const int nj,     // second source redshift bin
    const int limber  // 1 = Limber (required), 0 = not implemented
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
  if (NSIZE <= 0) {
    log_fatal("cosmic shear requested but tomo.shear_Npowerspectra = %d", NSIZE);
    exit(1);
  }

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

    const int lmin = 1;
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<lmin; l++) {
        Glpm[0][i][l] = 0.0;
        Glpm[1][i][l] = 0.0;
      }
    }
    // -----------------------------------------------------------------------
    // Bin-averaged Hankel transform kernels Gl_pm for xi_+(theta), xi_-(theta).
    //
    // MOTIVATION:
    //   The standard shear 2pt functions are defined at a single angle:
    //     xi_+(theta) = sum_l (2l+1)/(4pi) * C_l * d^l_{2,2}(cos(theta))
    //     xi_-(theta) = sum_l (2l+1)/(4pi) * C_l * d^l_{2,-2}(cos(theta))
    //   where d^l_{2,m'} are reduced Wigner d-matrix elements for spin-2
    //   fields. In practice, data is binned in angular bins [theta_min, theta_max].
    //   Using a point evaluation at the bin center introduces discretization
    //   error. The bin-averaged kernel integrates the exact kernel over the bin:
    //
    //     Gl_pm(i,l) = integral_{theta_min}^{theta_max} kernel_l_pm(theta) sin(theta) dtheta
    //                  ---------------------------------------------------------------
    //                  integral_{theta_min}^{theta_max} sin(theta) dtheta
    //
    //   Substituting x = cos(theta), dx = -sin(theta) dtheta, and noting
    //   xmin = cos(theta_max), xmax = cos(theta_min) (cosine reverses order):
    //
    //     Gl_pm(i,l) = 1/(xmin - xmax) * integral_{xmax}^{xmin} kernel_l_pm(x) dx
    //
    // THE KERNEL:
    //   The Wigner d-matrices for spin-2 fields can be decomposed into
    //   Legendre polynomials P_l(x) and their derivatives dP_l/dx.
    //   The spin-2 prefactor gives an overall 1/[l(l+1)]^2, so:
    //
    //     prefactor = (2l+1) / (2*pi * l^2 * (l+1)^2)
    //               = [(2l+1)/(4*pi)] * [1/(l(l+1))^2]
    //                  ~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~
    //                  Legendre norm     spin-2 factors (one per shear field)
    //
    // ANALYTIC BIN INTEGRATION:
    //   The key insight is that all terms in the kernel — P_l(x), x*P_l(x),
    //   P_l'(x), x*P_l'(x) — have closed-form antiderivatives via Legendre
    //   recurrence relations:
    //
    //     integral P_l(x) dx = [P_{l+1}(x) - P_{l-1}(x)] / (2l+1)
    //     integral x*P_l(x) dx  uses  x*P_l = [(l+1)*P_{l+1} + l*P_{l-1}] / (2l+1)
    //     integral P_l'(x) dx = P_l(x)
    //     integral x*P_l'(x) dx = x*P_l(x) - integral P_l(x) dx
    //
    //   So the bin-averaged kernel evaluates as differences of P_l and dP_l
    //   at the two bin edges, which is what the precomputed arrays provide:
    //     Pmin[i][l]  = P_l(xmin[i])       Pmax[i][l]  = P_l(xmax[i])
    //     dPmin[i][l] = P_l'(xmin[i])      dPmax[i][l] = P_l'(xmax[i])
    //
    // XI_+ vs XI_-:
    //   The two kernels differ only in the sign of the last two terms,
    //   corresponding to the difference between d^l_{2,+2} and d^l_{2,-2}:
    //     Gl_+: ... +2*(l-1)*(x*dP_l - P_l) - 2*(l+2)*dP_{l-1}
    //     Gl_-: ... -2*(l-1)*(x*dP_l - P_l) + 2*(l+2)*dP_{l-1}
    //   Physically, xi_+ = <E*E> + <B*B> and xi_- = <E*E> - <B*B>, so the
    //   sign flip selects additive vs subtractive mixing of E/B modes.
    //
    // NOTATION in the code below:
    //   Pmin[i][l±1], Pmax[i][l±1]   = P_{l±1} at bin edges
    //   dPmin[i][l], dPmax[i][l]     = P_l' at bin edges
    //   xmin[i], xmax[i]             = cos(theta_max), cos(theta_min)
    //   (xmin - xmax) in denominator = bin width in cos(theta)
    // -----------------------------------------------------------------------
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

    // init static vars
    (void) C_ss_tomo_limber((double) limits.LMIN_tab+1, Z1(0), Z2(0), 1);
    
    if (1 == limber) {
      C_ss_tomo_limber_nointerp_batch(lmin, limits.LMIN_tab, NSIZE, Cl);
      #pragma omp parallel for schedule(static)
      for (int nz = 0; nz < NSIZE; nz++) {
        C_ss_tomo_limber_fill(nz, limits.LMIN_tab, Ntable.LMAX,
                              lnell, Cl[0][nz], Cl[1][nz]);
      }
    }
    else {
      log_fatal("NonLimber not implemented"); exit(1);
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        const int q = nz * Ntable.Ntheta + i;
        const double* restrict c0 = Cl[0][nz];
        const double* restrict c1 = Cl[1][nz];
        const double* restrict gp = Glpm[0][i];
        const double* restrict gm = Glpm[1][i];
        double sum0 = 0.0;
        double sum1 = 0.0;
        #pragma omp simd reduction(+:sum0, sum1)
        for (int l=lmin; l<Ntable.LMAX; l++) {
          sum0 += gp[l] * (c0[l] + c1[l]);
          sum1 += gm[l] * (c0[l] - c1[l]);
        }
        xipm[0][q] = sum0;
        xipm[1][q] = sum1;
      }
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
  return (pm > 0) ? xipm[0][q] : xipm[1][q];
}

// ---------------------------------------------------------------------------
// Galaxy-shear (tangential shear) real-space two-point correlation function
// gamma_t(theta) with bin-averaged Hankel transform.
//
// Computes gamma_t by summing the galaxy-shear angular power spectrum C_l^gs
// against a bin-averaged Legendre polynomial kernel Pl:
//
//   gamma_t(theta_i) = sum_l Pl(i,l) * C_l^gs
//
// The kernel Pl encodes the bin-averaged P_2(cos(theta)) projection
// (spin-2 field × spin-0 field), computed from associated Legendre
// polynomials at the bin edges following Kilbinger+ (2017).
//
// The C_l array is filled in two stages:
//   1. Low-ell (l = 1..LMIN_tab): direct quadrature via C_gs_tomo_limber_nointerp
//   2. High-ell (l = LMIN_tab..LMAX): fast interpolation via C_gs_tomo_limber_fill
//
// The final Hankel sum is SIMD-vectorized via #pragma omp simd.
//
// Only lens-source pairs with redshift overlap contribute (test_zoverlap);
// non-overlapping pairs return 0.
//
// Cache invalidation: recomputes when cosmology, photo-z (shear or clustering),
// IA, redshift distributions, Ntable, or galaxy bias parameters change.
// ---------------------------------------------------------------------------
double w_gammat_tomo(
    const int nt,     // angular bin index (0..Ntheta-1)
    const int ni,     // lens redshift bin
    const int nj,     // source redshift bin
    const int limber  // 1 = Limber (required), 0 = not implemented
  )
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
  if (NSIZE <= 0) {
    log_fatal("ggl requested but tomo.ggl_Npowerspectra == %d", NSIZE);
    exit(1);
  }

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

    #pragma omp parallel for collapse(2) schedule(static)
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
    // -----------------------------------------------------------------------
    // Bin-averaged Hankel transform kernel Pl for gamma_t(theta) (tangential shear).
    //
    // MOTIVATION:
    //   The galaxy-shear (GGL) correlation function is:
    //     gamma_t(theta) = sum_l (2l+1)/(4pi*l*(l+1)) * C_l^gs * P_l^2(cos(theta))
    //   where P_l^2(x) is the associated Legendre polynomial of degree l, order 2.
    //   The prefactor 1/[l(l+1)] comes from the single spin-2 shear field
    //   (contrast with xi_pm which has two spin-2 fields giving 1/[l(l+1)]^2).
    //
    //   As with xi_pm, we bin-average the kernel over [theta_min, theta_max]:
    //
    //     Pl(i,l) = 1/(xmin - xmax) * integral_{xmax}^{xmin} kernel_l(x) dx
    //
    //   where xmin = cos(theta_max), xmax = cos(theta_min).
    //
    // ANALYTIC BIN INTEGRATION:
    //   The associated Legendre polynomial P_l^2(x) satisfies the recurrence:
    //     P_l^2(x) = [(2l+1)*x*P_l - (l+2)*P_{l-1}^2] / (l-1)
    //   and can be related to ordinary Legendre polynomials P_l(x) via:
    //     P_l^2(x) = (1-x^2) * P_l''(x) = ... (complicated)
    //   but the integral over a bin can be evaluated using the identity:
    //
    //     integral P_l^2(x) dx = (l+2/(2l+1)) * P_{l-1}(x)
    //                          + (2-l) * x * P_l(x)
    //                          - 2/(2l+1) * P_{l+1}(x)
    //
    //   which is derived from the Legendre recurrence relations. The
    //   bin-averaged kernel is then the difference of these antiderivatives
    //   evaluated at the bin edges, divided by (xmin - xmax).
    //
    // NOTATION:
    //   Pmin[i][l] = P_l(xmin[i])       Pmax[i][l] = P_l(xmax[i])
    //   xmin[i] = cos(theta_max)        xmax[i] = cos(theta_min)
    //   (xmin - xmax) in denominator = bin width in cos(theta)
    //
    // PREFACTOR:
    //   (2l+1) / (4*pi*l*(l+1))
    //   = [(2l+1)/(4*pi)] * [1/(l*(l+1))]
    //     ~~~~~~~~~~~~~~~~   ~~~~~~~~~~~~~~
    //     Legendre norm       single spin-2 field factor
    // -----------------------------------------------------------------------
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
      C_gs_tomo_limber_nointerp_batch(lmin, limits.LMIN_tab, NSIZE, Cl);
      #pragma omp parallel for schedule(static)
      for (int nz = 0; nz < NSIZE; nz++) {
        C_gs_tomo_limber_fill(nz, limits.LMIN_tab, Ntable.LMAX, lnell, Cl[nz]);
      }
    }
    else {
      log_fatal("NonLimber not implemented");
      exit(1);
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        const double* restrict pl = Pl[i];
        const double* restrict cl = Cl[nz];
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int l=lmin; l<Ntable.LMAX; l++) {
          sum += pl[l] * cl[l];
        }
        w_vec[nz*Ntable.Ntheta+i] = sum;
      }
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
  // ---------------------------------------------------------------------------
  if (nt < 0 || nt > Ntable.Ntheta - 1) {
    log_fatal("error in selecting bin number nt = %d (max %d)", nt, Ntable.Ntheta);
    exit(1); 
  }
  if (ni < 0 || 
      ni > redshift.clustering_nbin - 1 || 
      nj < 0 || 
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

// ---------------------------------------------------------------------------
// Galaxy clustering real-space two-point correlation function w(theta) with
// bin-averaged Hankel transform.
//
// Computes w(theta) by summing the galaxy clustering angular power spectrum
// C_l^gg against a bin-averaged Legendre polynomial kernel Pl:
//
//   w(theta_i) = sum_l Pl(i,l) * C_l^gg
//
// The kernel Pl encodes the bin-averaged P_0(cos(theta)) projection
// (spin-0 × spin-0), computed from Legendre polynomials at the bin edges.
//
// The C_l array is filled via two paths depending on the limber flag:
//   limber = 1: full Limber approximation
//     1. Low-ell (l = 1..LMIN_tab): C_gg_tomo_limber_nointerp
//     2. High-ell (l = LMIN_tab..LMAX): C_gg_tomo_limber_fill
//   limber = 0: non-Limber FFTLog (C_cl_tomo) for l < LMAX_NOLIMBER,
//     then Limber fill for l >= LMAX_NOLIMBER
//
// Only auto-correlations (ni = nj) are supported.
//
// Cache invalidation: recomputes when cosmology, clustering photo-z,
// clustering redshift distribution, Ntable, or galaxy bias change.
// ---------------------------------------------------------------------------
double w_gg_tomo(
    const int nt,     // angular bin index (0..Ntheta-1)
    const int ni,     // first lens redshift bin
    const int nj,     // second lens redshift bin (must equal ni)
    const int limber  // 1 = full Limber, 0 = non-Limber FFTLog + Limber hybrid
  )
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
  if (NSIZE <= 0) {
    log_fatal("wgg requested but tomo.clustering_Npowerspectra = %d", NSIZE);
    exit(1);
  }

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

    #pragma omp parallel for collapse(2) schedule(static)
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
    // -----------------------------------------------------------------------
    // Bin-averaged Hankel transform kernel Pl for w(theta) (galaxy clustering).
    //
    // MOTIVATION:
    //   The galaxy clustering correlation function is:
    //     w(theta) = sum_l (2l+1)/(4*pi) * C_l^gg * P_l(cos(theta))
    //   where P_l(x) is the ordinary Legendre polynomial (spin-0 × spin-0,
    //   no 1/[l(l+1)] prefactor unlike the shear probes).
    //
    //   Bin-averaging over [theta_min, theta_max]:
    //
    //     Pl(i,l) = 1/(xmin - xmax) * integral_{xmax}^{xmin} P_l(x) dx
    //
    // ANALYTIC BIN INTEGRATION:
    //   The Legendre recurrence relation gives a closed-form antiderivative:
    //
    //     integral P_l(x) dx = [P_{l+1}(x) - P_{l-1}(x)] / (2l+1)
    //
    //   The bin-averaged kernel is the difference at the two bin edges:
    //
    //     Pl(i,l) = [P_{l+1}(xmin) - P_{l+1}(xmax) - P_{l-1}(xmin) + P_{l-1}(xmax)]
    //               / [(2l+1) * (xmin - xmax)]
    //
    //   The (2l+1) from the antiderivative cancels with the (2l+1)/(4*pi)
    //   prefactor from the Legendre expansion, leaving just 1/(4*pi) as the
    //   overall normalization.
    //
    // NOTATION:
    //   Pmin[i][l] = P_l(xmin[i])       Pmax[i][l] = P_l(xmax[i])
    //   xmin[i] = cos(theta_max)        xmax[i] = cos(theta_min)
    //   (xmin - xmax) in denominator = bin width in cos(theta)
    // -----------------------------------------------------------------------
    #pragma omp parallel for collapse(2) schedule(static)
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
      #pragma omp parallel
      {
        #pragma omp for collapse(2) schedule(static) nowait
        for (int nz=0; nz<NSIZE; nz++) {
          for (int l=lmin; l<limits.LMIN_tab; l++) {
            Cl[nz][l] = C_gg_tomo_limber_nointerp((double) l, nz, nz, 0);
          }
        }
        #pragma omp for schedule(static) nowait
        for (int nz = 0; nz < NSIZE; nz++) {
          C_gg_tomo_limber_fill(nz, limits.LMIN_tab, Ntable.LMAX, lnell, Cl[nz]);
        }
      }
    }
    else {
      const double tolerance = 0.01;
      C_cl_tomo(Cl, tolerance);
      #pragma omp parallel for schedule(static)
      for (int nz=0; nz<NSIZE; nz++) { // LIMBER PART
        C_gg_tomo_limber_fill(nz, limits.LMAX_NOLIMBER, Ntable.LMAX, lnell, Cl[nz]);
      }
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        const double* restrict pl = Pl[i];
        const double* restrict cl = Cl[nz];
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int l=lmin; l<Ntable.LMAX; l++) {
          sum += pl[l] * cl[l];
        }
        w_vec[nz*Ntable.Ntheta + i] = sum;
      }
    }
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
  if (ni < 0 || 
      ni > redshift.clustering_nbin - 1 || 
      nj < 0 || 
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
// Galaxy-CMB lensing real-space two-point correlation function with
// bin-averaged Hankel transform.
//
// Computes the cross-correlation between the galaxy density field and the
// CMB convergence map by summing C_l^gk against a bin-averaged Legendre
// polynomial kernel (same kernel as w_gg — spin-0 × spin-0):
//
//   w_gk(theta_i) = sum_l Pl(i,l) * C_l^gk
//
// The C_l array is filled in two stages:
//   1. Low-ell (l = 1..LMIN_tab): C_gk_tomo_limber_nointerp
//   2. High-ell (l = LMIN_tab..LMAX): C_gk_tomo_limber_fill
//
// No intrinsic alignment contribution (CMB lensing kernel peaks at z ~ 2).
// One lens bin index only (no source bin — the CMB is a single source plane).
//
// Cache invalidation: recomputes when cosmology, clustering photo-z,
// clustering redshift distribution, Ntable, or galaxy bias change.
// ---------------------------------------------------------------------------
double w_gk_tomo(
    const int nt,     // angular bin index (0..Ntheta-1)
    const int ni,     // lens redshift bin
    const int limber  // 1 = Limber (required), 0 = not implemented
  )
{
  static double** Pl = NULL;
  static double* w_vec = NULL;
  static double** Cl = NULL; 
  static double* cmbf = NULL; // CMB filter
  static double* lnell = NULL;
  static uint64_t cache[MAX_SIZE_ARRAYS];
  

  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized");
    exit(1);
  }

  const int NSIZE = redshift.clustering_nbin;
  if (NSIZE <= 0) {
    log_fatal("wgk requested but redshift.clustering_nbin = %d", NSIZE);
    exit(1);
  }

  if (NULL == Pl ||
      NULL == w_vec || 
      NULL == Cl || 
      fdiff2(cache[3], Ntable.random))
  {
    if (Pl != NULL) free(Pl);
    Pl = (double**) malloc2d(Ntable.Ntheta, Ntable.LMAX);;

    if (w_vec != NULL) free(w_vec);
    w_vec = calloc1d(NSIZE*Ntable.Ntheta);

    if (cmbf != NULL) free(cmbf);
    cmbf = (double*) malloc1d(Ntable.LMAX); // CMB filter

    if (lnell != NULL) {
      free(lnell);
    }
    lnell = (double*) malloc1d(Ntable.LMAX + 1);
    for (int l = 1; l <= Ntable.LMAX; l++) {
      lnell[l] = log((double) l);
    }

    double*** P = (double***) malloc3d(2, Ntable.Ntheta, Ntable.LMAX+1);
    double** Pmin  = P[0]; double** Pmax  = P[1];

    double xmin[Ntable.Ntheta];
    double xmax[Ntable.Ntheta];
    for (int i=0; i<Ntable.Ntheta; i++)
    { // Cocoa: dont thread (init of static variables inside set_bin_average)
      bin_avg r = set_bin_average(i,0);
      xmin[i] = r.xmin;
      xmax[i] = r.xmax;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<(Ntable.LMAX+1); l++) {
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
    // -----------------------------------------------------------------------
    // Bin-averaged Hankel transform kernel Pl for w_gk(theta) (galaxy-CMB lensing).
    //
    // Same kernel as galaxy clustering w(theta): both are spin-0 × spin-0
    // correlations, so the projection uses ordinary Legendre polynomials P_l(x)
    // with no 1/[l(l+1)] spin factor. The only difference is the C_l being
    // summed against (C_l^gk instead of C_l^gg).
    //
    // See the w_gg kernel documentation for the derivation.
    // -----------------------------------------------------------------------
    #pragma omp parallel for collapse(2) schedule(static)
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
    #pragma omp parallel for
    for (int l=0; l<Ntable.LMAX; l++) {
      double f = beam_cmb(l);
      if (cmb.healpixwin_ncls > 0) {
        f *= w_pixel(l);
      }
      cmbf[l] = f;
    }
    const int lmin = 1;
    for (int i=0; i<NSIZE; i++) {
      for (int l=0; l<lmin; l++) {
        Cl[i][l] = 0.0;
      }
    } 
    (void) C_gk_tomo_limber((double) limits.LMIN_tab + 1, 0); // init static vars
    if (1 == limber) {
      #pragma omp parallel
      {
        #pragma omp for collapse(2) schedule(static) nowait
        for (int nz=0; nz<NSIZE; nz++) {
          for (int l=lmin; l<limits.LMIN_tab; l++) {
            Cl[nz][l] = C_gk_tomo_limber_nointerp((double) l, nz, 0)*cmbf[l];
          }
        }
        #pragma omp for schedule(static) nowait
        for (int nz=0; nz<NSIZE; nz++) {
          C_gk_tomo_limber_fill(nz, limits.LMIN_tab, Ntable.LMAX, lnell, Cl[nz]);
          for (int l=limits.LMIN_tab; l<Ntable.LMAX; l++) {
            Cl[nz][l] *= cmbf[l]; // multiply by CMB beam filter
          }
        }
      }
    }
    else {
      log_fatal("NonLimber not implemented");
      exit(1);
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        const double* restrict pl = Pl[i];
        const double* restrict cl = Cl[nz];
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int l=lmin; l<Ntable.LMAX; l++) {
          sum += pl[l] * cl[l];
        }
        w_vec[nz*Ntable.Ntheta+i] = sum;
      }
    }
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_clustering;
    cache[2] = redshift.random_clustering;
    cache[3] = Ntable.random;
    cache[4] = nuisance.random_galaxy_bias;
    cache[5] = cmb.random;
  }
  if (ni < 0 || ni > redshift.clustering_nbin-1) {
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
// CMB lensing-shear real-space two-point correlation function with
// bin-averaged Hankel transform.
//
// Computes the cross-correlation between the CMB convergence map and the
// shear field by summing C_l^ks against a bin-averaged Legendre polynomial
// kernel (same spin-2 kernel as gamma_t):
//
//   w_ks(theta_i) = sum_l Pl(i,l) * C_l^ks
//
// The C_l array is filled in two stages:
//   1. Low-ell (l = 1..LMIN_tab): C_ks_tomo_limber_nointerp
//   2. High-ell (l = LMIN_tab..LMAX): C_ks_tomo_limber_fill
//
// Includes NLA intrinsic alignment contribution (C1 * W_source × W_k_cmb).
// One source bin index only (the CMB is a single lens plane).
//
// Cache invalidation: recomputes when cosmology, shear photo-z, IA,
// shear redshift distribution, or Ntable change.
// ---------------------------------------------------------------------------
double w_ks_tomo(
    const int nt,     // angular bin index (0..Ntheta-1)
    const int ni,     // source redshift bin
    const int limber  // 1 = Limber (required), 0 = not implemented
  )
{
  static double** Pl = NULL;
  static double* w_vec = NULL;
  static double** Cl = NULL; 
  static double* cmbf = NULL; // CMB filter
  static double* lnell = NULL;
  static uint64_t cache[MAX_SIZE_ARRAYS];
  
  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized"); exit(1);
  }

  const int NSIZE = redshift.shear_nbin;
  if (NSIZE <= 0) {
    log_fatal("wks requested but redshift.shear_nbin = %d", NSIZE);
    exit(1);
  }

  if (Pl == NULL || 
      w_vec == NULL || 
      NULL == Cl || 
      fdiff2(cache[4], Ntable.random))
  {
    if (Pl != NULL) free(Pl);
    Pl = (double**) malloc2d(Ntable.Ntheta, Ntable.LMAX);;

    if (w_vec != NULL) free(w_vec);
    w_vec = calloc1d(NSIZE*Ntable.Ntheta);

    if (cmbf != NULL) free(cmbf);
    cmbf = (double*) malloc1d(Ntable.LMAX); // CMB filter

    if (lnell != NULL) {
      free(lnell);
    }
    lnell = (double*) malloc1d(Ntable.LMAX + 1);
    for (int l = 1; l <= Ntable.LMAX; l++) {
      lnell[l] = log((double) l);
    }

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

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=0; i<Ntable.Ntheta; i++) {
      for (int l=0; l<(Ntable.LMAX+1); l++) {
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

    // -----------------------------------------------------------------------
    // Bin-averaged Hankel transform kernel Pl for w_ks(theta) (CMB lensing-shear).
    //
    // Same kernel as galaxy-shear gamma_t(theta): both correlate a spin-0
    // field (CMB convergence here, galaxy density for GGL) with a spin-2
    // shear field, so the projection uses P_l^2(x) (associated Legendre
    // polynomial of order 2) with the single spin-2 prefactor 1/[l(l+1)].
    // The only difference is the C_l being summed against (C_l^ks instead
    // of C_l^gs).
    //
    // See the w_gammat kernel documentation for the derivation.
    // -----------------------------------------------------------------------
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
    #pragma omp parallel for
    for (int l=0; l<Ntable.LMAX; l++) {
      double f = beam_cmb(l);
      if (cmb.healpixwin_ncls > 0) {
        f *= w_pixel(l);
      }
      cmbf[l] = f;
    }
    const int lmin = 1;
    for (int i=0; i<NSIZE; i++) {
      for (int l=0; l<lmin; l++) {
        Cl[i][l] = 0.0;
      }
    } 
    (void) C_ks_tomo_limber((double) limits.LMIN_tab + 1, 0); // init static vars
    if (1 == limber) {
      #pragma omp parallel
      {
        #pragma omp for collapse(2) schedule(static) nowait
        for (int nz=0; nz<redshift.shear_nbin; nz++) {
          for (int l=lmin; l<limits.LMIN_tab; l++) {
            Cl[nz][l] = C_ks_tomo_limber_nointerp((double) l, nz, 0)*cmbf[l];
          }
        }
        #pragma omp for schedule(static) nowait
        for (int nz=0; nz<NSIZE; nz++) {
          C_ks_tomo_limber_fill(nz, limits.LMIN_tab, Ntable.LMAX, lnell, Cl[nz]);
          for (int l=limits.LMIN_tab; l<Ntable.LMAX; l++) {
            Cl[nz][l] *= cmbf[l]; // multiply by CMB beam filter
          }
        }
      }
    } 
    else {
      log_fatal("NonLimber not implemented");
      exit(1);
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        const double* restrict pl = Pl[i];
        const double* restrict cl = Cl[nz];
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int l=lmin; l<Ntable.LMAX; l++) {
          sum += pl[l] * cl[l];
        }
        w_vec[nz*Ntable.Ntheta+i] = sum;
      }
    }
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
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
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
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Limber Approximation (Angular Power Spectrum)
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Precomputed cosmological quantities at Gauss-Legendre quadrature nodes.
//
// The Limber integral for angular power spectra is evaluated as a weighted
// sum over quadrature points in scale factor a. Each point requires several
// expensive cosmological functions (comoving distance, growth factor,
// Hubble rate). Since these depend only on a (not on multipole l or
// tomographic bin), they can be computed once and reused across all
// (ell, bin-pair) combinations.
//
// This is the core data structure enabling the loop-inversion optimization
// in C_ss_tomo_limber_work and C_gs_tomo_limber: instead of recomputing
// chi(a), D(a), H(a)/H0 inside every per-ell integrand call, we evaluate
// them once at all quadrature nodes and pass flat arrays to the vectorized
// inner loops.
//
// Memory layout: data[CN_NPARAMS][npts], contiguous via malloc2d.
// ---------------------------------------------------------------------------
typedef struct {
  int npts;       // number of Gauss-Legendre quadrature points (= w->n)
  double** data;  // data[param][p]: cosmological quantities at each node
} cosmo_nodes;

// ---------------------------------------------------------------------------
// Column indices into cosmo_nodes.data[param][p].
// CN_NPARAMS is not a real parameter — it exploits enum auto-increment to
// give the total number of columns, used to size the malloc2d allocation.
// ---------------------------------------------------------------------------
enum {
  CN_A = 0,     // scale factor a (quadrature abscissa in [amin, amax])
  CN_WT,        // Gauss-Legendre quadrature weight
  CN_FK,        // comoving distance chi(a) (= f_K for flat cosmology)
  CN_GROWFAC,   // linear growth factor D(a) = growfac(a)
  CN_HOVERH0,   // H(a)/H0 computed via hoverh0v2(a, dchi/da)
  CN_DCHIDA,    // dchi/da from chi_all(a) — used in the Limber prefactor dchi/da / fK^2
  CN_NPARAMS    // total number of columns (auto-set by enum)
};

// ---------------------------------------------------------------------------
// Create cosmo_nodes by evaluating cosmological functions at all Gauss-Legendre
// quadrature points in the scale factor range [amin, amax].
//
// The quadrature points and weights come from the GSL fixed-order table w,
// which is shared with the Limber integration routines. The number of points
// (typically 64-512 depending on Ntable.high_def_integration) controls the
// accuracy of the numerical integration.
//
// Thread safety: the functions chi_all, growfac, and hoverh0v2 must have
// their internal static interpolation tables initialized before this is
// called in parallel. This is ensured by the init blocks in C_ss_tomo_limber_work
// and C_gs_tomo_limber's caller.
// ---------------------------------------------------------------------------
cosmo_nodes create_cosmo_nodes(
    const double amin,                      // minimum scale factor (integration lower bound)
    const double amax,                      // maximum scale factor (integration upper bound)
    const gsl_integration_glfixed_table* w  // GSL Gauss-Legendre table (provides nodes and weights)
  )
{
  cosmo_nodes cn;
  cn.npts = (int) w->n;
  cn.data = (double**) malloc2d(CN_NPARAMS, cn.npts);

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
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// SS = SHEAR SHEAR
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// NLA shear-shear integrand core (EE only, BB = 0 for NLA).
// Pure arithmetic on preloaded scalars — no branches, no table lookups —
// so GCC can vectorize the calling loop with #pragma omp simd.
//
// Computes: (WK1 - WS1*C11) * (WK2 - WS2*C12) * PK
// ---------------------------------------------------------------------------
static inline double int_for_C_ss_tomo_limber_nla_core(
    const double PK,   // P_delta(k, a): nonlinear matter power spectrum
    const double WK1,  // W_kappa(a, fK, n1): lensing convergence kernel, bin 1
    const double WK2,  // W_kappa(a, fK, n2): lensing convergence kernel, bin 2
    const double WS1,  // W_source(a, n1, h/h0): source galaxy distribution, bin 1
    const double WS2,  // W_source(a, n2, h/h0): source galaxy distribution, bin 2
    const double C11,  // IA_A1(a, D, n1): NLA alignment amplitude, bin 1
    const double C12   // IA_A1(a, D, n2): NLA alignment amplitude, bin 2
  ) // inline necessary for vectorization
{
  const double ans  = (  WK1*WK2 
                        - WS1*WK2*C11 
                        - WS2*WK1*C12
                        + WS1*WS2*C11*C12)*PK;
  return ans;
}

// ---------------------------------------------------------------------------
// TATT shear-shear EE integrand core.
// Pure arithmetic on preloaded scalars for vectorization.
//
// Extends NLA with tidal torquing (C2, bta) and one-loop IA kernels
// (tt, ta, ta_dE, mix). The formula expands the product
//   (WK1 - WS1*IA1) * (WK2 - WS2*IA2) * PK
// where IA_i includes linear (C1*PK), density-weighted (C1*bta*ta_dE),
// and quadratic (C2*mix, C2^2*tt) contributions.
// ---------------------------------------------------------------------------
static inline double int_for_C_ss_tomo_limber_tatt_EE_core(
    const double PK,     // P_delta(k, a): nonlinear matter power spectrum
    const double WK1,    // W_kappa(a, fK, n1): lensing convergence kernel, bin 1
    const double WK2,    // W_kappa(a, fK, n2): lensing convergence kernel, bin 2
    const double WS1,    // W_source(a, n1, h/h0): source distribution, bin 1
    const double WS2,    // W_source(a, n2, h/h0): source distribution, bin 2
    const double C11,    // IA_A1(a, D, n1): linear tidal alignment amplitude, bin 1
    const double C12,    // IA_A1(a, D, n2): linear tidal alignment amplitude, bin 2
    const double C21,    // IA_A2(a, D, n1): quadratic tidal alignment amplitude, bin 1
    const double C22,    // IA_A2(a, D, n2): quadratic tidal alignment amplitude, bin 2
    const double bta1,   // IA_BTA(a, D, n1): density weighting of tidal field, bin 1
    const double bta2,   // IA_BTA(a, D, n2): density weighting of tidal field, bin 2
    const double tt,     // g4 * P_tt(k): tidal-tidal one-loop kernel (FPTIA.tab[0])
    const double ta_dE1, // g4 * P_ta_dE1(k): tidal-density E-mode kernel (FPTIA.tab[2])
    const double ta_dE2, // g4 * P_ta_dE2(k): tidal-density E-mode kernel (FPTIA.tab[3])
    const double ta,     // g4 * P_ta(k): tidal-alignment one-loop kernel (FPTIA.tab[4])
    const double mixA,   // g4 * P_mixA(k): mixed A one-loop kernel (FPTIA.tab[6])
    const double mixB,   // g4 * P_mixB(k): mixed B one-loop kernel (FPTIA.tab[7])
    const double mixEE   // g4 * P_mixEE(k): mixed EE one-loop kernel (FPTIA.tab[8])
  ) // inline necessary for vectorization
{
  const double ans = WK1*WK2*PK 
              - WS1*WK2*(C11*PK + C11*bta1*(ta_dE1+ta_dE2) - 5*C21*(mixA+mixB))
              - WS2*WK1*(C12*PK + C12*bta2*(ta_dE1+ta_dE2) - 5*C22*(mixA+mixB))
              + WS1*WS2*(C11*C12*PK 
                         + C11*C12*(bta1*bta2*ta + (bta1+bta2)*(ta_dE1+ta_dE2))
                         - 5.*(C11*C22 + C12*C21)*(mixA+mixB)
                         - 5.*(C11*bta1*C22+C12*bta2*C21)*mixEE
                         + 25.*C21*C22*tt);
  return ans;
}

// ---------------------------------------------------------------------------
// TATT shear-shear BB integrand core.
// Pure arithmetic on preloaded scalars for vectorization.
//
// BB modes arise only from the quadratic IA terms (tidal torquing).
// There is no tree-level BB contribution, so WK does not appear:
//   BB = WS1*WS2 * (C11*C12*bta1*bta2*ta
//                    - 5*(C11*bta1*C22 + C12*bta2*C21)*mix
//                    + 25*C21*C22*tt)
// For NLA (C2 = 0, bta = 0), BB = 0 identically.
// ---------------------------------------------------------------------------
static inline double int_for_C_ss_tomo_limber_tatt_BB_core(
    const double PK,   // P_delta(k, a): NL MPS (unused but kept for API consistency)
    const double WK1,  // W_kappa(a, fK, n1): convergence kernel (unused, BB has no tree level)
    const double WK2,  // W_kappa(a, fK, n2): convergence kernel (unused, BB has no tree level)
    const double WS1,  // W_source(a, n1, h/h0): source distribution, bin 1
    const double WS2,  // W_source(a, n2, h/h0): source distribution, bin 2
    const double C11,  // IA_A1(a, D, n1): linear tidal alignment amplitude, bin 1
    const double C12,  // IA_A1(a, D, n2): linear tidal alignment amplitude, bin 2
    const double C21,  // IA_A2(a, D, n1): quadratic tidal alignment amplitude, bin 1
    const double C22,  // IA_A2(a, D, n2): quadratic tidal alignment amplitude, bin 2
    const double bta1, // IA_BTA(a, D, n1): density weighting of tidal field, bin 1
    const double bta2, // IA_BTA(a, D, n2): density weighting of tidal field, bin 2
    const double tt,   // g4 * P_tt_BB(k): tidal-tidal BB one-loop kernel (FPTIA.tab[1])
    const double ta,   // g4 * P_ta_BB(k): tidal-alignment BB one-loop kernel (FPTIA.tab[5])
    const double mix   // g4 * P_mix_BB(k): mixed BB one-loop kernel (FPTIA.tab[9])
  ) // inline necessary for vectorization 
{
  const double ans = WS1*WS2*(C11*C12*bta1*bta2*ta 
                       - 5.*(C11*bta1*C22+C12*bta2*C21)*mix 
                       + 25.*C21*C22*tt);
  return ans;
}

// ---------------------------------------------------------------------------
// Scalar integrand for C_ss: combines IA-model branching, FPTIA
// interpolation, and Limber arithmetic in one per-point function.
//
// Not vectorizable (via SIMD) because the switch on nuisance.IA_MODEL sits 
// inside the per-quadrature-point loop.
//
// Not used by the hot-path C_ss_tomo_limber (!!)
// Hot-path employs C_ss_tomo_limber_work. Why? C_ss_tomo_limber_work uses
// precomputed kernels and vectorized inner loops.
//
// This legacy scalar code is still used by:
//   - cosmo2D_scuts (dC/dlnk scale-cut derivatives via the deriv parameter)
//   - Jupyter notebooks for single-point diagnostic evaluations
//
// params is double[5]:
//   ar[0] = n1:    first source redshift bin index
//   ar[1] = n2:    second source redshift bin index
//   ar[2] = l:     multipole moment
//   ar[3] = EE:    1 for E-mode, 0 for B-mode
//   ar[4] = deriv: 0 for C_l, 1 for dC/dlnk (scale-cut diagnostic)
// ---------------------------------------------------------------------------
double int_for_C_ss_tomo_limber(
    double a,       // scale factor (integration variable, GSL interface)
    void* params    // double[5]: {n1, n2, l, EE, deriv} — see above
  )
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  double* ar = (double*) params;
  const int n1 = (int) ar[0]; // first source bin 
  const int n2 = (int) ar[1]; // second source bin 
  if (n1 < 0 || n1 > redshift.shear_nbin - 1 || 
      n2 < 0 || n2 > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (ni,nj) = [%d,%d]", n1,n2); 
    exit(1);
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

  double ans = 1.0;
  switch(nuisance.IA_MODEL) 
  {
    case IA_MODEL_TATT:
    {
      if (0 == nuisance.IA_code) {
        get_FPT_IA();
      }
      const double ell = l + 0.5;
      const double k = ell/fK;
      const double lnk = log(k);
      const double g4 = growfac_a*growfac_a*growfac_a*growfac_a;

      double lim[3];
      lim[0] = log(FPTIA.k_min);
      lim[1] = log(FPTIA.k_max);
      lim[2] = (lim[1] - lim[0])/FPTIA.N;

      double K[10] = {0};
      if (lnk >= lim[0] && lnk <= lim[1]) {
        const double r = (lnk - lim[0]) / lim[2];
        const int b = (int) floor(r);
        const double dr = (b+1 >= FPTIA.N) ? 0.0 : r - b;
        const int idx = (b+1 >= FPTIA.N) ? FPTIA.N - 2 : b;
        for (int m=0; m<10; m++) {
          K[m] = g4*LERP(FPTIA.tab[SS_IA_SRC[m]], idx, dr);
        }
      }

      if (1 == EE) {
        ans = int_for_C_ss_tomo_limber_tatt_EE_core(PK,WK1,WK2,WS1,WS2,C11,C12,
                                                    C21,C22,bta1,bta2,
                                                    K[0],K[1],K[2],K[3],
                                                    K[4],K[5],K[6]);
      }
      else {
        ans = int_for_C_ss_tomo_limber_tatt_BB_core(PK,WK1,WK2,WS1,WS2,C11,C12,
                                                    C21,C22,bta1,bta2,
                                                    K[7],K[8],K[9]);
      }
      break;
    }
    case IA_MODEL_NLA:
    { 
      if (1 == EE) { 
        ans = int_for_C_ss_tomo_limber_nla_core(PK,WK1,WK2,WS1,WS2,C11,C12);
      }
      else {
        ans = 0.0;
      }
      break;
    }
    default: {
      log_fatal("nuisance.IA_MODEL = %d not supported", nuisance.IA_MODEL); 
      exit(1);
    }
  }
  if (0 == deriv) {
    return ans*(chidchi.dchida/(fK*fK))*ell_prefactor;
  } 
  else { // dCXY/dlnk: important to determine scale cuts (2011.06469 eq 17)
    return ans*(chidchi.dchida/fK)*ell_prefactor;
  }
}
// ---------------------------------------------------------------------------
// Single-ell shear-shear C_l via GSL fixed-order Gauss-Legendre quadrature.
//
// Uses the legacy scalar integrand int_for_C_ss_tomo_limber directly,
// evaluating all cosmological functions per quadrature point. Much slower
// than C_ss_tomo_limber (which precomputes everything) or the batch version
// (which shares precomputed arrays across ell values), but self-contained
// and does not require prior initialization of the interpolation table.
//
// Used by:
//   - Jupyter notebook single-point diagnostics
//   - Backward compatibility with callers that expect the scalar interface
//
// Parameters:
//   l    - multipole moment
//   ni   - first source redshift bin index
//   nj   - second source redshift bin index
//   EE   - 1 for E-mode power spectrum, 0 for B-mode
//   init - 1: warm up static variables inside int_for_C_ss_tomo_limber
//             by evaluating the integrand once at amin (returns that value)
//          0: compute and return the full Gauss-Legendre integral
// ---------------------------------------------------------------------------
double C_ss_tomo_limber_nointerp(
    const double l, 
    const int ni, 
    const int nj, 
    const int EE, 
    const int init
  ) // slow - use the batch version - here just for jupyter notebook.
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL; 
  if (ni < 0 || ni > redshift.shear_nbin -1 || 
      nj < 0 || nj > redshift.shear_nbin -1) {
    log_fatal("invalid bin input (ni, nj) = (%d, %d)", ni, nj); exit(1);
  }
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 96 :
                         (1 == hdi) ? 128 :
                         (2 == hdi) ? 256 : 
                         (3 == hdi) ? 512 :  1024; // predefined GSL tables
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
// Core workhorse for all shear-shear C_l computations (both the interp table
// in C_ss_tomo_limber and the low-ell batch in C_ss_tomo_limber_nointerp_batch).
//
// Precomputes all expensive quantities (radial weights, IA amplitudes, matter
// power spectrum, TATT one-loop kernels) on a fixed grid of quadrature points,
// then evaluates the Limber integral for every (ell, tomo-pair) combination.
//
// The key optimization is inverting the loop nesting relative to the legacy
// scalar integrand (int_for_C_ss_tomo_limber):
//   Legacy: for each ell → for each quadrature point → switch(IA_MODEL)
//   Here:   precompute all points → switch(IA_MODEL) → for each ell → SIMD loop
// This lets the IA model branch sit outside the inner loop, and the inner loop
// becomes pure arithmetic on contiguous arrays — vectorizable with AVX2.
//
// Memory layout:
//   WC[5][shear_nbin][npts]:  radial weight functions and IA amplitudes
//     WC[0] = W_kappa    (lensing convergence kernel)
//     WC[1] = W_source   (source galaxy distribution)
//     WC[2] = IA_A1      (linear tidal alignment amplitude, C1)
//     WC[3] = IA_A2      (quadratic tidal alignment amplitude, C2)
//     WC[4] = IA_BTA     (density weighting of tidal field)
//   KIA[11][nell][npts]:  power spectrum and one-loop IA kernels
//     KIA[0..9] = TATT one-loop kernels (see SS_IA_SRC mapping), zero for NLA
//     KIA[10]   = P_delta(k, a), the nonlinear matter power spectrum
//
// Parameters:
//   cn     - precomputed cosmological quantities at quadrature nodes
//            (scale factor, comoving distance, growth factor, dchi/da, weights)
//   lx     - array of multipole values, length nell
//            (log-spaced for C_ss_tomo_limber, integer-spaced for batch)
//   nell   - number of multipole values
//   NSIZE  - number of tomographic shear power spectra (= shear_nbin*(shear_nbin+1)/2)
//   table  - output array [2][NSIZE][nell]: table[0] = EE, table[1] = BB
//   init   - if 1, only warm up functions with static variables and return
//            (no allocation, no computation — used to initialize interpolation
//            tables inside W_kappa, W_source, IA_A1, etc. before the first
//            parallel call)
// ---------------------------------------------------------------------------
static void C_ss_tomo_limber_work(
    const cosmo_nodes* cn,  // quadrature nodes with precomputed cosmo quantities
    const double* lx,       // multipole values (length nell)
    const int nell,         // number of multipole values
    const int NSIZE,        // number of tomo shear power spectra
    double*** table        // output [2][NSIZE][nell]: EE and BB
  )
{
  // -----------------------------------------------------------------------
  // Warm up all functions that lazily initialize internal static tables.
  // Must be called single-threaded before any parallel region touches them.
  // -----------------------------------------------------------------------
  {
    const double a    = cn->data[CN_A][0];
    const double fK   = cn->data[CN_FK][0];
    const double hoh0 = cn->data[CN_HOVERH0][0];
    const double gf   = cn->data[CN_GROWFAC][0];
    const double ell  = lx[0] + 0.5;
    (void) W_kappa(a, fK, 0);
    (void) W_source(a, 0, hoh0);
    (void) IA_A1_Z1(a, gf, 0);
    (void) IA_A2_Z1(a, gf, 0);
    (void) IA_BTA_Z1(a, gf, 0);
    (void) Pdelta(ell/fK, a);
    (void) Z1(0);
    (void) Z2(0);
    if (nuisance.IA_MODEL == IA_MODEL_TATT) {
      if (0 == nuisance.IA_code) get_FPT_IA();
    }
  }

  // -----------------------------------------------------------------------
  // Allocate precomputed arrays
  // -----------------------------------------------------------------------
  double*** WC = (double***) malloc3d(5, redshift.shear_nbin, cn->npts);
  double*** KIA = (double***) malloc3d(11, nell, cn->npts);
  zero3d(KIA, 11, nell, cn->npts);

  double limTATT[3];
  if (nuisance.IA_MODEL == IA_MODEL_TATT) {
    if (0 == nuisance.IA_code) get_FPT_IA();
    limTATT[0] = log(FPTIA.k_min);
    limTATT[1] = log(FPTIA.k_max);
    limTATT[2] = (limTATT[1] - limTATT[0])/FPTIA.N;
  }
  // -----------------------------------------------------------------------
  // Precompute: radial weights per (bin, quadrature point) and
  //             P(k,a) + TATT kernels per (ell, quadrature point)
  // -----------------------------------------------------------------------
  #pragma omp parallel for schedule(static)
  for (int p = 0; p < cn->npts; p++) {
    const double a    = cn->data[CN_A][p];
    const double fK   = cn->data[CN_FK][p];
    const double hoh0 = cn->data[CN_HOVERH0][p];
    const double gf   = cn->data[CN_GROWFAC][p];
    const double g4   = gf*gf*gf*gf;
    for (int b = 0; b < redshift.shear_nbin; b++) {
      WC[0][b][p] = W_kappa(a, fK, b);
      WC[1][b][p] = W_source(a, b, hoh0);
      WC[2][b][p] = IA_A1_Z1(a, gf, b);
      WC[3][b][p] = IA_A2_Z1(a, gf, b);
      WC[4][b][p] = IA_BTA_Z1(a, gf, b);
    }
    for (int i = 0; i<nell; i++) {
      const double ell = lx[i] + 0.5;
      const double k = ell / fK;
      const double lnk = log(k);
      KIA[10][i][p] = Pdelta(k, a);
      if (nuisance.IA_MODEL == IA_MODEL_TATT) {
        if (lnk >= limTATT[0] && lnk <= limTATT[1]) {
          const double r = (lnk - limTATT[0]) / limTATT[2];
          const int b = (int) floor(r);
          const double dr = (b+1 >= FPTIA.N) ? 0.0 : r - b;
          const int idx = (b+1 >= FPTIA.N) ? FPTIA.N - 2 : b;
          for (int m = 0; m < 10; m++) {
            KIA[m][i][p] = g4*LERP(FPTIA.tab[SS_IA_SRC[m]], idx, dr);
          }
        }
      }
    }
  }
  // -----------------------------------------------------------------------
  // Main integration loop.
  // Always uses the TATT core function, which reduces identically to NLA
  // when C2 = BTA = 0 (as enforced by the memset initialization of KIA).
  // This avoids the IA model switch inside the loop, so the SIMD reduction
  // over quadrature points (p) sees only pure arithmetic — no branches.
  // The restrict pointers are hoisted before the p-loop to eliminate
  // gather instructions and enable contiguous AVX2 vector loads.
  //
  // Ell prefactor: l*(l-1)*(l+1)*(l+2) / (l+0.5)^4
  //   = product of two shear-field prefactors (1812.05995 eqs 74-79)
  // -----------------------------------------------------------------------
  #pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < nell; i++) {
    for (int k = 0; k < NSIZE; k++) {
      const int Z1NZ = Z1(k);
      const int Z2NZ = Z2(k);
      const double* restrict fK     = cn->data[CN_FK];
      const double* restrict dchida = cn->data[CN_DCHIDA];
      const double* restrict wt     = cn->data[CN_WT];
      const double* restrict PK     = KIA[10][i];
      const double* restrict WK1    = WC[0][Z1NZ];
      const double* restrict WK2    = WC[0][Z2NZ];
      const double* restrict WS1    = WC[1][Z1NZ];
      const double* restrict WS2    = WC[1][Z2NZ];
      const double* restrict C11    = WC[2][Z1NZ];
      const double* restrict C12    = WC[2][Z2NZ];
      const double* restrict C21    = WC[3][Z1NZ];
      const double* restrict C22    = WC[3][Z2NZ];
      const double* restrict bta1   = WC[4][Z1NZ];
      const double* restrict bta2   = WC[4][Z2NZ];
      const double* restrict tt     = KIA[0][i];
      const double* restrict ta_dE1 = KIA[1][i];
      const double* restrict ta_dE2 = KIA[2][i];
      const double* restrict ta     = KIA[3][i];
      const double* restrict mixA   = KIA[4][i];
      const double* restrict mixB   = KIA[5][i];
      const double* restrict mixEE  = KIA[6][i];
      const double* restrict ttbb   = KIA[7][i];
      const double* restrict tabb   = KIA[8][i];
      const double* restrict mixbb  = KIA[9][i];
      const double l = lx[i];
      const double ell = l + 0.5;
      const double ell4 = ell*ell*ell*ell;
      const double ell_pf = l*(l-1.)*(l+1.)*(l+2.)/ell4;
      double sEE = 0.0, sBB = 0.0;
      #pragma omp simd reduction(+:sEE, sBB)
      for (int p = 0; p < cn->npts; p++) {
        const double amp = (dchida[p]/(fK[p]*fK[p]))*ell_pf;
        sEE += int_for_C_ss_tomo_limber_tatt_EE_core(
                 PK[p],WK1[p],WK2[p],WS1[p],WS2[p],
                 C11[p],C12[p],C21[p],C22[p],bta1[p],bta2[p],
                 tt[p],ta_dE1[p],ta_dE2[p],ta[p],
                 mixA[p],mixB[p],mixEE[p]) * amp * wt[p];
        sBB += int_for_C_ss_tomo_limber_tatt_BB_core(
                 PK[p],WK1[p],WK2[p],WS1[p],WS2[p],
                 C11[p],C12[p],C21[p],C22[p],bta1[p],bta2[p],
                 ttbb[p],tabb[p],mixbb[p]) * amp * wt[p];
      }
      table[0][k][i] = sEE;
      table[1][k][i] = sBB;
    }
  }
  free(WC);
  free(KIA);
}

// ---------------------------------------------------------------------------
// Batch computation of shear-shear C_l at arbitrary multipole values.
//
// Unlike C_ss_tomo_limber_nointerp_batch (which takes a contiguous integer
// range lmin..lmax-1 and writes into Cl[k][l] indexed by multipole), this
// version takes an arbitrary array of ell values and writes results into
// output arrays indexed 0..nell-1. This avoids the offset issue and allows
// non-integer or non-contiguous ell values.
//
// Designed for the fourier-space likelihood (roman_fourier) which evaluates
// C_l at a sparse set of ell values (like.ell[]).
//
// Parameters:
//   ells    - array of multipole values, length nell (need not be integers)
//   nell    - number of multipole values
//   NSIZE   - number of tomographic shear power spectra (= shear_Npowerspectra)
//   out_EE  - output array [NSIZE][nell], indexed as out_EE[nz][i], NULL if init=1
//   out_BB  - output array [NSIZE][nell], indexed as out_BB[nz][i], NULL if init=1
//   init    - if 1, only warm up static variables (no allocation, no computation)
//             if 0, perform full batch computation
// ---------------------------------------------------------------------------
void C_ss_tomo_limber_nointerp_ells(
    const double* ells,   // array of multipole values (length nell)
    const int nell,       // number of multipole values
    const int NSIZE,      // number of tomo shear power spectra
    double** out_EE,      // output EE [NSIZE][nell], NULL if init=1
    double** out_BB      // output BB [NSIZE][nell], NULL if init=1
  )
{
  static gsl_integration_glfixed_table* w = NULL;
  static uint64_t cache[MAX_SIZE_ARRAYS];
  if (NULL == w || fdiff2(cache[0], Ntable.random)) 
  {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 96 :
                         (1 == hdi) ? 128 :
                         (2 == hdi) ? 256 : 
                         (3 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  const double amin = 1./(redshift.shear_zdist_zmax_all+1.);
  const double amax = 1./(1.+fmax(redshift.shear_zdist_zmin_all,1e-6));

  cosmo_nodes cn = create_cosmo_nodes(amin, amax, w);

  if (nell <= 0) {
    log_fatal("nell = %d <= 0", nell);
    exit(1);
  }

  double*** tmp = (double***) malloc3d(2, NSIZE, nell);
  zero3d(tmp, 2, NSIZE, nell);

  C_ss_tomo_limber_work(&cn, ells, nell, NSIZE, tmp);

  for (int k = 0; k < NSIZE; k++) {
    for (int i = 0; i < nell; i++) {
      out_EE[k][i] = tmp[0][k][i];
      out_BB[k][i] = tmp[1][k][i];
    }
  }

  free(tmp); free_cosmo_nodes(&cn);
}

// ---------------------------------------------------------------------------
// Batch computation at integer multipoles lmin..lmax-1.
// Thin wrapper around C_XY_tomo_limber_nointerp_ells.
// ---------------------------------------------------------------------------
void C_ss_tomo_limber_nointerp_batch(
    const int lmin,
    const int lmax,
    const int NSIZE,
    double*** Cl
  )
{
  const int nell = lmax - lmin;
  if (nell <= 0) {
    log_fatal("lmax = %d <= lmin = %d", lmax, lmin);
    exit(1);
  }
  double* lx = (double*) malloc1d(nell);
  for (int i = 0; i < nell; i++) {
    lx[i] = (double)(lmin + i);
  }
  double** tmp_EE = (double**) malloc2d(NSIZE, nell);
  double** tmp_BB = (double**) malloc2d(NSIZE, nell);

  C_ss_tomo_limber_nointerp_ells(lx, nell, NSIZE, tmp_EE, tmp_BB);

  for (int k = 0; k < NSIZE; k++) {
    for (int i = 0; i < nell; i++) {
      Cl[0][k][lmin+i] = tmp_EE[k][i];
      Cl[1][k][lmin+i] = tmp_BB[k][i];
    }
  }

  free(tmp_EE);
  free(tmp_BB);
  free(lx);
}

// ---------------------------------------------------------------------------
// Shared state between C_ss_tomo_limber (which builds the interpolation table)
// and C_ss_tomo_limber_fill (which reads it to fill Cl arrays at ~100k ell
// values for real-space correlation functions).
//
// This avoids passing the table through function arguments, since both
// functions are called independently from different places (C_ss_tomo_limber
// from direct C_l queries, C_ss_tomo_limber_fill from xi_pm_tomo).
//
//   tab     - pointer to the cached table[2][shear_Npowerspectra][nell]
//             tab[0] = EE, tab[1] = BB (owned by C_ss_tomo_limber's static)
//   lim[0]  - log(l_min) of the interpolation grid
//   lim[1]  - log(l_max) of the interpolation grid
//   lim[2]  - uniform spacing in log(l): (lim[1] - lim[0]) / (nell - 1)
//   nell    - number of grid points in the interpolation table
// ---------------------------------------------------------------------------
static struct { double*** tab; double lim[3]; int nell; } ss_ = {0};

// ---------------------------------------------------------------------------
// Shear-shear angular power spectrum C_l^EE or C_l^BB with interpolation.
//
// On first call (or when cosmology/nuisance parameters change), builds a
// log-spaced interpolation table covering l = LMIN_tab..LMAX using
// C_ss_tomo_limber_work, then caches it for subsequent lookups. Returns
// the interpolated value at the requested l via interpol1d.
//
// The table is shared with C_ss_tomo_limber_fill via the ss_ static struct,
// so real-space correlation functions (xi_pm_tomo) can read the same table
// without recomputation.
//
// Cache invalidation: recomputes when any of these change:
//   cosmology.random, nuisance.random_photoz_shear, nuisance.random_ia,
//   redshift.random_shear, Ntable.random
//
// Parameters:
//   l  - multipole moment (continuous, interpolated from the cached table)
//   ni - first source redshift bin index
//   nj - second source redshift bin index
//   EE - 1 for E-mode, 0 for B-mode
//
// Returns:
//   C_l^EE (EE=1) or C_l^BB (EE=0) for the (ni, nj) bin pair
// ---------------------------------------------------------------------------
double C_ss_tomo_limber(
    const double l,  // multipole moment (continuous)
    const int ni,    // first source redshift bin
    const int nj,    // second source redshift bin
    const int EE     // 1 = E-mode, 0 = B-mode
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double*** table = NULL;
  static double lim[3];
  static int nell;
  static gsl_integration_glfixed_table* w = NULL;
  static double* lx = NULL;

  if (NULL == table || fdiff2(cache[4], Ntable.random))
  {
    nell = Ntable.N_ell;
    lim[0] = log(fmax(limits.LMIN_tab - 1., 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0]) / ((double) nell - 1.);
    
    if (table != NULL) free(table);
    table = (double***) malloc3d(2, tomo.shear_Npowerspectra, nell);
    zero3d(table, 2, tomo.shear_Npowerspectra, nell);

    ss_.tab = table; 
    ss_.lim[0] = lim[0]; 
    ss_.lim[1] = lim[1]; 
    ss_.lim[2] = lim[2]; 
    ss_.nell = nell;  

    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 96 :
                         (1 == hdi) ? 128 :
                         (2 == hdi) ? 256 : 
                         (3 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);

    if (lx != NULL) free(lx);
    lx = (double*) malloc1d(nell);
    for (int i = 0; i < nell; i++) {
      lx[i] = exp(lim[0] + i * lim[2]);
    }
  }

  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    const double amin = 1./(redshift.shear_zdist_zmax_all+1.);
    const double amax = 1./(1.+fmax(redshift.shear_zdist_zmin_all,1e-6));
    
    cosmo_nodes cn = create_cosmo_nodes(amin, amax, w);

    C_ss_tomo_limber_work(&cn, lx, nell, tomo.shear_Npowerspectra, table);

    free_cosmo_nodes(&cn);

    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear;
    cache[4] = Ntable.random;
  }

  if (ni < 0 || ni > redshift.shear_nbin - 1 || 
      nj < 0 || nj > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number (ni,nj) = [%d,%d]", ni, nj);
    exit(1);
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
// Fast batch interpolation of the shear-shear C_l table at integer multipoles.
//
// Called by xi_pm_tomo to fill ~100k ell values for the Hankel transform
// C_l → xi_±(theta). Reading the ss_ table one ell at a time via interpol1d
// would be too slow; this function processes 4 ells per iteration using AVX2
// gather instructions (i32gather_pd) through limber_fill_interp.
//
// The EE and BB tables are interpolated simultaneously, sharing the index
// arithmetic (log-space position, clamping, fractional offset) across both.
//
// Requires C_ss_tomo_limber to have been called first to populate ss_.tab.
//
// Parameters:
//   nz     - tomographic pair index (0..shear_Npowerspectra-1)
//   lmin   - first ell to fill (inclusive)
//   lmax   - last ell to fill (exclusive)
//   ln_ell - precomputed log(l) array, indexed as ln_ell[l]
//   out_EE - output array for E-mode C_l, indexed as out_EE[l]
//   out_BB - output array for B-mode C_l, indexed as out_BB[l]
// ---------------------------------------------------------------------------
void C_ss_tomo_limber_fill(
    const int nz,                       // tomographic pair index (0..shear_Npowerspectra-1)
    const int lmin,                     // first multipole to fill (inclusive)
    const int lmax,                     // last multipole to fill (exclusive)
    const double* restrict ln_ell,      // precomputed log(l) array, indexed by l
    double* restrict out_EE,            // output EE C_l array, indexed by l
    double* restrict out_BB             // output BB C_l array, indexed by l
  )
{
  const double* tab[2] = { ss_.tab[0][nz], ss_.tab[1][nz] };
  double* dst[2] = { out_EE, out_BB };
  limber_fill_interp(2, tab, dst, lmin, lmax, ln_ell,
                     ss_.lim[0], 1.0/ss_.lim[2], ss_.nell);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GS = GALAXY-SHEAR
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// One-loop galaxy bias correction to the galaxy-matter cross spectrum.
// Pure arithmetic on preloaded scalars for SIMD vectorization.
//
// Computes: 0.5 * g4 * (b2*d1d2 + bs2*d1s2 + b3*d1p3) + bk*k^2*PK
//
// The first three terms are the standard one-loop SPT contributions from
// second-order (b2), tidal (bs2), and third-order (b3) galaxy bias operators
// convolved with the corresponding matter field correlators (d1d2, d1s2, d1p3).
// The last term (bk*k^2*PK) is the higher-derivative counterterm that absorbs
// sensitivity to small-scale modes beyond the perturbative regime.
//
// This function returns the one-loop piece only — it does NOT include the
// tree-level b1*PK contribution, which is handled separately in the calling
// NLA/TATT core functions to enforce one-loop consistency (oneloop × linear IA
// only, avoiding two-loop cross terms).
// ---------------------------------------------------------------------------
static inline double int_for_C_gs_tomo_limber_bias_oneloop_core(
    const double k,    // wavenumber k = (l+0.5) / fK
    const double PK,   // P_delta(k, a): nonlinear matter power spectrum
    const double g4,   // D(a)^4: fourth power of the linear growth factor
    const double b2,   // gb2(z, nl): second-order galaxy bias
    const double bs2,  // gbs2(z, nl): tidal (s^2) galaxy bias
    const double b3,   // gb3(z, nl): third-order galaxy bias
    const double bk,   // gbK(z, nl): higher-derivative bias coefficient
    const double d1d2, // D^4 * P_{delta,delta2}(k): one-loop matter-b2 correlator
    const double d1s2, // D^4 * P_{delta,s2}(k): one-loop matter-tidal correlator
    const double d1p3  // D^4 * P_{delta,psi3}(k): one-loop matter-b3 correlator
  ) // inline necessary for vectorization
{
   return 0.5*g4*(b2*d1d2 + bs2*d1s2 + b3*d1p3) + (bk * k * k * PK);
}

// ---------------------------------------------------------------------------
// NLA galaxy-shear (galaxy-galaxy lensing) Limber integrand core.
// Pure arithmetic on preloaded scalars for SIMD vectorization.
//
// Three physical contributions to <delta_g, kappa + IA>:
//
//   ft (galaxy density):  WGAL * [b1*PK*(WK - WS*C1) + oneloop*(WK - WS*C1)]
//   st (RSD):             WRSD * PK * (WK - WS*C1)
//   tt (magnification):   WMAG * bmag * ell_pf * PK * (WK - WS*C1)
//
// One-loop consistency: the oneloop correction (b2, bs2, b3, bk terms)
// only multiplies the linear IA piece (WK - WS*C1), not C1*PK. If oneloop
// multiplied the full NLA term (WK - WS*C1)*PK, it would give
// oneloop*PK = (b1*P_mm^1loop + ...)*PK, mixing one-loop bias with P_NL
// which already contains P_mm^1loop — a double count. Instead, oneloop
// carries its own P(k) dimensions internally (from the d1d2, d1s2, d1p3
// correlators and bk*k^2*PK), so it multiplies (WK - WS*C1) without PK.
//
// RSD and magnification enter at tree level only — their one-loop cross
// terms with galaxy bias would be two-loop order, so WRSD and WMAG
// multiply b1*PK but not oneloop.
// ---------------------------------------------------------------------------
static inline double int_for_C_gs_tomo_limber_nla_core(
    const double PK,
    const double WK, 
    const double WS,
    const double WGAL,
    const double WMAG,
    const double WRSD,
    const double C1,
    const double b1,
    const double bmag,
    const double oneloop,
    const double bmag_ell_prefactor
  ) // inline necessary for vectorization
{
  // First term: delta_g_D x (delta_kappa + delta_IA)
  //             Here one-loop bias should only multiply
  //             linear part of IA (otherwise it is 2-loop)
  // Second Term: delta_RSD x (delta_kappa + delta_IA) (RSD)
  // Third Term:  delta_mu  x (delta_kappa + delta_IA) (magnification)
  // Where galaxy bias shows up? delta_g_D = b(z) x delta_m
  // For RSD delta_RSD \propto velocity divergence (not galaxy)
  // For delta_mu - magnification depends on matter
  const double ft = WGAL*b1*(WK*PK - WS*C1*PK) + WGAL*oneloop*(WK-WS*C1);
  const double st = WRSD*(WK*PK - WS*C1*PK);
  const double tt = WMAG*bmag_ell_prefactor*bmag*(WK*PK - WS*C1*PK);
  return ft + st + tt;
}

// ---------------------------------------------------------------------------
// TATT galaxy-shear (galaxy-galaxy lensing) Limber integrand core.
// Pure arithmetic on preloaded scalars for SIMD vectorization.
//
// Extends the NLA core with tidal torquing (C2, BTA) and one-loop IA kernels.
// The intrinsic alignment field is:
//   IA = C1*PK + IATATT
// where IATATT = C1*BTA*(ta_dE1 + ta_dE2) - 5*C2*(mixA + mixB) collects
// the density-weighted tidal alignment and quadratic tidal torquing terms.
//
// Three physical contributions to <delta_g, kappa + IA>:
//
//   ft (galaxy density):  WGAL * [b1*(WK*PK - WS*IA) + oneloop*(WK - WS*C1)]
//   st (RSD):             WRSD * (WK*PK - WS*IA)
//   tt (magnification):   WMAG * bmag * ell_pf * (WK*PK - WS*IA)
//
// One-loop consistency: the TATT terms (C2, BTA, ta_dE, mix) are already
// one-loop order in the perturbative fields (products of two first-order
// tidal/density fields). Crossing them with the one-loop galaxy bias would
// produce two-loop contributions. Therefore:
//   - Tree-level galaxy (b1*PK) multiplies the FULL IA (C1*PK + IATATT)
//   - One-loop galaxy (oneloop) multiplies only LINEAR IA (WK - WS*C1)
//   - RSD and magnification are tree-level, so they get the full IA
//
// Reduces to int_for_C_gs_tomo_limber_nla_core when C2 = 0, BTA = 0,
// and all TATT kernels are zero (as enforced by memset for NLA).
// ---------------------------------------------------------------------------
static inline double int_for_C_gs_tomo_limber_tatt_core(
    const double PK,                // P_delta(k, a): nonlinear matter power spectrum
    const double WK,                // W_kappa(a, fK, ns): lensing convergence kernel
    const double WS,                // W_source(a, ns, h/h0): source galaxy distribution
    const double WGAL,              // W_gal(a, nl, h/h0): lens galaxy distribution
    const double WMAG,              // W_mag(a, fK, nl): magnification lensing kernel
    const double WRSD,              // W_RSD(ell, a0, a1, nl): RSD kernel (0 if disabled)
    const double C1,                // IA_A1(a, D, ns): linear tidal alignment amplitude
    const double C2,                // IA_A2(a, D, ns): quadratic tidal alignment amplitude
    const double BTA,               // IA_BTA(a, D, ns): density weighting of tidal field
    const double ta_dE1,            // g4 * P_ta_dE1(k): tidal-density kernel (FPTIA.tab[2])
    const double ta_dE2,            // g4 * P_ta_dE2(k): tidal-density kernel (FPTIA.tab[3])
    const double mixA,              // g4 * P_mixA(k): mixed A kernel (FPTIA.tab[6])
    const double mixB,              // g4 * P_mixB(k): mixed B kernel (FPTIA.tab[7])
    const double b1,                // gb1(z, nl): linear galaxy bias
    const double bmag,              // gbmag(z, nl): magnification bias coefficient
    const double oneloop,           // one-loop galaxy bias correction (from bias_oneloop_core)
    const double bmag_ell_prefactor // l*(l+1)/(l+0.5)^2: magnification ell prefactor
  ) // inline necessary for vectorization
{
  // First term: delta_g_D x (delta_kappa + delta_IA)
  //             Here one-loop bias should only multiply
  //             linear part of IA (otherwise it is 2-loop)
  // Second Term: delta_RSD x (delta_kappa + delta_IA) (RSD)
  // Third Term:  delta_mu  x (delta_kappa + delta_IA) (magnification)
  // Where galaxy bias shows up? delta_g_D = b(z) x delta_m
  // For RSD delta_RSD \propto velocity divergence (not galaxy)
  // For delta_mu - magnification depends on matter
  const double IATATT = C1*BTA*(ta_dE1 + ta_dE2) - 5.0*C2*(mixA + mixB);
  const double IA = C1*PK + IATATT;
  const double ft = WGAL*b1*(WK*PK - WS*IA) + WGAL*oneloop*(WK - WS*C1); 
  const double st = WRSD*(WK*PK - WS*IA);
  const double tt = (WMAG*bmag_ell_prefactor*bmag)*(WK*PK - WS*IA);
  return ft + st + tt;
}

// ---------------------------------------------------------------------------
// Scalar integrand for C_gs (galaxy-galaxy lensing): combines IA-model
// branching, one-loop galaxy bias, FPTIA/FPTbias interpolation, and Limber
// arithmetic in one per-quadrature-point function.
//
// Not used by the hot-path C_gs_tomo_limber (which uses precomputed arrays
// with vectorized inner loops via the _tatt_core and _nla_core functions).
//
// Still used by:
//   - C_gs_tomo_limber_nointerp for single-ell GSL quadrature
//   - Jupyter notebooks for single-point diagnostic evaluations
//
// params is double[4]:
//   ar[0] = nl:             lens redshift bin index
//   ar[1] = ns:             source redshift bin index
//   ar[2] = l:              multipole moment
//   ar[3] = nonlinear_bias: 1 if one-loop bias is enabled, 0 otherwise
// ---------------------------------------------------------------------------
double int_for_C_gs_tomo_limber(
    double a,       // scale factor (integration variable, GSL interface)
    void* params    // double[4]: {nl, ns, l, nonlinear_bias} — see above
  )
{
  if (include_HOD_GX == 1) {
    log_fatal("HOD NOT IMPLEMENTED");
    exit(1);
  }
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
  const double g4 = growfac_a*growfac_a*growfac_a*growfac_a;
  struct chis chidchi = chi_all(a);
  const double hoverh0 = hoverh0v2(a, chidchi.dchida);
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

  double WRSD = 0.0;
  if (1 == include_RSD_GS) {
    const double k = ell/fK;
    const double chi_0 = ell/k;
    const double chi_1 = (ell+1.)/k;
    const double a_0 = a_chi(chi_0);
    const double a_1 = a_chi(chi_1);
    WRSD = W_RSD(ell, a_0, a_1, nl);
  }

  const double b1 = gb1(z, nl);
  const double bmag = gbmag(z, nl);

  const double C1  = IA_A1_Z1(a, growfac_a, ns);
  const double C2  = IA_A2_Z1(a, growfac_a, ns);
  const double BTA = IA_BTA_Z1(a, growfac_a, ns);

  if (1 == include_HOD_GX) {
    log_fatal("HOD NOT IMPLEMENTED");
    exit(1);
  }

  double b1l = 0.0;
  if (1 == nonlinear_bias) {
    if (0 == nuisance.IA_code) get_FPT_bias();
    const double lnk = log(k);
    double lim[3];
    lim[0] = log(FPTbias.k_min);
    lim[1] = log(FPTbias.k_max);
    lim[2] = (lim[1] - lim[0])/FPTbias.N;
    double d1d2 = 0.0, d1s2 = 0.0, d1p3 = 0.0;
    if (lnk >= lim[0] && lnk <= lim[1]) {
      const double r = (lnk - lim[0]) / lim[2];
      const int b = (int) floor(r);
      const double dr = (b+1 >= FPTbias.N) ? 0.0 : r - b;
      const int idx = (b+1 >= FPTbias.N) ? FPTbias.N - 2 : b;
      d1d2 = LERP(FPTbias.tab[0], idx, dr);
      d1s2 = LERP(FPTbias.tab[2], idx, dr);
      d1p3 = LERP(FPTbias.tab[5], idx, dr);
    }
    const double b2 = gb2(z, nl);
    const double bs2 = gbs2(z, nl);
    const double b3 = gb3(z, nl);
    const double bk = gbK(z, nl);
    b1l = int_for_C_gs_tomo_limber_bias_oneloop_core(k,PK,g4,b2,bs2,b3,
                                                     bk,d1d2,d1s2,d1p3);
  }

  double ans;
  switch(nuisance.IA_MODEL)
  {
    case IA_MODEL_TATT:
    {
      if (0 == nuisance.IA_code) get_FPT_IA();
      const double lnk = log(k);
      double lim[3];
      lim[0] = log(FPTIA.k_min);
      lim[1] = log(FPTIA.k_max);
      lim[2] = (lim[1] - lim[0])/FPTIA.N;
      double K[4] = {0};
      if (lnk >= lim[0] && lnk <= lim[1]) {
        const double r = (lnk - lim[0]) / lim[2];
        const int b = (int) floor(r);
        const double dr = (b+1 >= FPTIA.N) ? 0.0 : r - b;
        const int idx = (b+1 >= FPTIA.N) ? FPTIA.N - 2 : b;
        for (int m=0; m<4; m++) {
          K[m] = g4*LERP(FPTIA.tab[GS_IA_SRC[m]], idx, dr);
        }
      }
      ans = int_for_C_gs_tomo_limber_tatt_core(PK,WK,WS,WGAL,WMAG,WRSD,C1,C2,
                                               BTA,K[2],K[3],K[0],K[1],b1,
                                               bmag,b1l,ell_prefactor);
      break;
    }
    case IA_MODEL_NLA:
    {
      ans = int_for_C_gs_tomo_limber_nla_core(PK,WK,WS,WGAL,WMAG,WRSD,C1,
                                              b1,bmag,b1l,ell_prefactor);
      break;
    }
    default:
    {
      log_fatal("nuisance.IA_MODEL = %d not supported", nuisance.IA_MODEL);
      exit(1);
    }
  } 
  return ans*(chidchi.dchida/(fK*fK))*ell_prefactor2;
}

// ---------------------------------------------------------------------------
// Single-ell galaxy-shear C_l via GSL fixed-order Gauss-Legendre quadrature.
//
// Uses the legacy scalar integrand int_for_C_gs_tomo_limber directly,
// evaluating all cosmological functions per quadrature point. Much slower
// than C_gs_tomo_limber (which precomputes everything) or the batch version
// (which shares precomputed arrays across ell values), but self-contained
// and does not require prior initialization of the interpolation table.
//
// Used by:
//   - Jupyter notebook single-point diagnostics
//   - Backward compatibility with callers that expect the scalar interface
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
  if (nl < -1 || nl > redshift.clustering_nbin -1 || 
      ns < -1 || ns > redshift.shear_nbin -1)
  {
    log_fatal("invalid bin input (ni, nj) = (%d, %d)", nl, ns); exit(1);
  }
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 96 :
                         (1 == hdi) ? 128 :
                         (2 == hdi) ? 256 : 
                         (3 == hdi) ? 512 :  1024; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[4] = {(double) nl, (double) ns, l, has_b2_galaxies()};
  
  const double amin = amin_lens(nl);
  const double amax = amax_lens(nl);
  if (!(amin>0) || !(amin<1) || !(amax>0) || !(amax<1)) {
    log_fatal("0 < amin/amax < 1 not true"); exit(1);
  }
  if (!(amin < amax)) {
    log_fatal("amin < amax not true"); exit(1);
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
// Core workhorse for all galaxy-shear C_l computations (both the interpolation
// table in C_gs_tomo_limber and the low-ell batch in C_gs_tomo_limber_nointerp_batch).
//
// Same design philosophy as C_ss_tomo_limber_work: precomputes all expensive
// quantities on a fixed grid of quadrature points, then evaluates the Limber
// integral for every (ell, tomo-pair) combination with SIMD-vectorized inner loops.
//
// Key difference from SS: galaxy-shear has DIFFERENT integration limits per
// lens bin (amin_lens, amax_lens vary with nl), so cosmo_nodes are created
// per lens bin (cn_all[clustering_nbin]) rather than a single global cn.
//
// Memory layout:
//   WB[10][clustering_nbin][npts]: lens weights and galaxy bias parameters
//     WB[0] = W_gal      (lens galaxy radial kernel)
//     WB[1] = W_mag       (magnification lensing kernel)
//     WB[2] = b1          (linear galaxy bias)
//     WB[3] = bmag        (magnification bias coefficient)
//     WB[4] = b2          (second-order galaxy bias, 0 if no oneloop)
//     WB[5] = bs2         (tidal galaxy bias, 0 if no oneloop)
//     WB[6] = b3          (third-order galaxy bias, 0 if no oneloop)
//     WB[7] = bK          (higher-derivative bias, 0 if no oneloop)
//     WB[8..9] = unused (allocated to 10 for alignment)
//   WC[5][clustering_nbin][shear_nbin][npts]: source weights and IA amplitudes
//     WC[0] = W_kappa     (lensing convergence kernel)
//     WC[1] = W_source    (source galaxy distribution)
//     WC[2] = IA_A1       (linear tidal alignment, C1)
//     WC[3] = IA_A2       (quadratic tidal alignment, C2)
//     WC[4] = IA_BTA      (density weighting of tidal field)
//   KIA[10][clustering_nbin][nell][npts]: power spectrum, RSD, IA and bias kernels
//     KIA[0]   = P_delta(k, a)
//     KIA[1]   = W_RSD(ell, a0, a1, nl) (0 if RSD disabled)
//     KIA[2..5] = TATT IA kernels (mixA, mixB, ta_dE1, ta_dE2), via GS_IA_SRC
//     KIA[6..8] = one-loop bias kernels (d1d2, d1s2, d1p3), via GS_BIAS_SRC
//     KIA[9]   = unused
//
// One-loop consistency: the inner loop calls _tatt_core which ensures
//   - tree-level galaxy (b1*PK) multiplies the full IA (C1*PK + TATT terms)
//   - one-loop galaxy (b1l from _bias_oneloop_core) multiplies only linear IA
//   See the _tatt_core and _nla_core documentation for details.
//
// Parameters:
//   cn_all - array of cosmo_nodes, one per lens bin [clustering_nbin]
//   lx     - array of multipole values, length nell
//   nell   - number of multipole values
//   table  - output array [ggl_Npowerspectra][nell]
//   init   - if 1, only warm up functions with static variables and return
// ---------------------------------------------------------------------------
static void C_gs_tomo_limber_work(
    const cosmo_nodes* cn_all,  // quadrature nodes per lens bin [clustering_nbin]
    const double* lx,           // multipole values (length nell)
    const double* ell_prefactor,  // l*(l+1)/(l+0.5)^2 per ell (magnification)
    const double* ell_prefactor2, // sqrt(l*(l-1)*(l+1)*(l+2))/(l+0.5)^2 per ell
    const int nell,             // number of multipole values
    double** table             // output [ggl_Npowerspectra][nell]
  )
{
  // -----------------------------------------------------------------------
  // Warm up all functions that lazily initialize internal static tables.
  // Must be called single-threaded before any parallel region touches them.
  // -----------------------------------------------------------------------
  const int nonlinear_bias = has_b2_galaxies();
  {
    const cosmo_nodes* cn = &cn_all[0];
    const double a    = cn->data[CN_A][0];
    const double fK   = cn->data[CN_FK][0];
    const double hoh0 = cn->data[CN_HOVERH0][0];
    const double gf   = cn->data[CN_GROWFAC][0];
    const double ell  = lx[0] + 0.5;
    (void) W_gal(a, 0, hoh0);
    (void) W_mag(a, fK, 0);
    (void) W_kappa(a, fK, 0);
    (void) W_source(a, 0, hoh0);
    (void) IA_A1_Z1(a, gf, 0);
    (void) IA_A2_Z1(a, gf, 0);
    (void) IA_BTA_Z1(a, gf, 0);
    (void) Pdelta(ell/fK, a);
    (void) gb1(0.1, 0);
    (void) gbmag(0.1, 0);
    (void) ZL(0);
    (void) ZS(0);
    if (1 == nonlinear_bias) {
      (void) gb2(0.1, 0);
      (void) gbs2(0.1, 0);
      (void) gb3(0.1, 0);
      (void) gbK(0.1, 0);
    }
    if (1 == include_RSD_GS) {
      (void) a_chi(0.9);
      (void) W_RSD(100, 0.9, 0.95, 0);
    }
    if (nuisance.IA_MODEL == IA_MODEL_TATT) {
      if (0 == nuisance.IA_code) get_FPT_IA();
    }
    if (1 == nonlinear_bias) {
      if (0 == nuisance.IA_code) {
        get_FPT_bias();
      }
    }
  }

  // -----------------------------------------------------------------------
  // Allocate precomputed arrays
  // -----------------------------------------------------------------------
  const int npts = cn_all[0].npts;

  double*** WB = (double***) malloc3d(10, redshift.clustering_nbin, npts);
  zero3d(WB, 10, redshift.clustering_nbin, npts);

  double**** WC = (double****) malloc4d(5, 
                                        redshift.clustering_nbin,
                                        redshift.shear_nbin, 
                                        npts);
  zero4d(WC, 5, redshift.clustering_nbin, redshift.shear_nbin, npts);

  double**** KIA = (double****) malloc4d(10, 
                                         redshift.clustering_nbin, 
                                         nell, 
                                         npts);
  zero4d(KIA, 10, redshift.clustering_nbin, nell, npts);

  double limTATT[3];
  double limbias[3];
  if (nuisance.IA_MODEL == IA_MODEL_TATT) {
    if (0 == nuisance.IA_code) get_FPT_IA();
    limTATT[0] = log(FPTIA.k_min);
    limTATT[1] = log(FPTIA.k_max);
    limTATT[2] = (limTATT[1] - limTATT[0])/FPTIA.N;
  }
  if (1 == nonlinear_bias) {
    if (0 == nuisance.IA_code) get_FPT_bias();
    limbias[0] = log(FPTbias.k_min);
    limbias[1] = log(FPTbias.k_max);
    limbias[2] = (limbias[1] - limbias[0])/FPTbias.N;
  }
  
  #pragma omp parallel
  {
    // -----------------------------------------------------------------------
    // Precompute: lens weights, galaxy biases, source weights, IA amplitudes
    // -----------------------------------------------------------------------
    #pragma omp for collapse(2) schedule(static) nowait
    for (int zl = 0; zl < redshift.clustering_nbin; zl++) {
      for (int p = 0; p < npts; p++) {
        const cosmo_nodes* cn = &cn_all[zl];
        const double a  = cn->data[CN_A][p];
        const double z  = 1.0/a - 1.0;
        const double growfac_a = cn->data[CN_GROWFAC][p];
        WB[0][zl][p] = W_gal(cn->data[CN_A][p], zl, cn->data[CN_HOVERH0][p]);
        WB[1][zl][p] = W_mag(cn->data[CN_A][p], cn->data[CN_FK][p], zl);
        WB[2][zl][p] = gb1(z, zl);
        WB[3][zl][p] = gbmag(z, zl);
        if (1 == nonlinear_bias) {
          WB[4][zl][p] = gb2(z, zl);
          WB[5][zl][p] = gbs2(z, zl);
          WB[6][zl][p] = gb3(z, zl);
          WB[7][zl][p] = gbK(z, zl);
        }
        for (int zs = 0; zs < redshift.shear_nbin; zs++) {
          WC[0][zl][zs][p] = W_kappa(cn->data[CN_A][p], cn->data[CN_FK][p], zs);
          WC[1][zl][zs][p] = W_source(cn->data[CN_A][p], zs, cn->data[CN_HOVERH0][p]);
          WC[2][zl][zs][p] = IA_A1_Z1(a, growfac_a, zs);
          WC[3][zl][zs][p] = IA_A2_Z1(a, growfac_a, zs);
          WC[4][zl][zs][p] = IA_BTA_Z1(a, growfac_a, zs);
        }
      }
    }
    // -----------------------------------------------------------------------
    // Precompute: P(k,a), RSD, TATT kernels, one-loop bias kernels
    // -----------------------------------------------------------------------
    #pragma omp for collapse(3) schedule(static)
    for (int zl = 0; zl < redshift.clustering_nbin; zl++) {
      for (int p = 0; p < npts; p++) {
        for (int i = 0; i < nell; i++) {
          const cosmo_nodes* cn = &cn_all[zl];
          const double a  = cn->data[CN_A][p];
          const double fK = cn->data[CN_FK][p];
          const double ell = lx[i] + 0.5;
          const double k = ell / fK;
          const double lnk = log(k);
          KIA[0][zl][i][p] = Pdelta(k, a);
          if (1 == include_RSD_GS) {
            const double chi_0 = ell/k;
            const double chi_1 = (ell + 1.0)/k;
            const double a_0 = a_chi(chi_0);
            const double a_1 = a_chi(chi_1);
            KIA[1][zl][i][p] = W_RSD(ell, a_0, a_1, zl);
          }
          if (nuisance.IA_MODEL == IA_MODEL_TATT) {
            if (lnk >= limTATT[0] && lnk <= limTATT[1]) {
              const double r = (lnk - limTATT[0]) / limTATT[2];
              const int b = (int) floor(r);
              const double dr = (b+1 >= FPTIA.N) ? 0.0 : r - b;
              const int idx = (b+1 >= FPTIA.N) ? FPTIA.N - 2 : b;
              for (int m = 0; m < 4; m++) {
                KIA[2+m][zl][i][p] = LERP(FPTIA.tab[GS_IA_SRC[m]], idx, dr);
              }
            }
          }
          if (1 == nonlinear_bias) {
            if (lnk >= limbias[0] && lnk <= limbias[1]) {
              const double r = (lnk - limbias[0]) / limbias[2];
              const int b = (int) floor(r);
              const double dr = (b+1 >= FPTbias.N) ? 0.0 : r - b;
              const int idx = (b+1 >= FPTbias.N) ? FPTbias.N - 2 : b;
              for (int m = 0; m < 3; m++) {
                KIA[6+m][zl][i][p] = LERP(FPTbias.tab[GS_BIAS_SRC[m]], idx, dr);
              }
            }
          }
        }
      }
    }
  }

  // -----------------------------------------------------------------------
  // Main integration loop.
  // Always calls _tatt_core (reduces to NLA when C2=BTA=0 via memset).
  // restrict pointers hoisted for contiguous AVX2 loads.
  //
  // Ell prefactors (1812.05995 eqs 74-79):
  //   ell_prefactor  = l*(l+1)/(l+0.5)^2       (magnification)
  //   ell_prefactor2 = sqrt(l*(l-1)*(l+1)*(l+2))/(l+0.5)^2  (shear field)
  // -----------------------------------------------------------------------
  #pragma omp parallel for collapse(2) schedule(static)
  for (int j = 0; j < tomo.ggl_Npowerspectra; j++) {
    for (int i = 0; i < nell; i++) {
      const int ZLNZ = ZL(j);
      const int ZSNZ = ZS(j);
      const cosmo_nodes* cn = &cn_all[ZLNZ];

      const double ell = lx[i] + 0.5;
      const double ep  = ell_prefactor[i];
      const double ep2 = ell_prefactor2[i];

      const double* restrict fK      = cn->data[CN_FK];
      const double* restrict growfac = cn->data[CN_GROWFAC];
      const double* restrict dchida  = cn->data[CN_DCHIDA];
      const double* restrict wt      = cn->data[CN_WT];

      const double* restrict WK      = WC[0][ZLNZ][ZSNZ];
      const double* restrict WS      = WC[1][ZLNZ][ZSNZ];
      const double* restrict C1      = WC[2][ZLNZ][ZSNZ];
      const double* restrict C2      = WC[3][ZLNZ][ZSNZ];
      const double* restrict BTA     = WC[4][ZLNZ][ZSNZ];

      const double* restrict WRSD    = KIA[1][ZLNZ][i];
      const double* restrict WGAL    = WB[0][ZLNZ];
      const double* restrict WMAG    = WB[1][ZLNZ];
      const double* restrict b1      = WB[2][ZLNZ];
      const double* restrict bmag    = WB[3][ZLNZ];
      const double* restrict b2      = WB[4][ZLNZ];
      const double* restrict bs2     = WB[5][ZLNZ];
      const double* restrict b3      = WB[6][ZLNZ];
      const double* restrict bk      = WB[7][ZLNZ];

      const double* restrict PK      = KIA[0][ZLNZ][i];
      const double* restrict mixA    = KIA[2][ZLNZ][i];
      const double* restrict mixB    = KIA[3][ZLNZ][i];
      const double* restrict ta_dE1  = KIA[4][ZLNZ][i];
      const double* restrict ta_dE2  = KIA[5][ZLNZ][i];
      const double* restrict d1d2    = KIA[6][ZLNZ][i];
      const double* restrict d1s2    = KIA[7][ZLNZ][i];
      const double* restrict d1p3    = KIA[8][ZLNZ][i];

      double sum = 0.0;
      #pragma omp simd reduction(+:sum)
      for (int p = 0; p < npts; p++) {
        const double g4 = growfac[p]*growfac[p]*growfac[p]*growfac[p];
        const double k = ell / fK[p];
        const double amp = (dchida[p]/(fK[p]*fK[p]))*ep2;
        const double b1l =
            int_for_C_gs_tomo_limber_bias_oneloop_core(k,PK[p],g4,
              b2[p],bs2[p],b3[p],bk[p],d1d2[p],d1s2[p],d1p3[p]);
        const double ans =
            int_for_C_gs_tomo_limber_tatt_core(PK[p],WK[p],WS[p],
              WGAL[p],WMAG[p],WRSD[p],C1[p],C2[p],BTA[p],
              g4*ta_dE1[p],g4*ta_dE2[p],g4*mixA[p],g4*mixB[p],
              b1[p],bmag[p],b1l,ep);
        sum += ans*amp*wt[p];
      }
      table[j][i] = sum;
    }
  }
  free(WB); free(WC); free(KIA);
}

// ---------------------------------------------------------------------------
// Batch computation of galaxy-shear C_l at arbitrary multipole values.
//
// Same design as C_ss_tomo_limber_nointerp_ells but for galaxy-shear.
// Takes an arbitrary array of ell values and writes results into output
// arrays indexed 0..nell-1.
//
// Parameters:
//   ells    - array of multipole values, length nell (need not be integers)
//   nell    - number of multipole values
//   NSIZE   - number of ggl power spectra (= ggl_Npowerspectra)
//   out     - output array [NSIZE][nell], indexed as out[nz][i], NULL if init=1
//   init    - if 1, only warm up static variables
//             if 0, perform full batch computation
// ---------------------------------------------------------------------------
void C_gs_tomo_limber_nointerp_ells(
    const double* ells,  // array of multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of ggl power spectra
    double** out         // output [NSIZE][nell], NULL if init=1
  )
{
  static gsl_integration_glfixed_table* w = NULL;
  static uint64_t cache[MAX_SIZE_ARRAYS];
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 64 :
                         (1 == hdi) ? 128 :
                         (2 == hdi) ? 256 : 
                         (3 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  cosmo_nodes cn_all[redshift.clustering_nbin];
  for (int zl = 0; zl<redshift.clustering_nbin; zl++) {
    const double amin = amin_lens(zl);
    const double amax = amax_lens(zl);
    cn_all[zl] = create_cosmo_nodes(amin, amax, w);
  }
  for (int q = 1; q < redshift.clustering_nbin; q++) {
    if (cn_all[q].npts != cn_all[0].npts) {
      log_fatal("inconsistent quadrature size"); exit(1);
    }
  }
  if (nell <= 0) {
    log_fatal("nell = %d <= 0", nell); exit(1);
  }

  double** tmp_table = (double**) malloc2d(NSIZE, nell);
  zero2d(tmp_table, NSIZE, nell);

  double* ep1  = (double*) malloc1d(nell);
  double* ep2 = (double*) malloc1d(nell);
  for (int i=0; i<nell; i++) {
    ep1[i] = ells[i]*(ells[i] + 1.)/((ells[i] + 0.5)*(ells[i] + 0.5));
    
    const double tmp = (ells[i] - 1.)*ells[i]*(ells[i] + 1.)*(ells[i] + 2.);
    ep2[i] = (tmp > 0) ? sqrt(tmp)/((ells[i] + 0.5)*(ells[i] + 0.5)) : 0.0;
  }

  C_gs_tomo_limber_work(cn_all, ells, ep1, ep2, nell, tmp_table);

  for (int k = 0; k < NSIZE; k++) {
    for (int i = 0; i < nell; i++) {
      out[k][i] = tmp_table[k][i];
    }
  }
  
  free(tmp_table); free(ep1); free(ep2);

  for (int zl = 0; zl < redshift.clustering_nbin; zl++) {
    free_cosmo_nodes(&cn_all[zl]);
  }

  return;
}

// ---------------------------------------------------------------------------
// Batch computation at integer multipoles lmin..lmax-1.
// Thin wrapper around C_XY_tomo_limber_nointerp_ells.
// ---------------------------------------------------------------------------
void C_gs_tomo_limber_nointerp_batch(
    const int lmin,
    const int lmax,
    const int NSIZE,
    double** Cl
  )
{
  const int nell = lmax - lmin;
  if (nell <= 0) {
    log_fatal("lmax = %d <= lmin = %d", lmax, lmin);
    exit(1);
  }
  double* lx = (double*) malloc1d(nell);
  for (int i=0; i<nell; i++) {
    lx[i] = (double)(lmin + i);
  }

  double** tmp = (double**) malloc2d(NSIZE, nell);

  C_gs_tomo_limber_nointerp_ells(lx, nell, NSIZE, tmp);

  for (int k = 0; k < NSIZE; k++) {
    for (int i = 0; i < nell; i++) {
      Cl[k][lmin+i] = tmp[k][i];
    }
  }

  free(tmp); free(lx);
}

// ---------------------------------------------------------------------------
// Shared state between C_gs_tomo_limber (which builds the interpolation table)
// and C_gs_tomo_limber_fill (which reads it to fill Cl arrays at ~100k ell
// values for real-space correlation functions).
//
//   tab     - pointer to the cached table[ggl_Npowerspectra][nell]
//             (owned by C_gs_tomo_limber's static)
//   lim[0]  - log(l_min) of the interpolation grid
//   lim[1]  - log(l_max) of the interpolation grid
//   lim[2]  - uniform spacing in log(l): (lim[1] - lim[0]) / (nell - 1)
//   nell    - number of grid points in the interpolation table
// ---------------------------------------------------------------------------
static struct { double** tab; double lim[3]; int nell; } gs_ = {0};

// ---------------------------------------------------------------------------
// Galaxy-shear angular power spectrum C_l with interpolation.
// See C_ss_tomo_limber for the same pattern. Uses C_gs_tomo_limber_work
// with log-spaced ell values to build the cached interpolation table.
//
// Only lens-source pairs with redshift overlap contribute (test_zoverlap).
//
// Cache invalidation: recomputes when any of these change:
//   cosmology.random, nuisance.random_photoz_shear,
//   nuisance.random_photoz_clustering, nuisance.random_ia,
//   redshift.random_shear, redshift.random_clustering,
//   Ntable.random, nuisance.random_galaxy_bias
// ---------------------------------------------------------------------------
double C_gs_tomo_limber(
    const double l,   // multipole moment (continuous, interpolated)
    const int ni,     // lens redshift bin
    const int nj      // source redshift bin
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static int nell;
  static double lim[3];
  static gsl_integration_glfixed_table* w = NULL;
  static double* lx = NULL;
  static double* ep = NULL;
  static double* ep2 = NULL;

  if (NULL == table || fdiff2(cache[6], Ntable.random)) {
    nell   = Ntable.N_ell;
    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0]) / ((double) nell - 1.0);

    if (table != NULL) free(table);
    table = (double**) malloc2d(tomo.ggl_Npowerspectra, nell);
    zero2d(table, tomo.ggl_Npowerspectra, nell);

    gs_.tab    = table;
    gs_.lim[0] = lim[0];
    gs_.lim[1] = lim[1];
    gs_.lim[2] = lim[2];
    gs_.nell   = nell;

    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 64 :
                         (1 == hdi) ? 128 :
                         (2 == hdi) ? 256 : 
                         (3 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);

    if (lx != NULL) free(lx);
    lx = (double*) malloc1d(nell);
    if (ep != NULL) free(ep);
    ep = (double*) malloc1d(nell);
    if (ep2 != NULL) free(ep2);
    ep2 = (double*) malloc1d(nell);

    for (int i = 0; i < nell; i++) {
      lx[i] = exp(lim[0] + i * lim[2]);
      const double ell = lx[i] + 0.5;
      ep[i] = lx[i]*(lx[i]+1.)/(ell*ell);
      const double tmp = (lx[i]-1.)*lx[i]*(lx[i]+1.)*(lx[i]+2.);
      ep2[i] = (tmp > 0) ? sqrt(tmp)/(ell*ell) : 0.0;
    }
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
    cosmo_nodes cn_all[redshift.clustering_nbin];
    for (int zl = 0; zl < redshift.clustering_nbin; zl++) {
      const double amin = amin_lens(zl);
      const double amax = amax_lens(zl);
      cn_all[zl] = create_cosmo_nodes(amin, amax, w);
    }
    for (int q = 1; q < redshift.clustering_nbin; q++) {
      if (cn_all[q].npts != cn_all[0].npts) {
        log_fatal("inconsistent quadrature size"); exit(1);
      }
    }

    C_gs_tomo_limber_work(cn_all, lx, ep, ep2, nell, table);

    for (int zl = 0; zl < redshift.clustering_nbin; zl++) {
      free_cosmo_nodes(&cn_all[zl]);
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
  if (test_zoverlap(ni, nj)) {
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
// Fast batch interpolation of the galaxy-shear C_l table at integer multipoles.
//
// Called by w_gammat_tomo to fill ~100k ell values for the Hankel transform
// C_l → gamma_t(theta). Uses limber_fill_interp which processes 4 ells per
// iteration via AVX2 gather instructions (i32gather_pd).
//
// Requires C_gs_tomo_limber to have been called first to populate gs_.tab.
// ---------------------------------------------------------------------------
void C_gs_tomo_limber_fill(
    const int nz,                    // tomographic pair index (0..ggl_Npowerspectra-1)
    const int lmin,                  // first multipole to fill (inclusive)
    const int lmax,                  // last multipole to fill (exclusive)
    const double* restrict ln_ell,   // precomputed log(l) array, indexed by l
    double* restrict out             // output C_l array, indexed by l
  )
{
  const double* tab[1] = { gs_.tab[nz] };
  double* dst[1] = { out };
  limber_fill_interp(1, tab, dst, lmin, lmax, ln_ell,
                     gs_.lim[0], 1.0/gs_.lim[2], gs_.nell);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GG = GALAXY-GALAXY CLUSTERING
// GK = GALAXY-CMB LENSING
// KS = CMB LENSING-SHEAR
// KK = CMB LENSING-CMB LENSING
// ---------------------------------------------------------------------------
// Unlike SS and GS, these probes have Npowerspectra = nbin (auto-correlations
// only for GG, or one index per bin for GK/KS/KK), not nbin*(nbin+1)/2.
// The loop-inversion optimization (precomputing cosmo_nodes, radial weights,
// and IA kernels into arrays, then vectorizing the inner quadrature loop)
// has not been applied here — the smaller number of spectra makes the payoff
// marginal relative to the implementation effort. These probes still use
// the legacy pattern: C_xy_tomo_limber calls C_xy_tomo_limber_nointerp
// per (bin, ell) inside an OpenMP parallel loop.
//
// The vectorized _fill functions (C_gg_tomo_limber_fill, etc.) ARE used
// for the real-space Hankel transforms, sharing limber_fill_interp with
// the SS and GS probes.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Scalar integrand for C_gg (galaxy clustering auto-spectrum).
//
// Includes linear bias, magnification bias, RSD, HOD (if enabled), and
// one-loop galaxy bias corrections (b2, bs2, b3, bk, with sigma4 subtraction
// for the d2d2, d2s2, s2s2 correlators).
//
// Assumes ni = nj (auto-correlation only, cross-bin not supported).
//
// params is double[5]:
//   ar[0] = ni:             first lens redshift bin index
//   ar[1] = nj:             second lens redshift bin index (must equal ni)
//   ar[2] = l:              multipole moment
//   ar[3] = use_linear_ps:  1 to use P_lin instead of P_delta (for non-Limber)
//   ar[4] = nonlinear_bias: 1 if one-loop bias is enabled, 0 otherwise
// ---------------------------------------------------------------------------
double int_for_C_gg_tomo_limber(
    double a,       // scale factor (integration variable, GSL interface)
    void* params    // double[5]: {ni, nj, l, use_linear_ps, nonlinear_bias}
  )
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
// Single-ell galaxy clustering C_l via GSL fixed-order quadrature, with an
// option to use the linear power spectrum instead of P_delta.
//
// The use_linear_ps flag is needed by the non-Limber FFTLog calculation
// (C_gg_tomo_nolimber), which computes:
//   Cl_nonlimber = Cl_fftlog(P_lin) + Cl_limber(P_delta) - Cl_limber(P_lin)
// The last two terms require this function with use_linear_ps = 0 and 1
// respectively, evaluated at matching ell values.
//
// Only auto-correlations (ni = nj) are supported.
// ---------------------------------------------------------------------------
double C_gg_tomo_limber_linpsopt_nointerp(
    const double l,            // multipole moment
    const int ni,              // first lens redshift bin
    const int nj,              // second lens redshift bin (must equal ni)
    const int use_linear_ps,   // 1 = use P_lin(k,a), 0 = use P_delta(k,a)
    const int init             // 1 = warm up statics only, 0 = compute
  )
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
    const size_t szint = (0 == hdi) ? 128 : 
                         (1 == hdi) ? 256 : 
                         (2 == hdi) ? 512 : 1024; // predefined GSL tables
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
// Single-ell galaxy clustering C_l using the nonlinear power spectrum.
// Convenience wrapper around C_gg_tomo_limber_linpsopt_nointerp with
// use_linear_ps = 0, matching the API pattern of the other probes
// (C_ss_tomo_limber_nointerp, C_gs_tomo_limber_nointerp, etc.).
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
// Shared state between C_gg_tomo_limber (which builds the interpolation table)
// and C_gg_tomo_limber_fill (which reads it to fill Cl arrays at ~100k ell
// values for real-space correlation functions).
//
//   tab     - pointer to the cached table[clustering_nbin][nell]
//             (owned by C_gg_tomo_limber's static, auto-correlations only)
//   lim[0]  - log(l_min) of the interpolation grid
//   lim[1]  - log(l_max) of the interpolation grid
//   lim[2]  - uniform spacing in log(l): (lim[1] - lim[0]) / (nell - 1)
//   nell    - number of grid points in the interpolation table
// ---------------------------------------------------------------------------
static struct { double** tab; double lim[3]; int nell; } gg_ = {0};

// ---------------------------------------------------------------------------
// Shared state between C_gg_tomo_limber (which builds the interpolation table)
// and C_gg_tomo_limber_fill (which reads it to fill Cl arrays at ~100k ell
// values for real-space correlation functions).
//
//   tab     - pointer to the cached table[clustering_nbin][nell]
//             (owned by C_gg_tomo_limber's static, auto-correlations only)
//   lim[0]  - log(l_min) of the interpolation grid
//   lim[1]  - log(l_max) of the interpolation grid
//   lim[2]  - uniform spacing in log(l): (lim[1] - lim[0]) / (nell - 1)
//   nell    - number of grid points in the interpolation table
// ---------------------------------------------------------------------------
double C_gg_tomo_limber(
    const double l,   // multipole moment (continuous, interpolated)
    const int ni,     // first lens redshift bin
    const int nj      // second lens redshift bin (must equal ni)
  )
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
    
    gg_.tab    = table;
    gg_.lim[0] = lim[0];
    gg_.lim[1] = lim[1];
    gg_.lim[2] = lim[2];
    gg_.nell   = nell;
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
    #pragma omp parallel for collapse(2) schedule(static)
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
      nj < 0 || nj > redshift.clustering_nbin - 1) {
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
// Fast batch interpolation of the galaxy clustering C_l table at integer
// multipoles. Called by w_gg_tomo to fill ~100k ell values for the Hankel
// transform C_l → w(theta). Uses limber_fill_interp which processes 4 ells
// per iteration via AVX2 gather instructions (i32gather_pd).
//
// Requires C_gg_tomo_limber to have been called first to populate gg_.tab.
// ---------------------------------------------------------------------------
void C_gg_tomo_limber_fill(
    const int nz,                    // lens bin index (0..clustering_nbin-1)
    const int lmin,                  // first multipole to fill (inclusive)
    const int lmax,                  // last multipole to fill (exclusive)
    const double* restrict ln_ell,   // precomputed log(l) array, indexed by l
    double* restrict out             // output C_l array, indexed by l
  )
{
  const double* tab[1] = { gg_.tab[nz] };
  double* dst[1] = { out };
  limber_fill_interp(1, tab, dst, lmin, lmax, ln_ell,
                     gg_.lim[0], 1.0/gg_.lim[2], gg_.nell);
}

// ---------------------------------------------------------------------------
// Scalar integrand for C_gk (galaxy-CMB lensing cross spectrum).
//
// Correlates the galaxy density field (with linear bias, magnification bias,
// and optionally RSD) against the CMB convergence kernel W_k. Includes
// one-loop galaxy bias corrections (b2, bs2, b3, bk) when enabled.
//
// params is double[3]:
//   ar[0] = nl:             lens redshift bin index
//   ar[1] = l:              multipole moment
//   ar[2] = nonlinear_bias: 1 if one-loop bias is enabled, 0 otherwise
// ---------------------------------------------------------------------------
double int_for_C_gk_tomo_limber(
    double a,       // scale factor (integration variable, GSL interface)
    void* params    // double[3]: {nl, l, nonlinear_bias} — see above
  )
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

// ---------------------------------------------------------------------------
// Single-ell galaxy-CMB lensing C_l via GSL fixed-order quadrature.
// ---------------------------------------------------------------------------
double C_gk_tomo_limber_nointerp(
    const double l,   // multipole moment
    const int ni,     // lens redshift bin index
    const int init    // 1 = warm up statics only, 0 = compute
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;
  
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 64 : 
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

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// so C_gk_tomo_limber_fill can see C_gk_tomo_limber data
static struct { double** tab; double lim[3]; int nell; } gk_ = {0};

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

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

    gk_.tab    = table;
    gk_.lim[0] = lim[0];
    gk_.lim[1] = lim[1];
    gk_.lim[2] = lim[2];
    gk_.nell   = nell;
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
    #pragma omp parallel for collapse(2) schedule(static)
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
  
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
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

// ----------------------------------------------------------------------------
// optimization: real-space 2pt interpolates ~100,000 ells. The vectorized fill
// is faster than per-ell lookups. GCC cannot auto-vectorize the indirect 
// table access (gather), so we provide an explicit AVX2 path using i32gather_pd
// -----------------------------------------------------------------------------

void C_gk_tomo_limber_fill(
    const int nz, const int lmin, const int lmax,
    const double* restrict ln_ell, double* restrict out)
{
  const double* tab[1] = { gk_.tab[nz] };
  double* dst[1] = { out };
  limber_fill_interp(1, tab, dst, lmin, lmax, ln_ell,
                     gk_.lim[0], 1.0/gk_.lim[2], gk_.nell);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double C_ks_tomo_limber_nointerp(const double l, const int ni, const int init)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;  

  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  if (NULL ==  w || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 64 : 
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

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// so C_ks_tomo_limber_fill can see C_ks_tomo_limber data
static struct { double** tab; double lim[3]; int nell; } ks_ = {0};

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

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

    ks_.tab    = table;
    ks_.lim[0] = lim[0];
    ks_.lim[1] = lim[1];
    ks_.lim[2] = lim[2];
    ks_.nell   = nell;
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
    #pragma omp parallel for collapse(2) schedule(static)
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

// ----------------------------------------------------------------------------
// optimization: real-space 2pt interpolates ~100,000 ells. The vectorized fill
// is faster than per-ell lookups. GCC cannot auto-vectorize the indirect 
// table access (gather), so we provide an explicit AVX2 path using i32gather_pd
// -----------------------------------------------------------------------------

void C_ks_tomo_limber_fill(
    const int nz, const int lmin, const int lmax,
    const double* restrict ln_ell, double* restrict out)
{
  const double* tab[1] = { ks_.tab[nz] };
  double* dst[1] = { out };
  limber_fill_interp(1, tab, dst, lmin, lmax, ln_ell,
                     ks_.lim[0], 1.0/ks_.lim[2], ks_.nell);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
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
    const size_t szint = (0 == hdi) ? 64 : 
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
    #pragma omp parallel for schedule(static)
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
    #pragma omp parallel for collapse(2) schedule(static)
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
    #pragma omp parallel for collapse(2) schedule(static)
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
    (void) C_ky_limber_nointerp(exp(lim[0]), 1);  // init static vars
    #pragma omp parallel for schedule(static)
    for (int i=0; i<Ntable.N_ell; i++) {
      table[i] = C_ky_limber_nointerp(exp(lim[0] + i*lim[2]), 0);
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
  
  const double lnl = log(l);
  if (lnl < lim[0])
    log_warn("l = %e < lmin = %e. Extrapolation adopted", l, exp(lim[0]));
  if (lnl > lim[1])
    log_warn("l = %e > l_max = %e. Extrapolation adopted", l, exp(lim[1]));

  return interpol1d(table, Ntable.N_ell, lim[0], lim[1], lim[2], lnl);
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

  double ar[1] = {l};
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
    #pragma omp parallel for schedule(static)
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
  return interpol1d(table, Ntable.N_ell, lim[0], lim[1], lim[2], lnl);
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

// ---------------------------------------------------------------------------
// Configuration for one radial component in the FFTLog non-Limber calculation.
//
// Each radial component (galaxy density, RSD velocity, magnification) has
// its own config controlling the FFTLog bias parameter, windowing, and
// whether the transform computes C_l or its derivatives.
//
// Typical values used in C_gg_tomo_nolimber:
//   Galaxy density: nu=1.0,  c_window_width=0.25, derivative=0, N_pad=200
//   RSD velocity:   nu=1.01, c_window_width=0.25, derivative=2, N_pad=500
//   Magnification:  nu=1.0,  c_window_width=0.25, derivative=0, N_pad=500
// ---------------------------------------------------------------------------
typedef struct config 
{
  double nu;              // bias parameter: input f(x) is divided by x^nu before FFT
                          // (must be > 0; nu=1 is the standard Hankel transform;
                          // slight offsets like 1.01 improve convergence for RSD)
  double c_window_width;  // fractional width of the Fourier-space tapering window
                          // (0 < c_window_width < 1; typically 0.25; controls how
                          // aggressively high-frequency ringing is suppressed)
  int derivative;         // order of the spherical Bessel derivative in the kernel:
                          //   0: j_l(kr)          — standard projection (density, magnification)
                          //   1: j_l'(kr)         — first derivative
                          //   2: j_l''(kr)        — second derivative (RSD, which involves
                          //                         the second derivative of the velocity field)
  long N_pad;             // number of zero-padding elements added on each side of the
                          // input array before FFT (larger N_pad reduces edge effects
                          // but increases FFT cost; 200 for density, 500 for RSD/mag)
  long N_extrap_low;      // number of points for low-x power-law extrapolation (unused)
  long N_extrap_high;     // number of points for high-x power-law extrapolation (unused)
} config;

// ---------------------------------------------------------------------------
// Non-Limber C_gg via FFTLog: Phase 1 (ell-independent forward transform).
//
// The non-Limber galaxy clustering power spectrum uses the FFTLog algorithm
// to evaluate the Hankel-like integral that replaces the Limber approximation
// at low multipoles (l < LMAX_NOLIMBER). The computation is split into two
// phases to avoid redundant work:
//
//   Phase 1 (this function): forward FFT of the radial weight functions.
//     The input functions fx[i][j][q] — galaxy density (j=0), RSD velocity
//     (j=1), and optionally magnification (j=2) — are biased by x^(-nu),
//     zero-padded to an FFT-friendly size N[j][2], and forward-transformed.
//     None of this depends on multipole l, so it is done once.
//
//   Phase 2 (cfftlog_ells_p2): ell-dependent inverse transform.
//     For each multipole l, computes the Gamma-function kernel g_l(z),
//     multiplies it against the forward-transformed data, and inverse-FFTs
//     to obtain the projected power spectrum Fy[i][j][k][q]. This is called
//     repeatedly in blocks of BLOCK ells with early termination when the
//     non-Limber result converges to the Limber result.
//
// The split saves ~30-50% of total FFTLog time because the forward FFT
// (dominated by FFTW + windowing) is O(nbins * SIZE2 * N*logN), while
// the inverse transform is O(nbins * SIZE2 * BLOCK * N*logN) and runs
// many times as ells are processed in blocks.
//
// Memory layout:
//   fx[SIZE1][SIZE2][Nx]:  input radial weight functions on the chi grid
//     fx[i][0] = chi * n(z) * D(a) * (H/H0) * b1(z)    (galaxy density)
//     fx[i][1] = -chi * n(z) * D(a) * (H/H0) * f(a)    (RSD velocity)
//     fx[i][2] = (W_mag / fK / coverH0^2) * D(a)        (magnification, optional)
//   toutfwd[SIZE1*SIZE2][Nmax/2+1]: output forward FFT coefficients
//     indexed as toutfwd[i*SIZE2+j] for bin i, component j
//   eta_m[SIZE2][Nmax/2+1]: output Fourier-space frequencies
//     eta_m[j][q] = 2*pi*q / (dlnx * N[j][2])
//
// The c-window (Gaussian tapering in Fourier space) is applied after the
// forward FFT to suppress ringing from the finite chi range. The window
// width is controlled by cfg[j].c_window_width (typically 0.25).
//
// FFTW plans and window tables are cached in static variables and rebuilt
// only when SIZE2 or N[j][2] change (i.e., when Ntable settings change,
// not on every cosmology evaluation).
// ---------------------------------------------------------------------------
void cfftlog_ells_p1(
    double* const x,                        // chi grid values, length Nx (log-spaced)
    double* const* const* const fx,         // input functions fx[SIZE1][SIZE2][Nx]
    int const Nx,                           // number of chi grid points
    config* const cfg,                      // FFTLog config per component (nu, c_window_width, derivative, N_pad)
    fftw_complex* const* const toutfwd,     // output forward FFT coefficients [SIZE1*SIZE2][Nmax/2+1]
    double* const* const eta_m,             // output Fourier frequencies [SIZE2][Nmax/2+1]
    int N[][3],                             // per-component sizes: N[j][0]=N_pad, N[j][1]=Nx, N[j][2]=FFT size
    int const Nmax,                         // max(N[j][2]) across all components
    int const SIZE1,                        // number of lens redshift bins
    int const SIZE2                         // number of radial components (2 without magnification, 3 with)
  )
{
  static int cache[MAX_SIZE_ARRAYS];
  static int cached_N2[MAX_SIZE_ARRAYS];
  static fftw_plan* planf = NULL;
  static double** W_table = NULL;
  static int* W_kmax = NULL;

  int rebuild = (planf == NULL || SIZE2 != cache[0]);
  for (int j = 0; j < SIZE2 && !rebuild; j++) {
    if (cached_N2[j] != N[j][2]) rebuild = 1;
  }

  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------  

  double*** fb = (double***) malloc3d(SIZE1, SIZE2, Nmax); // biased input func
  #pragma omp parallel for collapse(2) schedule(static)
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

  if (rebuild) {
    if (planf != NULL) {
      for (int j = 0; j < cache[0]; j++) {
        fftw_destroy_plan(planf[j]);
      }
      free(planf);
    }
    //fftw_plan planf[SIZE2];
    planf = (fftw_plan*) malloc(sizeof(fftw_plan) * SIZE2);
    for (int j=0; j<SIZE2; j++) {
      planf[j] = fftw_plan_dft_r2c_1d(N[j][2], 
                                      fb[0][j], 
                                      toutfwd[0*SIZE2+j],
                                      FFTW_ESTIMATE);
      cached_N2[j] = N[j][2];
    }
    cache[0] = SIZE2; 

    if (W_table != NULL) {
        free(W_table);
    }
    W_table = (double**) malloc2d(SIZE2, Nmax/2 + 1);
    if (W_kmax != NULL) {
      free(W_kmax);
    }
    W_kmax  = (int*) malloc(sizeof(int) * SIZE2);

    const double inv_2pi = 1.0 / (2.0 * M_PI);
    for (int j=0; j<SIZE2; j++) {
      const double cww = cfg[j].c_window_width;
      if (!(cww > 0) || !(cww < 1)) {
          log_fatal("improper window width"); exit(1);
      }
      const int halfN = N[j][2] / 2;
      const int kmax = (int)(halfN * cww);
      if (kmax <= 0) {
          log_fatal("kmax <= 0 in c-window"); exit(1);
      }
      W_kmax[j] = kmax;
      for (int k = 0; k < kmax + 1; k++) {
        W_table[j][k] = (double)k / kmax - sin(2.0 * M_PI * k / kmax) * inv_2pi;
      }
    }   
  }

  #pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<SIZE1; i++) {
    for(int j=0; j<SIZE2; j++) {
      fftw_execute_dft_r2c(planf[j], fb[i][j], toutfwd[i*SIZE2+j]);
      // c_window_cfft function begins -----------------------------------------
      const int halfN = N[j][2]/2;
      const int kmax = W_kmax[j];
      const double* const W = W_table[j];
      for(int k=0; k<(kmax+1); k++) { // window for right-side
        toutfwd[i*SIZE2+j][halfN-k] *= W[k];
      }
    }
  }

  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------

  const double dlnx = log(x[1]/x[0]);
  for(int j=0; j<SIZE2; j++) {
    const double scale = (2.0*M_PI/(dlnx * N[j][2]));
    for(int q=0; q<N[j][2]/2+1; q++) {
      eta_m[j][q] = scale * q;  
    }
  }

  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  free((void*) fb);
}

// ---------------------------------------------------------------------------
// Non-Limber C_gg via FFTLog: Phase 2 (ell-dependent inverse transform).
//
// Processes a block of multipoles l = ks..ke-1, computing the projected
// radial functions Fy[i][j][k][q] and wavenumber grid y[i][k][q] for each
// lens bin i, radial component j, multipole k, and chi-node q.
//
// Called repeatedly from C_gg_tomo_nolimber in a while-loop over blocks
// of BLOCK multipoles, with early termination when the non-Limber result
// converges to the Limber result (checked per lens bin via the converged
// array).
//
// Algorithmic steps for each block:
//
//   1. Wavenumber grid: y[i][k][q] = (k+1) / x[Nx-1-q]
//      The output wavenumber at each chi-node, reversed relative to x.
//
//   2. Gamma-function kernel gl[j][k][q]:
//      The Bessel-function transform kernel in Fourier space, computed via
//      the Lanczos approximation to ln(Gamma). Depends on cfg[j].derivative:
//        case 0: gl = exp(z*ln2) * Gamma(½(k+z)) / Gamma(½(3+k-z))
//        case 1: gl = -(z-1) * exp((z-1)*ln2) * Gamma(½(k+z-1)) / Gamma(½(4+k-z))
//        case 2: gl = (z-1)(z-2) * exp((z-2)*ln2) * Gamma(½(k+z-2)) / Gamma(½(5+k-z))
//      where z = nu + i*eta_m[j][q] is the complex biasing parameter.
//
//      Optimization: only the first two ells (ks, ks+1) are computed from
//      the full Lanczos formula. Subsequent ells use the Gamma recurrence
//      relation Gamma(a+1) = a*Gamma(a), which gives:
//        gl[k+2] = gl[k] * (k + z + offset) / (k + 3 - z + offset)
//      This reduces O(BLOCK * N/2) Lanczos evaluations to O(2 * N/2)
//      plus O(BLOCK * N/2) complex multiplies — a major speedup since
//      Lanczos involves 9 complex divisions + clog + cexp per evaluation.
//
//   3. Inverse FFT: for each (bin i, component j, multipole k):
//      - Multiply forward FFT coefficients (from p1) by the phase shift
//        exp(-i * eta_m[q] * ln(base_j * y[i][k][0])) and by gl[j][k][q]
//      - Conjugate the result
//      - Inverse FFT via FFTW (c2r) using per-thread buffers
//      - Extract the unpadded region and normalize:
//        Fy[i][j][k][q] = outbcw[N_pad + q] * sqrt(pi) / (4*N * y^nu)
//
//      Phase rotation optimization (SIMD path): instead of computing
//      cos/sin per q, uses a rotating phasor (complex multiply per step)
//      with periodic exact recomputation every 1024 steps to prevent drift.
//
//      SIMD optimization for Fy normalization: the y^(-nu) factor is
//      rewritten using y = (k+1)/x[Nx-1-q], precomputing x^nu once in
//      x_pow_nu[j][q]. The inner loop then uses AVX2 vector multiply
//      instead of per-element pow().
//
// Thread safety: each OpenMP thread uses its own outfwd[id] and outbcw[id]
// buffers with fftw_execute_dft_c2r (new-array interface), avoiding races
// on shared FFTW arrays.
//
// Static caches: FFTW inverse plans, gl array, outfwd/outbcw buffers, and
// base_j are cached and rebuilt only when NTHREADS, Nmax, BLOCK, SIZE2,
// or N[j][2] change — not on every cosmology evaluation.
// ---------------------------------------------------------------------------
void cfftlog_ells_p2(
    double* const x,                        // chi grid values, length Nx (log-spaced, from p1)
    int const Nx,                           // number of chi grid points
    config* const cfg,                      // FFTLog config per component (nu, derivative, N_pad)
    int const LMAX,                         // maximum multipole for convergence clipping
    double* const* const* const y,          // output wavenumber grid y[SIZE1][LMAX][Nx]
    double* const* const* const* const Fy,  // output projected functions Fy[SIZE1][SIZE2][LMAX][Nx]
    fftw_complex* const* const toutfwd,     // forward FFT coefficients from p1 [SIZE1*SIZE2][Nmax/2+1]
    double* const* const eta_m,             // Fourier frequencies from p1 [SIZE2][Nmax/2+1]
    int  N[][3],                            // per-component sizes: N[j][0]=N_pad, N[j][1]=Nx, N[j][2]=FFT size
    int const Nmax,                         // max(N[j][2]) across all components
    int const ks,                           // first multipole in this block (inclusive)
    int const ke,                           // last multipole in this block (exclusive)
    const int* const converged,             // per-bin convergence flags [SIZE1]: skip bin if converged[i]=1
    int const SIZE1,                        // number of lens redshift bins
    int const SIZE2                         // number of radial components (2 or 3)
  )
{
  static int cache[MAX_SIZE_ARRAYS];
  static int cached_N2[MAX_SIZE_ARRAYS]; // track N[j][2] for FFTW plan validity
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
  #ifdef _OPENMP
  const int NTHREADS = omp_get_max_threads();
  #else
  const int NTHREADS = 1;
  #endif
  const double sqrtpi = sqrt(M_PI);
  const double ln2 = log(2.);
  const double x0   = x[0];
  const double dlnx = log(x[1]/x[0]);
  const double complex clogpi = clog(M_PI);
  const double ln2pio2 = 0.5*log(2*M_PI);

  int rebuild = (outfwd == NULL   || 
      NTHREADS != cache[0] ||
      Nmax != cache[1] || 
      BLOCK > cache[2] || 
      SIZE2 != cache[3]);
  for (int j=0; j<SIZE2 && !rebuild; j++) {
    if (cached_N2[j] != N[j][2]) rebuild = 1;
  }

  if (rebuild) 
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

    cache[0] = NTHREADS;
    cache[1] = Nmax;
    cache[2] = BLOCK; 
    cache[3] = SIZE2;
    for (int j=0; j<SIZE2; j++) {
      cached_N2[j] = N[j][2];
    }
  }

  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------  

  for(int j=0; j<SIZE2; j++) {
    base_j[j] = x0 / exp(2 * N[j][0] * dlnx); // x depends on cosmo (chi_min/max)
  }

  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------  
  
  #pragma omp parallel for collapse(2) schedule(static)
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

        #pragma omp parallel for collapse(2) schedule(static)
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
        #pragma omp parallel for schedule(static)
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
        #pragma omp parallel for collapse(2) schedule(static)
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
        #pragma omp parallel for schedule(static)
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
        #pragma omp parallel for collapse(2) schedule(static)
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
        #pragma omp parallel for schedule(static)
        for (int q = 0; q < N[j][2]/2+1; q++) {
          for (int k = ks + 2; k < ke; k++) {  
            const double complex z = nu + I * eta_m[j][q];
            gl[j][k-ks][q] = gl[j][k-ks-2][q] * (k - 4 + z) / (k + 3 - z);
          }
        }
        break;
      }
      default:
      {
        log_fatal("unsupported derivative = %d", cfg[j].derivative);
        exit(1);
      }
    }
  } 
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
  // ---------------------------------------------------------------------------
#ifndef COSMO2D_NOT_USE_SIMD 
  double** x_pow_nu = (double**) malloc2d(SIZE2, Nx);
  #pragma omp parallel for collapse(2) schedule(static)
  for (int j=0; j<SIZE2; j++) {
    for (int q=0; q<Nx; q++) {
      const double xr = x[Nx - 1 - q];
      x_pow_nu[j][q] = (cfg[j].nu == 1.0) ? xr : pow(xr, cfg[j].nu);
    }
  }
#endif

  for(int i=0; i<SIZE1; i++) {
    if (converged[i]) continue;
    #pragma omp parallel for collapse(2) schedule(static)
    for(int j=0; j<SIZE2; j++) {
      for (int k=ks; k<kmax; k++) { 
#ifdef _OPENMP
        const int id = omp_get_thread_num(); 
#else
        const int id = 0;
#endif 
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
          if (q > 0 && (q % 1024) == 0) {
            // recompute phasor exactly to prevent drift (numerical error)
            // if N/2 becomes >> 1000 (right now is <1000)
            // This is extra safety (paranoia!)
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

        // FFTW's new-array execute interface: The plan was created with
        // outfwd[0]/outbcw[0], but each OpenMP thread executes it on its 
        // own buffers outfwd[id]/outbcw[id]. Do not replace this with
        // fftw_execute(planb[j]), which would use the plan's original arrays.
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
        
        double* RESTRICT Fy_ijk = Fy[i][j][k];
        const double* RESTRICT ob = outbcw[id] + N[j][0];
        const double* RESTRICT xnu = x_pow_nu[j];
        
        v4d vpre = simde_mm256_set1_pd(prefactor); // [pf | pf | pf | pf]
        
        int q = 0;
        for (; q <= Nx - 4; q += 4) {
          v4d vob  = simde_mm256_loadu_pd(ob + q);  // ob[q..q+3]
          v4d vxnu = simde_mm256_loadu_pd(xnu + q); // xnu[q..q+3]
          // Two multiplies: (ob * prefactor) * xnu
          //   first:  vtmp = [ ob[q]*pf | ob[q+1]*pf | ... | ... ]
          //   second: vres = [vtmp[0]*xnu[q] | vtmp[1]*xnu[q+1] | ... | ...  ]
          v4d vtmp = simde_mm256_mul_pd(vob, vpre);
          v4d vres = simde_mm256_mul_pd(vtmp, vxnu);
          // Store 4 results back to Fy_ijk[q..q+3]
          simde_mm256_storeu_pd(Fy_ijk + q, vres);
        }
        for (; q < Nx; q++) { // Scalar tail
          Fy_ijk[q] = ob[q] * prefactor * xnu[q];
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

// ---------------------------------------------------------------------------
// Non-Limber galaxy clustering C_l via the FFTLog algorithm.
//
// At low multipoles (l < LMAX_NOLIMBER), the Limber approximation breaks
// down for galaxy clustering because the lens galaxy redshift distributions
// are narrow and the radial kernels oscillate on scales comparable to the
// kernel width. This function computes the exact (non-Limber) projection
// integral using FFTLog, which recasts the double-Bessel integral as a
// convolution evaluable via FFT.
//
// The result is combined with the Limber C_l to give the full answer:
//   Cl[i][l] = Cl_fftlog(P_lin) + Cl_limber(P_delta) - Cl_limber(P_lin)
// The last two terms (computed via C_gg_tomo_limber_linpsopt_nointerp)
// correct for the difference between the linear power spectrum used in
// FFTLog and the nonlinear power spectrum used in the Limber integral.
//
// Algorithm overview:
//   1. Build the log-spaced chi grid (chi_min..chi_max, dimensionless)
//   2. Evaluate the three radial weight functions per lens bin:
//        fx[i][0] = chi * n(z) * D(a) * (H/H0) * b1   (galaxy density)
//        fx[i][1] = -chi * n(z) * D(a) * (H/H0) * f    (RSD velocity)
//        fx[i][2] = (W_mag / fK / coverH0^2) * D        (magnification)
//      The magnification component is skipped when bmag = 0 for all bins.
//   3. Phase 1 (cfftlog_ells_p1): forward FFT of the radial functions
//      (ell-independent, done once)
//   4. Phase 2 (cfftlog_ells_p2): ell-dependent inverse transform,
//      called in blocks of BLOCK=16 multipoles with early termination
//   5. For each block, assemble the integrand:
//        F = Fy_density + Fy_RSD + bmag * l*(l+1) * Fy_mag / y^2
//      and integrate:
//        Cl_fftlog = (2/pi) * dlnk * sum_q[ F^2 * (k*c/H0)^3 * P_lin(k) ]
//   6. Combine: Cl = Cl_fftlog + Cl_limber(P_NL) - Cl_limber(P_lin)
//   7. Convergence: after each block, check |Cl_nolimber/Cl_limber - 1| < tol.
//      Once converged, fill remaining ells from the Limber table.
//
// Only auto-correlations (ni = nj) are supported.
//
// Parameters:
//   Cl  - output array Cl[nbins][LMAX_NOLIMBER]: non-Limber C_l per lens bin
//   tol - convergence tolerance for switching to Limber (typically 0.01)
// ---------------------------------------------------------------------------
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

  #pragma omp parallel for schedule(static)
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
    const int ke = (ks + BLOCK < limits.LMAX_NOLIMBER) ? ks + BLOCK : 
                                                         limits.LMAX_NOLIMBER;

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

      #pragma omp parallel for collapse(2) schedule(static)
      for (int k=ks; k<kk; k++) {
        for (int q=0; q<nchi; q++) {
          const double ell_prefactor = k * (k + 1.);
          const double ty    = y[i][k][q];
          const double k1cH0 = ty*real_coverH0;
          // check gbmag(0,i) (first argument)
          const double bmag = gbmag(0.,i);
          const double F = Fy[i][0][k][q] + Fy[i][1][k][q] + 
                           bmag*ell_prefactor*Fy[i][2][k][q]/(ty*ty);
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
      
      const double denom = C_gg_tomo_limber_nointerp((double) L, i, i, 0);
      if (fabs(denom) > 1e-300) {
        const double dev = Cl[i][L] / denom - 1.0;
        if (isfinite(dev) && fabs(dev) < tol) {
          converged[i] = 1;
          LMAX[i] = kk;
        }
      }
    }

    all_done = 1;
    for (int i=0; i <nbins; i++) {
      if (!converged[i]) all_done = 0;
    }
    ks = ke;
  }

  for (int i=0; i<nbins; i++) {
    for (int k=LMAX[i]; k<limits.LMAX_NOLIMBER; k++) {
      Cl[i][k] = (k > limits.LMIN_tab) ? C_gg_tomo_limber(k, i, i) :
                                         C_gg_tomo_limber_nointerp(k, i, i, 0);
    }
  }

  free((void*) toutfwd);
  free((void*) eta_m);
}
