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

#include "simde/x86/avx2.h"
#include "simde/x86/fma.h"

// Physics gates for the Limber integrands (0 or 1). include_HOD_GX is
// runtime-switchable (set_include_HOD_GX); the RSD gates are
// compile-time.
//
//   include_HOD_GX    = halo-model (HOD) galaxy power in the galaxy
//                       probes: gg reads p_gg/p_gm and gs reads p_gm
//                       from halo.c (Limber-only; no RSD, no one-loop
//                       bias; gs is NLA-only); the gk batched path
//                       still aborts. 0 by default.
//   include_halo_IA   = halo-model intrinsic alignments (Fortuna et al.
//                       2021; halo.c ia_* readers) in the Limber ss and
//                       gs engines: NLA 2-halo for red centrals times
//                       f_rc(a) and the k window, plus the satellite
//                       1-halo terms. NLA only, perturbative-bias
//                       galaxies only; ks, non-Limber gs and the
//                       scale-cut responses abort. Runtime
//                       (set_include_halo_IA); 0 by default.
//   include_RSD_GS/GK = add the W_RSD (redshift-space distortion)
//                       kernel to that probe's Limber integrand
//   include_RSD_GG    = same gate for gg; defaults to 1 so the Limber
//                       C_gg carries the RSD term the non-Limber
//                       C_cl_tomo always includes
//   include_RSD_GY    = never read (no gy probe in this file)
static int include_HOD_GX = 0; // 0 or 1
static int include_halo_IA = 0; // 0 or 1
static int include_RSD_GS = 0; // 0 or 1 
static int include_RSD_GG = 1; // 0 or 1 
static int include_RSD_GK = 0; // 0 or 1
static int include_RSD_GY = 0; // 0 or 1

// ---------------------------------------------------------------------------
// Runtime switch of include_HOD_GX (generic_interface.cpp
// init_include_HOD_GX; the likelihoods read the yaml key of the same
// name). The C_l^gg interpolation table keys its cache on the flag, so
// flipping it rebuilds the table on the next read.
// ---------------------------------------------------------------------------
void set_include_HOD_GX(const int flag)
{
  if (flag != 0 && flag != 1) {
    log_fatal("invalid include_HOD_GX = %d (0 or 1)", flag);
    exit(1);
  }
  include_HOD_GX = flag;
}

int get_include_HOD_GX(void)
{
  return include_HOD_GX;
}

// ---------------------------------------------------------------------------
// Runtime switch of include_halo_IA (generic_interface.cpp
// init_include_halo_IA; yaml key of the same name). The C_ss and C_gs
// interpolation tables key their caches on the flag and on
// nuisance.random_ia_halo.
// ---------------------------------------------------------------------------
void set_include_halo_IA(const int flag)
{
  if (flag != 0 && flag != 1) {
    log_fatal("invalid include_halo_IA = %d (0 or 1)", flag);
    exit(1);
  }
  include_halo_IA = flag;
}

int get_include_halo_IA(void)
{
  return include_halo_IA;
}

// refuse a path that has no halo-model IA implementation
static void halo_IA_unsupported(const char* where)
{
  if (1 == include_halo_IA) {
    log_fatal("include_halo_IA = 1 is not implemented in %s", where);
    exit(1);
  }
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

// ----------------------------------------------------------------------------
// ARCHITECTURE MAP: how a data-vector entry is computed.
//
// The probes are xy = ss (shear-shear), gs (galaxy-shear), gg (clustering),
// gk (galaxy x CMB lensing), ks (CMB lensing x shear), kk (CMB lensing auto).
//
// Real space (the public entry points the data vector calls):
//
//   xi_pm_tomo / w_gammat_tomo / w_gg_tomo / w_gk_tomo / w_ks_tomo
//     -> l = 1..LMIN_tab:        C_xy_tomo_limber_nointerp_batch
//                                (exact quadrature per integer multipole)
//     -> l = LMIN_tab..LMAX:     C_xy_tomo_limber builds the log-spaced
//                                C_l table (optionally: exact quadrature
//                                on a coarse grid -> cubic-spline
//                                upsampling onto the dense table)
//                                -> C_xy_tomo_limber_fill reads the
//                                table at every integer l (AVX2 batch)
//     -> Legendre sum over l = 1..LMAX against the bin-averaged kernels
//
// Limber engines (one chain per probe):
//
//   C_xy_tomo_limber_nointerp_ells
//     -> create_cosmo_nodes      (chi, D, H/H0, dchi/da at the
//                                Gauss-Legendre nodes; ell/bin
//                                independent, computed once; the lens
//                                bins of gs, gg, gk go through
//                                create_cosmo_nodes_lens, one rule on
//                                the n(z) support and one on the
//                                magnification foreground)
//     -> C_xy_tomo_limber_work   (precompute weights + kernels per node
//                                -> SIMD quadrature per (ell, pair))
//
// Non-Limber (gg and gs, l < LMAX_NOLIMBER, FKEM split):
//
//   C_cl_tomo / C_gs_tomo
//     -> radial kernels on the log-chi grid
//     -> cfftlog_ells_p1         (ell-independent forward FFT, once)
//     -> cfftlog_ells_p2         (ell-dependent inverse transform,
//                                blocks of 16 multipoles)
//     -> C_l = C^fftlog(P_lin) + C^Limber(P_NL) - C^Limber(P_lin)
//     -> early exit per bin/pair once the ratio to Limber converges;
//        the remaining multipoles take the Limber-path values
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
// Where the sqrt(16 ln 2) comes from: a real-space Gaussian beam
// exp(-theta^2/(2 sigma_b^2)) falls to half its peak at theta = FWHM/2,
//   exp(-(FWHM/2)^2 / (2 sigma_b^2)) = 1/2  ->  sigma_b = FWHM/sqrt(8 ln 2),
// and its harmonic transform is B_l = exp(-l(l+1) sigma_b^2 / 2). The
// code folds that 1/2 into the width: sigma^2 = sigma_b^2/2, i.e.
// sigma = FWHM/sqrt(16 ln 2).
//
// Parameters:
//   l - multipole moment
//
// Returns:
//   B_l for l inside [cmb.lk_wxk[RANGE_MIN], cmb.lk_wxk[RANGE_MAX]], the multipole range
//   used for the CMB lensing cross-correlations (gk, ks, kk); 0 outside it
// ---------------------------------------------------------------------------
double beam_cmb(
    const int l  // multipole moment
  )
{
  const double s = cmb.fwhm/sqrt(16.0*log(2.0));
  return ((l<cmb.lk_wxk[RANGE_MIN]) || (l>cmb.lk_wxk[RANGE_MAX])) ? 0.0 : exp(-l*(l+1.0)*s*s);
}

// ---------------------------------------------------------------------------
// HEALPix pixel window function.
//
// Returns the precomputed HEALPix pixel window function at multipole l,
// which accounts for the finite pixel size of the CMB convergence map.
// The window function is loaded at initialization into cmb.healpixwin[]
// with cmb.healpixwin_ncls entries (typically lmax+1).
//
// Parameters:
//   l - multipole moment
//
// Returns:
//   cmb.healpixwin[l] for l < cmb.healpixwin_ncls; 0 beyond the
//   precomputed range
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
// Used to guard the allocation and computation of FPTbias one-loop kernels
// (d1d2, d1s2, d1p3) and the higher-derivative counterterm (bk*k^2*PK) in
// the GS, GG and GK probes.
//
// Returns:
//   1 if one-loop galaxy bias corrections should be computed (at least
//   one bin has nuisance.gb[1][i] != 0), 0 otherwise
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
typedef simde__m256d v4d;   // 4 doubles, AVX2-width
typedef simde__m128d v2d;   // 2 doubles (SSE2)
typedef simde__m128i v4i;   // 4 int32s (SSE2) - used for SIMD gather indices

// ---------------------------------------------------------------------------
// Generic Limber table interpolation with SIMDe vector arithmetic.
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
//   the grid indices ic are data-dependent - different ells map to different
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
//
// Returns:
//   nothing; the interpolated values are written into
//   out[q][lmin..lmax-1] for every table q
// ---------------------------------------------------------------------------
void limber_fill_interp(
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
  // Vector constants: one copy of each scalar, broadcast to all 4 lanes
  // of a 256-bit register (a v4d holds 4 doubles side by side)
  const v4d va       = simde_mm256_set1_pd(a);       // ln(l_min) of the grid
  const v4d vinv_dx  = simde_mm256_set1_pd(inv_dx);  // 1 / grid spacing
  const v4d vzero    = simde_mm256_setzero_pd();     // lower index clamp: 0
  const v4d vmax_idx = simde_mm256_set1_pd((double)(n - 2)); // upper clamp
  const v4i vone     = simde_mm_set1_epi32(1);       // to form ic + 1
  // Why the clamp is n - 2 and not n - 1: linear interpolation reads the
  // PAIR (tab[ic], tab[ic + 1]), so a grid with n points has only n - 1
  // segments (slopes) between them, and the last valid left index is
  // ic = n - 2. Clamping to it makes a multipole beyond the grid follow
  // the last segment (edge extrapolation) instead of reading past the
  // end of the table.
  int l = lmin;
  for (; l <= lmax - 4; l += 4) { // 4 multipoles per iteration (AVX2 width)
    // load ln(l), ln(l+1), ln(l+2), ln(l+3) with one contiguous load
    v4d vlnell = simde_mm256_loadu_pd(ln_ell + l);
    // fractional grid position r = (ln(l) - a)/dx, all 4 lanes at once
    v4d vr = simde_mm256_mul_pd(simde_mm256_sub_pd(vlnell, va), vinv_dx);
    // i = floor(r): the grid cell each of the 4 multipoles falls into
    v4d vi = simde_mm256_floor_pd(vr);
    // clamp i to [0, n - 2] (see above); the clamp stays in double
    // precision because the min/max instructions act on doubles here
    v4d vicdb = simde_mm256_min_pd(simde_mm256_max_pd(vi, vzero), vmax_idx);
    // t = r - ic: fractional position inside the cell, in [0, 1) on the
    // interior; for a clamped index t leaves that range and the formula
    // below extrapolates the edge segment
    v4d vt = simde_mm256_sub_pd(vr, vicdb);
    // the 4 left indices as 32-bit integers (truncation is exact
    // because vicdb already holds whole numbers), and the 4 right
    // indices ic + 1
    v4i vic = simde_mm256_cvttpd_epi32(vicdb);
    v4i vicp1 = simde_mm_add_epi32(vic, vone);
    for (int q = 0; q < ntab; q++) {
      // fetch tab[q][ic] and tab[q][ic + 1] for the 4 lanes: the indices
      // differ per lane, so these are gather loads (the 8 is the index
      // scale, sizeof(double)), not contiguous vector loads
      v4d v0 = simde_mm256_i32gather_pd(tab[q], vic, 8);
      v4d v1 = simde_mm256_i32gather_pd(tab[q], vicp1, 8);
      // linear interpolation out = v0 + t*(v1 - v0) as one fused
      // multiply-add, stored back for the 4 multipoles at once
      simde_mm256_storeu_pd(out[q] + l,
        simde_mm256_fmadd_pd(vt, simde_mm256_sub_pd(v1, v0), v0));
    }
  }
  for (; l < lmax; l++) { // scalar tail: the last lmax % 4 multipoles
    const double r = (ln_ell[l] - a) * inv_dx;
    const int i = (int) floor(r);
    const int ic = i < 0 ? 0 : (i >= n - 1 ? n - 2 : i);
    const double t = r - ic;
    for (int q = 0; q < ntab; q++) {
      out[q][l] = tab[q][ic] + t * (tab[q][ic + 1] - tab[q][ic]);
    }
  }
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
//   KIA[0]  <- FPTIA.tab[0]  tt      (tidal-tidal, EE)
//   KIA[1]  <- FPTIA.tab[2]  ta_dE1  (tidal-density E-mode 1, EE)
//   KIA[2]  <- FPTIA.tab[3]  ta_dE2  (tidal-density E-mode 2, EE)
//   KIA[3]  <- FPTIA.tab[4]  ta      (tidal-alignment, EE)
//   KIA[4]  <- FPTIA.tab[6]  mixA    (mixed A, EE)
//   KIA[5]  <- FPTIA.tab[7]  mixB    (mixed B, EE)
//   KIA[6]  <- FPTIA.tab[8]  mixEE   (mixed EE)
//   KIA[7]  <- FPTIA.tab[1]  tt      (tidal-tidal, BB)
//   KIA[8]  <- FPTIA.tab[5]  ta      (tidal-alignment, BB)
//   KIA[9]  <- FPTIA.tab[9]  mix     (mixed, BB)
//
// GS (galaxy-shear): KIA[2..5] maps the 4 FPTIA kernels needed for GGL
//   KIA[2]  <- FPTIA.tab[6]  mixA    (mixed A)
//   KIA[3]  <- FPTIA.tab[7]  mixB    (mixed B)
//   KIA[4]  <- FPTIA.tab[2]  ta_dE1  (tidal-density 1)
//   KIA[5]  <- FPTIA.tab[3]  ta_dE2  (tidal-density 2)
//   (GS has no BB mode - galaxy density is spin-0)
//
// GS_BIAS (galaxy-shear one-loop bias): KIA[6..8] maps FPTbias correlators
//   KIA[6]  <- FPTbias.tab[0]  d1d2  (delta x delta_2 correlator)
//   KIA[7]  <- FPTbias.tab[2]  d1s2  (delta x s_2 correlator)
//   KIA[8]  <- FPTbias.tab[5]  d1p3  (delta x psi_3 correlator)
//
// These mappings are used with the LERP macro in the precompute loops:
//   for (int m = 0; m < N; m++)
//     KIA[offset+m][i][p] = g4 * LERP(FPTIA.tab[SRC[m]], idx, dr);
// ---------------------------------------------------------------------------
static const int SS_IA_SRC[] = {0, 2, 3, 4, 6, 7, 8, 1, 5, 9}; 
static const int GS_IA_SRC[] = {6, 7, 2, 3}; // KIA[2..5] <- FPTIA.tab
static const int GS_BIAS_SRC[] = {0, 2, 5};  // KIA[6..8] <- FPTbias.tab

// -------------------------------------------------------------------------
// Forward declarations of the per-probe batch table readers (defined next
// to their probes below). The real-space functions above the definitions
// (xi_pm_tomo, w_gammat_tomo, ...) call them to fill C_l at every integer
// multipole from the cached interpolation tables.
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
// Legendre sums of the real-space functions below, for every spectrum nz
// and every theta bin i:
//
//   w_vec[nz*ntheta + i] = sum_{l=lmin}^{lmax-1} Pl[i][l] * Cl[nz][l]
//
// The only sum is over l: every (nz, i) has its own.
//
// Why a scalar one-output loop is slow: at LMAX = 75000 a
// C_l array and a kernel array are 600 kB each, far more than the L1 cache
// holds. The reference loop makes one full pass over l per (nz, i), so it
// reads the whole of Cl[nz] again for every theta bin and the whole of
// Pl[i] again for every spectrum: two values fetched from memory per
// multiply-add, and the loop waits on memory, not on arithmetic.
//
// The default loop takes 4 spectra and 4 theta bins in one pass over l: 8
// values fetched per 16 multiply-adds, each Cl[nz] read ntheta/4 times and
// each Pl[i] read NSIZE/4 times. Each of the 16 sums adds the same products
// in the same order as the reference loop does for that (nz, i), so the
// results are bitwise those of the reference.
//
// Why 4 x 4: the 16 sums and the 8 fetched values must stay in the CPU's
// vector registers; a larger group spills them to memory.
//
// Thread safety: call outside parallel regions.
//
// Parameters:
//   NSIZE  - number of spectra (rows of Cl)
//   ntheta - number of theta bins (rows of Pl)
//   lmin   - first multipole of the sums
//   lmax   - one past the last multipole of the sums
//   Pl     - [ntheta][lmax] bin-averaged Legendre kernel
//   Cl     - [NSIZE][lmax] C_l at every integer l
//   w_vec  - output [NSIZE*ntheta], indexed nz*ntheta + i
// ---------------------------------------------------------------------------
void legendre_sums(
    const int NSIZE,
    const int ntheta,
    const int lmin,
    const int lmax,
    double** Pl,
    double** Cl,
    double* w_vec
  )
{
  // nz and i are the first spectrum and the first theta bin of the group,
  // which covers spectra nz .. nz+3 and theta bins i .. i+3
  #pragma omp parallel for collapse(2) schedule(static)
  for (int nz=0; nz<NSIZE; nz+=4) {
    for (int i=0; i<ntheta; i+=4) {
      // Past the end (NSIZE or ntheta not a multiple of 4) the extra slots
      // point at the last valid spectrum / theta bin: the loop below always
      // reads valid memory, and those repeats are computed but never stored
      const int nz1 = (nz + 1 < NSIZE) ? nz + 1 : NSIZE - 1;
      const int nz2 = (nz + 2 < NSIZE) ? nz + 2 : NSIZE - 1;
      const int nz3 = (nz + 3 < NSIZE) ? nz + 3 : NSIZE - 1;
      const int i1  = (i + 1 < ntheta) ? i + 1 : ntheta - 1;
      const int i2  = (i + 2 < ntheta) ? i + 2 : ntheta - 1;
      const int i3  = (i + 3 < ntheta) ? i + 3 : ntheta - 1;

      const double* restrict cl0 = Cl[nz];   // C_l of spectra nz .. nz+3
      const double* restrict cl1 = Cl[nz1];
      const double* restrict cl2 = Cl[nz2];
      const double* restrict cl3 = Cl[nz3];
      const double* restrict pl0 = Pl[i];    // kernel of theta bins i .. i+3
      const double* restrict pl1 = Pl[i1];
      const double* restrict pl2 = Pl[i2];
      const double* restrict pl3 = Pl[i3];

      // sum<a><b>: the sum of spectrum nz + a and theta bin i + b. Sixteen
      // separate sums over l; nothing is added across spectra or bins,
      // they only share the reads of cl and pl
      double sum00 = 0.0, sum01 = 0.0, sum02 = 0.0, sum03 = 0.0;
      double sum10 = 0.0, sum11 = 0.0, sum12 = 0.0, sum13 = 0.0;
      double sum20 = 0.0, sum21 = 0.0, sum22 = 0.0, sum23 = 0.0;
      double sum30 = 0.0, sum31 = 0.0, sum32 = 0.0, sum33 = 0.0;

      #pragma omp simd reduction(+:sum00,sum01,sum02,sum03,\
                                   sum10,sum11,sum12,sum13,\
                                   sum20,sum21,sum22,sum23,\
                                   sum30,sum31,sum32,sum33)
      for (int l=lmin; l<lmax; l++) {
        sum00 += pl0[l] * cl0[l];
        sum01 += pl1[l] * cl0[l];
        sum02 += pl2[l] * cl0[l];
        sum03 += pl3[l] * cl0[l];

        sum10 += pl0[l] * cl1[l];
        sum11 += pl1[l] * cl1[l];
        sum12 += pl2[l] * cl1[l];
        sum13 += pl3[l] * cl1[l];

        sum20 += pl0[l] * cl2[l];
        sum21 += pl1[l] * cl2[l];
        sum22 += pl2[l] * cl2[l];
        sum23 += pl3[l] * cl2[l];

        sum30 += pl0[l] * cl3[l];
        sum31 += pl1[l] * cl3[l];
        sum32 += pl2[l] * cl3[l];
        sum33 += pl3[l] * cl3[l];
      }

      // store the sums of the spectra and theta bins that exist
      const double sum[4][4] = {{sum00, sum01, sum02, sum03},
                                {sum10, sum11, sum12, sum13},
                                {sum20, sum21, sum22, sum23},
                                {sum30, sum31, sum32, sum33}};
      for (int a=0; a<4; a++) {
        for (int b=0; b<4; b++) {
          if (nz + a < NSIZE && i + b < ntheta) {
            w_vec[(nz + a)*ntheta + (i + b)] = sum[a][b];
          }
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Legendre sums of xi_pm_tomo, for every shear pair nz and theta bin i:
//
//   xip[nz*ntheta + i] = sum_l Glp[i][l] * (Cl_EE[nz][l] + Cl_BB[nz][l])
//   xim[nz*ntheta + i] = sum_l Glm[i][l] * (Cl_EE[nz][l] - Cl_BB[nz][l])
//
// Same idea as legendre_sums (see the note there): the reference loop
// makes one full pass over l per (nz, i), fetching
// four values (EE, BB, Gl+, Gl-) for two multiply-adds and recomputing
// EE + BB and EE - BB for every theta bin. The default loop takes 2 pairs
// and 4 theta bins in one pass: 12 values fetched per 16 multiply-adds,
// and EE +- BB computed once per pair. Each sum adds the same products in
// the same order as the reference, so the results are bitwise identical.
//
// Why 2 x 4: each pair brings two C_l arrays and each theta bin two
// kernels, so 2 x 4 already holds 16 sums, 12 fetched values and 4
// temporaries in the vector registers. Measured against 1x4, 2x2 and 4x2
// at 10 to 55 pairs: the fastest or within noise of it at 4 threads.
//
// Thread safety: call outside parallel regions.
//
// Parameters:
//   NSIZE  - number of shear pairs (rows of Cl_EE, Cl_BB)
//   ntheta - number of theta bins (rows of Glp, Glm)
//   lmin   - first multipole of the sums
//   lmax   - one past the last multipole of the sums
//   Glp    - [ntheta][lmax] bin-averaged kernel of xi_+
//   Glm    - [ntheta][lmax] bin-averaged kernel of xi_-
//   Cl_EE  - [NSIZE][lmax] E-mode C_l at every integer l
//   Cl_BB  - [NSIZE][lmax] B-mode C_l at every integer l
//   xip    - output [NSIZE*ntheta], indexed nz*ntheta + i
//   xim    - output [NSIZE*ntheta], indexed nz*ntheta + i
// ---------------------------------------------------------------------------
void legendre_sums_xipm(
    const int NSIZE,
    const int ntheta,
    const int lmin,
    const int lmax,
    double** Glp,
    double** Glm,
    double** Cl_EE,
    double** Cl_BB,
    double* xip,
    double* xim
  )
{
  // nz and i are the first pair and the first theta bin of the group,
  // which covers pairs nz, nz+1 and theta bins i .. i+3
  #pragma omp parallel for collapse(2) schedule(static)
  for (int nz=0; nz<NSIZE; nz+=2) {
    for (int i=0; i<ntheta; i+=4) {
      // past the end: repeats of the last valid pair / theta bin,
      // computed but never stored (see legendre_sums)
      const int nz1 = (nz + 1 < NSIZE) ? nz + 1 : NSIZE - 1;
      const int i1  = (i + 1 < ntheta) ? i + 1 : ntheta - 1;
      const int i2  = (i + 2 < ntheta) ? i + 2 : ntheta - 1;
      const int i3  = (i + 3 < ntheta) ? i + 3 : ntheta - 1;

      const double* restrict ee0 = Cl_EE[nz];   // pair nz
      const double* restrict bb0 = Cl_BB[nz];
      const double* restrict ee1 = Cl_EE[nz1];  // pair nz+1
      const double* restrict bb1 = Cl_BB[nz1];
      const double* restrict gp0 = Glp[i];      // Gl+ of theta bins i .. i+3
      const double* restrict gp1 = Glp[i1];
      const double* restrict gp2 = Glp[i2];
      const double* restrict gp3 = Glp[i3];
      const double* restrict gm0 = Glm[i];      // Gl- of theta bins i .. i+3
      const double* restrict gm1 = Glm[i1];
      const double* restrict gm2 = Glm[i2];
      const double* restrict gm3 = Glm[i3];

      // xip<a><b>, xim<a><b>: the sums of pair nz + a and theta bin i + b.
      // Sixteen separate sums over l; nothing is added across pairs or bins
      double xip00 = 0.0, xip01 = 0.0, xip02 = 0.0, xip03 = 0.0;
      double xip10 = 0.0, xip11 = 0.0, xip12 = 0.0, xip13 = 0.0;
      double xim00 = 0.0, xim01 = 0.0, xim02 = 0.0, xim03 = 0.0;
      double xim10 = 0.0, xim11 = 0.0, xim12 = 0.0, xim13 = 0.0;

      #pragma omp simd reduction(+:xip00,xip01,xip02,xip03,\
                                   xip10,xip11,xip12,xip13,\
                                   xim00,xim01,xim02,xim03,\
                                   xim10,xim11,xim12,xim13)
      for (int l=lmin; l<lmax; l++) {
        const double sum_eb0 = ee0[l] + bb0[l];  // EE + BB of pair nz
        const double dif_eb0 = ee0[l] - bb0[l];  // EE - BB of pair nz
        const double sum_eb1 = ee1[l] + bb1[l];  // EE + BB of pair nz+1
        const double dif_eb1 = ee1[l] - bb1[l];  // EE - BB of pair nz+1

        xip00 += gp0[l] * sum_eb0;
        xip01 += gp1[l] * sum_eb0;
        xip02 += gp2[l] * sum_eb0;
        xip03 += gp3[l] * sum_eb0;

        xip10 += gp0[l] * sum_eb1;
        xip11 += gp1[l] * sum_eb1;
        xip12 += gp2[l] * sum_eb1;
        xip13 += gp3[l] * sum_eb1;

        xim00 += gm0[l] * dif_eb0;
        xim01 += gm1[l] * dif_eb0;
        xim02 += gm2[l] * dif_eb0;
        xim03 += gm3[l] * dif_eb0;

        xim10 += gm0[l] * dif_eb1;
        xim11 += gm1[l] * dif_eb1;
        xim12 += gm2[l] * dif_eb1;
        xim13 += gm3[l] * dif_eb1;
      }

      // store the sums of the pairs and theta bins that exist
      const double sp[2][4] = {{xip00, xip01, xip02, xip03},
                               {xip10, xip11, xip12, xip13}};
      const double sm[2][4] = {{xim00, xim01, xim02, xim03},
                               {xim10, xim11, xim12, xim13}};
      for (int a=0; a<2; a++) {
        for (int b=0; b<4; b++) {
          if (nz + a < NSIZE && i + b < ntheta) {
            xip[(nz + a)*ntheta + (i + b)] = sp[a][b];
            xim[(nz + a)*ntheta + (i + b)] = sm[a][b];
          }
        }
      }
    }
  }
}

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
// bin edges (xmin = cos(theta_min), xmax = cos(theta_max), following
// set_bin_average in basics.c: the min/max names track the theta edges,
// so xmin > xmax numerically). This replaces
// the naive point-evaluation J_0/J_4 Hankel transform with an exact bin average.
//
// The C_l array is filled in two stages:
//   1. Low-ell (l = 1..LMIN_tab): direct quadrature via _nointerp (or batch)
//   2. High-ell (l = LMIN_tab..LMAX): fast interpolation from the cached
//      log-spaced table via C_ss_tomo_limber_fill with AVX2 gather
//
// The final Hankel sum over ~100k multipoles is legendre_sums_xipm (above):
// 2 pairs x 4 theta bins per pass over l, SIMD-vectorized via #pragma omp
// simd with one reduction accumulator per (pair, theta bin, +/-).
//
// Cache invalidation:
// recomputes when cosmology, shear photo-z, IA, shear
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

  // -------------------------------------------------------------------------
  // The cache mechanism (every cached function in this file follows it).
  //
  // Each parameter group in structs.h (cosmology, nuisance.*, redshift.*,
  // Ntable, cmb) carries a uint64 tag named random/random_*. The set_*
  // functions of the C++ interface (generic_interface) draw a fresh
  // random uint64 into the tag whenever they change that group. A cached
  // function keeps the tags it last computed with in a static uint64
  // cache[] and compares with fdiff2 (plain uint64 inequality): any
  // mismatch means that group changed after the cached values were built.
  //
  // Static storage zero-initializes cache[] and the table pointers, so
  // the first call always rebuilds (NULL pointers) and refills.
  //
  // Two tiers, one per if-block below:
  //   geometry tier - allocations and the Gl kernels, which depend only
  //     on the angular binning and Ntable.LMAX: rebuilt when a pointer
  //     is NULL or Ntable.random changed;
  //   physics tier - the C_l values and their Legendre sums: refilled
  //     when any physics tag (cosmology, photo-z, IA, n(z)) or
  //     Ntable.random changed.
  // -------------------------------------------------------------------------
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
    //   xmin = cos(theta_min), xmax = cos(theta_max) (set_bin_average's
    //   convention: the names track the theta edges, and cosine reverses
    //   order, so xmin > xmax numerically):
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
    //   The un-integrated kernel (what the big bracket below is the
    //   antiderivative of; upper signs xi_+, lower signs xi_-):
    //
    //     G_l^{+/-}(x) = l^2 (l^2-1)/2 * P_l(x)
    //                    - l (l-1) * x P_l'(x)
    //                    + (4-l) * P_l''(x)
    //                    + (l+2) * x P_{l-1}''(x)
    //                    +/- 2 [ (l-1) x P_l''(x) - (l+2) P_{l-1}''(x) ]
    //
    //   and it satisfies (checked term by term at l = 2 and l = 3)
    //
    //     G_l^{+/-}(x) = (1/2) * [(l+2)!/(l-2)!] * d^l_{2,+/-2}(x),
    //
    //   so prefactor * G_l^{+/-} = [(2l+1)/(4 pi)] * d^l_{2,+/-2} times
    //   the spin-2 conversion (l+2)!/(l-2)! / [l(l+1)]^2, which -> 1 at
    //   large l.
    //
    // ANALYTIC BIN INTEGRATION:
    //   Every term of G_l^{+/-} has a closed-form antiderivative via the
    //   Legendre identities
    //
    //     integral P_l(x) dx = [P_{l+1}(x) - P_{l-1}(x)] / (2l+1)
    //     integral x*P_l'(x) dx = x*P_l(x) - integral P_l(x) dx
    //     integral P_l''(x) dx = P_l'(x)
    //     integral x*P_l''(x) dx = x*P_l'(x) - P_l(x)
    //
    //   Term -> antiderivative map onto the seven coefficient lines of
    //   the code below (line n = n-th summand inside the bracket):
    //
    //     l^2(l^2-1)/2 P_l - l(l-1) x P_l'  ->  lines 1-3: integrate with
    //       the first two identities, then regroup with the recurrence
    //       x*P_l = [(l+1)*P_{l+1} + l*P_{l-1}]/(2l+1) into the
    //       P_{l-1} / x*P_l / P_{l+1} split the code uses (the split is
    //       not unique; any regrouping differs by that recurrence)
    //     (4-l)   P_l''            ->  line 4:  (4-l)   * dP_l
    //     (l+2)   x P_{l-1}''      ->  line 5:  (l+2)*(x dP_{l-1} - P_{l-1})
    //     +/-2(l-1) x P_l''        ->  line 6:  +/-2(l-1) * (x dP_l - P_l)
    //     -/+2(l+2) P_{l-1}''      ->  line 7:  -/+2(l+2) * dP_{l-1}
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
    //   Pmin[i][l+/-1], Pmax[i][l+/-1] = P_{l+/-1} at bin edges
    //   dPmin[i][l], dPmax[i][l]     = P_l' at bin edges
    //   xmin[i], xmax[i]             = cos(theta_min), cos(theta_max)
    //   (xmin - xmax) in denominator = bin width in cos(theta), positive
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

    // init static vars. Z1(nz)/Z2(nz) (redshift_spline.c) map the flat
    // shear pair index nz = 0..shear_Npowerspectra-1 to its two source
    // bins; N_shear(ni, nj) below is the inverse map.
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
    legendre_sums_xipm(NSIZE, Ntable.Ntheta, lmin, Ntable.LMAX,
                       Glpm[0], Glpm[1], Cl[0], Cl[1], xipm[0], xipm[1]);
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
  const int ntomo = N_shear(ni, nj); // flat pair index of (ni, nj)
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
// (spin-2 field x spin-0 field), computed from associated Legendre
// polynomials at the bin edges following Kilbinger+ (2017).
//
// The C_l array is filled via two paths depending on the limber flag:
//   limber = 1: full Limber approximation
//     1. Low-ell (l = 1..LMIN_tab-1): batch quadrature via
//        C_gs_tomo_limber_nointerp_batch
//     2. High-ell (l = LMIN_tab..LMAX-1): interpolation of the cached
//        log-spaced table via C_gs_tomo_limber_fill (AVX2 gather)
//   limber = 0: the non-Limber C_gs_tomo for l < LMAX_NOLIMBER (it already
//     continues converged pairs with the Limber values), then
//     C_gs_tomo_limber_fill for l = LMAX_NOLIMBER..LMAX-1
// The data vector selects the path with like.adopt_limber[LIMBER_GS] (yaml key
// adopt_limber_gs; 1 by default).
//
// The final Hankel sum is SIMD-vectorized via #pragma omp simd.
//
// Only lens-source pairs with redshift overlap contribute (test_zoverlap);
// non-overlapping pairs return 0.
//
// Cache invalidation:
// recomputes when cosmology, photo-z (shear or clustering),
// IA, redshift distributions, Ntable, galaxy bias parameters, or the limber
// flag change. The flag is part of the cache key so that one process can
// switch between the two paths (tests/test_nonlimber_ggl.py does).
//
// Parameters:
//   nt     - angular bin index (0..Ntable.Ntheta-1)
//   ni     - lens redshift bin
//   nj     - source redshift bin
//   limber - 1 for full Limber; 0 for non-Limber below LMAX_NOLIMBER
//
// Returns:
//   gamma_t(theta_nt) for the (ni, nj) pair; 0 for a pair excluded by
//   test_zoverlap
// ---------------------------------------------------------------------------
double w_gammat_tomo(
    const int nt,     // angular bin index (0..Ntheta-1)
    const int ni,     // lens redshift bin
    const int nj,     // source redshift bin
    const int limber  // 1 = full Limber, 0 = non-Limber FFTLog + Limber hybrid
  )
{
  static double** Pl = NULL;
  static double* w_vec = NULL;
  static double** Cl = NULL;
  static double* lnell = NULL;
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static int cache_limber = -1; // limber flag the cached w_vec was built with

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
    Pl = (double**) malloc2d(Ntable.Ntheta, Ntable.LMAX);
    
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
    //   where xmin = cos(theta_min), xmax = cos(theta_max)
    //   (set_bin_average's convention: min/max track the theta edges,
    //   so xmin > xmax numerically).
    //
    // ANALYTIC BIN INTEGRATION:
    //   The associated Legendre polynomial of order 2 is
    //     P_l^2(x) = (1-x^2) * P_l''(x),
    //   and Legendre's differential equation turns that into first
    //   derivatives and below:
    //     (1-x^2) P_l''(x) = 2x P_l'(x) - l(l+1) P_l(x).
    //   Integrating with
    //     integral x*P_l'(x) dx = x*P_l - integral P_l dx
    //     integral P_l(x) dx    = [P_{l+1} - P_{l-1}] / (2l+1)
    //   and regrouping with (2l+1)*x*P_l = (l+1)*P_{l+1} + l*P_{l-1}
    //   gives the antiderivative in the form the code uses:
    //
    //     integral P_l^2(x) dx = (l+2/(2l+1)) * P_{l-1}(x)
    //                          + (2-l) * x * P_l(x)
    //                          - 2/(2l+1) * P_{l+1}(x)
    //
    //   The bin-averaged kernel is then the difference of these
    //   antiderivatives evaluated at the two bin edges, divided by
    //   (xmin - xmax).
    //
    // NOTATION:
    //   Pmin[i][l] = P_l(xmin[i])       Pmax[i][l] = P_l(xmax[i])
    //   xmin[i] = cos(theta_min)        xmax[i] = cos(theta_max)
    //   (xmin - xmax) in denominator = bin width in cos(theta), positive
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
      fdiff2(cache[7], nuisance.random_galaxy_bias) ||
      cache_limber != limber)
  {
    const int lmin = 1;
    for (int i=0; i<NSIZE; i++) {
      for (int l=0; l<lmin; l++) {
        Cl[i][l] = 0.0;
      }
    }

    // init static vars. ZL(nz)/ZS(nz) (redshift_spline.c) map the flat
    // ggl pair index nz = 0..ggl_Npowerspectra-1 to its lens and source
    // bins; N_ggl(ni, nj) below is the inverse map (-1 for pairs the
    // data vector excludes).
    (void) C_gs_tomo_limber((double) limits.LMIN_tab + 1, ZL(0), ZS(0));
    if (1 == limber) {
      C_gs_tomo_limber_nointerp_batch(lmin, limits.LMIN_tab, NSIZE, Cl);
      #pragma omp parallel for schedule(static)
      for (int nz = 0; nz < NSIZE; nz++) {
        C_gs_tomo_limber_fill(nz, limits.LMIN_tab, Ntable.LMAX, lnell, Cl[nz]);
      }
    }
    else {
      const double tolerance = 0.01;
      C_gs_tomo(Cl, tolerance);
      #pragma omp parallel for schedule(static)
      for (int nz = 0; nz < NSIZE; nz++) { // LIMBER PART
        C_gs_tomo_limber_fill(nz, limits.LMAX_NOLIMBER, Ntable.LMAX, lnell, Cl[nz]);
      }
    }
    legendre_sums(NSIZE, Ntable.Ntheta, lmin, Ntable.LMAX, Pl, Cl, w_vec);
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_photoz_clustering;
    cache[3] = nuisance.random_ia;
    cache[4] = redshift.random_shear;
    cache[5] = redshift.random_clustering;
    cache[6] = Ntable.random;
    cache[7] = nuisance.random_galaxy_bias;
    cache_limber = limber;
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
// (spin-0 x spin-0), computed from Legendre polynomials at the bin edges.
//
// The C_l array is filled via two paths depending on the limber flag:
//   limber = 1: full Limber approximation
//     1. Low-ell (l = 1..LMIN_tab-1): batch quadrature via
//        C_gg_tomo_limber_nointerp_batch
//     2. High-ell (l = LMIN_tab..LMAX-1): C_gg_tomo_limber_fill
//   limber = 0: non-Limber FFTLog (C_cl_tomo) for l < LMAX_NOLIMBER,
//     then Limber fill for l >= LMAX_NOLIMBER
// The data vector selects the path with like.adopt_limber[LIMBER_GG] (yaml key
// adopt_limber_gg; 0 by default in the real-space projects).
//
// Only auto-correlations (ni = nj) are supported.
//
// Cache invalidation:
// recomputes when cosmology, clustering photo-z,
// clustering redshift distribution, Ntable, galaxy bias, or the limber
// flag change. The flag is part of the cache key so that one process can
// switch between the two paths (tests/test_nonlimber_gg.py does).
//
// Parameters:
//   nt     - angular bin index (0..Ntable.Ntheta-1)
//   ni     - first lens redshift bin
//   nj     - second lens redshift bin (must equal ni)
//   limber - 1 for full Limber; 0 for non-Limber below LMAX_NOLIMBER
//
// Returns:
//   w(theta_nt) for the auto pair (ni, ni)
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
  static int cache_limber = -1; // limber flag the cached w_vec was built with

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
    //   where P_l(x) is the ordinary Legendre polynomial (spin-0 x spin-0,
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
    //   xmin[i] = cos(theta_min)        xmax[i] = cos(theta_max)
    //   (xmin - xmax) in denominator = bin width in cos(theta), positive
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
      fdiff2(cache[4], nuisance.random_galaxy_bias) ||
      // redshift.random_shear: with magnification bias on, amax_lens
      // (redshift_spline.c) ends the lens range at the source z_min,
      // so a new source n(z) alone moves these integrals
      fdiff2(cache[5], redshift.random_shear) ||
      cache_limber != limber)
  {
    const int lmin = 1;
    for (int i=0; i<NSIZE; i++) {
      for (int l=0; l<lmin; l++) {
        Cl[i][l] = 0.0;
      }
    }               
    (void) C_gg_tomo_limber((double) limits.LMIN_tab + 1, 0, 0); // init static vars
    if (1 == limber) {
      C_gg_tomo_limber_nointerp_batch(lmin, limits.LMIN_tab, NSIZE, Cl);
      #pragma omp parallel for schedule(static)
      for (int nz = 0; nz < NSIZE; nz++) {
        C_gg_tomo_limber_fill(nz, limits.LMIN_tab, Ntable.LMAX, lnell, Cl[nz]);
      }
    }
    else {
      // Switch-to-Limber tolerance of the non-Limber gg path. The
      // per-bin early exit of C_cl_tomo hands the multipoles above its
      // switch point l_s to the Limber table while the exact C_l still
      // differs from Limber by up to the tolerance. That difference is
      // real beyond-Limber and RSD power (it scales as the extended-
      // Limber correction (chi0/sigma_chi)^2/(l + 0.5)^2 of each lens
      // kernel), so the tolerance bounds a physical modeling step at
      // l_s, not a numerical one: at 0.01 the step reaches 0.25-0.96%
      // of C_l at l_s = 32-80 (lsst_y1) and l_s = 32-112 (des_y3).
      // 0.002 keeps the step at the size of the residual the split
      // itself leaves at l = 149, at roughly twice the FFTLog cost.
      const double tolerance = 0.002;
      C_cl_tomo(Cl, tolerance);
      #pragma omp parallel for schedule(static)
      for (int nz=0; nz<NSIZE; nz++) { // LIMBER PART
        C_gg_tomo_limber_fill(nz, limits.LMAX_NOLIMBER, Ntable.LMAX, lnell, Cl[nz]);
      }
    }
    legendre_sums(NSIZE, Ntable.Ntheta, lmin, Ntable.LMAX, Pl, Cl, w_vec);
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_clustering;
    cache[2] = redshift.random_clustering;
    cache[3] = Ntable.random;
    cache[4] = nuisance.random_galaxy_bias;
    cache[5] = redshift.random_shear;
    cache_limber = limber;
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
// polynomial kernel (same kernel as w_gg - spin-0 x spin-0):
//
//   w_gk(theta_i) = sum_l Pl(i,l) * C_l^gk
//
// The C_l array is filled in two stages:
//   1. Low-ell (l = 1..LMIN_tab-1): batch quadrature via
//      C_gk_tomo_limber_nointerp_batch
//   2. High-ell (l = LMIN_tab..LMAX-1): interpolation of the cached
//      log-spaced table via C_gk_tomo_limber_fill (AVX2 gather)
// and then multiplied by the CMB filter cmbf[l] = beam_cmb(l), times
// w_pixel(l) when cmb.healpixwin_ncls > 0.
//
// No intrinsic alignment contribution: neither field is a galaxy shape
// (galaxy density x CMB convergence). One lens bin index only (no source
// bin - the CMB is a single source plane).
//
// Cache invalidation:
// recomputes when cosmology, clustering photo-z,
// clustering redshift distribution, Ntable, galaxy bias, or the CMB
// configuration (cmb.random) change.
//
// Parameters:
//   nt     - angular bin index (0..Ntable.Ntheta-1)
//   ni     - lens redshift bin
//   limber - 1 for full Limber (only supported option; 0 exits with error)
//
// Returns:
//   w_gk(theta_nt) for lens bin ni
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
    Pl = (double**) malloc2d(Ntable.Ntheta, Ntable.LMAX);

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
    // Bin-averaged Hankel transform kernel Pl for w_gk(theta)
    // (galaxy-CMB lensing).
    //
    // MOTIVATION:
    //   Galaxy density and CMB convergence are both spin-0 fields, so the
    //   cross-correlation function has the same Legendre expansion as
    //   galaxy clustering:
    //     w_gk(theta) = sum_l (2l+1)/(4*pi) * C_l^gk * P_l(cos(theta))
    //   where P_l(x) is the ordinary Legendre polynomial (spin-0 x spin-0,
    //   no 1/[l(l+1)] spin factor). Only the C_l summed against differs
    //   (C_l^gk instead of C_l^gg).
    //
    //   Bin-averaging over [theta_min, theta_max]:
    //
    //     Pl(i,l) = 1/(xmin - xmax) * integral_{xmax}^{xmin} P_l(x) dx
    //
    //   with x = cos(theta), so xmin = cos(theta_min), xmax = cos(theta_max)
    //   (set_bin_average's convention: the min/max names track the theta
    //   edges, so xmin > xmax numerically).
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
    //   xmin[i] = cos(theta_min)        xmax[i] = cos(theta_max)
    //   (xmin - xmax) in denominator = bin width in cos(theta), positive
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
      fdiff2(cache[5], cmb.random) ||
      // redshift.random_shear: with magnification bias on, amax_lens
      // (redshift_spline.c) ends the lens range at the source z_min,
      // so a new source n(z) alone moves these integrals
      fdiff2(cache[6], redshift.random_shear))
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
      C_gk_tomo_limber_nointerp_batch(lmin, limits.LMIN_tab, NSIZE, Cl);
      #pragma omp parallel for schedule(static)
      for (int nz=0; nz<NSIZE; nz++) {
        C_gk_tomo_limber_fill(nz, limits.LMIN_tab, Ntable.LMAX, lnell, Cl[nz]);
      }
      #pragma omp parallel for collapse(2) schedule(static)
      for (int nz=0; nz<NSIZE; nz++) {
        for (int l=lmin; l<Ntable.LMAX; l++) {
          Cl[nz][l] *= cmbf[l]; // multiply by CMB beam filter
        }
      }
    }
    else {
      log_fatal("NonLimber not implemented");
      exit(1);
    }
    legendre_sums(NSIZE, Ntable.Ntheta, lmin, Ntable.LMAX, Pl, Cl, w_vec);
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_clustering;
    cache[2] = redshift.random_clustering;
    cache[3] = Ntable.random;
    cache[4] = nuisance.random_galaxy_bias;
    cache[5] = cmb.random;
    cache[6] = redshift.random_shear;
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
//   1. Low-ell (l = 1..LMIN_tab-1): batch quadrature via
//      C_ks_tomo_limber_nointerp_batch
//   2. High-ell (l = LMIN_tab..LMAX-1): interpolation of the cached
//      log-spaced table via C_ks_tomo_limber_fill (AVX2 gather)
// and then multiplied by the CMB filter cmbf[l] = beam_cmb(l), times
// w_pixel(l) when cmb.healpixwin_ncls > 0.
//
// Includes the NLA intrinsic alignment contribution (C1 * W_source x
// W_k_cmb). One source bin index only (the CMB is a single lens plane).
//
// Cache invalidation:
// recomputes when cosmology, shear photo-z, IA,
// shear redshift distribution, Ntable, or the CMB configuration
// (cmb.random) change.
//
// Parameters:
//   nt     - angular bin index (0..Ntable.Ntheta-1)
//   ni     - source redshift bin
//   limber - 1 for full Limber (only supported option; 0 exits with error)
//
// Returns:
//   w_ks(theta_nt) for source bin ni
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
    Pl = (double**) malloc2d(Ntable.Ntheta, Ntable.LMAX);

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
    // Bin-averaged Hankel transform kernel Pl for w_ks(theta)
    // (CMB lensing-shear).
    //
    // MOTIVATION:
    //   The CMB convergence (spin-0) crossed with the shear field (spin-2)
    //   has the same harmonic expansion as galaxy-shear gamma_t(theta):
    //     w_ks(theta) = sum_l (2l+1)/(4*pi*l*(l+1)) * C_l^ks * P_l^2(cos(theta))
    //   where P_l^2(x) is the associated Legendre polynomial of degree l,
    //   order 2, and the prefactor 1/[l(l+1)] comes from the single spin-2
    //   shear field (contrast with xi_pm, two spin-2 fields, 1/[l(l+1)]^2).
    //   Only the C_l summed against differs (C_l^ks instead of C_l^gs).
    //
    //   Bin-averaging the kernel over [theta_min, theta_max]:
    //
    //     Pl(i,l) = 1/(xmin - xmax) * integral_{xmax}^{xmin} kernel_l(x) dx
    //
    //   with x = cos(theta), so xmin = cos(theta_min), xmax = cos(theta_max)
    //   (set_bin_average's convention: the min/max names track the theta
    //   edges, so xmin > xmax numerically).
    //
    // ANALYTIC BIN INTEGRATION:
    //   Same kernel and antiderivative as w_gammat_tomo (the derivation
    //   lives there: P_l^2 = (1-x^2) P_l'', reduced with Legendre's
    //   differential equation and integrated by parts):
    //
    //     integral P_l^2(x) dx = (l+2/(2l+1)) * P_{l-1}(x)
    //                          + (2-l) * x * P_l(x)
    //                          - 2/(2l+1) * P_{l+1}(x)
    //
    //   The bin-averaged kernel is the difference of these
    //   antiderivatives evaluated at the bin edges, divided by
    //   (xmin - xmax).
    //
    // NOTATION:
    //   Pmin[i][l] = P_l(xmin[i])       Pmax[i][l] = P_l(xmax[i])
    //   xmin[i] = cos(theta_min)        xmax[i] = cos(theta_max)
    //   (xmin - xmax) in denominator = bin width in cos(theta), positive
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
      C_ks_tomo_limber_nointerp_batch(lmin, limits.LMIN_tab, NSIZE, Cl);
      #pragma omp parallel for schedule(static)
      for (int nz=0; nz<NSIZE; nz++) {
        C_ks_tomo_limber_fill(nz, limits.LMIN_tab, Ntable.LMAX, lnell, Cl[nz]);
      }
      #pragma omp parallel for collapse(2) schedule(static)
      for (int nz=0; nz<NSIZE; nz++) {
        for (int l=lmin; l<Ntable.LMAX; l++) {
          Cl[nz][l] *= cmbf[l]; // multiply by CMB beam filter
        }
      }
    }
    else {
      log_fatal("NonLimber not implemented");
      exit(1);
    }
    legendre_sums(NSIZE, Ntable.Ntheta, lmin, Ntable.LMAX, Pl, Cl, w_vec);
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
  int npts;       // number of Gauss-Legendre quadrature points (= w->n;
                  // 2 w->n for a lens bin split by create_cosmo_nodes_lens)
  double** data;  // data[param][p]: cosmological quantities at each node
} cosmo_nodes;

// ---------------------------------------------------------------------------
// Column indices into cosmo_nodes.data[param][p].
// CN_NPARAMS is not a real parameter - it exploits enum auto-increment to
// give the total number of columns, used to size the malloc2d allocation.
// ---------------------------------------------------------------------------
enum {
  CN_A = 0,     // scale factor a (quadrature abscissa in [amin, amax])
  CN_WT,        // Gauss-Legendre quadrature weight
  CN_FK,        // comoving distance chi(a) (= f_K for flat cosmology)
  CN_GROWFAC,   // linear growth factor D(a) = growfac(a)
  CN_HOVERH0,   // H(a)/H0 computed via hoverh0v2(a, dchi/da)
  CN_DCHIDA,    // dchi/da from chi_all(a) - used in the Limber prefactor dchi/da / fK^2
  CN_NPARAMS    // total number of columns (auto-set by enum)
};

// ---------------------------------------------------------------------------
// Create cosmo_nodes by evaluating cosmological functions at all Gauss-Legendre
// quadrature points in the scale factor range [amin, amax].
//
// The quadrature points and weights come from the GSL fixed-order table w,
// which is shared with the Limber integration routines. The number of points
// (96-1024 for ss, 64-1024 for gs/gk/ks/kk, keyed on
// Ntable.high_def_integration) controls the
// accuracy of the numerical integration.
//
// Thread safety: the loop over nodes runs single-threaded, so it safely
// performs the first (lazy) initialization of the static interpolation
// tables inside chi_all, growfac and hoverh0v2 when needed.
//
// Parameters:
//   amin - minimum scale factor (integration lower bound)
//   amax - maximum scale factor (integration upper bound)
//   w    - GSL Gauss-Legendre table (provides nodes and weights)
//
// Returns:
//   a cosmo_nodes whose data[CN_NPARAMS][npts] is filled at every node;
//   the caller releases it with free_cosmo_nodes
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

// ---------------------------------------------------------------------------
// Release the data block of a cosmo_nodes created by create_cosmo_nodes.
// The struct itself is caller-owned storage; only data is heap allocated
// (one malloc2d block, so a single free releases it).
//
// Parameters:
//   cn - the cosmo_nodes whose data block to release
//
// Returns:
//   nothing
// ---------------------------------------------------------------------------
void free_cosmo_nodes(cosmo_nodes* cn) {
  free(cn->data);
}

// ---------------------------------------------------------------------------
// Near edge (largest scale factor) of the photo-z padded n(z) support of
// lens bin ni:
//
//   zmin = (zdist_zmin[ni] - zmean[ni]) * sigma_ni + zmean[ni]
//   a_nz = 1 / (1 + max(zmin - 2*|dz_ni|, 0.001))
//
// with sigma_ni = nuisance.photoz[1][1][ni] (stretch) and
// dz_ni = nuisance.photoz[1][0][ni] (shift): the stretched table edge,
// padded by one |dz_ni| for either sign of the shift and one more as
// margin. This is the bound amax_lens (redshift_spline.c) returns for a
// bin without magnification, written with the same operations in the
// same order, so the two are the same double: keep them in step. With
// magnification amax_lens moves past a_nz to the lower edge of the source
// sample, and a_nz stays the near edge of the padded lens n(z), the
// support of the density kernel W_gal.
//
// Parameters:
//   ni - lens tomographic bin index (0 .. clustering_nbin-1)
//
// Returns:
//   the largest scale factor of the bin's padded n(z) support
// ---------------------------------------------------------------------------
static double amax_lens_nz(const int ni)
{
  // the redshift floor of amax_lens: it keeps a < 1, where the kernels
  // are defined
  const double z_floor = 0.001;

  const double zmin =
    (redshift.clustering_zdist_z[RANGE_MIN][ni]
      - redshift.clustering_zdist_z[ZDIST_MEAN][ni])*nuisance.photoz[1][1][ni]
      + redshift.clustering_zdist_z[ZDIST_MEAN][ni];
  return 1. / (1 + fmax(zmin -2.*fabs(nuisance.photoz[1][0][ni]), z_floor));
}

// ---------------------------------------------------------------------------
// Gauss-Legendre nodes of lens bin ni on its Limber range [amin_lens(ni),
// amax_lens(ni)]: the nodes of the C_gs, C_gg and C_gk Limber integrals,
// and so of the Limber terms the non-Limber C_cl_tomo and C_gs_tomo
// subtract.
//
// Why the range is split. Without magnification the range is the bin's
// photo-z padded n(z) support, and one rule w covers it. With
// magnification (gbmag(0, ni) != 0) amax_lens widens the range down to the
// lower edge of the source sample, because the magnification kernel W_mag
// has support in front of the lens galaxies. One rule on the widened range
// spreads its nodes over the whole foreground and leaves fewer on the
// narrow density kernel W_gal: the density terms then depend on how far
// the range stretches, and b_mag = 0 against any b_mag != 0 changes the
// resolution of the density terms, not only the physics. The widened range
// is therefore split at a_nz = amax_lens_nz(ni), the near edge of the
// padded n(z) support, into two panels, each a full rule w on its own
// interval:
//
//   n(z) panel        [amin_lens, a_nz]: node for node the rule of the bin
//                     without magnification; every density term lives here
//   foreground panel  [a_nz, amax_lens]: in front of the padded n(z)
//                     support, where the magnification kernel carries the
//                     signal
//
// Both panels take the size of w, so each scales with
// Ntable.high_def_integration exactly as the single rule does, and the
// resolution of the density kernel no longer depends on the extent of the
// foreground.
//
// Nodes 0 .. w->n - 1 are the n(z) panel and the rest the foreground (the
// layout of the cluster bins in cosmo2D_cluster.c). A Limber sum runs over
// all cn.npts nodes of its bin, so the consumers need only the count,
// which is larger for a bin with the foreground panel.
//
// One rule over the whole range (create_cosmo_nodes) remains when
//   - the bin has no magnification: amax_lens = a_nz, nothing to split,
//     and the nodes are bitwise those of the unsplit design;
//   - the foreground is empty: the source sample starts behind a_nz, so
//     amax_lens is the nearer bound;
//   - COSMO2D_LENS_SINGLE_RULE is defined: every bin, the reference for
//     debugging the split.
//
// Thread safety: call it outside parallel regions; like create_cosmo_nodes
// it performs the lazy initialization of chi_all, growfac and hoverh0v2.
//
// Parameters:
//   ni - lens tomographic bin index (0 .. clustering_nbin-1)
//   w  - Gauss-Legendre rule of each panel
//
// Returns:
//   a cosmo_nodes the caller releases with free_cosmo_nodes
// ---------------------------------------------------------------------------
static cosmo_nodes create_cosmo_nodes_lens(
    const int ni,                            // lens tomographic bin
    const gsl_integration_glfixed_table* w   // Gauss-Legendre rule per panel
  )
{
  // --- 1. THE LIMBER RANGE AND THE NEAR EDGE OF THE N(Z) SUPPORT ---
  const double amin = amin_lens(ni);
  const double amax = amax_lens(ni);
  const double a_nz = amax_lens_nz(ni);

  const int has_magnification = (gbmag(0.0, ni) != 0);
  const int has_foreground    = (a_nz < amax);
  const int has_nz_panel      = (amin < a_nz);

  int split = 0;
  if (has_magnification && has_foreground && has_nz_panel) {
    split = 1;
  }
#ifdef COSMO2D_LENS_SINGLE_RULE
  split = 0; // the reference: one rule on every bin
#endif

  if (0 == split) {
    return create_cosmo_nodes(amin, amax, w);
  }

  // --- 2. TWO PANELS, ONE RULE EACH, CONCATENATED (N(Z) PANEL FIRST) ---
  const int nodes_per_panel = (int) w->n;

  const double panel_lower[2] = {amin, a_nz};  // n(z) panel, foreground
  const double panel_upper[2] = {a_nz, amax};

  cosmo_nodes cn;
  cn.npts = 2*nodes_per_panel;
  cn.data = (double**) malloc2d(CN_NPARAMS, cn.npts);

  for (int panel = 0; panel < 2; panel++) {
    for (int q = 0; q < nodes_per_panel; q++) {
      const int p = panel*nodes_per_panel + q;

      // node q of the rule, mapped onto the panel's interval
      gsl_integration_glfixed_point(panel_lower[panel],
                                    panel_upper[panel],
                                    q,
                                    &cn.data[CN_A][p],
                                    &cn.data[CN_WT][p],
                                    w);

      // the cosmology at the node, as in create_cosmo_nodes
      const double a      = cn.data[CN_A][p];
      struct chis chidchi = chi_all(a);
      cn.data[CN_FK][p]      = chidchi.chi;
      cn.data[CN_GROWFAC][p] = growfac(a);
      cn.data[CN_HOVERH0][p] = hoverh0v2(a, chidchi.dchida);
      cn.data[CN_DCHIDA][p]  = chidchi.dchida;
    }
  }
  return cn;
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

// ---------------------------------------------------------------------------
// TATT shear-shear EE integrand core.
// Pure arithmetic on preloaded scalars for vectorization.
//
// Extends NLA with tidal torquing (C2, bta) and one-loop IA kernels
// (tt, ta, ta_dE, mix). The formula expands the product
//   (WK1 - WS1*IA1) * (WK2 - WS2*IA2) * PK
// where IA_i includes linear (C1*PK), density-weighted (C1*bta*ta_dE),
// and quadratic (C2*mix, C2^2*tt) contributions.
//
// Where the factors 5 and 25 come from: Blazek et al. 2019
// (arXiv:1708.09247) define the tidal-torquing amplitude as
//   C2 = 5 * A2 * Cbar1 * rho_crit * Omega_m / D(z)^2
// while IA_A2_Z1 (IA.c) returns A2 * Omega_m * c1rhocrit_ia / D^2 with
// the 5 left out. The 5 is applied here instead, one per power of the
// quadratic field in each correlator: 5*C2 in the terms linear in C2,
// 25 = 5^2 in the C2*C2 (tt) term.
//
// Parameters:
//   PK     - P_delta(k, a): nonlinear matter power spectrum
//   WK1    - W_kappa(a, fK, n1): lensing convergence kernel, bin 1
//   WK2    - W_kappa(a, fK, n2): lensing convergence kernel, bin 2
//   WS1    - W_source(a, n1, h/h0): source distribution, bin 1
//   WS2    - W_source(a, n2, h/h0): source distribution, bin 2
//   C11    - IA_A1(a, D, n1): linear tidal alignment amplitude, bin 1
//   C12    - IA_A1(a, D, n2): linear tidal alignment amplitude, bin 2
//   C21    - IA_A2(a, D, n1): quadratic tidal alignment amplitude, bin 1
//   C22    - IA_A2(a, D, n2): quadratic tidal alignment amplitude, bin 2
//   bta1   - IA_BTA(a, D, n1): density weighting of tidal field, bin 1
//   bta2   - IA_BTA(a, D, n2): density weighting of tidal field, bin 2
//   tt     - g4 * P_tt(k): tidal-tidal one-loop kernel (FPTIA.tab[0])
//   ta_dE1 - g4 * P_ta_dE1(k): tidal-density E-mode kernel (FPTIA.tab[2])
//   ta_dE2 - g4 * P_ta_dE2(k): tidal-density E-mode kernel (FPTIA.tab[3])
//   ta     - g4 * P_ta(k): tidal-alignment one-loop kernel (FPTIA.tab[4])
//   mixA   - g4 * P_mixA(k): mixed A one-loop kernel (FPTIA.tab[6])
//   mixB   - g4 * P_mixB(k): mixed B one-loop kernel (FPTIA.tab[7])
//   mixEE  - g4 * P_mixEE(k): mixed EE one-loop kernel (FPTIA.tab[8])
//
// Returns:
//   the EE integrand value at one quadrature node (no dchida/fK^2
//   amplitude and no quadrature weight - the caller applies both)
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
// The 5/25 factors are the Blazek et al. 2019 C2 normalization, applied
// once per power of the quadratic field (see the EE core above).
//
// Parameters:
//   PK   - P_delta(k, a): NL MPS (unused but kept for API consistency)
//   WK1  - W_kappa(a, fK, n1): convergence kernel (unused, no tree level)
//   WK2  - W_kappa(a, fK, n2): convergence kernel (unused, no tree level)
//   WS1  - W_source(a, n1, h/h0): source distribution, bin 1
//   WS2  - W_source(a, n2, h/h0): source distribution, bin 2
//   C11  - IA_A1(a, D, n1): linear tidal alignment amplitude, bin 1
//   C12  - IA_A1(a, D, n2): linear tidal alignment amplitude, bin 2
//   C21  - IA_A2(a, D, n1): quadratic tidal alignment amplitude, bin 1
//   C22  - IA_A2(a, D, n2): quadratic tidal alignment amplitude, bin 2
//   bta1 - IA_BTA(a, D, n1): density weighting of tidal field, bin 1
//   bta2 - IA_BTA(a, D, n2): density weighting of tidal field, bin 2
//   tt   - g4 * P_tt_BB(k): tidal-tidal BB one-loop kernel (FPTIA.tab[1])
//   ta   - g4 * P_ta_BB(k): tidal-alignment BB kernel (FPTIA.tab[5])
//   mix  - g4 * P_mix_BB(k): mixed BB one-loop kernel (FPTIA.tab[9])
//
// Returns:
//   the BB integrand value at one quadrature node (no dchida/fK^2
//   amplitude and no quadrature weight - the caller applies both)
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
// Single-ell shear-shear C_l: a point diagnostic on the batch engine.
//
// Runs one C_ss_tomo_limber_nointerp_ells call at a single multipole and
// reads one entry, so it pays the WHOLE-TOMOGRAPHY batch cost per call
// (every enumerated Z1 <= Z2 pair is computed even though one number is
// returned). Never loop this over (l, ni, nj): call
// C_ss_tomo_limber_nointerp_ells once and index the result instead.
//
// Kept in the API as the exact per-multipole entry point a future
// non-Limber computation needs (the non-Limber pipelines evaluate the
// Limber part per integer multipole, the way C_gg_tomo consumes its
// scalar today).
//
// Parameters:
//   l    - multipole moment
//   ni   - first source redshift bin index
//   nj   - second source redshift bin index
//   EE   - 1 for E-mode power spectrum, 0 for B-mode
//
// Returns:
//   C_l^EE (EE = 1) or C_l^BB (EE = 0) of the (ni, nj) pair with the full
//   Limber model (nonlinear P_delta, the configured IA model)
// ---------------------------------------------------------------------------
double C_ss_tomo_limber_nointerp(
    const double l,
    const int ni,
    const int nj,
    const int EE
  ) // slow (whole-tomography batch per call) - use the batch version
{
  if (ni < 0 || ni > redshift.shear_nbin -1 ||
      nj < 0 || nj > redshift.shear_nbin -1) {
    log_fatal("invalid bin input (ni, nj) = (%d, %d)", ni, nj); exit(1);
  }
  const int NSIZE = tomo.shear_Npowerspectra;
  double** tmp_EE = (double**) malloc2d(NSIZE, 1);
  double** tmp_BB = (double**) malloc2d(NSIZE, 1);
  const double ell = l;

  C_ss_tomo_limber_nointerp_ells(&ell, 1, NSIZE, tmp_EE, tmp_BB);

  // N_shear is symmetric in (ni, nj), so no bin ordering is needed
  const int nz = N_shear(ni, nj);
  const double res = (1 == EE) ? tmp_EE[nz][0] : tmp_BB[nz][0];
  free(tmp_EE);
  free(tmp_BB);
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
// The key optimization is the loop nesting: precompute all quadrature
// points, resolve the IA model once, then run a SIMD loop per ell. A
// per-ell scalar quadrature would re-evaluate every kernel and branch on
// the IA model inside the innermost loop; here the IA model branch sits
// outside, and the inner loop is pure arithmetic on contiguous arrays -
// vectorizable with AVX2.
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
// Cache invalidation:
// none here - every input arrives precomputed; the
// warm-up prelude initializes the lazily-built statics of the kernel and
// power-spectrum functions single-threaded before the parallel regions.
//
// Parameters:
//   cn     - precomputed cosmological quantities at quadrature nodes
//            (scale factor, comoving distance, growth factor, dchi/da, weights)
//   lx     - array of multipole values, length nell
//            (log-spaced for C_ss_tomo_limber, integer-spaced for batch)
//   nell   - number of multipole values
//   NSIZE  - number of tomographic shear power spectra (= shear_nbin*(shear_nbin+1)/2)
//   table  - output array [2][NSIZE][nell]: table[0] = EE, table[1] = BB
//
// Returns:
//   nothing; the result is written into table
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
  // halo-model IA (include_halo_IA; Fortuna et al. 2021): the IA leg of
  // each source bin becomes f_rc(a) C1 P_delta f_2h + P_1h,dI and the
  // IA-IA term f_rc^2 C1 C1' P_delta f_2h + P_1h,II (halo.c readers)
  const int halo_ia = include_halo_IA;
  if (1 == halo_ia && nuisance.IA_MODEL == IA_MODEL_TATT) {
    log_fatal("include_halo_IA supports the NLA model only");
    exit(1);
  }
  {
    const double a    = cn->data[CN_A][0];
    const double fK   = cn->data[CN_FK][0];
    const double hoh0 = cn->data[CN_HOVERH0][0];
    const double gf   = cn->data[CN_GROWFAC][0];
    const double ell  = lx[0] + 0.5;
    if (1 == halo_ia) {
      // halo.c builds its IA tables in its own OpenMP regions: trigger
      // them here, before this function's parallel regions
      (void) ia_f_red_central(a);
      (void) ia_p1h_dI(ell/fK, a);
      (void) ia_p1h_II(ell/fK, a);
    }
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

  // halo IA at the nodes: KHI[0] = P_delta f_2h, KHI[1] = P_1h,dI,
  // KHI[2] = P_1h,II; FRC = f_rc(a)
  double*** KHI = NULL;
  double* FRC = NULL;
  if (1 == halo_ia) {
    KHI = (double***) malloc3d(3, nell, cn->npts);
    FRC = (double*) malloc1d(cn->npts);
  }

  double limTATT[3];
  if (nuisance.IA_MODEL == IA_MODEL_TATT) {
    if (0 == nuisance.IA_code) get_FPT_IA();
    limTATT[0] = log(FPTIA.krange[RANGE_MIN]);
    limTATT[1] = log(FPTIA.krange[RANGE_MAX]);
    limTATT[2] = (limTATT[1] - limTATT[0])/FPTIA.N;
  }
  // -----------------------------------------------------------------------
  // Precompute: radial weights per (bin, quadrature point) and
  //             P(k,a) + TATT kernels per (ell, quadrature point)
  // -----------------------------------------------------------------------
  // per-thread scratch of the batched P reads (Pdelta_at_a: one call per
  // node, the z half of the table read once per node instead of once per
  // multipole): KPN[2t] = the node's Limber wavenumbers, KPN[2t+1] = P
  double** KPN = (double**) malloc2d(2*omp_get_max_threads(), nell);
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
    double* restrict kn = KPN[2*omp_get_thread_num()];
    double* restrict pn = KPN[2*omp_get_thread_num() + 1];
    for (int i = 0; i<nell; i++) {
      kn[i] = (lx[i] + 0.5) / fK;
    }
    Pdelta_at_a(a, kn, nell, pn);
    for (int i = 0; i<nell; i++) {
      const double ell = lx[i] + 0.5;
      const double k = ell / fK;
      const double lnk = log(k);
      KIA[10][i][p] = pn[i];
      if (nuisance.IA_MODEL == IA_MODEL_TATT) {
        // Hold-last-node clamp (the idiom of every FPTIA/FPTbias LERP
        // read in this file): the table spacing is limTATT[2] =
        // range/FPTIA.N, so the gated k range's top lies up to one
        // spacing beyond the last node and b can reach FPTIA.N. When
        // b+1 would step past the table, the read clamps to
        // idx = N - 2 with dr = 0 - it holds a top-of-table node
        // instead of indexing out of bounds.
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
  // Precompute (halo-model IA only): its own loop nests, after P_delta is
  // in place, so the flag is tested once and the table reads thread over
  // every (node, ell) pair
  // -----------------------------------------------------------------------
  if (1 == halo_ia) {
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < cn->npts; p++) {
      FRC[p] = ia_f_red_central(cn->data[CN_A][p]);
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int p = 0; p < cn->npts; p++) {
      for (int i = 0; i < nell; i++) {
        const double a = cn->data[CN_A][p];
        const double k = (lx[i] + 0.5)/cn->data[CN_FK][p];

        KHI[0][i][p] = KIA[10][i][p]*ia_window_2h(k); // P_delta f_2h
        KHI[1][i][p] = ia_p1h_dI(k, a);               // satellites' dI
        KHI[2][i][p] = ia_p1h_II(k, a);               // satellites' II
      }
    }
  }

  // -----------------------------------------------------------------------
  // Main integration loop.
  // Always uses the TATT core function, which reduces identically to NLA
  // when C2 = BTA = 0 (as enforced by the memset initialization of KIA).
  // This avoids the IA model switch inside the loop, so the SIMD reduction
  // over quadrature points (p) sees only pure arithmetic - no branches.
  // The restrict pointers are hoisted before the p-loop to eliminate
  // gather instructions and enable contiguous AVX2 vector loads.
  //
  // Ell prefactor (1812.05995 eqs 74-79): two spin-2 shear fields, each
  // carrying the curved-sky factor sqrt((l-1)*l*(l+1)*(l+2))/(l+0.5)^2,
  // so
  //   ell_pf = [sqrt((l-1)*l*(l+1)*(l+2))/(l+0.5)^2]^2
  //          = l*(l-1)*(l+1)*(l+2)/(l+0.5)^4
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
      if (1 == halo_ia) {
        /* PHYSICAL DERIVATION & LOGIC FLOW (Fortuna et al. 2021)
           1. IA leg of bin j: f_rc C1_j P f_2h + P_1h,dI  (red centrals'
              NLA, windowed, plus the satellites' 1-halo term)
           2. EE = WK1 WK2 P - WS1 WK2 leg_1 - WS2 WK1 leg_2
                   + WS1 WS2 (f_rc^2 C1_1 C1_2 P f_2h + P_1h,II)
           3. BB = 0 (neither term sources B modes)                    */
        const double* restrict PKT  = KHI[0][i];
        const double* restrict P1DI = KHI[1][i];
        const double* restrict P1II = KHI[2][i];
        const double* restrict frc  = FRC;
        #pragma omp simd reduction(+:sEE)
        for (int p = 0; p < cn->npts; p++) {
          const double amp  = (dchida[p]/(fK[p]*fK[p]))*ell_pf;
          const double leg1 = frc[p]*C11[p]*PKT[p] + P1DI[p];
          const double leg2 = frc[p]*C12[p]*PKT[p] + P1DI[p];
          const double ii   = frc[p]*frc[p]*C11[p]*C12[p]*PKT[p] + P1II[p];
          const double ee   = WK1[p]*WK2[p]*PK[p]
                              - WS1[p]*WK2[p]*leg1
                              - WS2[p]*WK1[p]*leg2
                              + WS1[p]*WS2[p]*ii;
          sEE += ee*amp*wt[p];
        }
        table[0][k][i] = sEE;
        table[1][k][i] = 0.0;
        continue;
      }
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
  free(KPN);
  if (KHI != NULL) {
    free(KHI);
    free(FRC);
  }
}

// ---------------------------------------------------------------------------
// Batch computation of shear-shear C_l at arbitrary multipole values.
//
// Unlike C_ss_tomo_limber_nointerp_batch (which takes a contiguous integer
// range lmin..lmax-1 and writes into Cl[k][l] indexed by multipole), this
// version takes an arbitrary array of ell values and writes results into
// output arrays indexed 0..nell-1: entry i belongs to ells[i], with no
// multipole-to-index correspondence, so the ell values need not be
// integers, contiguous, or start anywhere in particular.
//
// Designed for the fourier-space likelihood (roman_fourier) which evaluates
// C_l at a sparse set of ell values (like.ell[]).
//
// Cache invalidation:
// the static Gauss-Legendre table w (96/128/256/512/
// 1024 nodes, keyed on abs(Ntable.high_def_integration)) rebuilds when
// Ntable.random changes; the cosmo_nodes are rebuilt on every call (they
// depend on the current cosmology).
//
// Parameters:
//   ells    - array of multipole values, length nell (need not be integers)
//   nell    - number of multipole values
//   NSIZE   - number of tomographic shear power spectra (= shear_Npowerspectra)
//   out_EE  - output array [NSIZE][nell], indexed as out_EE[nz][i]
//   out_BB  - output array [NSIZE][nell], indexed as out_BB[nz][i]
//
// Returns:
//   nothing; the results are written into out_EE and out_BB
// ---------------------------------------------------------------------------
void C_ss_tomo_limber_nointerp_ells(
    const double* ells,   // array of multipole values (length nell)
    const int nell,       // number of multipole values
    const int NSIZE,      // number of tomo shear power spectra
    double** out_EE,      // output EE [NSIZE][nell]
    double** out_BB      // output BB [NSIZE][nell]
  )
{
  static gsl_integration_glfixed_table* w = NULL;
  static uint64_t cache[MAX_SIZE_ARRAYS];
  if (NULL == w || fdiff2(cache[0], Ntable.random)) 
  {
    // Ntable.high_def_integration is the quadrature-accuracy knob the
    // interface writes from the yaml integration_accuracy key
    // (init_accuracy_boost). Its magnitude picks the fixed
    // Gauss-Legendre order in the ladder below; only abs() is ever
    // read - here and in every other ladder of this form.
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 96 :
                         (1 == hdi) ? 128 :
                         (2 == hdi) ? 256 : 
                         (3 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  const double amin = 1./(zmax_source_photoz() + 1.); // shifted support
  const double amax = 1./(1.+fmax(redshift.shear_zdist_zall[RANGE_MIN],1e-6));

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
// Batch shear-shear Limber C_l at the integer multipoles l = lmin..lmax-1,
// written at their own index: Cl[0][nz][l] (EE) and Cl[1][nz][l] (BB).
// Thin wrapper around C_ss_tomo_limber_nointerp_ells.
//
// Example: xi_pm_tomo calls it with lmin = 1 and lmax = limits.LMIN_tab
// for the multipoles below the interpolation table; Cl[0/1][nz][0] is
// left untouched.
//
// Parameters:
//   lmin  - first multipole (inclusive)
//   lmax  - last multipole (exclusive)
//   NSIZE - number of tomo shear power spectra (= shear_Npowerspectra)
//   Cl    - output [2][NSIZE][>= lmax], indexed Cl[0/1][nz][l] (EE/BB)
//
// Returns:
//   nothing; the result is written into Cl
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
// Batch computation of the scale-cut derivative dC_ss/dlnk on a
// (ln k, ell) grid (2011.06469 eq 17).
//
// In the Limber integral each scale factor maps one-to-one onto
// k = (l + 1/2)/chi(a), so dC_ss/dlnk at a given (k, ell) is the per-chi
// C_ss integrand core/fK^2 evaluated at the single node with
// chi(a) = (l + 1/2)/k, times |dchi/dlnk| = chi: the per-node amplitude
// is 1/fK. Equivalently, it is the quadrature's per-a amplitude
// dchida/fK^2 times |da/dlnk| = fK/dchida - the dchida cancels. There is
// no quadrature sum here - every (k, ell, tomo pair) output is one core
// evaluation.
//
// Same design as C_ss_tomo_limber_work: precompute every expensive
// quantity per node - the nodes are the nlnk*nell grid points, flattened
// as p = f*nell + i so each fixed f is one contiguous stretch - then fill
// every (tomo pair, node) output with the always-TATT cores, which reduce
// identically to NLA when the KIA kernels stay zero. A node whose scale
// factor falls outside the source support (a outside (amin, amax)) keeps
// AMP = 0 and zeroed kernels, so its outputs are exactly 0.
//
// With normalize = 0 the output is dC_ss/dlnk itself - what the real-space
// dlnxi machinery needs, since it Legendre-sums dC over ell before
// normalizing by xi(theta). With normalize = 1 the function also computes
// C_ss(ell, pair) - the same quadrature machinery as C_ss_tomo_limber_work
// -and writes dlnC_ss/dlnk = dC/C: one thread team computes the C_ss rows
// and then fills the dC rows, dividing each one right after filling it,
// while it is still cache-hot. No separate C_ss batch call, no
// intermediate dC table, no second pass over the output.
//
// Cache invalidation:
// none - no static state; every call recomputes
// from its arguments after its own single-threaded warm-up.
//
// Parameters:
//   lnkx      - ln k grid values (length nlnk), k in (Mpc/h)^-1
//   nlnk      - number of ln k grid values
//   lx        - multipole values (length nell)
//   nell      - number of multipole values
//   NSIZE     - number of tomo shear power spectra (= shear_Npowerspectra)
//   normalize - 1: write dlnC = dC/C_ss; 0: write dC
//   table     - output [2][NSIZE][nlnk][nell]: EE and BB
//
// Returns:
//   nothing; the result is written into table
// ---------------------------------------------------------------------------
void dC_ss_dlnk_tomo_limber_work(
    const double* lnkx,  // ln k grid values (length nlnk), k in (Mpc/h)^-1
    const int nlnk,      // number of ln k grid values
    const double* lx,    // multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of tomo shear power spectra
    const int normalize, // 1: write dlnC = dC/C_ss; 0: write dC
    double**** table     // output [2][NSIZE][nlnk][nell]: EE and BB
  )
{
  halo_IA_unsupported("dC_ss_dlnk_tomo_limber_work");
  const double amin = 1./(zmax_source_photoz() + 1.); // shifted support
  const double amax = 1./(1. + fmax(redshift.shear_zdist_zall[RANGE_MIN], 1e-6));

  // -----------------------------------------------------------------------
  // Warm up all functions that lazily initialize internal static tables.
  // Must be called single-threaded before any parallel region touches them.
  // -----------------------------------------------------------------------
  {
    const double a = 0.5*(amin + amax); // inside the source support
    struct chis chidchi = chi_all(a);
    const double fK   = chidchi.chi;
    const double hoh0 = hoverh0v2(a, chidchi.dchida);
    const double gf   = growfac(a);
    const double ell  = lx[0] + 0.5;
    (void) a_chi(fK);
    (void) f_K(fK);
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

  if (nlnk <= 0 || nell <= 0) {
    log_fatal("nlnk = %d and nell = %d must be positive", nlnk, nell);
    exit(1);
  }

  // -----------------------------------------------------------------------
  // Allocate precomputed arrays (one entry per node p = f*nell + i)
  // -----------------------------------------------------------------------
  const int npts = nlnk*nell;

  double* AMP = (double*) malloc1d(npts);
  double*** WC = (double***) malloc3d(5, redshift.shear_nbin, npts);
  zero3d(WC, 5, redshift.shear_nbin, npts);
  double** KIA = (double**) malloc2d(11, npts);
  zero2d(KIA, 11, npts);

  double limTATT[3];
  if (nuisance.IA_MODEL == IA_MODEL_TATT) {
    if (0 == nuisance.IA_code) get_FPT_IA();
    limTATT[0] = log(FPTIA.krange[RANGE_MIN]);
    limTATT[1] = log(FPTIA.krange[RANGE_MAX]);
    limTATT[2] = (limTATT[1] - limTATT[0])/FPTIA.N;
  }

  // -----------------------------------------------------------------------
  // Quadrature-side precompute (only when normalizing): C_ss needs its own
  // Gauss-Legendre node set along the line of sight, because the C_ell sum
  // runs over quadrature nodes, not (k, ell) grid nodes. Same machinery
  // and layouts as C_ss_tomo_limber_work: radial weights and IA amplitudes
  // per (source bin, node) in WCq, P_delta plus TATT kernels per
  // (ell, node) in KIAq (there k = (l + 1/2)/chi varies with ell at fixed
  // node, so KIAq keeps the ell dimension the grid-side KIA does not need)
  // -----------------------------------------------------------------------
  gsl_integration_glfixed_table* w = NULL;
  cosmo_nodes cn;
  double*** WCq = NULL;
  double*** KIAq = NULL;
  double** CEE = NULL;
  double** CBB = NULL;
  if (1 == normalize) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 96 :
                         (1 == hdi) ? 128 :
                         (2 == hdi) ? 256 :
                         (3 == hdi) ? 512 : 1024; // predefined GSL tables
    w = malloc_gslint_glfixed(szint);
    cn = create_cosmo_nodes(amin, amax, w);
    WCq = (double***) malloc3d(5, redshift.shear_nbin, cn.npts);
    KIAq = (double***) malloc3d(11, nell, cn.npts);
    zero3d(KIAq, 11, nell, cn.npts);
    CEE = (double**) malloc2d(NSIZE, nell);
    CBB = (double**) malloc2d(NSIZE, nell);
    #pragma omp parallel for schedule(static)
    for (int p = 0; p < cn.npts; p++) {
      const double a    = cn.data[CN_A][p];
      const double fK   = cn.data[CN_FK][p];
      const double hoh0 = cn.data[CN_HOVERH0][p];
      const double gf   = cn.data[CN_GROWFAC][p];
      const double g4   = gf*gf*gf*gf;
      for (int b = 0; b < redshift.shear_nbin; b++) {
        WCq[0][b][p] = W_kappa(a, fK, b);
        WCq[1][b][p] = W_source(a, b, hoh0);
        WCq[2][b][p] = IA_A1_Z1(a, gf, b);
        WCq[3][b][p] = IA_A2_Z1(a, gf, b);
        WCq[4][b][p] = IA_BTA_Z1(a, gf, b);
      }
      for (int i = 0; i < nell; i++) {
        const double ell = lx[i] + 0.5;
        const double k = ell / fK;
        const double lnk = log(k);
        KIAq[10][i][p] = Pdelta(k, a);
        if (nuisance.IA_MODEL == IA_MODEL_TATT) {
          if (lnk >= limTATT[0] && lnk <= limTATT[1]) {
            const double r = (lnk - limTATT[0]) / limTATT[2];
            const int b = (int) floor(r);
            const double dr = (b+1 >= FPTIA.N) ? 0.0 : r - b;
            const int idx = (b+1 >= FPTIA.N) ? FPTIA.N - 2 : b;
            for (int m = 0; m < 10; m++) {
              KIAq[m][i][p] = g4*LERP(FPTIA.tab[SS_IA_SRC[m]], idx, dr);
            }
          }
        }
      }
    }
  }

  // -----------------------------------------------------------------------
  // Precompute per node: the dlnk amplitude, radial weights and IA
  // amplitudes per source bin (WC, same layout as C_ss_tomo_limber_work),
  // and P_delta plus the TATT one-loop kernels (KIA; each node has a
  // single k, so KIA needs no separate ell dimension here)
  // -----------------------------------------------------------------------
  #pragma omp parallel for collapse(2) schedule(static)
  for (int f = 0; f < nlnk; f++) {
    for (int i = 0; i < nell; i++) {
      const int p = f*nell + i;
      const double l = lx[i];
      const double ell = l + 0.5;
      // the (k, ell) pair selects one Limber node: chi(a) = ell/k, with k
      // converted from (Mpc/h)^{-1} to ((Mpc/h)/(c/H0=100))^{-1}
      const double a = a_chi(f_K(ell/(exp(lnkx[f])*cosmology.coverH0)));
      if (!(a > amin && a < amax)) {
        AMP[p] = 0.0;
        continue;
      }
      struct chis chidchi = chi_all(a);
      const double growfac_a = growfac(a);
      const double hoverh0 = hoverh0v2(a, chidchi.dchida);
      const double fK = chidchi.chi;
      const double k = ell/fK;
      const double g4 = growfac_a*growfac_a*growfac_a*growfac_a;
      const double ell4 = ell*ell*ell*ell;
      // two spin-2 shear prefactors, as in C_ss_tomo_limber_work
      const double ell_prefactor = l*(l - 1.)*(l + 1.)*(l + 2.)/ell4;
      AMP[p] = ell_prefactor/fK;
      for (int b = 0; b < redshift.shear_nbin; b++) {
        WC[0][b][p] = W_kappa(a, fK, b);
        WC[1][b][p] = W_source(a, b, hoverh0);
        WC[2][b][p] = IA_A1_Z1(a, growfac_a, b);
        WC[3][b][p] = IA_A2_Z1(a, growfac_a, b);
        WC[4][b][p] = IA_BTA_Z1(a, growfac_a, b);
      }
      KIA[10][p] = Pdelta(k, a);
      if (nuisance.IA_MODEL == IA_MODEL_TATT) {
        const double lnk = log(k);
        if (lnk >= limTATT[0] && lnk <= limTATT[1]) {
          const double r = (lnk - limTATT[0]) / limTATT[2];
          const int b = (int) floor(r);
          const double dr = (b+1 >= FPTIA.N) ? 0.0 : r - b;
          const int idx = (b+1 >= FPTIA.N) ? FPTIA.N - 2 : b;
          for (int m = 0; m < 10; m++) {
            KIA[m][p] = g4*LERP(FPTIA.tab[SS_IA_SRC[m]], idx, dr);
          }
        }
      }
    }
  }

  // -----------------------------------------------------------------------
  // Main fill loop.
  //
  // Where the derivative differs from C_ss: in C_ss_tomo_limber_work each
  // output is a quadrature SUM over the line of sight,
  //
  //   C_ss(l) = sum_p core(p) * (dchida[p]/fK[p]^2) * ell_prefactor * wt[p],
  //
  // because every scale factor contributes to one C_ell. Here each output
  // is ONE core evaluation with no reduction,
  //
  //   dC_ss/dlnk(k, l) = core(p(k, l)) * (1/fK) * ell_prefactor,
  //
  // because at fixed ell the Limber relation k = (l + 1/2)/chi picks a
  // single node p(k, l), and changing variables from a to ln k multiplies
  // the per-a integrand core * (dchida/fK^2) by |da/dlnk| = fK/dchida:
  // the dchida cancels and one power of 1/fK survives (2011.06469 eq 17).
  // AMP carries that per-node amplitude, with AMP = 0 marking nodes
  // outside the source support. The core functions and their inputs
  // (WC, KIA) are exactly the ones the C_ell sum uses: only the amplitude
  // and the absence of the sum differ.
  //
  // When normalizing, one thread team does everything: its first loop
  // computes the C_ss rows (the quadrature sum below, one row per tomo
  // pair - C_ss does not depend on k, so each row serves every f), and
  // after the loop's implicit barrier the same team fills the dC rows and
  // divides each one to dlnC = dC/C while it is still cache-hot. No
  // intermediate dC table exists and no pass re-reads the output.
  //
  // Always uses the TATT core function, which reduces identically to NLA
  // when C2 = BTA = 0 (as enforced by the zero initialization of KIA).
  // This avoids the IA model switch inside the loop, so the SIMD body over
  // the nell contiguous nodes of each f sees only pure arithmetic.
  // The restrict pointers are hoisted before the inner loops to eliminate
  // gather instructions and enable contiguous vector loads.
  // -----------------------------------------------------------------------
  #pragma omp parallel
  {
  if (1 == normalize) { // C_ss rows first: the division below reads them
    #pragma omp for collapse(2) schedule(static)
    for (int nz = 0; nz < NSIZE; nz++) {
      for (int i = 0; i < nell; i++) {
        const int Z1NZ = Z1(nz);
        const int Z2NZ = Z2(nz);
        const double* restrict fKq    = cn.data[CN_FK];
        const double* restrict dchida = cn.data[CN_DCHIDA];
        const double* restrict wt     = cn.data[CN_WT];
        const double* restrict PK     = KIAq[10][i];
        const double* restrict tt     = KIAq[0][i];
        const double* restrict ta_dE1 = KIAq[1][i];
        const double* restrict ta_dE2 = KIAq[2][i];
        const double* restrict ta     = KIAq[3][i];
        const double* restrict mixA   = KIAq[4][i];
        const double* restrict mixB   = KIAq[5][i];
        const double* restrict mixEE  = KIAq[6][i];
        const double* restrict ttbb   = KIAq[7][i];
        const double* restrict tabb   = KIAq[8][i];
        const double* restrict mixbb  = KIAq[9][i];
        const double* restrict WK1    = WCq[0][Z1NZ];
        const double* restrict WK2    = WCq[0][Z2NZ];
        const double* restrict WS1    = WCq[1][Z1NZ];
        const double* restrict WS2    = WCq[1][Z2NZ];
        const double* restrict C11    = WCq[2][Z1NZ];
        const double* restrict C12    = WCq[2][Z2NZ];
        const double* restrict C21    = WCq[3][Z1NZ];
        const double* restrict C22    = WCq[3][Z2NZ];
        const double* restrict bta1   = WCq[4][Z1NZ];
        const double* restrict bta2   = WCq[4][Z2NZ];
        const double l = lx[i];
        const double ell = l + 0.5;
        const double ell4 = ell*ell*ell*ell;
        const double ell_pf = l*(l - 1.)*(l + 1.)*(l + 2.)/ell4;
        double sEE = 0.0;
        double sBB = 0.0;
        #pragma omp simd reduction(+:sEE, sBB)
        for (int p = 0; p < cn.npts; p++) {
          const double ampq = (dchida[p]/(fKq[p]*fKq[p]))*ell_pf;
          sEE += int_for_C_ss_tomo_limber_tatt_EE_core(
                   PK[p],WK1[p],WK2[p],WS1[p],WS2[p],
                   C11[p],C12[p],C21[p],C22[p],bta1[p],bta2[p],
                   tt[p],ta_dE1[p],ta_dE2[p],ta[p],
                   mixA[p],mixB[p],mixEE[p]) * ampq * wt[p];
          sBB += int_for_C_ss_tomo_limber_tatt_BB_core(
                   PK[p],WK1[p],WK2[p],WS1[p],WS2[p],
                   C11[p],C12[p],C21[p],C22[p],bta1[p],bta2[p],
                   ttbb[p],tabb[p],mixbb[p]) * ampq * wt[p];
        }
        CEE[nz][i] = sEE;
        CBB[nz][i] = sBB;
      }
    } // implicit barrier: C_ss rows complete before any division below
  }
  #pragma omp for collapse(2) schedule(static)
  for (int nz = 0; nz < NSIZE; nz++) {
    for (int f = 0; f < nlnk; f++) {
      const int Z1NZ = Z1(nz);
      const int Z2NZ = Z2(nz);
      const double* restrict amp    = &AMP[f*nell];
      const double* restrict PK     = &KIA[10][f*nell];
      const double* restrict tt     = &KIA[0][f*nell];
      const double* restrict ta_dE1 = &KIA[1][f*nell];
      const double* restrict ta_dE2 = &KIA[2][f*nell];
      const double* restrict ta     = &KIA[3][f*nell];
      const double* restrict mixA   = &KIA[4][f*nell];
      const double* restrict mixB   = &KIA[5][f*nell];
      const double* restrict mixEE  = &KIA[6][f*nell];
      const double* restrict ttbb   = &KIA[7][f*nell];
      const double* restrict tabb   = &KIA[8][f*nell];
      const double* restrict mixbb  = &KIA[9][f*nell];
      const double* restrict WK1    = &WC[0][Z1NZ][f*nell];
      const double* restrict WK2    = &WC[0][Z2NZ][f*nell];
      const double* restrict WS1    = &WC[1][Z1NZ][f*nell];
      const double* restrict WS2    = &WC[1][Z2NZ][f*nell];
      const double* restrict C11    = &WC[2][Z1NZ][f*nell];
      const double* restrict C12    = &WC[2][Z2NZ][f*nell];
      const double* restrict C21    = &WC[3][Z1NZ][f*nell];
      const double* restrict C22    = &WC[3][Z2NZ][f*nell];
      const double* restrict bta1   = &WC[4][Z1NZ][f*nell];
      const double* restrict bta2   = &WC[4][Z2NZ][f*nell];
      double* restrict outEE = table[0][nz][f];
      double* restrict outBB = table[1][nz][f];
      #pragma omp simd
      for (int i = 0; i < nell; i++) {
        outEE[i] = int_for_C_ss_tomo_limber_tatt_EE_core(
                     PK[i],WK1[i],WK2[i],WS1[i],WS2[i],
                     C11[i],C12[i],C21[i],C22[i],bta1[i],bta2[i],
                     tt[i],ta_dE1[i],ta_dE2[i],ta[i],
                     mixA[i],mixB[i],mixEE[i]) * amp[i];
        outBB[i] = int_for_C_ss_tomo_limber_tatt_BB_core(
                     PK[i],WK1[i],WK2[i],WS1[i],WS2[i],
                     C11[i],C12[i],C21[i],C22[i],bta1[i],bta2[i],
                     ttbb[i],tabb[i],mixbb[i]) * amp[i];
      }
      if (1 == normalize) { // dlnC = dC/C, dividing while the row is
        // cache-hot; a near-zero dC passes through and a near-zero C gives
        // 0, so the ratio never blows up where the spectra vanish
        const double* restrict cee = CEE[nz];
        const double* restrict cbb = CBB[nz];
        #pragma omp simd
        for (int i = 0; i < nell; i++) {
          const double dCEE = outEE[i];
          const double CEEv = (fabs(dCEE) > 1e-30) ? cee[i] : 1.0;
          outEE[i] = (fabs(CEEv) > 1e-30) ? dCEE/CEEv : 0.0;
          const double dCBB = outBB[i];
          const double CBBv = (fabs(dCBB) > 1e-30) ? cbb[i] : 1.0;
          outBB[i] = (fabs(CBBv) > 1e-30) ? dCBB/CBBv : 0.0;
        }
      }
    }
  }
  } // end of the parallel region
  free(AMP);
  free(WC);
  free(KIA);
  if (1 == normalize) {
    free(CEE);
    free(CBB);
    free(WCq);
    free(KIAq);
    free_cosmo_nodes(&cn);
    gsl_integration_glfixed_table_free(w);
  }
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
// When Ntable.N_ell[NODES_COARSE] is active, the exact quadrature instead
// runs on the internal coarse grid and the house cubic spline
// upsamples onto the unchanged N_ell nodes (the strategy block inside
// explains why this wins).
//
// Why the table is shared through the ss_ static struct: the
// real-space projection (xi_pm_tomo) needs C_l at every integer
// multipole up to Ntable.LMAX ~ 1e5, for every tomographic pair,
// inside its Legendre/Hankel sums - millions of table reads per
// likelihood evaluation. Only the vectorized batch reader
// (C_ss_tomo_limber_fill, which runs the interpol1d linear read four
// multipoles at a time through AVX2 gathers) sustains that rate;
// calling this function one multipole at a time would dominate the
// whole evaluation.
//
// The struct is how the table travels between the two functions.
// The builder (this function) and the reader (the _fill) never call
// each other - the real-space projection calls one, the C_ell paths
// call the other - so no argument list connects them. Instead the
// builder publishes the table pointer and the grid geometry (the
// ln(ell) limits, spacing and node count) in the file-scope struct,
// and the reader picks them up there.
//
// Cache invalidation:
// recomputes when any of these change:
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
  static int ncoarse = 0;  // active internal coarse grid size (0 = off)
  static double dlnc = 0.; // coarse grid spacing in ln(ell)
  static double* lxc = NULL;    // coarse ell nodes
  static int* qidx = NULL;      // fine node -> coarse interval (uniform
  static double* qdel = NULL;   //   grids: precomputed, no search)
  static double*** tabc = NULL; // coarse C_ell values
  static double*** cspl = NULL; // natural-cubic-spline c coefficients

  if (NULL == table || fdiff2(cache[4], Ntable.random))
  {
    nell = Ntable.N_ell[NODES_DENSE];
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

    // Coarse-grid workspace (the strategy is explained where the grid
    // is used, in the refill block below): every allocation lives
    // HERE, in the Ntable rebuild block; the per-cosmology refill only
    // fills. The pieces are:
    //   lxc        - the ncoarse ell nodes, log-spaced over the same
    //                [lim[0], lim[1]] range as the fine table
    //   tabc, cspl - the coarse C_ell values and their cubic-spline
    //                coefficients, one row per (EE/BB, bin pair)
    //   qidx, qdel - for each fine node, the coarse interval it falls
    //                in and its ln(ell) offset from that interval's
    //                left node: both grids are uniform in ln(ell) with
    //                shared endpoints, so this is pure grid geometry,
    //                computed once - no search of any kind at refill
    if (lxc  != NULL) { free(lxc);  lxc  = NULL; }
    if (qidx != NULL) { free(qidx); qidx = NULL; }
    if (qdel != NULL) { free(qdel); qdel = NULL; }
    if (tabc != NULL) { free(tabc); tabc = NULL; }
    if (cspl != NULL) { free(cspl); cspl = NULL; }
    const int nc = Ntable.N_ell[NODES_COARSE];
    ncoarse = (nc > 3 && nc < nell) ? nc : 0;
    if (ncoarse > 0) {
      dlnc = (lim[1] - lim[0]) / ((double) ncoarse - 1.0);
      lxc = (double*) malloc1d(ncoarse);
      for (int i=0; i<ncoarse; i++) {
        lxc[i] = exp(lim[0] + i*dlnc);
      }
      qidx = (int*) malloc(sizeof(int) * nell);
      qdel = (double*) malloc1d(nell);
      for (int i=0; i<nell; i++) {
        // Where does fine node i sit on the coarse grid? Both grids
        // run over the same [lim[0], lim[1]] in ln(ell), so the map
        // is pure arithmetic:
        //
        //   fine node i -> ln(ell) = lim[0] + i*lim[2]
        //               -> r = i*lim[2]/dlnc   (coarse spacings in)
        //               -> j = (int) r         (interval's left node)
        //               -> qdel = (r - j)*dlnc (offset inside it)
        //
        // The spline evaluates on interval [j, j+1], so the largest
        // legal j is ncoarse-2, the left node of the LAST interval.
        //
        // Why the clamp: at the shared top endpoint, i*lim[2] and
        // (ncoarse-1)*dlnc are two floating-point roundings of the
        // same length lim[1] - lim[0]. r can therefore land one ulp
        // above ncoarse-1 and truncate to j = ncoarse-1 - one past
        // the last interval. The clamp moves that node back onto the
        // last interval, where it evaluates at (at most one ulp
        // past) the interval's right endpoint.
        const double r = (double) i * lim[2] / dlnc;
        int j = (int) r;
        if (j > ncoarse - 2) {
          j = ncoarse - 2;
        }
        qidx[i] = j;
        qdel[i] = (r - j) * dlnc; // offset from node j, in ln(ell)
      }
      tabc = (double***) malloc3d(2, tomo.shear_Npowerspectra, ncoarse);
      cspl = (double***) malloc3d(2, tomo.shear_Npowerspectra, ncoarse);
    }
  }

  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random) ||
      fdiff2(cache[5], (uint64_t) include_halo_IA) ||
      fdiff2(cache[6], nuisance.random_ia_halo))
  {
    const double amin = 1./(zmax_source_photoz() + 1.); // shifted support
    const double amax = 1./(1.+fmax(redshift.shear_zdist_zall[RANGE_MIN],1e-6));
    
    cosmo_nodes cn = create_cosmo_nodes(amin, amax, w);

    if (ncoarse > 0) {
      // ---------------------------------------------------------------
      // The internal coarse grid: general strategy.
      //
      // The real-space projections (xi_pm_tomo, via the shared ss_
      // struct and C_ss_tomo_limber_fill) read this table at every
      // integer ell up to Ntable.LMAX ~ 1e5 inside their Legendre
      // sums. At that call rate only the optimized, vectorized LINEAR
      // read is affordable: a cubic-spline lookup per ell would
      // dominate the whole evaluation.
      //
      // A linear read, however, is only accurate on a DENSE table -
      // and each of the N_ell = 512 nodes costs one exact Limber
      // quadrature, which is the expensive part.
      //
      // The coarse grid splits the difference: a cubic spline carries
      // far more accuracy per node than a linear segment, so the
      // expensive quadratures run on few nodes and a cheap cubic
      // upsampling fills the dense table:
      //
      //   exact Limber quadrature on ncoarse nodes (default 192)
      //     -> spline_coeffs_uniform: one tridiagonal solve per row
      //     -> Horner evaluation at the 512 precomputed fine offsets
      //     -> the unchanged dense table
      //     -> the same fast linear reads by every consumer
      //
      // This is safe because
      // C_ss is smooth in ln(ell); C_gg keeps the exact grid - its
      // BAO wiggles would be undersampled (see its header).
      // ---------------------------------------------------------------
      zero3d(tabc, 2, tomo.shear_Npowerspectra, ncoarse);

      C_ss_tomo_limber_work(&cn, lxc, ncoarse, tomo.shear_Npowerspectra,
                            tabc);

      const double hc = dlnc;
      const double inv_hc = 1.0/dlnc;
      #pragma omp parallel for collapse(2) schedule(static)
      for (int c=0; c<2; c++) {
        for (int nz=0; nz<tomo.shear_Npowerspectra; nz++) {
          spline_coeffs_uniform(tabc[c][nz], ncoarse, hc, cspl[c][nz]);
        }
      }
      // Upsampling. On interval [x_j, x_j + h] the house spline
      // (spline_coeffs_uniform) is the cubic
      //
      //   S(x_j + dx) = y_j + b dx + c_j dx^2 + d dx^3
      //
      // where c is the coefficient array the tridiagonal solve above
      // produced: the spline's second derivative / 2, with natural
      // boundaries c_0 = c_{n-1} = 0.
      //
      // The other two coefficients follow from two conditions:
      //
      //   S'' runs linearly from 2 c_j to 2 c_{j+1}
      //     -> d = (c_{j+1} - c_j) / (3 h)
      //
      //   S(x_{j+1}) = y_{j+1}, interpolate the right node
      //     -> b = (y_{j+1} - y_j)/h - h (c_{j+1} + 2 c_j)/3
      //
      // The polynomial is evaluated in Horner form; qidx/qdel hold
      // each fine node's precomputed interval j and offset dx.
      #pragma omp parallel for collapse(3) schedule(static)
      for (int c=0; c<2; c++) {
        for (int nz=0; nz<tomo.shear_Npowerspectra; nz++) {
          for (int i=0; i<nell; i++) {
            const double* restrict y = tabc[c][nz];
            const double* restrict cc = cspl[c][nz];
            const int j = qidx[i];
            const double b = (y[j+1] - y[j])*inv_hc
                             - hc*(cc[j+1] + 2.0*cc[j])/3.0;
            const double d = (cc[j+1] - cc[j])/(3.0*hc);
            table[c][nz][i] =
                y[j] + qdel[i]*(b + qdel[i]*(cc[j] + qdel[i]*d));
          }
        }
      }
    }
    else {
      C_ss_tomo_limber_work(&cn, lx, nell, tomo.shear_Npowerspectra, table);
    }

    free_cosmo_nodes(&cn);

    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_shear;
    cache[2] = nuisance.random_ia;
    cache[3] = redshift.random_shear;
    cache[4] = Ntable.random;
    cache[5] = (uint64_t) include_halo_IA;
    cache[6] = nuisance.random_ia_halo;
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
// C_l -> xi_pm(theta). Reading the ss_ table one ell at a time via interpol1d
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
//
// Returns:
//   nothing; the interpolated C_l are written into out_EE and out_BB at
//   indices lmin..lmax-1
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
// GS = GALAXY SHEAR (GGL)
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
// The 1/2: the galaxy density is expanded as
//   delta_g = b1*d + (b2/2)*d^2 + (bs2/2)*s^2 + (b3/2)*psi3 + ...
// so in the galaxy-matter cross each second-order operator carries its
// 1/2 into the correlator (the gg auto squares the expansion instead:
// 1/4 on operator autos, 1/2 on the b2-bs2 cross, 1 on operator-matter
// crosses - see C_gg_tomo_limber_work).
//
// The first three terms are the standard one-loop SPT contributions from
// second-order (b2), tidal (bs2), and third-order (b3) galaxy bias operators
// convolved with the corresponding matter field correlators (d1d2, d1s2, d1p3).
// The last term (bk*k^2*PK) is the higher-derivative counterterm that absorbs
// sensitivity to small-scale modes beyond the perturbative regime.
//
// This function returns the one-loop piece only - it does NOT include the
// tree-level b1*PK contribution, which is handled separately in the calling
// NLA/TATT core functions to enforce one-loop consistency (oneloop x linear IA
// only, avoiding two-loop cross terms).
//
// Parameters:
//   k    - wavenumber k = (l+0.5) / fK
//   PK   - P_delta(k, a): nonlinear matter power spectrum
//   g4   - D(a)^4: fourth power of the linear growth factor
//   b2   - gb2(z, nl): second-order galaxy bias
//   bs2  - gbs2(z, nl): tidal (s^2) galaxy bias
//   b3   - gb3(z, nl): third-order galaxy bias
//   bk   - gbK(z, nl): higher-derivative bias coefficient
//   d1d2 - P_{delta,delta2}(k): one-loop matter-b2 correlator
//   d1s2 - P_{delta,s2}(k): one-loop matter-tidal correlator
//   d1p3 - P_{delta,psi3}(k): one-loop matter-b3 correlator
//
// Returns:
//   the one-loop galaxy bias correction at one quadrature node
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
// TATT galaxy-shear (galaxy-galaxy lensing) Limber integrand core.
// Pure arithmetic on preloaded scalars for SIMD vectorization.
//
// Extends the NLA core with tidal torquing (C2, BTA) and one-loop IA kernels.
// The intrinsic alignment field is:
//   IA = C1*PK + IATATT
// where IATATT = C1*BTA*(ta_dE1 + ta_dE2) - 5*C2*(mixA + mixB) collects
// the density-weighted tidal alignment and quadratic tidal torquing terms.
// The 5 is the Blazek et al. 2019 C2 normalization that IA_A2_Z1 leaves
// out, one factor per power of the quadratic field (see the ss EE core);
// gs correlators are linear in C2, so no 25 appears here.
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
// Reduces to the NLA product (WGAL*b1 + WMAG*ep*bmag + WRSD) *
// (WK - WS*C1) * PK when C2 = 0, BTA = 0,
// and all TATT kernels are zero (as enforced by memset for NLA).
//
// Parameters:
//   PK      - P_delta(k, a): nonlinear matter power spectrum
//   WK      - W_kappa(a, fK, ns): lensing convergence kernel
//   WS      - W_source(a, ns, h/h0): source galaxy distribution
//   WGAL    - W_gal(a, nl, h/h0): lens galaxy distribution
//   WMAG    - W_mag(a, fK, nl): magnification lensing kernel
//   WRSD    - W_RSD(ell, a0, a1, nl): RSD kernel (0 if disabled)
//   C1      - IA_A1(a, D, ns): linear tidal alignment amplitude
//   C2      - IA_A2(a, D, ns): quadratic tidal alignment amplitude
//   BTA     - IA_BTA(a, D, ns): density weighting of tidal field
//   ta_dE1  - g4 * P_ta_dE1(k): tidal-density kernel (FPTIA.tab[2])
//   ta_dE2  - g4 * P_ta_dE2(k): tidal-density kernel (FPTIA.tab[3])
//   mixA    - g4 * P_mixA(k): mixed A kernel (FPTIA.tab[6])
//   mixB    - g4 * P_mixB(k): mixed B kernel (FPTIA.tab[7])
//   b1      - gb1(z, nl): linear galaxy bias
//   bmag    - gbmag(z, nl): magnification bias coefficient
//   oneloop - one-loop galaxy bias correction (from bias_oneloop_core)
//   bmag_ell_prefactor - l*(l+1)/(l+0.5)^2: magnification ell prefactor
//
// Returns:
//   the integrand value at one quadrature node (no dchida/fK^2 amplitude
//   and no quadrature weight - the caller applies both)
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
// Single-ell galaxy-galaxy lensing C_l: a point diagnostic on the batch
// engine.
//
// Runs one C_gs_tomo_limber_nointerp_ells call at a single multipole and
// reads one entry, so it pays the WHOLE-TOMOGRAPHY batch cost per call
// (every enumerated lens-source pair is computed even though one number is
// returned). Never loop this over (l, nl, ns): call
// C_gs_tomo_limber_nointerp_ells once and index the result instead.
//
// Kept in the API as the exact per-multipole entry point of the probe (the
// C_ss_tomo_limber_nointerp pattern): notebooks evaluate single points
// here, and a per-integer-multipole Limber value is what a non-Limber
// pipeline consumes (C_gs_tomo backfills from the batch, not from this
// wrapper).
//
// Parameters:
//   l    - multipole moment
//   nl   - lens redshift bin index (0..redshift.clustering_nbin-1)
//   ns   - source redshift bin index (0..redshift.shear_nbin-1); (nl, ns)
//          must be an enumerated ggl pair (test_zoverlap true), the pairs
//          the data vector carries
//
// Returns:
//   C_l^gs of the (nl, ns) pair with the full Limber model (nonlinear
//   P_delta, one-loop bias when enabled, the configured IA model)
// ---------------------------------------------------------------------------
double C_gs_tomo_limber_nointerp(
    const double l,
    const int nl,
    const int ns
  ) // slow (whole-tomography batch per call) - use the batch version
{
  if (nl < 0 || nl > redshift.clustering_nbin - 1 ||
      ns < 0 || ns > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input (nl, ns) = (%d, %d)", nl, ns); exit(1);
  }
  const int nz = N_ggl(nl, ns);
  if (nz < 0) {
    log_fatal("(nl, ns) = (%d, %d) is not an enumerated ggl pair", nl, ns);
    exit(1);
  }
  const int NSIZE = tomo.ggl_Npowerspectra;
  double** tmp = (double**) malloc2d(NSIZE, 1);
  const double ell = l;

  C_gs_tomo_limber_nointerp_ells(&ell, 1, NSIZE, tmp);

  const double res = tmp[nz][0];
  free(tmp);
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
// The node count may differ between lens bins as well (cn_all[zl].npts):
// every per-node array below is padded to the largest count, npts_max,
// the precompute loops skip a bin's padding nodes, and each pair's sum
// runs over the nodes of its own lens bin only.
//
// Memory layout (npts = npts_max, the largest node count over the bins):
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
//     KIA[9]   = the separable P_lin of the linear term (table_lin)
//
// One-loop consistency: the inner loop calls _tatt_core which ensures
//   - tree-level galaxy (b1*PK) multiplies the full IA (C1*PK + TATT terms)
//   - one-loop galaxy (b1l from _bias_oneloop_core) multiplies only linear IA
//   (the TATT kernels and the one-loop bias are each already one-loop
//   order, so their cross terms would be two-loop and are dropped)
//
// Cache invalidation:
// none here - every input arrives precomputed; the
// warm-up prelude initializes the lazily-built statics of the kernel and
// power-spectrum functions single-threaded before the parallel regions.
//
// Parameters:
//   cn_all - array of cosmo_nodes, one per lens bin [clustering_nbin]
//   lx     - array of multipole values, length nell
//   ell_prefactor  - l*(l+1)/(l+0.5)^2 per ell (magnification)
//   ell_prefactor2 - sqrt(l*(l-1)*(l+1)*(l+2))/(l+0.5)^2 per ell (shear)
//   nell   - number of multipole values
//   table  - output array [ggl_Npowerspectra][nell], the full model
//            (Limber path; NULL: not computed)
//   table_lin - output array [ggl_Npowerspectra][nell], the linear
//            counterpart of the FFTLog term of C_gs_tomo ((D(a)/D(a_piv))^2
//            P_lin(k, a_piv) per lens bin, no one-loop bias, IA through C1
//            only; NULL: not computed)
//
// Why one call can fill both: the non-Limber C_gs_tomo needs both terms
// at the same multipoles. The nodes, the lens and source weights, the IA
// amplitudes and the RSD kernel are computed once; the sum runs once per
// output, the linear one reading the one-loop bias and TATT inputs at
// zero (their values in a linear-only call), so filling both at once is
// bitwise two separate calls.
//
// Returns:
//   nothing; the results are written into table and table_lin
// ---------------------------------------------------------------------------
static void C_gs_tomo_limber_work(
    const cosmo_nodes* cn_all,  // quadrature nodes per lens bin [clustering_nbin]
    const double* lx,           // multipole values (length nell)
    const double* ell_prefactor,  // l*(l+1)/(l+0.5)^2 per ell (magnification)
    const double* ell_prefactor2, // sqrt(l*(l-1)*(l+1)*(l+2))/(l+0.5)^2 per ell
    const int nell,             // number of multipole values
    double** table,             // output [ggl_Npowerspectra][nell], full model (or NULL)
    double** table_lin          // output [ggl_Npowerspectra][nell], linear term (or NULL)
  )
{
  // -----------------------------------------------------------------------
  // Warm up all functions that lazily initialize internal static tables.
  // Must be called single-threaded before any parallel region touches them.
  // -----------------------------------------------------------------------
  // The linear term (table_lin) reads the one-loop bias and the TATT
  // kernels at zero (ZERO below), so the cores reduce to b1*P_lin times
  // (WK - WS*C1): exactly the physics of the FFTLog term.
  // -------------------------------------------------------------------------
  // HOD mode (include_HOD_GX = 1): the lens galaxies are the halo-model
  // occupation field. The density leg's power with matter is p_gm, and
  // its cross with the NLA alignment field is C1 p_gm (delta_I = C1
  // times the linear tidal field, so <delta_g delta_I> = C1 <delta_g
  // delta_m>); magnification traces matter and keeps the standard
  // term. Per quadrature node:
  //
  //   [ W_gal p_gm(k, a, zl) + W_mag ep b_mag P_delta ] (W_K - W_S C1)
  //
  // TATT crosses the density with one-loop tidal operators, which has
  // no HOD counterpart: TATT + HOD aborts (NLA only). RSD and the
  // one-loop bias are off in this mode, and HOD C_l^gs is Limber-only
  // (the FFTLog linear term aborts): run with adopt_limber_gs = 1.
  // -------------------------------------------------------------------------
  const int hod = include_HOD_GX;
  // halo-model IA (include_halo_IA): the source IA leg becomes f_rc C1
  // P_delta f_2h + P_1h,dI (see C_ss_tomo_limber_work); Limber, NLA and
  // perturbative-bias galaxies only in this version
  const int halo_ia = include_halo_IA;
  if (1 == halo_ia && (1 == hod || NULL != table_lin)) {
    log_fatal("include_halo_IA: needs include_HOD_GX = 0 and the "
              "Limber gs (adopt_limber_gs = 1)");
    exit(1);
  }
  if (1 == hod && NULL != table_lin) {
    log_fatal("HOD C_l^gs is Limber-only: set adopt_limber_gs = 1");
    exit(1);
  }

  int nonlinear_bias = 0;
  if (NULL != table && 0 == hod) {
    nonlinear_bias = has_b2_galaxies();
  }

  // RSD is not part of the HOD model (see the note above)
  int rsd = 0;
  if (1 == include_RSD_GS && 0 == hod) {
    rsd = 1;
  }
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
    if (NULL != table_lin) {
      (void) p_lin(ell/fK, a);
    }
    (void) gb1(0.1, 0);
    (void) gbmag(0.1, 0);
    (void) ZL(0);
    (void) ZS(0);
    if (1 == hod) {
      // halo.c builds its (a, ln k) tables inside its own OpenMP
      // regions; trigger them before this function's parallel regions
      (void) p_gm(ell/fK, a, 0);
    }
    if (1 == halo_ia) {
      (void) ia_f_red_central(a);
      (void) ia_p1h_dI(ell/fK, a);
    }
    if (1 == nonlinear_bias) {
      (void) gb2(0.1, 0);
      (void) gbs2(0.1, 0);
      (void) gb3(0.1, 0);
      (void) gbK(0.1, 0);
    }
    if (1 == rsd) {
      (void) a_chi(0.9);
      (void) W_RSD(100, 0.9, 0.95, 0);
    }
    if (nuisance.IA_MODEL == IA_MODEL_TATT && NULL != table) {
      if (0 == nuisance.IA_code) get_FPT_IA();
    }
    if (1 == nonlinear_bias) {
      if (0 == nuisance.IA_code) {
        get_FPT_bias();
      }
    }
  }

  // -----------------------------------------------------------------------
  // Allocate precomputed arrays, padded to the largest node count over the
  // lens bins (the bins' counts may differ; see the header)
  // -----------------------------------------------------------------------
  int npts_max = 0;
  for (int zl = 0; zl < redshift.clustering_nbin; zl++) {
    if (cn_all[zl].npts > npts_max) {
      npts_max = cn_all[zl].npts;
    }
  }

  double*** WB = (double***) malloc3d(10, redshift.clustering_nbin, npts_max);
  zero3d(WB, 10, redshift.clustering_nbin, npts_max);

  double**** WC = (double****) malloc4d(5, 
                                        redshift.clustering_nbin,
                                        redshift.shear_nbin, 
                                        npts_max);
  zero4d(WC, 5, redshift.clustering_nbin, redshift.shear_nbin, npts_max);

  double**** KIA = (double****) malloc4d(10, 
                                         redshift.clustering_nbin, 
                                         nell, 
                                         npts_max);
  zero4d(KIA, 10, redshift.clustering_nbin, nell, npts_max);

  // HOD galaxy-matter spectrum at the nodes: KH[0] = p_gm(k, a, zl)
  double**** KH = NULL;
  if (1 == hod) {
    KH = (double****) malloc4d(1, redshift.clustering_nbin, nell, npts_max);
  }

  // halo IA at the nodes: KHI[0] = P_delta f_2h, KHI[1] = P_1h,dI;
  // FRC = f_rc(a) per lens-bin node
  double**** KHI = NULL;
  double** FRC = NULL;
  if (1 == halo_ia) {
    KHI = (double****) malloc4d(2, redshift.clustering_nbin, nell, npts_max);
    FRC = (double**) malloc2d(redshift.clustering_nbin, npts_max);
  }

  double limTATT[3];
  double limbias[3];
  const int tatt = (nuisance.IA_MODEL == IA_MODEL_TATT && NULL != table);
  if (1 == halo_ia && 1 == tatt) {
    log_fatal("include_halo_IA supports the NLA model only");
    exit(1);
  }
  if (1 == hod && 1 == tatt) {
    log_fatal("TATT with HOD has no tree-level density leg: "
              "HOD C_l^gs supports NLA only");
    exit(1);
  }
  if (tatt) {
    if (0 == nuisance.IA_code) get_FPT_IA();
    limTATT[0] = log(FPTIA.krange[RANGE_MIN]);
    limTATT[1] = log(FPTIA.krange[RANGE_MAX]);
    limTATT[2] = (limTATT[1] - limTATT[0])/FPTIA.N;
  }
  if (1 == nonlinear_bias) {
    if (0 == nuisance.IA_code) get_FPT_bias();
    limbias[0] = log(FPTbias.krange[RANGE_MIN]);
    limbias[1] = log(FPTbias.krange[RANGE_MAX]);
    limbias[2] = (limbias[1] - limbias[0])/FPTbias.N;
  }
  
  // FKEM pivot per lens bin, as in C_cl_tomo (see the note there);
  // COSMO2D_FKEM_PIVOT_Z0 restores the z = 0 anchor.
  double apivw[MAX_SIZE_ARRAYS];
  double invgf2w[MAX_SIZE_ARRAYS];
  if (NULL != table_lin) {
    for (int i=0; i<redshift.clustering_nbin; i++) {
#ifdef COSMO2D_FKEM_PIVOT_Z0
      apivw[i] = 1.0;
#else
      apivw[i] = 1.0/(1.0 + zmean(i));
#endif
      const double gfp = growfac(apivw[i]);
      invgf2w[i] = 1.0/(gfp*gfp);
    }
  }

  // per-thread scratch of the batched P reads (Pdelta_at_a: one call per
  // node, the z half of the table read once per node instead of once per
  // multipole): KPN[3t] = the node's Limber wavenumbers, KPN[3t+1] =
  // P_delta, KPN[3t+2] = P_lin(k, a_piv) of the linear term
  double** KPN = (double**) malloc2d(3*omp_get_max_threads(), nell);
  // the one-loop bias and TATT inputs of the linear term's sum: zero, as
  // in a linear-only call
  double* ZERO = (double*) malloc1d(npts_max);
  for (int p=0; p<npts_max; p++) {
    ZERO[p] = 0.0;
  }
  #pragma omp parallel
  {
    // -----------------------------------------------------------------------
    // Precompute: lens weights, galaxy biases, source weights, IA amplitudes
    // -----------------------------------------------------------------------
    #pragma omp for collapse(2) schedule(static) nowait
    for (int zl = 0; zl < redshift.clustering_nbin; zl++) {
      for (int p = 0; p < npts_max; p++) {
        const cosmo_nodes* cn = &cn_all[zl];
        if (p >= cn->npts) {
          continue; // padding node: bin zl has fewer nodes
        }
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
    //
    // Threaded over (bin, node), the multipoles in the inner loop: one
    // batched P read per node (see KPN).
    // -----------------------------------------------------------------------
    #pragma omp for collapse(2) schedule(static)
    for (int zl = 0; zl < redshift.clustering_nbin; zl++) {
      for (int p = 0; p < npts_max; p++) {
        const cosmo_nodes* cn = &cn_all[zl];
        if (p >= cn->npts) {
          continue; // padding node: bin zl has fewer nodes
        }
        // P_delta, and the P_lin(k, a_piv) of the separable linear term
        double* restrict kn = KPN[3*omp_get_thread_num()];
        double* restrict pn = KPN[3*omp_get_thread_num() + 1];
        double* restrict pl = KPN[3*omp_get_thread_num() + 2];
        for (int i = 0; i < nell; i++) {
          kn[i] = (lx[i] + 0.5) / cn->data[CN_FK][p];
        }
        if (NULL != table) {
          Pdelta_at_a(cn->data[CN_A][p], kn, nell, pn);
        }
        if (NULL != table_lin) {
          p_lin_at_a(apivw[zl], kn, nell, pl);
        }
        for (int i = 0; i < nell; i++) {
          const double a  = cn->data[CN_A][p];
          const double fK = cn->data[CN_FK][p];
          const double ell = lx[i] + 0.5;
          const double k = ell / fK;
          const double lnk = log(k);
          // Linear term: the separable spectrum of the FFTLog term,
          // (D(a)/D(a_piv))^2 * P_lin(k, a_piv) anchored per lens bin,
          // never p_lin(k,a): only an identical separable form on both
          // sides lets the FFTLog/Limber pair cancel at high l.
          //
          // CAMB's growth is scale dependent (massive neutrinos); the
          // scale-dependent part lives in the Limber P_delta term,
          // where the FKEM split puts it.
          const double gf = cn->data[CN_GROWFAC][p];
          if (NULL != table) {
            KIA[0][zl][i][p] = pn[i];
          }
          if (NULL != table_lin) {
            KIA[9][zl][i][p] = gf*gf*invgf2w[zl]*pl[i];
          }
          if (1 == hod) {
            KH[0][zl][i][p] = p_gm(k, a, zl);
          }
          // RSD in Limber samples the kernel at TWO radii: the j_l''
          // of the exact velocity term couples neighboring Bessel
          // orders, so the extended-Limber W_RSD (radial_weights.c)
          // combines n(z)*H*f at the j_l peak chi_0 = (l+1/2)/k and at
          // the j_{l+1} peak chi_1 = (l+3/2)/k (here ell = l + 1/2,
          // so chi_0 = ell/k and chi_1 = (ell+1)/k). The gg and gk
          // copies of this recipe add a reach mask (see
          // C_gg_tomo_limber_work).
          if (1 == rsd) {
            const double chi_0 = ell/k;
            const double chi_1 = (ell + 1.0)/k;
            const double a_0 = a_chi(chi_0);
            const double a_1 = a_chi(chi_1);
            KIA[1][zl][i][p] = W_RSD(ell, a_0, a_1, zl);
          }
          if (tatt) {
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
  // Precompute (halo-model IA only): its own loop nests over every (lens
  // bin, ell, node), after P_delta is in place (as in C_ss_tomo_limber_work)
  // -----------------------------------------------------------------------
  if (1 == halo_ia) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int zl = 0; zl < redshift.clustering_nbin; zl++) {
      for (int p = 0; p < npts_max; p++) {
        if (p >= cn_all[zl].npts) {
          continue; // padding node: bin zl has fewer nodes
        }
        FRC[zl][p] = ia_f_red_central(cn_all[zl].data[CN_A][p]);
      }
    }

    #pragma omp parallel for collapse(3) schedule(static)
    for (int zl = 0; zl < redshift.clustering_nbin; zl++) {
      for (int i = 0; i < nell; i++) {
        for (int p = 0; p < npts_max; p++) {
          if (p >= cn_all[zl].npts) {
            continue; // padding node: bin zl has fewer nodes
          }
          const double a = cn_all[zl].data[CN_A][p];
          const double k = (lx[i] + 0.5)/cn_all[zl].data[CN_FK][p];

          KHI[0][zl][i][p] = KIA[0][zl][i][p]*ia_window_2h(k); // P f_2h
          KHI[1][zl][i][p] = ia_p1h_dI(k, a);                  // sats' dI
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
      const int npts = cn->npts; // the nodes of this pair's lens bin

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

      // o = 0: the full model into table; o = 1: the linear term into
      // table_lin (PK = the separable P_lin, the one-loop bias and TATT
      // inputs at zero; no HOD and no halo IA, so always the plain sum)
      for (int o=0; o<2; o++) {
        double** out = (0 == o) ? table : table_lin;
        if (NULL == out) {
          continue;
        }
        const double* restrict b2      = (0 == o) ? WB[4][ZLNZ] : ZERO;
        const double* restrict bs2     = (0 == o) ? WB[5][ZLNZ] : ZERO;
        const double* restrict b3      = (0 == o) ? WB[6][ZLNZ] : ZERO;
        const double* restrict bk      = (0 == o) ? WB[7][ZLNZ] : ZERO;

        const double* restrict PK      = (0 == o) ? KIA[0][ZLNZ][i] : KIA[9][ZLNZ][i];
        const double* restrict mixA    = (0 == o) ? KIA[2][ZLNZ][i] : ZERO;
        const double* restrict mixB    = (0 == o) ? KIA[3][ZLNZ][i] : ZERO;
        const double* restrict ta_dE1  = (0 == o) ? KIA[4][ZLNZ][i] : ZERO;
        const double* restrict ta_dE2  = (0 == o) ? KIA[5][ZLNZ][i] : ZERO;
        const double* restrict d1d2    = (0 == o) ? KIA[6][ZLNZ][i] : ZERO;
        const double* restrict d1s2    = (0 == o) ? KIA[7][ZLNZ][i] : ZERO;
        const double* restrict d1p3    = (0 == o) ? KIA[8][ZLNZ][i] : ZERO;

        double sum = 0.0;
        if (0 == o && 1 == hod) {
          /* PHYSICAL DERIVATION & LOGIC FLOW
             1. lens leg = W_gal p_gm + W_mag ell_prefactor b_mag P_delta
                (bias inside p_gm; magnification traces matter)
             2. source leg = W_kappa - W_source C1           (NLA only)
             3. C_l^gs = sum_p lens x source x (dchi/da) ep2 w_p / f_K^2 */
          const double* restrict PGM = KH[0][ZLNZ][i];
          #pragma omp simd reduction(+:sum)
          for (int p = 0; p < npts; p++) {
            const double amp  = (dchida[p]/(fK[p]*fK[p]))*ep2;
            const double lens = WGAL[p]*PGM[p] + WMAG[p]*ep*bmag[p]*PK[p];
            const double ans  = lens*(WK[p] - WS[p]*C1[p]);
            sum += ans*amp*wt[p];
          }
        }
        else if (0 == o && 1 == halo_ia) {
          /* PHYSICAL DERIVATION & LOGIC FLOW (Fortuna et al. 2021)
             1. lens leg = WGAL b1 + WMAG ep bmag + WRSD (perturbative)
             2. source leg = WK P - WS (f_rc C1 P f_2h + P_1h,dI)
             3. C_l^gs = sum_p [lens x source + WGAL oneloop (WK - WS f_rc
                C1)] (dchi/da) ep2 w_p / f_K^2 - the one-loop bias meets
                the tree-level IA only, as in the NLA core               */
          const double* restrict PKT  = KHI[0][ZLNZ][i];
          const double* restrict P1DI = KHI[1][ZLNZ][i];
          const double* restrict frc  = FRC[ZLNZ];
          #pragma omp simd reduction(+:sum)
          for (int p = 0; p < npts; p++) {
            const double g4  = growfac[p]*growfac[p]*growfac[p]*growfac[p];
            const double k   = ell / fK[p];
            const double amp = (dchida[p]/(fK[p]*fK[p]))*ep2;
            const double b1l =
                int_for_C_gs_tomo_limber_bias_oneloop_core(k,PK[p],g4,
                  b2[p],bs2[p],b3[p],bk[p],d1d2[p],d1s2[p],d1p3[p]);
            const double ia     = frc[p]*C1[p]*PKT[p] + P1DI[p];
            const double lens   = WGAL[p]*b1[p] + WMAG[p]*ep*bmag[p] + WRSD[p];
            const double source = WK[p]*PK[p] - WS[p]*ia;
            const double ans    = lens*source
                                  + WGAL[p]*b1l*(WK[p] - WS[p]*frc[p]*C1[p]);
            sum += ans*amp*wt[p];
          }
        }
        else {
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
        }
        out[j][i] = sum;
      }
    }
  }
  free(WB); free(WC); free(KIA); free(KPN); free(ZERO);
  if (KH != NULL) {
    free(KH);
  }
  if (KHI != NULL) {
    free(KHI);
    free(FRC);
  }
}

// ---------------------------------------------------------------------------
// Batch galaxy-shear Limber C_l^gs at arbitrary multipole values, for every
// ggl pair: the entry point of C_gs_tomo_limber_work.
//
// Builds the Gauss-Legendre quadrature nodes of each lens bin on its Limber
// range [amin_lens, amax_lens] (64 nodes at the default accuracy, more with
// Ntable.high_def_integration; a bin whose range magnification widens
// gets one such rule on its n(z) support and one on the foreground, see
// create_cosmo_nodes_lens), the ell prefactors
//
//   ell_prefactor  = l (l+1) / (l + 1/2)^2                  (magnification)
//   ell_prefactor2 = sqrt((l-1) l (l+1) (l+2)) / (l + 1/2)^2 (spin-2 shear)
//
// and hands everything to C_gs_tomo_limber_work.
//
// Two outputs, either may be NULL:
//   out     - the full model (P_delta, one-loop galaxy bias, NLA or TATT):
//             the Limber C_l of the likelihood
//             (C_gs_tomo_limber_nointerp_ells);
//   out_lin - the linear term that the non-Limber C_gs_tomo subtracts:
//             (D(a)/D(a_piv))^2 * P_lin(k, a_piv) per lens bin, b1 only,
//             IA through C1 only (the exact content of the FFTLog term).
// With both, one pass shares the nodes, the weights and the RSD kernel
// between them (see C_gs_tomo_limber_work), bitwise two separate calls.
//
// Example: w_gammat_tomo with like.adopt_limber[LIMBER_GS] = 0 calls C_gs_tomo,
// which calls this function once with ells = 0, 1, ..., 149 and both
// outputs to get both Limber terms of the non-Limber split at once.
//
// Cache invalidation:
// the static Gauss-Legendre table w (64/128/256/512/
// 1024 nodes, keyed on abs(Ntable.high_def_integration)) rebuilds when
// Ntable.random changes; the per-lens-bin cosmo_nodes are rebuilt on
// every call (they depend on the current cosmology).
//
// Parameters:
//   ells  - multipole values, length nell (need not be integers)
//   nell  - number of multipole values
//   NSIZE - number of ggl power spectra (= tomo.ggl_Npowerspectra)
//   out     - output [NSIZE][nell], the full model, indexed out[nz][i];
//             pair nz is (ZL(nz), ZS(nz)) (NULL: not computed)
//   out_lin - output [NSIZE][nell], the linear term (NULL: not computed)
//
// Returns:
//   nothing; the results are written into out and out_lin
// ---------------------------------------------------------------------------
void C_gs_tomo_limber_nl_lin_nointerp_ells(
    const double* ells,      // array of multipole values (length nell)
    const int nell,          // number of multipole values
    const int NSIZE,         // number of ggl power spectra
    double** out,            // output [NSIZE][nell], full model (or NULL)
    double** out_lin         // output [NSIZE][nell], linear term (or NULL)
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
    cn_all[zl] = create_cosmo_nodes_lens(zl, w);
  }
  if (nell <= 0) {
    log_fatal("nell = %d <= 0", nell); exit(1);
  }
  if (NULL == out && NULL == out_lin) {
    log_fatal("no output requested"); exit(1);
  }

  double** tmp_table = NULL;
  if (NULL != out) {
    tmp_table = (double**) malloc2d(NSIZE, nell);
    zero2d(tmp_table, NSIZE, nell);
  }
  double** tmp_lin = NULL;
  if (NULL != out_lin) {
    tmp_lin = (double**) malloc2d(NSIZE, nell);
    zero2d(tmp_lin, NSIZE, nell);
  }

  double* ep1  = (double*) malloc1d(nell);
  double* ep2 = (double*) malloc1d(nell);
  for (int i=0; i<nell; i++) {
    ep1[i] = ells[i]*(ells[i] + 1.)/((ells[i] + 0.5)*(ells[i] + 0.5));
    
    const double tmp = (ells[i] - 1.)*ells[i]*(ells[i] + 1.)*(ells[i] + 2.);
    ep2[i] = (tmp > 0) ? sqrt(tmp)/((ells[i] + 0.5)*(ells[i] + 0.5)) : 0.0;
  }

  C_gs_tomo_limber_work(cn_all, ells, ep1, ep2, nell, tmp_table, tmp_lin);

  for (int k = 0; k < NSIZE; k++) {
    for (int i = 0; i < nell; i++) {
      if (NULL != out) {
        out[k][i] = tmp_table[k][i];
      }
      if (NULL != out_lin) {
        out_lin[k][i] = tmp_lin[k][i];
      }
    }
  }

  if (NULL != tmp_table) {
    free(tmp_table);
  }
  if (NULL != tmp_lin) {
    free(tmp_lin);
  }
  free(ep1); free(ep2);

  for (int zl = 0; zl < redshift.clustering_nbin; zl++) {
    free_cosmo_nodes(&cn_all[zl]);
  }

  return;
}

// ---------------------------------------------------------------------------
// One of the two outputs of C_gs_tomo_limber_nl_lin_nointerp_ells:
// use_linear_ps = 0 the full model, 1 the linear term of the non-Limber
// split.
//
// Parameters:
//   ells  - multipole values, length nell (need not be integers)
//   nell  - number of multipole values
//   NSIZE - number of ggl power spectra (= tomo.ggl_Npowerspectra)
//   use_linear_ps - 1 = linear term of the non-Limber split, 0 = full model
//   out   - output [NSIZE][nell], indexed out[nz][i]
//
// Returns:
//   nothing; the result is written into out
// ---------------------------------------------------------------------------
void C_gs_tomo_limber_linpsopt_nointerp_ells(
    const double* ells,      // array of multipole values (length nell)
    const int nell,          // number of multipole values
    const int NSIZE,         // number of ggl power spectra
    const int use_linear_ps, // 1 = P_lin + linear kernels, 0 = full model
    double** out             // output [NSIZE][nell]
  )
{
  if (0 == use_linear_ps) {
    C_gs_tomo_limber_nl_lin_nointerp_ells(ells, nell, NSIZE, out, NULL);
  }
  else {
    C_gs_tomo_limber_nl_lin_nointerp_ells(ells, nell, NSIZE, NULL, out);
  }
}

// ---------------------------------------------------------------------------
// Batch galaxy-shear Limber C_l^gs (full model) at arbitrary multipole
// values: C_gs_tomo_limber_linpsopt_nointerp_ells with use_linear_ps = 0.
// Used by the Fourier-space data vectors (Limber case), the notebook
// wrapper C_gs_tomo_limber_cpp, and C_gs_tomo_limber_nointerp_batch.
//
// Parameters:
//   ells  - multipole values, length nell (need not be integers)
//   nell  - number of multipole values
//   NSIZE - number of ggl power spectra (= tomo.ggl_Npowerspectra)
//   out   - output [NSIZE][nell], indexed out[nz][i]
//
// Returns:
//   nothing; the result is written into out
// ---------------------------------------------------------------------------
void C_gs_tomo_limber_nointerp_ells(
    const double* ells,  // array of multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of ggl power spectra
    double** out         // output [NSIZE][nell]
  )
{
  C_gs_tomo_limber_linpsopt_nointerp_ells(ells, nell, NSIZE, 0, out);
}

// ---------------------------------------------------------------------------
// Batch galaxy-shear Limber C_l at the integer multipoles l = lmin..lmax-1,
// written at their own index: Cl[nz][l]. Thin wrapper around
// C_gs_tomo_limber_nointerp_ells.
//
// Example: w_gammat_tomo calls it with lmin = 1 and lmax = limits.LMIN_tab
// for the multipoles below the interpolation table; Cl[nz][0] is left
// untouched.
//
// Parameters:
//   lmin  - first multipole (inclusive)
//   lmax  - last multipole (exclusive)
//   NSIZE - number of ggl power spectra (= tomo.ggl_Npowerspectra)
//   Cl    - output [NSIZE][>= lmax], indexed Cl[nz][l]
//
// Returns:
//   nothing; the result is written into Cl
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
// Galaxy-shear angular power spectrum C_l^gs with interpolation.
//
// On first call (or when cosmology/nuisance parameters change), builds a
// log-spaced interpolation table covering l = LMIN_tab..LMAX (Ntable.N_ell[NODES_DENSE]
// points) using C_gs_tomo_limber_work with per-lens-bin cosmo_nodes and
// precomputed ell prefactors, then caches it for subsequent lookups.
// Returns the interpolated value at the requested l via interpol1d.
//
// When Ntable.N_ell[NODES_COARSE] is active, the exact quadrature instead
// runs on the internal coarse grid and the house cubic spline
// upsamples onto the unchanged N_ell nodes (the strategy block inside
// explains why this wins).
//
// Why the table is shared through the gs_ static struct: the
// real-space projection (w_gammat_tomo) needs C_l at every integer
// multipole up to Ntable.LMAX ~ 1e5, for every tomographic pair,
// inside its Legendre/Hankel sums - millions of table reads per
// likelihood evaluation. Only the vectorized batch reader
// (C_gs_tomo_limber_fill, which runs the interpol1d linear read four
// multipoles at a time through AVX2 gathers) sustains that rate;
// calling this function one multipole at a time would dominate the
// whole evaluation.
//
// The struct is how the table travels between the two functions.
// The builder (this function) and the reader (the _fill) never call
// each other - the real-space projection calls one, the C_ell paths
// call the other - so no argument list connects them. Instead the
// builder publishes the table pointer and the grid geometry (the
// ln(ell) limits, spacing and node count) in the file-scope struct,
// and the reader picks them up there.
//
// Only lens-source pairs with redshift overlap contribute (test_zoverlap).
//
// Cache invalidation:
// the static table, grid limits, Gauss-Legendre table
// and ell arrays rebuild when the table is NULL or Ntable.random changes;
// the values refill when any of these change:
//   cosmology.random, nuisance.random_photoz_shear,
//   nuisance.random_photoz_clustering, nuisance.random_ia,
//   redshift.random_shear, redshift.random_clustering,
//   Ntable.random, nuisance.random_galaxy_bias
//
// Parameters:
//   l  - multipole moment (continuous; outside the grid the lookup warns
//        and extrapolates)
//   ni - lens redshift bin
//   nj - source redshift bin
//
// Returns:
//   C_l^gs of the (ni, nj) pair; 0 for a pair excluded by test_zoverlap
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
  static int ncoarse = 0;  // active internal coarse grid size (0 = off)
  static double dlnc = 0.; // coarse grid spacing in ln(ell)
  static double* lxc = NULL;   // coarse ell nodes
  static double* epc = NULL;   // coarse ell prefactors (as ep/ep2)
  static double* ep2c = NULL;
  static int* qidx = NULL;     // fine node -> coarse interval (uniform
  static double* qdel = NULL;  //   grids: precomputed, no search)
  static double** tabc = NULL; // coarse C_ell values
  static double** cspl = NULL; // natural-cubic-spline c coefficients

  if (NULL == table || fdiff2(cache[6], Ntable.random)) {
    nell   = Ntable.N_ell[NODES_DENSE];
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

    // Curved-sky (extended Limber) ell prefactors, tabulated per node
    // (1812.05995 eqs 74-79). The Limber kernel is evaluated at
    // k = (l + 1/2)/chi, and each projected field carries the exact
    // prefactor of its spin:
    //
    //   ep  = l(l+1)/(l+1/2)^2                  magnification (the
    //         angular Laplacian eigenvalue l(l+1) over the flat-sky
    //         (l+1/2)^2)
    //
    //   ep2 = sqrt((l-1)l(l+1)(l+2))/(l+1/2)^2  shear (the spin-2
    //         factor sqrt((l+2)!/(l-2)!) from two covariant
    //         derivatives acting on the lensing potential)
    //
    // Both approach 1 for l >> 1 (the flat-sky limit). At l = 1 the
    // (l-1) factor makes ep2 exactly zero - a spin-2 field has no
    // l < 2 multipoles - which the (tmp > 0) guard implements without
    // taking the sqrt of a negative rounding.
    for (int i = 0; i < nell; i++) {
      lx[i] = exp(lim[0] + i * lim[2]);
      const double ell = lx[i] + 0.5;
      ep[i] = lx[i]*(lx[i]+1.)/(ell*ell);
      const double tmp = (lx[i]-1.)*lx[i]*(lx[i]+1.)*(lx[i]+2.);
      ep2[i] = (tmp > 0) ? sqrt(tmp)/(ell*ell) : 0.0;
    }

    // Coarse-grid workspace (the strategy is explained where the grid
    // is used, in the refill block below): every allocation lives
    // HERE, in the Ntable rebuild block; the per-cosmology refill only
    // fills. The pieces are:
    //   lxc, epc, ep2c - the ncoarse ell nodes, log-spaced over the
    //                same [lim[0], lim[1]] range as the fine table,
    //                and their curved-sky prefactors: the same
    //                formulas tabulated for the fine grid above,
    //                evaluated on the coarse nodes
    //   tabc, cspl - the coarse C_ell values and their cubic-spline
    //                coefficients, one row per (lens, source) pair
    //   qidx, qdel - for each fine node, the coarse interval it falls
    //                in and its ln(ell) offset from that interval's
    //                left node: both grids are uniform in ln(ell) with
    //                shared endpoints, so this is pure grid geometry,
    //                computed once - no search of any kind at refill
    if (lxc  != NULL) { free(lxc);  lxc  = NULL; }
    if (epc  != NULL) { free(epc);  epc  = NULL; }
    if (ep2c != NULL) { free(ep2c); ep2c = NULL; }
    if (qidx != NULL) { free(qidx); qidx = NULL; }
    if (qdel != NULL) { free(qdel); qdel = NULL; }
    if (tabc != NULL) { free(tabc); tabc = NULL; }
    if (cspl != NULL) { free(cspl); cspl = NULL; }
    const int nc = Ntable.N_ell[NODES_COARSE];
    ncoarse = (nc > 3 && nc < nell) ? nc : 0;
    if (ncoarse > 0) {
      dlnc = (lim[1] - lim[0]) / ((double) ncoarse - 1.0);
      lxc  = (double*) malloc1d(ncoarse);
      epc  = (double*) malloc1d(ncoarse);
      ep2c = (double*) malloc1d(ncoarse);
      for (int i=0; i<ncoarse; i++) {
        lxc[i] = exp(lim[0] + i*dlnc);
        const double ell = lxc[i] + 0.5;
        epc[i] = lxc[i]*(lxc[i]+1.)/(ell*ell);
        const double tmp = (lxc[i]-1.)*lxc[i]*(lxc[i]+1.)*(lxc[i]+2.);
        ep2c[i] = (tmp > 0) ? sqrt(tmp)/(ell*ell) : 0.0;
      }
      qidx = (int*) malloc(sizeof(int) * nell);
      qdel = (double*) malloc1d(nell);
      for (int i=0; i<nell; i++) {
        // Where does fine node i sit on the coarse grid? Both grids
        // run over the same [lim[0], lim[1]] in ln(ell), so the map
        // is pure arithmetic:
        //
        //   fine node i -> ln(ell) = lim[0] + i*lim[2]
        //               -> r = i*lim[2]/dlnc   (coarse spacings in)
        //               -> j = (int) r         (interval's left node)
        //               -> qdel = (r - j)*dlnc (offset inside it)
        //
        // The spline evaluates on interval [j, j+1], so the largest
        // legal j is ncoarse-2, the left node of the LAST interval.
        //
        // Why the clamp: at the shared top endpoint, i*lim[2] and
        // (ncoarse-1)*dlnc are two floating-point roundings of the
        // same length lim[1] - lim[0]. r can therefore land one ulp
        // above ncoarse-1 and truncate to j = ncoarse-1 - one past
        // the last interval. The clamp moves that node back onto the
        // last interval, where it evaluates at (at most one ulp
        // past) the interval's right endpoint.
        const double r = (double) i * lim[2] / dlnc;
        int j = (int) r;
        if (j > ncoarse - 2) {
          j = ncoarse - 2;
        }
        qidx[i] = j;
        qdel[i] = (r - j) * dlnc; // offset from node j, in ln(ell)
      }
      tabc = (double**) malloc2d(tomo.ggl_Npowerspectra, ncoarse);
      cspl = (double**) malloc2d(tomo.ggl_Npowerspectra, ncoarse);
    }
  }

  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_photoz_clustering) ||
      fdiff2(cache[3], nuisance.random_ia) ||
      fdiff2(cache[4], redshift.random_shear) ||
      fdiff2(cache[5], redshift.random_clustering) ||
      fdiff2(cache[6], Ntable.random) ||
      fdiff2(cache[7], nuisance.random_galaxy_bias) ||
      fdiff2(cache[8], (uint64_t) include_HOD_GX) ||
      fdiff2(cache[9], (uint64_t) include_halo_IA) ||
      fdiff2(cache[10], nuisance.random_ia_halo))
  {
    // per-lens-bin nodes (split rule with magnification, see
    // create_cosmo_nodes_lens)
    cosmo_nodes cn_all[redshift.clustering_nbin];
    for (int zl = 0; zl < redshift.clustering_nbin; zl++) {
      cn_all[zl] = create_cosmo_nodes_lens(zl, w);
    }

    if (ncoarse > 0) {
      // ---------------------------------------------------------------
      // The internal coarse grid: general strategy.
      //
      // The real-space projections (w_gammat_tomo, via the shared gs_
      // struct and C_gs_tomo_limber_fill) read this table at every
      // integer ell up to Ntable.LMAX ~ 1e5 inside their Legendre
      // sums. At that call rate only the optimized, vectorized LINEAR
      // read is affordable: a cubic-spline lookup per ell would
      // dominate the whole evaluation.
      //
      // A linear read, however, is only accurate on a DENSE table -
      // and each of the N_ell = 512 nodes costs one exact Limber
      // quadrature, which is the expensive part.
      //
      // The coarse grid splits the difference: a cubic spline carries
      // far more accuracy per node than a linear segment, so the
      // expensive quadratures run on few nodes and a cheap cubic
      // upsampling fills the dense table:
      //
      //   exact Limber quadrature on ncoarse nodes (default 192)
      //     -> spline_coeffs_uniform: one tridiagonal solve per row
      //     -> Horner evaluation at the 512 precomputed fine offsets
      //     -> the unchanged dense table
      //     -> the same fast linear reads by every consumer
      //
      // This is safe because
      // C_gs is smooth in ln(ell); C_gg keeps the exact grid - its
      // BAO wiggles would be undersampled (see its header).
      // ---------------------------------------------------------------
      zero2d(tabc, tomo.ggl_Npowerspectra, ncoarse);

      C_gs_tomo_limber_work(cn_all, lxc, epc, ep2c, ncoarse, tabc, NULL);

      const double hc = dlnc;
      const double inv_hc = 1.0/dlnc;
      #pragma omp parallel for schedule(static)
      for (int nz=0; nz<tomo.ggl_Npowerspectra; nz++) {
        spline_coeffs_uniform(tabc[nz], ncoarse, hc, cspl[nz]);
      }
      // Upsampling. On interval [x_j, x_j + h] the house spline
      // (spline_coeffs_uniform) is the cubic
      //
      //   S(x_j + dx) = y_j + b dx + c_j dx^2 + d dx^3
      //
      // where c is the coefficient array the tridiagonal solve above
      // produced: the spline's second derivative / 2, with natural
      // boundaries c_0 = c_{n-1} = 0.
      //
      // The other two coefficients follow from two conditions:
      //
      //   S'' runs linearly from 2 c_j to 2 c_{j+1}
      //     -> d = (c_{j+1} - c_j) / (3 h)
      //
      //   S(x_{j+1}) = y_{j+1}, interpolate the right node
      //     -> b = (y_{j+1} - y_j)/h - h (c_{j+1} + 2 c_j)/3
      //
      // The polynomial is evaluated in Horner form; qidx/qdel hold
      // each fine node's precomputed interval j and offset dx.
      #pragma omp parallel for collapse(2) schedule(static)
      for (int nz=0; nz<tomo.ggl_Npowerspectra; nz++) {
        for (int i=0; i<nell; i++) {
          const double* restrict y = tabc[nz];
          const double* restrict cc = cspl[nz];
          const int j = qidx[i];
          const double b = (y[j+1] - y[j])*inv_hc
                           - hc*(cc[j+1] + 2.0*cc[j])/3.0;
          const double d = (cc[j+1] - cc[j])/(3.0*hc);
          table[nz][i] = y[j] + qdel[i]*(b + qdel[i]*(cc[j] + qdel[i]*d));
        }
      }
    }
    else {
      C_gs_tomo_limber_work(cn_all, lx, ep, ep2, nell, table, NULL);
    }

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
    cache[8] = (uint64_t) include_HOD_GX;
    cache[9] = (uint64_t) include_halo_IA;
    cache[10] = nuisance.random_ia_halo;
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
// C_l -> gamma_t(theta). Uses limber_fill_interp which processes 4 ells per
// iteration via AVX2 gather instructions (i32gather_pd).
//
// Requires C_gs_tomo_limber to have been called first to populate gs_.tab.
//
// Parameters:
//   nz     - tomographic pair index (0..ggl_Npowerspectra-1)
//   lmin   - first multipole to fill (inclusive)
//   lmax   - last multipole to fill (exclusive)
//   ln_ell - precomputed log(l) array, indexed by l
//   out    - output C_l array, indexed by l
//
// Returns:
//   nothing; the interpolated C_l are written into out at indices
//   lmin..lmax-1
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
// GG = GALAXY CLUSTERING
// ---------------------------------------------------------------------------
// GG opens the four remaining probe families (GG, GK, KS, KK). Unlike SS
// and GS, these have Npowerspectra = nbin (auto-correlations only for GG,
// or one index per bin for GK/KS/KK), not nbin*(nbin+1)/2. GG, GK and KS
// use the same _work batch design as SS and GS (cosmo_nodes, precomputed
// radial weights and kernels, SIMD-vectorized quadrature); KK keeps the
// legacy scalar pattern (a single spectrum, no tomography).
//
// The vectorized _fill functions (C_gg_tomo_limber_fill, etc.) are used
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
// Single-ell galaxy clustering C_l with the linear-spectrum switch: a point
// diagnostic on the batch engine.
//
// Runs one C_gg_tomo_limber_linpsopt_nointerp_ells call at a single
// multipole and reads one entry, so it pays the WHOLE-TOMOGRAPHY batch cost
// per call (every lens bin is computed even though one number is returned).
// Never loop this over (l, ni): call the batch once and index the result
// instead.
//
// Kept in the API as the exact per-multipole entry point of the probe (the
// C_ss_tomo_limber_nointerp pattern): notebooks evaluate single points
// here, and validation scripts compare the two use_linear_ps contents at
// one multipole (C_cl_tomo backfills from the batch, not from this
// wrapper).
//
// Parameters:
//   l             - multipole moment
//   ni            - lens redshift bin index (0..redshift.clustering_nbin-1)
//   nj            - second lens redshift bin index; must equal ni (the data
//                   vector carries clustering auto spectra only)
//   use_linear_ps - 1: separable linear spectrum
//                   (D(a)/D(a_piv))^2 P_lin(k, a_piv) per lens bin, no
//                   one-loop bias (the linear term C_cl_tomo subtracts);
//                   0: the full model
//
// Returns:
//   C_l^gg of lens bin ni with the selected power-spectrum content
// ---------------------------------------------------------------------------
double C_gg_tomo_limber_linpsopt_nointerp(
    const double l,
    const int ni,
    const int nj,
    const int use_linear_ps
  ) // slow (whole-tomography batch per call) - use the batch version
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1 ||
      nj < 0 || nj > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number (ni, nj) = [%d,%d]", ni, nj);
    exit(1);
  }
  if (ni != nj) {
    log_fatal("cross-tomography (ni,nj) = (%d,%d) bins not supported", ni, nj);
    exit(1);
  }
  const int NSIZE = redshift.clustering_nbin;
  double** tmp = (double**) malloc2d(NSIZE, 1);
  const double ell = l;

  C_gg_tomo_limber_linpsopt_nointerp_ells(&ell, 1, NSIZE, use_linear_ps, tmp);

  const double res = tmp[ni][0];
  free(tmp);
  return res;
}

// ---------------------------------------------------------------------------
// Single-ell galaxy clustering C_l using the nonlinear power spectrum.
//
// Convenience wrapper around C_gg_tomo_limber_linpsopt_nointerp with
// use_linear_ps = 0, matching the API pattern of the other probes
// (C_ss_tomo_limber_nointerp, C_gs_tomo_limber_nointerp): the same
// whole-tomography batch cost per call applies.
//
// Parameters:
//   l    - multipole moment
//   ni   - lens redshift bin index (0..redshift.clustering_nbin-1)
//   nj   - second lens redshift bin index; must equal ni (the data vector
//          carries clustering auto spectra only)
//
// Returns:
//   C_l^gg of lens bin ni with the full Limber model (nonlinear P_delta,
//   one-loop bias when enabled)
// ---------------------------------------------------------------------------
double C_gg_tomo_limber_nointerp(
    const double l,
    const int ni,
    const int nj
  ) // slow (whole-tomography batch per call) - use the batch version
{
  return C_gg_tomo_limber_linpsopt_nointerp(l, ni, nj, 0);
}

// ---------------------------------------------------------------------------
// Batched galaxy-clustering Limber C_l^gg (auto spectra, ni = nj): the core
// of the gg batch API. Callers: the interpolation table of C_gg_tomo_limber,
// the low-ell loop of w_gg_tomo (through C_gg_tomo_limber_nointerp_batch),
// the two Limber terms of the non-Limber C_cl_tomo, the Fourier-space data
// vectors, and the notebook wrapper C_gg_tomo_limber_cpp.
//
// Computes, for every lens bin and every multipole in lx,
//
//   C_l = SUM_p wt_p * mask_p * [ (WGAL*b1 + WMAG*ep*bmag + WRSD)^2 * PK
//                                 + oneloop ] * (dchi/da) / fK^2
//
// over the Gauss-Legendre nodes p of the bin (cn_all[bin], cn_all[bin].npts
// of them; the count may differ between bins), with
//   ep      = l (l+1) / (l + 1/2)^2           (magnification ell prefactor)
//   PK      = P_delta(k, a) for table, or (D(a)/D(a_piv))^2 *
//             P_lin(k, a_piv) per lens bin for the linear term table_lin,
//             at the Limber wavenumber k = (l + 1/2)/fK
//   WRSD    = W_RSD(l + 1/2, a_0, a_1, bin), chi_0 = fK, chi_1 = (l + 3/2)/k
//             (zero unless include_RSD_GG)
//   oneloop = WGAL^2 * [ D^4 (b1 b2 P_d1d2 + b2^2/4 P_d2d2 + b1 bs2 P_d1s2
//             + b2 bs2/2 P_d2s2 + bs2^2/4 P_s2s2 + b1 b3 P_d1p3)
//             + 2 b1 bK k^2 PK ]   (one-loop bias on, use_linear_ps = 0)
//   mask_p  = 0 where the RSD kernel would reach beyond chi(limits.a_min)
//             (chi_1 > chi(a_min)), 1 elsewhere.
// The scalar C_gg_tomo_limber_linpsopt_nointerp reads one entry of one
// batch call, so scalar and batch agree exactly by construction.
//
// The functions of the node alone (W_gal, W_mag, b1, bmag, b2, bs2, b3,
// bK) are evaluated once per node, and only P(k,a), W_RSD and the one-loop
// tables once per (node, ell); the sum over nodes is a vectorized loop.
// W_RSD and P_delta dominate the cost (lsst_y1, 1750 ells x 5 bins, one
// thread: 33 ms; the retired per-(node, ell) GSL path took 45 ms).
//
// Memory layout (npts = npts_max, the largest node count over the bins;
// the precompute loops skip a bin's padding nodes and its sum never reads
// them):
//   WB[4][nbin][npts]        W_gal, W_mag, b1, bmag
//   WO[4][nbin][npts]        b2, bs2, b3, bK            (one-loop bias only)
//   KG[4][nbin][nell][npts]  P_delta, W_RSD, mask, separable P_lin
//                            (slot 3 only with table_lin)
//   KB[6][nbin][nell][npts]  P_d1d2, P_d2d2 - 2 sigma4, P_d1s2,
//                            P_d2s2 - 4/3 sigma4, P_s2s2 - 8/9 sigma4, P_d1p3
//                            (one-loop bias only; zero outside the FPTbias
//                            k range, as in the scalar integrand)
//
// Cache invalidation:
// none here - every input arrives precomputed; the
// warm-up prelude initializes the lazily-built statics of the kernel and
// power-spectrum functions single-threaded before the parallel regions.
//
// Parameters:
//   cn_all        - quadrature nodes per lens bin [clustering_nbin]
//   lx            - multipole values (length nell)
//   ell_prefactor - l (l+1)/(l + 1/2)^2 per multipole
//   nell          - number of multipole values
//   table         - output [clustering_nbin][nell], the full model
//                   (NULL: not computed)
//   table_lin     - output [clustering_nbin][nell], the linear term
//                   C_cl_tomo subtracts: separable linear spectrum, no
//                   one-loop bias (NULL: not computed)
//
// Why one call can fill both: the non-Limber C_cl_tomo needs both terms
// at the same multipoles, and they differ only in PK (and in the one-loop
// terms, which the linear term never has). The nodes, the radial weights
// and the RSD kernel with its mask - the a_chi and W_RSD lookups that
// dominate the cost - are computed once and summed twice. Each output's
// sum is the same code it would be alone, so filling both at once is
// bitwise two separate calls.
//
// Returns:
//   nothing; the results are written into table and table_lin
// ---------------------------------------------------------------------------
static void C_gg_tomo_limber_work(
    const cosmo_nodes* cn_all,    // quadrature nodes per lens bin [clustering_nbin]
    const double* lx,             // multipole values (length nell)
    const double* ell_prefactor,  // l*(l+1)/(l+0.5)^2 per ell (magnification)
    const int nell,               // number of multipole values
    double** table,               // output [clustering_nbin][nell], full model (or NULL)
    double** table_lin            // output [clustering_nbin][nell], linear term (or NULL)
  )
{
  // -------------------------------------------------------------------------
  // HOD mode (include_HOD_GX = 1): the galaxies are the halo-model
  // occupation field, so the density weight is W_gal alone (the legacy
  // W_HOD weight n_i(z) H/H0; no bias factor - the bias lives inside
  // the HOD spectra) and the power comes from halo.c:
  //
  //   density-density         W_gal^2                  p_gg(k, a, zl, zl)
  //   density-magnification   2 W_gal W_mag ep b_mag   p_gm(k, a, zl)
  //   magnification-magnif.   (W_mag ep b_mag)^2       P_delta(k, a)
  //
  // The one-loop bias expansion and the RSD term have no HOD
  // counterpart (the legacy C_cl_HOD carried neither); both are off in
  // this mode. HOD C_l^gg is Limber-only, so the linear term of the
  // non-Limber split aborts: run with adopt_limber_gg = 1.
  // -------------------------------------------------------------------------
  const int hod = include_HOD_GX;
  if (1 == hod && NULL != table_lin) {
    log_fatal("HOD C_l^gg is Limber-only: set adopt_limber_gg = 1");
    exit(1);
  }

  const int nbin = redshift.clustering_nbin;

  // per-node arrays are padded to the largest node count over the bins
  int npts_max = 0;
  for (int zl=0; zl<nbin; zl++) {
    if (cn_all[zl].npts > npts_max) {
      npts_max = cn_all[zl].npts;
    }
  }

  int nonlinear_bias = 0;
  if (NULL != table && 0 == hod) {
    nonlinear_bias = has_b2_galaxies();
  }

  // RSD is not part of the HOD model (see the note above)
  int rsd = 0;
  if (1 == include_RSD_GG && 0 == hod) {
    rsd = 1;
  }
  // -----------------------------------------------------------------------
  // Warm up all functions that lazily initialize internal static tables.
  // Must be called single-threaded before any parallel region touches them.
  // -----------------------------------------------------------------------
  {
    const cosmo_nodes* cn = &cn_all[0];
    const double a    = cn->data[CN_A][0];
    const double fK   = cn->data[CN_FK][0];
    const double hoh0 = cn->data[CN_HOVERH0][0];
    const double ell  = lx[0] + 0.5;
    (void) W_gal(a, 0, hoh0);
    (void) W_mag(a, fK, 0);
    (void) Pdelta(ell/fK, a);
    if (NULL != table_lin) {
      (void) p_lin(ell/fK, a);
    }
    (void) gb1(0.1, 0);
    (void) gbmag(0.1, 0);
    if (1 == hod) {
      // halo.c builds its (a, ln k) tables inside its own OpenMP
      // regions; trigger the builds here, before this function's
      // parallel regions (one call fills every lens bin)
      (void) p_gg(ell/fK, a, 0, 0);
      (void) p_gm(ell/fK, a, 0);
    }
    if (1 == rsd) {
      (void) chi(limits.a_min);
      (void) a_chi(0.9);
      (void) W_RSD(100, 0.9, 0.95, 0);
    }
    if (1 == nonlinear_bias) {
      (void) gb2(0.1, 0);
      (void) gbs2(0.1, 0);
      (void) gb3(0.1, 0);
      (void) gbK(0.1, 0);
      if (0 == nuisance.IA_code) {
        get_FPT_bias();
      }
    }
  }
  const double chi_a_min = (1 == rsd) ? chi(limits.a_min) : 0.0;
  double limbias[3] = {0.0, 0.0, 0.0};
  double s4 = 0.0;
  if (1 == nonlinear_bias) {
    limbias[0] = log(FPTbias.krange[RANGE_MIN]);
    limbias[1] = log(FPTbias.krange[RANGE_MAX]);
    limbias[2] = (limbias[1] - limbias[0])/FPTbias.N;
    s4 = FPTbias.sigma4;
  }

  // -----------------------------------------------------------------------
  // Allocate precomputed arrays (padded to npts_max)
  // -----------------------------------------------------------------------
  double*** WB  = (double***) malloc3d(4, nbin, npts_max);
  // KG[3], the separable linear spectrum, exists only with table_lin
  double**** KG = (double****) malloc4d((NULL != table_lin) ? 4 : 3,
                                        nbin, nell, npts_max);
  // HOD spectra at the nodes: KH[0] = p_gg(k, a, zl, zl),
  // KH[1] = p_gm(k, a, zl)
  double**** KH = NULL;
  if (1 == hod) {
    KH = (double****) malloc4d(2, nbin, nell, npts_max);
  }
  double*** WO  = NULL;
  double**** KB = NULL;
  if (1 == nonlinear_bias) {
    WO = (double***) malloc3d(4, nbin, npts_max);
    KB = (double****) malloc4d(6, nbin, nell, npts_max);
  }

  // FKEM pivot per lens bin, as in C_cl_tomo (see the note there);
  // COSMO2D_FKEM_PIVOT_Z0 restores the z = 0 anchor.
  double apivw[MAX_SIZE_ARRAYS];
  double invgf2w[MAX_SIZE_ARRAYS];
  if (NULL != table_lin) {
    for (int i=0; i<nbin; i++) {
#ifdef COSMO2D_FKEM_PIVOT_Z0
      apivw[i] = 1.0;
#else
      apivw[i] = 1.0/(1.0 + zmean(i));
#endif
      const double gfp = growfac(apivw[i]);
      invgf2w[i] = 1.0/(gfp*gfp);
    }
  }

  // per-thread scratch of the batched P reads (Pdelta_at_a: one call per
  // node, the z half of the table read once per node instead of once per
  // multipole): KPN[2t] = the node's Limber wavenumbers, KPN[2t+1] = P
  double** KPN = (double**) malloc2d(2*omp_get_max_threads(), nell);
  #pragma omp parallel
  {
    // -----------------------------------------------------------------------
    // Precompute: lens weights and galaxy biases
    // -----------------------------------------------------------------------
    #pragma omp for collapse(2) schedule(static) nowait
    for (int zl=0; zl<nbin; zl++) {
      for (int p=0; p<npts_max; p++) {
        const cosmo_nodes* cn = &cn_all[zl];
        if (p >= cn->npts) {
          continue; // padding node: bin zl has fewer nodes
        }
        const double a = cn->data[CN_A][p];
        const double z = 1.0/a - 1.0;
        WB[0][zl][p] = W_gal(a, zl, cn->data[CN_HOVERH0][p]);
        WB[1][zl][p] = W_mag(a, cn->data[CN_FK][p], zl);
        WB[2][zl][p] = gb1(z, zl);
        WB[3][zl][p] = gbmag(z, zl);
        if (1 == nonlinear_bias) {
          WO[0][zl][p] = gb2(z, zl);
          WO[1][zl][p] = gbs2(z, zl);
          WO[2][zl][p] = gb3(z, zl);
          WO[3][zl][p] = gbK(z, zl);
        }
      }
    }
    // -----------------------------------------------------------------------
    // Precompute: P(k,a) of every node at every multipole, one batched read
    // per node (see KPN). Its own loop over (bin, node), since the fill
    // below runs over (bin, ell, node); nowait, because the two write
    // disjoint slots: a thread done here starts on the fill at once.
    // -----------------------------------------------------------------------
    #pragma omp for collapse(2) schedule(static) nowait
    for (int zl=0; zl<nbin; zl++) {
      for (int p=0; p<npts_max; p++) {
        const cosmo_nodes* cn = &cn_all[zl];
        if (p >= cn->npts) {
          continue; // padding node: bin zl has fewer nodes
        }
        double* restrict kn = KPN[2*omp_get_thread_num()];
        double* restrict pn = KPN[2*omp_get_thread_num() + 1];
        for (int i=0; i<nell; i++) {
          kn[i] = (lx[i] + 0.5)/cn->data[CN_FK][p];
        }
        if (NULL != table) {
          Pdelta_at_a(cn->data[CN_A][p], kn, nell, pn);
          for (int i=0; i<nell; i++) {
            KG[0][zl][i][p] = pn[i];
          }
        }
        if (NULL != table_lin) {
          // Linear term: the separable spectrum of the FFTLog term of
          // C_cl_tomo, (D(a)/D(a_piv))^2 * P_lin(k, a_piv) anchored per
          // lens bin, never p_lin(k,a): only an identical separable
          // form on both sides lets the FFTLog/Limber pair cancel at
          // high l.
          //
          // CAMB's growth is scale dependent (massive neutrinos); the
          // scale-dependent part lives in the Limber P_delta term,
          // where the FKEM split puts it.
          const double gf = cn->data[CN_GROWFAC][p];
          p_lin_at_a(apivw[zl], kn, nell, pn);
          for (int i=0; i<nell; i++) {
            KG[3][zl][i][p] = gf*gf*invgf2w[zl]*pn[i];
          }
        }
      }
    }
    // -----------------------------------------------------------------------
    // Precompute: RSD kernel and its support, one-loop kernels
    //
    // schedule(dynamic): the iterations cost very different amounts - a
    // padding node returns at once, a node the RSD mask drops skips
    // a_chi and W_RSD, a full node pays for all of them - so the static
    // split left threads idle at this loop's barrier (~4% of all cycles
    // of a des_cluster 6x2pt+N evaluation, perf on amypond, v5.00), and
    // one thread preempted by another process stalled the whole team.
    // Each iteration writes only its own (bin, ell, node) slots, so the
    // order the chunks run in cannot change a bit of the result. A chunk
    // of 128 iterations is ~50 us of work, far above the cost of taking
    // it from the shared counter.
    // -----------------------------------------------------------------------
    #pragma omp for collapse(3) schedule(dynamic, 128)
    for (int zl=0; zl<nbin; zl++) {
      for (int i=0; i<nell; i++) {
        for (int p=0; p<npts_max; p++) {
          const cosmo_nodes* cn = &cn_all[zl];
          if (p >= cn->npts) {
            continue; // padding node: bin zl has fewer nodes
          }
          const double a   = cn->data[CN_A][p];
          const double fK  = cn->data[CN_FK][p];
          const double ell = lx[i] + 0.5;
          const double k   = ell/fK;
          KG[1][zl][i][p] = 0.0;
          KG[2][zl][i][p] = 1.0;
          if (1 == hod) {
            KH[0][zl][i][p] = p_gg(k, a, zl, zl);
            KH[1][zl][i][p] = p_gm(k, a, zl);
          }
          if (1 == rsd) {
            // two-radius sampling: see the W_RSD note in
            // C_gs_tomo_limber_work. The mask: the distance tables
            // cover a >= limits.a_min, i.e. chi <= chi(a_min); a node
            // whose second radius chi_1 reaches beyond that cannot be
            // converted by a_chi, so the whole node is masked to zero
            // instead of extrapolated.
            const double chi_0 = ell/k;
            const double chi_1 = (ell + 1.0)/k;
            if (chi_1 > chi_a_min) {
              KG[2][zl][i][p] = 0.0;
            }
            else {
              const double a_0 = a_chi(chi_0);
              const double a_1 = a_chi(chi_1);
              KG[1][zl][i][p] = W_RSD(ell, a_0, a_1, zl);
            }
          }
          if (1 == nonlinear_bias) {
            const double lnk = log(k);
            const int in = (lnk >= limbias[0] && lnk <= limbias[1]);
            const double* lb = limbias;
            const int N = FPTbias.N;
            // sigma4 subtractions: the quadratic-operator spectra tend
            // to constants as k -> 0 (two Wick contractions of two
            // P_lin factors; each s^2 leg contributes the k -> 0 limit
            // of the tidal kernel, S2(q, -q) = 2/3):
            //   P_d2d2 -> 2*sigma4
            //   P_d2s2 -> (2/3)*2*sigma4  = 4/3*sigma4
            //   P_s2s2 -> (2/3)^2*2*sigma4 = 8/9*sigma4
            // with sigma4 = P_d2d2(k_min)/2 (pt_cfastpt.c). Subtracting
            // exactly these limits renormalizes the operators so every
            // one-loop correlator vanishes at large scales instead of
            // adding a constant, shot-noise-like power.
            KB[0][zl][i][p] = in ?
              interpol1d(FPTbias.tab[0], N, lb[0], lb[1], lb[2], lnk) : 0.0;
            KB[1][zl][i][p] = in ?
              interpol1d(FPTbias.tab[1], N, lb[0], lb[1], lb[2], lnk) - 2.*s4 : 0.0;
            KB[2][zl][i][p] = in ?
              interpol1d(FPTbias.tab[2], N, lb[0], lb[1], lb[2], lnk) : 0.0;
            KB[3][zl][i][p] = in ?
              interpol1d(FPTbias.tab[3], N, lb[0], lb[1], lb[2], lnk) - 4./3.*s4 : 0.0;
            KB[4][zl][i][p] = in ?
              interpol1d(FPTbias.tab[4], N, lb[0], lb[1], lb[2], lnk) - 8./9.*s4 : 0.0;
            KB[5][zl][i][p] = in ?
              interpol1d(FPTbias.tab[5], N, lb[0], lb[1], lb[2], lnk) : 0.0;
          }
        }
      }
    }
  }

  // -----------------------------------------------------------------------
  // Main integration loop. restrict pointers hoisted for contiguous loads.
  // -----------------------------------------------------------------------
  #pragma omp parallel for collapse(2) schedule(static)
  for (int zl=0; zl<nbin; zl++) {
    for (int i=0; i<nell; i++) {
      const cosmo_nodes* cn = &cn_all[zl];
      const int npts = cn->npts; // the nodes of bin zl
      const double ell = lx[i] + 0.5;
      const double ep  = ell_prefactor[i];

      const double* restrict fK     = cn->data[CN_FK];
      const double* restrict dchida = cn->data[CN_DCHIDA];
      const double* restrict wt     = cn->data[CN_WT];
      const double* restrict WGAL   = WB[0][zl];
      const double* restrict WMAG   = WB[1][zl];
      const double* restrict b1     = WB[2][zl];
      const double* restrict bmag   = WB[3][zl];
      const double* restrict WRSD   = KG[1][zl][i];
      const double* restrict mask   = KG[2][zl][i];

      // o = 0: the full model into table (PK = P_delta);
      // o = 1: the linear term into table_lin (PK = the separable P_lin;
      //        no HOD and no one-loop bias, so always the plain sum)
      for (int o=0; o<2; o++) {
        double** out = (0 == o) ? table : table_lin;
        if (NULL == out) {
          continue;
        }
        const double* restrict PK = (0 == o) ? KG[0][zl][i] : KG[3][zl][i];

        double sum = 0.0;
        if (0 == o && 1 == hod) {
          /* PHYSICAL DERIVATION & LOGIC FLOW
             1. galaxy density weight  W_d = W_gal    (no bias factor)
             2. magnification weight   W_m = W_mag ell_prefactor b_mag
             3. Limber sum over the nodes p:
                C_l^gg += [W_d^2 p_gg + 2 W_d W_m p_gm + W_m^2 P_delta]
                          (dchi/da) w_p / f_K^2                          */
          const double* restrict PGG = KH[0][zl][i];
          const double* restrict PGM = KH[1][zl][i];
          #pragma omp simd reduction(+:sum)
          for (int p=0; p<npts; p++) {
            const double Wd = WGAL[p];
            const double Wm = WMAG[p]*ep*bmag[p];
            const double P2 = Wd*Wd*PGG[p] + 2.0*Wd*Wm*PGM[p] + Wm*Wm*PK[p];
            sum += (P2*dchida[p]/(fK[p]*fK[p]))*wt[p];
          }
        }
        else if (0 == o && 1 == nonlinear_bias) {
          const double* restrict growfac = cn->data[CN_GROWFAC];
          const double* restrict b2   = WO[0][zl];
          const double* restrict bs2  = WO[1][zl];
          const double* restrict b3   = WO[2][zl];
          const double* restrict bk   = WO[3][zl];
          const double* restrict d1d2 = KB[0][zl][i];
          const double* restrict d2d2 = KB[1][zl][i];
          const double* restrict d1s2 = KB[2][zl][i];
          const double* restrict d2s2 = KB[3][zl][i];
          const double* restrict s2s2 = KB[4][zl][i];
          const double* restrict d1p3 = KB[5][zl][i];
          // One-loop coefficients from squaring
          //   delta_g = b1 d + (b2/2) d^2 + (bs2/2) s^2 + (b3/2) psi3:
          // operator autos get (1/2)^2 = 1/4, the b2-bs2 cross gets
          // 2*(1/2)*(1/2) = 1/2, operator-b1 crosses get 2*(1/2) = 1.
          #pragma omp simd reduction(+:sum)
          for (int p=0; p<npts; p++) {
            const double W = WGAL[p]*b1[p] + WMAG[p]*ep*bmag[p] + WRSD[p];
            const double k = ell/fK[p];
            const double g4 = growfac[p]*growfac[p]*growfac[p]*growfac[p];
            const double oneloop = (WGAL[p]*WGAL[p])*
              (g4*(b1[p]*b2[p]*d1d2[p] + 0.25*b2[p]*b2[p]*d2d2[p] +
                   b1[p]*bs2[p]*d1s2[p] + 0.5*b2[p]*bs2[p]*d2s2[p] +
                   0.25*bs2[p]*bs2[p]*s2s2[p] + b1[p]*b3[p]*d1p3[p]) +
               (2*b1[p]*bk[p]*k*k*PK[p]));
            sum += mask[p]*((W*W*PK[p] + oneloop)*dchida[p]/(fK[p]*fK[p]))*wt[p];
          }
        }
        else {
          #pragma omp simd reduction(+:sum)
          for (int p=0; p<npts; p++) {
            const double W = WGAL[p]*b1[p] + WMAG[p]*ep*bmag[p] + WRSD[p];
            sum += mask[p]*((W*W*PK[p])*dchida[p]/(fK[p]*fK[p]))*wt[p];
          }
        }
        out[zl][i] = sum;
      }
    }
  }
  free(WB); free(KG); free(KPN);
  if (WO != NULL) free(WO);
  if (KB != NULL) free(KB);
  if (KH != NULL) {
    free(KH);
  }
}

// ---------------------------------------------------------------------------
// Batch galaxy-clustering Limber C_l^gg (auto spectra) at arbitrary
// multipole values: the entry point of C_gg_tomo_limber_work.
//
// Builds the Gauss-Legendre nodes of each lens bin on [amin_lens,
// amax_lens] (128 nodes at the default accuracy; 256, 512, 1024 for
// Ntable.high_def_integration = 1, 2, 3+; a bin whose range magnification
// widens gets one such rule on its n(z) support and one on the
// foreground, see create_cosmo_nodes_lens) and the magnification prefactor
// l (l+1)/(l + 1/2)^2, then calls C_gg_tomo_limber_work.
//
// Two outputs, either may be NULL:
//   out     - the full model (P_delta, one-loop galaxy bias);
//   out_lin - the linear term that the non-Limber C_cl_tomo subtracts,
//             (D(a)/D(a_piv))^2 * P_lin(k, a_piv) per lens bin, b1 only.
// With both, one pass shares the nodes, the radial weights and the RSD
// kernel between them (see C_gg_tomo_limber_work), bitwise two separate
// calls.
//
// Example: C_cl_tomo calls it once with ells = 0, 1, ..., 149 to get both
// Limber terms of the non-Limber split for every lens bin at once.
//
// Cache invalidation:
// the static Gauss-Legendre table w (128/256/512/1024
// nodes, keyed on abs(Ntable.high_def_integration)) rebuilds when
// Ntable.random changes; the per-lens-bin cosmo_nodes are rebuilt on
// every call (they depend on the current cosmology).
//
// Parameters:
//   ells  - multipole values, length nell (need not be integers)
//   nell  - number of multipole values
//   NSIZE - number of gg power spectra; must equal redshift.clustering_nbin
//           (auto spectra only)
//   out     - output [NSIZE][nell], the full model, indexed out[nz][i]
//             (NULL: not computed)
//   out_lin - output [NSIZE][nell], the linear term (NULL: not computed)
//
// Returns:
//   nothing; the results are written into out and out_lin
// ---------------------------------------------------------------------------
void C_gg_tomo_limber_nl_lin_nointerp_ells(
    const double* ells,      // array of multipole values (length nell)
    const int nell,          // number of multipole values
    const int NSIZE,         // number of gg power spectra
    double** out,            // output [NSIZE][nell], full model (or NULL)
    double** out_lin         // output [NSIZE][nell], linear term (or NULL)
  )
{
  static gsl_integration_glfixed_table* w = NULL;
  static uint64_t cache[MAX_SIZE_ARRAYS];
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 128 :
                         (1 == hdi) ? 256 :
                         (2 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL) gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }
  if (NSIZE != redshift.clustering_nbin) {
    log_fatal("NSIZE = %d != clustering_nbin = %d (auto spectra only)",
              NSIZE, redshift.clustering_nbin);
    exit(1);
  }
  if (nell <= 0) {
    log_fatal("nell = %d <= 0", nell); exit(1);
  }
  if (NULL == out && NULL == out_lin) {
    log_fatal("no output requested"); exit(1);
  }

  cosmo_nodes cn_all[redshift.clustering_nbin];
  for (int zl=0; zl<redshift.clustering_nbin; zl++) {
    const double amin = amin_lens(zl);
    const double amax = amax_lens(zl);
    if (!(amin>0) || !(amin<1) || !(amax>0) || !(amax<1)) {
      log_fatal("0 < amin/amax < 1 not true"); exit(1);
    }
    if (!(amin < amax)) {
      log_fatal("amin < amax not true"); exit(1);
    }
    cn_all[zl] = create_cosmo_nodes_lens(zl, w);
  }

  double* ep = (double*) malloc1d(nell);
  for (int i=0; i<nell; i++) {
    ep[i] = ells[i]*(ells[i] + 1.)/((ells[i] + 0.5)*(ells[i] + 0.5));
  }

  C_gg_tomo_limber_work(cn_all, ells, ep, nell, out, out_lin);

  free(ep);
  for (int zl=0; zl<redshift.clustering_nbin; zl++) {
    free_cosmo_nodes(&cn_all[zl]);
  }
}

// ---------------------------------------------------------------------------
// One of the two outputs of C_gg_tomo_limber_nl_lin_nointerp_ells:
// use_linear_ps = 0 the full model, 1 the linear term of the non-Limber
// split.
//
// Parameters:
//   ells  - multipole values, length nell (need not be integers)
//   nell  - number of multipole values
//   NSIZE - number of gg power spectra (= redshift.clustering_nbin)
//   use_linear_ps - 1 = linear term of the non-Limber split, 0 = full model
//   out   - output [NSIZE][nell], indexed out[nz][i]
//
// Returns:
//   nothing; the result is written into out
// ---------------------------------------------------------------------------
void C_gg_tomo_limber_linpsopt_nointerp_ells(
    const double* ells,      // array of multipole values (length nell)
    const int nell,          // number of multipole values
    const int NSIZE,         // number of gg power spectra
    const int use_linear_ps, // 1 = P_lin, no one-loop bias; 0 = full model
    double** out             // output [NSIZE][nell]
  )
{
  if (0 == use_linear_ps) {
    C_gg_tomo_limber_nl_lin_nointerp_ells(ells, nell, NSIZE, out, NULL);
  }
  else {
    C_gg_tomo_limber_nl_lin_nointerp_ells(ells, nell, NSIZE, NULL, out);
  }
}

// ---------------------------------------------------------------------------
// Batch galaxy-clustering Limber C_l^gg (full model) at arbitrary multipole
// values: C_gg_tomo_limber_linpsopt_nointerp_ells with use_linear_ps = 0.
// Used by the interpolation table of C_gg_tomo_limber, the Fourier-space
// data vectors, the notebook wrapper C_gg_tomo_limber_cpp, and
// C_gg_tomo_limber_nointerp_batch.
//
// Parameters:
//   ells  - multipole values, length nell (need not be integers)
//   nell  - number of multipole values
//   NSIZE - number of gg power spectra (= redshift.clustering_nbin)
//   out   - output [NSIZE][nell], indexed out[nz][i]
//
// Returns:
//   nothing; the result is written into out
// ---------------------------------------------------------------------------
void C_gg_tomo_limber_nointerp_ells(
    const double* ells,  // array of multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of gg power spectra
    double** out         // output [NSIZE][nell]
  )
{
  C_gg_tomo_limber_linpsopt_nointerp_ells(ells, nell, NSIZE, 0, out);
}

// ---------------------------------------------------------------------------
// Batch galaxy-clustering Limber C_l^gg at the integer multipoles
// l = lmin, ..., lmax-1, written at their own index: Cl[nz][l].
// Thin wrapper around C_gg_tomo_limber_nointerp_ells.
//
// Example: w_gg_tomo (like.adopt_limber[LIMBER_GG] = 1) calls it with lmin = 1 and
// lmax = limits.LMIN_tab = 20 for the multipoles below the interpolation
// table; Cl[nz][0] is left untouched.
//
// Parameters:
//   lmin  - first multipole (inclusive)
//   lmax  - last multipole (exclusive)
//   NSIZE - number of gg power spectra (= redshift.clustering_nbin)
//   Cl    - output [NSIZE][>= lmax], indexed Cl[nz][l]
//
// Returns:
//   nothing; the result is written into Cl
// ---------------------------------------------------------------------------
void C_gg_tomo_limber_nointerp_batch(
    const int lmin,   // first multipole (inclusive)
    const int lmax,   // last multipole (exclusive)
    const int NSIZE,  // number of gg power spectra (= clustering_nbin)
    double** Cl       // output [NSIZE][>=lmax], indexed as Cl[nz][l]
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

  C_gg_tomo_limber_nointerp_ells(lx, nell, NSIZE, tmp);

  for (int k=0; k<NSIZE; k++) {
    for (int i=0; i<nell; i++) {
      Cl[k][lmin+i] = tmp[k][i];
    }
  }
  free(tmp); free(lx);
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
// Galaxy clustering angular power spectrum C_l^gg with interpolation
// (auto spectra only: ni must equal nj).
//
// Builds the (lens bin, log ell) table with one
// C_gg_tomo_limber_nointerp_ells call (log-spaced grid, Ntable.N_ell[NODES_DENSE]
// points covering l = LMIN_tab..LMAX), then caches it for subsequent
// lookups. Returns the interpolated value at the requested l via
// interpol1d.
//
// Why the table is shared through the gg_ static struct: the
// real-space projection (w_gg_tomo) needs C_l at every integer
// multipole up to Ntable.LMAX ~ 1e5, for every tomographic pair,
// inside its Legendre/Hankel sums - millions of table reads per
// likelihood evaluation. Only the vectorized batch reader
// (C_gg_tomo_limber_fill, which runs the interpol1d linear read four
// multipoles at a time through AVX2 gathers) sustains that rate;
// calling this function one multipole at a time would dominate the
// whole evaluation.
//
// The struct is how the table travels between the two functions.
// The builder (this function) and the reader (the _fill) never call
// each other - the real-space projection calls one, the C_ell paths
// call the other - so no argument list connects them. Instead the
// builder publishes the table pointer and the grid geometry (the
// ln(ell) limits, spacing and node count) in the file-scope struct,
// and the reader picks them up there.
//
// The table keeps the exact per-node quadrature at every one of its
// N_ell nodes: do NOT apply the internal coarse-grid upsampling of the
// ss/gs tables here (Ntable.N_ell[NODES_COARSE]) - the clustering auto
// spectra carry BAO wiggles in exactly the ell range the spline would
// smooth over.
//
// Cache invalidation:
// the static table and grid limits rebuild when the
// table is NULL or Ntable.random changes; the values refill when any of
// cosmology.random, nuisance.random_photoz_clustering,
// redshift.random_clustering, Ntable.random, or
// nuisance.random_galaxy_bias change.
//
// Parameters:
//   l  - multipole moment (continuous; outside the grid the lookup warns
//        and extrapolates)
//   ni - first lens redshift bin
//   nj - second lens redshift bin (must equal ni)
//
// Returns:
//   C_l^gg of lens bin ni (auto spectrum)
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
    nell   = Ntable.N_ell[NODES_DENSE];
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
      fdiff2(cache[4], nuisance.random_galaxy_bias) ||
      fdiff2(cache[5], (uint64_t) include_HOD_GX) ||
      // redshift.random_shear: with magnification bias on, amax_lens
      // (redshift_spline.c) ends the lens range at the source z_min,
      // so a new source n(z) alone moves these integrals
      fdiff2(cache[6], redshift.random_shear))
  {
    double* lx = (double*) malloc1d(nell);
    for (int i=0; i<nell; i++) {
      lx[i] = exp(lim[0] + i*lim[2]);
    }
    C_gg_tomo_limber_nointerp_ells(lx, nell, NSIZE, table);
    free(lx);
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_clustering;
    cache[2] = redshift.random_clustering;
    cache[3] = Ntable.random;
    cache[4] = nuisance.random_galaxy_bias;
    cache[5] = (uint64_t) include_HOD_GX;
    cache[6] = redshift.random_shear;
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
// transform C_l -> w(theta). Uses limber_fill_interp which processes 4 ells
// per iteration via AVX2 gather instructions (i32gather_pd).
//
// Requires C_gg_tomo_limber to have been called first to populate gg_.tab.
//
// Parameters:
//   nz     - lens bin index (0..clustering_nbin-1)
//   lmin   - first multipole to fill (inclusive)
//   lmax   - last multipole to fill (exclusive)
//   ln_ell - precomputed log(l) array, indexed by l
//   out    - output C_l array, indexed by l
//
// Returns:
//   nothing; the interpolated C_l are written into out at indices
//   lmin..lmax-1
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
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GK = GALAXY x CMB LENSING
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Core workhorse for all galaxy x CMB-lensing C_l computations (the interp
// table in C_gk_tomo_limber and the low-ell batch in
// C_gk_tomo_limber_nointerp_batch).
//
// Same design as C_gg_tomo_limber_work with one galaxy leg replaced by the
// CMB convergence kernel: precompute all expensive quantities on a fixed
// grid of quadrature points, then evaluate the Limber integral for every
// (ell, lens bin) combination with SIMD-vectorized inner loops.
//
// Computes, per lens bin and multipole l, with ep = l(l+1)/(l+0.5)^2 and
// k = (l + 1/2)/fK(a):
//
//   C_l^gk = ep * int da (dchida/fK^2) W_k *
//            [ (W_gal b1 + W_mag ep bmag + W_RSD) P_delta
//              + W_gal (D^4 (b2/2 P_d1d2 + bs2/2 P_d1s2 + b3/2 P_d1p3)
//                       + bK k^2 P_delta) ]
//
// The one-loop bias terms (second line) enter only when has_b2_galaxies()
// and read the FPTbias tables (zero outside their k range, as in the gg
// batch). RSD is gated on the file-scope include_RSD_GK with the same
// reach mask as the gg batch. HOD is not implemented in the batched path
// (log_fatal), as in gg.
//
// Memory layout (npts = npts_max, the largest node count over the lens
// bins, as in the gg batch: each bin's sum runs over its own nodes):
//   WB[5][clustering_nbin][npts]        W_gal, W_mag, b1, bmag, W_k
//   WO[4][clustering_nbin][npts]        b2, bs2, b3, bK    (one-loop only)
//   KG[3][clustering_nbin][nell][npts]  PK, W_RSD, mask
//   KB[3][clustering_nbin][nell][npts]  P_d1d2, P_d1s2, P_d1p3
//                                       (one-loop only; zero outside the
//                                       FPTbias k range)
//
// Cache invalidation:
// none here - every input arrives precomputed; the
// warm-up prelude initializes the lazily-built statics of the kernel and
// power-spectrum functions single-threaded before the parallel regions.
//
// Parameters:
//   cn_all        - quadrature nodes per lens bin [clustering_nbin]
//   lx            - multipole values (length nell)
//   ell_prefactor - l*(l+1)/(l+0.5)^2 per multipole (length nell)
//   nell          - number of multipole values
//   table         - output [clustering_nbin][nell]
//
// Returns:
//   nothing; the result is written into table
// ---------------------------------------------------------------------------
static void C_gk_tomo_limber_work(
    const cosmo_nodes* cn_all,    // quadrature nodes per lens bin
    const double* lx,             // multipole values (length nell)
    const double* ell_prefactor,  // l*(l+1)/(l+0.5)^2 per ell
    const int nell,               // number of multipole values
    double** table                // output [clustering_nbin][nell]
  )
{
  if (1 == include_HOD_GX) {
    log_fatal("HOD not implemented in the batched gk path"); exit(1);
  }
  const int nbin = redshift.clustering_nbin;
  const int nonlinear_bias = has_b2_galaxies();

  // per-node arrays are padded to the largest node count over the bins
  int npts_max = 0;
  for (int zl=0; zl<nbin; zl++) {
    if (cn_all[zl].npts > npts_max) {
      npts_max = cn_all[zl].npts;
    }
  }
  // -----------------------------------------------------------------------
  // Warm up all functions that lazily initialize internal static tables.
  // Must be called single-threaded before any parallel region touches them.
  // -----------------------------------------------------------------------
  {
    const cosmo_nodes* cn = &cn_all[0];
    const double a    = cn->data[CN_A][0];
    const double fK   = cn->data[CN_FK][0];
    const double hoh0 = cn->data[CN_HOVERH0][0];
    const double ell  = lx[0] + 0.5;
    (void) W_gal(a, 0, hoh0);
    (void) W_mag(a, fK, 0);
    (void) W_k(a, fK);
    (void) Pdelta(ell/fK, a);
    (void) gb1(0.1, 0);
    (void) gbmag(0.1, 0);
    if (1 == include_RSD_GK) {
      (void) chi(limits.a_min);
      (void) a_chi(0.9);
      (void) W_RSD(100, 0.9, 0.95, 0);
    }
    if (1 == nonlinear_bias) {
      (void) gb2(0.1, 0);
      (void) gbs2(0.1, 0);
      (void) gb3(0.1, 0);
      (void) gbK(0.1, 0);
      if (0 == nuisance.IA_code) {
        get_FPT_bias();
      }
    }
  }
  const double chi_a_min = (1 == include_RSD_GK) ? chi(limits.a_min) : 0.0;
  double limbias[3] = {0.0, 0.0, 0.0};
  if (1 == nonlinear_bias) {
    limbias[0] = log(FPTbias.krange[RANGE_MIN]);
    limbias[1] = log(FPTbias.krange[RANGE_MAX]);
    limbias[2] = (limbias[1] - limbias[0])/FPTbias.N;
  }

  // -----------------------------------------------------------------------
  // Allocate precomputed arrays (padded to npts_max)
  // -----------------------------------------------------------------------
  double*** WB  = (double***) malloc3d(5, nbin, npts_max);
  double**** KG = (double****) malloc4d(3, nbin, nell, npts_max);
  double*** WO  = NULL;
  double**** KB = NULL;
  if (1 == nonlinear_bias) {
    WO = (double***) malloc3d(4, nbin, npts_max);
    KB = (double****) malloc4d(3, nbin, nell, npts_max);
  }

  // per-thread scratch of the batched P reads (Pdelta_at_a: one call per
  // node, the z half of the table read once per node instead of once per
  // multipole): KPN[2t] = the node's Limber wavenumbers, KPN[2t+1] = P
  double** KPN = (double**) malloc2d(2*omp_get_max_threads(), nell);
  #pragma omp parallel
  {
    // ---------------------------------------------------------------------
    // Precompute: lens weights, galaxy biases, CMB convergence kernel
    // ---------------------------------------------------------------------
    #pragma omp for collapse(2) schedule(static) nowait
    for (int zl=0; zl<nbin; zl++) {
      for (int p=0; p<npts_max; p++) {
        const cosmo_nodes* cn = &cn_all[zl];
        if (p >= cn->npts) {
          continue; // padding node: bin zl has fewer nodes
        }
        const double a  = cn->data[CN_A][p];
        const double fK = cn->data[CN_FK][p];
        const double z  = 1.0/a - 1.0;
        WB[0][zl][p] = W_gal(a, zl, cn->data[CN_HOVERH0][p]);
        WB[1][zl][p] = W_mag(a, fK, zl);
        WB[2][zl][p] = gb1(z, zl);
        WB[3][zl][p] = gbmag(z, zl);
        WB[4][zl][p] = W_k(a, fK);
        if (1 == nonlinear_bias) {
          WO[0][zl][p] = gb2(z, zl);
          WO[1][zl][p] = gbs2(z, zl);
          WO[2][zl][p] = gb3(z, zl);
          WO[3][zl][p] = gbK(z, zl);
        }
      }
    }
    // ---------------------------------------------------------------------
    // Precompute: P_delta of every node at every multipole, one batched
    // read per node (see KPN). Its own loop over (bin, node), since the
    // fill below runs over (bin, ell, node); nowait, because the two write
    // disjoint slots.
    // ---------------------------------------------------------------------
    #pragma omp for collapse(2) schedule(static) nowait
    for (int zl=0; zl<nbin; zl++) {
      for (int p=0; p<npts_max; p++) {
        const cosmo_nodes* cn = &cn_all[zl];
        if (p >= cn->npts) {
          continue; // padding node: bin zl has fewer nodes
        }
        double* restrict kn = KPN[2*omp_get_thread_num()];
        double* restrict pn = KPN[2*omp_get_thread_num() + 1];
        for (int i=0; i<nell; i++) {
          kn[i] = (lx[i] + 0.5)/cn->data[CN_FK][p];
        }
        Pdelta_at_a(cn->data[CN_A][p], kn, nell, pn);
        for (int i=0; i<nell; i++) {
          KG[0][zl][i][p] = pn[i];
        }
      }
    }
    // ---------------------------------------------------------------------
    // Precompute: RSD kernel and its support, one-loop kernels
    // ---------------------------------------------------------------------
    #pragma omp for collapse(3) schedule(static)
    for (int zl=0; zl<nbin; zl++) {
      for (int i=0; i<nell; i++) {
        for (int p=0; p<npts_max; p++) {
          const cosmo_nodes* cn = &cn_all[zl];
          if (p >= cn->npts) {
            continue; // padding node: bin zl has fewer nodes
          }
          const double fK  = cn->data[CN_FK][p];
          const double ell = lx[i] + 0.5;
          const double k   = ell/fK;
          KG[1][zl][i][p] = 0.0;
          KG[2][zl][i][p] = 1.0;
          if (1 == include_RSD_GK) {
            const double chi_0 = ell/k;
            const double chi_1 = (ell + 1.0)/k;
            if (chi_1 > chi_a_min) {
              KG[2][zl][i][p] = 0.0;
            }
            else {
              const double a_0 = a_chi(chi_0);
              const double a_1 = a_chi(chi_1);
              KG[1][zl][i][p] = W_RSD(ell, a_0, a_1, zl);
            }
          }
          if (1 == nonlinear_bias) {
            const double lnk = log(k);
            const int in = (lnk >= limbias[0] && lnk <= limbias[1]);
            const double* lb = limbias;
            const int N = FPTbias.N;
            KB[0][zl][i][p] = in ?
              interpol1d(FPTbias.tab[0], N, lb[0], lb[1], lb[2], lnk) : 0.0;
            KB[1][zl][i][p] = in ?
              interpol1d(FPTbias.tab[2], N, lb[0], lb[1], lb[2], lnk) : 0.0;
            KB[2][zl][i][p] = in ?
              interpol1d(FPTbias.tab[5], N, lb[0], lb[1], lb[2], lnk) : 0.0;
          }
        }
      }
    }
  }

  // -----------------------------------------------------------------------
  // Main integration loop. restrict pointers hoisted for contiguous loads.
  // -----------------------------------------------------------------------
  #pragma omp parallel for collapse(2) schedule(static)
  for (int zl=0; zl<nbin; zl++) {
    for (int i=0; i<nell; i++) {
      const cosmo_nodes* cn = &cn_all[zl];
      const int npts = cn->npts; // the nodes of bin zl
      const double ell = lx[i] + 0.5;
      const double ep  = ell_prefactor[i];

      const double* restrict fK     = cn->data[CN_FK];
      const double* restrict dchida = cn->data[CN_DCHIDA];
      const double* restrict wt     = cn->data[CN_WT];
      const double* restrict WGAL   = WB[0][zl];
      const double* restrict WMAG   = WB[1][zl];
      const double* restrict b1     = WB[2][zl];
      const double* restrict bmag   = WB[3][zl];
      const double* restrict WKC    = WB[4][zl];
      const double* restrict PK     = KG[0][zl][i];
      const double* restrict WRSD   = KG[1][zl][i];
      const double* restrict mask   = KG[2][zl][i];

      double sum = 0.0;
      if (1 == nonlinear_bias) {
        const double* restrict growfac = cn->data[CN_GROWFAC];
        const double* restrict b2   = WO[0][zl];
        const double* restrict bs2  = WO[1][zl];
        const double* restrict b3   = WO[2][zl];
        const double* restrict bk   = WO[3][zl];
        const double* restrict d1d2 = KB[0][zl][i];
        const double* restrict d1s2 = KB[1][zl][i];
        const double* restrict d1p3 = KB[2][zl][i];
        #pragma omp simd reduction(+:sum)
        for (int p=0; p<npts; p++) {
          const double W = WGAL[p]*b1[p] + WMAG[p]*ep*bmag[p] + WRSD[p];
          const double k = ell/fK[p];
          const double g4 = growfac[p]*growfac[p]*growfac[p]*growfac[p];
          const double oneloop = WGAL[p]*
            (g4*(0.5*b2[p]*d1d2[p] + 0.5*bs2[p]*d1s2[p] + 0.5*b3[p]*d1p3[p]) +
             (bk[p]*k*k*PK[p]));
          sum += mask[p]*WKC[p]*((W*PK[p] + oneloop)*dchida[p]/(fK[p]*fK[p]))*wt[p];
        }
      }
      else {
        #pragma omp simd reduction(+:sum)
        for (int p=0; p<npts; p++) {
          const double W = WGAL[p]*b1[p] + WMAG[p]*ep*bmag[p] + WRSD[p];
          sum += mask[p]*WKC[p]*((W*PK[p])*dchida[p]/(fK[p]*fK[p]))*wt[p];
        }
      }
      table[zl][i] = sum*ep;
    }
  }
  free(WB); free(KG); free(KPN);
  if (WO != NULL) free(WO);
  if (KB != NULL) free(KB);
}

// ---------------------------------------------------------------------------
// Batch computation of galaxy x CMB-lensing C_l at arbitrary multipole
// values: the entry point of C_gk_tomo_limber_work.
//
// Builds the Gauss-Legendre nodes of each lens bin on [amin_lens,
// amax_lens] (64 nodes at the default accuracy; 128, 256, 512, 1024 for
// Ntable.high_def_integration = 1, 2, 3, 4+; a bin whose range
// magnification widens gets one such rule on its n(z) support and one on
// the foreground, see create_cosmo_nodes_lens) and the spin-0 prefactor
// l*(l+1)/(l + 1/2)^2 per multipole, then calls C_gk_tomo_limber_work.
//
// Cache invalidation:
// the static Gauss-Legendre table rebuilds when
// Ntable.random changes; the quadrature nodes are rebuilt on every call
// (they depend on the current cosmology through create_cosmo_nodes_lens).
//
// Parameters:
//   ells  - multipole values, length nell (need not be integers)
//   nell  - number of multipole values
//   NSIZE - number of lens tomographic bins; must equal
//           redshift.clustering_nbin (one spectrum per lens bin - the
//           CMB is a single source plane)
//   out   - output [NSIZE][nell], indexed as out[nz][i]
//
// Returns:
//   nothing; the result is written into out
// ---------------------------------------------------------------------------
void C_gk_tomo_limber_nointerp_ells(
    const double* ells,  // array of multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of lens tomographic bins
    double** out         // output [NSIZE][nell], indexed as out[nz][i]
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

  if (NSIZE != redshift.clustering_nbin) {
    log_fatal("NSIZE = %d != clustering_nbin = %d", NSIZE,
              redshift.clustering_nbin);
    exit(1);
  }
  if (nell <= 0) {
    log_fatal("nell = %d <= 0", nell); exit(1);
  }

  cosmo_nodes cn_all[redshift.clustering_nbin];
  for (int b = 0; b < redshift.clustering_nbin; b++) {
    const double amin = amin_lens(b);
    const double amax = amax_lens(b);
    if (!(amin>0) || !(amin<1) || !(amax>0) || !(amax<1)) {
      log_fatal("0 < amin/amax < 1 not true"); exit(1);
    }
    cn_all[b] = create_cosmo_nodes_lens(b, w);
  }

  double* epf = (double*) malloc1d(nell);
  for (int i=0; i<nell; i++) {
    const double l = ells[i];
    const double ell = l + 0.5;
    epf[i] = l*(l + 1.0)/(ell*ell);
  }

  C_gk_tomo_limber_work(cn_all, ells, epf, nell, out);

  free(epf);
  for (int b = 0; b < redshift.clustering_nbin; b++) {
    free_cosmo_nodes(&cn_all[b]);
  }
}

// ---------------------------------------------------------------------------
// Batch computation of galaxy x CMB-lensing C_l at the integer multipoles
// lmin..lmax-1: a thin wrapper around C_gk_tomo_limber_nointerp_ells.
//
// Builds the integer multipole list, runs one batch call, and scatters the
// results into Cl at their own multipole indices (Cl[nz][l], not packed
// from zero) - the layout the low-ell loop of w_gk_tomo consumes.
//
// Parameters:
//   lmin  - first multipole (inclusive)
//   lmax  - last multipole (exclusive)
//   NSIZE - number of lens tomographic bins (= redshift.clustering_nbin)
//   Cl    - output [NSIZE][>=lmax], written at indices lmin..lmax-1
//
// Returns:
//   nothing; the result is written into Cl
// ---------------------------------------------------------------------------
void C_gk_tomo_limber_nointerp_batch(
    const int lmin,   // first multipole (inclusive)
    const int lmax,   // last multipole (exclusive)
    const int NSIZE,  // number of lens tomographic bins (= clustering_nbin)
    double** Cl       // output [NSIZE][>=lmax], indexed as Cl[nz][l]
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

  C_gk_tomo_limber_nointerp_ells(lx, nell, NSIZE, tmp);

  for (int k = 0; k < NSIZE; k++) {
    for (int i = 0; i < nell; i++) {
      Cl[k][lmin+i] = tmp[k][i];
    }
  }

  free(tmp); free(lx);
}

// ---------------------------------------------------------------------------
// Single-ell galaxy x CMB-lensing C_l: a point diagnostic on the batch
// engine.
//
// Runs one C_gk_tomo_limber_nointerp_ells call at a single multipole and
// reads one entry, so it pays the WHOLE-TOMOGRAPHY batch cost per call
// (every lens bin is computed even though one number is returned). Never
// loop this over (l, ni): call C_gk_tomo_limber_nointerp_ells once and
// index the result instead.
//
// Kept in the API as the exact per-multipole entry point of the probe (the
// C_ss_tomo_limber_nointerp pattern): notebooks evaluate single points
// here, and a per-integer-multipole Limber value is what a non-Limber
// pipeline consumes.
//
// Parameters:
//   l    - multipole moment
//   ni   - lens redshift bin index (0..redshift.clustering_nbin-1)
//
// Returns:
//   C_l^gk of lens bin ni with the full Limber model (nonlinear P_delta,
//   one-loop bias when enabled)
// ---------------------------------------------------------------------------
double C_gk_tomo_limber_nointerp(
    const double l,
    const int ni
  ) // slow (whole-tomography batch per call) - use the batch version
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni); exit(1);
  }
  const int NSIZE = redshift.clustering_nbin;
  double** tmp = (double**) malloc2d(NSIZE, 1);
  const double ell = l;

  C_gk_tomo_limber_nointerp_ells(&ell, 1, NSIZE, tmp);

  const double res = tmp[ni][0];
  free(tmp);
  return res;
}

// ---------------------------------------------------------------------------
// Shared state between C_gk_tomo_limber (which builds the interpolation table)
// and C_gk_tomo_limber_fill (which reads it to fill Cl arrays at ~100k ell
// values for real-space correlation functions).
//
//   tab     - pointer to the cached table[clustering_nbin][nell]
//             (owned by C_gk_tomo_limber's static)
//   lim[0]  - log(l_min) of the interpolation grid
//   lim[1]  - log(l_max) of the interpolation grid
//   lim[2]  - uniform spacing in log(l): (lim[1] - lim[0]) / (nell - 1)
//   nell    - number of grid points in the interpolation table
// ---------------------------------------------------------------------------
static struct { double** tab; double lim[3]; int nell; } gk_ = {0};

// ---------------------------------------------------------------------------
// Galaxy x CMB-lensing angular power spectrum C_l^gk with interpolation.
//
// Builds the (lens bin, log ell) table with one
// C_gk_tomo_limber_nointerp_ells call (log-spaced grid, Ntable.N_ell[NODES_DENSE]
// points covering l = LMIN_tab..LMAX), then caches it for subsequent
// lookups. Returns the interpolated value at the requested l via
// interpol1d.
//
// Why the table is shared through the gk_ static struct: the
// real-space projection (w_gk_tomo) needs C_l at every integer
// multipole up to Ntable.LMAX ~ 1e5, for every tomographic pair,
// inside its Legendre/Hankel sums - millions of table reads per
// likelihood evaluation. Only the vectorized batch reader
// (C_gk_tomo_limber_fill, which runs the interpol1d linear read four
// multipoles at a time through AVX2 gathers) sustains that rate;
// calling this function one multipole at a time would dominate the
// whole evaluation.
//
// The struct is how the table travels between the two functions.
// The builder (this function) and the reader (the _fill) never call
// each other - the real-space projection calls one, the C_ell paths
// call the other - so no argument list connects them. Instead the
// builder publishes the table pointer and the grid geometry (the
// ln(ell) limits, spacing and node count) in the file-scope struct,
// and the reader picks them up there.
//
// Stored values carry no CMB beam or pixel window; w_gk_tomo
// multiplies its own copy by the beam_cmb/w_pixel filter.
//
// When Ntable.N_ell[NODES_COARSE] is active, the exact quadrature instead
// runs on the internal coarse grid and the house cubic spline
// upsamples onto the unchanged N_ell nodes (the strategy block inside
// explains why this wins).
//
// Cache invalidation:
// the static table and grid limits rebuild when the
// table is NULL or Ntable.random changes; the values refill when any of
// cosmology.random, nuisance.random_photoz_clustering,
// redshift.random_clustering, Ntable.random, or
// nuisance.random_galaxy_bias change.
//
// Parameters:
//   l  - multipole moment (continuous; outside the grid the lookup warns
//        and extrapolates)
//   ni - lens redshift bin (the CMB is a single source plane, so there
//        is no second bin index)
//
// Returns:
//   C_l^gk of lens bin ni
// ---------------------------------------------------------------------------
double C_gk_tomo_limber(const double l, const int ni)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static int nell;
  static double lim[3];
  static double* lx = NULL;
  static int ncoarse = 0;  // active internal coarse grid size (0 = off)
  static double dlnc = 0.; // coarse grid spacing in ln(ell)
  static double* lxc = NULL;   // coarse ell nodes
  static int* qidx = NULL;     // fine node -> coarse interval (uniform
  static double* qdel = NULL;  //   grids: precomputed, no search)
  static double** tabc = NULL; // coarse C_ell values
  static double** cspl = NULL; // natural-cubic-spline c coefficients

  if (NULL == table || fdiff2(cache[3], Ntable.random)) {
    nell = Ntable.N_ell[NODES_DENSE];
    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_ell[NODES_DENSE] - 1.0);
    if (table != NULL) free(table);
    table = (double**) malloc2d(redshift.clustering_nbin, Ntable.N_ell[NODES_DENSE]);

    gk_.tab    = table;
    gk_.lim[0] = lim[0];
    gk_.lim[1] = lim[1];
    gk_.lim[2] = lim[2];
    gk_.nell   = nell;

    if (lx != NULL) free(lx);
    lx = (double*) malloc1d(nell);
    for (int i=0; i<nell; i++) {
      lx[i] = exp(lim[0] + i*lim[2]);
    }

    // Coarse-grid workspace (the strategy is explained where the grid
    // is used, in the refill block below): every allocation lives
    // HERE, in the Ntable rebuild block; the per-cosmology refill only
    // fills. The pieces are:
    //   lxc        - the ncoarse ell nodes, log-spaced over the same
    //                [lim[0], lim[1]] range as the fine table
    //   tabc, cspl - the coarse C_ell values and their cubic-spline
    //                coefficients, one row per lens (clustering) bin
    //   qidx, qdel - for each fine node, the coarse interval it falls
    //                in and its ln(ell) offset from that interval's
    //                left node: both grids are uniform in ln(ell) with
    //                shared endpoints, so this is pure grid geometry,
    //                computed once - no search of any kind at refill
    if (lxc  != NULL) { free(lxc);  lxc  = NULL; }
    if (qidx != NULL) { free(qidx); qidx = NULL; }
    if (qdel != NULL) { free(qdel); qdel = NULL; }
    if (tabc != NULL) { free(tabc); tabc = NULL; }
    if (cspl != NULL) { free(cspl); cspl = NULL; }
    const int nc = Ntable.N_ell[NODES_COARSE];
    ncoarse = (nc > 3 && nc < nell) ? nc : 0;
    if (ncoarse > 0) {
      dlnc = (lim[1] - lim[0]) / ((double) ncoarse - 1.0);
      lxc = (double*) malloc1d(ncoarse);
      for (int i=0; i<ncoarse; i++) {
        lxc[i] = exp(lim[0] + i*dlnc);
      }
      qidx = (int*) malloc(sizeof(int) * nell);
      qdel = (double*) malloc1d(nell);
      for (int i=0; i<nell; i++) {
        // Where does fine node i sit on the coarse grid? Both grids
        // run over the same [lim[0], lim[1]] in ln(ell), so the map
        // is pure arithmetic:
        //
        //   fine node i -> ln(ell) = lim[0] + i*lim[2]
        //               -> r = i*lim[2]/dlnc   (coarse spacings in)
        //               -> j = (int) r         (interval's left node)
        //               -> qdel = (r - j)*dlnc (offset inside it)
        //
        // The spline evaluates on interval [j, j+1], so the largest
        // legal j is ncoarse-2, the left node of the LAST interval.
        //
        // Why the clamp: at the shared top endpoint, i*lim[2] and
        // (ncoarse-1)*dlnc are two floating-point roundings of the
        // same length lim[1] - lim[0]. r can therefore land one ulp
        // above ncoarse-1 and truncate to j = ncoarse-1 - one past
        // the last interval. The clamp moves that node back onto the
        // last interval, where it evaluates at (at most one ulp
        // past) the interval's right endpoint.
        const double r = (double) i * lim[2] / dlnc;
        int j = (int) r;
        if (j > ncoarse - 2) {
          j = ncoarse - 2;
        }
        qidx[i] = j;
        qdel[i] = (r - j) * dlnc; // offset from node j, in ln(ell)
      }
      tabc = (double**) malloc2d(redshift.clustering_nbin, ncoarse);
      cspl = (double**) malloc2d(redshift.clustering_nbin, ncoarse);
    }
  }

  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_clustering) ||
      fdiff2(cache[2], redshift.random_clustering) ||
      fdiff2(cache[3], Ntable.random) ||
      fdiff2(cache[4], nuisance.random_galaxy_bias) ||
      // redshift.random_shear: with magnification bias on, amax_lens
      // (redshift_spline.c) ends the lens range at the source z_min,
      // so a new source n(z) alone moves these integrals
      fdiff2(cache[5], redshift.random_shear))
  {
    if (ncoarse > 0) {
      // ---------------------------------------------------------------
      // The internal coarse grid: general strategy.
      //
      // The real-space projection (w_gk_tomo, via the shared gk_
      // struct and C_gk_tomo_limber_fill) reads this table at every
      // integer ell up to Ntable.LMAX ~ 1e5 inside its Legendre
      // sums. At that call rate only the optimized, vectorized LINEAR
      // read is affordable: a cubic-spline lookup per ell would
      // dominate the whole evaluation.
      //
      // A linear read, however, is only accurate on a DENSE table -
      // and each of the N_ell = 512 nodes costs one exact Limber
      // quadrature, which is the expensive part.
      //
      // The coarse grid splits the difference: a cubic spline carries
      // far more accuracy per node than a linear segment, so the
      // expensive quadratures run on few nodes and a cheap cubic
      // upsampling fills the dense table:
      //
      //   exact Limber quadrature on ncoarse nodes (default 192)
      //     -> spline_coeffs_uniform: one tridiagonal solve per row
      //     -> Horner evaluation at the 512 precomputed fine offsets
      //     -> the unchanged dense table
      //     -> the same fast linear reads by every consumer
      //
      // This is safe because the cross-spectrum's BAO features are
      // mild; C_gg - the auto-spectrum, where the wiggles are
      // strongest - keeps the exact grid (see its header).
      // ---------------------------------------------------------------
      C_gk_tomo_limber_nointerp_ells(lxc, ncoarse, redshift.clustering_nbin, tabc);

      const double hc = dlnc;
      const double inv_hc = 1.0/dlnc;
      #pragma omp parallel for schedule(static)
      for (int nz=0; nz<redshift.clustering_nbin; nz++) {
        spline_coeffs_uniform(tabc[nz], ncoarse, hc, cspl[nz]);
      }
      // Upsampling. On interval [x_j, x_j + h] the house spline
      // (spline_coeffs_uniform) is the cubic
      //
      //   S(x_j + dx) = y_j + b dx + c_j dx^2 + d dx^3
      //
      // where c is the coefficient array the tridiagonal solve above
      // produced: the spline's second derivative / 2, with natural
      // boundaries c_0 = c_{n-1} = 0.
      //
      // The other two coefficients follow from two conditions:
      //
      //   S'' runs linearly from 2 c_j to 2 c_{j+1}
      //     -> d = (c_{j+1} - c_j) / (3 h)
      //
      //   S(x_{j+1}) = y_{j+1}, interpolate the right node
      //     -> b = (y_{j+1} - y_j)/h - h (c_{j+1} + 2 c_j)/3
      //
      // The polynomial is evaluated in Horner form; qidx/qdel hold
      // each fine node's precomputed interval j and offset dx.
      #pragma omp parallel for collapse(2) schedule(static)
      for (int nz=0; nz<redshift.clustering_nbin; nz++) {
        for (int i=0; i<nell; i++) {
          const double* restrict y = tabc[nz];
          const double* restrict cc = cspl[nz];
          const int j = qidx[i];
          const double b = (y[j+1] - y[j])*inv_hc
                           - hc*(cc[j+1] + 2.0*cc[j])/3.0;
          const double d = (cc[j+1] - cc[j])/(3.0*hc);
          table[nz][i] = y[j] + qdel[i]*(b + qdel[i]*(cc[j] + qdel[i]*d));
        }
      }
    }
    else {
      C_gk_tomo_limber_nointerp_ells(lx, nell, redshift.clustering_nbin, table);
    }
    cache[0] = cosmology.random;
    cache[1] = nuisance.random_photoz_clustering;
    cache[2] = redshift.random_clustering;
    cache[3] = Ntable.random;
    cache[4] = nuisance.random_galaxy_bias;
    cache[5] = redshift.random_shear;
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

// ---------------------------------------------------------------------------
// Fast batch interpolation of the galaxy x CMB-lensing C_l table at
// integer multipoles. Called by w_gk_tomo to fill ~100k ell values for the
// Hankel transform C_l -> w_gk(theta): the vectorized fill is faster than
// per-ell lookups, and GCC cannot auto-vectorize the indirect (gather)
// table access, so limber_fill_interp provides the explicit AVX2 path
// (i32gather_pd, 4 ells per iteration).
//
// Requires C_gk_tomo_limber to have been called first to populate gk_.tab.
// The output carries no CMB beam or pixel window (w_gk_tomo applies it).
//
// Parameters:
//   nz     - lens bin index (0..clustering_nbin-1)
//   lmin   - first multipole to fill (inclusive)
//   lmax   - last multipole to fill (exclusive)
//   ln_ell - precomputed log(l) array, indexed by l
//   out    - output C_l array, indexed by l
//
// Returns:
//   nothing; the interpolated C_l are written into out at indices
//   lmin..lmax-1
// ---------------------------------------------------------------------------
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
// KS = CMB LENSING x SHEAR
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// CMB-lensing x shear integrand core.
// Pure arithmetic on preloaded scalars - no branches, no table lookups -
// so GCC can vectorize the calling loop with #pragma omp simd.
//
// The IA contribution is the linear alignment amplitude only (C1 * Z1):
// the CMB convergence is a single spin-0 lens plane, so the cross keeps
// the NLA term under every IA model.
//
// Computes: (WK1 - WS1) * WKC * PK
//
// Parameters:
//   PK  - P_delta(k, a): nonlinear matter power spectrum
//   WK1 - W_kappa(a, fK, ni): lensing convergence kernel, source bin
//   WKC - W_k(a, fK): CMB lensing convergence kernel
//   WS1 - W_source(a, ni, h/h0) * IA_A1(a, D, ni): IA-weighted source
//         distribution
//
// Returns:
//   the integrand value at one quadrature node (no dchida/fK^2 amplitude
//   and no quadrature weight - the caller applies both)
// ---------------------------------------------------------------------------
static inline double int_for_C_ks_tomo_limber_core(
    const double PK,   // P_delta(k, a): nonlinear matter power spectrum
    const double WK1,  // W_kappa(a, fK, ni): lensing convergence kernel, source bin
    const double WKC,  // W_k(a, fK): CMB lensing convergence kernel
    const double WS1   // W_source(a, ni, h/h0) * IA_A1(a, D, ni): IA-weighted source distribution
  ) // inline necessary for vectorization
{
  return (WK1 - WS1)*WKC*PK;
}

// ---------------------------------------------------------------------------
// Single-ell CMB-lensing x shear C_l: a point diagnostic on the batch
// engine.
//
// Runs one C_ks_tomo_limber_nointerp_ells call at a single multipole and
// reads one entry, so it pays the WHOLE-TOMOGRAPHY batch cost per call
// (every source bin is computed even though one number is returned). Never
// loop this over (l, ns): call C_ks_tomo_limber_nointerp_ells once and
// index the result instead.
//
// Kept in the API as the exact per-multipole entry point of the probe (the
// C_ss_tomo_limber_nointerp pattern): notebooks evaluate single points
// here, and a per-integer-multipole Limber value is what a non-Limber
// pipeline consumes.
//
// Parameters:
//   l    - multipole moment
//   ns   - source redshift bin index (0..redshift.shear_nbin-1)
//
// Returns:
//   C_l^ks of source bin ns with the full Limber model (nonlinear P_delta;
//   the NLA intrinsic-alignment term under every IA model)
// ---------------------------------------------------------------------------
double C_ks_tomo_limber_nointerp(
    const double l,
    const int ns
  ) // slow (whole-tomography batch per call) - use the batch version
{
  if (ns < 0 || ns > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input ns = %d", ns); exit(1);
  }
  const int NSIZE = redshift.shear_nbin;
  double** tmp = (double**) malloc2d(NSIZE, 1);
  const double ell = l;

  C_ks_tomo_limber_nointerp_ells(&ell, 1, NSIZE, tmp);

  const double res = tmp[ns][0];
  free(tmp);
  return res;
}

// ---------------------------------------------------------------------------
// Core workhorse for all CMB-lensing x shear C_l computations (the interp
// table in C_ks_tomo_limber and the low-ell batch in
// C_ks_tomo_limber_nointerp_batch).
//
// Same design as C_ss_tomo_limber_work: precompute all expensive quantities
// (radial weights, IA amplitude, matter power spectrum) on a fixed grid of
// quadrature points, then evaluate the Limber integral for every
// (ell, source bin) combination with SIMD-vectorized inner loops.
//
// Key difference from SS (shared with GS): the integration limits differ
// per source bin (amin_source, amax_source vary with ni), so cosmo_nodes
// are created per source bin (cn_all[shear_nbin]) rather than a single
// global cn.
//
// Memory layout:
//   WC[3][shear_nbin][npts]: radial weights at each bin's quadrature nodes
//     WC[0] = W_kappa           (lensing convergence kernel, source bin)
//     WC[1] = W_source * IA_A1  (IA-weighted source galaxy distribution;
//                                only this product enters the ks integrand,
//                                see int_for_C_ks_tomo_limber_core)
//     WC[2] = W_k               (CMB lensing convergence kernel)
//   KP[shear_nbin][nell][npts]: P_delta(k, a) at k = (l + 1/2)/chi(a)
//
// Ell prefactor: [l*(l+1)/(l+0.5)^2] * [sqrt((l-1)*l*(l+1)*(l+2))/(l+0.5)^2]
//   = product of the convergence (spin-0) and shear (spin-2) field
//     prefactors (1812.05995 eqs 74-79)
//
// Cache invalidation:
// none here - every input arrives precomputed; the
// warm-up prelude initializes the lazily-built statics of the kernel and
// power-spectrum functions single-threaded before the parallel regions.
//
// Parameters:
//   cn_all - quadrature nodes per source bin [shear_nbin] with precomputed
//            cosmological quantities (scale factor, chi, D, H/H0, dchi/da)
//   lx     - array of multipole values, length nell
//   nell   - number of multipole values
//   table  - output array [shear_nbin][nell]
//
// Returns:
//   nothing; the result is written into table
// ---------------------------------------------------------------------------
static void C_ks_tomo_limber_work(
    const cosmo_nodes* cn_all,  // quadrature nodes per source bin [shear_nbin]
    const double* lx,           // multipole values (length nell)
    const int nell,             // number of multipole values
    double** table              // output [shear_nbin][nell]
  )
{
  halo_IA_unsupported("C_ks_tomo_limber_work");
  // -----------------------------------------------------------------------
  // Warm up all functions that lazily initialize internal static tables.
  // Must be called single-threaded before any parallel region touches them.
  // -----------------------------------------------------------------------
  {
    const cosmo_nodes* cn = &cn_all[0];
    const double a    = cn->data[CN_A][0];
    const double fK   = cn->data[CN_FK][0];
    const double hoh0 = cn->data[CN_HOVERH0][0];
    const double gf   = cn->data[CN_GROWFAC][0];
    const double ell  = lx[0] + 0.5;
    (void) W_kappa(a, fK, 0);
    (void) W_source(a, 0, hoh0);
    (void) IA_A1_Z1(a, gf, 0);
    (void) W_k(a, fK);
    (void) Pdelta(ell/fK, a);
  }

  const int npts = cn_all[0].npts;

  double*** WC = (double***) malloc3d(3, redshift.shear_nbin, npts);
  double*** KP = (double***) malloc3d(redshift.shear_nbin, nell, npts);

  #pragma omp parallel
  {
    // -----------------------------------------------------------------------
    // Precompute: radial weights and IA amplitude per (bin, quadrature point)
    // -----------------------------------------------------------------------
    #pragma omp for collapse(2) schedule(static) nowait
    for (int b = 0; b < redshift.shear_nbin; b++) {
      for (int p = 0; p < npts; p++) {
        const cosmo_nodes* cn = &cn_all[b];
        const double a    = cn->data[CN_A][p];
        const double fK   = cn->data[CN_FK][p];
        const double hoh0 = cn->data[CN_HOVERH0][p];
        const double gf   = cn->data[CN_GROWFAC][p];
        WC[0][b][p] = W_kappa(a, fK, b);
        WC[1][b][p] = W_source(a, b, hoh0)*IA_A1_Z1(a, gf, b);
        WC[2][b][p] = W_k(a, fK);
      }
    }
    // -----------------------------------------------------------------------
    // Precompute: P(k, a) per (bin, ell, quadrature point). The bin index
    // matters because each bin has its own quadrature nodes.
    // -----------------------------------------------------------------------
    #pragma omp for collapse(3) schedule(static)
    for (int b = 0; b < redshift.shear_nbin; b++) {
      for (int i = 0; i < nell; i++) {
        for (int p = 0; p < npts; p++) {
          const cosmo_nodes* cn = &cn_all[b];
          const double a   = cn->data[CN_A][p];
          const double fK  = cn->data[CN_FK][p];
          const double ell = lx[i] + 0.5;
          KP[b][i][p] = Pdelta(ell/fK, a);
        }
      }
    }
  }

  // -----------------------------------------------------------------------
  // Main integration loop.
  // The restrict pointers are hoisted before the p-loop to eliminate
  // gather instructions and enable contiguous AVX2 vector loads.
  // -----------------------------------------------------------------------
  #pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < nell; i++) {
    for (int b = 0; b < redshift.shear_nbin; b++) {
      const double* restrict fK     = cn_all[b].data[CN_FK];
      const double* restrict dchida = cn_all[b].data[CN_DCHIDA];
      const double* restrict wt     = cn_all[b].data[CN_WT];
      const double* restrict PK     = KP[b][i];
      const double* restrict WK1    = WC[0][b];
      const double* restrict WS1    = WC[1][b];
      const double* restrict WKC    = WC[2][b];
      const double l = lx[i];
      const double ell = l + 0.5;
      const double ell2 = ell*ell;
      const double ell_pf1 = l*(l + 1.)/ell2;
      const double tmp = (l - 1.)*l*(l + 1.)*(l + 2.);
      const double ell_pf2 = (tmp > 0) ? sqrt(tmp)/ell2 : 0.0;
      const double ell_pf = ell_pf1*ell_pf2;
      double sum = 0.0;
      #pragma omp simd reduction(+:sum)
      for (int p = 0; p < npts; p++) {
        const double amp = (dchida[p]/(fK[p]*fK[p]))*ell_pf;
        sum += int_for_C_ks_tomo_limber_core(PK[p],
                                             WK1[p],
                                             WKC[p],
                                             WS1[p]) * amp * wt[p];
      }
      table[b][i] = sum;
    }
  }
  free(WC);
  free(KP);
}

// ---------------------------------------------------------------------------
// Batch computation of CMB-lensing x shear C_l at arbitrary multipole
// values. Same design as C_ss_tomo_limber_nointerp_ells: takes an
// arbitrary array of ell values and writes results indexed 0..nell-1.
//
// Cache invalidation:
// the static Gauss-Legendre table w (64/128/256/512/
// 1024 nodes, keyed on abs(Ntable.high_def_integration)) rebuilds when
// Ntable.random changes; the per-bin cosmo_nodes are rebuilt on every
// call (they depend on the current cosmology).
//
// Parameters:
//   ells    - array of multipole values, length nell (need not be integers)
//   nell    - number of multipole values
//   NSIZE   - number of source tomographic bins (= redshift.shear_nbin;
//             the CMB is a single lens plane, so one spectrum per bin)
//   out     - output array [NSIZE][nell], indexed as out[nz][i]
//
// Returns:
//   nothing; the result is written into out
// ---------------------------------------------------------------------------
void C_ks_tomo_limber_nointerp_ells(
    const double* ells,  // array of multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of source tomographic bins (= shear_nbin)
    double** out         // output [NSIZE][nell], indexed as out[nz][i]
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

  if (NSIZE != redshift.shear_nbin) {
    log_fatal("NSIZE = %d != shear_nbin = %d", NSIZE, redshift.shear_nbin);
    exit(1);
  }
  if (nell <= 0) {
    log_fatal("nell = %d <= 0", nell); exit(1);
  }

  cosmo_nodes cn_all[redshift.shear_nbin];
  for (int b = 0; b < redshift.shear_nbin; b++) {
    const double amin = amin_source(b);
    const double amax = amax_source(b);
    if (!(amin>0) || !(amin<1) || !(amax>0) || !(amax<1)) {
      log_fatal("0 < amin/amax < 1 not true"); exit(1);
    }
    cn_all[b] = create_cosmo_nodes(amin, amax, w);
  }
  for (int q = 1; q < redshift.shear_nbin; q++) {
    if (cn_all[q].npts != cn_all[0].npts) {
      log_fatal("inconsistent quadrature size"); exit(1);
    }
  }

  C_ks_tomo_limber_work(cn_all, ells, nell, out);

  for (int b = 0; b < redshift.shear_nbin; b++) {
    free_cosmo_nodes(&cn_all[b]);
  }
}

// ---------------------------------------------------------------------------
// Batch CMB-lensing x shear Limber C_l at the integer multipoles
// l = lmin..lmax-1, written at their own index: Cl[nz][l]. Thin wrapper
// around C_ks_tomo_limber_nointerp_ells.
//
// Example: w_ks_tomo calls it with lmin = 1 and lmax = limits.LMIN_tab
// for the multipoles below the interpolation table; Cl[nz][0] is left
// untouched. The output carries no CMB beam or pixel window (w_ks_tomo
// applies it).
//
// Parameters:
//   lmin  - first multipole (inclusive)
//   lmax  - last multipole (exclusive)
//   NSIZE - number of source tomographic bins (= shear_nbin)
//   Cl    - output [NSIZE][>= lmax], indexed Cl[nz][l]
//
// Returns:
//   nothing; the result is written into Cl
// ---------------------------------------------------------------------------
void C_ks_tomo_limber_nointerp_batch(
    const int lmin,   // first multipole (inclusive)
    const int lmax,   // last multipole (exclusive)
    const int NSIZE,  // number of source tomographic bins (= shear_nbin)
    double** Cl       // output [NSIZE][>=lmax], indexed as Cl[nz][l]
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

  C_ks_tomo_limber_nointerp_ells(lx, nell, NSIZE, tmp);

  for (int k = 0; k < NSIZE; k++) {
    for (int i = 0; i < nell; i++) {
      Cl[k][lmin+i] = tmp[k][i];
    }
  }

  free(tmp); free(lx);
}

// ---------------------------------------------------------------------------
// Batch computation of the scale-cut derivative dC_ks/dlnk on a
// (ln k, ell) grid (2011.06469 eq 17).
//
// In the Limber integral each scale factor maps one-to-one onto
// k = (l + 1/2)/chi(a), so dC_ks/dlnk at a given (k, ell) is the per-chi
// C_ks integrand core/fK^2 evaluated at the single node with
// chi(a) = (l + 1/2)/k, times |dchi/dlnk| = chi: the per-node amplitude
// is 1/fK. Equivalently, it is the quadrature's per-a amplitude
// dchida/fK^2 times |da/dlnk| = fK/dchida - the dchida cancels. There is
// no quadrature sum here - every (k, ell, source bin) output is one core
// evaluation.
//
// Same integrand as C_ks_tomo_limber_work: (W_kappa - W_source*IA_A1) *
// W_k * P_delta with the spin-0 x spin-2 ell prefactor pf1*pf2
// (1812.05995 eqs 74-79). The ks cross keeps the NLA term under every IA
// model, so there is no EE/BB split and no TATT kernel table - one
// component per source bin. Unlike ss, the source support differs per
// bin (amin_source, amax_source vary with b), so the kernels are gated
// per (bin, node): a node whose scale factor falls outside bin b's
// support keeps that bin's kernels zero, and its output is exactly 0.
//
// With normalize = 0 the output is dC_ks/dlnk itself - what the
// real-space dlnw_ks machinery needs, since it Legendre-sums dC over ell
// before normalizing by w_ks(theta). With normalize = 1 the function
// also computes C_ks(ell, bin) - the same per-bin Gauss-Legendre
// quadrature as C_ks_tomo_limber_work - and writes dlnC_ks/dlnk = dC/C:
// one thread team computes the C_ks rows and then fills the dC rows,
// dividing each one right after filling it, while it is still cache-hot.
//
// Cache invalidation:
// none - no static state; every call recomputes
// from its arguments after its own single-threaded warm-up.
//
// Parameters:
//   lnkx      - ln k grid values (length nlnk), k in (Mpc/h)^-1
//   nlnk      - number of ln k grid values
//   lx        - multipole values (length nell)
//   nell      - number of multipole values
//   NSIZE     - number of source tomographic bins (= shear_nbin)
//   normalize - 1: write dlnC = dC/C_ks; 0: write dC
//   table     - output [NSIZE][nlnk][nell]
//
// Returns:
//   nothing; the result is written into table
// ---------------------------------------------------------------------------
void dC_ks_dlnk_tomo_limber_work(
    const double* lnkx,  // ln k grid values (length nlnk), k in (Mpc/h)^-1
    const int nlnk,      // number of ln k grid values
    const double* lx,    // multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of source tomographic bins (= shear_nbin)
    const int normalize, // 1: write dlnC = dC/C_ks; 0: write dC
    double*** table      // output [NSIZE][nlnk][nell]
  )
{
  halo_IA_unsupported("dC_ks_dlnk_tomo_limber_work");
  if (NSIZE != redshift.shear_nbin) {
    log_fatal("NSIZE = %d != shear_nbin = %d", NSIZE, redshift.shear_nbin);
    exit(1);
  }
  if (nlnk <= 0 || nell <= 0) {
    log_fatal("nlnk = %d and nell = %d must be positive", nlnk, nell);
    exit(1);
  }

  // per-bin source support, plus the union window that gates whether a
  // node is worth computing at all (outside it no bin contributes)
  double aminb[redshift.shear_nbin];
  double amaxb[redshift.shear_nbin];
  double aminw = 1.0;
  double amaxw = 0.0;
  for (int b = 0; b < redshift.shear_nbin; b++) {
    aminb[b] = amin_source(b);
    amaxb[b] = amax_source(b);
    if (!(aminb[b]>0) || !(aminb[b]<1) || !(amaxb[b]>0) || !(amaxb[b]<1)) {
      log_fatal("0 < amin/amax < 1 not true"); exit(1);
    }
    aminw = fmin(aminw, aminb[b]);
    amaxw = fmax(amaxw, amaxb[b]);
  }

  // -----------------------------------------------------------------------
  // Warm up all functions that lazily initialize internal static tables.
  // Must be called single-threaded before any parallel region touches them.
  // -----------------------------------------------------------------------
  {
    const double a = 0.5*(aminw + amaxw); // inside the source support
    struct chis chidchi = chi_all(a);
    const double fK   = chidchi.chi;
    const double hoh0 = hoverh0v2(a, chidchi.dchida);
    const double gf   = growfac(a);
    const double ell  = lx[0] + 0.5;
    (void) a_chi(fK);
    (void) f_K(fK);
    (void) W_kappa(a, fK, 0);
    (void) W_source(a, 0, hoh0);
    (void) IA_A1_Z1(a, gf, 0);
    (void) W_k(a, fK);
    (void) Pdelta(ell/fK, a);
  }

  // -----------------------------------------------------------------------
  // Allocate precomputed arrays (one entry per node p = f*nell + i)
  // -----------------------------------------------------------------------
  const int npts = nlnk*nell;

  double* AMP = (double*) malloc1d(npts);
  double* PKn = (double*) malloc1d(npts);
  double*** WC = (double***) malloc3d(3, redshift.shear_nbin, npts);
  zero3d(WC, 3, redshift.shear_nbin, npts);

  // -----------------------------------------------------------------------
  // Quadrature-side precompute (only when normalizing): C_ks needs its own
  // Gauss-Legendre node set along the line of sight, because the C_ell sum
  // runs over quadrature nodes, not (k, ell) grid nodes. Same machinery
  // and layouts as C_ks_tomo_limber_work: per-bin cosmo_nodes (the
  // integration limits differ per source bin), radial weights per
  // (source bin, node) in WCq, P_delta per (bin, ell, node) in KPq
  // (there k = (l + 1/2)/chi varies with ell at fixed node, so KPq keeps
  // the ell dimension the grid-side PKn does not need)
  // -----------------------------------------------------------------------
  gsl_integration_glfixed_table* w = NULL;
  cosmo_nodes cn_all[redshift.shear_nbin];
  int cnpts = 0;
  double*** WCq = NULL;
  double*** KPq = NULL;
  double** CKS = NULL;
  if (1 == normalize) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 64 :
                         (1 == hdi) ? 128 :
                         (2 == hdi) ? 256 :
                         (3 == hdi) ? 512 : 1024; // predefined GSL tables
    w = malloc_gslint_glfixed(szint);
    for (int b = 0; b < redshift.shear_nbin; b++) {
      cn_all[b] = create_cosmo_nodes(aminb[b], amaxb[b], w);
    }
    for (int q = 1; q < redshift.shear_nbin; q++) {
      if (cn_all[q].npts != cn_all[0].npts) {
        log_fatal("inconsistent quadrature size"); exit(1);
      }
    }
    cnpts = cn_all[0].npts;
    WCq = (double***) malloc3d(3, redshift.shear_nbin, cnpts);
    KPq = (double***) malloc3d(redshift.shear_nbin, nell, cnpts);
    CKS = (double**) malloc2d(NSIZE, nell);
    #pragma omp parallel
    {
      #pragma omp for collapse(2) schedule(static) nowait
      for (int b = 0; b < redshift.shear_nbin; b++) {
        for (int p = 0; p < cnpts; p++) {
          const cosmo_nodes* cn = &cn_all[b];
          const double a    = cn->data[CN_A][p];
          const double fK   = cn->data[CN_FK][p];
          const double hoh0 = cn->data[CN_HOVERH0][p];
          const double gf   = cn->data[CN_GROWFAC][p];
          WCq[0][b][p] = W_kappa(a, fK, b);
          WCq[1][b][p] = W_source(a, b, hoh0)*IA_A1_Z1(a, gf, b);
          WCq[2][b][p] = W_k(a, fK);
        }
      }
      #pragma omp for collapse(3) schedule(static)
      for (int b = 0; b < redshift.shear_nbin; b++) {
        for (int i = 0; i < nell; i++) {
          for (int p = 0; p < cnpts; p++) {
            const cosmo_nodes* cn = &cn_all[b];
            const double a   = cn->data[CN_A][p];
            const double fK  = cn->data[CN_FK][p];
            const double ell = lx[i] + 0.5;
            KPq[b][i][p] = Pdelta(ell/fK, a);
          }
        }
      }
    }
  }

  // -----------------------------------------------------------------------
  // Precompute per node: the dlnk amplitude, P_delta, and the radial
  // weights per source bin (WC, same layout as C_ks_tomo_limber_work;
  // each node has a single k, so P_delta needs no separate ell dimension
  // here). WC entries stay zero for the bins whose source support does
  // not contain the node's scale factor.
  // -----------------------------------------------------------------------
  #pragma omp parallel for collapse(2) schedule(static)
  for (int f = 0; f < nlnk; f++) {
    for (int i = 0; i < nell; i++) {
      const int p = f*nell + i;
      const double l = lx[i];
      const double ell = l + 0.5;
      // the (k, ell) pair selects one Limber node: chi(a) = ell/k, with k
      // converted from (Mpc/h)^{-1} to ((Mpc/h)/(c/H0=100))^{-1}
      const double a = a_chi(f_K(ell/(exp(lnkx[f])*cosmology.coverH0)));
      if (!(a > aminw && a < amaxw)) {
        AMP[p] = 0.0;
        PKn[p] = 0.0;
        continue;
      }
      struct chis chidchi = chi_all(a);
      const double growfac_a = growfac(a);
      const double hoverh0 = hoverh0v2(a, chidchi.dchida);
      const double fK = chidchi.chi;
      const double k = ell/fK;
      const double ell2 = ell*ell;
      const double ell_pf1 = l*(l + 1.)/ell2;
      const double tmp = (l - 1.)*l*(l + 1.)*(l + 2.);
      const double ell_pf2 = (tmp > 0) ? sqrt(tmp)/ell2 : 0.0;
      AMP[p] = (ell_pf1*ell_pf2)/fK;
      for (int b = 0; b < redshift.shear_nbin; b++) {
        if (!(a > aminb[b] && a < amaxb[b])) {
          continue; // outside bin b's source support: kernels stay 0
        }
        WC[0][b][p] = W_kappa(a, fK, b);
        WC[1][b][p] = W_source(a, b, hoverh0)*IA_A1_Z1(a, growfac_a, b);
        WC[2][b][p] = W_k(a, fK);
      }
      PKn[p] = Pdelta(k, a);
    }
  }

  // -----------------------------------------------------------------------
  // Main fill loop.
  //
  // Where the derivative differs from C_ks: in C_ks_tomo_limber_work each
  // output is a quadrature SUM over the line of sight,
  //
  //   C_ks(l) = sum_p core(p) * (dchida[p]/fK[p]^2) * ell_prefactor * wt[p],
  //
  // because every scale factor contributes to one C_ell. Here each output
  // is ONE core evaluation with no reduction,
  //
  //   dC_ks/dlnk(k, l) = core(p(k, l)) * ell_prefactor / fK,
  //
  // because at fixed ell the Limber relation k = (l + 1/2)/chi picks a
  // single node p(k, l), and changing variables from a to ln k multiplies
  // the per-a integrand core * (dchida/fK^2) by |da/dlnk| = fK/dchida:
  // the dchida cancels and one power of 1/fK survives (2011.06469 eq 17).
  // AMP carries that per-node amplitude,
  // with AMP = 0 marking nodes outside every bin's source support and
  // zeroed WC kernels marking the per-bin exclusions. The core function
  // and its inputs (WC, PKn) are exactly the ones the C_ell sum uses:
  // only the amplitude and the absence of the sum differ.
  //
  // When normalizing, one thread team does everything: its first loop
  // computes the C_ks rows (the quadrature sum below, one row per source
  // bin - C_ks does not depend on k, so each row serves every f), and
  // after the loop's implicit barrier the same team fills the dC rows and
  // divides each one to dlnC = dC/C while it is still cache-hot. No
  // intermediate dC table exists and no pass re-reads the output.
  // The restrict pointers are hoisted before the inner loops to eliminate
  // gather instructions and enable contiguous vector loads.
  // -----------------------------------------------------------------------
  #pragma omp parallel
  {
  if (1 == normalize) { // C_ks rows first: the division below reads them
    #pragma omp for collapse(2) schedule(static)
    for (int b = 0; b < NSIZE; b++) {
      for (int i = 0; i < nell; i++) {
        const double* restrict fKq    = cn_all[b].data[CN_FK];
        const double* restrict dchida = cn_all[b].data[CN_DCHIDA];
        const double* restrict wtq    = cn_all[b].data[CN_WT];
        const double* restrict PKq    = KPq[b][i];
        const double* restrict WK1q   = WCq[0][b];
        const double* restrict WS1q   = WCq[1][b];
        const double* restrict WKCq   = WCq[2][b];
        const double l = lx[i];
        const double ell = l + 0.5;
        const double ell2 = ell*ell;
        const double ell_pf1 = l*(l + 1.)/ell2;
        const double tmp = (l - 1.)*l*(l + 1.)*(l + 2.);
        const double ell_pf2 = (tmp > 0) ? sqrt(tmp)/ell2 : 0.0;
        const double ell_pf = ell_pf1*ell_pf2;
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int p = 0; p < cnpts; p++) {
          const double ampq = (dchida[p]/(fKq[p]*fKq[p]))*ell_pf;
          sum += int_for_C_ks_tomo_limber_core(PKq[p],
                                               WK1q[p],
                                               WKCq[p],
                                               WS1q[p]) * ampq * wtq[p];
        }
        CKS[b][i] = sum;
      }
    } // implicit barrier: C_ks rows complete before any division below
  }
  #pragma omp for collapse(2) schedule(static)
  for (int b = 0; b < NSIZE; b++) {
    for (int f = 0; f < nlnk; f++) {
      const double* restrict amp = &AMP[f*nell];
      const double* restrict PK  = &PKn[f*nell];
      const double* restrict WK1 = &WC[0][b][f*nell];
      const double* restrict WS1 = &WC[1][b][f*nell];
      const double* restrict WKC = &WC[2][b][f*nell];
      double* restrict out = table[b][f];
      #pragma omp simd
      for (int i = 0; i < nell; i++) {
        out[i] = int_for_C_ks_tomo_limber_core(PK[i],
                                               WK1[i],
                                               WKC[i],
                                               WS1[i]) * amp[i];
      }
      if (1 == normalize) { // dlnC = dC/C, dividing while the row is
        // cache-hot; a near-zero dC passes through and a near-zero C gives
        // 0, so the ratio never blows up where the spectrum vanishes
        const double* restrict cks = CKS[b];
        #pragma omp simd
        for (int i = 0; i < nell; i++) {
          const double dCKS = out[i];
          const double CKSv = (fabs(dCKS) > 1e-30) ? cks[i] : 1.0;
          out[i] = (fabs(CKSv) > 1e-30) ? dCKS/CKSv : 0.0;
        }
      }
    }
  }
  } // end of the parallel region
  free(AMP);
  free(PKn);
  free(WC);
  if (1 == normalize) {
    free(CKS);
    free(WCq);
    free(KPq);
    for (int b = 0; b < redshift.shear_nbin; b++) {
      free_cosmo_nodes(&cn_all[b]);
    }
    gsl_integration_glfixed_table_free(w);
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared state between C_ks_tomo_limber (which builds the interpolation table)
// and C_ks_tomo_limber_fill (which reads it to fill Cl arrays at ~100k ell
// values for real-space correlation functions).
//
//   tab     - pointer to the cached table[shear_nbin][nell]
//             (owned by C_ks_tomo_limber's static)
//   lim[0]  - log(l_min) of the interpolation grid
//   lim[1]  - log(l_max) of the interpolation grid
//   lim[2]  - uniform spacing in log(l): (lim[1] - lim[0]) / (nell - 1)
//   nell    - number of grid points in the interpolation table
// ---------------------------------------------------------------------------
static struct { double** tab; double lim[3]; int nell; } ks_ = {0};

// ---------------------------------------------------------------------------
// CMB-lensing x shear angular power spectrum C_l with interpolation.
// Builds the (source bin, log ell) table with one
// C_ks_tomo_limber_nointerp_ells call, then caches it for subsequent
// lookups. Returns the interpolated value at the requested l via
// interpol1d.
//
// Why the table is shared through the ks_ static struct: the
// real-space projection (w_ks_tomo) needs C_l at every integer
// multipole up to Ntable.LMAX ~ 1e5, for every tomographic pair,
// inside its Legendre/Hankel sums - millions of table reads per
// likelihood evaluation. Only the vectorized batch reader
// (C_ks_tomo_limber_fill, which runs the interpol1d linear read four
// multipoles at a time through AVX2 gathers) sustains that rate;
// calling this function one multipole at a time would dominate the
// whole evaluation.
//
// The struct is how the table travels between the two functions.
// The builder (this function) and the reader (the _fill) never call
// each other - the real-space projection calls one, the C_ell paths
// call the other - so no argument list connects them. Instead the
// builder publishes the table pointer and the grid geometry (the
// ln(ell) limits, spacing and node count) in the file-scope struct,
// and the reader picks them up there.
//
// When Ntable.N_ell[NODES_COARSE] is active, the exact quadrature instead
// runs on the internal coarse grid and the house cubic spline
// upsamples onto the unchanged N_ell nodes (the strategy block inside
// explains why this wins).
//
// Cache invalidation:
// recomputes when any of these change:
//   cosmology.random, nuisance.random_photoz_shear, nuisance.random_ia,
//   redshift.random_shear, Ntable.random
// Stored values carry no CMB beam or pixel window; w_ks_tomo multiplies
// its own copy by the beam_cmb/w_pixel filter.
//
// Parameters:
//   l  - multipole moment (continuous; outside the grid the lookup warns
//        and extrapolates)
//   ni - source redshift bin (the CMB is a single lens plane, so there
//        is no second bin index)
//
// Returns:
//   C_l^ks of source bin ni
// ---------------------------------------------------------------------------
double C_ks_tomo_limber(
    const double l,  // multipole moment (continuous)
    const int ni     // source redshift bin
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static int nell;
  static double lim[3];
  static double* lx = NULL;
  static int ncoarse = 0;  // active internal coarse grid size (0 = off)
  static double dlnc = 0.; // coarse grid spacing in ln(ell)
  static double* lxc = NULL;   // coarse ell nodes
  static int* qidx = NULL;     // fine node -> coarse interval (uniform
  static double* qdel = NULL;  //   grids: precomputed, no search)
  static double** tabc = NULL; // coarse C_ell values
  static double** cspl = NULL; // natural-cubic-spline c coefficients

  if (NULL == table || fdiff2(cache[4], Ntable.random)) {
    nell = Ntable.N_ell[NODES_DENSE];
    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_ell[NODES_DENSE] - 1.0);

    if (table != NULL) free(table);
    table = (double**) malloc2d(redshift.shear_nbin, Ntable.N_ell[NODES_DENSE]);

    ks_.tab    = table;
    ks_.lim[0] = lim[0];
    ks_.lim[1] = lim[1];
    ks_.lim[2] = lim[2];
    ks_.nell   = nell;

    if (lx != NULL) free(lx);
    lx = (double*) malloc1d(nell);
    for (int i = 0; i < nell; i++) {
      lx[i] = exp(lim[0] + i*lim[2]);
    }

    // Coarse-grid workspace (the strategy is explained where the grid
    // is used, in the refill block below): every allocation lives
    // HERE, in the Ntable rebuild block; the per-cosmology refill only
    // fills. The pieces are:
    //   lxc        - the ncoarse ell nodes, log-spaced over the same
    //                [lim[0], lim[1]] range as the fine table
    //   tabc, cspl - the coarse C_ell values and their cubic-spline
    //                coefficients, one row per source bin
    //   qidx, qdel - for each fine node, the coarse interval it falls
    //                in and its ln(ell) offset from that interval's
    //                left node: both grids are uniform in ln(ell) with
    //                shared endpoints, so this is pure grid geometry,
    //                computed once - no search of any kind at refill
    if (lxc  != NULL) { free(lxc);  lxc  = NULL; }
    if (qidx != NULL) { free(qidx); qidx = NULL; }
    if (qdel != NULL) { free(qdel); qdel = NULL; }
    if (tabc != NULL) { free(tabc); tabc = NULL; }
    if (cspl != NULL) { free(cspl); cspl = NULL; }
    const int nc = Ntable.N_ell[NODES_COARSE];
    ncoarse = (nc > 3 && nc < nell) ? nc : 0;
    if (ncoarse > 0) {
      dlnc = (lim[1] - lim[0]) / ((double) ncoarse - 1.0);
      lxc = (double*) malloc1d(ncoarse);
      for (int i=0; i<ncoarse; i++) {
        lxc[i] = exp(lim[0] + i*dlnc);
      }
      qidx = (int*) malloc(sizeof(int) * nell);
      qdel = (double*) malloc1d(nell);
      for (int i=0; i<nell; i++) {
        // Where does fine node i sit on the coarse grid? Both grids
        // run over the same [lim[0], lim[1]] in ln(ell), so the map
        // is pure arithmetic:
        //
        //   fine node i -> ln(ell) = lim[0] + i*lim[2]
        //               -> r = i*lim[2]/dlnc   (coarse spacings in)
        //               -> j = (int) r         (interval's left node)
        //               -> qdel = (r - j)*dlnc (offset inside it)
        //
        // The spline evaluates on interval [j, j+1], so the largest
        // legal j is ncoarse-2, the left node of the LAST interval.
        //
        // Why the clamp: at the shared top endpoint, i*lim[2] and
        // (ncoarse-1)*dlnc are two floating-point roundings of the
        // same length lim[1] - lim[0]. r can therefore land one ulp
        // above ncoarse-1 and truncate to j = ncoarse-1 - one past
        // the last interval. The clamp moves that node back onto the
        // last interval, where it evaluates at (at most one ulp
        // past) the interval's right endpoint.
        const double r = (double) i * lim[2] / dlnc;
        int j = (int) r;
        if (j > ncoarse - 2) {
          j = ncoarse - 2;
        }
        qidx[i] = j;
        qdel[i] = (r - j) * dlnc; // offset from node j, in ln(ell)
      }
      tabc = (double**) malloc2d(redshift.shear_nbin, ncoarse);
      cspl = (double**) malloc2d(redshift.shear_nbin, ncoarse);
    }
  }

  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    if (ncoarse > 0) {
      // ---------------------------------------------------------------
      // The internal coarse grid: general strategy.
      //
      // The real-space projection (w_ks_tomo, via the shared ks_
      // struct and C_ks_tomo_limber_fill) reads this table at every
      // integer ell up to Ntable.LMAX ~ 1e5 inside its Legendre
      // sums. At that call rate only the optimized, vectorized LINEAR
      // read is affordable: a cubic-spline lookup per ell would
      // dominate the whole evaluation.
      //
      // A linear read, however, is only accurate on a DENSE table -
      // and each of the N_ell = 512 nodes costs one exact Limber
      // quadrature, which is the expensive part.
      //
      // The coarse grid splits the difference: a cubic spline carries
      // far more accuracy per node than a linear segment, so the
      // expensive quadratures run on few nodes and a cheap cubic
      // upsampling fills the dense table:
      //
      //   exact Limber quadrature on ncoarse nodes (default 192)
      //     -> spline_coeffs_uniform: one tridiagonal solve per row
      //     -> Horner evaluation at the 512 precomputed fine offsets
      //     -> the unchanged dense table
      //     -> the same fast linear reads by every consumer
      //
      // This is safe because C_ks is smooth in ln(ell) - both
      // fields are lensing kernels, no galaxy density enters; C_gg
      // keeps the exact grid (see its header).
      // ---------------------------------------------------------------
      C_ks_tomo_limber_nointerp_ells(lxc, ncoarse, redshift.shear_nbin, tabc);

      const double hc = dlnc;
      const double inv_hc = 1.0/dlnc;
      #pragma omp parallel for schedule(static)
      for (int nz=0; nz<redshift.shear_nbin; nz++) {
        spline_coeffs_uniform(tabc[nz], ncoarse, hc, cspl[nz]);
      }
      // Upsampling. On interval [x_j, x_j + h] the house spline
      // (spline_coeffs_uniform) is the cubic
      //
      //   S(x_j + dx) = y_j + b dx + c_j dx^2 + d dx^3
      //
      // where c is the coefficient array the tridiagonal solve above
      // produced: the spline's second derivative / 2, with natural
      // boundaries c_0 = c_{n-1} = 0.
      //
      // The other two coefficients follow from two conditions:
      //
      //   S'' runs linearly from 2 c_j to 2 c_{j+1}
      //     -> d = (c_{j+1} - c_j) / (3 h)
      //
      //   S(x_{j+1}) = y_{j+1}, interpolate the right node
      //     -> b = (y_{j+1} - y_j)/h - h (c_{j+1} + 2 c_j)/3
      //
      // The polynomial is evaluated in Horner form; qidx/qdel hold
      // each fine node's precomputed interval j and offset dx.
      #pragma omp parallel for collapse(2) schedule(static)
      for (int nz=0; nz<redshift.shear_nbin; nz++) {
        for (int i=0; i<nell; i++) {
          const double* restrict y = tabc[nz];
          const double* restrict cc = cspl[nz];
          const int j = qidx[i];
          const double b = (y[j+1] - y[j])*inv_hc
                           - hc*(cc[j+1] + 2.0*cc[j])/3.0;
          const double d = (cc[j+1] - cc[j])/(3.0*hc);
          table[nz][i] = y[j] + qdel[i]*(b + qdel[i]*(cc[j] + qdel[i]*d));
        }
      }
    }
    else {
      C_ks_tomo_limber_nointerp_ells(lx, nell, redshift.shear_nbin, table);
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
  return interpol1d(table[q], Ntable.N_ell[NODES_DENSE], lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// Fast batch interpolation of the CMB-lensing x shear C_l table at
// integer multipoles. Called by w_ks_tomo to fill ~100k ell values for the
// Hankel transform C_l -> w_ks(theta): the vectorized fill is faster than
// per-ell lookups, and GCC cannot auto-vectorize the indirect (gather)
// table access, so limber_fill_interp provides the explicit AVX2 path
// (i32gather_pd, 4 ells per iteration).
//
// Requires C_ks_tomo_limber to have been called first to populate ks_.tab.
// The output carries no CMB beam or pixel window (w_ks_tomo applies it).
//
// Parameters:
//   nz     - source bin index (0..shear_nbin-1)
//   lmin   - first multipole to fill (inclusive)
//   lmax   - last multipole to fill (exclusive)
//   ln_ell - precomputed log(l) array, indexed by l
//   out    - output C_l array, indexed by l
//
// Returns:
//   nothing; the interpolated C_l are written into out at indices
//   lmin..lmax-1
// ---------------------------------------------------------------------------
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
// KK = CMB LENSING AUTO
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Scalar integrand for C_kk (CMB lensing convergence auto spectrum).
//
// At the Limber wavenumber k = (l + 1/2)/fK:
//
//   integrand(a) = W_k(a, fK)^2 * P_delta(k, a) * (dchi/da / fK^2)
//                  * [l*(l+1)/(l+0.5)^2]^2
//
// with two powers of the spin-0 convergence ell prefactor, one per
// convergence field (1812.05995 eqs 74-79). The CMB is a single source
// plane, so there is no tomographic bin index, and no CMB beam or pixel
// window enters here.
//
// Parameters:
//   a      - scale factor (integration variable; must satisfy 0 < a < 1)
//   params - double[1]: ar[0] = l, the multipole moment
//
// Returns:
//   the Limber integrand at scale factor a (GSL integrand interface)
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
  
  const double ell_prefactor = l*(l + 1.0)/(ell*ell); // 1812.05995 eqs 74-79

  return WK*WK*PK*(chidchi.dchida/(fK*fK))*ell_prefactor*ell_prefactor;
}

// ---------------------------------------------------------------------------
// Single-ell CMB lensing convergence auto C_l via GSL fixed-order
// Gauss-Legendre quadrature of int_for_C_kk_limber over the scale factor
// range [limits.a_min*(1 + 1e-5), 0.99999]: the full line of sight, since
// the CMB is a single source plane with no per-bin integration range.
//
// With init = 1 the integrand is evaluated once at amin instead, warming
// the statics inside the distance, lensing kernel and power spectrum
// functions so later calls can run inside parallel regions (C_kk_limber
// does this before its parallel table fill).
//
// Cache invalidation:
// the static GSL table w (64/128/256/512/1024 nodes,
// keyed on abs(Ntable.high_def_integration)) rebuilds when Ntable.random
// changes.
//
// Parameters:
//   l    - multipole moment
//   init - 1 = warm up statics only, 0 = compute
//
// Returns:
//   C_l^kk (integrand value at amin when init = 1); no CMB beam or pixel
//   window is applied
// ---------------------------------------------------------------------------
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
  // Endpoints nudged off the exact limits: the distance/growth tables
  // cover [limits.a_min, 1] and the integrand requires a strictly
  // inside (0, 1) - at a = 1, chi -> 0 and the Limber wavenumber
  // k = (l + 1/2)/chi diverges. The relative 1e-5 and the 0.99999 keep
  // every quadrature node inside both domains.
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

// ---------------------------------------------------------------------------
// CMB lensing convergence auto power spectrum C_l^kk with interpolation.
//
// Builds a single log-spaced table (Ntable.N_ell[NODES_DENSE] points covering
// l = LMIN_tab..LMAX) filled per ell by C_kk_limber_nointerp inside an
// OpenMP loop, after one single-threaded init call warms the statics.
// There is no tomography, so the table is one-dimensional.
//
// Cache invalidation:
// the table allocation and grid limits rebuild when
// the table is NULL or Ntable.random changes; the values refill when
// cosmology.random or Ntable.random change.
//
// Parameters:
//   l - multipole moment (continuous; outside the grid the lookup warns
//       and extrapolates)
//
// Returns:
//   C_l^kk, interpolated from the cached table via interpol1d; no CMB
//   beam or pixel window is applied
// ---------------------------------------------------------------------------
double C_kk_limber(const double l)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* table = NULL;
  static double lim[3];

  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    lim[1] = log(Ntable.LMAX + 1);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_ell[NODES_DENSE] - 1.0);
    if (table != NULL) free(table);
    table = (double*) malloc1d(Ntable.N_ell[NODES_DENSE]);
  }
  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random)) {
    (void) C_kk_limber_nointerp(exp(lim[0]), 1); // init static vars    
    #pragma omp parallel for schedule(static)
    for (int i=0; i<Ntable.N_ell[NODES_DENSE]; i++) {
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
  return interpol1d(table, Ntable.N_ell[NODES_DENSE], lim[0], lim[1], lim[2], lnl);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------


// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// Non-Limber (Angular Power Spectrum)
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------

// ----------------------------------------------------------------------------
// The non-Limber pipeline (C_cl_tomo for gg, C_gs_tomo for gs):
//
//   radial kernels fx on the log-chi grid
//     -> cfftlog_ells_p1   ell-independent forward FFT of every active
//                          (bin, component) row, done once per call
//     -> cfftlog_ells_p2   ell-dependent inverse transform, blocks of
//                          16 multipoles: Gamma kernel x forward
//                          coefficients -> inverse FFT -> Fy
//     -> assemble          per pair: sum_q F_1 F_2 k^3 P_lin dlnk
//                          gives C^fftlog(P_lin)
//     -> combine           C_l = C^fftlog(P_lin) + C^Limber(P_NL)
//                                - C^Limber(P_lin)  (FKEM split)
//     -> converge          after each block, a bin/pair whose C_l
//                          matches its Limber value within tol freezes;
//                          its remaining multipoles take the
//                          Limber-path values
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
//
// For example:
//   N = 10240 = 2^11 x 5  -> 11 radix-2 stages + 1 radix-5 stage, all fast
//   N = 10201 = 101 x 101 -> two levels of prime-101 sub-transforms, slow
//
// Padding to a slightly larger FFT-friendly size does not affect the convolution
// result: the extra elements are zeros, and we read the same output indices
// regardless of the padded size.
//
// Parameters:
//   n - minimum acceptable transform size
//
// Returns:
//   the smallest m >= n whose prime factorization contains only 2, 3, 5
//   and 7
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
// Typical values used in C_cl_tomo and C_gs_tomo:
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
                          //   0: j_l(kr)   - standard projection (density, magnification)
                          //   1: j_l'(kr)  - first derivative
                          //   2: j_l''(kr) - second derivative (RSD, which involves
                          //                  the second derivative of the velocity field)
  long N_pad;             // number of zero-padding elements added on each side of the
                          // input array before FFT (larger N_pad reduces edge effects
                          // but increases FFT cost; 200 for density, 500 for RSD/mag)
  long N_extrap_low;      // number of points for low-x power-law extrapolation (unused)
  long N_extrap_high;     // number of points for high-x power-law extrapolation (unused)
} config;

// ---------------------------------------------------------------------------
// Non-Limber projection via FFTLog: Phase 1 (ell-independent forward
// transform). Shared by C_cl_tomo (gg) and C_gs_tomo (gs).
//
// The non-Limber angular power spectra use the FFTLog algorithm
// to evaluate the Hankel-like integral that replaces the Limber approximation
// at low multipoles (l < LMAX_NOLIMBER). The computation is split into two
// phases to avoid redundant work:
//
//   Phase 1 (this function): forward FFT of the radial weight functions.
//     The input functions fx[i][j][q] - galaxy density (j=0), RSD velocity
//     (j=1), and optionally magnification (j=2) - are biased by x^(-nu),
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
//     (rows are lens bins in C_cl_tomo; C_gs_tomo appends source rows
//     whose slot 2 carries the lensing + IA kernel instead)
//   toutfwd[SIZE1*SIZE2][Nmax/2+1]: output forward FFT coefficients
//     indexed as toutfwd[i*SIZE2+j] for bin i, component j
//   eta_m[SIZE2][Nmax/2+1]: output Fourier-space frequencies
//     eta_m[j][q] = 2*pi*q / (dlnx * N[j][2])
//
// The c-window - a C^1 smoothed-step taper in Fourier space,
// W(t) = t - sin(2 pi t)/(2 pi), the normalized integral of a raised
// cosine - is applied after the forward FFT to suppress ringing from
// the finite chi range (the formula and its properties are derived at
// the taper table below). The width of the tapered band is controlled
// by cfg[j].c_window_width (typically 0.25).
//
// Cache invalidation:
// the static FFTW forward plans (planf) and the
// c-window tables (W_table, W_kmax) rebuild only when SIZE2 or any
// N[j][2] changes (i.e., when Ntable settings change, not on every
// cosmology evaluation).
//
// Parameters:
//   x       - chi grid values, length Nx (log-spaced)
//   fx      - input functions fx[SIZE1][SIZE2][Nx]
//   Nx      - number of chi grid points
//   cfg     - FFTLog config per component (nu, c_window_width, N_pad)
//   toutfwd - output forward FFT coefficients [SIZE1*SIZE2][Nmax/2+1]
//   eta_m   - output Fourier frequencies [SIZE2][Nmax/2+1]
//   N       - per-component sizes: N[j][0] = N_pad, N[j][1] = Nx,
//             N[j][2] = FFT size
//   Nmax    - max(N[j][2]) across all components
//   active  - per-(row, component) activity mask [SIZE1][SIZE2], or NULL
//             for all active: an inactive (i, j) slot skips its forward
//             FFT and writes zero coefficients (callers whose slots hold
//             identically zero kernels save the transform)
//   SIZE1   - number of radial rows (bins)
//   SIZE2   - number of radial components per row (2 or 3)
//
// Returns:
//   nothing; toutfwd and eta_m are filled
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
    const int* const* const active,         // activity mask [SIZE1][SIZE2]; NULL = all active
    int const SIZE1,                        // number of radial rows (bins)
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

  // Bias each active row by x^-nu and zero-pad it into fb: N[j][0]
  // guard zeros, the Nx biased samples, then zeros up to the FFT size.
  double*** fb = (double***) malloc3d(SIZE1, SIZE2, Nmax); // biased input func
  #pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<SIZE1; i++) {
    for(int j=0; j<SIZE2; j++) {
      if (active != NULL && !active[i][j]) {
        continue; // slot skipped: fb never read (the FFT below skips too)
      }
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

  // FFTW forward plans and the c-window taper table: rebuilt only when
  // SIZE2 or an FFT size changed, never per cosmology.
  if (rebuild) {
    if (planf != NULL) {
      for (int j = 0; j < cache[0]; j++) {
        fftw_destroy_plan(planf[j]);
      }
      free(planf);
    }
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

    // The taper table. With t = k/kmax in [0, 1],
    //
    //   W(t) = t - sin(2 pi t)/(2 pi)
    //
    // is the normalized integral of the raised cosine 1 - cos(2 pi t):
    // W(0) = 0, W(1) = 1, and W'(0) = W'(1) = 0, so the taper joins
    // both ends with continuous slope (C^1) and adds no ringing of its
    // own. It is applied below from the Nyquist entry downward over
    // the top c_window_width fraction of the stored spectrum, running
    // from W = 1 (interior untouched) to W = 0 at Nyquist.
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
      if (active != NULL && !active[i][j]) { // zero coefficients, no FFT
        for (int q=0; q<Nmax/2+1; q++) {
          toutfwd[i*SIZE2+j][q] = 0.0;
        }
        continue;
      }
      fftw_execute_dft_r2c(planf[j], fb[i][j], toutfwd[i*SIZE2+j]);
      // Apply the c-window taper: entry halfN - k is scaled by W[k],
      // so the highest |m| modes - the ones carrying the finite-range
      // ringing - are suppressed down to zero at Nyquist, while the
      // low-|m| modes, which hold the smooth physical kernel, pass
      // untouched. The r2c layout stores only m = 0..N/2 of the real
      // input, so this top-of-spectrum sweep is the whole taper.
      const int halfN = N[j][2]/2;
      const int kmax = W_kmax[j];
      const double* const W = W_table[j];
      for(int k=0; k<(kmax+1); k++) { // window for right-side
        toutfwd[i*SIZE2+j][halfN-k] *= W[k];
      }
    }
  }

  // Fourier-space frequencies of the log grid, per component:
  // eta_m[j][q] = 2 pi q / (dlnx * N[j][2]).
  const double dlnx = log(x[1]/x[0]);
  for(int j=0; j<SIZE2; j++) {
    const double scale = (2.0*M_PI/(dlnx * N[j][2]));
    for(int q=0; q<N[j][2]/2+1; q++) {
      eta_m[j][q] = scale * q;  
    }
  }

  free((void*) fb);
}

// ---------------------------------------------------------------------------
// Non-Limber projection via FFTLog: Phase 2 (ell-dependent inverse
// transform). Shared by C_cl_tomo (gg) and C_gs_tomo (gs).
//
// Processes a block of multipoles l = ks..ke-1, computing the projected
// radial functions Fy[i][j][k][q] and wavenumber grid y[i][k][q] for each
// radial row i, radial component j, multipole k, and chi-node q.
//
// Called repeatedly from C_cl_tomo and C_gs_tomo in a while-loop over
// blocks of BLOCK multipoles, with early termination when the non-Limber
// result converges to the Limber result (rows whose spectra have all
// converged are skipped via the converged array).
//
// Algorithmic steps for each block:
//
//   1. Wavenumber grid: y[i][k][q] = (k+1) / x[Nx-1-q]
//      The output wavenumber at each chi-node, reversed relative to x.
//
//   2. Gamma-function kernel gl[j][k][q]:
//      The Bessel-function transform kernel in Fourier space, computed via
//      the Lanczos approximation to ln(Gamma). Depends on cfg[j].derivative:
//        case 0: gl = exp(z*ln2) * Gamma((k+z)/2) / Gamma((3+k-z)/2)
//        case 1: gl = -(z-1) * exp((z-1)*ln2) * Gamma((k+z-1)/2) / Gamma((4+k-z)/2)
//        case 2: gl = (z-1)(z-2) * exp((z-2)*ln2) * Gamma((k+z-2)/2) / Gamma((5+k-z)/2)
//      where z = nu + i*eta_m[j][q] is the complex biasing parameter.
//
//      Optimization: only the first two ells (ks, ks+1) are computed from
//      the full Lanczos formula. Subsequent ells use the Gamma recurrence
//      relation Gamma(a+1) = a*Gamma(a), which gives:
//        gl[k+2] = gl[k] * (k + z + offset) / (k + 3 - z + offset)
//      This reduces O(BLOCK * N/2) Lanczos evaluations to O(2 * N/2)
//      plus O(BLOCK * N/2) complex multiplies - a major speedup since
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
// Cache invalidation:
// the static FFTW inverse plans (planb), the gl
// kernel array, the per-thread outfwd/outbcw buffers and base_j rebuild
// when NTHREADS, Nmax or SIZE2 change, when any N[j][2] changes, or when
// the block size ke - ks exceeds the cached one - not on every cosmology
// evaluation (the base_j values themselves are recomputed every call:
// x depends on the cosmology through chi_min/chi_max).
//
// Parameters:
//   x         - chi grid values, length Nx (log-spaced, as passed to p1)
//   Nx        - number of chi grid points
//   cfg       - FFTLog config per component (nu, derivative, N_pad)
//   LMAX      - highest multipole the y/Fy arrays hold (clips ke)
//   y         - output wavenumber grid y[SIZE1][LMAX][Nx]
//   Fy        - output projected functions Fy[SIZE1][SIZE2][LMAX][Nx]
//   toutfwd   - forward FFT coefficients from p1 [SIZE1*SIZE2][Nmax/2+1]
//   eta_m     - Fourier frequencies from p1 [SIZE2][Nmax/2+1]
//   N         - per-component sizes: N[j][0] = N_pad, N[j][1] = Nx,
//               N[j][2] = FFT size
//   Nmax      - max(N[j][2]) across all components
//   ks        - first multipole in this block (inclusive)
//   ke        - last multipole in this block (exclusive)
//   converged - per-row skip flags [SIZE1]: row i is skipped when
//               converged[i] = 1
//   active    - per-(row, component) activity mask [SIZE1][SIZE2], or
//               NULL for all active: an inactive (i, j) slot skips its
//               inverse FFTs and zero-fills Fy[i][j] for the block
//   SIZE1     - number of radial rows (bins)
//   SIZE2     - number of radial components per row (2 or 3)
//
// Returns:
//   nothing; y and Fy are filled for the block's multipoles
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
    const int* const converged,             // per-row skip flags [SIZE1]; 1 = skip row i
    const int* const* const active,         // activity mask [SIZE1][SIZE2]; NULL = all active
    int const SIZE1,                        // number of radial rows (bins)
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
  // Lanczos approximation to lnGamma, the standard g = 7, n = 9
  // coefficient set: with t = a + 6.5, i.e. t = (a-1) + g + 1/2,
  //
  //   lnGamma(a) = ln sqrt(2 pi) + (a - 1/2)*ln(t) - t
  //                + ln[ pfac[0] + sum_{w=1..8} pfac[w]/((a-1) + w) ]
  //
  // accurate to ~1e-15 for Re(a) >= 1/2; arguments left of that strip
  // go through the reflection formula instead (see the branch pair in
  // the cases below).
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

  // base_j = the log-grid origin shifted DOWN by the two N_pad guard
  // bands: x0 * exp(-2 * N_pad * dlnx). It enters only through the
  // phase exp(-i * eta_m * ln(base_j * y0)) applied to the forward
  // coefficients below, which rotates the circular convolution so the
  // physical (unpadded) window of the output lands at indices
  // N_pad..N_pad+Nx-1, where Fy is read. Recomputed every call: x
  // depends on the cosmology through chi_min/chi_max.
  for(int j=0; j<SIZE2; j++) {
    base_j[j] = x0 / exp(2 * N[j][0] * dlnx); // x depends on cosmo (chi_min/max)
  }

  // Output wavenumber grid, log-spaced like x but REVERSED relative to
  // it: y[i][k][q] = (k+1)/x[Nx-1-q], so q runs from
  // (k+1)/x_max up to (k+1)/x_min. The FFTLog convolution naturally
  // produces the transform on this reciprocal grid; (k+1) rescales it
  // per multipole.
  #pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<SIZE1; i++) {
    for(int q=0; q<Nx; q++) { // q < Nx
      for (int k=ks; k<kmax; k++) {
        y[i][k][q] = (k + 1.) / x[Nx -1 -q];
      }
    }
  } 

  // ---------------------------------------------------------------------------
  // Gamma function recurrence Gamma(a+1) = a*Gamma(a)
  //   For case 0:
  //     gl[k] = exp(z*ln2) * Gamma((k+z)/2) / Gamma((3+k-z)/2)
  //     When k -> k+2, both arguments shift by 1:
  //     Gamma((k+2+z)/2) = Gamma((k+z)/2 + 1) = (k+z)/2 * Gamma((k+z)/2)
  //     Gamma((5+k-z)/2) = Gamma((3+k-z)/2 + 1) = (3+k-z)/2 * Gamma((3+k-z)/2)
  //   So:
  //     gl[k+2] / gl[k] = ((k+z)/2) / ((3+k-z)/2) = (k+z) / (k+3-z)
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
            double complex part1; // lnGamma((k+z)/2)
            {
              const double complex a = 0.5*(k + z);
              // Re(a) < 1/2: reflection Gamma(a)*Gamma(1-a) =
              // pi/sin(pi a) in log form,
              //   lnGamma(a) = ln(pi) - ln(sin(pi a)) - lnGamma(1-a),
              // with lnGamma(1-a) from the direct Lanczos expression
              // below at argument 1-a (so (a-1) -> -a, t = -a + 7.5).
              if(creal(a) < 0.5) {
                double complex tmp = pfac[0];
                for(int w=1; w<9; w++) tmp += pfac[w] / ((-a) + w);
                const double complex t = (-a) + 7.5;
                part1 = clogpi - clog(csin(M_PI*a)) - 
                        (ln2pio2 + ((-a) + 0.5)*clog(t) - t + clog(tmp));
              }
              // Re(a) >= 1/2: direct Lanczos,
              //   lnGamma(a) = ln sqrt(2 pi) + (a-1/2)*ln(t) - t
              //                + ln(series), t = (a-1) + 7.5
              // (the coefficient set is documented at pfac above; the
              // same two branches repeat for every part1/part2 pair in
              // the cases below).
              else {
                double complex tmp = pfac[0];
                for(int w=1; w<9; w++) tmp += pfac[w] / ((a-1) + w);
                const double complex t = (a-1) + 7.5;
                part1 = ln2pio2 + ((a-1) + 0.5)*clog(t) - t + clog(tmp);
              }
            }
            double complex part2; // lnGamma((3+k-z)/2), same branches
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
        // the case-0 kernel with z -> z-1, so the same
        // Gamma(a+1) = a*Gamma(a) step gives
        // gl[k]/gl[k-2] = (k-3+z)/(k+2-z)
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
        // the case-0 kernel with z -> z-2, so the same
        // Gamma(a+1) = a*Gamma(a) step gives
        // gl[k]/gl[k-2] = (k-4+z)/(k+3-z)
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
  // Per-component table of x^nu on the reversed grid, precomputed so
  // the SIMD normalization below multiplies instead of calling pow()
  // per element (y^-nu = x^nu / (k+1)^nu on this grid).
  double** x_pow_nu = (double**) malloc2d(SIZE2, Nx);
  #pragma omp parallel for collapse(2) schedule(static)
  for (int j=0; j<SIZE2; j++) {
    for (int q=0; q<Nx; q++) {
      const double xr = x[Nx - 1 - q];
      x_pow_nu[j][q] = (cfg[j].nu == 1.0) ? xr : pow(xr, cfg[j].nu);
    }
  }

  for(int i=0; i<SIZE1; i++) {
    if (converged[i]) continue;
    #pragma omp parallel for collapse(2) schedule(static)
    for(int j=0; j<SIZE2; j++) {
      for (int k=ks; k<kmax; k++) { 
        if (active != NULL && !active[i][j]) { // zero kernel: no inverse FFT
          for (int q=0; q<Nx; q++) {
            Fy[i][j][k][q] = 0.0;
          }
          continue;
        }
#ifdef _OPENMP
        const int id = omp_get_thread_num(); 
#else
        const int id = 0;
#endif 
        const double lnbase = log(base_j[j] * y[i][k][0]);    
        // Explore the fact that the phase eta_m[j][q] is linear in q
        const double delta_phase = -eta_m[j][1] * lnbase;
        double step_re, step_im;
        cosmo_sincos(delta_phase, &step_im, &step_re);
        double phasor_re = 1.0; // Phasor starts at exp(i * 0) = 1 + 0i.
        double phasor_im = 0.0; // Phasor starts at exp(i * 0) = 1 + 0i.
        for(int q=0; q<(N[j][2]/2+1); q++) {
          fftw_complex val = toutfwd[i*SIZE2+j][q];
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
          val *= gl[j][k-ks][q];
          outfwd[id][q] = conj(val);
        }

        // FFTW's new-array execute interface: The plan was created with
        // outfwd[0]/outbcw[0], but each OpenMP thread executes it on its 
        // own buffers outfwd[id]/outbcw[id]. Do not replace this with
        // fftw_execute(planb[j]), which would use the plan's original arrays.
        fftw_execute_dft_c2r(planb[j], outfwd[id], outbcw[id]);

        // Normalization sqrt(pi)/4: the Mellin transform of the
        // spherical Bessel kernel is
        //
        //   integral_0^inf t^(z-1) j_l(t) dt
        //     = (sqrt(pi)/4) * 2^z * Gamma((l+z)/2) / Gamma((l+3-z)/2)
        //
        // (check l = 0, z = 1: integral j_0 = pi/2 = (sqrt(pi)/4)*2*
        // Gamma(1/2)). The 2^z * Gamma/Gamma part is carried by gl
        // above; the constant sqrt(pi)/4 is applied here. The
        // 1/N[j][2] normalizes FFTW's unnormalized c2r, and y^-nu
        // undoes the x^nu bias applied in p1. The output is read at
        // offset N[j][0], past the front guard band (see base_j).
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
      }
    }
  }
  free((void*) x_pow_nu);
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
// The last two terms correct for the difference between the linear power
// spectrum used in FFTLog and the nonlinear power spectrum used in the
// Limber integral. Both come from one batched call each
// (C_gg_tomo_limber_linpsopt_nointerp_ells at l = 0..LMAX_NOLIMBER-1).
//
// The FFTLog term needs separable growth, P(k; z1, z2) = D(z1)*D(z2)*
// P_lin(k, a_piv)/D(a_piv)^2 anchored per lens bin at
// a_piv = 1/(1 + zmean(bin)), so the subtracted Limber term uses the
// identical separable form; with P_lin(k, a) in the subtracted term the
// pair never cancels at high l. The scale-dependent part of the growth
// is carried by the Limber P_delta term.
//
// Size of the non-separability: CAMB's P_lin(k, a) has scale-dependent
// growth (massive neutrinos), and D^2 ratios differ from
// P_lin(k, a)/P_lin(k, a_piv) by 0.7% to 1.6% at z = 0.3 to 1 for the
// wavenumbers of l ~ 100.
//
// Algorithm overview:
//   1. Build the log-spaced chi grid (chi_min..chi_max, dimensionless)
//   2. Evaluate the three radial weight functions per lens bin:
//        fx[i][0] = chi * n(z) * D(a) * (H/H0) * b1   (galaxy density)
//        fx[i][1] = -chi * n(z) * D(a) * (H/H0) * f    (RSD velocity)
//        fx[i][2] = (W_mag / fK / coverH0^2) * D        (magnification)
//      The magnification component is skipped when bmag = 0 for all bins.
//      The RSD component is zero unless include_RSD_GG = 1 and the HOD is
//      off, the gate of the Limber terms it is paired with.
//      All three are nonzero on the Limber integration range of the lens
//      bin, z in [1/amax_lens - 1, 1/amin_lens - 1], the range of the two
//      Limber terms of step 6. With bmag != 0 that range reaches down to
//      the lowest source redshift, because W_mag has support in front of
//      the lens galaxies. On the n(z) support alone the FFTLog term would
//      miss the foreground magnification x magnification power that the
//      linear Limber term keeps, and the pair would not cancel at high l
//      (C_gs_tomo uses the same range for its lens rows).
//   3. Phase 1 (cfftlog_ells_p1): forward FFT of the radial functions
//      (ell-independent, done once)
//   4. Phase 2 (cfftlog_ells_p2): ell-dependent inverse transform,
//      called in blocks of BLOCK=16 multipoles with early termination
//   5. For each block, assemble the integrand:
//        F = Fy_density + Fy_RSD + bmag * l*(l+1) * Fy_mag / y^2
//      and integrate:
//        Cl_fftlog = (2/pi) * dlnk * sum_q[ F^2 * (k*c/H0)^3 * P_lin(k) ]
//   6. Combine: Cl = Cl_fftlog + Cl_limber(P_NL) - Cl_limber(P_lin)
//   7. Convergence: after each block, check |Cl_nolimber/Cl_limber - 1| < tol
//      at the block's last multipole. Once converged, fill the remaining
//      multipoles below LMAX_NOLIMBER with the Limber values (the batch
//      below LMIN_tab, the table C_gg_tomo_limber above).
//
// Only auto-correlations (ni = nj) are supported.
//
// Cache invalidation:
// the static work arrays (LMAX, x, fx, y, Fy, vres,
// CLnl, CLlin, lx) and the FFTLog configuration cfg rebuild when any of
// them is NULL or Ntable.random changes. Kernels and transforms are
// recomputed at every call (the caller caches the output instead).
//
// Parameters:
//   Cl  - output array Cl[nbins][LMAX_NOLIMBER]: non-Limber C_l per lens bin
//   tol - convergence tolerance for switching to Limber (typically 0.01)
//
// Returns:
//   nothing; the result is written into Cl
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// The FFTLog machinery of C_cl_tomo on runs of integer multipoles: run r
// covers l = runs[2r] .. runs[2r+1] - 1, the runs ascend, and none is longer
// than the block of 16. The early-exit test runs at the last multipole of
// each run, per lens bin, and the loop stops once every bin has converged.
//
// C_cl_tomo hands it blocks of 16 from 0 to LMAX_NOLIMBER - 1; the
// Fourier-space C_gg_tomo_ells only the integers next to its band centers,
// so the Limber terms, the inverse transforms and the y-sums run at those
// few multipoles instead of all of them.
//
// Parameters:
//   Cl    - output [clustering_nbin][>= LMAX_NOLIMBER]: the non-Limber C_l
//           at the run multipoles a bin reached before converging
//   LMAX  - output [clustering_nbin]: the end of the run in which the bin
//           converged (its freeze point), LMAX_NOLIMBER if it never did
//   tol   - early-exit tolerance on |C_l / C_l^limber(P_delta) - 1|;
//           tol <= 0 disables the exit (every run is computed)
//   runs  - the runs, [2*nruns] (see above)
//   nruns - number of runs
//
// Returns:
//   CLnl, the Limber C_l(P_delta) at the run multipoles, indexed
//   CLnl[i][l] (static: valid until the next call)
// ---------------------------------------------------------------------------
static double** C_cl_tomo_core(
    double* const* const Cl,  // output [nbins][>= LMAX_NOLIMBER]
    int* const LMAX,          // output [nbins], the freeze points
    const double tol,         // early-exit tolerance
    const int* const runs,    // runs of multipoles [2*nruns]
    const int nruns           // number of runs
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* x = NULL;
  static double*** fx= NULL;
  static double*** y = NULL;
  static double**** Fy = NULL;
  static double*** vres = NULL;
  static double** CLnl = NULL;
  static double** CLlin = NULL;
  static double* lx = NULL;
  static config cfg[3];
  
  const int nbins = redshift.clustering_nbin; 
  const int nchi  = Ntable.NL_Nchi;
  
  if (NULL == x ||  
      NULL == y || 
      NULL == Fy || 
      NULL == fx ||
      fdiff2(cache[0], Ntable.random))
  {
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
    if (CLnl != NULL) free((void*) CLnl);
    CLnl = (double**) malloc2d(nbins, limits.LMAX_NOLIMBER);
    if (CLlin != NULL) free((void*) CLlin);
    CLlin = (double**) malloc2d(nbins, limits.LMAX_NOLIMBER);
    if (lx != NULL) free((void*) lx);
    lx = (double*) malloc1d(limits.LMAX_NOLIMBER);
    // FFTLog computes the Bessel convolution of these integrals with a
    // circular FFT, and a circular FFT is periodic: whatever leaks past
    // one end of the chi array re-enters at the other (wrap-around
    // aliasing). The zero-padding N_pad is the guard band that absorbs
    // that leakage. What protects the integral is the guard band's
    // LOG-LENGTH N_pad*dlnchi, not its point count: the chi range is
    // fixed, so dlnchi shrinks like 1/nchi when the accuracy boost
    // raises nchi = Ntable.NL_Nchi, and a constant N_pad would shrink
    // the guard band until the long low-ell kernel tails wrap into the
    // non-Limber C_gg (measured on a 6x2pt data vector: chi2 shifts of
    // +27 at accuracy boost 5). Scaling N_pad with nchi keeps the guard
    // band at the log-length these constants gave at the unboosted
    // grid, nchi = 512, so results at accuracy boost 1 are unchanged.
    cfg[0].nu = 1.;
    cfg[0].c_window_width = 0.25;
    cfg[0].derivative = 0;
    cfg[0].N_pad = (long) ceil(200.0*nchi/512.0);
    // RSD
    cfg[1].nu = 1.01;
    cfg[1].c_window_width = 0.25;
    cfg[1].derivative = 2;
    cfg[1].N_pad = (long) ceil(500.0*nchi/512.0);
    // MAG
    cfg[2].nu = 1.;
    cfg[2].c_window_width = 0.25;
    cfg[2].derivative = 0;
    cfg[2].N_pad = (long) ceil(500.0*nchi/512.0);
    cache[0] = Ntable.random;
  }
  // chi() returns comoving distance in c/H0 units and real_coverH0 =
  // coverH0/h0 = 2997.92458/h0 is c/H0 in Mpc, so chi_min/chi_max are
  // in Mpc (matching the [Mpc^-2] kernels below). The fixed
  // z = 0.002..4.0 window defines the log-spaced FFTLog chi grid; it
  // covers the radial kernels' support from z ~ 0 up to the highest
  // lens redshifts.
  const double real_coverH0 = cosmology.coverH0/cosmology.h0;
  const double chi_min = chi(1./(1.0 + 0.002))*real_coverH0; // Mpc
  const double chi_max = chi(1./(1.0 + 4.0))*real_coverH0;   // Mpc
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
  }

  // Limber terms at the multipoles of the runs, batched in one pass: full
  // model (P_delta) and the linear counterpart of the FFTLog term. They
  // share the nodes, the radial weights and the RSD kernel (the a_chi and
  // W_RSD lookups that dominate the cost), which one call computes once
  // for both. Indexed CLnl[i][l], CLlin[i][l] (entries of other l are not
  // written).
  int nlx = 0;
  for (int r=0; r<nruns; r++) {
    for (int l=runs[2*r]; l<runs[2*r+1]; l++) {
      lx[nlx++] = (double) l;
    }
  }
  {
    double** tnl  = (double**) malloc2d(nbins, nlx);
    double** tlin = (double**) malloc2d(nbins, nlx);
    C_gg_tomo_limber_nl_lin_nointerp_ells(lx, nlx, nbins, tnl, tlin);
    for (int i=0; i<nbins; i++) {
      for (int m=0; m<nlx; m++) {
        const int l = (int) lx[m];
        CLnl[i][l]  = tnl[i][m];
        CLlin[i][l] = tlin[i][m];
      }
    }
    free((void*) tnl);
    free((void*) tlin);
  }

  double zlo[nbins]; // Limber lens range (see the header comment)
  double zhi[nbins];
  for (int i=0; i<nbins; i++) {
    zlo[i] = 1./amax_lens(i) - 1.;
    zhi[i] = 1./amin_lens(i) - 1.;
  }
  // The RSD slot follows the gate of the two Limber terms it is paired
  // with (C_gg_tomo_limber_linpsopt_nointerp_ells: include_RSD_GG and no
  // HOD); an RSD row the Limber terms lack would not cancel at high l
  const int rsd = (1 == include_RSD_GG && 0 == include_HOD_GX) ? 1 : 0;

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
      if (z < zlo[i] || z > zhi[i]) {
        fx[i][0][j] = 0.;
        fx[i][1][j] = 0.;
        fx[i][2][j] = 0.;
      }
      else {
        const double pf = nz_lens_photoz(z,i);
        const double WM = W_mag(a, fK, i);
        fx[i][0][j] =  chi*pf*D*hoverh0_a*gb1(z,i);
        fx[i][1][j] = (1 == rsd) ? -chi*pf*D*hoverh0_a*f : 0.0;
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

  // FKEM pivot: the separable linear spectrum is anchored per lens bin at
  // a_piv = 1/(1 + zmean(bin)), where the separable form is exact; the
  // residual of the separable-growth approximation then grows only across
  // the bin width instead of from z = 0 (it matters when growth is scale
  // dependent: massive neutrinos). growfac(1) = 1, so the
  // COSMO2D_FKEM_PIVOT_Z0 fallback reproduces the z = 0 anchor exactly.
  double apiv[nbins];
  double invgf2piv[nbins];
  for (int i=0; i<nbins; i++) {
#ifdef COSMO2D_FKEM_PIVOT_Z0
    apiv[i] = 1.0;
#else
    apiv[i] = 1.0/(1.0 + zmean(i));
#endif
    const double gfp = growfac(apiv[i]);
    invgf2piv[i] = 1.0/(gfp*gfp);
  }

  cfftlog_ells_p1((double* const) x, 
                  (double* const* const* const) fx, 
                  nchi, 
                  cfg, 
                  (fftw_complex* const* const) toutfwd,
                  (double* const* const) eta_m, 
                  N, 
                  Nmax, 
                  NULL, // SIZE2 already drops the mag slot when bmag = 0
                  nbins, 
                  SIZE2);

  const int BLOCK = 16;
  for (int r=0; r<nruns; r++) {
    if (runs[2*r] < 0 || runs[2*r+1] > limits.LMAX_NOLIMBER ||
        runs[2*r+1] <= runs[2*r] || runs[2*r+1] - runs[2*r] > BLOCK ||
        (r > 0 && runs[2*r] < runs[2*r-1])) {
      log_fatal("bad multipole run %d: [%d, %d)", r, runs[2*r], runs[2*r+1]);
      exit(1);
    }
  }
  int converged[nbins];
  for (int i=0; i<nbins; i++) {
    converged[i] = 0;
    LMAX[i] = limits.LMAX_NOLIMBER;
  } 
  int all_done = 0;

  for (int r=0; r<nruns && !all_done; r++)
  {
    const int ks = runs[2*r];
    const int ke = runs[2*r+1];

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
                     NULL, // all slots active (see the p1 call)
                     nbins, 
                     SIZE2);
    // With SIZE2 = 2 (every |gbmag| below the 1e-12 threshold) p2
    // never writes the magnification slot, so the static Fy[i][2]
    // still holds a previous SIZE2 = 3 call's values or
    // fresh-allocation garbage. The assembly below reads Fy[i][2]
    // unconditionally, and bmag can be tiny but nonzero, so the slot
    // is zeroed explicitly.
    if (0 != is_bmag_zero) {
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
          // bmag is per-bin only (BMAG_PER_BIN is the one supported
          // model, and gbmag ignores its z argument there), so the
          // z = 0 evaluation is exact at every chi node
          const double bmag = gbmag(0.,i);
          const double F = Fy[i][0][k][q] + Fy[i][1][k][q] + 
                           bmag*ell_prefactor*Fy[i][2][k][q]/(ty*ty);
          vres[i][k][q] = F*F*(k1cH0*k1cH0*k1cH0)*
                          p_lin(k1cH0, apiv[i])*invgf2piv[i];
        }
      }
      #pragma omp parallel for
      for (int k=ks; k<kk; k++) {  
        const double tcl = simd_array_sum(vres[i][k], nchi);
        Cl[i][k] = tcl * dlnk * 2. / M_PI + CLnl[i][k] - CLlin[i][k];
      }
      const int L = kk - 1; // check convergeence
      
      const double denom = CLnl[i][L];
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
  }

  free((void*) toutfwd);
  free((void*) eta_m);
  return CLnl;
}


// ---------------------------------------------------------------------------
void C_cl_tomo(
    double* const* const Cl,
    double tol
  )
{
  const int LNL   = limits.LMAX_NOLIMBER;
  const int nbins = redshift.clustering_nbin;
  const int BLOCK = 16; // C_cl_tomo_core's per-block arrays
  const int nruns = (LNL + BLOCK - 1)/BLOCK;
  int runs[2*nruns];
  for (int r=0; r<nruns; r++) {
    runs[2*r]   = r*BLOCK;
    runs[2*r+1] = ((r + 1)*BLOCK < LNL) ? (r + 1)*BLOCK : LNL;
  }
  int LMAX[nbins];
  double** CLnl = C_cl_tomo_core(Cl, LMAX, tol, runs, nruns);

  // Limber continuation of converged bins: the Limber-path values (the
  // batch up to LMIN_tab, the interpolation table above)
  for (int i=0; i<nbins; i++) {
    for (int k=LMAX[i]; k<LNL; k++) {
      Cl[i][k] = (k > limits.LMIN_tab) ? C_gg_tomo_limber(k, i, i) : CLnl[i][k];
    }
  }
}

// ---------------------------------------------------------------------------
// Galaxy clustering C_l^gg (auto spectra) at arbitrary multipoles, with the
// non-Limber correction below limits.LMAX_NOLIMBER. The Fourier-space data
// vectors call it (generic_interface.hpp, like.adopt_limber[LIMBER_GG] = 0):
// their multipoles like.ell are band centers, not integers. The gg
// counterpart of C_gs_tomo_ells.
//
// At a band center l this function returns the exact Limber value at l
// plus the non-Limber correction interpolated linearly between the two
// integers around l:
//
//   C(l)  = C^limber(l) + (1 - t)*dC(l0) + t*dC(l0 + 1)
//   dC(n) = C^nonlimber(n) - C^limber(n),   l0 = floor(l),  t = l - l0
//
// Everything is computed at the multipoles it is needed, never tabulated:
// C^limber at the band centers (C_gg_tomo_limber_nointerp_ells), and
// C^nonlimber and C^limber at the integers l0, l0 + 1 only, through
// C_cl_tomo_core (its Limber P_delta values there are exact too).
//
// No early exit: the core runs with tol = 0, so every bin gets its exact
// dC at every needed integer. The real-space early exit (freeze a bin once
// |dC/C^limber| < tol and take Limber from there) is a truncation error of
// that size, and at the band centers it is visible: with tol = 0.01 it
// moved the roman_kl 3x2pt chi2 by 1.9 and roman_fourier's by 0.10
// (delta^T C^-1 delta, measured 2026-10-01). With ~2 integers per band
// center below LMAX_NOLIMBER, computing them all costs little.
//
// Band centers with l < 1 or l >= LMAX_NOLIMBER - 1 keep the Limber value.
//
// Parameters:
//   ells  - multipole values, length nell (band centers; need not be integers)
//   nell  - number of multipole values
//   NSIZE - number of gg power spectra (= clustering_nbin)
//   out   - output [NSIZE][nell], indexed out[nz][i]
//
// Returns:
//   nothing; the result is written into out
// ---------------------------------------------------------------------------
void C_gg_tomo_ells(
    const double* ells,  // array of multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of gg power spectra (= clustering_nbin)
    double** out         // output [NSIZE][nell]
  )
{
  const int LNL   = limits.LMAX_NOLIMBER;
  const int BLOCK = 16; // C_cl_tomo_core's per-block arrays
  C_gg_tomo_limber_nointerp_ells(ells, nell, NSIZE, out);

  // the integers next to every corrected band center, as runs of
  // consecutive multipoles no longer than a block
  int need[LNL];
  for (int l=0; l<LNL; l++) {
    need[l] = 0;
  }
  for (int i=0; i<nell; i++) {
    if (ells[i] >= 1.0 && ells[i] < LNL - 1.0) {
      const int l0 = (int) floor(ells[i]);
      need[l0] = 1;
      need[l0 + 1] = 1;
    }
  }
  int runs[2*LNL];
  int nruns = 0;
  for (int l=0; l<LNL; l++) {
    if (!need[l]) continue;
    if (nruns > 0 && runs[2*nruns-1] == l &&
        runs[2*nruns-1] - runs[2*nruns-2] < BLOCK) {
      runs[2*nruns-1] = l + 1;  // extends the open run
    }
    else {
      runs[2*nruns]   = l;
      runs[2*nruns+1] = l + 1;
      nruns++;
    }
  }
  if (0 == nruns) {
    return;
  }

  double** Cnl = (double**) malloc2d(NSIZE, LNL);
  int LMAX[NSIZE];
  double** CLnl = C_cl_tomo_core(Cnl, LMAX, 0.0, runs, nruns);

  for (int nz=0; nz<NSIZE; nz++) {
    for (int i=0; i<nell; i++) {
      if (ells[i] >= 1.0 && ells[i] < LNL - 1.0) {
        const int l0 = (int) floor(ells[i]);
        const double t = ells[i] - l0;
        // (tol = 0: no bin freezes, LMAX[nz] = LMAX_NOLIMBER)
        const double d0 = (l0 < LMAX[nz]) ?
          Cnl[nz][l0] - CLnl[nz][l0] : 0.0;
        const double d1 = (l0 + 1 < LMAX[nz]) ?
          Cnl[nz][l0 + 1] - CLnl[nz][l0 + 1] : 0.0;
        out[nz][i] += (1.0 - t)*d0 + t*d1;
      }
    }
  }
  free((void*) Cnl);
}

// ---------------------------------------------------------------------------
// Non-Limber galaxy-galaxy lensing angular power spectrum C_l^gs at every
// integer multipole l < limits.LMAX_NOLIMBER, for every lens-source pair.
//
// The Limber approximation replaces the product of two spherical Bessel
// functions in the exact projection by a delta function at k*chi = l + 1/2.
// It fails at low l when the lens and source kernels overlap in redshift:
// the pairs with lens bin = source bin, and the pairs whose source bin lies
// in front of the lens bin (their signal is the intrinsic alignment of the
// sources times the lens density, two narrow kernels, as in C_gg). The
// exact projection uses the split of Fang, Krause, Eifler & MacCrann
// (arXiv:1911.11947, Sec. 4.2):
//
//   C_l = C_l^fftlog(P_lin) + C_l^limber(P_delta) - C_l^limber(P_lin)
//
// The first term is the exact projection of the linear power spectrum with
// separable growth, P(k; z1, z2) = D(z1)*D(z2)/D(z_piv)^2 *
// P_lin(k, z_piv) anchored per lens bin at z_piv = zmean(bin). The last two
// terms add in Limber what linear theory misses (nonlinear P_delta, one-loop
// galaxy bias, TATT terms beyond the linear amplitude C1). The third term
// uses the same separable spectrum as the first, so at high l the two
// converge to the same number and cancel, and C_l returns to the Limber
// C_l^limber(P_delta).
//
// EXACT (FFTLOG) TERM:
//
//   C_l^fftlog = (2/pi) * INT dlnk k^3 P_lin(k, a_piv)/D(a_piv)^2 *
//                F_lens(k) * F_src(k)
//
//   F_lens(k) = INT dlnchi fx_dens(chi) j_l(k chi)
//             + INT dlnchi fx_rsd(chi) j_l''(k chi)
//             + bmag * l(l+1)/k^2 * INT dlnchi fx_mag(chi) j_l(k chi)
//   F_src(k)  = sqrt((l-1) l (l+1) (l+2))/k^2 * INT dlnchi fx_src(chi) j_l(k chi)
//
//   with the radial kernels (chi in c/H0 units, W_* from radial_weights.c)
//     fx_dens =  chi * n_lens(z) * D * (H/H0) * b1
//     fx_rsd  = -chi * n_lens(z) * D * (H/H0) * f      (zero unless include_RSD_GS)
//     fx_mag  = (W_mag / chi) * D
//     fx_src  = ((W_kappa - W_source*C1) / chi) * D
//
//   Lensing and the linear (NLA) intrinsic alignment share the spin-2 radial
//   structure j_l(k chi)/(k chi)^2, so one source kernel carries both: the
//   exact counterpart of the (WK - WS*C1) factor of the NLA integrand core.
//   The IA piece does not grow with D because C1 carries 1/D. Replacing j_l
//   in F_src by the Limber delta function gives back the Limber prefactor
//   sqrt((l-1) l (l+1) (l+2))/(l + 1/2)^2.
//
// ALGORITHM:
//   1. Radial kernels on a log-spaced chi grid (z = 0.002 to 4, Ntable.NL_Nchi
//      points). Rows 0..nlens-1 are the lens bins (slots density, RSD,
//      magnification, as in C_cl_tomo); rows nlens..nlens+nsrc-1 are the
//      source bins (slot 2 only; slots 0 and 1 stay zero).
//   2. The two Limber terms at l = 0..LMAX_NOLIMBER-1 with the batched
//      integrator of the Limber path, one pass for both
//      (C_gs_tomo_limber_nl_lin_nointerp_ells: P_delta and the separable
//      linear spectrum). The P_delta term is therefore bit-identical to what
//      w_gammat_tomo uses when like.adopt_limber[LIMBER_GS] = 1.
//   3. cfftlog_ells_p1: forward FFT of every row (independent of l, done once).
//   4. Blocks of BLOCK = 16 multipoles: cfftlog_ells_p2 computes the Hankel
//      transforms of every active row; k^3 P_lin(k) and 1/y^2 are tabulated
//      once per (l, y) and shared by all pairs (y is the same for every row);
//      each pair sums F_lens*F_src*k^3*P_lin over the y grid.
//   5. Early exit: at the last multipole L of each block, a pair with
//      |C_L / C_L^limber(P_delta) - 1| < tol is frozen. A radial row is
//      skipped in later blocks once every pair that uses it is frozen.
//   6. The multipoles of a frozen pair, from its freeze point up to
//      LMAX_NOLIMBER - 1, take the Limber-path values: the batch below
//      LMIN_tab, the interpolation table (C_gs_tomo_limber_fill) above.
//
//   Steps 1-5 live in C_gs_tomo_core, which works on runs of integer
//   multipoles: this function hands it blocks of 16 from 0 to
//   LMAX_NOLIMBER - 1 (the loop above), the Fourier-space C_gs_tomo_ells
//   only the integers next to its band centers.
//
//   Example (lsst_y1 3x2pt fiducial, 25 pairs, tol = 0.01): the 10 pairs with
//   the lens bin in front of the source bin freeze at l = 16 or 32; the 5
//   pairs with lens bin = source bin freeze between l = 32 and l = 128; 8 of
//   the 10 pairs with the source bin in front never freeze and run to
//   l = 149.
//
// DIFFERENCES FROM C_cl_tomo (the galaxy clustering counterpart):
//   - Lens rows are nonzero on the Limber integration range of the lens bin,
//     z in [1/amax_lens - 1, 1/amin_lens - 1], as in C_cl_tomo. With
//     bmag != 0 this range reaches down to the lowest source redshift. On
//     the n(z) support alone the magnification row would miss the
//     foreground that the Limber terms keep, and the pair would never
//     cancel.
//   - Convergence is tested per lens-source pair, not per bin.
//   - SIZE2 = 3 always: the broad source kernel uses slot 2, which has the
//     wide N_pad guard band. When C_cl_tomo runs in the same evaluation with
//     SIZE2 = 2 (all bmag = 0), the FFTW plans cached inside
//     cfftlog_ells_p1/p2 are rebuilt at each switch between the two calls.
//
// The unit test tests/test_nonlimber_ggl.py of every project compares the
// data vectors with and without this function.
//
// Cache invalidation:
// the chi grid, work arrays and FFT configuration are
// static and rebuilt when Ntable.random, the number of radial rows
// (nlens + nsrc), or tomo.ggl_Npowerspectra change. Kernels and transforms
// are recomputed at every call (the caller, w_gammat_tomo, caches its
// output instead).
//
// Parameters:
//   Cl  - output [ggl_Npowerspectra][>= LMAX_NOLIMBER], indexed Cl[nz][l];
//         pair nz is (ZL(nz), ZS(nz)); Cl[nz][0] = Cl[nz][1] = 0 (spin 2)
//   tol - early-exit tolerance (w_gammat_tomo passes 0.01); tol <= 0
//         disables the exit, so every pair runs through LMAX_NOLIMBER - 1
//         (the reference path for debugging)
//
// Returns:
//   nothing; the result is written into Cl
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// The FFTLog machinery of C_gs_tomo (steps 1-5 of its header) on runs of
// integer multipoles: run r covers l = runs[2r] .. runs[2r+1] - 1, the runs
// ascend, and none is longer than the block of 16 the per-block arrays
// hold. The early-exit test runs at the last multipole of each run, and the
// loop stops once every pair has converged.
//
// C_gs_tomo hands it blocks of 16 from 0 to LMAX_NOLIMBER - 1; the
// Fourier-space C_gs_tomo_ells only the integers next to its band centers,
// so the Limber terms, the inverse transforms and the y-sums run at those
// few multipoles instead of all of them.
//
// Parameters:
//   Cl    - output [ggl_Npowerspectra][>= LMAX_NOLIMBER]: the non-Limber C_l
//           at the run multipoles a pair reached before converging
//   LMAX  - output [ggl_Npowerspectra]: the end of the run in which the pair
//           converged (its freeze point), LMAX_NOLIMBER if it never did
//   tol   - early-exit tolerance on |C_l / C_l^limber(P_delta) - 1|;
//           tol <= 0 disables the exit (every run is computed)
//   runs  - the runs, [2*nruns] (see above)
//   nruns - number of runs
//
// Returns:
//   CLnl, the Limber C_l(P_delta) at the run multipoles, indexed
//   CLnl[nz][l] (static: valid until the next call)
// ---------------------------------------------------------------------------
static double** C_gs_tomo_core(
    double* const* const Cl,  // output [NSIZE][>= LMAX_NOLIMBER]
    int* const LMAX,          // output [NSIZE], the freeze points
    const double tol,         // early-exit tolerance
    const int* const runs,    // runs of multipoles [2*nruns]
    const int nruns           // number of runs
  )
{
  halo_IA_unsupported("C_gs_tomo");
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* x = NULL;
  static double*** fx = NULL;
  static double*** y = NULL;
  static double**** Fy = NULL;
  static double** CLnl = NULL;
  static double** CLlin = NULL;
  static double* lx = NULL;
  static config cfg[3];

  const int nlens = redshift.clustering_nbin;
  const int nsrc  = redshift.shear_nbin;
  const int SIZE1 = nlens + nsrc;
  const int SIZE2 = 3;
  const int NSIZE = tomo.ggl_Npowerspectra;
  const int nchi  = Ntable.NL_Nchi;
  const int LNL   = limits.LMAX_NOLIMBER;

  if (NSIZE <= 0) {
    log_fatal("ggl requested but tomo.ggl_Npowerspectra == %d", NSIZE);
    exit(1);
  }
  if (NULL == x ||
      NULL == y ||
      NULL == Fy ||
      NULL == fx ||
      fdiff2(cache[0], Ntable.random) ||
      cache[1] != (uint64_t) SIZE1 ||
      cache[2] != (uint64_t) NSIZE)
  {
    if (x != NULL) free((void*) x);
    x = (double*) malloc1d(nchi);
    if (y != NULL) free((void*) y);
    y = (double***) malloc3d(SIZE1, LNL, nchi);
    if (Fy != NULL) free((void*) Fy);
    Fy = (double****) malloc4d(SIZE1, SIZE2, LNL, nchi);
    if (fx != NULL) free((void*) fx);
    fx = (double***) malloc3d(SIZE1, SIZE2, nchi);
    if (CLnl != NULL) free((void*) CLnl);
    CLnl = (double**) malloc2d(NSIZE, LNL);
    if (CLlin != NULL) free((void*) CLlin);
    CLlin = (double**) malloc2d(NSIZE, LNL);
    if (lx != NULL) free((void*) lx);
    lx = (double*) malloc1d(LNL);
    // The same three configurations as C_cl_tomo, so the p1/p2 plan
    // caches are shared with the gg calls. N_pad scales with nchi because
    // the circular FFT is periodic and the zero-padded guard band absorbs
    // its wrap-around leakage; what protects the integral is the band's
    // LOG-length N_pad*dlnchi. The chi range is fixed, so dlnchi shrinks
    // like 1/nchi when the accuracy boost raises nchi = Ntable.NL_Nchi,
    // and a constant N_pad would shrink the band until the long low-ell
    // kernel tails wrap into the output. These constants reproduce the
    // guard band of the unboosted grid, nchi = 512.
    cfg[0].nu = 1.;
    cfg[0].c_window_width = 0.25;
    cfg[0].derivative = 0;
    cfg[0].N_pad = (long) ceil(200.0*nchi/512.0);
    // RSD
    cfg[1].nu = 1.01;
    cfg[1].c_window_width = 0.25;
    cfg[1].derivative = 2;
    cfg[1].N_pad = (long) ceil(500.0*nchi/512.0);
    // MAG (lens rows) and lensing + IA (source rows)
    cfg[2].nu = 1.;
    cfg[2].c_window_width = 0.25;
    cfg[2].derivative = 0;
    cfg[2].N_pad = (long) ceil(500.0*nchi/512.0);
    cache[0] = Ntable.random;
    cache[1] = (uint64_t) SIZE1;
    cache[2] = (uint64_t) NSIZE;
  }
  // chi in c/H0 units times real_coverH0 = c/H0 in Mpc gives Mpc; the
  // z = 0.002..4.0 window is the same FFTLog chi grid as C_cl_tomo
  // (the units note lives there).
  const double real_coverH0 = cosmology.coverH0/cosmology.h0;
  const double chi_min = chi(1./(1.0 + 0.002))*real_coverH0; // Mpc
  const double chi_max = chi(1./(1.0 + 4.0))*real_coverH0;   // Mpc
  const double dlnchi  = log(chi_max/chi_min) / ((double) nchi - 1.0);
  const double dlnk    = dlnchi;
  { // INIT: make sure static init variables inside the funcs are defined
    const double chi = chi_min/real_coverH0;
    const double a   = a_chi(chi);
    const double z   = 1. / a - 1.;
    const double fK  = chi;
    const double hoh0 = hoverh0(a);
    const double D   = growfac_all(a).D;
    for (int i=0; i<nlens; i++) {
      (void) nz_lens_photoz(z,i);
      (void) W_mag(a,fK,i);
      (void) gb1(z,i);
      (void) gbmag(0.,i);
    }
    for (int j=0; j<nsrc; j++) {
      (void) W_kappa(a, fK, j);
      (void) W_source(a, j, hoh0);
      (void) IA_A1_Z1(a, D, j);
    }
    (void) p_lin(0.1, 1.0);
    (void) ZL(0);
    (void) ZS(0);
  }

  // Limber terms at the multipoles of the runs, in one pass: full model
  // (P_delta) and the linear counterpart of the FFTLog term (P_lin), which
  // share the nodes, the weights and the RSD kernel. Indexed CLnl[nz][l],
  // CLlin[nz][l] (entries of other l are not written).
  int nlx = 0;
  for (int r=0; r<nruns; r++) {
    for (int l=runs[2*r]; l<runs[2*r+1]; l++) {
      lx[nlx++] = (double) l;
    }
  }
  {
    double** tnl  = (double**) malloc2d(NSIZE, nlx);
    double** tlin = (double**) malloc2d(NSIZE, nlx);
    C_gs_tomo_limber_nl_lin_nointerp_ells(lx, nlx, NSIZE, tnl, tlin);
    for (int nz=0; nz<NSIZE; nz++) {
      for (int m=0; m<nlx; m++) {
        const int l = (int) lx[m];
        CLnl[nz][l]  = tnl[nz][m];
        CLlin[nz][l] = tlin[nz][m];
      }
    }
    free((void*) tnl);
    free((void*) tlin);
  }

  double zlo[nlens]; // Limber lens range (see the header comment)
  double zhi[nlens];
  for (int i=0; i<nlens; i++) {
    zlo[i] = 1./amax_lens(i) - 1.;
    zhi[i] = 1./amin_lens(i) - 1.;
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
    for (int i=0; i<nlens; i++) {
      if (z < zlo[i] || z > zhi[i]) {
        fx[i][0][j] = 0.;
        fx[i][1][j] = 0.;
        fx[i][2][j] = 0.;
      }
      else {
        const double pf = nz_lens_photoz(z,i);
        const double WM = W_mag(a, fK, i);
        fx[i][0][j] = chi*pf*D*hoverh0_a*gb1(z,i);
        fx[i][1][j] = (1 == include_RSD_GS) ? -chi*pf*D*hoverh0_a*f : 0.0;
        fx[i][2][j] = (WM/fK/(real_coverH0*real_coverH0))*D; // [Mpc^-2]
      }
    }
    for (int js=0; js<nsrc; js++) {
      const int i = nlens + js;
      const double WK = W_kappa(a, fK, js);
      const double WS = W_source(a, js, hoverh0_a);
      const double C1 = IA_A1_Z1(a, D, js);
      fx[i][0][j] = 0.;
      fx[i][1][j] = 0.;
      fx[i][2][j] = ((WK - WS*C1)/fK/(real_coverH0*real_coverH0))*D; // [Mpc^-2]
    }
  }

  int Nmax = 0;
  int N[SIZE2][3];
  for(int j=0; j<SIZE2; j++) {
    N[j][0] = cfg[j].N_pad;
    N[j][1] = nchi;
    // FFT-friendly even size: only prime factors 2, 3, 5, 7, so FFTW
    // uses its fast small-radix codelets; the extra elements are
    // zero-padded in cfftlog_ells_p1 and do not affect the result
    long raw = 2*N[j][0] + N[j][1];
    if (raw % 2) raw++;
    N[j][2] = (int) next_fft_size(raw);
    if (N[j][2] % 2) {
      N[j][2] = (int) next_fft_size(N[j][2] + 1);
    }
    if (N[j][2] > Nmax)
      Nmax = N[j][2];
  }

  // Per-(row, slot) activity mask: most slots hold identically zero
  // kernels (see the fx assembly above), and cfftlog would transform
  // zeros. Lens rows: density always; RSD only when include_RSD_GS;
  // magnification only when some gbmag != 0 (the kernel is nonzero
  // either way, but the Cl sum multiplies it by bmag). Source rows:
  // the combined lensing + IA kernel lives in slot 2 alone.
  int gs_bmag_zero = 1;
  for (int i=0; i<nlens; i++) {
    if (gbmag(0., i) != 0) {
      gs_bmag_zero = 0;
    }
  }
  int** active = (int**) malloc2d_int(SIZE1, SIZE2);
  for (int i=0; i<nlens; i++) {
    active[i][0] = 1;
    active[i][1] = (1 == include_RSD_GS) ? 1 : 0;
    active[i][2] = (0 == gs_bmag_zero) ? 1 : 0;
  }
  for (int js=0; js<nsrc; js++) {
    active[nlens + js][0] = 0;
    active[nlens + js][1] = 0;
    active[nlens + js][2] = 1;
  }

  fftw_complex** toutfwd = (fftw_complex**) malloc2d_fftwc(SIZE1*SIZE2, Nmax/2+1);

  double** eta_m = (double**) malloc2d(SIZE2, Nmax/2+1);

  cfftlog_ells_p1((double* const) x,
                  (double* const* const* const) fx,
                  nchi,
                  cfg,
                  (fftw_complex* const* const) toutfwd,
                  (double* const* const) eta_m,
                  N,
                  Nmax,
                  (const int* const* const) active,
                  SIZE1,
                  SIZE2);

  const int BLOCK = 16;
  for (int r=0; r<nruns; r++) {
    if (runs[2*r] < 0 || runs[2*r+1] > LNL || runs[2*r+1] <= runs[2*r] ||
        runs[2*r+1] - runs[2*r] > BLOCK ||
        (r > 0 && runs[2*r] < runs[2*r-1])) {
      log_fatal("bad multipole run %d: [%d, %d)", r, runs[2*r], runs[2*r+1]);
      exit(1);
    }
  }
  double*** PK = (double***) malloc3d(redshift.clustering_nbin, BLOCK,
                                      nchi); // per lens bin (pivot spectrum)
  // FKEM pivot per lens bin, as in C_cl_tomo (see the note there);
  // COSMO2D_FKEM_PIVOT_Z0 restores the z = 0 anchor.
  double apivL[redshift.clustering_nbin];
  double invgf2L[redshift.clustering_nbin];
  for (int i=0; i<redshift.clustering_nbin; i++) {
#ifdef COSMO2D_FKEM_PIVOT_Z0
    apivL[i] = 1.0;
#else
    apivL[i] = 1.0/(1.0 + zmean(i));
#endif
    const double gfp = growfac(apivL[i]);
    invgf2L[i] = 1.0/(gfp*gfp);
  }
  double** IY2 = (double**) malloc2d(BLOCK, nchi); // 1/y^2
  int converged[NSIZE]; // per lens-source pair
  int row_done[SIZE1];  // per radial row: every pair using it converged
  for (int nz=0; nz<NSIZE; nz++) {
    converged[nz] = 0;
    LMAX[nz] = LNL;
  }
  int all_done = 0;

  for (int r=0; r<nruns && !all_done; r++)
  {
    const int ks = runs[2*r];
    const int ke = runs[2*r+1];

    for (int i=0; i<SIZE1; i++) {
      row_done[i] = 1;
    }
    for (int nz=0; nz<NSIZE; nz++) {
      if (!converged[nz]) {
        row_done[ZL(nz)] = 0;
        row_done[nlens + ZS(nz)] = 0;
      }
    }

    cfftlog_ells_p2((double* const) x,
                     nchi,
                     cfg,
                     LNL,
                     (double* const* const* const) y,
                     (double* const* const* const* const) Fy,
                     (fftw_complex* const* const) toutfwd,
                     (double* const* const) eta_m,
                     N,
                     Nmax,
                     ks,
                     ke,
                     row_done,
                     (const int* const* const) active,
                     SIZE1,
                     SIZE2);

    // y does not depend on the row, so P_lin is evaluated once per
    // (lens bin, l, q) and shared by every source pair of that lens bin
    // (the pivot spectrum differs per lens bin).
    #pragma omp parallel for collapse(2) schedule(static)
    for (int k=ks; k<ke; k++) {
      for (int q=0; q<nchi; q++) {
        const double ty = y[0][k][q];
        IY2[k-ks][q] = 1.0/(ty*ty);
      }
    }
    #pragma omp parallel for collapse(3) schedule(static)
    for (int zl=0; zl<redshift.clustering_nbin; zl++) {
      for (int k=ks; k<ke; k++) {
        for (int q=0; q<nchi; q++) {
          const double k1cH0 = y[0][k][q]*real_coverH0;
          PK[zl][k-ks][q] = (k1cH0*k1cH0*k1cH0)*
                            p_lin(k1cH0, apivL[zl])*invgf2L[zl];
        }
      }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int k=ks; k<ke; k++) {
        if (converged[nz]) continue;
        if (k < 2) { // spin-2: C_gs(l < 2) = 0
          Cl[nz][k] = 0.0;
          continue;
        }
        const int zl = ZL(nz);
        const int zs = ZS(nz);
        const double bmag = gbmag(0.,zl);
        const double lp   = k*(k + 1.);
        const double ep2x = sqrt((k - 1.)*k*(k + 1.)*(k + 2.));
        const double* restrict fd  = Fy[zl][0][k];
        const double* restrict fr  = Fy[zl][1][k];
        const double* restrict fm  = Fy[zl][2][k];
        const double* restrict fs  = Fy[nlens + zs][2][k];
        const double* restrict pk  = PK[zl][k-ks];
        const double* restrict iy2 = IY2[k-ks];
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int q=0; q<nchi; q++) {
          const double FL = fd[q] + fr[q] + bmag*lp*fm[q]*iy2[q];
          sum += FL*fs[q]*iy2[q]*pk[q];
        }
        Cl[nz][k] = ep2x*sum*dlnk*2./M_PI + CLnl[nz][k] - CLlin[nz][k];
      }
    }

    for (int nz=0; nz<NSIZE; nz++) { // check convergence
      if (converged[nz]) continue;
      const int L = ke - 1;
      const double denom = CLnl[nz][L];
      if (fabs(denom) > 1e-300) {
        const double dev = Cl[nz][L] / denom - 1.0;
        if (isfinite(dev) && fabs(dev) < tol) {
          converged[nz] = 1;
          LMAX[nz] = ke;
        }
      }
    }

    all_done = 1;
    for (int nz=0; nz<NSIZE; nz++) {
      if (!converged[nz]) all_done = 0;
    }
  }

  free((void*) active);
  free((void*) toutfwd);
  free((void*) eta_m);
  free((void*) PK);
  free((void*) IY2);
  return CLnl;
}

// ---------------------------------------------------------------------------
void C_gs_tomo(
    double* const* const Cl,
    double tol
  )
{
  const int LNL   = limits.LMAX_NOLIMBER;
  const int NSIZE = tomo.ggl_Npowerspectra;
  const int BLOCK = 16; // C_gs_tomo_core's per-block arrays
  static double* lnell = NULL;
  static int lnell_n = 0;
  if (NULL == lnell || lnell_n != LNL) {
    if (lnell != NULL) free((void*) lnell);
    lnell = (double*) malloc1d(LNL);
    for (int l=0; l<LNL; l++) {
      lnell[l] = (l > 0) ? log((double) l) : 0.0;
    }
    lnell_n = LNL;
  }
  const int nruns = (LNL + BLOCK - 1)/BLOCK;
  int runs[2*nruns];
  for (int r=0; r<nruns; r++) {
    runs[2*r]   = r*BLOCK;
    runs[2*r+1] = ((r + 1)*BLOCK < LNL) ? (r + 1)*BLOCK : LNL;
  }
  int LMAX[NSIZE];
  double** CLnl = C_gs_tomo_core(Cl, LMAX, tol, runs, nruns);

  // Limber continuation of converged pairs: the same values the Limber path
  // of w_gammat_tomo writes (batch below LMIN_tab, table above). The
  // table is built here, single-threaded, before the parallel region reads
  // it.
  (void) C_gs_tomo_limber((double) limits.LMIN_tab + 1, ZL(0), ZS(0));
  #pragma omp parallel for schedule(static)
  for (int nz=0; nz<NSIZE; nz++) {
    const int lo = (LMAX[nz] > limits.LMIN_tab) ? LMAX[nz] : limits.LMIN_tab;
    for (int k=LMAX[nz]; k<lo && k<LNL; k++) {
      Cl[nz][k] = CLnl[nz][k];
    }
    if (lo < LNL) {
      C_gs_tomo_limber_fill(nz, lo, LNL, lnell, Cl[nz]);
    }
  }
}

// ---------------------------------------------------------------------------
// Galaxy-galaxy lensing C_l^gs at arbitrary multipoles, with the non-Limber
// correction below limits.LMAX_NOLIMBER. The Fourier-space data vectors call
// it (generic_interface.hpp, like.adopt_limber[LIMBER_GS] = 0): their
// multipoles like.ell are band centers, not integers.
//
// At a band center l this function returns the exact Limber value at l plus
// the non-Limber correction interpolated linearly between the two integers
// around l:
//
//   C(l)  = C^limber(l) + (1 - t)*dC(l0) + t*dC(l0 + 1)
//   dC(n) = C^nonlimber(n) - C^limber(n),   l0 = floor(l),  t = l - l0
//
// Everything is computed at the multipoles it is needed, never tabulated:
// C^limber at the band centers (C_gs_tomo_limber_nointerp_ells), and
// C^nonlimber and C^limber at the integers l0, l0 + 1 only, through
// C_gs_tomo_core (its Limber P_delta values there are exact too).
//
// No early exit: the core runs with tol = 0, so every pair gets its exact
// dC at every needed integer (see C_gg_tomo_ells for why: the real-space
// early exit is a truncation error that the band centers see).
//
// Example: l = 35.4 gives l0 = 35, t = 0.4 and
//   C(35.4) = C^limber(35.4) + 0.6*dC(35) + 0.4*dC(36).
//
// Band centers with l < 2 (C^gs vanishes at l = 0, 1 for a spin-2 field)
// or l >= LMAX_NOLIMBER - 1 keep the Limber value.
//
// Parameters:
//   ells  - multipole values, length nell (band centers; need not be integers)
//   nell  - number of multipole values
//   NSIZE - number of ggl power spectra (= tomo.ggl_Npowerspectra)
//   out   - output [NSIZE][nell], indexed out[nz][i]; pair nz is (ZL(nz), ZS(nz))
//
// Returns:
//   nothing; the result is written into out
// ---------------------------------------------------------------------------
void C_gs_tomo_ells(
    const double* ells,  // array of multipole values (length nell)
    const int nell,      // number of multipole values
    const int NSIZE,     // number of ggl power spectra
    double** out         // output [NSIZE][nell]
  )
{
  const int LNL   = limits.LMAX_NOLIMBER;
  const int BLOCK = 16; // C_gs_tomo_core's per-block arrays
  C_gs_tomo_limber_nointerp_ells(ells, nell, NSIZE, out);

  // the integers next to every corrected band center, as runs of
  // consecutive multipoles no longer than a block
  int need[LNL];
  for (int l=0; l<LNL; l++) {
    need[l] = 0;
  }
  for (int i=0; i<nell; i++) {
    if (ells[i] >= 2.0 && ells[i] < LNL - 1.0) {
      const int l0 = (int) floor(ells[i]);
      need[l0] = 1;
      need[l0 + 1] = 1;
    }
  }
  int runs[2*LNL];
  int nruns = 0;
  for (int l=0; l<LNL; l++) {
    if (!need[l]) continue;
    if (nruns > 0 && runs[2*nruns-1] == l &&
        runs[2*nruns-1] - runs[2*nruns-2] < BLOCK) {
      runs[2*nruns-1] = l + 1;  // extends the open run
    }
    else {
      runs[2*nruns]   = l;
      runs[2*nruns+1] = l + 1;
      nruns++;
    }
  }
  if (0 == nruns) {
    return;
  }

  double** Cnl = (double**) malloc2d(NSIZE, LNL);
  int LMAX[NSIZE];
  double** CLnl = C_gs_tomo_core(Cnl, LMAX, 0.0, runs, nruns);

  for (int nz=0; nz<NSIZE; nz++) {
    for (int i=0; i<nell; i++) {
      if (ells[i] >= 2.0 && ells[i] < LNL - 1.0) {
        const int l0 = (int) floor(ells[i]);
        const double t = ells[i] - l0;
        // (tol = 0: no pair freezes, LMAX[nz] = LMAX_NOLIMBER)
        const double d0 = (l0 < LMAX[nz]) ?
          Cnl[nz][l0] - CLnl[nz][l0] : 0.0;
        const double d1 = (l0 + 1 < LMAX[nz]) ?
          Cnl[nz][l0 + 1] - CLnl[nz][l0 + 1] : 0.0;
        out[nz][i] += (1.0 - t)*d0 + t*d1;
      }
    }
  }
  free((void*) Cnl);
}
