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
// Legendre sums with a multipole filter, for every spectrum nz and theta
// bin i:
//
//   w_vec[nz*ntheta + i] = sum_{l=lmin}^{lmax-1} (Pl[i][l]*filter[l])*Cl[nz][l]
//
// The filtered counterpart of legendre_sums (cosmo2D.c; the reasons for
// the grouping are given there): 4 spectra and 4 theta bins per pass over
// l. The product keeps the order of the reference loop, (Pl*filter)*Cl,
// and every (nz, i) keeps its own sum, so the results are bitwise those
// of the reference (COSMO2D_NOT_USE_SIMD). Pl*filter is formed once per
// theta bin of the group instead of once per (nz, i).
//
// Thread safety: call outside parallel regions.
//
// Parameters:
//   NSIZE  - number of spectra (rows of Cl)
//   ntheta - number of theta bins (rows of Pl)
//   lmin   - first multipole of the sums
//   lmax   - one past the last multipole of the sums
//   Pl     - [ntheta][lmax] bin-averaged Legendre kernel
//   filter - [lmax] multipole filter (the CMB beam/filter)
//   Cl     - [NSIZE][lmax] spectra at every integer l
//   w_vec  - output [NSIZE*ntheta], indexed nz*ntheta + i
// ---------------------------------------------------------------------------
static void legendre_sums_filtered(
    const int NSIZE,
    const int ntheta,
    const int lmin,
    const int lmax,
    double** Pl,
    const double* filter,
    double** Cl,
    double* w_vec
  )
{
#ifdef COSMO2D_NOT_USE_SIMD
  #pragma omp parallel for collapse(2) schedule(static)
  for (int nz=0; nz<NSIZE; nz++) {
    for (int i=0; i<ntheta; i++) {
      // Local restrict pointers: without these, GCC cannot prove the
      // Pl, filter and Cl rows don't alias (pointer-to-pointer
      // indirection inside a collapse(2) OpenMP region) and gives up
      // on the SIMD reduction below
      const double* restrict c0 = Cl[nz];
      const double* restrict cf = filter;
      const double* restrict g0 = Pl[i];
      double sum = 0.0;
      #pragma omp simd reduction(+:sum)
      for (int l=lmin; l<lmax; l++) {
        sum += g0[l] * cf[l] * c0[l];
      }
      w_vec[nz*ntheta + i] = sum;
    }
  }
#else
  // nz and i are the first spectrum and the first theta bin of the group,
  // which covers spectra nz .. nz+3 and theta bins i .. i+3
  #pragma omp parallel for collapse(2) schedule(static)
  for (int nz=0; nz<NSIZE; nz+=4) {
    for (int i=0; i<ntheta; i+=4) {
      // past the end: repeats of the last valid spectrum / theta bin,
      // computed but never stored (see legendre_sums in cosmo2D.c)
      const int nz1 = (nz + 1 < NSIZE) ? nz + 1 : NSIZE - 1;
      const int nz2 = (nz + 2 < NSIZE) ? nz + 2 : NSIZE - 1;
      const int nz3 = (nz + 3 < NSIZE) ? nz + 3 : NSIZE - 1;
      const int i1  = (i + 1 < ntheta) ? i + 1 : ntheta - 1;
      const int i2  = (i + 2 < ntheta) ? i + 2 : ntheta - 1;
      const int i3  = (i + 3 < ntheta) ? i + 3 : ntheta - 1;

      const double* restrict cf  = filter;
      const double* restrict cl0 = Cl[nz];   // spectra nz .. nz+3
      const double* restrict cl1 = Cl[nz1];
      const double* restrict cl2 = Cl[nz2];
      const double* restrict cl3 = Cl[nz3];
      const double* restrict pl0 = Pl[i];    // kernel of theta bins i .. i+3
      const double* restrict pl1 = Pl[i1];
      const double* restrict pl2 = Pl[i2];
      const double* restrict pl3 = Pl[i3];

      // sum<a><b>: the sum of spectrum nz + a and theta bin i + b
      double sum00 = 0.0, sum01 = 0.0, sum02 = 0.0, sum03 = 0.0;
      double sum10 = 0.0, sum11 = 0.0, sum12 = 0.0, sum13 = 0.0;
      double sum20 = 0.0, sum21 = 0.0, sum22 = 0.0, sum23 = 0.0;
      double sum30 = 0.0, sum31 = 0.0, sum32 = 0.0, sum33 = 0.0;

      #pragma omp simd reduction(+:sum00,sum01,sum02,sum03,\
                                   sum10,sum11,sum12,sum13,\
                                   sum20,sum21,sum22,sum23,\
                                   sum30,sum31,sum32,sum33)
      for (int l=lmin; l<lmax; l++) {
        // Pl*filter of the four theta bins: (Pl*filter)*Cl below is the
        // product order of the reference loop
        const double pf0 = pl0[l] * cf[l];
        const double pf1 = pl1[l] * cf[l];
        const double pf2 = pl2[l] * cf[l];
        const double pf3 = pl3[l] * cf[l];

        sum00 += pf0 * cl0[l];
        sum01 += pf1 * cl0[l];
        sum02 += pf2 * cl0[l];
        sum03 += pf3 * cl0[l];

        sum10 += pf0 * cl1[l];
        sum11 += pf1 * cl1[l];
        sum12 += pf2 * cl1[l];
        sum13 += pf3 * cl1[l];

        sum20 += pf0 * cl2[l];
        sum21 += pf1 * cl2[l];
        sum22 += pf2 * cl2[l];
        sum23 += pf3 * cl2[l];

        sum30 += pf0 * cl3[l];
        sum31 += pf1 * cl3[l];
        sum32 += pf2 * cl3[l];
        sum33 += pf3 * cl3[l];
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
#endif
}

// ---------------------------------------------------------------------------
// Physical scale cuts from the response function RF (2011.06469 eq 17).
//
// For an observable X, the weight of the modes below a candidate cut
// kmax is measured by
//
//   RF(kmax) = int_{-infty}^{ln kmax} dlnk |dlnX/dlnk|
//              / int_{-infty}^{+infty} dlnk |dlnX/dlnk|,
//
// the fraction of X's total logarithmic response contributed by
// k < kmax (RF grows monotonically from 0 to 1). A data point's scale
// cut is the kk solving RF(kk) = alpha for a chosen threshold alpha:
// the modes beyond kk carry less than the fraction 1 - alpha of the
// response.
//
// Pipeline:
//
//   CAMB P(k)
//     -> dC_X/dlnk tables, one Limber node per (k, l) entry
//        (filled by the _work functions in cosmo2D.c; cached here)
//     -> dlnC_X/dlnk = (dC_X/dlnk)/C_X (Fourier space) and, through
//        the Legendre sums below, dlnxi_pm/dlnk and dlnw_ks/dlnk
//        (real space)
//     -> RF(kmax, l) and RF(kmax, theta) tables (the RF_*_work
//        functions)
//     -> root-find RF = alpha, one kmax per data point.
//
// Observables: Fourier C_ss (EE and BB) and C_ks; real-space xi_pm and
// w_ks. This file only tabulates RF: the root-find happens downstream
// of the cosmo2D_scuts_wrapper.cpp exports.
// ---------------------------------------------------------------------------

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
// Shared state between the cached dC tables (which own the static
// storage) and the dln*_nointerp node fills (which read it).
//
// At one fixed k, evaluating a dC table on its own ell nodes is a
// single fixed-weight blend of the two k-rows that bracket k - the
// same weight for every (pair, ell) entry. The nointerp fills read
// the rows through these structs instead of paying the scalar
// lookup (cache-key checks, bin mapping, one bilinear read) once
// per entry. Builder and reader never call each other, so the
// file-scope struct carries the table pointer and the grid geometry
// between them - the ss_/gs_ pattern of cosmo2D.c.
//
//   tab        - the cached table ([2][pairs][nlnk][nell] for ss,
//                [nbin][nlnk][nell] for ks), owned by the builder
//   lim        - its grid: [0..2] = ln l (min, max, step),
//                [3..5] = ln k (min, max, step)
//   nlnk, nell - node counts along ln k and ln l
// ---------------------------------------------------------------------------
static struct { double**** tab; double lim[6]; int nlnk; int nell; }
    dCss_ = {0};
static struct { double*** tab; double lim[6]; int nlnk; int nell; }
    dCks_ = {0};

// Under COSMO2D_NOT_USE_SIMD (the DEBUG build) basics.h does not include
// the SIMDe headers, so the type below does not exist there; it is used
// only inside the SIMD branches.
#ifndef COSMO2D_NOT_USE_SIMD
typedef simde__m256d v4d; // 4 doubles, AVX2-width (as in cosmo2D.c)
#endif

// ---------------------------------------------------------------------------
// Blend two k-rows of a cached (ln k, ln l) table at one fixed weight.
//
// At a fixed k, reading a bilinear table on its own ell nodes
// reduces to out[i] = r0[i] + t (r1[i] - r0[i]) with the SAME t for
// every entry: linear interpolation between the two k-rows that
// bracket k. The nointerp fills below call this once per (plane,
// pair) row instead of one scalar table lookup per entry.
//
// Why explicit SIMDe instead of an omp simd pragma: the fill loops
// of this family read through pointer-to-pointer tables, and
// experience with limber_fill_interp (cosmo2D.c) showed compilers
// refuse to auto-vectorize them - the pragma silently produces
// scalar code. The SIMDe intrinsics guarantee the vector form from
// one source (AVX2 on x86, NEON on Apple Silicon).
//
// Why this is simpler than limber_fill_interp: there, every output
// multipole lands at a DIFFERENT grid position, so each vector lane
// needs its own index and the loads must be gathers. Here the
// weight t and the row offset are the same for every entry - the k
// bracket is fixed - so the body is two contiguous 4-wide loads and
// one fused multiply-add per lane, no gathers, plus a scalar tail
// for the last nell % 4 entries.
//
// Parameters:
//   ntab - number of planes blended together (1 or 2)
//   row0 - left k-row per plane [ntab][nell]
//   row1 - right k-row per plane [ntab][nell]
//   out  - output rows [ntab][nell]
//   t    - the fixed blend weight, in [0, 1) on the interior
//   nell - row length
//
// Returns:
//   void (out filled)
// ---------------------------------------------------------------------------
static void limber_krow_blend(
    const int ntab,               // number of planes (1 or 2)
    const double** restrict row0, // left k-row per plane [ntab][nell]
    const double** restrict row1, // right k-row per plane [ntab][nell]
    double** restrict out,        // output rows [ntab][nell]
    const double t,               // fixed blend weight
    const int nell                // row length
  )
{
#ifdef COSMO2D_NOT_USE_SIMD
  for (int q = 0; q < ntab; q++) {
    for (int i = 0; i < nell; i++) {
      out[q][i] = row0[q][i] + t*(row1[q][i] - row0[q][i]);
    }
  }
#else
  const v4d vt = simde_mm256_set1_pd(t); // the weight in all 4 lanes
  for (int q = 0; q < ntab; q++) {
    const double* restrict a = row0[q];
    const double* restrict b = row1[q];
    double* restrict o = out[q];
    int i = 0;
    for (; i <= nell - 4; i += 4) { // 4 entries per iteration
      const v4d v0 = simde_mm256_loadu_pd(a + i);
      const v4d v1 = simde_mm256_loadu_pd(b + i);
      // out = v0 + t*(v1 - v0): one fused multiply-add per lane
      simde_mm256_storeu_pd(o + i,
        simde_mm256_fmadd_pd(vt, simde_mm256_sub_pd(v1, v0), v0));
    }
    for (; i < nell; i++) { // scalar tail
      o[i] = a[i] + t*(b[i] - a[i]);
    }
  }
#endif
}

// ---------------------------------------------------------------------------
// Cached dC_ss/dlnk (2011.06469 eq 17): interpolates a (ln k, ln l)
// table filled by dC_ss_dlnk_tomo_limber_work (cosmo2D.c).
//
// One table entry = one Limber node. At fixed l, the Limber relation
// k = (l + 1/2)/chi makes each k select a single line-of-sight point:
//
//   fixed l:  k -> chi = (l + 1/2)/k -> a(chi) -> one evaluation of
//   the C_ss integrand core at that node
//
// where the "core" is the radial-weights x P_delta combination of the
// C_ss quadrature, TATT IA terms included (reducing identically to
// NLA) - everything under the C_ss line-of-sight integral except the
// measure and the ell prefactor. Each entry is the core times the
// change-of-variables amplitude:
//
//   dC_ss/dlnk(k, l) = core * ell_prefactor/fK,
//
// with fK the comoving angular diameter distance of the node and
// ell_prefactor = l*(l-1)*(l+1)*(l+2)/(l+1/2)^4 the curved-sky spin-2
// prefactor: one factor sqrt((l+2)!/(l-2)!)/(l+1/2)^2 per shear field,
// -> 1 for l >> 1 (see the prefactor blocks in cosmo2D.c).
//
// The entry is exactly 0 when the node falls outside the source
// support. See the _work header for the derivation (the dchida
// cancellation) and the normalize-mode contract; this lookup function fills
// its table in the raw mode (normalize = 0), so the entries are dC
// itself, not dC/C.
//
// Table design: [2][shear_Npowerspectra][nlnk][nell] (EE and BB), with
// nlnk = Ntable.dCX_dlnk_nlnk[NODES_DENSE] log-spaced k in [Ntable.dCX_dlnk_kmin,
// Ntable.dCX_dlnk_kmax] and nell = Ntable.N_ell[NODES_DENSE] log-spaced multipoles
// covering every l >= 1; lookups interpolate bilinearly in (ln k, ln l)
// and a (k, l) outside the table returns 0.
//
// When the internal coarse grids are active (Ntable.N_ell[NODES_COARSE] on
// the ell axis, Ntable.dCX_dlnk_nlnk[NODES_COARSE] on ln k), the exact
// evaluations run on the coarse nodes and a tensor-product bicubic
// upsamples onto the unchanged dense table (see the strategy note in
// the refill block).
//
// Cache invalidation:
// recomputes when any of these change:
//   cosmology.random, nuisance.random_photoz_shear, nuisance.random_ia,
//   redshift.random_shear, Ntable.random
// (allocation and grid limits rebuild on Ntable.random alone).
//
// Parameters:
//   k  - wavenumber in (Mpc/h)^-1
//   l  - multipole (continuous)
//   ni - first source redshift bin
//   nj - second source redshift bin
//   EE - 1 = E-mode, 0 = B-mode
//
// Returns:
//   dC_ss/dlnk at (k, l) for the (ni, nj) pair; 0 outside the table
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
  // lim = table grid: [0..2] = ln l (min, max, step), [3..5] = ln k
  static double lim[6];
  static int nell;
  static int nlnk;
  static double* lnkx = NULL; // fine ln k nodes
  static double* lxv = NULL;  // fine ell values
  static int nkc = 0;         // used ln k node count (= nlnk when exact)
  static int nlc = 0;         // used ell node count  (= nell when exact)
  static double dkc = 0.;     // used grid spacings (ln k, ln l)
  static double dlc = 0.;
  static double* lnkc = NULL; // coarse ln k nodes
  static double* lxc = NULL;  // coarse ell values
  static double**** tabc = NULL; // coarse dC values
  
  if (NULL == table || fdiff2(cache[4], Ntable.random)) {
    nell = Ntable.N_ell[NODES_DENSE];
    lim[0] = 0.0; // ln(l = 1): the grid covers every multipole l >= 1
    lim[1] = log(Ntable.LMAX + 1.);
    lim[2] = (lim[1] - lim[0]) / ((double) nell - 1.);

    nlnk = Ntable.dCX_dlnk_nlnk[NODES_DENSE];
    lim[3] = log(Ntable.dCX_dlnk_kmin);
    lim[4] = log(Ntable.dCX_dlnk_kmax);
    lim[5] = (lim[4] - lim[3]) / ((double) nlnk - 1.);

    if (table != NULL) free(table);
    table = (double****) malloc4d(2, tomo.shear_Npowerspectra, nlnk, nell);

    dCss_.tab = table;
    for (int j=0; j<6; j++) {
      dCss_.lim[j] = lim[j];
    }
    dCss_.nlnk = nlnk;
    dCss_.nell = nell;
  
    if (lnkx != NULL) free(lnkx);
    lnkx = (double*) malloc1d(nlnk);
    for (int f=0; f<nlnk; f++) {
      lnkx[f] = lim[3] + f*lim[5];
    }
    if (lxv != NULL) free(lxv);
    lxv = (double*) malloc1d(nell);
    for (int i=0; i<nell; i++) {
      lxv[i] = exp(lim[0] + i*lim[2]);
    }

    // Coarse-grid workspace: allocations live HERE, in the Ntable
    // rebuild block; the per-cosmology refill only fills. Each axis
    // coarsens independently: Ntable.N_ell[NODES_COARSE] on the ell axis
    // (smooth) and Ntable.dCX_dlnk_nlnk[NODES_COARSE] on the ln k axis
    // (where the BAO wiggles live; the default 128 keeps the response
    // error at the level the retired fixed quadrature imposed). An
    // axis whose knob is 0 (off) or out of range - fewer than the 4
    // nodes a natural cubic spline needs, or not below the exact
    // count - keeps its exact count.
    if (lnkc != NULL) { free(lnkc); lnkc = NULL; }
    if (lxc  != NULL) { free(lxc);  lxc  = NULL; }
    if (tabc != NULL) { free(tabc); tabc = NULL; }
    const int nk_int = Ntable.dCX_dlnk_nlnk[NODES_COARSE];
    const int nl_int = Ntable.N_ell[NODES_COARSE];
    nkc = (nk_int > 3 && nk_int < nlnk) ? nk_int : nlnk;
    nlc = (nl_int > 3 && nl_int < nell) ? nl_int : nell;
    dkc = (lim[4] - lim[3]) / ((double) nkc - 1.0);
    dlc = (lim[1] - lim[0]) / ((double) nlc - 1.0);
    if (nkc < nlnk || nlc < nell) {
      lnkc = (double*) malloc1d(nkc);
      for (int f=0; f<nkc; f++) {
        lnkc[f] = lim[3] + f*dkc;
      }
      lxc = (double*) malloc1d(nlc);
      for (int i=0; i<nlc; i++) {
        lxc[i] = exp(lim[0] + i*dlc);
      }
      tabc = (double****) malloc4d(2, tomo.shear_Npowerspectra, nkc, nlc);
    }
  }
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    if (nkc < nlnk || nlc < nell) {
      // ---------------------------------------------------------------
      // The internal coarse grid, in 2D: general strategy.
      //
      // Every entry of this (ln k, ln l) table costs one exact
      // single-node Limber evaluation, and the full table is
      // 2 x pairs x nlnk x nell of them - the expensive part. The
      // consumers then read the table through bilinear interpolation
      // (interpol2d), which needs DENSE nodes to be accurate.
      //
      // So, exactly as in the 1D C_XY tables (cosmo2D.c): run the
      // exact evaluations on a coarse grid and upsample with a cubic
      // spline - here the tensor-product natural bicubic of
      // spline2d_upsample_uniform (basics.c), two 1D passes of the
      // house spline - and hand every consumer the same dense table:
      //
      //   exact single-node Limber on (nkc x nlc) nodes
      //     -> spline pass along ln l, one per coarse k row
      //     -> spline pass along ln k, one per fine ell column
      //     -> the unchanged (nlnk x nell) dense table
      //     -> the same bilinear reads by every consumer
      //
      // The two axes coarsen independently because their smoothness
      // differs: the ell direction is smooth (as in cosmo2D.c), but
      // the ln k direction carries the BAO wiggles of P(k), so its
      // knob needs the denser default (128 of 256, set in structs.c;
      // the workspace note above states the accuracy target).
      // ---------------------------------------------------------------
      dC_ss_dlnk_tomo_limber_work(lnkc, nkc, lxc, nlc,
                                tomo.shear_Npowerspectra, 0, tabc);

      // one tensor-product bicubic upsample per stored plane (each
      // call reads and writes only its own plane, so the planes
      // thread freely); an axis left exact passes through
      // (near-)unchanged
      #pragma omp parallel for collapse(2) schedule(static)
      for (int c=0; c<2; c++) {
        for (int q=0; q<tomo.shear_Npowerspectra; q++) {
          spline2d_upsample_uniform(tabc[c][q], nkc, nlc, dkc, dlc,
                                    table[c][q], nlnk, nell);
        }
      }
    }
    else {
      // exact fill: one single-node Limber evaluation per dense
      // (ln k, ln l) node, no upsampling
      dC_ss_dlnk_tomo_limber_work(lnkx, nlnk, lxv, nell,
                                tomo.shear_Npowerspectra, 0, table);
    }
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
// a (ln k, ln l) table filled by the normalized (normalize = 1) mode of
// dC_ss_dlnk_tomo_limber_work, which computes the C_ss rows with its own
// quadrature and divides each dC row in place (see
// dC_ss_dlnk_tomo_limber above for the tabulated node amplitude and
// the _work header for the fill).
//
// Each tabulated numerator entry is the single Limber-node evaluation
//
//   dC_ss/dlnk(k, l) = core * ell_prefactor/fK,
//
// at chi(a) = (l + 1/2)/k (see dC_ss_dlnk_tomo_limber above), and the
// denominator C_ss(l) row comes from the _work function's own
// Gauss-Legendre quadrature. The division is guarded: an effectively
// zero dC entry passes through unnormalized and an effectively zero
// C_ss maps the entry to 0.
//
// Table design: [2][shear_Npowerspectra][nlnk][nell] (EE and BB), with
// nlnk = Ntable.dCX_dlnk_nlnk[NODES_DENSE] log-spaced k in [Ntable.dCX_dlnk_kmin,
// Ntable.dCX_dlnk_kmax] and nell = Ntable.N_ell[NODES_DENSE] log-spaced multipoles
// covering every l >= 1 (the same grid as dC_ss_dlnk_tomo_limber);
// lookups interpolate bilinearly in (ln k, ln l) and a (k, l) outside
// the table returns 0.
//
// When the internal coarse grids are active (Ntable.N_ell[NODES_COARSE] on
// the ell axis, Ntable.dCX_dlnk_nlnk[NODES_COARSE] on ln k), the exact
// evaluations run on the coarse nodes and a tensor-product bicubic
// upsamples onto the unchanged dense table (see the strategy note in
// the refill block).
//
// Cache invalidation:
// recomputes when any of these change:
//   cosmology.random, nuisance.random_photoz_shear, nuisance.random_ia,
//   redshift.random_shear, Ntable.random
// (allocation and grid limits rebuild on Ntable.random alone).
//
// Parameters:
//   k  - wavenumber in (Mpc/h)^-1
//   l  - multipole (continuous)
//   ni - first source redshift bin
//   nj - second source redshift bin
//   EE - 1 = E-mode, 0 = B-mode
//
// Returns:
//   dlnC_ss/dlnk at (k, l) for the (ni, nj) pair; 0 outside the table
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
  // lim = table grid: [0..2] = ln l (min, max, step), [3..5] = ln k
  static double lim[6];
  static int nell;
  static int nlnk;
  static double* lnkx = NULL; // fine ln k nodes
  static double* lxv = NULL;  // fine ell values
  static int nkc = 0;         // used ln k node count (= nlnk when exact)
  static int nlc = 0;         // used ell node count  (= nell when exact)
  static double dkc = 0.;     // used grid spacings (ln k, ln l)
  static double dlc = 0.;
  static double* lnkc = NULL; // coarse ln k nodes
  static double* lxc = NULL;  // coarse ell values
  static double**** tabc = NULL; // coarse dC values
  
  if (NULL == table || fdiff2(cache[4], Ntable.random)) {
    nell = Ntable.N_ell[NODES_DENSE];
    lim[0] = 0.0; // ln(l = 1): the grid covers every multipole l >= 1
    lim[1] = log(Ntable.LMAX + 1.);
    lim[2] = (lim[1] - lim[0]) / ((double) nell - 1.);

    nlnk = Ntable.dCX_dlnk_nlnk[NODES_DENSE];
    lim[3] = log(Ntable.dCX_dlnk_kmin);
    lim[4] = log(Ntable.dCX_dlnk_kmax);
    lim[5] = (lim[4] - lim[3]) / ((double) nlnk - 1.);

    if (table != NULL) free(table);
    table = (double****) malloc4d(2, tomo.shear_Npowerspectra, nlnk, nell);
  
    if (lnkx != NULL) free(lnkx);
    lnkx = (double*) malloc1d(nlnk);
    for (int f=0; f<nlnk; f++) {
      lnkx[f] = lim[3] + f*lim[5];
    }
    if (lxv != NULL) free(lxv);
    lxv = (double*) malloc1d(nell);
    for (int i=0; i<nell; i++) {
      lxv[i] = exp(lim[0] + i*lim[2]);
    }

    // Coarse-grid workspace: allocations live HERE, in the Ntable
    // rebuild block; the per-cosmology refill only fills. Each axis
    // coarsens independently: Ntable.N_ell[NODES_COARSE] on the ell axis
    // (smooth) and Ntable.dCX_dlnk_nlnk[NODES_COARSE] on the ln k axis
    // (where the BAO wiggles live; the default 128 keeps the response
    // error at the level the retired fixed quadrature imposed). An
    // axis whose knob is 0 (off) or out of range - fewer than the 4
    // nodes a natural cubic spline needs, or not below the exact
    // count - keeps its exact count.
    if (lnkc != NULL) { free(lnkc); lnkc = NULL; }
    if (lxc  != NULL) { free(lxc);  lxc  = NULL; }
    if (tabc != NULL) { free(tabc); tabc = NULL; }
    const int nk_int = Ntable.dCX_dlnk_nlnk[NODES_COARSE];
    const int nl_int = Ntable.N_ell[NODES_COARSE];
    nkc = (nk_int > 3 && nk_int < nlnk) ? nk_int : nlnk;
    nlc = (nl_int > 3 && nl_int < nell) ? nl_int : nell;
    dkc = (lim[4] - lim[3]) / ((double) nkc - 1.0);
    dlc = (lim[1] - lim[0]) / ((double) nlc - 1.0);
    if (nkc < nlnk || nlc < nell) {
      lnkc = (double*) malloc1d(nkc);
      for (int f=0; f<nkc; f++) {
        lnkc[f] = lim[3] + f*dkc;
      }
      lxc = (double*) malloc1d(nlc);
      for (int i=0; i<nlc; i++) {
        lxc[i] = exp(lim[0] + i*dlc);
      }
      tabc = (double****) malloc4d(2, tomo.shear_Npowerspectra, nkc, nlc);
    }
  }
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    if (nkc < nlnk || nlc < nell) {
      // Internal coarse grid, in 2D: exact single-node Limber on the
      // (nkc x nlc) coarse nodes, then the tensor-product natural
      // bicubic of spline2d_upsample_uniform fills the unchanged
      // dense table (full strategy note: dC_ss_dlnk_tomo_limber's
      // refill above; the ell axis is smooth, the ln k axis carries
      // the BAO wiggles; the workspace note above states the
      // default's accuracy target).
      dC_ss_dlnk_tomo_limber_work(lnkc, nkc, lxc, nlc,
                                tomo.shear_Npowerspectra, 1, tabc);

      // one tensor-product bicubic upsample per stored plane (each
      // call reads and writes only its own plane, so the planes
      // thread freely); an axis left exact passes through
      // (near-)unchanged
      #pragma omp parallel for collapse(2) schedule(static)
      for (int c=0; c<2; c++) {
        for (int q=0; q<tomo.shear_Npowerspectra; q++) {
          spline2d_upsample_uniform(tabc[c][q], nkc, nlc, dkc, dlc,
                                    table[c][q], nlnk, nell);
        }
      }
    }
    else {
      // exact fill: one single-node Limber evaluation per dense
      // (ln k, ln l) node, no upsampling
      dC_ss_dlnk_tomo_limber_work(lnkx, nlnk, lxv, nell,
                                tomo.shear_Npowerspectra, 1, table);
    }
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
// Cached dC_ks/dlnk (2011.06469 eq 17): interpolates a (ln k, ln l)
// table filled by dC_ks_dlnk_tomo_limber_work (cosmo2D.c).
//
// One table entry = one Limber node, as in the ss case:
//
//   fixed l:  k -> chi = (l + 1/2)/k -> a(chi) -> one evaluation of
//   the C_ks integrand core (W_kappa - W_source*IA_A1)*W_k*P_delta
//
// times the change-of-variables amplitude:
//
//   dC_ks/dlnk(k, l) = core * pf1*pf2/fK,
//
// with fK the comoving angular diameter distance of the node and the
// curved-sky prefactors of the spin-0 x spin-2 cross (1812.05995
// eqs 74-79):
//
//   pf1 = l*(l+1)/(l+1/2)^2                    (CMB kappa, spin-0)
//   pf2 = sqrt((l-1)*l*(l+1)*(l+2))/(l+1/2)^2  (shear, spin-2)
//
// The entry is exactly 0 when the node falls outside bin ni's source
// support. See the _work header for the derivation and the
// normalize-mode contract; this lookup function fills its table in the raw
// mode (normalize = 0), so the entries are dC itself, not dC/C.
//
// Table design: [shear_nbin][nlnk][nell] (the ks cross has one component
// per source bin, so there is no EE/BB leading dimension), on the same
// (ln k, ln l) grid as the ss tables: the Ntable.dCX_dlnk k range and
// every multipole l >= 1; a (k, l) outside the table returns 0.
//
// When the internal coarse grids are active (Ntable.N_ell[NODES_COARSE] on
// the ell axis, Ntable.dCX_dlnk_nlnk[NODES_COARSE] on ln k), the exact
// evaluations run on the coarse nodes and a tensor-product bicubic
// upsamples onto the unchanged dense table (see the strategy note in
// the refill block).
//
// Cache invalidation:
// recomputes when any of these change:
//   cosmology.random, nuisance.random_photoz_shear, nuisance.random_ia,
//   redshift.random_shear, Ntable.random
// (the same keys as the C_ks C_ell cache; the CMB beam enters only
// downstream, in the w_ks projection, so cmb.random is not a key here).
//
// Parameters:
//   k  - wavenumber in (Mpc/h)^-1
//   l  - multipole (continuous)
//   ni - source redshift bin
//
// Returns:
//   dC_ks/dlnk at (k, l) for bin ni; 0 outside the table
// ---------------------------------------------------------------------------
double dC_ks_dlnk_tomo_limber(
    const double k,   // wavenumber in (Mpc/h)^-1
    const double l,   // multipole
    const int ni      // source redshift bin
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double*** table;
  // lim = table grid: [0..2] = ln l (min, max, step), [3..5] = ln k
  static double lim[6];
  static int nell;
  static int nlnk;
  static double* lnkx = NULL; // fine ln k nodes
  static double* lxv = NULL;  // fine ell values
  static int nkc = 0;         // used ln k node count (= nlnk when exact)
  static int nlc = 0;         // used ell node count  (= nell when exact)
  static double dkc = 0.;     // used grid spacings (ln k, ln l)
  static double dlc = 0.;
  static double* lnkc = NULL; // coarse ln k nodes
  static double* lxc = NULL;  // coarse ell values
  static double*** tabc = NULL; // coarse dC values

  if (NULL == table || fdiff2(cache[4], Ntable.random)) {
    nell = Ntable.N_ell[NODES_DENSE];
    lim[0] = 0.0; // ln(l = 1): the grid covers every multipole l >= 1
    lim[1] = log(Ntable.LMAX + 1.);
    lim[2] = (lim[1] - lim[0]) / ((double) nell - 1.);

    nlnk = Ntable.dCX_dlnk_nlnk[NODES_DENSE];
    lim[3] = log(Ntable.dCX_dlnk_kmin);
    lim[4] = log(Ntable.dCX_dlnk_kmax);
    lim[5] = (lim[4] - lim[3]) / ((double) nlnk - 1.);

    if (table != NULL) free(table);
    table = (double***) malloc3d(redshift.shear_nbin, nlnk, nell);

    dCks_.tab = table;
    for (int j=0; j<6; j++) {
      dCks_.lim[j] = lim[j];
    }
    dCks_.nlnk = nlnk;
    dCks_.nell = nell;
  
    if (lnkx != NULL) free(lnkx);
    lnkx = (double*) malloc1d(nlnk);
    for (int f=0; f<nlnk; f++) {
      lnkx[f] = lim[3] + f*lim[5];
    }
    if (lxv != NULL) free(lxv);
    lxv = (double*) malloc1d(nell);
    for (int i=0; i<nell; i++) {
      lxv[i] = exp(lim[0] + i*lim[2]);
    }

    // Coarse-grid workspace: allocations live HERE, in the Ntable
    // rebuild block; the per-cosmology refill only fills. Each axis
    // coarsens independently: Ntable.N_ell[NODES_COARSE] on the ell axis
    // (smooth) and Ntable.dCX_dlnk_nlnk[NODES_COARSE] on the ln k axis
    // (where the BAO wiggles live; the default 128 keeps the response
    // error at the level the retired fixed quadrature imposed). An
    // axis whose knob is 0 (off) or out of range - fewer than the 4
    // nodes a natural cubic spline needs, or not below the exact
    // count - keeps its exact count.
    if (lnkc != NULL) { free(lnkc); lnkc = NULL; }
    if (lxc  != NULL) { free(lxc);  lxc  = NULL; }
    if (tabc != NULL) { free(tabc); tabc = NULL; }
    const int nk_int = Ntable.dCX_dlnk_nlnk[NODES_COARSE];
    const int nl_int = Ntable.N_ell[NODES_COARSE];
    nkc = (nk_int > 3 && nk_int < nlnk) ? nk_int : nlnk;
    nlc = (nl_int > 3 && nl_int < nell) ? nl_int : nell;
    dkc = (lim[4] - lim[3]) / ((double) nkc - 1.0);
    dlc = (lim[1] - lim[0]) / ((double) nlc - 1.0);
    if (nkc < nlnk || nlc < nell) {
      lnkc = (double*) malloc1d(nkc);
      for (int f=0; f<nkc; f++) {
        lnkc[f] = lim[3] + f*dkc;
      }
      lxc = (double*) malloc1d(nlc);
      for (int i=0; i<nlc; i++) {
        lxc[i] = exp(lim[0] + i*dlc);
      }
      tabc = (double***) malloc3d(redshift.shear_nbin, nkc, nlc);
    }
  }
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    if (nkc < nlnk || nlc < nell) {
      // Internal coarse grid, in 2D: exact single-node Limber on the
      // (nkc x nlc) coarse nodes, then the tensor-product natural
      // bicubic of spline2d_upsample_uniform fills the unchanged
      // dense table (full strategy note: dC_ss_dlnk_tomo_limber's
      // refill above; the ell axis is smooth, the ln k axis carries
      // the BAO wiggles; the workspace note above states the
      // default's accuracy target).
      dC_ks_dlnk_tomo_limber_work(lnkc, nkc, lxc, nlc,
                                redshift.shear_nbin, 0, tabc);

      // one tensor-product bicubic upsample per stored plane (each
      // call reads and writes only its own plane, so the planes
      // thread freely); an axis left exact passes through
      // (near-)unchanged
      #pragma omp parallel for schedule(static)
      for (int nz=0; nz<redshift.shear_nbin; nz++) {
        spline2d_upsample_uniform(tabc[nz], nkc, nlc, dkc, dlc,
                                  table[nz], nlnk, nell);
      }
    }
    else {
      // exact fill: one single-node Limber evaluation per dense
      // (ln k, ln l) node, no upsampling
      dC_ks_dlnk_tomo_limber_work(lnkx, nlnk, lxv, nell,
                                redshift.shear_nbin, 0, table);
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
// a (ln k, ln l) table filled by the normalized (normalize = 1) mode of
// dC_ks_dlnk_tomo_limber_work, which computes the C_ks rows with its own
// per-bin quadrature and divides each dC row in place (see
// dC_ks_dlnk_tomo_limber above for the tabulated node amplitude).
//
// Each tabulated numerator entry is the single Limber-node evaluation
//
//   dC_ks/dlnk(k, l) = core * pf1*pf2/fK,
//
// at chi(a) = (l + 1/2)/k (see dC_ks_dlnk_tomo_limber above for the
// spin-0 x spin-2 prefactors), and the denominator C_ks(l) row comes
// from the _work function's own per-bin Gauss-Legendre quadrature. The
// division is guarded: an effectively zero dC entry passes through
// unnormalized and an effectively zero C_ks maps the entry to 0.
//
// Table design: [shear_nbin][nlnk][nell] (one component per source
// bin), with nlnk = Ntable.dCX_dlnk_nlnk[NODES_DENSE] log-spaced k in
// [Ntable.dCX_dlnk_kmin, Ntable.dCX_dlnk_kmax] and nell = Ntable.N_ell[NODES_DENSE]
// log-spaced multipoles covering every l >= 1 (the same grid as
// dC_ks_dlnk_tomo_limber); lookups interpolate bilinearly in
// (ln k, ln l) and a (k, l) outside the table returns 0.
//
// When the internal coarse grids are active (Ntable.N_ell[NODES_COARSE] on
// the ell axis, Ntable.dCX_dlnk_nlnk[NODES_COARSE] on ln k), the exact
// evaluations run on the coarse nodes and a tensor-product bicubic
// upsamples onto the unchanged dense table (see the strategy note in
// the refill block).
//
// Cache invalidation:
// recomputes when any of these change:
//   cosmology.random, nuisance.random_photoz_shear, nuisance.random_ia,
//   redshift.random_shear, Ntable.random
// (cmb.random is not a key: the CMB beam enters only downstream, in the
// w_ks projection).
//
// Parameters:
//   k  - wavenumber in (Mpc/h)^-1
//   l  - multipole (continuous)
//   ni - source redshift bin
//
// Returns:
//   dlnC_ks/dlnk at (k, l) for bin ni; 0 outside the table
// ---------------------------------------------------------------------------
double dlnC_ks_dlnk_tomo_limber(
    const double k,   // wavenumber in (Mpc/h)^-1
    const double l,   // multipole
    const int ni      // source redshift bin
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double*** table;
  // lim = table grid: [0..2] = ln l (min, max, step), [3..5] = ln k
  static double lim[6];
  static int nell;
  static int nlnk;
  static double* lnkx = NULL; // fine ln k nodes
  static double* lxv = NULL;  // fine ell values
  static int nkc = 0;         // used ln k node count (= nlnk when exact)
  static int nlc = 0;         // used ell node count  (= nell when exact)
  static double dkc = 0.;     // used grid spacings (ln k, ln l)
  static double dlc = 0.;
  static double* lnkc = NULL; // coarse ln k nodes
  static double* lxc = NULL;  // coarse ell values
  static double*** tabc = NULL; // coarse dC values

  if (NULL == table || fdiff2(cache[4], Ntable.random)) {
    nell = Ntable.N_ell[NODES_DENSE];
    lim[0] = 0.0; // ln(l = 1): the grid covers every multipole l >= 1
    lim[1] = log(Ntable.LMAX + 1.);
    lim[2] = (lim[1] - lim[0]) / ((double) nell - 1.);

    nlnk = Ntable.dCX_dlnk_nlnk[NODES_DENSE];
    lim[3] = log(Ntable.dCX_dlnk_kmin);
    lim[4] = log(Ntable.dCX_dlnk_kmax);
    lim[5] = (lim[4] - lim[3]) / ((double) nlnk - 1.);

    if (table != NULL) free(table);
    table = (double***) malloc3d(redshift.shear_nbin, nlnk, nell);
  
    if (lnkx != NULL) free(lnkx);
    lnkx = (double*) malloc1d(nlnk);
    for (int f=0; f<nlnk; f++) {
      lnkx[f] = lim[3] + f*lim[5];
    }
    if (lxv != NULL) free(lxv);
    lxv = (double*) malloc1d(nell);
    for (int i=0; i<nell; i++) {
      lxv[i] = exp(lim[0] + i*lim[2]);
    }

    // Coarse-grid workspace: allocations live HERE, in the Ntable
    // rebuild block; the per-cosmology refill only fills. Each axis
    // coarsens independently: Ntable.N_ell[NODES_COARSE] on the ell axis
    // (smooth) and Ntable.dCX_dlnk_nlnk[NODES_COARSE] on the ln k axis
    // (where the BAO wiggles live; the default 128 keeps the response
    // error at the level the retired fixed quadrature imposed). An
    // axis whose knob is 0 (off) or out of range - fewer than the 4
    // nodes a natural cubic spline needs, or not below the exact
    // count - keeps its exact count.
    if (lnkc != NULL) { free(lnkc); lnkc = NULL; }
    if (lxc  != NULL) { free(lxc);  lxc  = NULL; }
    if (tabc != NULL) { free(tabc); tabc = NULL; }
    const int nk_int = Ntable.dCX_dlnk_nlnk[NODES_COARSE];
    const int nl_int = Ntable.N_ell[NODES_COARSE];
    nkc = (nk_int > 3 && nk_int < nlnk) ? nk_int : nlnk;
    nlc = (nl_int > 3 && nl_int < nell) ? nl_int : nell;
    dkc = (lim[4] - lim[3]) / ((double) nkc - 1.0);
    dlc = (lim[1] - lim[0]) / ((double) nlc - 1.0);
    if (nkc < nlnk || nlc < nell) {
      lnkc = (double*) malloc1d(nkc);
      for (int f=0; f<nkc; f++) {
        lnkc[f] = lim[3] + f*dkc;
      }
      lxc = (double*) malloc1d(nlc);
      for (int i=0; i<nlc; i++) {
        lxc[i] = exp(lim[0] + i*dlc);
      }
      tabc = (double***) malloc3d(redshift.shear_nbin, nkc, nlc);
    }
  }
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    if (nkc < nlnk || nlc < nell) {
      // Internal coarse grid, in 2D: exact single-node Limber on the
      // (nkc x nlc) coarse nodes, then the tensor-product natural
      // bicubic of spline2d_upsample_uniform fills the unchanged
      // dense table (full strategy note: dC_ss_dlnk_tomo_limber's
      // refill above; the ell axis is smooth, the ln k axis carries
      // the BAO wiggles; the workspace note above states the
      // default's accuracy target).
      dC_ks_dlnk_tomo_limber_work(lnkc, nkc, lxc, nlc,
                                redshift.shear_nbin, 1, tabc);

      // one tensor-product bicubic upsample per stored plane (each
      // call reads and writes only its own plane, so the planes
      // thread freely); an axis left exact passes through
      // (near-)unchanged
      #pragma omp parallel for schedule(static)
      for (int nz=0; nz<redshift.shear_nbin; nz++) {
        spline2d_upsample_uniform(tabc[nz], nkc, nlc, dkc, dlc,
                                  table[nz], nlnk, nell);
      }
    }
    else {
      // exact fill: one single-node Limber evaluation per dense
      // (ln k, ln l) node, no upsampling
      dC_ks_dlnk_tomo_limber_work(lnkx, nlnk, lxv, nell,
                                redshift.shear_nbin, 1, table);
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
// (RF is normalized by its kk -> infty value, so RF runs from 0 to 1;
// alpha = the kept response fraction. The functions below tabulate
// RF(kmax, l) on a kmax grid - the root-find for kk happens downstream.
// See the file preamble.)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Exact integrals of |v(x)| over one grid interval, v linear inside.
//
// The RF numerators and denominators integrate the ABSOLUTE log
// response |dlnX/dlnk|, and the tabulated response is piecewise
// LINEAR in ln k (the dln caches store nodes on the uniform
// Ntable.dCX_dlnk grid and interpolate linearly between them). The
// integral of |v| over one interval of width dx is therefore closed
// form:
//
//   v keeps its sign across the interval
//     -> the plain trapezoid: 0.5 (|va| + |vb|) dx
//
//   v crosses zero inside (va vb < 0), at t* = |va|/(|va|+|vb|) dx
//     -> two triangles: 0.5 |va| t* + 0.5 |vb| (dx - t*)
//
// scuts_abs_lin_part integrates only [0, tt] of the interval - the
// cut last piece of a partial integral: v(tt) closes the trapezoid,
// or, when the crossing t* lies before tt, the second triangle is
// cut at tt.
//
// These make the RF integrals EXACT for the tabulated integrand:
// summing them interval by interval is not a quadrature rule
// approximating the tables - it is the integral of what the tables
// define. A 512-node Gauss-Legendre sweep per kmax would be an
// expensive approximation of a function whose integral has a closed
// form; that is why no quadrature rule appears in the RF workers.
// ---------------------------------------------------------------------------
static inline double scuts_abs_lin_full(
    const double va, // v at the interval's left node
    const double vb, // v at the interval's right node
    const double dx  // interval width (the uniform ln k spacing)
  )
{
  // no sign change on [0, dx]: |v| is the trapezoid over the interval
  if (va*vb >= 0.0) {
    return 0.5*(fabs(va) + fabs(vb))*dx;
  }
  // v crosses zero at ts (similar triangles: |va| : |vb| splits dx),
  // so |v| is two triangles, one on each side of the crossing
  const double ts = fabs(va)/(fabs(va) + fabs(vb))*dx;
  return 0.5*fabs(va)*ts + 0.5*fabs(vb)*(dx - ts);
}

static inline double scuts_abs_lin_part(
    const double va, // v at the interval's left node
    const double vb, // v at the interval's right node
    const double dx, // interval width (the uniform ln k spacing)
    const double tt  // integrate |v| over [0, tt], 0 <= tt <= dx
  )
{
  // v at the cut point: the same straight line, evaluated at tt
  const double vt = va + (vb - va)*(tt/dx);
  // no sign change on [0, tt]: the trapezoid closed by v(tt)
  if (va*vt >= 0.0) {
    return 0.5*(fabs(va) + fabs(vt))*tt;
  }
  // v changes sign inside [0, tt]. The zero crossing is a property
  // of the LINE, not of where the integral stops, so the
  // full-interval formula (from va and vb) still locates it, and
  // va*vt < 0 guarantees ts < tt. Two triangles again, the second
  // one cut at tt with height |v(tt)|.
  const double ts = fabs(va)/(fabs(va) + fabs(vb))*dx;
  return 0.5*fabs(va)*ts + 0.5*fabs(vt)*(tt - ts);
}

// ---------------------------------------------------------------------------
// RF of C_ss over kmax, exact from the tabulated response.
//
// RF(kmax) is the fraction of the total absolute log response below
// kmax,
//
//   RF(kmax) = int_(-inf)^(ln kmax) |dlnX/dlnk| dlnk
//            / int_(-inf)^(+inf)    |dlnX/dlnk| dlnk,
//
// and the response it integrates is the dlnC_ss_dlnk table read at
// fixed l: a table on the uniform Ntable.dCX_dlnk grid in ln k,
// read by BILINEAR interpolation in (ln k, ln l) - at fixed l that
// read is LINEAR between the ln k nodes - and exactly zero outside
// the grid. The integrand is therefore piecewise linear, and both
// integrals are CLOSED FORM (see scuts_abs_lin_full/_part above):
// no quadrature rule, no error.
//
// Per (EE/BB, pair, multipole) row:
//
//   sample the response at the grid's own ln k nodes
//     -> cumulative sum of the per-interval |v| integrals
//        (trapezoids, sign-crossing triangles)
//     -> denominator = the full cumulative
//     -> every requested kmax = prefix + the cut last piece,
//        found by one multiply and a cast (uniform grid, no search)
//
// Vanishing-denominator guard: the BB response is identically zero
// under NLA, so a zero full cumulative writes 0, never 0/0 = NaN.
//
// Cache invalidation:
// none (stateless); each call first builds (or reuses) the cached
// dlnC_ss table with one single-threaded read, and the parallel
// rows then sample it read-only.
//
// Parameters:
//   lnkmaxx - ln kmax values (length nkmax), k in (Mpc/h)^-1
//   nkmax   - number of ln kmax values
//   lx      - multipole values (length nl)
//   nl      - number of multipole values
//   NSIZE   - number of tomo shear power spectra
//   table   - output [2][NSIZE][nkmax][nl]: EE and BB
//
// Returns:
//   void (table filled for every kmax and multipole)
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
  if (nkmax <= 0) {
    log_fatal("nkmax = %d must be positive", nkmax);
    exit(1);
  }
  const int nlnk = Ntable.dCX_dlnk_nlnk[NODES_DENSE];
  const double lnk0 = log(Ntable.dCX_dlnk_kmin);
  const double dx = (log(Ntable.dCX_dlnk_kmax) - lnk0)
                    / ((double) nlnk - 1.0);
  const double lnk1 = lnk0 + (nlnk - 1)*dx;
  double* kv = (double*) malloc1d(nlnk); // the grid's own k nodes
  for (int f = 0; f < nlnk; f++) {
    kv[f] = exp(lnk0 + f*dx);
  }
  // build the cached dlnC table (and the statics it warms) single-threaded
  (void) dlnC_ss_dlnk_tomo_limber(1.0, lx[0], Z1(0), Z2(0), 1);
  #pragma omp parallel
  {
    double* prof = (double*) malloc1d(nlnk); // one row's response
    double* cum  = (double*) malloc1d(nlnk); // its running integral
    #pragma omp for collapse(3) schedule(static)
    for (int ee = 0; ee < 2; ee++) {       // 0 = EE, 1 = BB output row
      for (int nz = 0; nz < NSIZE; nz++) {
        for (int i = 0; i < nl; i++) {
          const int Z1NZ = Z1(nz);
          const int Z2NZ = Z2(nz);
          const double l = lx[i];
          const int EEflag = (0 == ee) ? 1 : 0;
          for (int f = 0; f < nlnk; f++) {
            prof[f] = dlnC_ss_dlnk_tomo_limber(kv[f], l, Z1NZ, Z2NZ, EEflag);
          }
          cum[0] = 0.0;
          for (int f = 1; f < nlnk; f++) {
            cum[f] = cum[f-1] + scuts_abs_lin_full(prof[f-1], prof[f], dx);
          }
          const double dden = cum[nlnk-1];
          for (int m = 0; m < nkmax; m++) {
            const double L = lnkmaxx[m];
            double num;
            if (L <= lnk0) {
              num = 0.0;
            }
            else if (L >= lnk1) {
              num = dden;
            }
            else {
              const double r = (L - lnk0)/dx;
              int j = (int) r; // interval's left node (uniform grid)
              if (j > nlnk - 2) { // 1-ulp division overshoot near lnk1
                j = nlnk - 2;
              }
              num = cum[j] +
                    scuts_abs_lin_part(prof[j], prof[j+1], dx, (r - j)*dx);
            }
            table[ee][nz][m][i] = (fabs(dden) > 1e-300) ?
                                  num/dden : 0.0;
          }
        }
      }
    }
    free(prof);
    free(cum);
  } // end of the parallel region
  free(kv);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// RF of C_ks over kmax, exact from the tabulated response.
//
// RF(kmax) is the fraction of the total absolute log response below
// kmax,
//
//   RF(kmax) = int_(-inf)^(ln kmax) |dlnX/dlnk| dlnk
//            / int_(-inf)^(+inf)    |dlnX/dlnk| dlnk,
//
// and the response it integrates is the dlnC_ks_dlnk table read at
// fixed l: a table on the uniform Ntable.dCX_dlnk grid in ln k,
// read by BILINEAR interpolation in (ln k, ln l) - at fixed l that
// read is LINEAR between the ln k nodes - and exactly zero outside
// the grid. The integrand is therefore piecewise linear, and both
// integrals are CLOSED FORM (see scuts_abs_lin_full/_part above):
// no quadrature rule, no error.
//
// Per (source bin, multipole) row:
//
//   sample the response at the grid's own ln k nodes
//     -> cumulative sum of the per-interval |v| integrals
//        (trapezoids, sign-crossing triangles)
//     -> denominator = the full cumulative
//     -> every requested kmax = prefix + the cut last piece,
//        found by one multiply and a cast (uniform grid, no search)
//
// Vanishing-denominator guard kept for symmetry with RF_C_ss: a zero
// full cumulative writes 0, never 0/0 = NaN.
//
// Cache invalidation:
// none (stateless); each call first builds (or reuses) the cached
// dlnC_ks table with one single-threaded read, and the parallel
// rows then sample it read-only.
//
// Parameters:
//   lnkmaxx - ln kmax values (length nkmax), k in (Mpc/h)^-1
//   nkmax   - number of ln kmax values
//   lx      - multipole values (length nl)
//   nl      - number of multipole values
//   NSIZE   - number of source tomographic bins (= shear_nbin)
//   table   - output [NSIZE][nkmax][nl]
//
// Returns:
//   void (table filled for every kmax and multipole)
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
  if (nkmax <= 0) {
    log_fatal("nkmax = %d must be positive", nkmax);
    exit(1);
  }
  const int nlnk = Ntable.dCX_dlnk_nlnk[NODES_DENSE];
  const double lnk0 = log(Ntable.dCX_dlnk_kmin);
  const double dx = (log(Ntable.dCX_dlnk_kmax) - lnk0)
                    / ((double) nlnk - 1.0);
  const double lnk1 = lnk0 + (nlnk - 1)*dx;
  double* kv = (double*) malloc1d(nlnk); // the grid's own k nodes
  for (int f = 0; f < nlnk; f++) {
    kv[f] = exp(lnk0 + f*dx);
  }
  // build the cached dlnC_ks table single-threaded before the
  // parallel region below reads it
  (void) dlnC_ks_dlnk_tomo_limber(1.0, lx[0], 0);
  #pragma omp parallel
  {
    double* prof = (double*) malloc1d(nlnk); // one row's response
    double* cum  = (double*) malloc1d(nlnk); // its running integral
    #pragma omp for collapse(2) schedule(static)
    for (int nz = 0; nz < NSIZE; nz++) {
      for (int i = 0; i < nl; i++) {
        const double l = lx[i];
        for (int f = 0; f < nlnk; f++) {
          prof[f] = dlnC_ks_dlnk_tomo_limber(kv[f], l, nz);
        }
        cum[0] = 0.0;
        for (int f = 1; f < nlnk; f++) {
          cum[f] = cum[f-1] + scuts_abs_lin_full(prof[f-1], prof[f], dx);
        }
        const double dden = cum[nlnk-1];
        for (int m = 0; m < nkmax; m++) {
          const double L = lnkmaxx[m];
          double num;
          if (L <= lnk0) {
            num = 0.0;
          }
          else if (L >= lnk1) {
            num = dden;
          }
          else {
            const double r = (L - lnk0)/dx;
            int j = (int) r; // interval's left node (uniform grid)
            if (j > nlnk - 2) { // 1-ulp division overshoot near lnk1
              j = nlnk - 2;
            }
            num = cum[j] +
                  scuts_abs_lin_part(prof[j], prof[j+1], dx, (r - j)*dx);
          }
          table[nz][m][i] = (fabs(dden) > 1e-300) ? num/dden : 0.0;
        }
      }
    }
    free(prof);
    free(cum);
  } // end of the parallel region
  free(kv);
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
// Steps 1 and 2 combined equal one bilinear (ln k, ln l) read of the
// 2D table at each integer multipole — same knots, same weights — so
// splitting the read loses no accuracy and no per-multipole cache is
// needed.
//
// Static state, rebuilt when Ntable or the tomography change:
//   Glpm[2][Ntheta][LMAX] - bin-averaged Legendre kernels (Gl+ and Gl-)
//   ln_ell[LMAX]          - log(l) at every integer multipole
//   dCgrid[2][NSIZE][N_ell], cx[2][NSIZE][LMAX] - work arrays the
//                           pipeline overwrites on every call
//
// Cache invalidation:
// Ntable.random (or a tomography-size change)
// rebuilds the static state above. No cosmology key is needed: the
// cosmology enters only through the cached dC and xi tables read on
// every call.
//
// Parameters:
//   k - wavenumber in (Mpc/h)^-1
//
// Returns:
//   newly allocated [2][NSIZE*Ntheta] array of dlnxi_pm/dlnk values
//   (caller frees), one row per xi component, each row flattened as
//   nz*Ntheta + i over (tomo pair nz, angular bin i) with
//   NSIZE = tomo.shear_Npowerspectra; all zeros when k is outside the
//   open interval (Ntable.dCX_dlnk_kmin, Ntable.dCX_dlnk_kmax)
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
  // the kernel formulas below divide by l (and the l < lmin rows are
  // zeroed): the monopole is excluded, so the sums start at lmin = 1
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
    // Exact analytic bin averages of the spin-2 Legendre kernels for
    // xi_+ (Glpm[0]) and xi_- (Glpm[1]): the point kernels of the sums
    //
    //   xi_+/-(theta) = sum_l Gl_+/-(theta, l) (C_EE +/- C_BB)
    //
    // integrated in x = cos(theta) over the angular bin and divided by
    // the bin width - hence the trailing /(xmin - xmax), with
    // xmin = cos(theta) at the bin's lower angular edge and xmax at
    // the upper (cosine reverses the order, so xmin > xmax). Every
    // term reduces, via Legendre recurrences, to P_l and dP_l at the
    // two edges - the Pmin/Pmax/dPmin/dPmax arrays above. The two
    // formulas differ only in the sign of their last two terms (the
    // d^l_{2,+2} vs d^l_{2,-2} parts of the spin-2 kernel). Identical
    // to the kernels xi_pm_tomo sums against - see the derivation
    // block inside xi_pm_tomo (cosmo2D.c).
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
    dCgrid = (double***) malloc3d(2, NSIZE, Ntable.N_ell[NODES_DENSE]);
    zero3d(dCgrid, 2, NSIZE, Ntable.N_ell[NODES_DENSE]);
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
    // dC_ss/dlnk at this k on the dC table's own multipole log-grid.
    // la/ldx must reproduce the dC table's ln l grid (lim[0], lim[2])
    // exactly: the exact-node read here, and limber_fill_interp's
    // inverse map below, rely on the two grids being the same
    const int nell = Ntable.N_ell[NODES_DENSE];
    const double la = 0.0; // ln(l = 1): the dC table's multipole grid start
    const double ldx = (log(Ntable.LMAX + 1.) - la)/((double) nell - 1.);
    // At fixed k the bilinear table read at an exact ell node is one
    // fixed-weight blend of the two bracketing k-rows - the same
    // weight tk for every (pair, ell) entry - so read the rows
    // through the dCss_ struct instead of one scalar lookup per
    // entry (cache-key checks, bin mapping, bilinear read, nell x
    // pairs times per k)
    const double rk = (log(k) - dCss_.lim[3])/dCss_.lim[5];
    int fk = (int) rk;
    if (fk > dCss_.nlnk - 2) { // 1-ulp division overshoot near kmax
      fk = dCss_.nlnk - 2;
    }
    const double tk = rk - fk;
    // SIMDe-vectorized fixed-weight blend (limber_krow_blend, the
    // limber_fill_interp companion), both planes per pair at once
    #pragma omp parallel for schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      const double* r0[2] = {dCss_.tab[0][nz][fk], dCss_.tab[1][nz][fk]};
      const double* r1[2] = {dCss_.tab[0][nz][fk+1],
                             dCss_.tab[1][nz][fk+1]};
      double* out2[2] = {dCgrid[0][nz], dCgrid[1][nz]};
      limber_krow_blend(2, r0, r1, out2, tk, nell);
    }
    // gather onto every integer multipole (vectorized linear interpolation)
    #pragma omp parallel for schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      const double* tab2[2] = {dCgrid[0][nz], dCgrid[1][nz]};
      double* out2[2] = {cx[0][nz], cx[1][nz]};
      limber_fill_interp(2, tab2, out2, lmin, Ntable.LMAX, ln_ell,
                         la, 1.0/ldx, nell);
    }
    // the grouped xi+- sums of xi_pm_tomo (cosmo2D.c): same sums, 2 pairs
    // x 4 theta bins per pass over l instead of one (pair, theta) per pass
    legendre_sums_xipm(NSIZE, Ntable.Ntheta, lmin, Ntable.LMAX,
                       Glpm[0], Glpm[1], cx[0], cx[1], ans[0], ans[1]);
    // warm xi_pm_tomo's cached tables single-threaded (one call builds
    // both xi+ and xi-); the parallel loop below then only reads them
    (void) xi_pm_tomo(0, 0, Z1(0), Z2(0), 1);
    #pragma omp parallel for collapse(3) schedule(static)
    for (int p=0; p<2; p++) {
      for (int nz=0; nz<NSIZE; nz++) {
        for (int i=0; i<Ntable.Ntheta; i++) {
          const int q = nz * Ntable.Ntheta + i;
          // ans rows are p = 0 -> xi+, p = 1 -> xi-, while xi_pm_tomo
          // takes 1 = xi+, 0 = xi-: hence the 1 - p index flip (its
          // last argument is the limber flag, 1 = Limber). Double
          // guard, as in the dlnC fill (cosmo2D.c): a ~0 derivative
          // stays 0 without reading xi, and a ~0 xi denominator maps
          // to 0, so the ratio never blows up where the signal
          // vanishes
          const double dxipmdlnk = ans[p][q];
          if (fabs(ans[p][q])>1.e-50) {
            const double xipm = xi_pm_tomo(1 - p, i, Z1(nz), Z2(nz), 1);
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
// dlnxi_dlnk_pm_tomo_nointerp results (see that header for the quantity
// and the pipeline). This is what the RF_xi integrals read.
//
// The tabulated quantity (2011.06469 eq 17) is
//
//   dxi_pm/dlnk(theta) = sum_l Glpm(theta, l) * (dC_EE +- dC_BB)(k, l)
//   dlnxi_pm/dlnk      = (dxi_pm/dlnk) / xi_pm(theta),
//
// with each dC read from the cached dC_ss_dlnk_tomo_limber table (one
// Limber node per (k, l), amplitude ell_prefactor/fK) and Glpm the
// bin-averaged Legendre kernels of xi_pm_tomo.
//
// Table design: [2][shear_Npowerspectra*Ntheta][nlnk], one row per
// (xi component, tomo pair x angular bin), with nlnk =
// Ntable.dCX_dlnk_nlnk[NODES_DENSE] log-spaced k in [Ntable.dCX_dlnk_kmin,
// Ntable.dCX_dlnk_kmax]; lookups interpolate linearly in ln k and a k
// outside the grid returns 0. The fill calls the nointerp pipeline once
// per k node (each call returns every pair and angular bin).
//
// Cache invalidation:
// recomputes when any of these change:
//   cosmology.random, nuisance.random_photoz_shear, nuisance.random_ia,
//   redshift.random_shear, Ntable.random
// (allocation and grid limits rebuild on Ntable.random alone).
//
// Parameters:
//   k  - wavenumber in (Mpc/h)^-1
//   pm - 1 = xi_+, 0 = xi_-
//   nt - angular bin index (0..Ntheta-1)
//   ni - first source redshift bin
//   nj - second source redshift bin
//
// Returns:
//   dlnxi_pm/dlnk at k for (theta_nt, ni, nj); 0 outside the grid
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
  // lim = ln k grid: [0..2] = (min, max, step); [3..5] unused here
  static double lim[6];
  static int nlnk;
  static int nkc = 0;     // used coarse ln k count (= nlnk when exact)
  static double dkc = 0.; // coarse grid spacing in ln k
  static int* qidx = NULL;      // fine node -> coarse interval (uniform
  static double* qdel = NULL;   //   grids: precomputed, no search)
  static double*** tabc = NULL; // coarse dlnxi values
  static double*** cspl = NULL; // natural-cubic-spline c coefficients
  const int NSIZE = tomo.shear_Npowerspectra;
  if (NULL == table || fdiff2(cache[4], Ntable.random)) {
    nlnk = Ntable.dCX_dlnk_nlnk[NODES_DENSE];
    lim[0] = log(Ntable.dCX_dlnk_kmin);
    lim[1] = log(Ntable.dCX_dlnk_kmax);
    lim[2] = (lim[1] - lim[0]) / ((double) nlnk - 1.);
    if (table != NULL) free(table);
    table = (double***) malloc3d(2, NSIZE*Ntable.Ntheta, nlnk);

    // Coarse-grid workspace: allocations live HERE, in the Ntable
    // rebuild block; the per-cosmology refill only fills. Unlike the
    // dC tables (whose nodes are cheap), every ln k node of THIS
    // cache costs one full nointerp build - the expensive-node case
    // the coarse-exact + cubic-upsample pattern exists for. The same
    // scale-cut k knob gates both grids
    // (Ntable.dCX_dlnk_nlnk[NODES_COARSE]; 0 or out of range = exact;
    // > 3 because a natural cubic spline needs 4 nodes).
    if (qidx != NULL) { free(qidx); qidx = NULL; }
    if (qdel != NULL) { free(qdel); qdel = NULL; }
    if (tabc != NULL) { free(tabc); tabc = NULL; }
    if (cspl != NULL) { free(cspl); cspl = NULL; }
    const int nk_int = Ntable.dCX_dlnk_nlnk[NODES_COARSE];
    nkc = (nk_int > 3 && nk_int < nlnk) ? nk_int : nlnk;
    dkc = (lim[1] - lim[0]) / ((double) nkc - 1.0);
    if (nkc < nlnk) {
      qidx = (int*) malloc(sizeof(int)*nlnk);
      qdel = (double*) malloc1d(nlnk);
      for (int f=0; f<nlnk; f++) {
        // fine node f -> its coarse interval and offset: both grids
        // are uniform in ln k with shared endpoints, so the map is
        // one multiply and a cast, clamped onto the last interval
        // against a 1-ulp division overshoot near the top endpoint
        const double r = (double) f * lim[2] / dkc;
        int j = (int) r;
        if (j > nkc - 2) {
          j = nkc - 2;
        }
        qidx[f] = j;
        qdel[f] = (r - j) * dkc;
      }
      tabc = (double***) malloc3d(2, NSIZE*Ntable.Ntheta, nkc);
      cspl = (double***) malloc3d(2, NSIZE*Ntable.Ntheta, nkc);
    }
  }
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random))
  {
    if (nkc < nlnk) {
      // exact nointerp builds on the coarse ln k nodes only - the
      // expensive part - then the house natural cubic spline
      // upsamples every (xi_+/-, pair, angular bin) row onto the
      // unchanged dense ln k grid (spline_coeffs_uniform + Horner at
      // the offsets precomputed in the rebuild block above)
      //
      // the f loop stays serial: every nointerp call runs its own
      // OpenMP-parallel fills inside, so threading it here would
      // only nest parallel regions
      for (int f=0; f<nkc; f++) {
        double** tmp = dlnxi_dlnk_pm_tomo_nointerp(exp(lim[0] + f*dkc));
        for (int p=0; p<2; p++) {
          for (int nz=0; nz<NSIZE; nz++) {
            for (int i=0; i<Ntable.Ntheta; i++) {
              const int q = nz * Ntable.Ntheta + i;
              tabc[p][q][f] = tmp[p][q];
            }
          }
        }
        free(tmp);
      }
      const int nrows = NSIZE*Ntable.Ntheta;
      #pragma omp parallel for collapse(2) schedule(static)
      for (int p=0; p<2; p++) {
        for (int q=0; q<nrows; q++) {
          spline_coeffs_uniform(tabc[p][q], nkc, dkc, cspl[p][q]);
        }
      }
      // Evaluate each row's spline at every fine ln k node, in
      // Horner form. On coarse interval [j, j+1] the cubic is
      //
      //   S(x_j + u) = y_j + b u + c_j u^2 + d u^3,
      //
      // with c_j = S''(x_j)/2 from spline_coeffs_uniform, and two
      // conditions fix the remaining coefficients:
      //
      //   S(x_{j+1}) = y_{j+1} (hit the right node)  -> b
      //   S'' linear from 2 c_j to 2 c_{j+1}         -> d
      //
      // - the same evaluation spline2d_upsample_uniform uses
      // (basics.c). qidx/qdel hold each fine node's coarse interval
      // j and offset u, precomputed once in the rebuild block.
      #pragma omp parallel for collapse(3) schedule(static)
      for (int p=0; p<2; p++) {
        for (int q=0; q<nrows; q++) {
          for (int f=0; f<nlnk; f++) {
            const double* restrict y = tabc[p][q];
            const double* restrict cc = cspl[p][q];
            const int j = qidx[f];
            const double b = (y[j+1] - y[j])/dkc
                             - dkc*(cc[j+1] + 2.0*cc[j])/3.0;
            const double d = (cc[j+1] - cc[j])/(3.0*dkc);
            table[p][q][f] =
                y[j] + qdel[f]*(b + qdel[f]*(cc[j] + qdel[f]*d));
          }
        }
      }
    }
    else {
      // exact build: one nointerp call per dense ln k node (the f
      // loop stays serial - each call parallelizes internally - and
      // each call returns every row at that node)
      for (int f=0; f<nlnk; f++) {
        double** tmp = dlnxi_dlnk_pm_tomo_nointerp(exp(lim[0] + f*lim[2]));
        for (int p=0; p<2; p++) {
          for (int nz=0; nz<NSIZE; nz++) {
            for (int i=0; i<Ntable.Ntheta; i++) {
              const int q = nz * Ntable.Ntheta + i;
              table[p][q][f] = tmp[p][q];
            }
          }
        }
        free(tmp);
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
// Steps 1 and 2 combined equal one bilinear (ln k, ln l) read of the
// 2D table at each integer multipole — same knots, same weights — so
// splitting the read loses no accuracy and no per-multipole cache is
// needed.
//
// Static state, rebuilt when Ntable or the tomography change:
//   Pl[Ntheta][LMAX] - bin-averaged Legendre kernel (as in w_ks_tomo)
//   ln_ell[LMAX]     - log(l) at every integer multipole
//   dCgrid[NSIZE][N_ell], cx[NSIZE][LMAX] - work arrays the pipeline
//                      overwrites on every call
//   cmbf[LMAX]       - CMB filter, refilled when cmb or Ntable change
//
// Cache invalidation:
// Ntable.random (or a tomography-size change)
// rebuilds the static geometry; cmbf refills on cmb.random or
// Ntable.random. No cosmology key is needed: the cosmology enters only
// through the cached dC and w_ks tables read on every call.
//
// Parameters:
//   k - wavenumber in (Mpc/h)^-1
//
// Returns:
//   newly allocated [NSIZE*Ntheta] array of dlnw_ks/dlnk values (caller
//   frees), flattened as nz*Ntheta + i over (source bin nz, angular
//   bin i) with NSIZE = redshift.shear_nbin; all zeros when k is
//   outside the open interval (Ntable.dCX_dlnk_kmin,
//   Ntable.dCX_dlnk_kmax)
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
  // the kernel formulas below divide by l (and the l < lmin rows are
  // zeroed): the monopole is excluded, so the sums start at lmin = 1
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
    dCgrid = (double**) malloc2d(NSIZE, Ntable.N_ell[NODES_DENSE]);
    zero2d(dCgrid, NSIZE, Ntable.N_ell[NODES_DENSE]);
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
    // dC_ks/dlnk at this k on the dC table's own multipole log-grid.
    // la/ldx must reproduce the dC table's ln l grid (lim[0], lim[2])
    // exactly: the exact-node read here, and limber_fill_interp's
    // inverse map below, rely on the two grids being the same
    const int nell = Ntable.N_ell[NODES_DENSE];
    const double la = 0.0; // ln(l = 1): the dC table's multipole grid start
    const double ldx = (log(Ntable.LMAX + 1.) - la)/((double) nell - 1.);
    // the same fixed-weight two-row blend as the ss fill above, on
    // the dCks_ struct (one weight tk for every (bin, ell) entry)
    const double rk = (log(k) - dCks_.lim[3])/dCks_.lim[5];
    int fk = (int) rk;
    if (fk > dCks_.nlnk - 2) { // 1-ulp division overshoot near kmax
      fk = dCks_.nlnk - 2;
    }
    const double tk = rk - fk;
    // SIMDe-vectorized fixed-weight blend (limber_krow_blend), one
    // plane per source bin
    #pragma omp parallel for schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      const double* r0[1] = {dCks_.tab[nz][fk]};
      const double* r1[1] = {dCks_.tab[nz][fk+1]};
      double* out1[1] = {dCgrid[nz]};
      limber_krow_blend(1, r0, r1, out1, tk, nell);
    }
    // gather onto every integer multipole (vectorized linear interpolation)
    #pragma omp parallel for schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      const double* tab1[1] = {dCgrid[nz]};
      double* out1[1] = {cx[nz]};
      limber_fill_interp(1, tab1, out1, lmin, Ntable.LMAX, ln_ell,
                         la, 1.0/ldx, nell);
    }
    legendre_sums_filtered(NSIZE, Ntable.Ntheta, lmin, Ntable.LMAX,
                           Pl, cmbf, cx, ans);
    // warm w_ks_tomo's cached table single-threaded (one call builds
    // every bin; the third argument is the limber flag, 1 = Limber);
    // the parallel loop below then only reads it
    (void) w_ks_tomo(0, 0, 1);
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nz=0; nz<NSIZE; nz++) {
      for (int i=0; i<Ntable.Ntheta; i++) {
        const int q = nz * Ntable.Ntheta + i;
        // double guard, as in the dlnC fill (cosmo2D.c): a ~0
        // derivative stays 0 without reading w_ks, and a ~0 w_ks
        // denominator maps to 0, so the ratio never blows up where
        // the signal vanishes (w_ks_tomo's last argument is the
        // limber flag, 1 = Limber)
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
// dlnw_ks_dlnk_tomo_nointerp results (see that header for the quantity
// and the pipeline). This is what the RF_w_ks integrals read.
//
// The tabulated quantity (2011.06469 eq 17) is
//
//   dw_ks/dlnk(theta) = sum_l Pl(theta, l) * cmbf(l) * dC_ks(k, l)
//   dlnw_ks/dlnk      = (dw_ks/dlnk) / w_ks(theta),
//
// with each dC read from the cached dC_ks_dlnk_tomo_limber table (one
// Limber node per (k, l), amplitude pf1*pf2/fK), Pl the bin-averaged
// spin-0 x spin-2 Legendre kernel of w_ks_tomo, and cmbf the CMB
// beam/pixel-window filter w_ks_tomo applies.
//
// Table design: [shear_nbin*Ntheta][nlnk], one row per (source bin x
// angular bin), with nlnk = Ntable.dCX_dlnk_nlnk[NODES_DENSE] log-spaced k in
// [Ntable.dCX_dlnk_kmin, Ntable.dCX_dlnk_kmax]; lookups interpolate
// linearly in ln k and a k outside the grid returns 0. The fill calls
// the nointerp pipeline once per k node (each call returns every bin).
//
// Cache invalidation:
// recomputes when any of these change:
//   cosmology.random, nuisance.random_photoz_shear, nuisance.random_ia,
//   redshift.random_shear, Ntable.random, cmb.random
// (cmb.random is a key here, unlike the Fourier-space dC tables, because
// the CMB beam enters the w_ks projection; allocation and grid limits
// rebuild on Ntable.random alone).
//
// Parameters:
//   k  - wavenumber in (Mpc/h)^-1
//   nt - angular bin index (0..Ntheta-1)
//   ni - source redshift bin
//
// Returns:
//   dlnw_ks/dlnk at k for (theta_nt, ni); 0 outside the grid
// ---------------------------------------------------------------------------
double dlnw_ks_dlnk_tomo(
    const double k,   // wavenumber in (Mpc/h)^-1
    const int nt,     // angular bin index (0..Ntheta-1)
    const int ni      // source redshift bin
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  // lim = ln k grid: (min, max, step)
  static double lim[3];
  static int nlnk;
  static int nkc = 0;     // used coarse ln k count (= nlnk when exact)
  static double dkc = 0.; // coarse grid spacing in ln k
  static int* qidx = NULL;      // fine node -> coarse interval (uniform
  static double* qdel = NULL;   //   grids: precomputed, no search)
  static double** tabc = NULL;  // coarse dlnw values
  static double** cspl = NULL;  // natural-cubic-spline c coefficients
  const int NSIZE = redshift.shear_nbin;
  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized"); exit(1);
  }
  if (NULL == table || fdiff2(cache[4], Ntable.random)) {
    nlnk = Ntable.dCX_dlnk_nlnk[NODES_DENSE];
    lim[0] = log(Ntable.dCX_dlnk_kmin);
    lim[1] = log(Ntable.dCX_dlnk_kmax);
    lim[2] = (lim[1] - lim[0]) / ((double) nlnk - 1.);
    if (table != NULL) free(table);
    table = (double**) malloc2d(NSIZE*Ntable.Ntheta, nlnk);

    // Coarse-grid workspace: allocations live HERE, in the Ntable
    // rebuild block; the per-cosmology refill only fills. Unlike the
    // dC tables (whose nodes are cheap), every ln k node of THIS
    // cache costs one full dlnw nointerp build - the expensive-node case
    // the coarse-exact + cubic-upsample pattern exists for. The same
    // scale-cut k knob gates both grids
    // (Ntable.dCX_dlnk_nlnk[NODES_COARSE]; 0 or out of range = exact;
    // > 3 because a natural cubic spline needs 4 nodes).
    if (qidx != NULL) { free(qidx); qidx = NULL; }
    if (qdel != NULL) { free(qdel); qdel = NULL; }
    if (tabc != NULL) { free(tabc); tabc = NULL; }
    if (cspl != NULL) { free(cspl); cspl = NULL; }
    const int nk_int = Ntable.dCX_dlnk_nlnk[NODES_COARSE];
    nkc = (nk_int > 3 && nk_int < nlnk) ? nk_int : nlnk;
    dkc = (lim[1] - lim[0]) / ((double) nkc - 1.0);
    if (nkc < nlnk) {
      qidx = (int*) malloc(sizeof(int)*nlnk);
      qdel = (double*) malloc1d(nlnk);
      for (int f=0; f<nlnk; f++) {
        // fine node f -> its coarse interval and offset: both grids
        // are uniform in ln k with shared endpoints, so the map is
        // one multiply and a cast, clamped onto the last interval
        // against a 1-ulp division overshoot near the top endpoint
        const double r = (double) f * lim[2] / dkc;
        int j = (int) r;
        if (j > nkc - 2) {
          j = nkc - 2;
        }
        qidx[f] = j;
        qdel[f] = (r - j) * dkc;
      }
      tabc = (double**) malloc2d(NSIZE*Ntable.Ntheta, nkc);
      cspl = (double**) malloc2d(NSIZE*Ntable.Ntheta, nkc);
    }
  }
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], nuisance.random_photoz_shear) ||
      fdiff2(cache[2], nuisance.random_ia) ||
      fdiff2(cache[3], redshift.random_shear) ||
      fdiff2(cache[4], Ntable.random) ||
      fdiff2(cache[5], cmb.random))
  {
    if (nkc < nlnk) {
      // coarse exact nointerp builds + house cubic upsample in ln k,
      // as in dlnxi_dlnk_pm_tomo above (serial f loop for the same
      // nesting reason: each nointerp call parallelizes internally)
      for (int f=0; f<nkc; f++) {
        double* tmp = dlnw_ks_dlnk_tomo_nointerp(exp(lim[0] + f*dkc));
        for (int nz=0; nz<NSIZE; nz++) {
          for (int i=0; i<Ntable.Ntheta; i++) {
            const int q = nz * Ntable.Ntheta + i;
            tabc[q][f] = tmp[q];
          }
        }
        free(tmp);
      }
      const int nrows = NSIZE*Ntable.Ntheta;
      #pragma omp parallel for schedule(static)
      for (int q=0; q<nrows; q++) {
        spline_coeffs_uniform(tabc[q], nkc, dkc, cspl[q]);
      }
      // same Horner spline evaluation as dlnxi_dlnk_pm_tomo above
      // (b hits the right node, d makes S'' linear; qidx/qdel are
      // the precomputed interval and offset of each fine node)
      #pragma omp parallel for collapse(2) schedule(static)
      for (int q=0; q<nrows; q++) {
        for (int f=0; f<nlnk; f++) {
          const double* restrict y = tabc[q];
          const double* restrict cc = cspl[q];
          const int j = qidx[f];
          const double b = (y[j+1] - y[j])/dkc
                           - dkc*(cc[j+1] + 2.0*cc[j])/3.0;
          const double d = (cc[j+1] - cc[j])/(3.0*dkc);
          table[q][f] = y[j] + qdel[f]*(b + qdel[f]*(cc[j] + qdel[f]*d));
        }
      }
    }
    else {
      // exact build: one nointerp call per dense ln k node, as in
      // dlnxi_dlnk_pm_tomo above
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
// (RF is normalized by its kk -> infty value, so RF runs from 0 to 1;
// alpha = the kept response fraction. The functions below tabulate
// RF(kmax, theta) on a kmax grid - the root-find for kk happens
// downstream. See the file preamble.)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// RF of xi_pm over kmax, exact from the tabulated response.
//
// RF(kmax) is the fraction of the total absolute log response below
// kmax,
//
//   RF(kmax) = int_(-inf)^(ln kmax) |dlnX/dlnk| dlnk
//            / int_(-inf)^(+inf)    |dlnX/dlnk| dlnk,
//
// and the response it integrates is the dlnxi_dlnk_pm_tomo cache:
// a table on the uniform Ntable.dCX_dlnk grid in ln k, read by
// LINEAR interpolation and exactly zero outside the grid. The
// integrand is therefore piecewise linear, and both integrals are
// CLOSED FORM (see scuts_abs_lin_full/_part above): no quadrature
// rule, no error.
//
// Per (xi_+/xi_-, pair, angular bin) row:
//
//   sample the response at the grid's own ln k nodes
//     -> cumulative sum of the per-interval |v| integrals
//        (trapezoids, sign-crossing triangles)
//     -> denominator = the full cumulative
//     -> every requested kmax = prefix + the cut last piece,
//        found by one multiply and a cast (uniform grid, no search)
//
// No vanishing-denominator guard: the xi_+/- responses mix the EE
// rows into every entry, so the full cumulative is positive whenever
// the tabulated response is not identically zero.
//
// Cache invalidation:
// none (stateless); each call first builds (or reuses) the cached
// dlnxi table with one single-threaded read, and the parallel rows
// then sample it read-only.
//
// Parameters:
//   lnkmaxx - ln kmax values (length nkmax), k in (Mpc/h)^-1
//   nkmax   - number of ln kmax values
//   NSIZE   - number of tomo shear power spectra
//   table   - output [2][NSIZE][nkmax][Ntheta]: xi+ and xi-
//
// Returns:
//   void (table filled for every kmax and angular bin)
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
  const int nlnk = Ntable.dCX_dlnk_nlnk[NODES_DENSE];
  const double lnk0 = log(Ntable.dCX_dlnk_kmin);
  const double dx = (log(Ntable.dCX_dlnk_kmax) - lnk0)
                    / ((double) nlnk - 1.0);
  const double lnk1 = lnk0 + (nlnk - 1)*dx;
  double* kv = (double*) malloc1d(nlnk); // the grid's own k nodes
  for (int f = 0; f < nlnk; f++) {
    kv[f] = exp(lnk0 + f*dx);
  }
  // build the k-cached dlnxi table single-threaded before the parallel
  // region below reads it
  (void) dlnxi_dlnk_pm_tomo(1.0, 1, 0, Z1(0), Z2(0));
  #pragma omp parallel
  {
    double* prof = (double*) malloc1d(nlnk); // one row's response
    double* cum  = (double*) malloc1d(nlnk); // its running integral
    #pragma omp for collapse(3) schedule(static)
    for (int sp = 0; sp < 2; sp++) {       // 0 = xi_+, 1 = xi_-
      for (int nz = 0; nz < NSIZE; nz++) {
        for (int nt = 0; nt < Ntable.Ntheta; nt++) {
          const int Z1NZ = Z1(nz);
          const int Z2NZ = Z2(nz);
          const int pm = 1 - sp; // the lookup's flag: 1 = xi_+
          for (int f = 0; f < nlnk; f++) {
            prof[f] = dlnxi_dlnk_pm_tomo(kv[f], pm, nt, Z1NZ, Z2NZ);
          }
          cum[0] = 0.0;
          for (int f = 1; f < nlnk; f++) {
            cum[f] = cum[f-1] + scuts_abs_lin_full(prof[f-1], prof[f], dx);
          }
          const double dden = cum[nlnk-1];
          for (int m = 0; m < nkmax; m++) {
            const double L = lnkmaxx[m];
            double num;
            if (L <= lnk0) {
              num = 0.0;
            }
            else if (L >= lnk1) {
              num = dden;
            }
            else {
              const double r = (L - lnk0)/dx;
              int j = (int) r; // interval's left node (uniform grid)
              if (j > nlnk - 2) { // 1-ulp division overshoot near lnk1
                j = nlnk - 2;
              }
              num = cum[j] +
                    scuts_abs_lin_part(prof[j], prof[j+1], dx, (r - j)*dx);
            }
            table[sp][nz][m][nt] = num/dden;
          }
        }
      }
    }
    free(prof);
    free(cum);
  } // end of the parallel region
  free(kv);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// RF of w_ks over kmax, exact from the tabulated response.
//
// RF(kmax) is the fraction of the total absolute log response below
// kmax,
//
//   RF(kmax) = int_(-inf)^(ln kmax) |dlnX/dlnk| dlnk
//            / int_(-inf)^(+inf)    |dlnX/dlnk| dlnk,
//
// and the response it integrates is the dlnw_ks_dlnk_tomo cache:
// a table on the uniform Ntable.dCX_dlnk grid in ln k, read by
// LINEAR interpolation and exactly zero outside the grid. The
// integrand is therefore piecewise linear, and both integrals are
// CLOSED FORM (see scuts_abs_lin_full/_part above): no quadrature
// rule, no error.
//
// Per (source bin, angular bin) row:
//
//   sample the response at the grid's own ln k nodes
//     -> cumulative sum of the per-interval |v| integrals
//        (trapezoids, sign-crossing triangles)
//     -> denominator = the full cumulative
//     -> every requested kmax = prefix + the cut last piece,
//        found by one multiply and a cast (uniform grid, no search)
//
// No vanishing-denominator guard: w_ks has no identically-zero rows
// (there is no BB analog here), so the full cumulative is positive
// whenever the tabulated response is not identically zero.
//
// Cache invalidation:
// none (stateless); each call first builds (or reuses) the cached
// dlnw_ks table with one single-threaded read, and the parallel
// rows then sample it read-only.
//
// Parameters:
//   lnkmaxx - ln kmax values (length nkmax), k in (Mpc/h)^-1
//   nkmax   - number of ln kmax values
//   NSIZE   - number of source tomographic bins (= shear_nbin)
//   table   - output [NSIZE][nkmax][Ntheta]
//
// Returns:
//   void (table filled for every kmax and angular bin)
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
  const int nlnk = Ntable.dCX_dlnk_nlnk[NODES_DENSE];
  const double lnk0 = log(Ntable.dCX_dlnk_kmin);
  const double dx = (log(Ntable.dCX_dlnk_kmax) - lnk0)
                    / ((double) nlnk - 1.0);
  const double lnk1 = lnk0 + (nlnk - 1)*dx;
  double* kv = (double*) malloc1d(nlnk); // the grid's own k nodes
  for (int f = 0; f < nlnk; f++) {
    kv[f] = exp(lnk0 + f*dx);
  }
  // build the k-cached dlnw_ks table single-threaded before the
  // parallel region below reads it
  (void) dlnw_ks_dlnk_tomo(1.0, 0, 0);
  #pragma omp parallel
  {
    double* prof = (double*) malloc1d(nlnk); // one row's response
    double* cum  = (double*) malloc1d(nlnk); // its running integral
    #pragma omp for collapse(2) schedule(static)
    for (int nz = 0; nz < NSIZE; nz++) {
      for (int nt = 0; nt < Ntable.Ntheta; nt++) {
        for (int f = 0; f < nlnk; f++) {
          prof[f] = dlnw_ks_dlnk_tomo(kv[f], nt, nz);
        }
        cum[0] = 0.0;
        for (int f = 1; f < nlnk; f++) {
          cum[f] = cum[f-1] + scuts_abs_lin_full(prof[f-1], prof[f], dx);
        }
        const double dden = cum[nlnk-1];
        for (int m = 0; m < nkmax; m++) {
          const double L = lnkmaxx[m];
          double num;
          if (L <= lnk0) {
            num = 0.0;
          }
          else if (L >= lnk1) {
            num = dden;
          }
          else {
            const double r = (L - lnk0)/dx;
            int j = (int) r; // interval's left node (uniform grid)
            if (j > nlnk - 2) { // 1-ulp division overshoot near lnk1
              j = nlnk - 2;
            }
            num = cum[j] +
                  scuts_abs_lin_part(prof[j], prof[j+1], dx, (r - j)*dx);
          }
          table[nz][m][nt] = num/dden;
        }
      }
    }
    free(prof);
    free(cum);
  } // end of the parallel region
  free(kv);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
