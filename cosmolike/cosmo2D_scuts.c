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
// nlnk = Ntable.dCX_dlnk_nlnk log-spaced k in [Ntable.dCX_dlnk_kmin,
// Ntable.dCX_dlnk_kmax] and nell = Ntable.N_ell log-spaced multipoles
// covering every l >= 1; lookups interpolate bilinearly in (ln k, ln l)
// and a (k, l) outside the table returns 0.
//
// When the internal coarse grids are active (Ntable.N_ell_internal on
// the ell axis, Ntable.dCX_dlnk_nlnk_internal on ln k), the exact
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
    // coarsens independently: Ntable.N_ell_internal on the ell axis
    // (smooth) and Ntable.dCX_dlnk_nlnk_internal on the ln k axis
    // (where the BAO wiggles live, so its default stays exact). An
    // axis whose knob is 0 or out of range keeps its exact count.
    if (lnkc != NULL) { free(lnkc); lnkc = NULL; }
    if (lxc  != NULL) { free(lxc);  lxc  = NULL; }
    if (tabc != NULL) { free(tabc); tabc = NULL; }
    const int nk_int = Ntable.dCX_dlnk_nlnk_internal;
    const int nl_int = Ntable.N_ell_internal;
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
      // knob defaults to exact (see structs.c).
      // ---------------------------------------------------------------
      dC_ss_dlnk_tomo_limber_work(lnkc, nkc, lxc, nlc,
                                tomo.shear_Npowerspectra, 0, tabc);

      // one tensor-product bicubic upsample per stored plane; an
      // axis left exact passes through (near-)unchanged
      for (int c=0; c<2; c++) {
        for (int q=0; q<tomo.shear_Npowerspectra; q++) {
          spline2d_upsample_uniform(tabc[c][q], nkc, nlc, dkc, dlc,
                                    table[c][q], nlnk, nell);
        }
      }
    }
    else {
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
// quadrature and divides each dC row in place (see the dC_ss_dlnk lookup function
// above for the tabulated node amplitude and the _work header for the
// fill).
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
// nlnk = Ntable.dCX_dlnk_nlnk log-spaced k in [Ntable.dCX_dlnk_kmin,
// Ntable.dCX_dlnk_kmax] and nell = Ntable.N_ell log-spaced multipoles
// covering every l >= 1 (the same grid as dC_ss_dlnk_tomo_limber);
// lookups interpolate bilinearly in (ln k, ln l) and a (k, l) outside
// the table returns 0.
//
// When the internal coarse grids are active (Ntable.N_ell_internal on
// the ell axis, Ntable.dCX_dlnk_nlnk_internal on ln k), the exact
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
    // coarsens independently: Ntable.N_ell_internal on the ell axis
    // (smooth) and Ntable.dCX_dlnk_nlnk_internal on the ln k axis
    // (where the BAO wiggles live, so its default stays exact). An
    // axis whose knob is 0 or out of range keeps its exact count.
    if (lnkc != NULL) { free(lnkc); lnkc = NULL; }
    if (lxc  != NULL) { free(lxc);  lxc  = NULL; }
    if (tabc != NULL) { free(tabc); tabc = NULL; }
    const int nk_int = Ntable.dCX_dlnk_nlnk_internal;
    const int nl_int = Ntable.N_ell_internal;
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
      // the BAO wiggles and defaults to exact).
      dC_ss_dlnk_tomo_limber_work(lnkc, nkc, lxc, nlc,
                                tomo.shear_Npowerspectra, 1, tabc);

      // one tensor-product bicubic upsample per stored plane; an
      // axis left exact passes through (near-)unchanged
      for (int c=0; c<2; c++) {
        for (int q=0; q<tomo.shear_Npowerspectra; q++) {
          spline2d_upsample_uniform(tabc[c][q], nkc, nlc, dkc, dlc,
                                    table[c][q], nlnk, nell);
        }
      }
    }
    else {
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
// When the internal coarse grids are active (Ntable.N_ell_internal on
// the ell axis, Ntable.dCX_dlnk_nlnk_internal on ln k), the exact
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
    // coarsens independently: Ntable.N_ell_internal on the ell axis
    // (smooth) and Ntable.dCX_dlnk_nlnk_internal on the ln k axis
    // (where the BAO wiggles live, so its default stays exact). An
    // axis whose knob is 0 or out of range keeps its exact count.
    if (lnkc != NULL) { free(lnkc); lnkc = NULL; }
    if (lxc  != NULL) { free(lxc);  lxc  = NULL; }
    if (tabc != NULL) { free(tabc); tabc = NULL; }
    const int nk_int = Ntable.dCX_dlnk_nlnk_internal;
    const int nl_int = Ntable.N_ell_internal;
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
      // the BAO wiggles and defaults to exact).
      dC_ks_dlnk_tomo_limber_work(lnkc, nkc, lxc, nlc,
                                redshift.shear_nbin, 0, tabc);

      // one tensor-product bicubic upsample per stored plane; an
      // axis left exact passes through (near-)unchanged
      for (int nz=0; nz<redshift.shear_nbin; nz++) {
        spline2d_upsample_uniform(tabc[nz], nkc, nlc, dkc, dlc,
                                  table[nz], nlnk, nell);
      }
    }
    else {
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
// per-bin quadrature and divides each dC row in place (see the dC_ks_dlnk
// lookup function above for the tabulated node amplitude).
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
// bin), with nlnk = Ntable.dCX_dlnk_nlnk log-spaced k in
// [Ntable.dCX_dlnk_kmin, Ntable.dCX_dlnk_kmax] and nell = Ntable.N_ell
// log-spaced multipoles covering every l >= 1 (the same grid as
// dC_ks_dlnk_tomo_limber); lookups interpolate bilinearly in
// (ln k, ln l) and a (k, l) outside the table returns 0.
//
// When the internal coarse grids are active (Ntable.N_ell_internal on
// the ell axis, Ntable.dCX_dlnk_nlnk_internal on ln k), the exact
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
    // coarsens independently: Ntable.N_ell_internal on the ell axis
    // (smooth) and Ntable.dCX_dlnk_nlnk_internal on the ln k axis
    // (where the BAO wiggles live, so its default stays exact). An
    // axis whose knob is 0 or out of range keeps its exact count.
    if (lnkc != NULL) { free(lnkc); lnkc = NULL; }
    if (lxc  != NULL) { free(lxc);  lxc  = NULL; }
    if (tabc != NULL) { free(tabc); tabc = NULL; }
    const int nk_int = Ntable.dCX_dlnk_nlnk_internal;
    const int nl_int = Ntable.N_ell_internal;
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
      // the BAO wiggles and defaults to exact).
      dC_ks_dlnk_tomo_limber_work(lnkc, nkc, lxc, nlc,
                                redshift.shear_nbin, 1, tabc);

      // one tensor-product bicubic upsample per stored plane; an
      // axis left exact passes through (near-)unchanged
      for (int nz=0; nz<redshift.shear_nbin; nz++) {
        spline2d_upsample_uniform(tabc[nz], nkc, nlc, dkc, dlc,
                                  table[nz], nlnk, nell);
      }
    }
    else {
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
// Batch computation of the response function (2011.06469 eq 17)
//
//   RF(kmax, l) = int_{-infty}^{ln kmax} dlnk |dlnC_ss/dlnk|
//                 / int_{-infty}^{+infty} dlnk |dlnC_ss/dlnk|
//
// on a (ln kmax, ell) grid, for every tomographic pair. Both integrals
// map onto t in (0, 1] — the numerator through lnk = ln kmax - (1-t)/t,
// the denominator through lnk = +-(1-t)/t — and use a fixed
// Gauss-Legendre rule with the nodes and weights precomputed into
// plain arrays (no GSL integrand callback per point).
//
// The denominator does not depend on kmax, so one thread team first
// fills one denominator value per (tomo pair, ell) and then, after the
// loop's implicit barrier, accumulates the numerator for every
// (pair, kmax, ell) output and divides. Every integrand evaluation reads
// the cached dlnC_ss_dlnk_tomo_limber table, built once, single-threaded,
// before the parallel region.
//
// Cache invalidation:
// none (stateless); it only warms the cached dlnC
// table it reads.
//
// Parameters:
//   lnkmaxx - ln kmax values (length nkmax), k in (Mpc/h)^-1
//   nkmax   - number of ln kmax values
//   lx      - multipole values (length nl)
//   nl      - number of multipole values
//   NSIZE   - number of tomographic shear power spectra
//   table   - output [2][NSIZE][nkmax][nl], RF at each (kmax, l): EE and BB
//
// Returns:
//   nothing; the result is written into table
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
  // Gauss-Legendre nodes and weights on t in [1e-5, 1] as plain arrays.
  // Both RF integrals map onto t via lnk = (const) -/+ (1-t)/t, whose
  // Jacobian dlnk = dt/t^2 is the wt = wq/t^2 factor in the loops; the
  // 1e-5 lower limit keeps 1/t^2 finite and drops only an exactly-zero
  // tail (|lnk| > ~1e5, far outside the k range where the cached dln
  // tables are nonzero)
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
  // the denominator's k nodes depend on nothing: precompute them once.
  // kd1/kd2 realize the split of int_{-inf}^{+inf} dlnk at lnk = 0,
  // i.e. k = 1 (Mpc/h)^-1: kd1 = exp(+(1-t)/t) covers [0, +inf) and
  // kd2 = exp(-(1-t)/t) the mirror half
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
        // a vanishing denominator row (the BB spectrum under NLA is
        // identically 0) writes 0, never 0/0 = NaN
        table[0][nz][m][i] = (fabs(den[0][nz][i]) > 1e-300) ?
                             sEE/den[0][nz][i] : 0.0;
        table[1][nz][m][i] = (fabs(den[1][nz][i]) > 1e-300) ?
                             sBB/den[1][nz][i] : 0.0;
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
//
// Cache invalidation:
// none (stateless); it only warms the cached dlnC
// table it reads.
//
// Parameters:
//   lnkmaxx - ln kmax values (length nkmax), k in (Mpc/h)^-1
//   nkmax   - number of ln kmax values
//   lx      - multipole values (length nl)
//   nl      - number of multipole values
//   NSIZE   - number of source tomographic bins (= shear_nbin)
//   table   - output [NSIZE][nkmax][nl], RF at each (kmax, l)
//
// Returns:
//   nothing; the result is written into table
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
  // Gauss-Legendre nodes and weights on t in [1e-5, 1] as plain arrays.
  // Both RF integrals map onto t via lnk = (const) -/+ (1-t)/t, whose
  // Jacobian dlnk = dt/t^2 is the wt = wq/t^2 factor in the loops; the
  // 1e-5 lower limit keeps 1/t^2 finite and drops only an exactly-zero
  // tail (|lnk| > ~1e5, far outside the k range where the cached dln
  // tables are nonzero)
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
  // the denominator's k nodes depend on nothing: precompute them once.
  // kd1/kd2 realize the split of int_{-inf}^{+inf} dlnk at lnk = 0,
  // i.e. k = 1 (Mpc/h)^-1: kd1 = exp(+(1-t)/t) covers [0, +inf) and
  // kd2 = exp(-(1-t)/t) the mirror half
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
        // a vanishing denominator writes 0, never 0/0 = NaN
        table[nz][m][i] = (fabs(den[nz][i]) > 1e-300) ?
                          sKS/den[nz][i] : 0.0;
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
    // dC_ss/dlnk at this k on the dC table's own multipole log-grid.
    // la/ldx must reproduce the dC table's ln l grid (lim[0], lim[2])
    // exactly: the exact-node read here, and limber_fill_interp's
    // inverse map below, rely on the two grids being the same
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
// Ntable.dCX_dlnk_nlnk log-spaced k in [Ntable.dCX_dlnk_kmin,
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
    // dC_ks/dlnk at this k on the dC table's own multipole log-grid.
    // la/ldx must reproduce the dC table's ln l grid (lim[0], lim[2])
    // exactly: the exact-node read here, and limber_fill_interp's
    // inverse map below, rely on the two grids being the same
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
// angular bin), with nlnk = Ntable.dCX_dlnk_nlnk log-spaced k in
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
// Batch computation of the real-space response function (2011.06469 eq 17)
//
//   RF(kmax, theta) = int_{-infty}^{ln kmax} dlnk |dlnxi_pm/dlnk|
//                     / int_{-infty}^{+infty} dlnk |dlnxi_pm/dlnk|
//
// on a ln kmax grid, for every tomographic pair and angular bin. Both
// integrals map onto t in (0, 1] — the numerator through
// lnk = ln kmax - (1-t)/t, the denominator through lnk = +-(1-t)/t — and
// use a fixed Gauss-Legendre rule with the nodes and weights
// precomputed into plain arrays (no GSL integrand callback per point).
//
// The denominator does not depend on kmax, so one thread team first fills
// one denominator value per (tomo pair, angular bin) and then, after the
// loop's implicit barrier, accumulates the numerator for every
// (pair, kmax, angular bin) output and divides. Every integrand
// evaluation reads the k-cached dlnxi_dlnk_pm_tomo table, built once,
// single-threaded, before the parallel region.
//
// Cache invalidation:
// none (stateless); it only warms the k-cached
// dlnxi table it reads.
//
// Parameters:
//   lnkmaxx - ln kmax values (length nkmax), k in (Mpc/h)^-1
//   nkmax   - number of ln kmax values
//   NSIZE   - number of tomographic shear power spectra
//   table   - output [2][NSIZE][nkmax][Ntheta], RF at each (kmax, theta)
//             for both xi components
//
// Returns:
//   nothing; the result is written into table
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
  // Gauss-Legendre nodes and weights on t in [1e-5, 1] as plain arrays.
  // Both RF integrals map onto t via lnk = (const) -/+ (1-t)/t, whose
  // Jacobian dlnk = dt/t^2 is the wt = wq/t^2 factor in the loops; the
  // 1e-5 lower limit keeps 1/t^2 finite and drops only an exactly-zero
  // tail (|lnk| > ~1e5, far outside the k range where the cached dln
  // tables are nonzero)
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
  // the denominator's k nodes depend on nothing: precompute them once.
  // kd1/kd2 realize the split of int_{-inf}^{+inf} dlnk at lnk = 0,
  // i.e. k = 1 (Mpc/h)^-1: kd1 = exp(+(1-t)/t) covers [0, +inf) and
  // kd2 = exp(-(1-t)/t) the mirror half
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
        // no vanishing-denominator guard, unlike the Fourier workers:
        // their zero rows come from dlnC_BB = 0 under NLA, while xi+/-
        // mix the EE rows into every entry, so den > 0 whenever the
        // tabulated response is not identically zero
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
//
// Cache invalidation:
// none (stateless); it only warms the k-cached
// dlnw_ks table it reads.
//
// Parameters:
//   lnkmaxx - ln kmax values (length nkmax), k in (Mpc/h)^-1
//   nkmax   - number of ln kmax values
//   NSIZE   - number of source tomographic bins (= shear_nbin)
//   table   - output [NSIZE][nkmax][Ntheta], RF at each (kmax, theta)
//
// Returns:
//   nothing; the result is written into table
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
  // Gauss-Legendre nodes and weights on t in [1e-5, 1] as plain arrays.
  // Both RF integrals map onto t via lnk = (const) -/+ (1-t)/t, whose
  // Jacobian dlnk = dt/t^2 is the wt = wq/t^2 factor in the loops; the
  // 1e-5 lower limit keeps 1/t^2 finite and drops only an exactly-zero
  // tail (|lnk| > ~1e5, far outside the k range where the cached dln
  // tables are nonzero)
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
  // the denominator's k nodes depend on nothing: precompute them once.
  // kd1/kd2 realize the split of int_{-inf}^{+inf} dlnk at lnk = 0,
  // i.e. k = 1 (Mpc/h)^-1: kd1 = exp(+(1-t)/t) covers [0, +inf) and
  // kd2 = exp(-(1-t)/t) the mirror half
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
        // no vanishing-denominator guard, unlike the Fourier workers:
        // w_ks has no identically-zero component (the NLA dlnC_BB = 0
        // rows), so den > 0 whenever the tabulated response is not
        // identically zero
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
