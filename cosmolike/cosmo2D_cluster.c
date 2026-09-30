// ============================================================================
// cosmo2D_cluster.c: ANGULAR STATISTICS AND NUMBER COUNTS OF GALAXY CLUSTERS
// ============================================================================
//
// The cluster part of the DES 4x2pt + N analysis, written on the cosmo2D.c
// design. Model: DES Y6 methods paper, arXiv 2503.13631 (the equation
// numbers below are that paper's); the Y1 switches of structs_cluster.h
// recover arXiv 2008.10757. cosmo2D_cluster.h is the contract.
//
// What this file computes:
//
//   C_cs, C_cc, C_cg   Limber angular spectra (eq 10) of cluster lensing,
//                      cluster clustering and cluster-galaxy clustering
//   w_gammat_cluster_tomo, w_cc_tomo, w_cg_tomo
//                      full-sky, angular-bin-averaged real-space statistics
//                      (eqs 11 and 12)
//   N_cluster_tomo     expected number counts (eq 16)
//
// ARCHITECTURE MAP (the same chain as cosmo2D.c):
//
//   w_*_tomo                      one cached real-space block per statistic
//     -> l = 1 .. LMIN_tab - 1:   C_xx_tomo_limber_batch_rows, the exact
//                                 Limber quadrature at every integer l
//     -> l = LMIN_tab .. LMAX-1:  C_xx_tomo_limber_table (exact quadrature
//                                 on a log-l grid, cubic spline onto a
//                                 denser log-l grid) read by
//                                 limber_fill_interp (cosmo2D.c, AVX2
//                                 gathers)
//     -> Legendre sum against the bin-averaged kernels
//
//   C_xx_tomo_limber_batch_rows
//     -> create_cosmo_nodes_cluster  Gauss-Legendre nodes per cluster bin
//                                    in TWO panels: the bin's support and
//                                    its foreground (magnification)
//     -> C_xx_tomo_limber_work       weights per node, spectra per (node,
//                                    l), one vectorized sum per (row, l)
//
// Units (the library's): chi and f_K in c/H0, k in (c/H0)^-1, P(k) in
// (c/H0)^3, cluster number densities in (c/H0)^-3, survey area in deg^2.
//
// Determinism: every lazily built table is refilled outside parallel
// regions, every output number is one serial sum inside one thread, and no
// reduction crosses threads, so no result depends on OMP_NUM_THREADS.
// ============================================================================

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <gsl/gsl_integration.h>

#include "basics.h"
#include "bias.h"
#include "cosmo3D.h"
#include "cosmo2D.h"
#include "IA.h"
#include "radial_weights.h"
#include "redshift_spline.h"
#include "structs.h"

#include "structs_cluster.h"
#include "redshift_spline_cluster.h"
#include "radial_weights_cluster.h"
#include "halo_cluster.h"
#include "cosmo2D_cluster.h"

#include "log.c/src/log.h"



// ============================================================================
// [SECTION] CONSTANTS AND SMALL HELPERS
// ============================================================================

// Lowest redshift of the foreground (magnification) panel. It is the floor
// the core puts under every line-of-sight integral that reaches the
// observer (amax_source and amax_lens in redshift_spline.c): it keeps
// a < 1, where the lensing kernels are defined.
static const double CLUSTER_FOREGROUND_ZMIN = 0.001;

// Capacity of every cache-key array of this file (each statistic uses
// fewer keys; see the CACHE KEYS section).
#define CLUSTER_NKEYS_MAX 24


// ---------------------------------------------------------------------------
// The bits of a double as a uint64 cache key. Two keys are equal exactly
// when the two doubles are identical, so a switch stored as a double (the
// magnification coefficient C_c, the survey area) refills a table the
// moment it changes, even if no setter drew a new random key.
// ---------------------------------------------------------------------------
static uint64_t double_bits_key(const double x)
{
  uint64_t bits;
  memcpy(&bits, &x, sizeof(bits));
  return bits;
}


// ---------------------------------------------------------------------------
// 1 when any key differs from the key the table was built with (fdiff2,
// the core's comparison of uint64 cache keys), 0 otherwise.
// ---------------------------------------------------------------------------
static int keys_changed(
    const uint64_t* cache,  // keys the table was built with
    const uint64_t* keys,   // current keys
    const int nkeys
  )
{
  int changed = 0;
  for (int k = 0; k < nkeys; k++) {
    if (fdiff2(cache[k], keys[k])) {
      changed = 1;
    }
  }
  return changed;
}


// ---------------------------------------------------------------------------
// Record the keys a table was just built with.
// ---------------------------------------------------------------------------
static void keys_stamp(
    uint64_t* cache,       // output: keys the table is now built with
    const uint64_t* keys,  // current keys
    const int nkeys
  )
{
  for (int k = 0; k < nkeys; k++) {
    cache[k] = keys[k];
  }
}


// ---------------------------------------------------------------------------
// Curved-sky (extended Limber) multipole prefactors (1812.05995 eqs
// 74-79), the formulas of cosmo2D.c. The Limber kernel is evaluated at
// k = (l + 1/2)/f_K, and each projected field carries the exact factor of
// its spin:
//
//   magnification  l (l+1)/(l+1/2)^2: the angular Laplacian eigenvalue
//                  l(l+1) over its flat-sky value (l+1/2)^2
//   shear          sqrt((l-1) l (l+1) (l+2))/(l+1/2)^2: two covariant
//                  derivatives of the lensing potential (spin 2); exactly
//                  0 at l = 1, where a spin-2 field has no multipole
//
// Both tend to 1 for l >> 1 (the flat-sky limit).
// ---------------------------------------------------------------------------
static double ell_prefactor_magnification(const double l)
{
  const double ell = l + 0.5;
  return l*(l + 1.0)/(ell*ell);
}


static double ell_prefactor_shear(const double l)
{
  const double ell = l + 0.5;
  const double spin2_product = (l - 1.0)*l*(l + 1.0)*(l + 2.0);
  // the guard keeps sqrt away from a negative rounding of 0 at l = 1
  if (spin2_product > 0.0) {
    return sqrt(spin2_product)/(ell*ell);
  }
  return 0.0;
}


// ---------------------------------------------------------------------------
// Gauss-Legendre table shared by every quadrature of this file (both
// Limber panels and the counts). Its size is always one GSL has
// precomputed (the house rule); Ntable.high_def_integration climbs the
// ladder. The smallest rung must resolve the erf edges of the selection
// kernels <phi_i|z> (width sigma_z ~ 0.006 (1+z)) on a cluster bin.
//
// Cache invalidation: rebuilt when Ntable.random changes.
// ---------------------------------------------------------------------------
static const gsl_integration_glfixed_table* limber_gl_table_cluster(void)
{
  static gsl_integration_glfixed_table* w = NULL;
  static uint64_t cache_ntable = 0;

  if (NULL == w || fdiff2(cache_ntable, Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);

    // predefined GSL tables only
    int nodes_per_panel = 1024;
    if (0 == hdi) {
      nodes_per_panel = 128;
    }
    else if (1 == hdi) {
      nodes_per_panel = 256;
    }
    else if (2 == hdi) {
      nodes_per_panel = 512;
    }

    if (w != NULL) {
      gsl_integration_glfixed_table_free(w);
    }
    w = malloc_gslint_glfixed(nodes_per_panel);
    cache_ntable = Ntable.random;
  }
  return w;
}



// ============================================================================
// [SECTION] QUADRATURE NODES (PRIVATE COPY OF cosmo2D.c's cosmo_nodes)
// ============================================================================
//
// The Limber integrals run over Gauss-Legendre nodes in the scale factor.
// Everything that depends on the node alone (a, weight, f_K, D, H/H0,
// dchi/da) is computed once per node and reused for every (row, l). This
// is cosmo2D.c's cosmo_nodes, copied here so the core exports nothing for
// clusters.
//
// Two panels per cluster bin ni, concatenated into one node list:
//
//   support    [amin_cluster(ni), amax_cluster(ni)]: where <phi_ni|z> is
//              nonzero; the density terms (b W_c, W_c) live here
//   foreground [amax_cluster(ni), a(z = CLUSTER_FOREGROUND_ZMIN)]: the
//              cluster magnification kernel W_mag,c extends over the whole
//              foreground, down to the observer, not only over the bin
//
// Nodes 0 .. nsupport-1 are the support panel, the rest the foreground.
// Each panel is a full Gauss-Legendre rule on its own interval, so the
// support panel (and every density term) is the same whether or not the
// foreground is present: switching magnification off drops the foreground
// panel without moving a single support node, so C_c -> 0 is a smooth
// limit (the lens-bin range of the core, amax_lens in redshift_spline.c,
// instead jumps when b_mag crosses 0). Without magnification the
// foreground carries no signal: the density kernels vanish there and every
// integrand has a cluster leg.

typedef struct {
  int npts;       // number of nodes (support + foreground)
  int nsupport;   // nodes 0 .. nsupport-1 lie on the cluster-bin support
  double** data;  // data[CN_*][p]: node quantities (malloc2d)
} cosmo_nodes;

enum {
  CN_A = 0,     // scale factor a
  CN_WT,        // Gauss-Legendre weight of the node's panel
  CN_FK,        // comoving distance chi(a) (= f_K, flat cosmology)
  CN_GROWFAC,   // linear growth factor D(a)
  CN_HOVERH0,   // H(a)/H0
  CN_DCHIDA,    // dchi/da (the Limber measure dchi/da / f_K^2)
  CN_NPARAMS    // number of columns
};


// ---------------------------------------------------------------------------
// Fill the nodes of one Gauss-Legendre panel [amin, amax] into cn, starting
// at node index offset. Single-threaded: it performs the first (lazy)
// initialization of the chi_all, growfac and hoverh0v2 tables.
// ---------------------------------------------------------------------------
static void fill_cosmo_nodes_panel(
    cosmo_nodes* cn,                         // nodes being filled
    const int offset,                        // index of the panel's first node
    const double amin,                       // panel lower bound in a
    const double amax,                       // panel upper bound in a
    const gsl_integration_glfixed_table* w   // Gauss-Legendre rule
  )
{
  const int nodes = (int) w->n;

  for (int q = 0; q < nodes; q++) {
    const int p = offset + q;

    gsl_integration_glfixed_point(amin,
                                  amax,
                                  q,
                                  &cn->data[CN_A][p],
                                  &cn->data[CN_WT][p],
                                  w);

    const double a         = cn->data[CN_A][p];
    const struct chis cdca = chi_all(a);

    cn->data[CN_FK][p]      = cdca.chi;
    cn->data[CN_GROWFAC][p] = growfac(a);
    cn->data[CN_HOVERH0][p] = hoverh0v2(a, cdca.dchida);
    cn->data[CN_DCHIDA][p]  = cdca.dchida;
  }
}


// ---------------------------------------------------------------------------
// Nodes of cluster bin ni: the support panel, then (with_foreground = 1)
// the foreground panel. If the support already reaches the foreground floor
// the support is clipped there and the foreground panel is dropped.
//
// Returns: the nodes; the caller releases them (free_cosmo_nodes_cluster_all)
// ---------------------------------------------------------------------------
static cosmo_nodes create_cosmo_nodes_cluster(
    const int ni,                            // cluster redshift bin
    const int with_foreground,               // 1: add the foreground panel
    const gsl_integration_glfixed_table* w   // Gauss-Legendre rule per panel
  )
{
  const double a_foreground_max = 1.0/(1.0 + CLUSTER_FOREGROUND_ZMIN);

  // support of <phi_ni|z> in scale factor: far edge, near edge
  const double a_far = amin_cluster(ni);
  double a_near      = amax_cluster(ni);

  int use_foreground = with_foreground;
  if (a_near >= a_foreground_max) {
    a_near = a_foreground_max;
    use_foreground = 0;
  }

  if (!(a_far > 0.0) || !(a_far < a_near)) {
    log_fatal("invalid support of cluster bin %d: a = [%e, %e]",
      ni, a_far, a_near);
    exit(1);
  }

  const int nodes_per_panel = (int) w->n;

  cosmo_nodes cn;
  cn.nsupport = nodes_per_panel;
  cn.npts     = nodes_per_panel;
  if (1 == use_foreground) {
    cn.npts = 2*nodes_per_panel;
  }
  cn.data = (double**) malloc2d(CN_NPARAMS, cn.npts);

  fill_cosmo_nodes_panel(&cn, 0, a_far, a_near, w);

  if (1 == use_foreground) {
    fill_cosmo_nodes_panel(&cn, cn.nsupport, a_near, a_foreground_max, w);
  }
  return cn;
}


// ---------------------------------------------------------------------------
// Nodes of every cluster bin, cn_all[0 .. zdist_nbin-1].
//
// Returns: the largest node count over the bins (the padded size of every
// per-node array of the work functions)
// ---------------------------------------------------------------------------
static int create_cosmo_nodes_cluster_all(
    cosmo_nodes* cn_all,        // output [cluster.zdist_nbin]
    const int with_foreground   // 1: add the foreground (magnification) panel
  )
{
  const gsl_integration_glfixed_table* w = limber_gl_table_cluster();

  int npts_max = 0;
  for (int ni = 0; ni < cluster.zdist_nbin; ni++) {
    cn_all[ni] = create_cosmo_nodes_cluster(ni, with_foreground, w);
    if (cn_all[ni].npts > npts_max) {
      npts_max = cn_all[ni].npts;
    }
  }
  return npts_max;
}


static void free_cosmo_nodes_cluster_all(cosmo_nodes* cn_all)
{
  for (int ni = 0; ni < cluster.zdist_nbin; ni++) {
    free(cn_all[ni].data);
  }
}



// ============================================================================
// [SECTION] CACHE KEYS
// ============================================================================
//
// A table stores the keys it was built with and refills when any differs
// (the core's two-tier pattern: Ntable.random reallocates, the physics keys
// refill). Each statistic keys on exactly what its integrand reads.

// ---------------------------------------------------------------------------
// Keys every cluster statistic shares: cluster model, selection kernels,
// MOR, pair lists, and the switches themselves (the include_HOD_GX pattern
// of cosmo2D.c: a switch flipped without a key bump still refills).
//
// Selection bias: the Y1 model sits inside the bias mass integral
// (halo_cluster.c), so its parameters change the spectra; the Y6 model acts
// on the data vector only (the interface), and keying on it would refill
// every cluster table at every MCMC step for nothing.
//
// Returns: the new number of keys (the position count stays fixed)
// ---------------------------------------------------------------------------
static int append_cluster_keys(
    uint64_t* keys,  // key array (CLUSTER_NKEYS_MAX)
    int nkeys        // keys already written
  )
{
  keys[nkeys++] = cluster.random_model;
  keys[nkeys++] = cluster.random_zdist;
  keys[nkeys++] = cluster.random_mor;
  keys[nkeys++] = cluster.random_pairs;
  keys[nkeys++] = (uint64_t) cluster.selection_model;
  if (CLUSTER_SELECTION_Y1 == cluster.selection_model) {
    keys[nkeys++] = cluster.random_selection;
  }
  else {
    keys[nkeys++] = 0;
  }
  keys[nkeys++] = (uint64_t) cluster.kernel_mode;
  keys[nkeys++] = (uint64_t) cluster.include_ia;
  keys[nkeys++] = double_bits_key(cluster.magnification);
  return nkeys;
}


// cluster lensing: cosmology, source n(z) and its photo-z shifts, NLA
// amplitudes, plus the cluster keys
static int cluster_keys_cs(uint64_t* keys)
{
  int nkeys = 0;
  keys[nkeys++] = Ntable.random;
  keys[nkeys++] = cosmology.random;
  keys[nkeys++] = nuisance.random_photoz_shear;
  keys[nkeys++] = nuisance.random_ia;
  keys[nkeys++] = redshift.random_shear;
  return append_cluster_keys(keys, nkeys);
}


// cluster clustering: cosmology plus the cluster keys
static int cluster_keys_cc(uint64_t* keys)
{
  int nkeys = 0;
  keys[nkeys++] = Ntable.random;
  keys[nkeys++] = cosmology.random;
  return append_cluster_keys(keys, nkeys);
}


// cluster-galaxy clustering: cosmology, lens n(z) and its photo-z
// parameters, galaxy bias and magnification, plus the cluster keys
static int cluster_keys_cg(uint64_t* keys)
{
  int nkeys = 0;
  keys[nkeys++] = Ntable.random;
  keys[nkeys++] = cosmology.random;
  keys[nkeys++] = nuisance.random_photoz_clustering;
  keys[nkeys++] = nuisance.random_galaxy_bias;
  keys[nkeys++] = redshift.random_clustering;
  return append_cluster_keys(keys, nkeys);
}



// ============================================================================
// [SECTION] CACHED LOG-ELL TABLES (shared by the three statistics)
// ============================================================================
//
// One table per statistic: rows = the statistic's block (e.g. cs: (pair,
// richness)), columns = log-spaced multipoles over [LMIN_tab, LMAX + 1]
// (the range of cosmo2D.c). Two uniform grids in ln l share those ends:
//
//   exact grid  the Limber quadrature runs here:
//                 C_cs        Ntable.N_ell_internal nodes (pattern P1b, as
//                             cosmo2D.c's C_gs: smooth in ln l with one
//                             broad lensing kernel; the 1-halo/2-halo
//                             transition spans about an e-fold in l)
//                 C_cc, C_cg  Ntable.N_ell nodes (two narrow density kernels
//                             leave BAO wiggles in l: the node count of
//                             cosmo2D.c's exact C_gg)
//   dense grid  CLUSTER_ELL_REFINEMENT (N_ell - 1) + 1 nodes, filled by the
//               house natural cubic spline in ln l (spline_coeffs_uniform +
//               Horner) through the exact nodes. The N_ell exact nodes of
//               cc and cg sit exactly on dense nodes (the x (N - 1) + 1
//               alignment rule).
//
// The real-space sums read the dense grid at every integer l through
// limber_fill_interp (linear in ln l); the scalar readers through
// interpol1d. Why the refinement: a linear read leaves an error
// (d ln l)^2/8 x (d^2 C/d ln l^2) inside every cell, and the Y transform
// of cluster lensing (eq 15) takes differences of gamma_t across
// neighbouring angular bins, which amplifies any error that is not smooth.
// Refining cuts that error by CLUSTER_ELL_REFINEMENT^2 at one Horner
// evaluation per dense node and refill; the integer-l reads cost the same
// at any table size.

// dense nodes per exact N_ell interval (see above)
static const int CLUSTER_ELL_REFINEMENT = 8;

// A statistic's exact Limber batch: C_l of every row of its block at nell
// multipoles, rows[row][i] (the *_batch_rows functions below).
typedef void (*limber_batch_rows)(const double* ells, const int nell,
  double** rows);

typedef struct
{
  // --- dense grid: what the readers see ---
  double** tab;         // [nrows][nell]: C_l at l_i = exp(lim[0] + i lim[2])
  int nrows;            // rows of the statistic's block
  int nell;             // dense nodes
  double lim[3];        // ln l_min, ln l_max, uniform spacing in ln l

  // --- exact grid: where the Limber quadrature runs ---
  int smooth_in_ln_ell; // the statistic's choice when allocated (1: cs)
  int nexact;           // exact nodes
  double dlnx;          // exact spacing in ln l
  double* lxe;          // the nexact multipoles of the exact grid
  double** tabe;        // [nrows][nexact]: exact C_l
  double** cspl;        // [nrows][nexact]: spline c coefficients
  int* qidx;            // dense node -> exact interval (left node)
  double* qdel;         // dense node -> ln l offset inside that interval

  // --- cache ---
  uint64_t cache_ntable;               // Ntable.random of the allocation
  uint64_t cache[CLUSTER_NKEYS_MAX];   // keys of the values
} limber_table_cluster;

// how the real-space functions reach a statistic's table
typedef const limber_table_cluster* (*limber_table_getter)(void);


// ---------------------------------------------------------------------------
// Release every array of a table (the struct itself is static storage).
// ---------------------------------------------------------------------------
static void limber_table_cluster_free(limber_table_cluster* T)
{
  if (T->tab != NULL) {
    free(T->tab);
    T->tab = NULL;
  }
  if (T->lxe != NULL) {
    free(T->lxe);
    T->lxe = NULL;
  }
  if (T->tabe != NULL) {
    free(T->tabe);
    T->tabe = NULL;
  }
  if (T->cspl != NULL) {
    free(T->cspl);
    T->cspl = NULL;
  }
  if (T->qidx != NULL) {
    free(T->qidx);
    T->qidx = NULL;
  }
  if (T->qdel != NULL) {
    free(T->qdel);
    T->qdel = NULL;
  }
}


// ---------------------------------------------------------------------------
// Fill the dense grid from the exact rows with the house natural cubic
// spline in ln l.
//
// On interval [x_j, x_j + h] the spline is
//
//   S(x_j + dx) = y_j + b dx + c_j dx^2 + d dx^3
//
// with c from spline_coeffs_uniform (S''/2, natural ends c = 0) and
//
//   d = (c_{j+1} - c_j)/(3 h)                        (S'' linear in the cell)
//   b = (y_{j+1} - y_j)/h - h (c_{j+1} + 2 c_j)/3    (S hits y_{j+1})
//
// evaluated in Horner form at each dense node's precomputed (j, dx).
// ---------------------------------------------------------------------------
static void limber_table_cluster_upsample(limber_table_cluster* T)
{
  const double h     = T->dlnx;
  const double inv_h = 1.0/T->dlnx;

  #pragma omp parallel for schedule(static)
  for (int row = 0; row < T->nrows; row++) {
    spline_coeffs_uniform(T->tabe[row], T->nexact, h, T->cspl[row]);
  }

  #pragma omp parallel for collapse(2) schedule(static)
  for (int row = 0; row < T->nrows; row++) {
    for (int i = 0; i < T->nell; i++) {
      const double* restrict y = T->tabe[row];
      const double* restrict c = T->cspl[row];

      const int j     = T->qidx[i];
      const double dx = T->qdel[i];

      const double b = (y[j+1] - y[j])*inv_h - h*(c[j+1] + 2.0*c[j])/3.0;
      const double d = (c[j+1] - c[j])/(3.0*h);

      T->tab[row][i] = y[j] + dx*(b + dx*(c[j] + dx*d));
    }
  }
}


// ---------------------------------------------------------------------------
// Bring a statistic's table up to date.
//
//   1. GEOMETRY (Ntable.random, block size): both grids, the dense-node map
//      onto the exact grid, every array. Every allocation lives here; a
//      refill only fills.
//   2. VALUES (the statistic's keys): exact Limber batch on the exact grid,
//      spline onto the dense grid.
//
// Thread safety: call outside parallel regions (the batch runs its own).
// ---------------------------------------------------------------------------
static void limber_table_cluster_update(
    limber_table_cluster* T,        // the statistic's static table
    const int nrows,                // rows of the statistic's block
    const int smooth_in_ln_ell,     // 1: exact grid N_ell_internal (P1b)
    const uint64_t* keys,           // the statistic's current keys
    const int nkeys,                // number of keys
    limber_batch_rows batch_rows    // the statistic's exact Limber batch
  )
{
  int refill = 0;

  // --- 1. GEOMETRY ---
  if (NULL == T->tab ||
      fdiff2(T->cache_ntable, Ntable.random) ||
      nrows != T->nrows ||
      smooth_in_ln_ell != T->smooth_in_ln_ell)
  {
    limber_table_cluster_free(T);

    // the spline needs at least four nodes
    const int nell_house = Ntable.N_ell;
    if (nell_house < 4) {
      log_fatal("Ntable.N_ell = %d < 4", nell_house);
      exit(1);
    }

    int nexact = nell_house;
    if (1 == smooth_in_ln_ell &&
        Ntable.N_ell_internal > 3 &&
        Ntable.N_ell_internal < nell_house)
    {
      nexact = Ntable.N_ell_internal;
    }
    const int nell = CLUSTER_ELL_REFINEMENT*(nell_house - 1) + 1;

    T->nrows            = nrows;
    T->nell             = nell;
    T->nexact           = nexact;
    T->smooth_in_ln_ell = smooth_in_ln_ell;

    // same l range as cosmo2D.c: from the first multipole the real-space
    // sums read from the table to one past the last one they need
    T->lim[0] = log(fmax(limits.LMIN_tab, 1.0));
    T->lim[1] = log(Ntable.LMAX + 1.0);
    T->lim[2] = (T->lim[1] - T->lim[0])/((double) nell - 1.0);

    T->dlnx = (T->lim[1] - T->lim[0])/((double) nexact - 1.0);

    T->lxe = (double*) malloc1d(nexact);
    for (int i = 0; i < nexact; i++) {
      T->lxe[i] = exp(T->lim[0] + i*T->dlnx);
    }

    // Where does dense node i sit on the exact grid? Both grids span the
    // same [lim[0], lim[1]] in ln l, so the map is arithmetic:
    //   r = i lim[2]/dlnx (exact spacings from the left end),
    //   j = (int) r (left node), qdel = (r - j) dlnx (offset in ln l).
    // The clamp: at the shared top end r can round one ulp above
    // nexact - 1; the last legal interval starts at nexact - 2.
    T->qidx = (int*) malloc1d_int(nell);
    T->qdel = (double*) malloc1d(nell);
    for (int i = 0; i < nell; i++) {
      const double r = (double) i*T->lim[2]/T->dlnx;
      int j = (int) r;
      if (j > nexact - 2) {
        j = nexact - 2;
      }
      T->qidx[i] = j;
      T->qdel[i] = (r - j)*T->dlnx;
    }

    T->tab  = (double**) malloc2d(nrows, nell);
    T->tabe = (double**) malloc2d(nrows, nexact);
    T->cspl = (double**) malloc2d(nrows, nexact);
    zero2d(T->tab, nrows, nell);
    zero2d(T->tabe, nrows, nexact);
    zero2d(T->cspl, nrows, nexact);

    T->cache_ntable = Ntable.random;
    refill = 1;
  }

  // --- 2. VALUES ---
  if (1 == refill || keys_changed(T->cache, keys, nkeys)) {
    batch_rows(T->lxe, T->nexact, T->tabe);
    limber_table_cluster_upsample(T);
    keys_stamp(T->cache, keys, nkeys);
  }
}


// ---------------------------------------------------------------------------
// Read row `row` of a table at multipole l (linear in ln l, the core's
// interpol1d; outside the grid it warns and extrapolates, as cosmo2D.c).
// ---------------------------------------------------------------------------
static double limber_table_cluster_read(
    const limber_table_cluster* T,
    const int row,
    const double l
  )
{
  const double lnl = log(l);

  if (lnl < T->lim[0]) {
    log_warn("l = %e < lmin = %e. Extrapolation adopted", l, exp(T->lim[0]));
  }
  if (lnl > T->lim[1]) {
    log_warn("l = %e > lmax = %e. Extrapolation adopted", l, exp(T->lim[1]));
  }
  return interpol1d(T->tab[row], T->nell, T->lim[0], T->lim[1], T->lim[2],
    lnl);
}



// ============================================================================
// [SECTION] CLUSTER LEG AT THE NODES (shared by C_cs, C_cc, C_cg)
// ============================================================================

// ---------------------------------------------------------------------------
// Node quantities of the cluster field, per (cluster bin, richness bin,
// node):
//
//   limber_weight          w_p (dchi/da) / f_K^2    (the Limber measure)
//   cluster_kernel         W_c = n_c(z) H/H0        (W_cluster; may be NULL)
//   cluster_density        b_nl(a) W_c              (eq 21 bias x kernel)
//   cluster_magnification  C_c W_mag,c              (eqs 27-28)
//
// The density terms are evaluated on the support panel only (they vanish
// in the foreground), the magnification on both panels. The caller zeroes
// the arrays, so padding nodes (p >= cn->npts) and switched-off terms stay 0.
// ---------------------------------------------------------------------------
static void cluster_leg_at_nodes(
    const cosmo_nodes* cn_all,       // nodes per cluster bin
    const int nbin_cluster,          // cluster bins 0 .. nbin_cluster-1
    const int npts_max,              // padded node count
    double** limber_weight,          // out [nbin_cluster][npts_max]
    double*** cluster_kernel,        // out [nbin_cluster][richness][npts_max]
    double*** cluster_density,       // out [nbin_cluster][richness][npts_max]
    double*** cluster_magnification  // out [nbin_cluster][richness][npts_max]
  )
{
  const int nbin_richness = cluster.richness_nbin;
  const double C_c        = cluster.magnification;  // eq 28: -2

  #pragma omp parallel for collapse(2) schedule(static)
  for (int ni = 0; ni < nbin_cluster; ni++) {
    for (int p = 0; p < npts_max; p++) {
      const cosmo_nodes* cn = &cn_all[ni];
      if (p >= cn->npts) {
        continue; // padding node: every array stays 0 there
      }

      const double a       = cn->data[CN_A][p];
      const double wt      = cn->data[CN_WT][p];
      const double fK      = cn->data[CN_FK][p];
      const double hoverh0 = cn->data[CN_HOVERH0][p];
      const double dchida  = cn->data[CN_DCHIDA][p];

      limber_weight[ni][p] = wt*dchida/(fK*fK);

      const int on_support = (p < cn->nsupport);

      for (int nl = 0; nl < nbin_richness; nl++) {
        if (1 == on_support) {
          const double W_c = W_cluster(a, ni, nl, hoverh0);
          const double b_c = bcl_richness(a, nl);

          if (cluster_kernel != NULL) {
            cluster_kernel[ni][nl][p] = W_c;
          }
          cluster_density[ni][nl][p] = b_c*W_c;
        }
        if (0.0 != C_c) {
          cluster_magnification[ni][nl][p] = C_c*W_mag_cluster(a, fK, ni, nl);
        }
      }
    }
  }
}


// ---------------------------------------------------------------------------
// Nonlinear matter power at every (cluster bin, multipole, node), at the
// Limber wavenumber k = (l + 1/2)/f_K: the P_delta accessor the galaxy
// spectra of cosmo2D.c read.
// ---------------------------------------------------------------------------
static void nonlinear_power_at_nodes(
    const cosmo_nodes* cn_all,   // nodes per cluster bin
    const int nbin_cluster,      // cluster bins 0 .. nbin_cluster-1
    const int npts_max,          // padded node count
    const double* lx,            // multipoles (length nell)
    const int nell,              // number of multipoles
    double*** p_nonlinear        // out [nbin_cluster][nell][npts_max]
  )
{
  #pragma omp parallel for collapse(3) schedule(static)
  for (int ni = 0; ni < nbin_cluster; ni++) {
    for (int i = 0; i < nell; i++) {
      for (int p = 0; p < npts_max; p++) {
        const cosmo_nodes* cn = &cn_all[ni];
        if (p >= cn->npts) {
          continue; // padding node
        }
        const double a  = cn->data[CN_A][p];
        const double fK = cn->data[CN_FK][p];
        const double k  = (lx[i] + 0.5)/fK;

        p_nonlinear[ni][i][p] = Pdelta(k, a);
      }
    }
  }
}


// ---------------------------------------------------------------------------
// Warm the lazily built tables every cluster Limber integrand reads, single
// threaded, before any parallel region: every cluster table
// (cluster_warmup, the contract of halo_cluster.h), then one call of each
// reader at a support node.
// ---------------------------------------------------------------------------
static void warmup_cluster_leg(
    const cosmo_nodes* cn_all,  // nodes per cluster bin
    const double* lx            // multipoles (at least one)
  )
{
  const cosmo_nodes* cn = &cn_all[0];

  const double a       = cn->data[CN_A][0];
  const double fK      = cn->data[CN_FK][0];
  const double hoverh0 = cn->data[CN_HOVERH0][0];
  const double k       = (lx[0] + 0.5)/fK;

  (void) W_cluster(a, 0, 0, hoverh0);
  (void) W_mag_cluster(a, fK, 0, 0);
  (void) bcl_richness(a, 0);
  (void) Pdelta(k, a);
}


// basic checks shared by every cluster statistic
static void check_cluster_setup(void)
{
  if (cluster.zdist_nbin <= 0) {
    log_fatal("cluster.zdist_nbin = %d: cluster redshift bins not set",
      cluster.zdist_nbin);
    exit(1);
  }
  if (cluster.richness_nbin <= 0) {
    log_fatal("cluster.richness_nbin = %d: richness bins not set",
      cluster.richness_nbin);
    exit(1);
  }
}



// ============================================================================
// [SECTION] CLUSTER LENSING: C_cs
// ============================================================================
//
// PHYSICAL DERIVATION & LOGIC FLOW (2503.13631 eqs 4, 10, 20-22, 27-28)
//   1. Limber (eq 10) at k = (l + 1/2)/f_K(chi):
//        C_cs(l) = ep_shear(l) int da (dchi/da)/f_K^2 [two-halo + one-halo]
//   2. two-halo (eqs 20-21, 27-28): the cluster field is b_nl delta_m plus
//      its magnification C_c kappa_c; the source field is kappa minus the
//      NLA alignment term (eq 4; the A_1 part only, IA_A1_Z1, as the gs
//      engine of cosmo2D.c builds it):
//        [W_kappa - W_source A_1] [b_nl W_c + C_c ep_mag W_mag,c] P_NL
//   3. one-halo (eq 22): the clusters' own halo profile; no bias, no
//      magnification, no alignment:
//        W_kappa W_c P1h_nl
//   4. The magnification enters with a PLUS sign and the coefficient C_c,
//      exactly as cosmo2D.c's galaxy term W_mag ep b_mag: C_c = -2 gives
//      b W_c - 2 W_mag,c, the geometric dilution of a flux-free sample.

static void check_cluster_setup_cs(void)
{
  check_cluster_setup();
  if (cluster.cs_npowerspectra <= 0) {
    log_fatal("cluster lensing requested but cs_npowerspectra = %d",
      cluster.cs_npowerspectra);
    exit(1);
  }
  if (redshift.shear_nbin <= 0) {
    log_fatal("cluster lensing requested but shear_nbin = %d",
      redshift.shear_nbin);
    exit(1);
  }
}


// ---------------------------------------------------------------------------
// Core of every cluster-lensing C_l (the C_gs_tomo_limber_work design):
// node quantities once per (bin, node), spectra once per (bin, l, node),
// then one vectorized sum over the nodes per (row, l).
//
// Memory layout (padded to npts_max; unused entries are 0):
//   limber_weight          [cluster bin][node]
//   cluster_kernel         [cluster bin][richness][node]   W_c
//   cluster_density        [cluster bin][richness][node]   b W_c
//   cluster_magnification  [cluster bin][richness][node]   C_c W_mag,c
//   source_kappa           [cluster bin][source bin][node] W_kappa
//   source_alignment       [cluster bin][source bin][node] W_source A_1
//   p_nonlinear            [cluster bin][l][node]          P_NL(k, a)
//   p_one_halo             [cluster bin][richness][l][node] P1h_nl(k, a)
//
// Output: table[n*richness_nbin + nl][i] for cs pair n = (ZC_cs(n),
// ZS_cs(n)), richness bin nl, multipole lx[i].
// ---------------------------------------------------------------------------
static void C_cs_tomo_limber_work(
    const cosmo_nodes* cn_all,   // nodes per cluster bin [zdist_nbin]
    const int npts_max,          // largest node count over the bins
    const double* lx,            // multipoles (length nell)
    const double* ep_mag,        // l(l+1)/(l+1/2)^2 per multipole
    const double* ep_shear,      // sqrt((l-1)l(l+1)(l+2))/(l+1/2)^2
    const int nell,              // number of multipoles
    double** table               // out [cs_npowerspectra*richness_nbin][nell]
  )
{
  const int nbin_cluster  = cluster.zdist_nbin;
  const int nbin_richness = cluster.richness_nbin;
  const int nbin_source   = redshift.shear_nbin;
  const int npairs        = cluster.cs_npowerspectra;
  const int nrows         = npairs*nbin_richness;
  const int include_ia    = cluster.include_ia;

  // --- 1. WARM-UP (single-threaded, before any parallel region) ---
  warmup_cluster_leg(cn_all, lx);
  {
    const cosmo_nodes* cn = &cn_all[0];

    const double a         = cn->data[CN_A][0];
    const double fK        = cn->data[CN_FK][0];
    const double hoverh0   = cn->data[CN_HOVERH0][0];
    const double growfac_a = cn->data[CN_GROWFAC][0];
    const double k         = (lx[0] + 0.5)/fK;

    (void) pcm_1h_richness(k, a, 0);
    (void) W_kappa(a, fK, 0);
    (void) W_source(a, 0, hoverh0);
    (void) IA_A1_Z1(a, growfac_a, 0);
  }

  // --- 2. PAIR LIST, read once outside the parallel regions ---
  int cluster_bin_of_pair[npairs];
  int source_bin_of_pair[npairs];
  for (int n = 0; n < npairs; n++) {
    cluster_bin_of_pair[n] = ZC_cs(n);
    source_bin_of_pair[n]  = ZS_cs(n);

    if (cluster_bin_of_pair[n] < 0 ||
        cluster_bin_of_pair[n] > nbin_cluster - 1 ||
        source_bin_of_pair[n] < 0 ||
        source_bin_of_pair[n] > nbin_source - 1)
    {
      log_fatal("invalid cs pair %d: (cluster, source) = (%d, %d)",
        n, cluster_bin_of_pair[n], source_bin_of_pair[n]);
      exit(1);
    }
  }

  // --- 3. ALLOCATION ---
  double** limber_weight = (double**) malloc2d(nbin_cluster, npts_max);
  double*** cluster_kernel =
    (double***) malloc3d(nbin_cluster, nbin_richness, npts_max);
  double*** cluster_density =
    (double***) malloc3d(nbin_cluster, nbin_richness, npts_max);
  double*** cluster_magnification =
    (double***) malloc3d(nbin_cluster, nbin_richness, npts_max);
  double*** source_kappa =
    (double***) malloc3d(nbin_cluster, nbin_source, npts_max);
  double*** source_alignment =
    (double***) malloc3d(nbin_cluster, nbin_source, npts_max);
  double*** p_nonlinear =
    (double***) malloc3d(nbin_cluster, nell, npts_max);
  double**** p_one_halo =
    (double****) malloc4d(nbin_cluster, nbin_richness, nell, npts_max);

  zero2d(limber_weight, nbin_cluster, npts_max);
  zero3d(cluster_kernel, nbin_cluster, nbin_richness, npts_max);
  zero3d(cluster_density, nbin_cluster, nbin_richness, npts_max);
  zero3d(cluster_magnification, nbin_cluster, nbin_richness, npts_max);
  zero3d(source_kappa, nbin_cluster, nbin_source, npts_max);
  zero3d(source_alignment, nbin_cluster, nbin_source, npts_max);
  zero3d(p_nonlinear, nbin_cluster, nell, npts_max);
  zero4d(p_one_halo, nbin_cluster, nbin_richness, nell, npts_max);

  // --- 4. NODE QUANTITIES: cluster leg, source leg ---
  cluster_leg_at_nodes(cn_all, nbin_cluster, npts_max, limber_weight,
    cluster_kernel, cluster_density, cluster_magnification);

  #pragma omp parallel for collapse(2) schedule(static)
  for (int ni = 0; ni < nbin_cluster; ni++) {
    for (int p = 0; p < npts_max; p++) {
      const cosmo_nodes* cn = &cn_all[ni];
      if (p >= cn->npts) {
        continue; // padding node
      }

      const double a         = cn->data[CN_A][p];
      const double fK        = cn->data[CN_FK][p];
      const double hoverh0   = cn->data[CN_HOVERH0][p];
      const double growfac_a = cn->data[CN_GROWFAC][p];

      for (int ns = 0; ns < nbin_source; ns++) {
        source_kappa[ni][ns][p] = W_kappa(a, fK, ns);

        // NLA contamination of the source shapes (eq 4): W_source A_1,
        // the C1 term of the cosmo2D.c gs engine
        if (1 == include_ia) {
          source_alignment[ni][ns][p] =
            W_source(a, ns, hoverh0)*IA_A1_Z1(a, growfac_a, ns);
        }
      }
    }
  }

  // --- 5. SPECTRA AT (NODE, l): P_NL everywhere, P1h on the support ---
  nonlinear_power_at_nodes(cn_all, nbin_cluster, npts_max, lx, nell,
    p_nonlinear);

  #pragma omp parallel for collapse(3) schedule(static)
  for (int ni = 0; ni < nbin_cluster; ni++) {
    for (int i = 0; i < nell; i++) {
      for (int p = 0; p < npts_max; p++) {
        const cosmo_nodes* cn = &cn_all[ni];
        if (p >= cn->nsupport) {
          continue; // the one-halo term lives where W_c does
        }
        const double a  = cn->data[CN_A][p];
        const double fK = cn->data[CN_FK][p];
        const double k  = (lx[i] + 0.5)/fK;

        for (int nl = 0; nl < nbin_richness; nl++) {
          p_one_halo[ni][nl][i][p] = pcm_1h_richness(k, a, nl);
        }
      }
    }
  }

  // --- 6. LIMBER SUM: one row (pair, richness) and one l per task ---
  #pragma omp parallel for collapse(2) schedule(static)
  for (int row = 0; row < nrows; row++) {
    for (int i = 0; i < nell; i++) {
      const int n  = row/nbin_richness;
      const int nl = row - n*nbin_richness;
      const int ni = cluster_bin_of_pair[n];
      const int ns = source_bin_of_pair[n];

      const int npts  = cn_all[ni].npts;
      const double ep = ep_mag[i];

      // Local restrict pointers: without them the compiler cannot prove
      // that rows reached through pointer-to-pointer indirection do not
      // alias inside the collapse(2) region, and it reloads every operand
      // (the cosmo2D.c idiom; see SKILL.md)
      const double* restrict weight        = limber_weight[ni];
      const double* restrict kernel        = cluster_kernel[ni][nl];
      const double* restrict density       = cluster_density[ni][nl];
      const double* restrict magnification = cluster_magnification[ni][nl];
      const double* restrict kappa         = source_kappa[ni][ns];
      const double* restrict alignment     = source_alignment[ni][ns];
      const double* restrict pnl           = p_nonlinear[ni][i];
      const double* restrict p1h           = p_one_halo[ni][nl][i];

      double sum = 0.0;
      #pragma omp simd reduction(+:sum)
      for (int p = 0; p < npts; p++) {
        // two-halo: [W_kappa - W_source A_1][b W_c + C_c ep W_mag,c] P_NL
        const double cluster_leg = density[p] + ep*magnification[p];
        const double source_leg  = kappa[p] - alignment[p];
        const double two_halo    = source_leg*cluster_leg*pnl[p];

        // one-halo: W_kappa W_c P1h
        const double one_halo = kappa[p]*kernel[p]*p1h[p];

        sum += (two_halo + one_halo)*weight[p];
      }
      table[row][i] = ep_shear[i]*sum;
    }
  }

  // --- 7. RELEASE ---
  free(limber_weight);
  free(cluster_kernel);
  free(cluster_density);
  free(cluster_magnification);
  free(source_kappa);
  free(source_alignment);
  free(p_nonlinear);
  free(p_one_halo);
}


// ---------------------------------------------------------------------------
// Exact cluster-lensing C_l of every row (pair n, richness nl) at nell
// multipoles: nodes, prefactors, work function. rows[n*richness_nbin +
// nl][i].
// ---------------------------------------------------------------------------
static void C_cs_tomo_limber_batch_rows(
    const double* ells,  // multipoles (length nell; need not be integers)
    const int nell,      // number of multipoles
    double** rows        // out [cs_npowerspectra*richness_nbin][nell]
  )
{
  if (nell <= 0) {
    log_fatal("nell = %d <= 0", nell);
    exit(1);
  }
  check_cluster_setup_cs();

  // every lazily built cluster table, before the nodes read the support
  cluster_warmup();

  const int with_foreground = (0.0 != cluster.magnification);

  cosmo_nodes cn_all[cluster.zdist_nbin];
  const int npts_max = create_cosmo_nodes_cluster_all(cn_all, with_foreground);

  double* ep_mag   = (double*) malloc1d(nell);
  double* ep_shear = (double*) malloc1d(nell);
  for (int i = 0; i < nell; i++) {
    ep_mag[i]   = ell_prefactor_magnification(ells[i]);
    ep_shear[i] = ell_prefactor_shear(ells[i]);
  }

  C_cs_tomo_limber_work(cn_all, npts_max, ells, ep_mag, ep_shear, nell, rows);

  free(ep_mag);
  free(ep_shear);
  free_cosmo_nodes_cluster_all(cn_all);
}


// ---------------------------------------------------------------------------
// Public batch: out[n][nl][i] = C_cs(ells[i]) of cs pair n = (ZC_cs(n),
// ZS_cs(n)) and richness bin nl (out: malloc3d by the caller).
// ---------------------------------------------------------------------------
void C_cs_tomo_limber_nointerp_ells(
    const double* ells,  // multipoles (length nell)
    const int nell,      // number of multipoles
    double*** out        // out [cs_npowerspectra][richness_nbin][nell]
  )
{
  check_cluster_setup_cs();

  const int nbin_richness = cluster.richness_nbin;
  const int npairs        = cluster.cs_npowerspectra;
  const int nrows         = npairs*nbin_richness;

  double** rows = (double**) malloc2d(nrows, nell);

  C_cs_tomo_limber_batch_rows(ells, nell, rows);

  for (int n = 0; n < npairs; n++) {
    for (int nl = 0; nl < nbin_richness; nl++) {
      for (int i = 0; i < nell; i++) {
        out[n][nl][i] = rows[n*nbin_richness + nl][i];
      }
    }
  }
  free(rows);
}


// ---------------------------------------------------------------------------
// The cached cluster-lensing table (exact at N_ell_internal nodes, spline
// onto the dense grid).
// ---------------------------------------------------------------------------
static const limber_table_cluster* C_cs_tomo_limber_table(void)
{
  static limber_table_cluster table;

  check_cluster_setup_cs();

  uint64_t keys[CLUSTER_NKEYS_MAX];
  const int nkeys = cluster_keys_cs(keys);

  const int nrows            = cluster.cs_npowerspectra*cluster.richness_nbin;
  const int smooth_in_ln_ell = 1; // exact grid: N_ell_internal nodes

  limber_table_cluster_update(&table, nrows, smooth_in_ln_ell, keys, nkeys,
    C_cs_tomo_limber_batch_rows);

  return &table;
}


// ---------------------------------------------------------------------------
// C_cs at multipole l for richness bin nl, cluster bin ni, source bin ns,
// read from the cached table. 0 for a (ni, ns) outside the pair list.
// ---------------------------------------------------------------------------
double C_cs_tomo_limber(
    const double l,  // multipole
    const int nl,    // richness bin
    const int ni,    // cluster redshift bin
    const int ns     // source redshift bin
  )
{
  check_cluster_setup_cs();

  if (nl < 0 || nl > cluster.richness_nbin - 1 ||
      ni < 0 || ni > cluster.zdist_nbin - 1 ||
      ns < 0 || ns > redshift.shear_nbin - 1)
  {
    log_fatal("error in selecting bin number (nl, ni, ns) = (%d, %d, %d)",
      nl, ni, ns);
    exit(1);
  }

  const limber_table_cluster* T = C_cs_tomo_limber_table();

  const int n = N_cs(ni, ns);
  if (n < 0) {
    return 0.0;
  }
  const int row = n*cluster.richness_nbin + nl;
  return limber_table_cluster_read(T, row, l);
}



// ============================================================================
// [SECTION] CLUSTER CLUSTERING: C_cc
// ============================================================================
//
// PHYSICAL DERIVATION & LOGIC FLOW (2503.13631 eqs 10, 20-21, 27-28)
//   1. Each cluster field of cluster bin ni and richness bin nl is its
//      biased density plus its magnification:
//        W_nl = b_nl W_c + C_c ep_mag W_mag,c
//   2. Limber (eq 10), auto redshift bin, every richness pair nl1 <= nl2:
//        C_cc(l) = int da (dchi/da)/f_K^2 W_nl1 W_nl2 P_NL(k, a)
//   3. The block covers cluster bins 0 .. cluster.cc_npowerspectra - 1
//      (the w_cc data vector: [z][nl1 <= nl2][theta]).

// number of richness pairs nl1 <= nl2 of a cluster bin
static int cc_richness_npairs(void)
{
  return cluster.richness_nbin*(cluster.richness_nbin + 1)/2;
}


static void check_cluster_setup_cc(void)
{
  check_cluster_setup();
  if (cluster.cc_npowerspectra <= 0 ||
      cluster.cc_npowerspectra > cluster.zdist_nbin)
  {
    log_fatal("cluster clustering requested but cc_npowerspectra = %d "
      "(zdist_nbin = %d)", cluster.cc_npowerspectra, cluster.zdist_nbin);
    exit(1);
  }
}


// ---------------------------------------------------------------------------
// Core of every cluster-clustering C_l (the C_gg_tomo_limber_work design).
//
// Memory layout (padded to npts_max; unused entries are 0):
//   limber_weight          [cluster bin][node]
//   cluster_density        [cluster bin][richness][node]   b W_c
//   cluster_magnification  [cluster bin][richness][node]   C_c W_mag,c
//   p_nonlinear            [cluster bin][l][node]          P_NL(k, a)
//
// Output: table[ni*npairs_richness + n][i] for cluster bin ni, richness
// pair n = (NL1_cc(n), NL2_cc(n)), multipole lx[i].
// ---------------------------------------------------------------------------
static void C_cc_tomo_limber_work(
    const cosmo_nodes* cn_all,   // nodes per cluster bin [zdist_nbin]
    const int npts_max,          // largest node count over the bins
    const double* lx,            // multipoles (length nell)
    const double* ep_mag,        // l(l+1)/(l+1/2)^2 per multipole
    const int nell,              // number of multipoles
    double** table               // out [cc_npowerspectra*npairs][nell]
  )
{
  const int nbin_cluster     = cluster.cc_npowerspectra;
  const int nbin_richness    = cluster.richness_nbin;
  const int npairs_richness  = cc_richness_npairs();
  const int nrows            = nbin_cluster*npairs_richness;

  // --- 1. WARM-UP (single-threaded) ---
  warmup_cluster_leg(cn_all, lx);

  // --- 2. RICHNESS PAIR LIST, read once outside the parallel regions ---
  int richness1_of_pair[npairs_richness];
  int richness2_of_pair[npairs_richness];
  for (int n = 0; n < npairs_richness; n++) {
    richness1_of_pair[n] = NL1_cc(n);
    richness2_of_pair[n] = NL2_cc(n);

    if (richness1_of_pair[n] < 0 ||
        richness1_of_pair[n] > nbin_richness - 1 ||
        richness2_of_pair[n] < 0 ||
        richness2_of_pair[n] > nbin_richness - 1)
    {
      log_fatal("invalid cc richness pair %d: (%d, %d)",
        n, richness1_of_pair[n], richness2_of_pair[n]);
      exit(1);
    }
  }

  // --- 3. ALLOCATION ---
  double** limber_weight = (double**) malloc2d(nbin_cluster, npts_max);
  double*** cluster_density =
    (double***) malloc3d(nbin_cluster, nbin_richness, npts_max);
  double*** cluster_magnification =
    (double***) malloc3d(nbin_cluster, nbin_richness, npts_max);
  double*** p_nonlinear =
    (double***) malloc3d(nbin_cluster, nell, npts_max);

  zero2d(limber_weight, nbin_cluster, npts_max);
  zero3d(cluster_density, nbin_cluster, nbin_richness, npts_max);
  zero3d(cluster_magnification, nbin_cluster, nbin_richness, npts_max);
  zero3d(p_nonlinear, nbin_cluster, nell, npts_max);

  // --- 4. NODE QUANTITIES AND SPECTRA ---
  cluster_leg_at_nodes(cn_all, nbin_cluster, npts_max, limber_weight,
    NULL, cluster_density, cluster_magnification);

  nonlinear_power_at_nodes(cn_all, nbin_cluster, npts_max, lx, nell,
    p_nonlinear);

  // --- 5. LIMBER SUM: one row (cluster bin, richness pair) and one l ---
  #pragma omp parallel for collapse(2) schedule(static)
  for (int row = 0; row < nrows; row++) {
    for (int i = 0; i < nell; i++) {
      const int ni  = row/npairs_richness;
      const int n   = row - ni*npairs_richness;
      const int nl1 = richness1_of_pair[n];
      const int nl2 = richness2_of_pair[n];

      const int npts  = cn_all[ni].npts;
      const double ep = ep_mag[i];

      // local restrict pointers (the cosmo2D.c idiom); for nl1 = nl2 two
      // of them read the same row, which restrict allows (no writes)
      const double* restrict weight         = limber_weight[ni];
      const double* restrict density1       = cluster_density[ni][nl1];
      const double* restrict density2       = cluster_density[ni][nl2];
      const double* restrict magnification1 = cluster_magnification[ni][nl1];
      const double* restrict magnification2 = cluster_magnification[ni][nl2];
      const double* restrict pnl            = p_nonlinear[ni][i];

      double sum = 0.0;
      #pragma omp simd reduction(+:sum)
      for (int p = 0; p < npts; p++) {
        const double cluster_leg1 = density1[p] + ep*magnification1[p];
        const double cluster_leg2 = density2[p] + ep*magnification2[p];

        sum += cluster_leg1*cluster_leg2*pnl[p]*weight[p];
      }
      table[row][i] = sum;
    }
  }

  // --- 6. RELEASE ---
  free(limber_weight);
  free(cluster_density);
  free(cluster_magnification);
  free(p_nonlinear);
}


// ---------------------------------------------------------------------------
// Exact cluster-clustering C_l of every row (cluster bin ni, richness pair
// n) at nell multipoles: rows[ni*npairs_richness + n][i].
// ---------------------------------------------------------------------------
static void C_cc_tomo_limber_batch_rows(
    const double* ells,  // multipoles (length nell; need not be integers)
    const int nell,      // number of multipoles
    double** rows        // out [cc_npowerspectra*npairs_richness][nell]
  )
{
  if (nell <= 0) {
    log_fatal("nell = %d <= 0", nell);
    exit(1);
  }
  check_cluster_setup_cc();

  cluster_warmup();

  const int with_foreground = (0.0 != cluster.magnification);

  cosmo_nodes cn_all[cluster.zdist_nbin];
  const int npts_max = create_cosmo_nodes_cluster_all(cn_all, with_foreground);

  double* ep_mag = (double*) malloc1d(nell);
  for (int i = 0; i < nell; i++) {
    ep_mag[i] = ell_prefactor_magnification(ells[i]);
  }

  C_cc_tomo_limber_work(cn_all, npts_max, ells, ep_mag, nell, rows);

  free(ep_mag);
  free_cosmo_nodes_cluster_all(cn_all);
}


// ---------------------------------------------------------------------------
// Public batch: out[ni][n][i] = C_cc(ells[i]) of cluster bin ni and
// richness pair n = (NL1_cc(n), NL2_cc(n)) (out: malloc3d by the caller).
// ---------------------------------------------------------------------------
void C_cc_tomo_limber_nointerp_ells(
    const double* ells,  // multipoles (length nell)
    const int nell,      // number of multipoles
    double*** out        // out [cc_npowerspectra][npairs_richness][nell]
  )
{
  check_cluster_setup_cc();

  const int nbin_cluster    = cluster.cc_npowerspectra;
  const int npairs_richness = cc_richness_npairs();
  const int nrows           = nbin_cluster*npairs_richness;

  double** rows = (double**) malloc2d(nrows, nell);

  C_cc_tomo_limber_batch_rows(ells, nell, rows);

  for (int ni = 0; ni < nbin_cluster; ni++) {
    for (int n = 0; n < npairs_richness; n++) {
      for (int i = 0; i < nell; i++) {
        out[ni][n][i] = rows[ni*npairs_richness + n][i];
      }
    }
  }
  free(rows);
}


// ---------------------------------------------------------------------------
// The cached cluster-clustering table (exact at N_ell nodes for the BAO
// wiggles, spline onto the dense grid).
// ---------------------------------------------------------------------------
static const limber_table_cluster* C_cc_tomo_limber_table(void)
{
  static limber_table_cluster table;

  check_cluster_setup_cc();

  uint64_t keys[CLUSTER_NKEYS_MAX];
  const int nkeys = cluster_keys_cc(keys);

  const int nrows            = cluster.cc_npowerspectra*cc_richness_npairs();
  const int smooth_in_ln_ell = 0; // BAO wiggles: exact grid N_ell nodes

  limber_table_cluster_update(&table, nrows, smooth_in_ln_ell, keys, nkeys,
    C_cc_tomo_limber_batch_rows);

  return &table;
}


// ---------------------------------------------------------------------------
// Row of (cluster bin ni, richness pair nl1, nl2) in the cc block; the
// spectrum is symmetric in the richness pair, so the order is free.
// ---------------------------------------------------------------------------
static int cc_row(const int nl1, const int nl2, const int ni)
{
  int nl_low  = nl1;
  int nl_high = nl2;
  if (nl1 > nl2) {
    nl_low  = nl2;
    nl_high = nl1;
  }

  const int n = N_cc_richness(nl_low, nl_high);
  if (n < 0 || n > cc_richness_npairs() - 1) {
    log_fatal("(nl1, nl2) = (%d, %d) is not a cc richness pair", nl1, nl2);
    exit(1);
  }
  return ni*cc_richness_npairs() + n;
}


// ---------------------------------------------------------------------------
// C_cc at multipole l for richness bins (nl1, nl2) in cluster bin ni, read
// from the cached table.
// ---------------------------------------------------------------------------
double C_cc_tomo_limber(
    const double l,  // multipole
    const int nl1,   // first richness bin
    const int nl2,   // second richness bin
    const int ni     // cluster redshift bin (0 .. cc_npowerspectra-1)
  )
{
  check_cluster_setup_cc();

  if (nl1 < 0 || nl1 > cluster.richness_nbin - 1 ||
      nl2 < 0 || nl2 > cluster.richness_nbin - 1 ||
      ni < 0 || ni > cluster.cc_npowerspectra - 1)
  {
    log_fatal("error in selecting bin number (nl1, nl2, ni) = (%d, %d, %d)",
      nl1, nl2, ni);
    exit(1);
  }

  const limber_table_cluster* T = C_cc_tomo_limber_table();

  return limber_table_cluster_read(T, cc_row(nl1, nl2, ni), l);
}



// ============================================================================
// [SECTION] CLUSTER-GALAXY CLUSTERING: C_cg
// ============================================================================
//
// PHYSICAL DERIVATION & LOGIC FLOW (2503.13631 eqs 7, 9, 10, 20-21, 27-28)
//   1. cluster leg (bin ni, richness nl): b_nl W_c + C_c ep_mag W_mag,c
//   2. galaxy leg (lens bin ng = ZG_cg(n)), exactly cosmo2D.c's lens side:
//        b_1 W_gal + ep_mag b_mag W_mag     (gb1, gbmag, W_gal, W_mag)
//   3. Limber (eq 10):
//        C_cg(l) = int da (dchi/da)/f_K^2 [cluster leg] [galaxy leg] P_NL
//   4. Range: the cluster leg vanishes behind the cluster bin (both W_c and
//      the lensing efficiency g_c), so the cluster panels (support +
//      foreground) cover every nonzero integrand, the galaxy tails and the
//      galaxy magnification included.

static void check_cluster_setup_cg(void)
{
  check_cluster_setup();
  if (cluster.cg_npowerspectra <= 0) {
    log_fatal("cluster-galaxy clustering requested but cg_npowerspectra = %d",
      cluster.cg_npowerspectra);
    exit(1);
  }
  if (redshift.clustering_nbin <= 0) {
    log_fatal("cluster-galaxy clustering requested but clustering_nbin = %d",
      redshift.clustering_nbin);
    exit(1);
  }
}


// ---------------------------------------------------------------------------
// Core of every cluster-galaxy C_l.
//
// Memory layout (padded to npts_max; unused entries are 0):
//   limber_weight          [cluster bin][node]
//   cluster_density        [cluster bin][richness][node]   b W_c
//   cluster_magnification  [cluster bin][richness][node]   C_c W_mag,c
//   galaxy_density         [cg pair][node]   b_1 W_gal     (on the nodes of
//   galaxy_magnification   [cg pair][node]   b_mag W_mag    the pair's
//                                                            cluster bin)
//   p_nonlinear            [cluster bin][l][node]          P_NL(k, a)
//
// Output: table[n*richness_nbin + nl][i] for cg pair n = (ZC_cg(n),
// ZG_cg(n)), richness bin nl, multipole lx[i].
// ---------------------------------------------------------------------------
static void C_cg_tomo_limber_work(
    const cosmo_nodes* cn_all,   // nodes per cluster bin [zdist_nbin]
    const int npts_max,          // largest node count over the bins
    const double* lx,            // multipoles (length nell)
    const double* ep_mag,        // l(l+1)/(l+1/2)^2 per multipole
    const int nell,              // number of multipoles
    double** table               // out [cg_npowerspectra*richness_nbin][nell]
  )
{
  const int nbin_cluster  = cluster.zdist_nbin;
  const int nbin_richness = cluster.richness_nbin;
  const int nbin_lens     = redshift.clustering_nbin;
  const int npairs        = cluster.cg_npowerspectra;
  const int nrows         = npairs*nbin_richness;

  // --- 1. WARM-UP (single-threaded) ---
  warmup_cluster_leg(cn_all, lx);
  {
    const cosmo_nodes* cn = &cn_all[0];

    const double a       = cn->data[CN_A][0];
    const double fK      = cn->data[CN_FK][0];
    const double hoverh0 = cn->data[CN_HOVERH0][0];

    (void) W_gal(a, 0, hoverh0);
    (void) W_mag(a, fK, 0);
    (void) gb1(0.1, 0);
    (void) gbmag(0.1, 0);
  }

  // --- 2. PAIR LIST, read once outside the parallel regions ---
  int cluster_bin_of_pair[npairs];
  int lens_bin_of_pair[npairs];
  for (int n = 0; n < npairs; n++) {
    cluster_bin_of_pair[n] = ZC_cg(n);
    lens_bin_of_pair[n]    = ZG_cg(n);

    if (cluster_bin_of_pair[n] < 0 ||
        cluster_bin_of_pair[n] > nbin_cluster - 1 ||
        lens_bin_of_pair[n] < 0 ||
        lens_bin_of_pair[n] > nbin_lens - 1)
    {
      log_fatal("invalid cg pair %d: (cluster, lens) = (%d, %d)",
        n, cluster_bin_of_pair[n], lens_bin_of_pair[n]);
      exit(1);
    }
  }

  // --- 3. ALLOCATION ---
  double** limber_weight = (double**) malloc2d(nbin_cluster, npts_max);
  double*** cluster_density =
    (double***) malloc3d(nbin_cluster, nbin_richness, npts_max);
  double*** cluster_magnification =
    (double***) malloc3d(nbin_cluster, nbin_richness, npts_max);
  double** galaxy_density       = (double**) malloc2d(npairs, npts_max);
  double** galaxy_magnification = (double**) malloc2d(npairs, npts_max);
  double*** p_nonlinear =
    (double***) malloc3d(nbin_cluster, nell, npts_max);

  zero2d(limber_weight, nbin_cluster, npts_max);
  zero3d(cluster_density, nbin_cluster, nbin_richness, npts_max);
  zero3d(cluster_magnification, nbin_cluster, nbin_richness, npts_max);
  zero2d(galaxy_density, npairs, npts_max);
  zero2d(galaxy_magnification, npairs, npts_max);
  zero3d(p_nonlinear, nbin_cluster, nell, npts_max);

  // --- 4. NODE QUANTITIES: cluster leg, galaxy leg ---
  cluster_leg_at_nodes(cn_all, nbin_cluster, npts_max, limber_weight,
    NULL, cluster_density, cluster_magnification);

  #pragma omp parallel for collapse(2) schedule(static)
  for (int n = 0; n < npairs; n++) {
    for (int p = 0; p < npts_max; p++) {
      const int ni = cluster_bin_of_pair[n];
      const int ng = lens_bin_of_pair[n];

      const cosmo_nodes* cn = &cn_all[ni];
      if (p >= cn->npts) {
        continue; // padding node
      }

      const double a       = cn->data[CN_A][p];
      const double z       = 1.0/a - 1.0;
      const double fK      = cn->data[CN_FK][p];
      const double hoverh0 = cn->data[CN_HOVERH0][p];

      galaxy_density[n][p]       = W_gal(a, ng, hoverh0)*gb1(z, ng);
      galaxy_magnification[n][p] = W_mag(a, fK, ng)*gbmag(z, ng);
    }
  }

  nonlinear_power_at_nodes(cn_all, nbin_cluster, npts_max, lx, nell,
    p_nonlinear);

  // --- 5. LIMBER SUM: one row (pair, richness) and one l per task ---
  #pragma omp parallel for collapse(2) schedule(static)
  for (int row = 0; row < nrows; row++) {
    for (int i = 0; i < nell; i++) {
      const int n  = row/nbin_richness;
      const int nl = row - n*nbin_richness;
      const int ni = cluster_bin_of_pair[n];

      const int npts  = cn_all[ni].npts;
      const double ep = ep_mag[i];

      // local restrict pointers (the cosmo2D.c idiom)
      const double* restrict weight        = limber_weight[ni];
      const double* restrict density       = cluster_density[ni][nl];
      const double* restrict magnification = cluster_magnification[ni][nl];
      const double* restrict gal_density   = galaxy_density[n];
      const double* restrict gal_magnif    = galaxy_magnification[n];
      const double* restrict pnl           = p_nonlinear[ni][i];

      double sum = 0.0;
      #pragma omp simd reduction(+:sum)
      for (int p = 0; p < npts; p++) {
        const double cluster_leg = density[p] + ep*magnification[p];
        const double galaxy_leg  = gal_density[p] + ep*gal_magnif[p];

        sum += cluster_leg*galaxy_leg*pnl[p]*weight[p];
      }
      table[row][i] = sum;
    }
  }

  // --- 6. RELEASE ---
  free(limber_weight);
  free(cluster_density);
  free(cluster_magnification);
  free(galaxy_density);
  free(galaxy_magnification);
  free(p_nonlinear);
}


// ---------------------------------------------------------------------------
// Exact cluster-galaxy C_l of every row (cg pair n, richness nl) at nell
// multipoles: rows[n*richness_nbin + nl][i].
// ---------------------------------------------------------------------------
static void C_cg_tomo_limber_batch_rows(
    const double* ells,  // multipoles (length nell; need not be integers)
    const int nell,      // number of multipoles
    double** rows        // out [cg_npowerspectra*richness_nbin][nell]
  )
{
  if (nell <= 0) {
    log_fatal("nell = %d <= 0", nell);
    exit(1);
  }
  check_cluster_setup_cg();

  cluster_warmup();

  const int with_foreground = (0.0 != cluster.magnification);

  cosmo_nodes cn_all[cluster.zdist_nbin];
  const int npts_max = create_cosmo_nodes_cluster_all(cn_all, with_foreground);

  double* ep_mag = (double*) malloc1d(nell);
  for (int i = 0; i < nell; i++) {
    ep_mag[i] = ell_prefactor_magnification(ells[i]);
  }

  C_cg_tomo_limber_work(cn_all, npts_max, ells, ep_mag, nell, rows);

  free(ep_mag);
  free_cosmo_nodes_cluster_all(cn_all);
}


// ---------------------------------------------------------------------------
// Public batch: out[n][nl][i] = C_cg(ells[i]) of cg pair n = (ZC_cg(n),
// ZG_cg(n)) and richness bin nl (out: malloc3d by the caller).
// ---------------------------------------------------------------------------
void C_cg_tomo_limber_nointerp_ells(
    const double* ells,  // multipoles (length nell)
    const int nell,      // number of multipoles
    double*** out        // out [cg_npowerspectra][richness_nbin][nell]
  )
{
  check_cluster_setup_cg();

  const int nbin_richness = cluster.richness_nbin;
  const int npairs        = cluster.cg_npowerspectra;
  const int nrows         = npairs*nbin_richness;

  double** rows = (double**) malloc2d(nrows, nell);

  C_cg_tomo_limber_batch_rows(ells, nell, rows);

  for (int n = 0; n < npairs; n++) {
    for (int nl = 0; nl < nbin_richness; nl++) {
      for (int i = 0; i < nell; i++) {
        out[n][nl][i] = rows[n*nbin_richness + nl][i];
      }
    }
  }
  free(rows);
}


// ---------------------------------------------------------------------------
// The cached cluster-galaxy table (exact at N_ell nodes for the BAO
// wiggles, spline onto the dense grid).
// ---------------------------------------------------------------------------
static const limber_table_cluster* C_cg_tomo_limber_table(void)
{
  static limber_table_cluster table;

  check_cluster_setup_cg();

  uint64_t keys[CLUSTER_NKEYS_MAX];
  const int nkeys = cluster_keys_cg(keys);

  const int nrows            = cluster.cg_npowerspectra*cluster.richness_nbin;
  const int smooth_in_ln_ell = 0; // BAO wiggles: exact grid N_ell nodes

  limber_table_cluster_update(&table, nrows, smooth_in_ln_ell, keys, nkeys,
    C_cg_tomo_limber_batch_rows);

  return &table;
}


// ---------------------------------------------------------------------------
// C_cg at multipole l for richness bin nl, cluster bin ni, lens bin ng,
// read from the cached table. 0 for a (ni, ng) outside the pair list.
// ---------------------------------------------------------------------------
double C_cg_tomo_limber(
    const double l,  // multipole
    const int nl,    // richness bin
    const int ni,    // cluster redshift bin
    const int ng     // lens (galaxy) redshift bin
  )
{
  check_cluster_setup_cg();

  if (nl < 0 || nl > cluster.richness_nbin - 1 ||
      ni < 0 || ni > cluster.zdist_nbin - 1 ||
      ng < 0 || ng > redshift.clustering_nbin - 1)
  {
    log_fatal("error in selecting bin number (nl, ni, ng) = (%d, %d, %d)",
      nl, ni, ng);
    exit(1);
  }

  const limber_table_cluster* T = C_cg_tomo_limber_table();

  const int n = N_cg(ni, ng);
  if (n < 0) {
    return 0.0;
  }
  const int row = n*cluster.richness_nbin + nl;
  return limber_table_cluster_read(T, row, l);
}



// ============================================================================
// [SECTION] BIN-AVERAGED LEGENDRE KERNELS (private copies of cosmo2D.c's)
// ============================================================================
//
// The real-space statistics are full-sky Legendre sums, averaged over each
// angular bin (the theta binning of the galaxy statistics: Ntable.Ntheta
// log bins in [Ntable.vtmin, Ntable.vtmax], set_bin_average of basics.c):
//
//   spin 0 (w_cc, w_cg; eq 11, the w_gg_tomo kernel):
//     w(theta_i) = sum_l Pl0[i][l] C_l
//     Pl0[i][l]  = [P_{l+1}(xmin) - P_{l+1}(xmax) - P_{l-1}(xmin)
//                   + P_{l-1}(xmax)] / (4 pi (xmin - xmax))
//     from int P_l dx = [P_{l+1} - P_{l-1}]/(2l+1): the (2l+1) of the
//     antiderivative cancels the (2l+1)/(4 pi) of the Legendre series.
//
//   spin 2 (gamma_t of clusters; eq 12, the w_gammat_tomo kernel):
//     gamma_t(theta_i) = sum_l Pl2[i][l] C_l
//     Pl2[i][l] = (2l+1)/(4 pi l(l+1)(xmin - xmax))
//                 [ (l + 2/(2l+1)) (P_{l-1}(xmin) - P_{l-1}(xmax))
//                 + (2 - l) (xmin P_l(xmin) - xmax P_l(xmax))
//                 - 2/(2l+1) (P_{l+1}(xmin) - P_{l+1}(xmax)) ]
//     the bin average of P_l^2(x) = 2x P_l'(x) - l(l+1) P_l(x) (derivation
//     in cosmo2D.c's w_gammat_tomo).
//
// xmin = cos(theta_min) > xmax = cos(theta_max) (names track theta). The
// Legendre edge values are needed at l + 1 for l up to LMAX - 1, so the
// edge arrays hold LMAX + 1 entries (l = 0 .. LMAX).

// ---------------------------------------------------------------------------
// Pl[Ntheta][LMAX] of the given spin (0 or 2), l = 0 set to 0.
//
// Cache invalidation: rebuilt when Ntable.random, Ntable.Ntheta or
// Ntable.LMAX change. Call outside parallel regions.
// ---------------------------------------------------------------------------
static double** legendre_kernel_cluster(const int spin)
{
  static double** Pl[2]            = {NULL, NULL};  // [0]: spin 0, [1]: spin 2
  static uint64_t cache_ntable[2]  = {0, 0};
  static int ntheta_built[2]       = {0, 0};
  static int lmax_built[2]         = {0, 0};

  int kind = 0;
  if (2 == spin) {
    kind = 1;
  }
  else if (0 != spin) {
    log_fatal("spin = %d not supported (0 or 2)", spin);
    exit(1);
  }

  if (NULL == Pl[kind] ||
      fdiff2(cache_ntable[kind], Ntable.random) ||
      ntheta_built[kind] != Ntable.Ntheta ||
      lmax_built[kind] != Ntable.LMAX)
  {
    const int ntheta = Ntable.Ntheta;
    const int lmax   = Ntable.LMAX;
    const int lmin   = 1;

    if (Pl[kind] != NULL) {
      free(Pl[kind]);
    }
    Pl[kind] = (double**) malloc2d(ntheta, lmax);
    double** kernel = Pl[kind];

    // Legendre polynomials at the bin edges, l = 0 .. LMAX
    double*** P = (double***) malloc3d(2, ntheta, lmax + 1);
    double** Pmin = P[0];
    double** Pmax = P[1];

    double xmin[ntheta];
    double xmax[ntheta];
    for (int i = 0; i < ntheta; i++) {
      // serial: the first call initializes set_bin_average's statics
      const bin_avg r = set_bin_average(i, 0);
      xmin[i] = r.xmin;
      xmax[i] = r.xmax;
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < ntheta; i++) {
      for (int l = 0; l < lmax + 1; l++) {
        const bin_avg r = set_bin_average(i, l);
        Pmin[i][l] = r.Pmin;
        Pmax[i][l] = r.Pmax;
      }
    }

    // no monopole in the sums
    for (int i = 0; i < ntheta; i++) {
      kernel[i][0] = 0.0;
    }

    if (0 == kind) {
      #pragma omp parallel for collapse(2) schedule(static)
      for (int i = 0; i < ntheta; i++) {
        for (int l = lmin; l < lmax; l++) {
          const double norm = (1.0/(xmin[i] - xmax[i]))*(1.0/(4.0*M_PI));
          kernel[i][l] = norm*(Pmin[i][l + 1] - Pmax[i][l + 1]
                               - Pmin[i][l - 1] + Pmax[i][l - 1]);
        }
      }
    }
    else {
      #pragma omp parallel for collapse(2) schedule(static)
      for (int i = 0; i < ntheta; i++) {
        for (int l = lmin; l < lmax; l++) {
          const double ll = (double) l;

          // (2l+1)/(4 pi l (l+1)): Legendre norm x single spin-2 field
          const double prefactor =
            (2.0*ll + 1.0)/(4.0*M_PI*ll*(ll + 1.0)*(xmin[i] - xmax[i]));

          const double term_lm1 =
            (ll + 2.0/(2.0*ll + 1.0))*(Pmin[i][l - 1] - Pmax[i][l - 1]);
          const double term_l =
            (2.0 - ll)*(xmin[i]*Pmin[i][l] - xmax[i]*Pmax[i][l]);
          const double term_lp1 =
            2.0/(2.0*ll + 1.0)*(Pmin[i][l + 1] - Pmax[i][l + 1]);

          kernel[i][l] = prefactor*(term_lm1 + term_l - term_lp1);
        }
      }
    }

    free(P);

    cache_ntable[kind] = Ntable.random;
    ntheta_built[kind] = ntheta;
    lmax_built[kind]   = lmax;
  }
  return Pl[kind];
}



// ============================================================================
// [SECTION] REAL SPACE: gamma_t OF CLUSTERS, w_cc, w_cg
// ============================================================================
//
// One cached block per statistic: w_vec[row][theta] for every row of the
// statistic's Limber block. The C_l of a row at every integer l < LMAX:
//
//   l = 0                      : 0 (no monopole)
//   l = 1 .. LMIN_tab - 1      : exact Limber batch at each integer l
//   l = LMIN_tab .. LMAX - 1   : the cached log-l table, read with
//                                limber_fill_interp (cosmo2D.c)
//
// then one Legendre sum per (row, theta bin). The table's first node sits
// exactly at LMIN_tab, so the two C_l pieces join without a step.

typedef struct
{
  double** Cl;       // [nrows][LMAX]: C_l at every integer l
  double* w_vec;     // [nrows][Ntheta]: the real-space block
  double* lnell;     // [LMAX + 1]: ln l, read by limber_fill_interp
  int nrows;
  int ntheta;
  int lmax;
  uint64_t cache_ntable;              // Ntable.random of the allocation
  uint64_t cache[CLUSTER_NKEYS_MAX];  // keys of the values
} real_space_cluster;


// ---------------------------------------------------------------------------
// Bring a real-space block up to date.
//
//   1. GEOMETRY (Ntable.random, Ntheta, LMAX, block size): allocate.
//   2. VALUES (the statistic's keys): C_l at every integer l (exact batch
//      below the table, table fill above), then the Legendre sums,
//      collapse(2) over (row, theta) with a vectorized sum over l.
//
// Thread safety: call outside parallel regions.
// ---------------------------------------------------------------------------
static void real_space_cluster_update(
    real_space_cluster* R,          // the statistic's static block
    const int nrows,                // rows of the statistic's block
    const int spin,                 // 0 (w) or 2 (gamma_t)
    const uint64_t* keys,           // the statistic's current keys
    const int nkeys,                // number of keys
    limber_table_getter get_table,  // the statistic's cached l table
    limber_batch_rows batch_rows    // the statistic's exact Limber batch
  )
{
  int refill = 0;

  // --- 1. GEOMETRY ---
  if (NULL == R->Cl ||
      fdiff2(R->cache_ntable, Ntable.random) ||
      nrows != R->nrows ||
      Ntable.Ntheta != R->ntheta ||
      Ntable.LMAX != R->lmax)
  {
    if (Ntable.LMAX <= limits.LMIN_tab) {
      log_fatal("Ntable.LMAX = %d <= limits.LMIN_tab = %d",
        Ntable.LMAX, limits.LMIN_tab);
      exit(1);
    }
    if (R->Cl != NULL) {
      free(R->Cl);
    }
    if (R->w_vec != NULL) {
      free(R->w_vec);
    }
    if (R->lnell != NULL) {
      free(R->lnell);
    }

    R->nrows  = nrows;
    R->ntheta = Ntable.Ntheta;
    R->lmax   = Ntable.LMAX;

    R->lnell = (double*) malloc1d(Ntable.LMAX + 1);
    R->lnell[0] = 0.0; // never read (the fill starts at l >= 1)
    for (int l = 1; l <= Ntable.LMAX; l++) {
      R->lnell[l] = log((double) l);
    }

    R->Cl = (double**) malloc2d(nrows, Ntable.LMAX);
    zero2d(R->Cl, nrows, Ntable.LMAX);

    R->w_vec = (double*) calloc1d(nrows*Ntable.Ntheta);

    R->cache_ntable = Ntable.random;
    refill = 1;
  }

  // --- 2. VALUES ---
  if (1 == refill || keys_changed(R->cache, keys, nkeys)) {
    double** Pl = legendre_kernel_cluster(spin);
    const limber_table_cluster* T = get_table();

    const int lmax = Ntable.LMAX;

    // first multipole read from the table
    int l_table_min = limits.LMIN_tab;
    if (l_table_min < 1) {
      l_table_min = 1;
    }

    // l = 0: no monopole
    for (int row = 0; row < nrows; row++) {
      R->Cl[row][0] = 0.0;
    }

    // l = 1 .. l_table_min - 1: exact Limber quadrature at each integer l
    const int nexact = l_table_min - 1;
    if (nexact > 0) {
      double* ells = (double*) malloc1d(nexact);
      for (int i = 0; i < nexact; i++) {
        ells[i] = (double) (i + 1);
      }
      double** exact = (double**) malloc2d(nrows, nexact);

      batch_rows(ells, nexact, exact);

      for (int row = 0; row < nrows; row++) {
        for (int i = 0; i < nexact; i++) {
          R->Cl[row][i + 1] = exact[row][i];
        }
      }
      free(exact);
      free(ells);
    }

    // l = l_table_min .. LMAX - 1: the log-l table, read at every integer
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < nrows; row++) {
      const double* tab[1] = { T->tab[row] };
      double* dst[1]       = { R->Cl[row] };
      limber_fill_interp(1, tab, dst, l_table_min, lmax, R->lnell,
        T->lim[0], 1.0/T->lim[2], T->nell);
    }

    // Legendre sums: one (row, theta bin) per task, serial sum over l
    const int ntheta = Ntable.Ntheta;

    #pragma omp parallel for collapse(2) schedule(static)
    for (int row = 0; row < nrows; row++) {
      for (int i = 0; i < ntheta; i++) {
        // Local restrict pointers: without them the compiler cannot prove
        // that Pl[i] and Cl[row] do not alias (pointer-to-pointer
        // indirection inside a collapse(2) region) and emits
        // reload-checking code; the body is a single multiply-add with
        // nothing to hide that overhead behind (SKILL.md, pitfall 1)
        const double* restrict pl = Pl[i];
        const double* restrict cl = R->Cl[row];

        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int l = 1; l < lmax; l++) {
          sum += pl[l]*cl[l];
        }
        R->w_vec[row*ntheta + i] = sum;
      }
    }

    keys_stamp(R->cache, keys, nkeys);
  }
}


// basic checks of a real-space request
static void check_real_space_setup(const int nt)
{
  if (0 == Ntable.Ntheta) {
    log_fatal("Ntable.Ntheta not initialized");
    exit(1);
  }
  if (nt < 0 || nt > Ntable.Ntheta - 1) {
    log_fatal("error in selecting bin number nt = %d (max %d)",
      nt, Ntable.Ntheta);
    exit(1);
  }
}


// ---------------------------------------------------------------------------
// Cluster tangential shear gamma_t(theta_nt) of richness bin nl, cluster
// bin ni and source bin ns: full-sky, bin-averaged spin-2 Legendre sum of
// C_cs (eq 12). This is gamma_t BEFORE the Y transform (eq 15), the
// selection bias (eq 23) and the shear calibration (1 + m): the interface
// applies all three on the data vector. It is computed at every theta bin
// regardless of the mask, because the Y transform mixes neighbouring bins.
//
// Returns: gamma_t; 0 for a (ni, ns) outside the cs pair list.
// ---------------------------------------------------------------------------
double w_gammat_cluster_tomo(
    const int nt,  // theta bin
    const int nl,  // richness bin
    const int ni,  // cluster redshift bin
    const int ns   // source redshift bin
  )
{
  static real_space_cluster block;

  check_cluster_setup_cs();
  check_real_space_setup(nt);

  if (nl < 0 || nl > cluster.richness_nbin - 1 ||
      ni < 0 || ni > cluster.zdist_nbin - 1 ||
      ns < 0 || ns > redshift.shear_nbin - 1)
  {
    log_fatal("error in selecting bin number (nl, ni, ns) = (%d, %d, %d)",
      nl, ni, ns);
    exit(1);
  }

  uint64_t keys[CLUSTER_NKEYS_MAX];
  const int nkeys = cluster_keys_cs(keys);

  const int nrows = cluster.cs_npowerspectra*cluster.richness_nbin;
  const int spin  = 2;

  real_space_cluster_update(&block, nrows, spin, keys, nkeys,
    C_cs_tomo_limber_table, C_cs_tomo_limber_batch_rows);

  const int n = N_cs(ni, ns);
  if (n < 0) {
    return 0.0;
  }
  const int row = n*cluster.richness_nbin + nl;
  return block.w_vec[row*Ntable.Ntheta + nt];
}


// ---------------------------------------------------------------------------
// Cluster-cluster angular correlation w_cc(theta_nt) of richness bins
// (nl1, nl2) in cluster bin ni: full-sky, bin-averaged spin-0 Legendre sum
// of C_cc (eq 11), before the selection bias (the interface multiplies by
// its square, eq 23). limber = 0 (the FKEM non-Limber split) is Phase 4.
// ---------------------------------------------------------------------------
double w_cc_tomo(
    const int nt,     // theta bin
    const int nl1,    // first richness bin
    const int nl2,    // second richness bin
    const int ni,     // cluster redshift bin (0 .. cc_npowerspectra-1)
    const int limber  // 1: Limber; 0: non-Limber (not implemented yet)
  )
{
  static real_space_cluster block;

  if (1 != limber) {
    log_fatal("w_cc_tomo: non-Limber not implemented yet (limber = %d)",
      limber);
    exit(1);
  }

  check_cluster_setup_cc();
  check_real_space_setup(nt);

  if (nl1 < 0 || nl1 > cluster.richness_nbin - 1 ||
      nl2 < 0 || nl2 > cluster.richness_nbin - 1 ||
      ni < 0 || ni > cluster.cc_npowerspectra - 1)
  {
    log_fatal("error in selecting bin number (nl1, nl2, ni) = (%d, %d, %d)",
      nl1, nl2, ni);
    exit(1);
  }

  uint64_t keys[CLUSTER_NKEYS_MAX];
  const int nkeys = cluster_keys_cc(keys);

  const int nrows = cluster.cc_npowerspectra*cc_richness_npairs();
  const int spin  = 0;

  real_space_cluster_update(&block, nrows, spin, keys, nkeys,
    C_cc_tomo_limber_table, C_cc_tomo_limber_batch_rows);

  const int row = cc_row(nl1, nl2, ni);
  return block.w_vec[row*Ntable.Ntheta + nt];
}


// ---------------------------------------------------------------------------
// Cluster-galaxy angular correlation w_cg(theta_nt) of richness bin nl,
// cluster bin ni and lens bin ng: full-sky, bin-averaged spin-0 Legendre
// sum of C_cg (eq 11), before the selection bias (the interface, eq 23).
// limber = 0 (the non-Limber option) is Phase 4.
//
// Returns: w_cg; 0 for a (ni, ng) outside the cg pair list.
// ---------------------------------------------------------------------------
double w_cg_tomo(
    const int nt,     // theta bin
    const int nl,     // richness bin
    const int ni,     // cluster redshift bin
    const int ng,     // lens (galaxy) redshift bin
    const int limber  // 1: Limber; 0: non-Limber (not implemented yet)
  )
{
  static real_space_cluster block;

  if (1 != limber) {
    log_fatal("w_cg_tomo: non-Limber not implemented yet (limber = %d)",
      limber);
    exit(1);
  }

  check_cluster_setup_cg();
  check_real_space_setup(nt);

  if (nl < 0 || nl > cluster.richness_nbin - 1 ||
      ni < 0 || ni > cluster.zdist_nbin - 1 ||
      ng < 0 || ng > redshift.clustering_nbin - 1)
  {
    log_fatal("error in selecting bin number (nl, ni, ng) = (%d, %d, %d)",
      nl, ni, ng);
    exit(1);
  }

  uint64_t keys[CLUSTER_NKEYS_MAX];
  const int nkeys = cluster_keys_cg(keys);

  const int nrows = cluster.cg_npowerspectra*cluster.richness_nbin;
  const int spin  = 0;

  real_space_cluster_update(&block, nrows, spin, keys, nkeys,
    C_cg_tomo_limber_table, C_cg_tomo_limber_batch_rows);

  const int n = N_cg(ni, ng);
  if (n < 0) {
    return 0.0;
  }
  const int row = n*cluster.richness_nbin + nl;
  return block.w_vec[row*Ntable.Ntheta + nt];
}



// ============================================================================
// [SECTION] CLUSTER NUMBER COUNTS
// ============================================================================
//
// PHYSICAL DERIVATION & LOGIC FLOW (2503.13631 eq 16)
//   1. Comoving volume per unit redshift and solid angle (flat):
//        dV/dz dOmega = f_K(chi)^2 / (H/H0)            [(c/H0)^3 / sr]
//   2. Clusters of richness bin nl per unit volume at true redshift z:
//        n_nl(z) = int dlnM (dn/dlnM) P(nl|M, z)       (ncl_richness)
//   3. Probability that such a cluster lands in redshift bin ni:
//        <phi_ni|z>                                     (phi_cluster)
//   4. N_{ni, nl} = Omega_s int dz [dV/dz dOmega] <phi_ni|z> n_nl(z)
//      over the support of <phi_ni|z>, with Omega_s = survey.area (deg^2)
//      times (pi/180)^2 sr per deg^2. The conversion is a local constant:
//      survey.area_conversion_factor is set only by reset_survey_struct,
//      which the interface never calls, so it is 0 at run time.
//   5. Gauss-Legendre in z on the support: the same tabulated-size rule
//      (hdi ladder) as the Limber support panel; a top-hat kernel is
//      smooth inside its support, an erf kernel has its edges resolved.

// ---------------------------------------------------------------------------
// Fill counts[ni][nl] for every cluster bin and richness bin (eq 16),
// serially: a few hundred node evaluations per bin, and no thread can
// change the summation order.
// ---------------------------------------------------------------------------
static void cluster_counts_fill(
    double** counts  // out [zdist_nbin][richness_nbin]
  )
{
  const int nbin_cluster  = cluster.zdist_nbin;
  const int nbin_richness = cluster.richness_nbin;

  const gsl_integration_glfixed_table* w = limber_gl_table_cluster();
  const int nodes = (int) w->n;

  // survey solid angle in steradians
  const double deg2_to_sr = (M_PI/180.0)*(M_PI/180.0);
  const double omega_s = survey.area*deg2_to_sr;

  zero2d(counts, nbin_cluster, nbin_richness);

  for (int ni = 0; ni < nbin_cluster; ni++) {
    // support of <phi_ni|z> in redshift (the far edge in a is amin_cluster)
    double z_min       = 1.0/amax_cluster(ni) - 1.0;
    const double z_max = 1.0/amin_cluster(ni) - 1.0;
    if (z_min < 0.0) {
      z_min = 0.0; // a rounding of a = 1 must not reach negative z
    }
    if (!(z_min < z_max)) {
      log_fatal("invalid support of cluster bin %d: z = [%e, %e]",
        ni, z_min, z_max);
      exit(1);
    }

    for (int q = 0; q < nodes; q++) {
      double z;
      double wq;
      gsl_integration_glfixed_point(z_min, z_max, q, &z, &wq, w);

      const double a         = 1.0/(1.0 + z);
      const struct chis cdca = chi_all(a);
      const double fK        = cdca.chi;
      const double hoverh0   = hoverh0v2(a, cdca.dchida);

      // dV/dz dOmega = f_K^2 / (H/H0), and the bin's selection kernel
      const double dVdz = fK*fK/hoverh0;
      const double phi  = phi_cluster(z, ni);

      for (int nl = 0; nl < nbin_richness; nl++) {
        counts[ni][nl] += wq*dVdz*phi*ncl_richness(a, nl);
      }
    }

    for (int nl = 0; nl < nbin_richness; nl++) {
      counts[ni][nl] *= omega_s;
    }
  }
}


// ---------------------------------------------------------------------------
// Expected number of clusters in redshift bin ni and richness bin nl
// (eq 16), read from a table of every (ni, nl).
//
// Cache invalidation: Ntable.random, cosmology.random, cluster.random_model,
// cluster.random_zdist, cluster.random_mor and the survey area.
// ---------------------------------------------------------------------------
double N_cluster_tomo(
    const int nl,  // richness bin
    const int ni   // cluster redshift bin
  )
{
  static double** counts = NULL;      // [zdist_nbin][richness_nbin]
  static int nbin_cluster_built  = 0;
  static int nbin_richness_built = 0;
  static uint64_t cache[CLUSTER_NKEYS_MAX];

  check_cluster_setup();

  if (nl < 0 || nl > cluster.richness_nbin - 1 ||
      ni < 0 || ni > cluster.zdist_nbin - 1)
  {
    log_fatal("error in selecting bin number (nl, ni) = (%d, %d)", nl, ni);
    exit(1);
  }
  if (!(survey.area > 0.0)) {
    log_fatal("survey.area = %e deg^2 not set", survey.area);
    exit(1);
  }

  uint64_t keys[CLUSTER_NKEYS_MAX];
  int nkeys = 0;
  keys[nkeys++] = Ntable.random;
  keys[nkeys++] = cosmology.random;
  keys[nkeys++] = cluster.random_model;
  keys[nkeys++] = cluster.random_zdist;
  keys[nkeys++] = cluster.random_mor;
  keys[nkeys++] = double_bits_key(survey.area);

  int refill = 0;

  // --- 1. GEOMETRY ---
  if (NULL == counts ||
      cluster.zdist_nbin != nbin_cluster_built ||
      cluster.richness_nbin != nbin_richness_built)
  {
    if (counts != NULL) {
      free(counts);
    }
    counts = (double**) malloc2d(cluster.zdist_nbin, cluster.richness_nbin);
    nbin_cluster_built  = cluster.zdist_nbin;
    nbin_richness_built = cluster.richness_nbin;
    refill = 1;
  }

  // --- 2. VALUES ---
  if (1 == refill || keys_changed(cache, keys, nkeys)) {
    cluster_warmup();
    cluster_counts_fill(counts);
    keys_stamp(cache, keys, nkeys);
  }

  return counts[ni][nl];
}
