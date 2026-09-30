#ifndef __COSMOLIKE_COSMO2D_CLUSTER_H
#define __COSMOLIKE_COSMO2D_CLUSTER_H
#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// [SECTION] CLUSTER ANGULAR STATISTICS (the cosmo2D.c design)
// ============================================================================
//
// Naming: c = clusters, s = shear (sources), g = galaxy positions (lenses).
// Index names: nt = theta bin, nl = richness bin, ni = cluster redshift bin,
// ns = source bin, ng = lens bin. Units: chi in c/H0, k in (c/H0)^-1.
//
// Limber: k = (l + 1/2)/f_K(chi), batched over every pair and multipole in
// one work function per statistic (the C_gs_tomo_limber_work design), with
// cosmo_nodes kept private to cosmo2D_cluster.c (no core export).
//
//   C_cs = int dchi/f_K^2 { [W_kappa_ns - W_IA_ns] (b_nl W_c - C_c W_mag,c)
//                           P_NL + W_kappa_ns W_c P1h_nl }
//   C_cc = int dchi/f_K^2 (b_nl1 W_c + C_c W_mag,c)(b_nl2 W_c + C_c W_mag,c)
//                           P_NL
//   C_cg = int dchi/f_K^2 (b_nl W_c + C_c W_mag,c)(b_g W_gal + b_mag W_mag)
//                           P_NL
// with W_c = W_cluster(a, ni, nl, H/H0), W_mag,c = W_mag_cluster(...),
// C_c = cluster.magnification (the sign convention of the magnification
// term follows cosmo2D.c's galaxy b_mag term), W_IA only when
// cluster.include_ia. Magnification kernels extend over the whole
// foreground (a up to 1), not only the cluster bin.
//
// Real space: full-sky, angular-bin-averaged Legendre sums with the same
// kernels as w_gammat_tomo (P_l^2, spin 2) and w_gg_tomo (P_l, spin 0),
// over the same theta binning (Ntable.Ntheta, vtmin, vtmax) and
// l < Ntable.LMAX.

// ---------------------------------------------------------------------------
// Limber batches at arbitrary multipoles (the *_nointerp_ells design).
// Output layouts (malloc3d by the caller):
//   cs: out[cs pair n][nl][ell]        (ZC_cs(n), ZS_cs(n))
//   cc: out[cluster bin ni][richness pair n][ell]  (NL1_cc(n), NL2_cc(n))
//   cg: out[cg pair n][nl][ell]        (ZC_cg(n), ZG_cg(n))
// ---------------------------------------------------------------------------
void C_cs_tomo_limber_nointerp_ells(const double* ells, const int nell,
  double*** out);

void C_cc_tomo_limber_nointerp_ells(const double* ells, const int nell,
  double*** out);

void C_cg_tomo_limber_nointerp_ells(const double* ells, const int nell,
  double*** out);

// ---------------------------------------------------------------------------
// Cached Limber tables (log-spaced l, refilled on the cache keys), read
// at one multipole: wrappers and diagnostics.
// ---------------------------------------------------------------------------
double C_cs_tomo_limber(const double l, const int nl, const int ni,
  const int ns);

double C_cc_tomo_limber(const double l, const int nl1, const int nl2,
  const int ni);

double C_cg_tomo_limber(const double l, const int nl, const int ni,
  const int ng);

// ---------------------------------------------------------------------------
// Real-space statistics at theta bin nt (cached per block).
// w_gammat_cluster_tomo is the cluster tangential shear gamma_t BEFORE the
// Y transform and the selection bias (the interface applies both on the
// data vector, eqs 15 and 23). limber = 0 selects the non-Limber w_cc
// (FKEM split, the C_cl_tomo design; Phase 4).
// ---------------------------------------------------------------------------
double w_gammat_cluster_tomo(const int nt, const int nl, const int ni,
  const int ns);

double w_cc_tomo(const int nt, const int nl1, const int nl2, const int ni,
  const int limber);

double w_cg_tomo(const int nt, const int nl, const int ni, const int ng,
  const int limber);

// ---------------------------------------------------------------------------
// Expected number of clusters in redshift bin ni and richness bin nl
// (eq 16):  N = Omega_s int dz dV/dz dOmega <phi_ni|z> n_nl(z),
// Omega_s = survey.area in steradians.
// ---------------------------------------------------------------------------
double N_cluster_tomo(const int nl, const int ni);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
