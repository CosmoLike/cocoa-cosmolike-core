#ifndef __COSMOLIKE_HALO_CLUSTER_H
#define __COSMOLIKE_HALO_CLUSTER_H
#ifdef __cplusplus
extern "C" {
#endif

// ============================================================================
// [SECTION] HALO-MODEL CLUSTER QUANTITIES (the halo.c design)
// ============================================================================
//
// Built on the Tinker 2010 multiplicity and halo.c's bias hb1nu (M200m,
// Delta = 200 mean), sigma2 (cosmo3D.c), dlognudlogm, conc (Bhattacharya
// 2013) and the NFW profile. The multiplicity amplitude follows
// cluster.hmf_alpha_mode: alpha = 0.368 at every z (DES; default) or
// halo.c's fnu, alpha(a) from int b f dnu = 1. Units: M in Msun/h, k in
// (c/H0)^-1, number densities in (c/H0)^-3, P(k) in (c/H0)^3.
//
// The tables below come out of one deep-unrolled fill over (richness bin,
// a node) with Gauss-Legendre nodes in ln M on [ln cluster.m[RANGE_MIN],
// ln cluster.m[RANGE_MAX]] (halo_nm / high_def_integration ladder). Their
// a-range covers every cluster redshift bin's support; outside it they
// return 0.
// Cache keys: Ntable.random, cosmology.random, cluster.random_model,
// cluster.random_zdist, cluster.random_mor (and cluster.random_selection
// when cluster.selection_model == CLUSTER_SELECTION_Y1).

// ---------------------------------------------------------------------------
// Probability that a halo of mass M at redshift z has observed richness in
// bin nl, for the lognormal MOR of eqs (18)-(19):
//   <ln lambda|M,z> = mor[0] + mor[1] ln(M/M_piv)
//                     + mor[3] ln((1+z)/(1+z_piv))
//   sigma^2 = mor[2]^2 + (exp(<ln lambda>) - 1)/exp(2 <ln lambda>)
//             (the Poisson term only when <ln lambda> > 0)
//   P = [erf(x_max) - erf(x_min)]/2,  x = (ln lambda_edge - <ln lambda>)
//                                        /(sqrt(2) sigma)
// (evaluated with erfc of the tails: accurate where P is tiny)
// Closed form: no table.
// ---------------------------------------------------------------------------
double prob_richness_bin_given_m(const double lnM, const double z,
  const int nl);

// ---------------------------------------------------------------------------
// Comoving number density of clusters in richness bin nl (eq 16 inner
// integrals):  n_nl(a) = int dlnM (dn/dlnM)(M, a) P(nl|M, z(a))
// ---------------------------------------------------------------------------
double ncl_richness(const double a, const int nl);

// ---------------------------------------------------------------------------
// Richness-weighted linear bias (eq 21):
//   b_nl(a) = int dlnM (dn/dlnM) P(nl|M) b_h(M, a) S(M, a) / n_nl(a)
// with S = 1, or the Y1 mass-dependent selection bias
// b_s0 (M/M_piv)^b_s1 ((1+z)/(1+z_piv))^b_s2 (cluster.mor_pivot_mass,
// cluster.mor_pivot_1pz) when
// cluster.selection_model == CLUSTER_SELECTION_Y1.
// ---------------------------------------------------------------------------
double bcl_richness(const double a, const int nl);

// ---------------------------------------------------------------------------
// One-halo cluster-matter power spectrum of richness bin nl (eq 22):
//   P1h_nl(k, a) = int dlnM (dn/dlnM) P(nl|M) (M/rho_m) u_NFW(k|M, a)
//                  / n_nl(a)
// Exact ln k nodes (the halo_nk_step ladder, refined: a richness bin
// selects a narrow mass window, so P1h keeps NFW wiggles in k), house
// spline in ln k, linear in a.
// ---------------------------------------------------------------------------
double pcm_1h_richness(const double k, const double a, const int nl);

// ---------------------------------------------------------------------------
// The same spectrum at n points and every richness bin in one call:
//   out[nl][q] = pcm_1h_richness(k[q], a[q], nl),  q = 0 .. n-1,
//   nl = 0 .. cluster.richness_nbin - 1
// (out: cluster.richness_nbin rows of length >= n; k, a and the rows of
// out must not overlap). The read of the cluster-lensing Limber integrand:
// the n points are the quadrature nodes of one multipole. A point's place
// on the table is found once and serves every richness bin, four points
// per SIMDe vector; every value is bitwise the scalar pcm_1h_richness
// call at that point. After cluster_warmup, calls from threaded loops
// only read the cached table, as in pcm_1h_richness.
// ---------------------------------------------------------------------------
void pcm_1h_richness_fill(const double* k, const double* a, const int n,
  double** out);

// ---------------------------------------------------------------------------
// Builds the lazily filled cluster tables serially: the mass tables of
// this file (its one-halo table only when cluster lensing is in the data
// vector; the cluster-lensing Limber code warms that table itself), then
// the selection-kernel, n(z) and lensing-efficiency tables of
// redshift_spline_cluster.c. The interface calls it before any threaded
// loop reads a cluster table.
// ---------------------------------------------------------------------------
void cluster_warmup(void);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
