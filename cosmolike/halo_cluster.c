#include <float.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_sf.h>

#include "basics.h"
#include "cosmo3D.h"
#include "halo.h"
#include "halo_cluster.h"
#include "redshift_spline_cluster.h"
#include "structs.h"
#include "structs_cluster.h"

#include "log.c/src/log.h"

// ---------------------------------------------------------------------------
// Halo-model quantities of galaxy clusters selected in observed richness:
// the cluster analog of halo.c for the DES cluster analyses (model of
// arXiv 2503.13631, equation numbers below; Y1 switches from arXiv
// 2008.10757). It replaces the legacy cluster_util.c.
//
// Physics. A cluster is a dark-matter halo of mass M (M200m) whose
// observed richness lambda scatters around a mean set by M and z, the
// lognormal mass-observable relation (MOR) of eqs (18)-(19):
//
//   <ln lambda|M, z> = mor[0] + mor[1] ln(M/M_piv)
//                      + mor[3] ln((1 + z)/(1 + z_piv))
//   sigma^2_lnlambda = mor[2]^2 + (e^<ln lambda> - 1)/e^(2 <ln lambda>)
//                      (the Poisson term only when <ln lambda> > 0)
//
// so the probability that the halo lands in richness bin nl,
// [lambda_min, lambda_max), is a difference of two error functions,
// P(nl|M, z). Weighting the halo mass function with it gives the three
// tables of this file:
//
//   (16) n_nl(a)      = int dlnM (dn/dlnM) P(nl|M, z)
//   (21) b_nl(a)      = int dlnM (dn/dlnM) P(nl|M, z) b_h(M, a) / n_nl
//   (22) P1h_nl(k, a) = int dlnM (dn/dlnM) P(nl|M, z) (M/rho_m) u(k|M, a)
//                       / n_nl
//
// with the ingredients of halo.c, used exactly as its mass integrals use
// them (hod_tables, p_gm):
//
//   dn/dlnM = (rho_m/M) nu f(nu) dln nu/dln M    mass function (fnu,
//                                                dlognudlogm)
//   nu      = delta_c/(sigma(M) D(a))            peak height (sigma2 of
//                                                cosmo3D.c, D = growfac)
//   b_h     = hb1nu(nu, a)                       Tinker 2010 halo bias
//   u(k|M)  = truncated NFW transform, c = conc(M, D) (Bhattacharya 2013)
//   rho_m   = rho_crit Omega_m                   total matter (the halo.c
//                                                convention, neutrinos
//                                                included)
//
// and, when cluster.selection_model == CLUSTER_SELECTION_Y1, the Y1
// mass-dependent selection bias (Y1 eqs 1 and 31) inside the bias
// integral:
//
//   b_h -> b_h S,   S = b_s0 (M/M_piv)^b_s1 ((1 + z)/(1 + z_piv))^b_s2
//
// with b_s0, b_s1, b_s2 = cluster.selection[0], [1], [2] and the MOR
// pivots M_piv, 1 + z_piv (Y1 uses the same 5e14 Msun/h and 1.45).
//
// Units (the library's): M in Msun/h, k in (c/H0)^-1, n in (c/H0)^-3,
// P(k) in (c/H0)^3.
//
// Glossary of the file:
//
//   prob_richness_bin_given_m = P(nl|M, z), closed form, no table
//   ncl_richness   = n_nl(a), the comoving number density ("n cluster")
//   bcl_richness   = b_nl(a), the richness-weighted linear bias
//   pcm_1h_richness = P1h_nl(k, a), the one-halo cluster-matter spectrum
//                    ("p" power, "cm" cluster-matter, "1h" one halo)
//   cluster_mass_tables = THE fill: one deep-unrolled loop nest over
//                    (richness bin, a node) with the Gauss-Legendre mass
//                    nodes innermost; it fills n_nl, b_nl and the 1-halo
//                    mass weights
//   cluster_p1h_table = P1h on exact ln k nodes from those weights
//   cluster_nfw_*  = a private copy of halo.c's NFW kernel (nfw_um and its
//                    f, G table), verified against halo.c's u_nfw_c
//   cluster_warmup = builds every lazy cluster table on one thread
//
// Tables and reads:
//
//   a grid  Ntable.halo_na_lens nodes uniform in a over every cluster
//           bin's support, [1/(1 + max_i zdist_zmax[i]),
//           1/(1 + min_i zdist_zmin[i])]; all three tables return 0
//           outside it
//   n_nl    ln n on the a nodes plus Ntable.halo_spline_pad exact pad
//           nodes beyond each end, read with the house natural cubic
//           spline (spline_coeffs_uniform + Horner)
//   b_nl    b on the same padded nodes, same spline
//   P1h_nl  ln P1h on (a node, exact ln k node); natural cubic spline in
//           ln k (pads beyond each end), linear between a nodes
//
// Cache keys (every table): cosmology.random, Ntable.random,
// cluster.random_model, cluster.random_zdist, cluster.random_mor, and
// cluster.random_selection when the Y1 selection bias is on.
//
// Threading: every table value is one serial sum over its quadrature
// nodes; loops are threaded across table nodes only (collapse over
// (richness bin, a node) or (a node, k node)), so no table depends on the
// thread count. Tables are built lazily on the first read after a key
// changed: that first read must run outside any parallel region, which is
// what cluster_warmup is for.
// ---------------------------------------------------------------------------



// ============================================================================
// [SECTION] CONSTANTS
// ============================================================================

// halo.c's collapse threshold and halo overdensity (its delta_c and Delta
// macros are private to halo.c). The private copies must hold the same
// values: nu = delta_c/(sigma D) feeds halo.c's fnu, hb1nu and conc, and
// Delta fixes r_Delta = (3M/(4 pi Delta rho_m))^(1/3) of the NFW profile
// (checked against u_nfw_c at every P1h refill).
static const double CLUSTER_DELTA_C    = 1.686;  // linear collapse threshold
static const double CLUSTER_DELTA_HALO = 200.0;  // M200m: 200 x mean density

// Highest scale factor of the tables: halo.c's fnu requires a < 1 (the
// bias_norm convention of halo.c).
static const double CLUSTER_A_TOP = 0.9999999;

// Floor on the richness-bin density n_nl in (c/H0)^-3: 1e-10 clusters per
// Hubble volume (c/H0)^3 = 2.7e10 (Mpc/h)^3, i.e. no cluster in any
// survey. The ratios b_nl and P1h_nl divide by max(n_nl, floor), so a bin
// that empties (an extreme MOR draw, or an erf tail beyond double
// precision) gives b_nl, P1h_nl -> 0 smoothly instead of 0/0; n_nl itself
// reads back as the floor there.
static const double CLUSTER_N_FLOOR = 1.0e-10;

// A mass node whose 1-halo weight is below this fraction of the weight
// sum of its row, in every richness bin, is left out of the P1h kernel
// sums (the light halos far below the richness cut). u <= 1 for every
// node and P1h stays above ~1e-6 of its k -> 0 value over the whole
// table, so a skipped node's share of any tabulated value is below 1e-13.
static const double CLUSTER_1H_SKIP = 1.0e-20;

// P1h ln k sampling: the p_gm coarse step (Ntable.halo_nk_step ladder) on
// the p_gm dense ln k grid, made CLUSTER_K_REFINE times denser. A richness
// bin selects a narrow mass window, so the mass integral does not wash out
// the ringing of the truncated NFW transform in k the way the broad HOD
// window of p_gm does: ln P1h keeps wiggles of period ~2 pi/(k r_Delta) in
// ln k at k ~ 1-30 h/Mpc, which the p_gm step does not resolve.
static const int CLUSTER_K_REFINE = 4;

// NFW f, G table of the private kernel: the halo.c values (NFW_TMIN,
// NFW_TASY), which the check against u_nfw_c requires.
static const double CLUSTER_NFW_TMIN = 1e-10; // reads clamp below
static const double CLUSTER_NFW_TASY = 50.0;  // asymptotic series above



// ============================================================================
// [SECTION] MASS-OBSERVABLE RELATION
// ============================================================================
//
// P(nl|M, z) = int_{lambda_min}^{lambda_max} dlambda
//              LogNormal(lambda; <ln lambda|M, z>, sigma_lnlambda)
//            = [erf(x_max) - erf(x_min)]/2,
//   x = (ln lambda_edge - <ln lambda>)/(sqrt(2) sigma_lnlambda)
//
// sigma does not depend on the observed lambda (only on the mean), so the
// integral over lambda is exact in closed form: no table, no quadrature.
// <ln lambda> splits into a mass part and a redshift part (only mor[3]
// carries z), which the fill hoists to the mass-node and a-node levels.


// the mass part of <ln lambda|M, z>: mor[0] + mor[1] ln(M/M_piv)
static inline double mor_mean_mass_part(
    const double lnM  // ln(M/(Msun/h))
  )
{
  return cluster.mor[0] + cluster.mor[1]*(lnM - log(cluster.mor_pivot_mass));
}


// the redshift part of <ln lambda|M, z>: mor[3] ln((1 + z)/(1 + z_piv))
static inline double mor_mean_redshift_part(
    const double z  // redshift
  )
{
  return cluster.mor[3]*log((1.0 + z)/cluster.mor_pivot_1pz);
}


// 1/(sqrt(2) sigma_lnlambda) of eq (18) at mean mu = <ln lambda>: the
// factor that turns ln lambda - mu into an erf argument. The Poisson term
// is written e^-mu (1 - e^-mu), the same (e^mu - 1)/e^(2 mu) without
// overflow at large mu or cancellation at small mu.
static inline double mor_inv_sqrt2_sigma(
    const double mu  // <ln lambda|M, z>
  )
{
  const double sigma_int = cluster.mor[2];

  double variance = sigma_int*sigma_int;
  if (mu > 0.0) {
    const double e_minus_mu = exp(-mu);
    variance += e_minus_mu*(-expm1(-mu)); // (e^mu - 1)/e^(2 mu)
  }

  return 1.0/(M_SQRT2*sqrt(variance));
}


// [erf(x_max) - erf(x_min)]/2, x_min <= x_max, without cancellation: when
// both edges sit on the same side of the mean the difference is taken
// between the two complementary tails (erfc), which keeps full relative
// precision down to probabilities of 1e-300 (the far tail of a richness
// bin); when the mean lies inside the bin both erf terms have opposite
// signs and add.
static inline double richness_bin_probability(
    const double x_min,  // (ln lambda_min - mu)/(sqrt(2) sigma)
    const double x_max   // (ln lambda_max - mu)/(sqrt(2) sigma)
  )
{
  if (x_min >= 0.0) {
    // bin above the mean: difference of upper tails
    return 0.5*(erfc(x_min) - erfc(x_max));
  }
  else if (x_max <= 0.0) {
    // bin below the mean: difference of lower tails
    return 0.5*(erfc(-x_max) - erfc(-x_min));
  }
  else {
    // mean inside the bin: erf(x_max) > 0 > erf(x_min)
    return 0.5*(erf(x_max) - erf(x_min));
  }
}


// Aborts unless the MOR model and its parameters make sigma > 0 and every
// richness bin a valid interval.
static void mor_check(void)
{
  if (cluster.mor_model != CLUSTER_MOR_LOGNORMAL) {
    log_fatal("cluster.mor_model = %d not supported", cluster.mor_model);
    exit(1);
  }
  if (!(cluster.mor[2] > 0.0)) {
    log_fatal("MOR scatter sigma_int = cluster.mor[2] = %g must be > 0",
              cluster.mor[2]);
    exit(1);
  }
  if (!(cluster.mor_pivot_mass > 0.0) || !(cluster.mor_pivot_1pz > 0.0)) {
    log_fatal("MOR pivots M_piv = %g, 1 + z_piv = %g must be > 0",
              cluster.mor_pivot_mass, cluster.mor_pivot_1pz);
    exit(1);
  }
  for (int nl=0; nl<cluster.richness_nbin; nl++) {
    if (!(cluster.richness_min[nl] > 0.0) ||
        !(cluster.richness_max[nl] > cluster.richness_min[nl])) {
      log_fatal("richness bin %d = [%g, %g) is not a valid interval", nl,
                cluster.richness_min[nl], cluster.richness_max[nl]);
      exit(1);
    }
  }
}


// ---------------------------------------------------------------------------
// P(nl|M, z) of the section header. The fill evaluates the same helpers
// in the same order, so this function and the tables agree bitwise at the
// table's own (ln M, z) nodes.
//
// Parameters:
//   lnM - ln of the halo mass in Msun/h (M200m)
//   z   - redshift
//   nl  - richness bin, 0 <= nl < cluster.richness_nbin (aborts otherwise)
//
// Returns:
//   the probability, in [0, 1]
// ---------------------------------------------------------------------------
double prob_richness_bin_given_m(
    const double lnM,
    const double z,
    const int nl
  )
{
  if (nl < 0 || nl > cluster.richness_nbin - 1) {
    log_fatal("error in selecting richness bin nl = %d", nl);
    exit(1);
  }
  mor_check();

  // <ln lambda|M, z> and the erf scale 1/(sqrt 2 sigma) at that mean
  const double mu = mor_mean_mass_part(lnM) + mor_mean_redshift_part(z);
  const double inv_sqrt2_sigma = mor_inv_sqrt2_sigma(mu);

  // erf arguments at the two edges of the bin
  const double x_min = (log(cluster.richness_min[nl]) - mu)*inv_sqrt2_sigma;
  const double x_max = (log(cluster.richness_max[nl]) - mu)*inv_sqrt2_sigma;

  return richness_bin_probability(x_min, x_max);
}



// ============================================================================
// [SECTION] NFW KERNEL (private copy of halo.c's nfw_um)
// ============================================================================
//
// u(k|M) of the NFW profile truncated at r_Delta (astro-ph/0206508
// Eq. 81), written with the smooth auxiliary functions f, g of Abramowitz
// & Stegun 5.2.6-5.2.7 (the full derivation: halo.c, u_nfw_c header):
//
//   u m(c) = [g(x) - g(xu)] + 2 g(xu) sin^2(c x/2) + [f(xu) - 1/xu] sin(c x)
//   x = k r_s,  xu = (1 + c) x,  m(c) = ln(1 + c) - c/(1 + c)
//
// halo.c keeps its kernel (nfw_um) and its f, G table static, and its
// exported u_nfw_c recomputes r_Delta, ln(1 + c) and ln x on every call.
// The P1h sums call the kernel once per (a node, k node, mass node), so
// this file holds a copy of the table and of the scalar kernel with the
// per-node factors hoisted out; cluster_nfw_check verifies the copy
// against u_nfw_c at every P1h refill.
//
//   tab[0][i] = f(t_i),  tab[1][i] = G(t_i) = g(t_i) + ln t_i,
//   ln t_i uniform on [ln CLUSTER_NFW_TMIN, ln CLUSTER_NFW_TASY],
//   Ntable.halo_nfw_n nodes, read linearly in ln t; the asymptotic series
//   above CLUSTER_NFW_TASY.

static struct {
  uint64_t cache;      // Ntable.random of the table
  int n_nodes;         // number of ln t nodes
  double lim[3];       // ln t axis: first, last, spacing
  double inv_spacing;  // 1/spacing
  double** tab;        // [2][n_nodes] f(t), G(t) = g(t) + ln t
} cluster_nfw_ = {0};


// Builds cluster_nfw_ (first call, or Ntable.random changed); runs a
// threaded loop, so it must be called outside any parallel region.
static void cluster_nfw_table(void)
{
  if (NULL == cluster_nfw_.tab || fdiff2(cluster_nfw_.cache, Ntable.random)) {
    if (cluster_nfw_.tab != NULL) {
      free(cluster_nfw_.tab);
    }

    // --- 1. AXIS SETUP ---
    const int n_nodes = Ntable.halo_nfw_n;
    double** tab = (double**) malloc2d(2, n_nodes);
    double* lim  = cluster_nfw_.lim;
    lim[0] = log(CLUSTER_NFW_TMIN);
    lim[1] = log(CLUSTER_NFW_TASY);
    lim[2] = (lim[1] - lim[0])/((double) n_nodes - 1.0);

    // --- 2. TABLE FILL ---
    /* PHYSICAL DERIVATION & LOGIC FLOW
       1. node: ln t_i = lim[0] + i lim[2],   t_i = exp(ln t_i)
       2. f(t) =  Ci(t) sin t + (pi/2 - Si(t)) cos t
       3. G(t) = -Ci(t) cos t + (pi/2 - Si(t)) sin t + ln t = g(t) + ln t */
    #pragma omp parallel for schedule(static)
    for (int i=0; i<n_nodes; i++) {
      const double lnt = lim[0] + i*lim[2];
      const double t   = exp(lnt);
      const double si  = gsl_sf_Si(t);
      const double ci  = gsl_sf_Ci(t);
      tab[0][i] = ci*sin(t) + (M_PI_2 - si)*cos(t);        // f(t)
      tab[1][i] = -ci*cos(t) + (M_PI_2 - si)*sin(t) + lnt; // G(t)
    }

    // --- 3. PUBLISH ---
    cluster_nfw_.n_nodes     = n_nodes;
    cluster_nfw_.inv_spacing = 1.0/lim[2];
    cluster_nfw_.tab         = tab;
    cluster_nfw_.cache       = Ntable.random;
  }
}


// Position of ln t on the table: node index i and the fraction frac of
// the interval [i, i + 1]; clamped below CLUSTER_NFW_TMIN and onto the
// last interval.
static inline int cluster_nfw_pos(
    const double lnt,  // ln t
    double* frac       // output: fraction of the interval [i, i + 1]
  )
{
  const double lnt_min = cluster_nfw_.lim[0];
  const double pos = (fmax(lnt, lnt_min) - lnt_min)*cluster_nfw_.inv_spacing;

  int i = (int) pos;
  if (i > cluster_nfw_.n_nodes - 2) {
    i = cluster_nfw_.n_nodes - 2;
  }

  *frac = pos - i;
  return i;
}


// u m(c) of the section header for one halo at one k: f(xu), G(x), G(xu)
// read from the table up to CLUSTER_NFW_TASY, the asymptotic series
// (A&S 5.2.34-35, nested; 2, 12, 30, 56 and 6, 20, 42, 72 are the ratios
// of consecutive coefficients) above it. The caller passes ln x and
// ln(1 + c) (the table axis is ln t) and divides by m(c).
static inline double cluster_nfw_um(
    const double c,    // concentration r_Delta/r_s
    const double x,    // k r_s
    const double lnx,  // ln x
    const double ln1c  // ln(1 + c)
  )
{
  const double* restrict tab_f = cluster_nfw_.tab[0];  // f(t)
  const double* restrict tab_G = cluster_nfw_.tab[1];  // G(t) = g(t) + ln t
  const double lnxu = lnx + ln1c;  // ln xu, xu = (1 + c) x
  const double xu   = (1.0 + c)*x;

  double fu, Gx, Gu;
  if (xu <= CLUSTER_NFW_TASY) {
    double frac_u, frac_x;
    const int iu = cluster_nfw_pos(lnxu, &frac_u);
    const int ix = cluster_nfw_pos(lnx, &frac_x);
    Gu = frac_u*(tab_G[iu + 1] - tab_G[iu]) + tab_G[iu];
    fu = frac_u*(tab_f[iu + 1] - tab_f[iu]) + tab_f[iu];
    Gx = frac_x*(tab_G[ix + 1] - tab_G[ix]) + tab_G[ix];
  }
  else {
    const double v = 1.0/(xu*xu); // series variable 1/xu^2
    fu = (1.0 - 2.0*v*(1.0 - 12.0*v*(1.0 - 30.0*v*(1.0 - 56.0*v))))/xu;
    Gu = v*(1.0 - 6.0*v*(1.0 - 20.0*v*(1.0 - 42.0*v*(1.0 - 72.0*v)))) + lnxu;
    if (x <= CLUSTER_NFW_TASY) {
      double frac_x;
      const int ix = cluster_nfw_pos(lnx, &frac_x);
      Gx = frac_x*(tab_G[ix + 1] - tab_G[ix]) + tab_G[ix];
    }
    else {
      const double w = 1.0/(x*x); // series variable 1/x^2
      Gx = w*(1.0 - 6.0*w*(1.0 - 20.0*w*(1.0 - 42.0*w*(1.0 - 72.0*w)))) + lnx;
    }
  }

  // u m(c) = [g(x) - g(xu)] + 2 g(xu) sin^2(c x/2) + [f(xu) - 1/xu] sin(c x),
  // with g(x) - g(xu) = Gx - Gu + ln(1 + c)
  const double gu = Gu - lnxu;           // g(xu)
  const double sin_half = sin(0.5*c*x);  // sin(c x/2)
  return (Gx - Gu + ln1c) + 2.0*gu*sin_half*sin_half + (fu - 1.0/xu)*sin(c*x);
}


// The private kernel against halo.c's u_nfw_c for a cluster-mass halo at
// three k (the linear table, its large-t end, the asymptotic series);
// aborts on a mismatch (halo.c's NFW machinery or its Delta changed).
// Needs the cosmology (rho_m) and both tables built: call it serially.
static void cluster_nfw_check(void)
{
  const double TEST_MASS = 1.0e14;  // Msun/h
  const double TEST_CONC = 4.0;
  const double TEST_K[3] = {1.0e2, 1.0e4, 1.0e6};  // (c/H0)^-1: k r_s ~
                                                   // 1e-2, 1, 1e2
  const double TOLERANCE = 1.0e-12;  // two roundings of x apart

  const double rho_delta = CLUSTER_DELTA_HALO*cosmology.rho_crit*
                           cosmology.Omega_m;
  const double r_delta = pow(3./(4.0*M_PI)*(TEST_MASS/rho_delta), 1./3.);
  const double r_s     = r_delta/TEST_CONC;
  const double ln1c    = log1p(TEST_CONC);
  const double m_c     = ln1c - TEST_CONC/(1.0 + TEST_CONC);

  for (int i=0; i<3; i++) {
    const double k = TEST_K[i];
    const double u_private =
        cluster_nfw_um(TEST_CONC, k*r_s, log(k) + log(r_s), ln1c)/m_c;
    const double u_halo = u_nfw_c(TEST_CONC, k, TEST_MASS, CLUSTER_A_TOP);

    if (!(fabs(u_private/u_halo - 1.0) < TOLERANCE)) {
      log_fatal("private NFW kernel u = %.17g differs from halo.c u_nfw_c "
                "= %.17g at k = %g: resync halo_cluster.c with halo.c",
                u_private, u_halo, k);
      exit(1);
    }
  }
}



// ============================================================================
// [SECTION] RICHNESS-BIN MASS INTEGRALS: n_nl(a), b_nl(a), 1-HALO WEIGHTS
// ============================================================================
//
// One fill (cluster_mass_tables) computes, at every node of the a grid and
// for every richness bin, the Gauss-Legendre sums in ln M on
// [ln cluster.m_min, ln cluster.m_max] (nodes ln M_q, weights w_q):
//
//   n_nl(a)    = sum_q dn_q(a) P_q
//   b_nl(a)    = sum_q dn_q(a) P_q b_q(a) / n_nl(a)
//   W_nl(a, q) = dn_q(a) P_q (M_q/rho_m) / (m(c_q) n_nl(a))
//
//   dn_q(a) = w_q (rho_m/M_q) nu f(nu) dln nu/dln M,  nu = nu0_q/D(a):
//             the quadrature weight times dn/dlnM (halo.c's product order)
//   P_q     = P(nl|M_q, z(a))
//   b_q(a)  = b_h(nu) (x the Y1 selection factor S)
//   m(c)    = ln(1 + c) - c/(1 + c), c = conc(M_q, D(a))
//
// W is the 1-halo mass weight: P1h_nl(k, a) = sum_q W_nl(a, q) u m(c)
// (NFW kernel section), with the 1/n_nl normalization and the 1/m(c) of
// u = u m(c)/m(c) folded in.
//
// Loop levels (each quantity at the outermost level it depends on):
//
//   per Ntable rebuild     x_q, w_q on [-1, 1]
//   per refill, per q      ln M_q, M_q, w_q (rho_m/M_q) dln nu/dln M,
//     (mass_node)          nu0_q = delta_c/sigma(M_q), the mass part of
//                          <ln lambda>, r_Delta(M_q), the mass part of S
//   per a row              a, z, D(a), the redshift parts of <ln lambda>
//     (a_row)              and of S
//   per (a row, q)         dn_q(a), b_q(a), <ln lambda>, 1/(sqrt 2 sigma),
//     (a_node)             c, ln(1 + c), r_s, ln r_s, dn_q (M_q/rho_m)/m(c)
//   per (nl, a row), sum   P_q (two erf), n_nl, b_nl, W_nl
//     over q
//
// The a grid: Ntable.halo_na_lens nodes on [a_lo, a_hi] (every cluster
// bin's support), extended by up to Ntable.halo_spline_pad exact nodes
// beyond each end ("pads": the natural spline's S'' = 0 end condition is
// wrong for the curved n(a), and its error decays by 2 - sqrt(3) per
// interval away from the end, so the pads keep it out of [a_lo, a_hi]).
// Pads are dropped where they would reach a <= 0 or a > CLUSTER_A_TOP
// (only a support touching z = 0 loses its upper pads).
//
// Reads: n_nl and b_nl through the house natural cubic spline in a (of
// ln max(n_nl, floor) and of b_nl): n varies by a large factor over a
// cluster bin (steeply for the rare, massive halos of the high-richness
// bins), so a linear read in a would miss the 1e-4 target, while the
// spline error scales as h^4 and sits far below it.

// rows of cl_.mass_node: per mass node q, set per refill
enum {
  MN_LNM,       // ln M_q
  MN_M,         // M_q, Msun/h
  MN_WEIGHT,    // half_width w_q (rho_m/M_q) dln nu/dln M
  MN_NU0,       // nu0_q = delta_c/sigma(M_q): the peak height at D = 1
  MN_MU_MASS,   // mor[0] + mor[1] ln(M_q/M_piv)
  MN_RDELTA,    // r_Delta(M_q) in c/H0
  MN_SEL_MASS,  // Y1: b_s0 (M_q/M_piv)^b_s1; 1 otherwise
  MN_NROWS
};

// rows of cl_.a_row: per a row r of the padded grid
enum {
  AR_A,         // scale factor a_r
  AR_Z,         // redshift 1/a_r - 1
  AR_GROWTH,    // D(a_r)
  AR_MU_Z,      // mor[3] ln((1 + z_r)/(1 + z_piv))
  AR_SEL_Z,     // Y1: ((1 + z_r)/(1 + z_piv))^b_s2; 1 otherwise
  AR_NROWS
};

// rows of cl_.a_node[r]: per (a row r, mass node q)
enum {
  AN_DN,        // dn_q(a) = w_q dn/dlnM (with the GL weight)
  AN_BIAS,      // b_h(nu) S
  AN_MU,        // <ln lambda|M_q, z_r>
  AN_INV_SIG,   // 1/(sqrt(2) sigma_lnlambda)
  AN_DN_1H,     // dn_q(a) (M_q/rho_m)/m(c)
  AN_CONC,      // c = conc(M_q, D)
  AN_LN1C,      // ln(1 + c)
  AN_RS,        // r_s = r_Delta/c
  AN_LNRS,      // ln r_s
  AN_NROWS
};

// Tables of the fill; zero at program start, so the first call builds.
static struct {
  uint64_t cache[MAX_SIZE_ARRAYS]; // [0] cosmology [1] Ntable [2] model
                                   // [3] zdist [4] mor [5] selection tags
  int nl_bins;         // richness bins of the allocation
  int n_mass;          // Gauss-Legendre nodes in ln M
  int n_a;             // a nodes on [a_lo, a_hi]
  int pad;             // pad nodes allocated beyond each end
  int pad_lo;          // pad nodes in use below a_lo
  int pad_hi;          // pad nodes in use above a_hi
  int n_rows;          // rows filled: pad_lo + n_a + pad_hi
  double a_lim[3];     // a_lo, a_hi, spacing h
  double a_first;      // a of row 0: a_lo - pad_lo h
  double lnlam[2][MAX_SIZE_ARRAYS]; // ln lambda_min (0), ln lambda_max (1)
  double** gl;         // [2][n_mass] GL nodes x_q (0), weights w_q (1)
                       // on [-1, 1]
  double** mass_node;  // [MN_NROWS][n_mass]
  double** a_row;      // [AR_NROWS][n_a + 2 pad]
  double*** a_node;    // [n_a + 2 pad][AN_NROWS][n_mass]
  double** ln_n;       // [nl][n_a + 2 pad] ln max(n_nl, floor) per row
  double** ln_n_curv;  // [nl][n_a + 2 pad] its spline coefficients
  double** bias;       // [nl][n_a + 2 pad] b_nl per row
  double** bias_curv;  // [nl][n_a + 2 pad] its spline coefficients
  double*** w1h;       // [nl][n_a][n_mass] W_nl(a_i, q) on the a nodes
                       // (row i = padded row pad_lo + i)
} cl_ = {0};


// 1 when a table stamped with cache[] must be refilled: any key differs
// (the selection tag only counts when the Y1 selection bias is on; a
// switch of the selection model changes cluster.random_model).
static int cluster_keys_differ(
    const uint64_t* cache  // [0..5] tags stamped by cluster_keys_stamp
  )
{
  int differ = fdiff2(cache[0], cosmology.random) ||
               fdiff2(cache[1], Ntable.random) ||
               fdiff2(cache[2], cluster.random_model) ||
               fdiff2(cache[3], cluster.random_zdist) ||
               fdiff2(cache[4], cluster.random_mor);

  if (CLUSTER_SELECTION_Y1 == cluster.selection_model) {
    differ = differ || fdiff2(cache[5], cluster.random_selection);
  }

  return differ;
}


static void cluster_keys_stamp(
    uint64_t* cache  // output: the current tags
  )
{
  cache[0] = cosmology.random;
  cache[1] = Ntable.random;
  cache[2] = cluster.random_model;
  cache[3] = cluster.random_zdist;
  cache[4] = cluster.random_mor;
  cache[5] = cluster.random_selection;
}


// Gauss-Legendre node count of the mass integrals: the ladder of halo.c's
// spectra (p_gm), Ntable.halo_nm at high_def_integration 0, doubling per
// step, the largest GSL rule from 3 on; snapped up to the nearest size GSL
// has precomputed (a non-tabulated size is computed on the fly with
// weights good to only ~5e-7).
static int cluster_mass_node_count(void)
{
  // the Gauss-Legendre sizes GSL stores as precomputed tables
  static const int GL_TABULATED[] = {64, 96, 128, 256, 512, 1024};
  const int n_tabulated = (int) (sizeof(GL_TABULATED)/sizeof(GL_TABULATED[0]));

  const int accuracy = abs(Ntable.high_def_integration);

  int n_requested;
  if (0 == accuracy) {
    n_requested = Ntable.halo_nm;
  }
  else if (1 == accuracy) {
    n_requested = 2*Ntable.halo_nm;
  }
  else if (2 == accuracy) {
    n_requested = 4*Ntable.halo_nm;
  }
  else {
    n_requested = 1024;
  }

  for (int t=0; t<n_tabulated; t++) {
    if (GL_TABULATED[t] >= n_requested) {
      return GL_TABULATED[t];
    }
  }
  return GL_TABULATED[n_tabulated - 1];
}


// House natural-spline read on a uniform grid, interval j, offset t from
// node j (y the values, curv = S''/2 from spline_coeffs_uniform, h the
// spacing):
//
//   S(x_j + t) = y_j + t (b + t (c_j + t d)),
//   b = (y_{j+1} - y_j)/h - h (c_{j+1} + 2 c_j)/3,  d = (c_{j+1} - c_j)/(3 h)
static inline double spline_horner(
    const double* y,     // node values
    const double* curv,  // spline coefficients c_j = S''(x_j)/2
    const int j,         // interval [x_j, x_{j+1}]
    const double t,      // offset from x_j, 0 <= t <= h
    const double h       // grid spacing
  )
{
  const double b = (y[j+1] - y[j])/h - h*(curv[j+1] + 2.0*curv[j])/3.0;
  const double d = (curv[j+1] - curv[j])/(3.0*h);
  return y[j] + t*(b + t*(curv[j] + t*d));
}


// ---------------------------------------------------------------------------
// Fills cl_ (section header). Called by every reader; rebuilds and
// refills only when a key changed, and then must run outside any parallel
// region (it owns threaded loops and reads halo.c tables that build
// lazily): cluster_warmup's first call does that.
//
// Cache invalidation:
//   rebuild (sizes, allocations, GL rule): Ntable.random,
//     cluster.random_model (richness bin count)
//   refill: the keys of cluster_keys_differ. The inputs each key stands
//     for (the setters' contract, structs_cluster.h): random_model the
//     richness edges, mass range, MOR and selection models and pivots;
//     random_zdist the supports zdist_zmin/zmax (the a grid);
//     random_mor mor[]; random_selection selection[] (Y1 only)
// ---------------------------------------------------------------------------
static void cluster_mass_tables(void)
{
  int rebuilt = 0;

  // --- 1. REBUILD: SIZES, ALLOCATIONS, GAUSS-LEGENDRE RULE ---
  if (NULL == cl_.ln_n ||
      fdiff2(cl_.cache[1], Ntable.random) ||
      fdiff2(cl_.cache[2], cluster.random_model))
  {
    if (cl_.ln_n != NULL) {
      free(cl_.gl);
      free(cl_.mass_node);
      free(cl_.a_row);
      free(cl_.a_node);
      free(cl_.ln_n);
      free(cl_.ln_n_curv);
      free(cl_.bias);
      free(cl_.bias_curv);
      free(cl_.w1h);
    }

    if (cluster.richness_nbin < 1 || cluster.richness_nbin > MAX_SIZE_ARRAYS) {
      log_fatal("cluster.richness_nbin = %d not in [1, %d]",
                cluster.richness_nbin, MAX_SIZE_ARRAYS);
      exit(1);
    }
    if (Ntable.halo_na_lens < 2) {
      log_fatal("Ntable.halo_na_lens = %d: the a grid needs >= 2 nodes",
                Ntable.halo_na_lens);
      exit(1);
    }

    cl_.nl_bins = cluster.richness_nbin;
    cl_.n_mass  = cluster_mass_node_count();
    cl_.n_a     = Ntable.halo_na_lens;
    cl_.pad     = Ntable.halo_spline_pad;

    const int n_rows_max = cl_.n_a + 2*cl_.pad;

    cl_.gl        = (double**)  malloc2d(2, cl_.n_mass);
    cl_.mass_node = (double**)  malloc2d(MN_NROWS, cl_.n_mass);
    cl_.a_row     = (double**)  malloc2d(AR_NROWS, n_rows_max);
    cl_.a_node    = (double***) malloc3d(n_rows_max, AN_NROWS, cl_.n_mass);
    cl_.ln_n      = (double**)  malloc2d(cl_.nl_bins, n_rows_max);
    cl_.ln_n_curv = (double**)  malloc2d(cl_.nl_bins, n_rows_max);
    cl_.bias      = (double**)  malloc2d(cl_.nl_bins, n_rows_max);
    cl_.bias_curv = (double**)  malloc2d(cl_.nl_bins, n_rows_max);
    cl_.w1h       = (double***) malloc3d(cl_.nl_bins, cl_.n_a, cl_.n_mass);

    // x_q, w_q on [-1, 1]; the refill maps them onto [ln m_min, ln m_max]
    gsl_integration_glfixed_table* gauss_table =
        malloc_gslint_glfixed(cl_.n_mass);
    for (int q=0; q<cl_.n_mass; q++) {
      gsl_integration_glfixed_point(-1.0, 1.0, q, &cl_.gl[0][q],
                                    &cl_.gl[1][q], gauss_table);
    }
    gsl_integration_glfixed_table_free(gauss_table);

    rebuilt = 1;
  }

  // --- 2. REFILL ---
  if (1 == rebuilt || cluster_keys_differ(cl_.cache))
  {
    const int nl_bins = cl_.nl_bins;
    const int n_mass  = cl_.n_mass;
    const int n_a     = cl_.n_a;

    const int selection_y1 = (CLUSTER_SELECTION_Y1 == cluster.selection_model);

    // --- 2a. INPUT CHECKS ---
    if (cluster.zdist_nbin < 1) {
      log_fatal("cluster redshift bins not set (cluster.zdist_nbin = %d)",
                cluster.zdist_nbin);
      exit(1);
    }
    // inside the ln M range of halo.c's sigma2 and dlognudlogm tables,
    // which clamp outside it
    if (!(cluster.m_min >= limits.halo_m_min) ||
        !(cluster.m_max <= limits.halo_m_max) ||
        !(cluster.m_max > cluster.m_min)) {
      log_fatal("cluster mass range [%g, %g] not inside the halo.c tables' "
                "[%g, %g]", cluster.m_min, cluster.m_max,
                limits.halo_m_min, limits.halo_m_max);
      exit(1);
    }
    mor_check();

    for (int nl=0; nl<nl_bins; nl++) {
      cl_.lnlam[0][nl] = log(cluster.richness_min[nl]);
      cl_.lnlam[1][nl] = log(cluster.richness_max[nl]);
    }

    // --- 2b. THE a GRID: EVERY CLUSTER BIN'S SUPPORT ---
    // [a_lo, a_hi] = [1/(1 + max_i zmax_i), 1/(1 + min_i zmin_i)], a_hi
    // capped below 1 (fnu); pads dropped where they would leave (0, A_TOP]
    double zmin_support = cluster.zdist_zmin[0];
    double zmax_support = cluster.zdist_zmax[0];
    for (int i=1; i<cluster.zdist_nbin; i++) {
      zmin_support = fmin(zmin_support, cluster.zdist_zmin[i]);
      zmax_support = fmax(zmax_support, cluster.zdist_zmax[i]);
    }

    const double a_lo = 1.0/(1.0 + zmax_support);
    const double a_hi = fmin(1.0/(1.0 + zmin_support), CLUSTER_A_TOP);
    if (!(a_hi > a_lo)) {
      log_fatal("cluster support z in [%g, %g] is empty",
                zmin_support, zmax_support);
      exit(1);
    }
    const double h = (a_hi - a_lo)/((double) n_a - 1.0);

    int pad_lo = cl_.pad;
    while (pad_lo > 0 && !(a_lo - pad_lo*h > 0.0)) {
      pad_lo--;
    }
    int pad_hi = cl_.pad;
    while (pad_hi > 0 && a_hi + pad_hi*h > CLUSTER_A_TOP) {
      pad_hi--;
    }

    cl_.a_lim[0] = a_lo;
    cl_.a_lim[1] = a_hi;
    cl_.a_lim[2] = h;
    cl_.pad_lo   = pad_lo;
    cl_.pad_hi   = pad_hi;
    cl_.n_rows   = pad_lo + n_a + pad_hi;
    cl_.a_first  = a_lo - pad_lo*h;

    const int n_rows = cl_.n_rows;

    // --- 2c. WARM-UP: THE LAZY TABLES OF halo.c AND cosmo3D.c READ BELOW ---
    // sigma2 and dlognudlogm (ln M tables; conc reads sigma2 too) and
    // tinker_alpha (inside fnu) build here, before the threads start
    (void) sigma2(cluster.m_min);
    (void) dlognudlogm(cluster.m_min);
    (void) fnu(1.0, cl_.a_first);

    /* PHYSICAL DERIVATION & LOGIC FLOW (the section header)
       1. per mass node q: M_q, the a-free factor of dn/dlnM, nu0_q, the
          mass parts of <ln lambda> and S, r_Delta
       2. per a row: a, z, D(a), the redshift parts of <ln lambda> and S
       3. per (a row, q): nu = nu0_q/D, dn_q = w (rho_m/M) dlnnu/dlnM
          f(nu) nu, b_q = b_h(nu) S, <ln lambda>, sigma, c(M, D), r_s
       4. per (nl, a row): P_q = [erf(x_max) - erf(x_min)]/2,
          n = sum dn_q P_q, b = sum dn_q P_q b_q / n,
          W_q = dn_q P_q (M_q/rho_m)/(m(c) n)
       5. natural splines of ln n and b across the padded a rows */

    // --- 2d. PER MASS NODE (serial: sigma2, dlognudlogm reads) ---
    const double rho_m     = cosmology.rho_crit*cosmology.Omega_m;
    const double rho_delta = CLUSTER_DELTA_HALO*rho_m;

    const double lnM_min    = log(cluster.m_min);
    const double lnM_max    = log(cluster.m_max);
    const double half_width = 0.5*(lnM_max - lnM_min);
    const double mid        = 0.5*(lnM_max + lnM_min);

    for (int q=0; q<n_mass; q++) {
      const double lnM = mid + half_width*cl_.gl[0][q];
      const double m   = exp(lnM);

      cl_.mass_node[MN_LNM][q]     = lnM;
      cl_.mass_node[MN_M][q]       = m;
      cl_.mass_node[MN_WEIGHT][q]  = half_width*cl_.gl[1][q]
                                     *(rho_m/m)*dlognudlogm(m);
      cl_.mass_node[MN_NU0][q]     = CLUSTER_DELTA_C/sqrt(sigma2(m));
      cl_.mass_node[MN_MU_MASS][q] = mor_mean_mass_part(lnM);
      cl_.mass_node[MN_RDELTA][q]  = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);

      // Y1 selection, mass part: b_s0 (M/M_piv)^b_s1
      double selection_mass = 1.0;
      if (selection_y1) {
        selection_mass = cluster.selection[0]*
                         pow(m/cluster.mor_pivot_mass, cluster.selection[1]);
      }
      cl_.mass_node[MN_SEL_MASS][q] = selection_mass;
    }

    // --- 2e. PER a ROW (serial: growfac) ---
    for (int r=0; r<n_rows; r++) {
      const double a = cl_.a_first + r*h;
      const double z = 1.0/a - 1.0;

      cl_.a_row[AR_A][r]      = a;
      cl_.a_row[AR_Z][r]      = z;
      cl_.a_row[AR_GROWTH][r] = growfac(a);
      cl_.a_row[AR_MU_Z][r]   = mor_mean_redshift_part(z);

      // Y1 selection, redshift part: ((1 + z)/(1 + z_piv))^b_s2
      double selection_z = 1.0;
      if (selection_y1) {
        selection_z = pow((1.0 + z)/cluster.mor_pivot_1pz,
                          cluster.selection[2]);
      }
      cl_.a_row[AR_SEL_Z][r] = selection_z;
    }

    // --- 2f. PER (a ROW, MASS NODE): THREADED OVER THE a ROWS ---
    #pragma omp parallel for schedule(static)
    for (int r=0; r<n_rows; r++) {
      const double a           = cl_.a_row[AR_A][r];
      const double D           = cl_.a_row[AR_GROWTH][r];
      const double mu_z        = cl_.a_row[AR_MU_Z][r];
      const double selection_z = cl_.a_row[AR_SEL_Z][r];

      // per-node rows of this a row (restrict: distinct rows)
      double* restrict dn_q      = cl_.a_node[r][AN_DN];
      double* restrict bias_q    = cl_.a_node[r][AN_BIAS];
      double* restrict mu_q      = cl_.a_node[r][AN_MU];
      double* restrict inv_sig_q = cl_.a_node[r][AN_INV_SIG];
      double* restrict dn_1h_q   = cl_.a_node[r][AN_DN_1H];
      double* restrict conc_q    = cl_.a_node[r][AN_CONC];
      double* restrict ln1c_q    = cl_.a_node[r][AN_LN1C];
      double* restrict rs_q      = cl_.a_node[r][AN_RS];
      double* restrict lnrs_q    = cl_.a_node[r][AN_LNRS];

      for (int q=0; q<n_mass; q++) {
        const double m  = cl_.mass_node[MN_M][q];
        const double nu = cl_.mass_node[MN_NU0][q]/D;  // nu0_q/D(a)

        // quadrature weight x dn/dlnM, in halo.c's order: the a-free
        // factor, then f(nu), then nu
        const double dn = cl_.mass_node[MN_WEIGHT][q]*fnu(nu, a)*nu;

        // halo bias, times the Y1 selection factor S (1 otherwise)
        const double bias = hb1nu(nu, a)*cl_.mass_node[MN_SEL_MASS][q]
                            *selection_z;

        // <ln lambda|M, z> and 1/(sqrt 2 sigma) at that mean
        const double mu = cl_.mass_node[MN_MU_MASS][q] + mu_z;

        // NFW profile: c, ln(1 + c), m(c) = ln(1 + c) - c/(1 + c), r_s
        const double c    = conc(m, D);
        const double ln1c = log1p(c);
        const double m_c  = ln1c - c/(1.0 + c);
        const double r_s  = cl_.mass_node[MN_RDELTA][q]/c;

        dn_q[q]      = dn;
        bias_q[q]    = bias;
        mu_q[q]      = mu;
        inv_sig_q[q] = mor_inv_sqrt2_sigma(mu);
        dn_1h_q[q]   = dn*(m/rho_m)/m_c;
        conc_q[q]    = c;
        ln1c_q[q]    = ln1c;
        rs_q[q]      = r_s;
        lnrs_q[q]    = log(r_s);
      }
    }

    // --- 2g. THE FILL: (RICHNESS BIN, a ROW) PAIRS COLLAPSED AND THREADED ---
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nl=0; nl<nl_bins; nl++) {
      for (int r=0; r<n_rows; r++) {
        const double lnlam_min = cl_.lnlam[0][nl];
        const double lnlam_max = cl_.lnlam[1][nl];

        const double* restrict dn_q      = cl_.a_node[r][AN_DN];
        const double* restrict bias_q    = cl_.a_node[r][AN_BIAS];
        const double* restrict mu_q      = cl_.a_node[r][AN_MU];
        const double* restrict inv_sig_q = cl_.a_node[r][AN_INV_SIG];
        const double* restrict dn_1h_q   = cl_.a_node[r][AN_DN_1H];

        // the 1-halo weights are kept on the a nodes only (not the pads)
        const int i_node   = r - cl_.pad_lo;
        const int has_w1h  = (i_node >= 0 && i_node < n_a);
        double* restrict w = NULL;
        if (has_w1h) {
          w = cl_.w1h[nl][i_node];
        }

        double n_sum = 0.0;  // sum_q dn_q P_q        -> n_nl(a)
        double b_sum = 0.0;  // sum_q dn_q P_q b_q    -> b_nl(a) n_nl(a)

        for (int q=0; q<n_mass; q++) {
          // P(nl|M_q, z): the erf arguments at the two richness edges
          const double x_min = (lnlam_min - mu_q[q])*inv_sig_q[q];
          const double x_max = (lnlam_max - mu_q[q])*inv_sig_q[q];
          const double prob  = richness_bin_probability(x_min, x_max);

          // node q's share of n_nl(a)
          const double dn_prob = dn_q[q]*prob;
          n_sum += dn_prob;
          b_sum += dn_prob*bias_q[q];

          if (has_w1h) {
            w[q] = dn_1h_q[q]*prob;
          }
        }

        // the density floor (CONSTANTS) guards the two ratios
        const double n_norm = fmax(n_sum, CLUSTER_N_FLOOR);

        cl_.ln_n[nl][r] = log(n_norm);
        cl_.bias[nl][r] = b_sum/n_norm;

        if (has_w1h) {
          const double inv_n = 1.0/n_norm;
          for (int q=0; q<n_mass; q++) {
            w[q] *= inv_n;
          }
        }
      }
    }

    // --- 2h. NATURAL SPLINES IN a ACROSS THE PADDED ROWS ---
    for (int nl=0; nl<nl_bins; nl++) {
      spline_coeffs_uniform(cl_.ln_n[nl], n_rows, h, cl_.ln_n_curv[nl]);
      spline_coeffs_uniform(cl_.bias[nl], n_rows, h, cl_.bias_curv[nl]);
    }

    // --- 2i. CACHE TAGS: THE INPUTS THE TABLES NOW HOLD ---
    cluster_keys_stamp(cl_.cache);
  }
}


// Spline read on the padded a grid of cl_ (a inside [a_lo, a_hi]).
static inline double cluster_a_spline(
    const double* y,     // [n_rows] node values
    const double* curv,  // [n_rows] spline coefficients
    const double a       // scale factor
  )
{
  const double h = cl_.a_lim[2];
  const double r = (a - cl_.a_first)/h;

  // interval index, clamped onto the grid (a rounding guard at a_hi)
  int j = (int) r;
  if (j > cl_.n_rows - 2) {
    j = cl_.n_rows - 2;
  }
  if (j < 0) {
    j = 0;
  }

  return spline_horner(y, curv, j, (r - j)*h, h);
}


// ---------------------------------------------------------------------------
// n_nl(a), eq (16) inner integrals (cluster_mass_tables header): the
// comoving number density of clusters in richness bin nl.
//
// Parameters:
//   a  - scale factor
//   nl - richness bin, 0 <= nl < cluster.richness_nbin (aborts otherwise)
//
// Returns:
//   n_nl(a) in (c/H0)^-3; 0 outside the a grid; the density floor where
//   the bin is empty
// ---------------------------------------------------------------------------
double ncl_richness(
    const double a,
    const int nl
  )
{
  if (nl < 0 || nl > cluster.richness_nbin - 1) {
    log_fatal("error in selecting richness bin nl = %d", nl);
    exit(1);
  }

  cluster_mass_tables();

  if (a < cl_.a_lim[0] || a > cl_.a_lim[1]) {
    return 0.0;
  }

  return exp(cluster_a_spline(cl_.ln_n[nl], cl_.ln_n_curv[nl], a));
}


// ---------------------------------------------------------------------------
// b_nl(a), eq (21) (Y1 eq 16 with the Y1 selection bias): the linear bias
// of clusters in richness bin nl (cluster_mass_tables header).
//
// Parameters:
//   a  - scale factor
//   nl - richness bin, 0 <= nl < cluster.richness_nbin (aborts otherwise)
//
// Returns:
//   b_nl(a), dimensionless; 0 outside the a grid
// ---------------------------------------------------------------------------
double bcl_richness(
    const double a,
    const int nl
  )
{
  if (nl < 0 || nl > cluster.richness_nbin - 1) {
    log_fatal("error in selecting richness bin nl = %d", nl);
    exit(1);
  }

  cluster_mass_tables();

  if (a < cl_.a_lim[0] || a > cl_.a_lim[1]) {
    return 0.0;
  }

  return cluster_a_spline(cl_.bias[nl], cl_.bias_curv[nl], a);
}



// ============================================================================
// [SECTION] ONE-HALO CLUSTER-MATTER POWER SPECTRUM
// ============================================================================
//
//   P1h_nl(k, a) = int dlnM (dn/dlnM) P(nl|M) (M/rho_m) u(k|M, a) / n_nl(a)
//               = sum_q W_nl(a, q) u m(c)(k r_s,q)        (eq 22)
//
// with the 1-halo weights W of the fill (they carry dn/dlnM, P, M/rho_m,
// 1/m(c) and 1/n_nl) and the NFW kernel u m(c) of the NFW section. No bias,
// no selection factor: the 1-halo term is the halo's own mass profile.
// At k -> 0, u -> 1 and P1h -> <M>/rho_m of the bin.
//
// Table: ln P1h_nl on (a node i, ln k node c), exact at every node:
//
//   a nodes     the n_a nodes of the fill (no pads)
//   ln k nodes  uniform, spacing = the p_gm coarse step on the p_gm dense
//               grid (Ntable.N_k_nlin nodes on [ln limits.k_min_cH0,
//               ln limits.k_max_cH0]; step Ntable.halo_nk_step, halved at
//               high_def_integration 1, 1 from 2 on) divided by
//               CLUSTER_K_REFINE (CONSTANTS: why), plus
//               Ntable.halo_spline_pad pads beyond each end
//
// Reads: natural cubic spline in ln k on each of the two bracketing a
// rows, linear in a between them, then exp. Below ln k_min the value at
// k_min (P1h is flat there: u = 1 - O((k r_Delta)^2)); above the last
// ln k node, ln P1h continued linearly in ln k with the slope of the last
// interval (P1h falls as a power law once every halo is resolved).
//
// Loop levels:
//
//   per refill, per a node i      the active mass nodes (weight above
//     (pass A)                    CLUSTER_1H_SKIP of the row in some bin)
//                                 and their c, ln(1 + c), r_s, ln r_s,
//                                 W_nl(a_i, q) compacted
//   per (i, k node), threaded     um = u m(c) per active node, shared by
//     (pass B)                    every richness bin: sum_q W_nl um
//   per (nl, i), threaded         the spline coefficients in ln k
//     (pass C)
//
// Cache invalidation: as the fill (cluster_keys_differ); rebuild on
// Ntable.random or cluster.random_model.

// rows of p1h_.node[i]: per (a node i, active mass node j)
enum {
  PN_CONC,   // c
  PN_LN1C,   // ln(1 + c)
  PN_RS,     // r_s
  PN_LNRS,   // ln r_s
  PN_NROWS
};

static struct {
  uint64_t cache[MAX_SIZE_ARRAYS]; // tags, as cl_.cache
  int nl_bins;         // richness bins of the allocation
  int n_mass;          // mass nodes of the fill
  int n_a;             // a nodes of the fill
  int n_k;             // ln k nodes, pads included
  int k_last;          // index of the last node inside the range (the
                       // pads start after it)
  double lnk_first;    // ln k of node 0
  double dlnk;         // ln k spacing
  double lnk_min;      // ln limits.k_min_cH0: reads clamp below
  double lnk_last;     // ln k of node k_last: reads extrapolate above
  double*** ln_p;      // [nl][n_a][n_k] ln P1h
  double*** curv;      // [nl][n_a][n_k] spline coefficients in ln k
  double*** node;      // [n_a][PN_NROWS][n_mass] profile of active nodes
  double*** weight;    // [n_a][n_mass][nl] W of active nodes, bins last
  int* n_active;       // [n_a] active mass nodes per a node
  int k_warned;        // 1 once the coverage warning was printed
} p1h_ = {0};


// ln k spacing of the P1h nodes (section header).
static double cluster_p1h_lnk_spacing(void)
{
  const double dlnk_dense = (log(limits.k_max_cH0) - log(limits.k_min_cH0))
                            /((double) Ntable.N_k_nlin - 1.0);

  const int accuracy = abs(Ntable.high_def_integration);

  // p_gm's coarse step on its dense grid
  int k_step;
  if (0 == accuracy) {
    k_step = Ntable.halo_nk_step;
  }
  else if (1 == accuracy) {
    k_step = Ntable.halo_nk_step/2;
  }
  else {
    k_step = 1;
  }
  if (k_step < 1) {
    k_step = 1;
  }

  return k_step*dlnk_dense/CLUSTER_K_REFINE;
}


// ---------------------------------------------------------------------------
// Fills p1h_ (section header), refilling the fill first. Same thread
// rule as cluster_mass_tables: the first call after a key changed runs
// outside any parallel region (cluster_warmup).
// ---------------------------------------------------------------------------
static void cluster_p1h_table(void)
{
  // the 1-halo weights (and the a grid) this table is built from
  cluster_mass_tables();

  int rebuilt = 0;

  // --- 1. REBUILD: SIZES, ALLOCATIONS, THE ln k GRID, THE NFW TABLE ---
  if (NULL == p1h_.ln_p ||
      fdiff2(p1h_.cache[1], Ntable.random) ||
      fdiff2(p1h_.cache[2], cluster.random_model))
  {
    if (p1h_.ln_p != NULL) {
      free(p1h_.ln_p);
      free(p1h_.curv);
      free(p1h_.node);
      free(p1h_.weight);
      free(p1h_.n_active);
    }

    const int K_PAD = Ntable.halo_spline_pad;

    p1h_.nl_bins = cl_.nl_bins;
    p1h_.n_mass  = cl_.n_mass;
    p1h_.n_a     = cl_.n_a;

    // uniform ln k nodes from ln k_min (node K_PAD) past ln k_max (node
    // k_last), K_PAD pads beyond each end
    p1h_.dlnk    = cluster_p1h_lnk_spacing();
    p1h_.lnk_min = log(limits.k_min_cH0);

    const double lnk_span = log(limits.k_max_cH0) - p1h_.lnk_min;
    const int n_inside = (int) ceil(lnk_span/p1h_.dlnk) + 1;

    p1h_.n_k       = n_inside + 2*K_PAD;
    p1h_.k_last    = K_PAD + n_inside - 1;
    p1h_.lnk_first = p1h_.lnk_min - K_PAD*p1h_.dlnk;
    p1h_.lnk_last  = p1h_.lnk_first + p1h_.k_last*p1h_.dlnk;

    p1h_.ln_p     = (double***) malloc3d(p1h_.nl_bins, p1h_.n_a, p1h_.n_k);
    p1h_.curv     = (double***) malloc3d(p1h_.nl_bins, p1h_.n_a, p1h_.n_k);
    p1h_.node     = (double***) malloc3d(p1h_.n_a, PN_NROWS, p1h_.n_mass);
    p1h_.weight   = (double***) malloc3d(p1h_.n_a, p1h_.n_mass, p1h_.nl_bins);
    p1h_.n_active = (int*) malloc1d_int(p1h_.n_a);

    // the private NFW f, G table (a threaded build: serial context here)
    cluster_nfw_table();

    rebuilt = 1;
  }

  // --- 2. REFILL ---
  if (1 == rebuilt || cluster_keys_differ(p1h_.cache))
  {
    const int nl_bins = p1h_.nl_bins;
    const int n_mass  = p1h_.n_mass;
    const int n_a     = p1h_.n_a;
    const int n_k     = p1h_.n_k;

    // --- 2a. GUARDS ---
    // the rows read the NFW kernel directly (as halo.c's p_gm)
    if (like.halo_model[3] != HALO_PROFILE_NFW) {
      log_fatal("like.halo_model[3] = %d not supported", like.halo_model[3]);
      exit(1);
    }
    cluster_nfw_table();
    cluster_nfw_check();

    // the Limber integrals reach k = (l + 1/2)/f_K(chi) up to l = LMAX at
    // the near edge of the cluster support; beyond the last node ln P1h
    // is extrapolated (section header), so say so once
    const double k_limber_max =
        (Ntable.LMAX + 0.5)/f_K(chi(cl_.a_lim[1]));
    if (log(k_limber_max) > p1h_.lnk_last && 0 == p1h_.k_warned) {
      log_warn("P1h cluster table ends at k = %g (c/H0)^-1 below the Limber "
               "k_max = %g of the cluster support: extrapolated as a power "
               "law (raise limits.k_max_cH0 to tabulate it)",
               exp(p1h_.lnk_last), k_limber_max);
      p1h_.k_warned = 1;
    }

    /* PHYSICAL DERIVATION & LOGIC FLOW (the section header)
       1. per a node: keep the mass nodes that carry weight, with their
          NFW factors c, ln(1 + c), r_s, ln r_s and W_nl of every bin
       2. per (a node, k node): um = u m(c)(k r_s) of each kept node,
          P1h_nl = sum W_nl um for every bin at once
       3. per (bin, a node): natural spline of ln P1h in ln k */

    // --- 2b. PASS A: ACTIVE MASS NODES OF EACH a NODE ---
    #pragma omp parallel for schedule(static)
    for (int i=0; i<n_a; i++) {
      const int r = cl_.pad_lo + i;  // the padded row of the fill

      // weight sum of the row, per bin: the scale of the skip test
      double row_sum[MAX_SIZE_ARRAYS];
      for (int nl=0; nl<nl_bins; nl++) {
        row_sum[nl] = 0.0;
        for (int q=0; q<n_mass; q++) {
          row_sum[nl] += cl_.w1h[nl][i][q];
        }
      }

      int j = 0;  // compacted index of the active nodes
      for (int q=0; q<n_mass; q++) {
        int active = 0;
        for (int nl=0; nl<nl_bins; nl++) {
          if (cl_.w1h[nl][i][q] > CLUSTER_1H_SKIP*row_sum[nl]) {
            active = 1;
          }
        }

        if (1 == active) {
          p1h_.node[i][PN_CONC][j] = cl_.a_node[r][AN_CONC][q];
          p1h_.node[i][PN_LN1C][j] = cl_.a_node[r][AN_LN1C][q];
          p1h_.node[i][PN_RS][j]   = cl_.a_node[r][AN_RS][q];
          p1h_.node[i][PN_LNRS][j] = cl_.a_node[r][AN_LNRS][q];
          for (int nl=0; nl<nl_bins; nl++) {
            p1h_.weight[i][j][nl] = cl_.w1h[nl][i][q];
          }
          j++;
        }
      }
      p1h_.n_active[i] = j;
    }

    // --- 2c. PASS B: (a NODE, k NODE) PAIRS COLLAPSED AND THREADED ---
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i=0; i<n_a; i++) {
      for (int c=0; c<n_k; c++) {
        const double lnk = p1h_.lnk_first + c*p1h_.dlnk;
        const double k   = exp(lnk);

        const int n_active = p1h_.n_active[i];
        const double* restrict conc = p1h_.node[i][PN_CONC];
        const double* restrict ln1c = p1h_.node[i][PN_LN1C];
        const double* restrict r_s  = p1h_.node[i][PN_RS];
        const double* restrict lnrs = p1h_.node[i][PN_LNRS];

        // sum_q W_nl um, all bins at once (one kernel call per node)
        double sum[MAX_SIZE_ARRAYS];
        for (int nl=0; nl<nl_bins; nl++) {
          sum[nl] = 0.0;
        }

        for (int j=0; j<n_active; j++) {
          // u m(c) at x = k r_s, ln x = ln k + ln r_s
          const double um = cluster_nfw_um(conc[j], k*r_s[j], lnk + lnrs[j],
                                           ln1c[j]);

          const double* restrict w = p1h_.weight[i][j];
          for (int nl=0; nl<nl_bins; nl++) {
            sum[nl] += w[nl]*um;
          }
        }

        for (int nl=0; nl<nl_bins; nl++) {
          if (isnan(sum[nl])) {
            log_fatal("NaN in the 1-halo cluster sum at ln k = %g", lnk);
            exit(1);
          }

          // u > 0 for the truncated NFW, so the sum is positive whenever
          // the bin holds any weight; an empty bin (every weight 0) is
          // held at the smallest double, P1h ~ 0, instead of ln 0
          double p1h = sum[nl];
          if (!(p1h > 0.0)) {
            p1h = DBL_MIN;
          }
          p1h_.ln_p[nl][i][c] = log(p1h);
        }
      }
    }

    // --- 2d. PASS C: NATURAL SPLINES IN ln k ---
    #pragma omp parallel for collapse(2) schedule(static)
    for (int nl=0; nl<nl_bins; nl++) {
      for (int i=0; i<n_a; i++) {
        spline_coeffs_uniform(p1h_.ln_p[nl][i], n_k, p1h_.dlnk,
                              p1h_.curv[nl][i]);
      }
    }

    // --- 2e. CACHE TAGS ---
    cluster_keys_stamp(p1h_.cache);
  }
}


// ln P1h of richness bin nl on a node i at ln k (inside [ln k_min,
// ln k_last] after the caller's clamp), plus the linear continuation
// beyond ln k_last by lnk_beyond.
static inline double cluster_p1h_row(
    const int nl,             // richness bin
    const int i,              // a node
    const double lnk,         // ln k, clamped to [ln k_min, ln k_last]
    const double lnk_beyond   // ln k - ln k_last above the table, else 0
  )
{
  const double* y    = p1h_.ln_p[nl][i];
  const double* curv = p1h_.curv[nl][i];
  const double h     = p1h_.dlnk;

  const double r = (lnk - p1h_.lnk_first)/h;

  int j = (int) r;
  if (j > p1h_.n_k - 2) {
    j = p1h_.n_k - 2;
  }

  double ln_p = spline_horner(y, curv, j, (r - j)*h, h);

  if (lnk_beyond > 0.0) {
    // slope d ln P1h/d ln k of the last interval inside the range
    const int jl = p1h_.k_last;
    ln_p += lnk_beyond*(y[jl] - y[jl-1])/h;
  }

  return ln_p;
}


// ---------------------------------------------------------------------------
// P1h_nl(k, a), eq (22): the one-halo cluster-matter power spectrum of
// richness bin nl (section header).
//
// Parameters:
//   k  - wavenumber in (c/H0)^-1
//   a  - scale factor
//   nl - richness bin, 0 <= nl < cluster.richness_nbin (aborts otherwise)
//
// Returns:
//   P1h_nl(k, a) in (c/H0)^3; 0 outside the a grid
// ---------------------------------------------------------------------------
double pcm_1h_richness(
    const double k,
    const double a,
    const int nl
  )
{
  if (nl < 0 || nl > cluster.richness_nbin - 1) {
    log_fatal("error in selecting richness bin nl = %d", nl);
    exit(1);
  }

  cluster_p1h_table();

  if (a < cl_.a_lim[0] || a > cl_.a_lim[1]) {
    return 0.0;
  }

  // --- a: node i and fraction t_a (linear between the two rows) ---
  const double ra = (a - cl_.a_lim[0])/cl_.a_lim[2];
  int i = (int) ra;
  if (i > p1h_.n_a - 2) {
    i = p1h_.n_a - 2;
  }
  const double t_a = ra - i;

  // --- ln k: flat below k_min, power-law continuation above k_last ---
  double lnk = log(k);
  double lnk_beyond = 0.0;
  if (lnk < p1h_.lnk_min) {
    lnk = p1h_.lnk_min;
  }
  else if (lnk > p1h_.lnk_last) {
    lnk_beyond = lnk - p1h_.lnk_last;
    lnk = p1h_.lnk_last;
  }

  const double ln_p_lo = cluster_p1h_row(nl, i, lnk, lnk_beyond);
  const double ln_p_hi = cluster_p1h_row(nl, i + 1, lnk, lnk_beyond);

  return exp(ln_p_lo + t_a*(ln_p_hi - ln_p_lo));
}



// ============================================================================
// [SECTION] WARM-UP
// ============================================================================

// ---------------------------------------------------------------------------
// Builds every lazily filled cluster table on the calling thread, so that
// threaded loops only read them (the halo_warmup rule of halo.c): the fill
// and the P1h table of this file (with the halo.c and cosmo3D.c tables
// they read: sigma2, dlognudlogm, tinker_alpha, u_nfw_c's table), then the
// selection-kernel, n(z) and lensing-efficiency tables of
// redshift_spline_cluster.c through one read per (cluster bin, richness
// bin) at the middle of the bin's support. nz_cluster comes after the
// fill: the abundance-weighted kernel reads n_nl. The pair maps are
// warmed by the interface. Does nothing while no cluster sample (redshift
// or richness bins) is set.
//
// Must be called outside any parallel region, after every cluster setter
// and cosmology update of the step.
// ---------------------------------------------------------------------------
void cluster_warmup(void)
{
  if (cluster.richness_nbin < 1 || cluster.zdist_nbin < 1) {
    return;
  }

  // --- 1. THIS FILE ---
  cluster_mass_tables();
  cluster_p1h_table();

  // --- 2. redshift_spline_cluster.c ---
  for (int ni=0; ni<cluster.zdist_nbin; ni++) {
    const double z_mid = 0.5*(cluster.zdist_zmin[ni] + cluster.zdist_zmax[ni]);
    const double a_mid = 1.0/(1.0 + z_mid);

    (void) phi_cluster(z_mid, ni);

    for (int nl=0; nl<cluster.richness_nbin; nl++) {
      (void) nz_cluster(z_mid, ni, nl);
      (void) g_cluster(a_mid, ni, nl);
    }
  }
}
