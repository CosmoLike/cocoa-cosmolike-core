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
//   dn/dlnM = (rho_hmf/M) nu f(nu) dln nu/dln M  mass function (f(nu),
//                                                dlognudlogm)
//   nu      = delta_c/(sigma(M) D(a))            peak height (sigma2 of
//                                                cosmo3D.c, D = growfac)
//   b_h     = hb1nu(nu, a)                       Tinker 2010 halo bias
//   u(k|M)  = truncated NFW transform, c = conc(M, D) (Bhattacharya 2013)
//   rho_m   = rho_crit Omega_m                   total matter, neutrinos
//                                                included: r_Delta and
//                                                the matter window M/rho_m
//   rho_hmf = rho_crit omega_halo_field()        the halo field's mean
//                                                density: rho_m, or
//                                                rho_crit (Omega_m -
//                                                Omega_nu) under cb
//
// The halo field is halo.c's (like.halo_model[4], halo.h): under
// HALO_FIELD_CB (the DES Y1 model, 2010.01138) sigma(M) is the variance
// of cold dark matter + baryons and rho_hmf their mean density, the
// counts and the bias follow nu_cb, and r_Delta, the 1-halo window
// M/rho_m and the 2-halo spectrum (cosmo2D_cluster.c) stay total
// matter. The default HALO_FIELD_MATTER makes rho_hmf = rho_m.
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
// One ingredient departs from halo.c: the amplitude alpha of the Tinker
// 2010 multiplicity f(nu) follows cluster.hmf_alpha_mode
// (structs_cluster.h). The DES analyses fix alpha = 0.368 (1001.3162
// Table 4) at every z: the default, CLUSTER_HMF_ALPHA_FIXED, evaluates a
// private copy of halo.c's f(nu) shape at that amplitude (TINKER 2010
// MULTIPLICITY section). CLUSTER_HMF_ALPHA_NORMALIZED calls halo.c's fnu,
// whose alpha(a) satisfies int b f dnu = 1 (Eq. 7) at every z. The shape
// is the same in both modes, so only n_nl (and the counts) depend on it.
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
//   pcm_1h_richness_fill = that read at many (k, a) points and every
//                    richness bin in one call (the Limber nodes of one
//                    multipole), four points per SIMDe vector
//   cluster_tinker_* = a private copy of halo.c's Tinker 2010 f(nu) shape,
//                    at the fixed amplitude 0.368 (CLUSTER_HMF_ALPHA_FIXED)
//   cluster_mass_tables = THE fill: one deep-unrolled loop nest over
//                    (richness bin, a node) with the Gauss-Legendre mass
//                    nodes innermost; it fills n_nl, b_nl and the 1-halo
//                    mass weights
//   cluster_p1h_table = P1h on exact ln k nodes from those weights
//   cluster_nfw_*  = a private copy of halo.c's NFW kernel (nfw_um and its
//                    f, G table), verified against halo.c's u_nfw_c
//   cluster_nfw_um4 = that kernel on four mass nodes per SIMDe vector (a
//                    private copy of halo.c's nfw_um4), bitwise the scalar
//                    kernel on each node
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
//
// SIMD: the P1h sums evaluate the NFW kernel on four mass nodes per SIMDe
// vector, and the batch read pcm_1h_richness_fill takes four (k, a)
// points per vector (AVX2 on x86-64, NEON on arm64, from one source).
// Both vector paths perform the scalar path's floating-point operations
// in the scalar order on every element, so their values are bitwise the
// scalar path's. COSMO2D_NOT_USE_SIMD (the DEBUG build; basics.h then
// leaves SIMDe out) selects the scalar loops, the reference.
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

// Amplitude of the Tinker 2010 multiplicity f(nu) under
// CLUSTER_HMF_ALPHA_FIXED: 1001.3162 Table 4 at Delta = 200 (mean), the
// value of the DES cluster analyses, held at every z.
static const double CLUSTER_TINKER_ALPHA_FIXED = 0.368;

// Lowest scale factor of the Tinker 2010 parameter evolution: beyond z = 3
// the paper recommends the z = 3 parameters (text after its Eq. 12), as
// halo.c's fnu does.
static const double CLUSTER_TINKER_A_FLOOR = 0.25;

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
// [SECTION] TINKER 2010 MULTIPLICITY AT FIXED AMPLITUDE (private copy of
//           halo.c's fnu_shape and fnu_core)
// ============================================================================
//
// The halo multiplicity of Tinker et al. 2010 (1001.3162 Eq. 8), the mass
// function per unit peak height nu (full derivation: halo.c, fnu header):
//
//   f(nu) = alpha [1 + (beta nu)^(-2 phi)] nu^(2 eta) exp(-gamma nu^2/2)
//
// with the Table 4 parameters at Delta = 200 (mean) evolved by Eqs. 9-12,
// 1 + z = 1/aa, aa = max(a, 0.25) (the fit is frozen at z = 3):
//
//   beta  = 0.589 aa^-0.20,  gamma = 0.864 aa^0.01,
//   phi   = -0.729 aa^0.08,  eta   = -0.243 aa^-0.27
//
// halo.c's fnu takes alpha(a) from the peak-background relation
// int b f dnu = 1 (Eq. 7): 0.3684 at z = 0, falling with z. The DES
// cluster analyses hold alpha = 0.368, the Table 4 value, at every z
// (CLUSTER_HMF_ALPHA_FIXED), and halo.c keeps its shape functions static,
// so this file holds a copy of them at that fixed amplitude. The
// parameters depend on a only: the fill evaluates them once per a row.
// cluster_tinker_check verifies the copy against halo.c's fnu at every
// refill of the mass tables.

// Tinker 2010 parameters at one a (the four shape parameters and alpha)
typedef struct {
  double alpha;  // amplitude: CLUSTER_TINKER_ALPHA_FIXED
  double beta;   // the four shape parameters, Eqs. 9-12 + Table 4 of
  double gamma;  //   1001.3162 at Delta = 200
  double phi;
  double eta;
} cluster_tinker_params;


// Eqs. 9-12 + Table 4 at aa = max(a, 0.25), alpha = 0.368
static inline cluster_tinker_params cluster_tinker_params_fixed(
    const double a  // scale factor, 0 < a < 1
  )
{
  const double aa = fmax(CLUSTER_TINKER_A_FLOOR, a);

  cluster_tinker_params p;
  p.alpha = CLUSTER_TINKER_ALPHA_FIXED;
  p.beta  = 0.589*pow(aa, -0.2);    // Eq. 9:  beta_0  (1+z)^0.20
  p.gamma = 0.864*pow(aa, 0.01);    // Eq. 12: gamma_0 (1+z)^-0.01
  p.phi   = -0.729*pow(aa, .08);    // Eq. 10: phi_0   (1+z)^-0.08
  p.eta   = -0.243*pow(aa, -0.27);  // Eq. 11: eta_0   (1+z)^0.27
  return p;
}


// Eq. 8 itself, in halo.c's operation order (fnu_core)
static inline double cluster_tinker_fnu(
    const double nu,                   // peak height delta_c/(sigma D)
    const cluster_tinker_params* p     // from cluster_tinker_params_fixed
  )
{
  return p->alpha*(1. + pow(p->beta*nu,-2*p->phi))*pow(nu,2*p->eta)*
         exp(-p->gamma*nu*nu/2.);
}


// The private shape against halo.c's fnu at two scale factors (one inside
// the fit range, one below its z = 3 floor) and four peak heights: the
// ratio fnu/cluster_tinker_fnu = alpha(a)/0.368 must not depend on nu.
// Aborts on a mismatch (halo.c's Tinker parameters changed). Builds
// halo.c's lazy tinker_alpha table: call it serially.
static void cluster_tinker_check(void)
{
  const double TEST_A[2]  = {0.7, 0.2};
  const double TEST_NU[4] = {0.3, 1.0, 2.5, 5.0};
  const double TOLERANCE  = 1.0e-12;  // a few roundings of the product

  if (like.halo_model[0] != HMF_TINKER_2010) {
    log_fatal("like.halo_model[0] = %d: the cluster mass function is "
              "Tinker 2010 only (HMF_TINKER_2010)", like.halo_model[0]);
    exit(1);
  }

  for (int i=0; i<2; i++) {
    const cluster_tinker_params p = cluster_tinker_params_fixed(TEST_A[i]);
    const double ratio0 = fnu(TEST_NU[0], TEST_A[i])/
                          cluster_tinker_fnu(TEST_NU[0], &p);

    for (int j=1; j<4; j++) {
      const double ratio = fnu(TEST_NU[j], TEST_A[i])/
                           cluster_tinker_fnu(TEST_NU[j], &p);

      if (!(fabs(ratio/ratio0 - 1.0) < TOLERANCE)) {
        log_fatal("private Tinker 2010 shape differs from halo.c fnu at "
                  "a = %g, nu = %g (ratio %.17g vs %.17g at nu = %g): "
                  "resync halo_cluster.c with halo.c", TEST_A[i],
                  TEST_NU[j], ratio, ratio0, TEST_NU[0]);
        exit(1);
      }
    }
  }
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


#ifndef COSMO2D_NOT_USE_SIMD
// ============================================================================
// [SECTION] SIMDe PATH OF THE NFW KERNEL (private copy of halo.c's nfw_um4)
// ============================================================================
//
// cluster_nfw_um on four mass nodes at once, for the P1h sums of
// cluster_p1h_table (pass B): cluster_nfw_um4 is cluster_nfw_um with each
// of its four arguments carrying four nodes, and each lane of its result
// bitwise the scalar cluster_nfw_um of that node. halo.c keeps its vector
// kernel (nfw_um4 and its helpers) static, as it keeps the scalar one, so
// this file holds a copy of them that reads the private table cluster_nfw_.
// The operations are halo.c's, statement for statement; its headers carry
// the long form of every explanation below.
//
// SIMDe (simde/x86/avx2.h and fma.h) gives AVX2 on x86-64 and NEON on
// arm64 from one source. basics.h includes it only when
// COSMO2D_NOT_USE_SIMD is not defined (the DEBUG build defines it), so
// every SIMDe type and call of this file sits inside
// #ifndef COSMO2D_NOT_USE_SIMD, with the scalar loop, the reference, in
// the other branch.
//
// A v4d holds four doubles side by side, its "lanes" 0, 1, 2, 3 (one AVX2
// register on x86-64, two NEON registers on arm64); a v2d holds two: one
// half of a v4d, lanes 0,1 (the low half) or lanes 2,3 (the high half).
// Vector variables carry a v prefix. One simde_mm256_* call applies the
// same operation to all four lanes, so a v4d line does what the scalar
// line quoted above it does for one node, four nodes at a time.
//
// The helpers, in reading order:
//
//   cluster_fmadd4, cluster_fnmadd4 - a*b + c and c - a*b with one rounding
//   cluster_nfw_pos4         - position (node index, fraction) of ln t on
//                              the table grid
//   cluster_nfw_read4        - the linear table read at that position
//   cluster_nfw_sin4         - libm sin on each lane
//   cluster_nfw_series_step4 - one bracket of the asymptotic series
//   cluster_nfw_G_asym4      - the asymptotic series of G(t)
//   cluster_nfw_um4          - the kernel itself
//
// Why each lane is bitwise the scalar path, under the strict IEEE flags of
// the default build (-frounding-math -ftrapping-math): a fused
// multiply-add exactly where the compiler fuses the scalar a*b + c,
// sign-bit masks instead of floating-point compares (the strict flags
// split a vector compare into scalar compares per lane), table indices
// truncated and clamped in double, libm sin on every lane, and every
// helper always inlined (on arm64 a v4d is a union of two NEON registers
// and a real call would pass it through memory).
typedef simde__m256d v4d;   // 4 doubles
typedef simde__m128d v2d;   // 2 doubles: one half of a v4d


// ---------------------------------------------------------------------------
// cluster_fmadd4: a*b + c on four lanes with one rounding.
//
// A fused multiply-add keeps the product a*b exact and rounds only the
// final sum; a separate multiply and add rounds twice, and the two results
// can differ in the last bit. The compiler fuses the scalar path's
// a*b + c (cluster_nfw_um, the spline reads), so the vector path must fuse
// the same products at the same places to stay bitwise equal to it.
//
// With native x86 FMA, simde_mm256_fmadd_pd is one AVX2 instruction.
// Without it (arm64) SIMDe writes that call as a multiply and then an
// add, two roundings, while the two-lane simde_mm_fmadd_pd is a real
// fused NEON instruction: the v4d is split into its two v2d halves, each
// half is fused, and the halves are joined again. Lane l of the result is
// a[l]*b[l] + c[l] either way.
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d cluster_fmadd4(
    const v4d va,   // a on four lanes
    const v4d vb,   // b on four lanes
    const v4d vc    // c on four lanes
  )
{
#ifdef SIMDE_X86_FMA_NATIVE
  // a*b + c on all four lanes, one fused instruction
  return simde_mm256_fmadd_pd(va, vb, vc);
#else
  // the low half of each input (castpd256_pd128 keeps the lower two
  // doubles; it moves no data)
  const v2d va_low = simde_mm256_castpd256_pd128(va);  // a, lanes 0,1
  const v2d vb_low = simde_mm256_castpd256_pd128(vb);  // b, lanes 0,1
  const v2d vc_low = simde_mm256_castpd256_pd128(vc);  // c, lanes 0,1

  // the high half of each input (extractf128_pd(v, 1) takes the upper
  // two doubles)
  const v2d va_high = simde_mm256_extractf128_pd(va, 1);  // a, lanes 2,3
  const v2d vb_high = simde_mm256_extractf128_pd(vb, 1);  // b, lanes 2,3
  const v2d vc_high = simde_mm256_extractf128_pd(vc, 1);  // c, lanes 2,3

  // a*b + c fused on lanes 0,1
  const v2d vlow  = simde_mm_fmadd_pd(va_low, vb_low, vc_low);

  // a*b + c fused on lanes 2,3
  const v2d vhigh = simde_mm_fmadd_pd(va_high, vb_high, vc_high);

  // join the halves: set_m128d(high, low) puts vlow in lanes 0,1 and
  // vhigh in lanes 2,3
  return simde_mm256_set_m128d(vhigh, vlow);
#endif
}


// ---------------------------------------------------------------------------
// cluster_fnmadd4: c - a*b on four lanes with one rounding (cluster_fmadd4
// with the product negated). The asymptotic series of cluster_nfw_um is a
// chain of 1 - k v (...) steps that the compiler fuses this way in the
// scalar path.
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d cluster_fnmadd4(
    const v4d va,   // a on four lanes
    const v4d vb,   // b on four lanes
    const v4d vc    // c on four lanes
  )
{
#ifdef SIMDE_X86_FMA_NATIVE
  // c - a*b on all four lanes, one fused instruction
  return simde_mm256_fnmadd_pd(va, vb, vc);
#else
  // the low half of each input, as in cluster_fmadd4
  const v2d va_low = simde_mm256_castpd256_pd128(va);  // a, lanes 0,1
  const v2d vb_low = simde_mm256_castpd256_pd128(vb);  // b, lanes 0,1
  const v2d vc_low = simde_mm256_castpd256_pd128(vc);  // c, lanes 0,1

  // the high half of each input
  const v2d va_high = simde_mm256_extractf128_pd(va, 1);  // a, lanes 2,3
  const v2d vb_high = simde_mm256_extractf128_pd(vb, 1);  // b, lanes 2,3
  const v2d vc_high = simde_mm256_extractf128_pd(vc, 1);  // c, lanes 2,3

  // c - a*b fused on lanes 0,1
  const v2d vlow  = simde_mm_fnmadd_pd(va_low, vb_low, vc_low);

  // c - a*b fused on lanes 2,3
  const v2d vhigh = simde_mm_fnmadd_pd(va_high, vb_high, vc_high);

  // join the halves: vlow in lanes 0,1, vhigh in lanes 2,3
  return simde_mm256_set_m128d(vhigh, vlow);
#endif
}


// ---------------------------------------------------------------------------
// cluster_nfw_pos4: cluster_nfw_pos on four lanes, the position of ln t on
// the cluster_nfw_ grid for four values of t at once:
//
//   pos  = (max(ln t, ln t_min) - ln t_min)/spacing
//   i    = min(trunc(pos), n_nodes - 2)      (clamped onto the last cell)
//   frac = pos - i
//
// Lane l of vlnt is one ln t; index[l] and lane l of the result are its i
// and frac.
//
// Why it is bitwise cluster_nfw_pos: the scalar path computes pos in
// double, converts it to int (which truncates) and clamps. Here trunc and
// min are taken in double, and both are exact for the non-negative
// positions of the grid, so pos - i is the same double and the int
// conversion, done lane by lane, yields the same i. A vector
// double-to-int conversion is avoided on purpose: the strict IEEE flags
// split it into scalar conversions per lane.
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d cluster_nfw_pos4(
    const v4d vlnt,   // ln t on four lanes
    int index[4]      // output: node index i of each lane
  )
{
  // ln t_min, the first grid node, in all four lanes (set1 copies one
  // scalar into every lane)
  const v4d vlnt_min = simde_mm256_set1_pd(cluster_nfw_.lim[0]);

  // 1/spacing of the grid in all four lanes
  const v4d vinv_spacing = simde_mm256_set1_pd(cluster_nfw_.inv_spacing);

  // n_nodes - 2, the index of the last cell, in all four lanes
  const v4d vlast_cell =
      simde_mm256_set1_pd((double) (cluster_nfw_.n_nodes - 2));

  // max(ln t, ln t_min): the table clamp below ln t_min
  const v4d vlnt_clamped = simde_mm256_max_pd(vlnt, vlnt_min);

  // ln t - ln t_min, the distance from the first grid node
  const v4d vlnt_offset = simde_mm256_sub_pd(vlnt_clamped, vlnt_min);

  // pos = (ln t - ln t_min)/spacing, the position in grid cells
  const v4d vpos = simde_mm256_mul_pd(vlnt_offset, vinv_spacing);

  // trunc(pos): round toward zero, the cell number as a double
  const v4d vpos_trunc = simde_mm256_round_pd(vpos, SIMDE_MM_FROUND_TO_ZERO);

  // i = min(trunc(pos), n - 2): the last-cell clamp
  const v4d vnode = simde_mm256_min_pd(vpos_trunc, vlast_cell);

  double node[4];

  // the four cell numbers to the plain double[4] (storeu writes the four
  // lanes to memory), then to int lane by lane
  simde_mm256_storeu_pd(node, vnode);
  for (int lane=0; lane<4; lane++) {
    index[lane] = (int) node[lane];
  }

  // frac = pos - i on each lane
  return simde_mm256_sub_pd(vpos, vnode);
}


// ---------------------------------------------------------------------------
// cluster_nfw_read4: the linear table read of cluster_nfw_um on four
// lanes, tab[i] + frac*(tab[i + 1] - tab[i]); lane l reads tab at index[l]
// with lane l of vfrac.
//
// Memory access: each lane needs the two neighbours tab[i], tab[i + 1],
// which sit side by side, so one 16-byte load per lane fetches both (a
// v2d pair); the four pairs are then regrouped into a v4d of left nodes
// and a v4d of right nodes. A gather instruction would do the same, but
// it is slow on several x86 cores and is lane-by-lane loads on NEON
// anyway.
//
// Why it is bitwise cluster_nfw_um: the scalar read is
// frac*(tab[i + 1] - tab[i]) + tab[i], which the compiler fuses into one
// multiply-add; the vector read fuses the same product (cluster_fmadd4).
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d cluster_nfw_read4(
    const double* restrict tab,  // table row on the ln t grid: f or G
    const int index[4],          // node index i of each lane
    const v4d vfrac              // fraction of the cell of each lane
  )
{
  // the pair (tab[i], tab[i + 1]) of each lane, one two-double load each
  // (loadu reads two consecutive doubles from memory into a v2d)
  const v2d vpair0 = simde_mm_loadu_pd(tab + index[0]);  // lane 0's pair
  const v2d vpair1 = simde_mm_loadu_pd(tab + index[1]);  // lane 1's pair
  const v2d vpair2 = simde_mm_loadu_pd(tab + index[2]);  // lane 2's pair
  const v2d vpair3 = simde_mm_loadu_pd(tab + index[3]);  // lane 3's pair

  // regroup the pairs into left nodes tab[i] and right nodes tab[i + 1]:
  // unpacklo takes the first double of each pair, unpackhi the second

  // (tab[i0], tab[i1]): the left nodes of lanes 0,1
  const v2d vleft_low = simde_mm_unpacklo_pd(vpair0, vpair1);

  // (tab[i2], tab[i3]): the left nodes of lanes 2,3
  const v2d vleft_high = simde_mm_unpacklo_pd(vpair2, vpair3);

  // (tab[i0 + 1], tab[i1 + 1]): the right nodes of lanes 0,1
  const v2d vright_low = simde_mm_unpackhi_pd(vpair0, vpair1);

  // (tab[i2 + 1], tab[i3 + 1]): the right nodes of lanes 2,3
  const v2d vright_high = simde_mm_unpackhi_pd(vpair2, vpair3);

  // tab[i] on all four lanes (set_m128d joins the halves, low first)
  const v4d vleft = simde_mm256_set_m128d(vleft_high, vleft_low);

  // tab[i + 1] on all four lanes
  const v4d vright = simde_mm256_set_m128d(vright_high, vright_low);

  // tab[i + 1] - tab[i], the rise across the cell
  const v4d vrise = simde_mm256_sub_pd(vright, vleft);

  // frac*(tab[i + 1] - tab[i]) + tab[i], fused as the scalar read
  return cluster_fmadd4(vfrac, vrise, vleft);
}


// ---------------------------------------------------------------------------
// cluster_nfw_sin4: sin on each of four lanes, with the libm sin of the
// scalar path.
//
// cluster_nfw_um needs sin(c x/2) and sin(c x) per node, about half of
// the kernel's cost. There is no vector sine here (the rule of halo.c's
// nfw_sin4): a vector math library would give a different last bit from
// libm, and the vector path must be bitwise cluster_nfw_um. So the four
// angles are written out of the v4d into a plain double[4], sin is called
// on each one exactly as the scalar path does, and the four sines are
// read back into a v4d.
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d cluster_nfw_sin4(
    const v4d vangle   // the four angles, in radians
  )
{
  double angle[4];
  double sine[4];

  // the four angles to a plain double[4]
  simde_mm256_storeu_pd(angle, vangle);

  // sin of each angle, the same libm call as the scalar path
  for (int lane=0; lane<4; lane++) {
    sine[lane] = sin(angle[lane]);
  }

  // the four sines back into one v4d (loadu reads four doubles from memory)
  return simde_mm256_loadu_pd(sine);
}


// ---------------------------------------------------------------------------
// cluster_nfw_series_step4: one step of the nested asymptotic series of
// cluster_nfw_um, poly -> 1 - k v poly, on four lanes, fused as the
// scalar 1.0 - k*v*(...) (cluster_fnmadd4). Starting from the innermost
// bracket (1 - 56v or 1 - 72v), each call wraps the series in one more
// bracket, k taking the next coefficient ratio (cluster_nfw_um header).
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d cluster_nfw_series_step4(
    const double k,   // coefficient ratio of this bracket
    const v4d vv,     // series variable v = 1/t^2 on four lanes
    const v4d vpoly   // the nested bracket built so far
  )
{
  // the coefficient ratio k in all four lanes
  const v4d vratio = simde_mm256_set1_pd(k);

  // k v on each lane
  const v4d vkv = simde_mm256_mul_pd(vratio, vv);

  // 1 in all four lanes
  const v4d vone = simde_mm256_set1_pd(1.0);

  // 1 - (k v) poly, one rounding, as the scalar 1.0 - k*v*(...)
  return cluster_fnmadd4(vkv, vpoly, vone);
}


// ---------------------------------------------------------------------------
// cluster_nfw_G_asym4: the asymptotic series of G(t) = g(t) + ln t on
// four lanes,
//
//   G(t) = v(1 - 6v(1 - 20v(1 - 42v(1 - 72v)))) + ln t,   v = 1/t^2,
//
// the same brackets in the same order as cluster_nfw_um, each
// 1 - k v (..) fused (cluster_fnmadd4) and the final v poly + ln t fused
// (cluster_fmadd4), as the compiler fuses the scalar expression.
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d cluster_nfw_G_asym4(
    const v4d vv,    // series variable v = 1/t^2 on four lanes
    const v4d vlnt   // ln t on four lanes
  )
{
  // 1 in all four lanes
  const v4d vone = simde_mm256_set1_pd(1.0);

  // the innermost coefficient ratio 72 in all four lanes
  const v4d vratio72 = simde_mm256_set1_pd(72.0);

  // 1 - 72v
  v4d vpoly = cluster_fnmadd4(vratio72, vv, vone);

  // the outer brackets, one per step
  vpoly = cluster_nfw_series_step4(42.0, vv, vpoly);  // 1 - 42v(1 - 72v)
  vpoly = cluster_nfw_series_step4(20.0, vv, vpoly);  // 1 - 20v(...)
  vpoly = cluster_nfw_series_step4(6.0, vv, vpoly);   // 1 - 6v(...)

  // v poly + ln t, fused
  return cluster_fmadd4(vv, vpoly, vlnt);
}


// ---------------------------------------------------------------------------
// cluster_nfw_um4: cluster_nfw_um on four mass nodes at once, u m(c) of
// four halos at one k. Lane l of every argument belongs to one mass node
// and lane l of the result is u m(c) of that node.
//
// Algorithm, in order (the banners in the body):
//   1. branch masks: per lane, table (t <= CLUSTER_NFW_TASY) or asymptotic
//      series (t > CLUSTER_NFW_TASY) for t = xu and for t = x, from the
//      sign bit of CLUSTER_NFW_TASY - t (movemask), no FP compare;
//   2. table reads: f(xu), G(xu), G(x) (cluster_nfw_pos4,
//      cluster_nfw_read4), skipped when every lane is past the table;
//   3. asymptotic series: f(xu), G(xu), G(x) in v = 1/t^2, skipped when no
//      lane needs it; per lane, blendv keeps the table value or takes the
//      series;
//   4. the two sines (cluster_nfw_sin4), then the combination.
//
// Why it is bitwise cluster_nfw_um: the same operations in the same order
// on every lane, a fused multiply-add exactly where the compiler fuses
// the scalar a*b + c, and libm sin on every lane. The branch is taken per
// lane by masks, so a lane past CLUSTER_NFW_TASY gets the series value
// and a lane below it the table value, as the scalar if/else would give.
//
// cluster_nfw_table must have run (the build is not thread-safe; this
// read is).
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d cluster_nfw_um4(
    const v4d vc,     // concentration r_Delta/r_s on four lanes
    const v4d vx,     // k r_s on four lanes
    const v4d vlnx,   // ln x on four lanes
    const v4d vln1c   // ln(1 + c) on four lanes
  )
{
  const double* restrict tab_f = cluster_nfw_.tab[0];  // f(t)
  const double* restrict tab_G = cluster_nfw_.tab[1];  // G(t) = g(t) + ln t

  // 1 in all four lanes
  const v4d vone = simde_mm256_set1_pd(1.0);

  // CLUSTER_NFW_TASY, the top of the table, in all four lanes
  const v4d vtasy = simde_mm256_set1_pd(CLUSTER_NFW_TASY);

  // scalar: lnxu = lnx + ln1c;  xu = (1.0 + c)*x

  // ln xu = ln x + ln(1 + c)
  const v4d vlnxu = simde_mm256_add_pd(vlnx, vln1c);

  // 1 + c
  const v4d vone_plus_c = simde_mm256_add_pd(vone, vc);

  // xu = (1 + c) x
  const v4d vxu = simde_mm256_mul_pd(vone_plus_c, vx);

  // --- 1. BRANCH MASKS ---
  // scalar: if (xu <= CLUSTER_NFW_TASY) read the table, else the series;
  // the same for x. CLUSTER_NFW_TASY - t has its sign bit set exactly
  // where t > CLUSTER_NFW_TASY (the series) and clear where the table is
  // read (+0 at t = CLUSTER_NFW_TASY). The masks must see the rounded xu
  // of the scalar compare: xu = (1 + c) x stays unfused because it has
  // other uses (1/xu, the blend), which GCC's -ffp-contract=fast needs to
  // leave the product alone

  // CLUSTER_NFW_TASY - xu: negative (sign bit set) on the series lanes
  const v4d vasym_u = simde_mm256_sub_pd(vtasy, vxu);

  // CLUSTER_NFW_TASY - x
  const v4d vasym_x = simde_mm256_sub_pd(vtasy, vx);

  // movemask collects the sign bit of each lane into bit l of an int:
  // 0 = every lane reads the table, 0xF = every lane takes the series
  const int asym_u = simde_mm256_movemask_pd(vasym_u);  // for t = xu
  const int asym_x = simde_mm256_movemask_pd(vasym_x);  // for t = x

  // --- 2. TABLE READS: f(xu), G(xu), G(x) ---
  // lanes past CLUSTER_NFW_TASY read the clamped last interval; the blend
  // below replaces them

  // (0, 0, 0, 0) until a branch fills them
  v4d vfu = simde_mm256_setzero_pd();  // f(xu)
  v4d vGu = simde_mm256_setzero_pd();  // G(xu)
  v4d vGx = simde_mm256_setzero_pd();  // G(x)

  if (asym_u != 0xF) {
    int index_u[4];

    // scalar: iu = cluster_nfw_pos(lnxu, &frac_u), f and G share the grid
    const v4d vfrac_u = cluster_nfw_pos4(vlnxu, index_u);

    // scalar: Gu = frac_u*(tab_G[iu + 1] - tab_G[iu]) + tab_G[iu]
    vGu = cluster_nfw_read4(tab_G, index_u, vfrac_u);

    // scalar: fu = frac_u*(tab_f[iu + 1] - tab_f[iu]) + tab_f[iu]
    vfu = cluster_nfw_read4(tab_f, index_u, vfrac_u);
  }
  if (asym_x != 0xF) {
    int index_x[4];

    // scalar: ix = cluster_nfw_pos(lnx, &frac_x)
    const v4d vfrac_x = cluster_nfw_pos4(vlnx, index_x);

    // scalar: Gx = frac_x*(tab_G[ix + 1] - tab_G[ix]) + tab_G[ix]
    vGx = cluster_nfw_read4(tab_G, index_x, vfrac_x);
  }

  // --- 3. ASYMPTOTIC SERIES (A&S 5.2.34-35, cluster_nfw_um) ---
  // lanes that read the table evaluate the series at t = CLUSTER_NFW_TASY,
  // a finite stand-in (no 1/t^2 overflow at tiny t) that the blend discards
  if (asym_u != 0) {
    // t = xu on the series lanes, CLUSTER_NFW_TASY on the table lanes
    // (blendv takes lane l from its second argument where the sign bit of
    // lane l of the mask is set, from its first argument otherwise)
    const v4d vt = simde_mm256_blendv_pd(vtasy, vxu, vasym_u);

    // t^2
    const v4d vt2 = simde_mm256_mul_pd(vt, vt);

    // scalar: v = 1.0/(xu*xu), the series variable
    const v4d vv = simde_mm256_div_pd(vone, vt2);

    // scalar: fu = (1 - 2v(1 - 12v(1 - 30v(1 - 56v))))/xu, innermost first

    // the innermost coefficient ratio 56 in all four lanes
    const v4d vratio56 = simde_mm256_set1_pd(56.0);

    // 1 - 56v
    v4d vpoly = cluster_fnmadd4(vratio56, vv, vone);

    // the outer brackets, one per step
    vpoly = cluster_nfw_series_step4(30.0, vv, vpoly);  // 1 - 30v(1 - 56v)
    vpoly = cluster_nfw_series_step4(12.0, vv, vpoly);  // 1 - 12v(...)
    vpoly = cluster_nfw_series_step4(2.0, vv, vpoly);   // 1 - 2v(...)

    // poly/t
    const v4d vfu_asym = simde_mm256_div_pd(vpoly, vt);

    // f(xu): the series on the series lanes, the table value elsewhere
    vfu = simde_mm256_blendv_pd(vfu, vfu_asym, vasym_u);

    // scalar: Gu = v*(1 - 6v(1 - 20v(1 - 42v(1 - 72v)))) + lnxu
    const v4d vGu_asym = cluster_nfw_G_asym4(vv, vlnxu);

    // G(xu): the series on the series lanes, the table value elsewhere
    vGu = simde_mm256_blendv_pd(vGu, vGu_asym, vasym_u);
  }
  if (asym_x != 0) {
    // t = x on the series lanes, CLUSTER_NFW_TASY on the table lanes
    const v4d vt = simde_mm256_blendv_pd(vtasy, vx, vasym_x);

    // t^2
    const v4d vt2 = simde_mm256_mul_pd(vt, vt);

    // scalar: w = 1.0/(x*x), the series variable
    const v4d vw = simde_mm256_div_pd(vone, vt2);

    // scalar: Gx = w*(1 - 6w(1 - 20w(1 - 42w(1 - 72w)))) + lnx
    const v4d vGx_asym = cluster_nfw_G_asym4(vw, vlnx);

    // G(x): the series on the series lanes, the table value elsewhere
    vGx = simde_mm256_blendv_pd(vGx, vGx_asym, vasym_x);
  }

  // --- 4. u m(c) ---
  // u m(c) = [Gx - Gu + ln(1+c)] + 2 g(xu) sin^2(c x/2)
  //          + [f(xu) - 1/xu] sin(c x),
  // g(xu) = Gu - ln xu; the scalar sum fuses both products:
  //   gu = Gu - lnxu;  sin_half = sin(0.5*c*x);
  //   (Gx - Gu + ln1c) + 2.0*gu*sin_half*sin_half + (fu - 1.0/xu)*sin(c*x)

  // 2 in all four lanes
  const v4d vtwo = simde_mm256_set1_pd(2.0);

  // gu = Gu - ln xu, that is g(xu)
  const v4d vgu = simde_mm256_sub_pd(vGu, vlnxu);

  // 2 gu
  const v4d vtwo_gu = simde_mm256_mul_pd(vtwo, vgu);

  // 0.5 in all four lanes
  const v4d vhalf = simde_mm256_set1_pd(0.5);

  // 0.5 c
  const v4d vhalf_c = simde_mm256_mul_pd(vhalf, vc);

  // 0.5 c x, the half angle
  const v4d vhalf_cx = simde_mm256_mul_pd(vhalf_c, vx);

  // sin(c x/2)
  const v4d vsin_half = cluster_nfw_sin4(vhalf_cx);

  // c x, the full angle
  const v4d vcx = simde_mm256_mul_pd(vc, vx);

  // sin(c x)
  const v4d vsin_full = cluster_nfw_sin4(vcx);

  // Gx - Gu
  const v4d vGx_minus_Gu = simde_mm256_sub_pd(vGx, vGu);

  // Gx - Gu + ln(1 + c), that is g(x) - g(xu)
  const v4d vg_diff = simde_mm256_add_pd(vGx_minus_Gu, vln1c);

  // 1/xu
  const v4d vinv_xu = simde_mm256_div_pd(vone, vxu);

  // fu - 1/xu
  const v4d vf_term = simde_mm256_sub_pd(vfu, vinv_xu);

  // 2 gu sin(c x/2)
  const v4d vtwo_gu_sin = simde_mm256_mul_pd(vtwo_gu, vsin_half);

  // (2 gu sin_half) sin_half + g_diff, fused as the scalar sum
  const v4d vsum = cluster_fmadd4(vtwo_gu_sin, vsin_half, vg_diff);

  // (fu - 1/xu) sin(c x) + the rest, fused: u m(c) on the four lanes
  return cluster_fmadd4(vf_term, vsin_full, vsum);
}
#endif


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
//   dn_q(a) = w_q (rho_hmf/M_q) nu f(nu) dln nu/dln M,  nu = nu0_q/D(a):
//             the quadrature weight times dn/dlnM (halo.c's product order);
//             f(nu) = cluster_tinker_fnu (alpha = 0.368) or halo.c's fnu
//             (alpha(a) of Eq. 7), per cluster.hmf_alpha_mode
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
//   per refill, per q      ln M_q, M_q, w_q (rho_hmf/M_q) dln nu/dln M,
//     (mass_node)          nu0_q = delta_c/sigma(M_q), the mass part of
//                          <ln lambda>, r_Delta(M_q), the mass part of S
//   per a row              a, z, D(a), the redshift parts of <ln lambda>
//     (a_row)              and of S; the Tinker 2010 parameters (fixed
//                          amplitude mode; inside the threaded row loop)
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
  MN_WEIGHT,    // half_width w_q (rho_hmf/M_q) dln nu/dln M
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
//     richness edges, mass range, MOR and selection models and pivots,
//     the mass-function amplitude mode (hmf_alpha_mode); random_zdist the supports zdist_zmin/zmax (the a grid);
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

    // 1: f(nu) at alpha = 0.368 (the private copy); 0: halo.c's fnu
    const int alpha_fixed = (CLUSTER_HMF_ALPHA_FIXED == cluster.hmf_alpha_mode);

    // --- 2a. INPUT CHECKS ---
    if (cluster.zdist_nbin < 1) {
      log_fatal("cluster redshift bins not set (cluster.zdist_nbin = %d)",
                cluster.zdist_nbin);
      exit(1);
    }
    if (cluster.hmf_alpha_mode != CLUSTER_HMF_ALPHA_FIXED &&
        cluster.hmf_alpha_mode != CLUSTER_HMF_ALPHA_NORMALIZED) {
      log_fatal("cluster.hmf_alpha_mode = %d not supported",
                cluster.hmf_alpha_mode);
      exit(1);
    }
    // inside the ln M range of halo.c's sigma2 and dlognudlogm tables,
    // which clamp outside it
    if (!(cluster.m_min >= limits.halo_m[RANGE_MIN]) ||
        !(cluster.m_max <= limits.halo_m[RANGE_MAX]) ||
        !(cluster.m_max > cluster.m_min)) {
      log_fatal("cluster mass range [%g, %g] not inside the halo.c tables' "
                "[%g, %g]", cluster.m_min, cluster.m_max,
                limits.halo_m[RANGE_MIN], limits.halo_m[RANGE_MAX]);
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
    // tinker_alpha (inside fnu, which cluster_tinker_check calls while it
    // checks the private Tinker copy) build here, before the threads start
    (void) sigma2(cluster.m_min);
    (void) dlognudlogm(cluster.m_min);
    cluster_tinker_check();

    /* PHYSICAL DERIVATION & LOGIC FLOW (the section header)
       1. per mass node q: M_q, the a-free factor of dn/dlnM, nu0_q, the
          mass parts of <ln lambda> and S, r_Delta
       2. per a row: a, z, D(a), the redshift parts of <ln lambda> and S,
          the Tinker 2010 parameters at alpha = 0.368 (fixed mode)
       3. per (a row, q): nu = nu0_q/D, dn_q = w (rho_hmf/M) dlnnu/dlnM
          f(nu) nu, b_q = b_h(nu) S, <ln lambda>, sigma, c(M, D), r_s
       4. per (nl, a row): P_q = [erf(x_max) - erf(x_min)]/2,
          n = sum dn_q P_q, b = sum dn_q P_q b_q / n,
          W_q = dn_q P_q (M_q/rho_m)/(m(c) n)
       5. natural splines of ln n and b across the padded a rows */

    // --- 2d. PER MASS NODE (serial: sigma2, dlognudlogm reads) ---
    const double rho_m     = cosmology.rho_crit*cosmology.Omega_m;
    const double rho_delta = CLUSTER_DELTA_HALO*rho_m;
    // mean density of the halo field in the rho/M of dn/dlnM (section
    // header): rho_m, or rho_cb under like.halo_model[4] = HALO_FIELD_CB
    const double rho_hmf   = cosmology.rho_crit*omega_halo_field();

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
                                     *(rho_hmf/m)*dlognudlogm(m);
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

      // Tinker 2010 parameters of this a row at alpha = 0.368 (read only
      // when alpha_fixed; halo.c's fnu builds its own from a)
      const cluster_tinker_params tinker = cluster_tinker_params_fixed(a);

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

        // Tinker 2010 f(nu): alpha = 0.368 (the DES convention) or
        // halo.c's alpha(a) of Eq. 7 (cluster.hmf_alpha_mode)
        const double f_nu = alpha_fixed ? cluster_tinker_fnu(nu, &tinker)
                                        : fnu(nu, a);

        // quadrature weight x dn/dlnM, in halo.c's order: the a-free
        // factor, then f(nu), then nu
        const double dn = cl_.mass_node[MN_WEIGHT][q]*f_nu*nu;

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
//     (pass B)                    every richness bin: sum_q W_nl um (the
//                                 kernel four nodes per SIMDe vector,
//                                 cluster_nfw_um4; the sums in node order)
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

#ifndef COSMO2D_NOT_USE_SIMD
        // The reference loop (the #else branch below) with the kernel on
        // four active nodes j, j+1, j+2, j+3 per step (one per lane of a
        // v4d; cluster_nfw_um4 = cluster_nfw_um on each lane, bitwise).
        // Only the kernel is vectorized: its four values go back to a
        // plain double[4] and enter the sums one node at a time, in the
        // order j, j+1, j+2, j+3 of the reference loop, so every sum[nl]
        // adds the same terms in the same order and the table is bitwise
        // the reference's. (halo.c's spectra keep one partial sum per
        // lane instead, which changes the last digits of the sums; here
        // the sums are nl_bins multiply-adds per node next to a kernel
        // with two sines, so keeping the scalar order costs nothing.)

        // k and ln k of this column in all four lanes (set1 copies one
        // scalar into every lane)
        const v4d vk   = simde_mm256_set1_pd(k);    // k
        const v4d vlnk = simde_mm256_set1_pd(lnk);  // ln k

        int j = 0;
        for (; j<=n_active-4; j+=4) {
          // the four arguments of cluster_nfw_um at nodes j..j+3; scalar:
          //   cluster_nfw_um(conc[j], k*r_s[j], lnk + lnrs[j], ln1c[j])

          // c, the concentrations of nodes j..j+3 (loadu reads four
          // consecutive doubles from memory into the lanes)
          const v4d vconc = simde_mm256_loadu_pd(conc + j);

          // r_s of nodes j..j+3
          const v4d vrs = simde_mm256_loadu_pd(r_s + j);

          // x = k r_s
          const v4d vkrs = simde_mm256_mul_pd(vk, vrs);

          // ln r_s of nodes j..j+3
          const v4d vlnrs = simde_mm256_loadu_pd(lnrs + j);

          // ln x = ln k + ln r_s
          const v4d vlnkrs = simde_mm256_add_pd(vlnk, vlnrs);

          // ln(1 + c) of nodes j..j+3
          const v4d vln1c = simde_mm256_loadu_pd(ln1c + j);

          // um = u m(c) at nodes j..j+3
          const v4d vum = cluster_nfw_um4(vconc, vkrs, vlnkrs, vln1c);

          // the four kernel values to a plain double[4] (storeu writes
          // the four lanes to memory)
          double um[4];
          simde_mm256_storeu_pd(um, vum);

          // scalar: sum[nl] += w[nl]*um, node by node in the reference
          // order (lane 0 is node j, lane 3 is node j+3)
          for (int lane=0; lane<4; lane++) {
            const double* restrict w = p1h_.weight[i][j + lane];
            for (int nl=0; nl<nl_bins; nl++) {
              sum[nl] += w[nl]*um[lane];
            }
          }
        }

        // scalar tail: n_active not a multiple of four (the reference
        // loop's body on the leftover nodes)
        for (; j<n_active; j++) {
          const double um = cluster_nfw_um(conc[j], k*r_s[j], lnk + lnrs[j],
                                           ln1c[j]);

          const double* restrict w = p1h_.weight[i][j];
          for (int nl=0; nl<nl_bins; nl++) {
            sum[nl] += w[nl]*um;
          }
        }
#else
        // the reference: one scalar kernel call per active node
        for (int j=0; j<n_active; j++) {
          // u m(c) at x = k r_s, ln x = ln k + ln r_s
          const double um = cluster_nfw_um(conc[j], k*r_s[j], lnk + lnrs[j],
                                           ln1c[j]);

          const double* restrict w = p1h_.weight[i][j];
          for (int nl=0; nl<nl_bins; nl++) {
            sum[nl] += w[nl]*um;
          }
        }
#endif

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


#ifndef COSMO2D_NOT_USE_SIMD
// ---------------------------------------------------------------------------
// cluster_load_pairs4: two neighbouring table values per lane, regrouped.
//
// A spline read needs the two nodes that bracket its point: (y_j, y_{j+1})
// and (c_j, c_{j+1}). They sit side by side in memory, so one 16-byte
// load per lane fetches a pair; the four pairs are then regrouped into a
// v4d of left values and a v4d of right values. This is the memory access
// of cluster_nfw_read4 with one difference: there the four lanes read one
// table row at four indices, here each lane has its own row as well (its
// own a node), so the caller passes the four addresses.
//
// Why no gather instruction (the limber_fill_interp idiom of cosmo2D.c):
// a gather takes one base address, and the lanes read different rows;
// paired loads also fetch both neighbours at once.
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) void cluster_load_pairs4(
    const double* const left_node[4],  // address of each lane's left value
    v4d* vleft,                        // output: left_node[l][0] on lane l
    v4d* vright                        // output: left_node[l][1] on lane l
  )
{
  // the pair of each lane, one two-double load each (loadu reads two
  // consecutive doubles from memory into a v2d)
  const v2d vpair0 = simde_mm_loadu_pd(left_node[0]);  // lane 0's pair
  const v2d vpair1 = simde_mm_loadu_pd(left_node[1]);  // lane 1's pair
  const v2d vpair2 = simde_mm_loadu_pd(left_node[2]);  // lane 2's pair
  const v2d vpair3 = simde_mm_loadu_pd(left_node[3]);  // lane 3's pair

  // unpacklo takes the first double of each pair, unpackhi the second

  // the left values of lanes 0,1
  const v2d vleft_low = simde_mm_unpacklo_pd(vpair0, vpair1);

  // the left values of lanes 2,3
  const v2d vleft_high = simde_mm_unpacklo_pd(vpair2, vpair3);

  // the right values of lanes 0,1
  const v2d vright_low = simde_mm_unpackhi_pd(vpair0, vpair1);

  // the right values of lanes 2,3
  const v2d vright_high = simde_mm_unpackhi_pd(vpair2, vpair3);

  // the left values on all four lanes (set_m128d joins the halves, low
  // first)
  *vleft = simde_mm256_set_m128d(vleft_high, vleft_low);

  // the right values on all four lanes
  *vright = simde_mm256_set_m128d(vright_high, vright_low);
}


// ---------------------------------------------------------------------------
// cluster_spline_horner4: spline_horner on four lanes, the house
// natural-spline read
//
//   S(x_j + t) = y_j + t (b + t (c_j + t d)),
//   b = (y_{j+1} - y_j)/h - h (c_{j+1} + 2 c_j)/3,  d = (c_{j+1} - c_j)/(3 h)
//
// with lane l holding one read: its own interval (the caller loaded y_j,
// y_{j+1}, c_j, c_{j+1} of that interval into lane l) and its own offset t.
//
// Why it is bitwise spline_horner: the same operations in the same order,
// the divisions kept as divisions (a vector division rounds as the scalar
// one does), and the three multiply-adds of the Horner form fused
// (cluster_fmadd4), as the compiler fuses the scalar expression. In b,
// c_{j+1} + 2 c_j is the same double fused or not (2 c_j is exact), and
// neither difference of b has a product as a direct operand, so nothing
// else can fuse.
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d cluster_spline_horner4(
    const v4d vy0,   // y_j of each lane
    const v4d vy1,   // y_{j+1}
    const v4d vc0,   // c_j = S''(x_j)/2
    const v4d vc1,   // c_{j+1}
    const v4d vt,    // offset t from x_j of each lane, 0 <= t <= h
    const double h   // grid spacing
  )
{
  // h, 2, 3 and 3 h in all four lanes (set1 copies one scalar into every
  // lane; 3.0*h is the scalar path's product, computed once)
  const v4d vh     = simde_mm256_set1_pd(h);
  const v4d vtwo   = simde_mm256_set1_pd(2.0);
  const v4d vthree = simde_mm256_set1_pd(3.0);
  const v4d v3h    = simde_mm256_set1_pd(3.0*h);

  // scalar: b = (y[j+1] - y[j])/h - h*(curv[j+1] + 2.0*curv[j])/3.0

  // y_{j+1} - y_j
  const v4d vdy = simde_mm256_sub_pd(vy1, vy0);

  // (y_{j+1} - y_j)/h, the chord slope
  const v4d vslope = simde_mm256_div_pd(vdy, vh);

  // 2 c_j
  const v4d vtwo_c0 = simde_mm256_mul_pd(vtwo, vc0);

  // c_{j+1} + 2 c_j
  const v4d vcsum = simde_mm256_add_pd(vc1, vtwo_c0);

  // h (c_{j+1} + 2 c_j)
  const v4d vh_csum = simde_mm256_mul_pd(vh, vcsum);

  // h (c_{j+1} + 2 c_j)/3
  const v4d vcurv_term = simde_mm256_div_pd(vh_csum, vthree);

  // b
  const v4d vb = simde_mm256_sub_pd(vslope, vcurv_term);

  // scalar: d = (curv[j+1] - curv[j])/(3.0*h)

  // c_{j+1} - c_j
  const v4d vdc = simde_mm256_sub_pd(vc1, vc0);

  // d
  const v4d vd = simde_mm256_div_pd(vdc, v3h);

  // scalar: y[j] + t*(b + t*(curv[j] + t*d)), innermost bracket first,
  // each t*(..) + .. fused

  // c_j + t d
  const v4d vinner = cluster_fmadd4(vt, vd, vc0);

  // b + t (c_j + t d)
  const v4d vouter = cluster_fmadd4(vt, vinner, vb);

  // y_j + t (b + t (c_j + t d))
  return cluster_fmadd4(vt, vouter, vy0);
}
#endif


// ---------------------------------------------------------------------------
// P1h_nl at n points and every richness bin in one call:
//
//   out[nl][q] = pcm_1h_richness(k[q], a[q], nl),
//   q = 0 .. n-1,  nl = 0 .. cluster.richness_nbin - 1
//
// the read of the cluster-lensing Limber integrand, whose n points are
// the quadrature nodes of one multipole: a_q along the line of sight,
// k_q = (l + 1/2)/f_K(a_q) (C_cs_tomo_limber_work of cosmo2D_cluster.c).
//
// Why a batch: a point's place on the table does not depend on the
// richness bin. ln k, the a node i with its fraction t_a and the ln k
// interval j with its offset t are found once per point and serve every
// bin, and four points are read per SIMDe vector (one per lane of a v4d).
//
// The vector body takes the groups of four points that read the table
// the ordinary way: every a inside the a grid and every k at or below
// the last ln k node. A group with a point outside the a grid (P1h = 0)
// or above the table (the power-law continuation), and the last n % 4
// points, go through pcm_1h_richness itself. Both are the scalar read,
// so every value is bitwise pcm_1h_richness's: the same operations in
// the same order per point (cluster_spline_horner4), libm log and exp on
// every lane.
//
// Thread rule of pcm_1h_richness: the first call after a key changed must
// run outside any parallel region (cluster_warmup); after it, calls from
// threaded loops only read.
//
// Parameters:
//   k   - [n] wavenumbers in (c/H0)^-1
//   a   - [n] scale factors
//   n   - number of points
//   out - [cluster.richness_nbin][>= n] output: P1h_nl(k_q, a_q) in
//         (c/H0)^3; 0 outside the a grid
// ---------------------------------------------------------------------------
void pcm_1h_richness_fill(
    const double* restrict k,
    const double* restrict a,
    const int n,
    double** out
  )
{
  const int nl_bins = cluster.richness_nbin;

  cluster_p1h_table();

#ifndef COSMO2D_NOT_USE_SIMD
  // --- table geometry: the numbers of pcm_1h_richness, cluster_p1h_row ---
  const double a_lo     = cl_.a_lim[0];   // first a node
  const double a_hi     = cl_.a_lim[1];   // last a node
  const double lnk_min  = p1h_.lnk_min;   // reads clamp below
  const double lnk_last = p1h_.lnk_last;  // reads extrapolate above
  const double h        = p1h_.dlnk;      // ln k spacing

  // the same numbers in all four lanes (set1 copies one scalar into
  // every lane)

  // a_lo
  const v4d va_lo = simde_mm256_set1_pd(a_lo);

  // the a spacing
  const v4d vh_a = simde_mm256_set1_pd(cl_.a_lim[2]);

  // n_a - 2, the index of the last a interval
  const v4d vlast_a = simde_mm256_set1_pd((double) (p1h_.n_a - 2));

  // ln k of node 0
  const v4d vlnk_first = simde_mm256_set1_pd(p1h_.lnk_first);

  // the ln k spacing
  const v4d vh = simde_mm256_set1_pd(h);

  // n_k - 2, the index of the last ln k interval
  const v4d vlast_k = simde_mm256_set1_pd((double) (p1h_.n_k - 2));

  int q = 0;
  for (; q<=n-4; q+=4) {
    // --- 1. ln k OF EACH POINT, AND THE GROUP TEST (scalar) ---
    // scalar (pcm_1h_richness): 0 outside the a grid; lnk = log(k),
    // clamped to ln k_min below the table, continued as a power law above
    // ln k_last. The libm log runs per lane, as the scalar path calls it.
    double lnk[4];
    int ordinary = 1;  // 1: every point of the group reads the table
    for (int lane=0; lane<4; lane++) {
      const double a_lane = a[q + lane];
      if (a_lane < a_lo || a_lane > a_hi) {
        ordinary = 0;  // outside the a grid
      }

      double lnk_lane = log(k[q + lane]);
      if (lnk_lane < lnk_min) {
        lnk_lane = lnk_min;  // flat below k_min
      }
      else if (lnk_lane > lnk_last) {
        ordinary = 0;  // above the table
      }
      lnk[lane] = lnk_lane;
    }

    if (0 == ordinary) {
      // the scalar read for the four points of this group
      for (int lane=0; lane<4; lane++) {
        for (int nl=0; nl<nl_bins; nl++) {
          out[nl][q + lane] = pcm_1h_richness(k[q + lane], a[q + lane], nl);
        }
      }
      continue;
    }

    // --- 2. a: NODE i AND FRACTION t_a OF EACH POINT ---
    // scalar: ra = (a - a_lo)/h_a;  i = min((int) ra, n_a - 2);
    //         t_a = ra - i
    // trunc and min in double, exact for the non-negative ra of a point
    // inside the grid (cluster_nfw_pos4 header: why not a vector
    // double-to-int conversion)

    // a of points q..q+3 (loadu reads four consecutive doubles from
    // memory into the lanes)
    const v4d va = simde_mm256_loadu_pd(a + q);

    // a - a_lo
    const v4d va_offset = simde_mm256_sub_pd(va, va_lo);

    // ra = (a - a_lo)/h_a, the position in a intervals
    const v4d vra = simde_mm256_div_pd(va_offset, vh_a);

    // trunc(ra): round toward zero, the interval number as a double
    const v4d vra_trunc = simde_mm256_round_pd(vra, SIMDE_MM_FROUND_TO_ZERO);

    // i = min(trunc(ra), n_a - 2): the last-interval clamp
    const v4d vnode_a = simde_mm256_min_pd(vra_trunc, vlast_a);

    // t_a = ra - i
    const v4d vt_a = simde_mm256_sub_pd(vra, vnode_a);

    // --- 3. ln k: INTERVAL j AND OFFSET t OF EACH POINT ---
    // scalar (cluster_p1h_row): r = (lnk - lnk_first)/h;
    //         j = min((int) r, n_k - 2);  t = (r - j)*h

    // the four clamped ln k
    const v4d vlnk = simde_mm256_loadu_pd(lnk);

    // ln k - ln k of node 0
    const v4d vlnk_offset = simde_mm256_sub_pd(vlnk, vlnk_first);

    // r = (ln k - ln k_first)/h, the position in ln k intervals
    const v4d vr = simde_mm256_div_pd(vlnk_offset, vh);

    // trunc(r)
    const v4d vr_trunc = simde_mm256_round_pd(vr, SIMDE_MM_FROUND_TO_ZERO);

    // j = min(trunc(r), n_k - 2): the last-interval clamp
    const v4d vnode_k = simde_mm256_min_pd(vr_trunc, vlast_k);

    // r - j
    const v4d vr_frac = simde_mm256_sub_pd(vr, vnode_k);

    // t = (r - j) h, the offset from node j in ln k
    const v4d vt = simde_mm256_mul_pd(vr_frac, vh);

    // i and j of each lane to plain arrays (storeu writes the four lanes
    // to memory), then to int lane by lane
    double node_a[4];
    double node_k[4];
    simde_mm256_storeu_pd(node_a, vnode_a);
    simde_mm256_storeu_pd(node_k, vnode_k);

    int index_a[4];  // a node i of each lane
    int index_k[4];  // ln k interval j of each lane
    for (int lane=0; lane<4; lane++) {
      index_a[lane] = (int) node_a[lane];
      index_k[lane] = (int) node_k[lane];
    }

    // --- 4. EVERY RICHNESS BIN AT THESE FOUR POINTS ---
    for (int nl=0; nl<nl_bins; nl++) {
      // where each lane reads: its interval j on its two a rows i and
      // i + 1, in ln P1h (y) and in the spline coefficients (c)
      const double* y_lo[4];  // &ln_p[nl][i][j]
      const double* c_lo[4];  // &curv[nl][i][j]
      const double* y_hi[4];  // &ln_p[nl][i + 1][j]
      const double* c_hi[4];  // &curv[nl][i + 1][j]
      for (int lane=0; lane<4; lane++) {
        const int ia = index_a[lane];
        const int jk = index_k[lane];

        y_lo[lane] = p1h_.ln_p[nl][ia] + jk;
        c_lo[lane] = p1h_.curv[nl][ia] + jk;
        y_hi[lane] = p1h_.ln_p[nl][ia + 1] + jk;
        c_hi[lane] = p1h_.curv[nl][ia + 1] + jk;
      }

      v4d vy0;  // y_j
      v4d vy1;  // y_{j+1}
      v4d vc0;  // c_j
      v4d vc1;  // c_{j+1}

      // scalar: ln_p_lo = cluster_p1h_row(nl, i, lnk, 0), the spline on
      // the lower a row
      cluster_load_pairs4(y_lo, &vy0, &vy1);
      cluster_load_pairs4(c_lo, &vc0, &vc1);
      const v4d vln_p_lo = cluster_spline_horner4(vy0, vy1, vc0, vc1, vt, h);

      // scalar: ln_p_hi = cluster_p1h_row(nl, i + 1, lnk, 0), the upper
      // a row
      cluster_load_pairs4(y_hi, &vy0, &vy1);
      cluster_load_pairs4(c_hi, &vc0, &vc1);
      const v4d vln_p_hi = cluster_spline_horner4(vy0, vy1, vc0, vc1, vt, h);

      // scalar: exp(ln_p_lo + t_a*(ln_p_hi - ln_p_lo)), linear in a

      // ln_p_hi - ln_p_lo
      const v4d vrise = simde_mm256_sub_pd(vln_p_hi, vln_p_lo);

      // ln_p_lo + t_a (ln_p_hi - ln_p_lo), fused as the scalar sum
      const v4d vln_p = cluster_fmadd4(vt_a, vrise, vln_p_lo);

      // the four ln P1h to a plain double[4], then the libm exp of the
      // scalar path on each
      double ln_p[4];
      simde_mm256_storeu_pd(ln_p, vln_p);

      double* restrict out_nl = out[nl];
      for (int lane=0; lane<4; lane++) {
        out_nl[q + lane] = exp(ln_p[lane]);
      }
    }
  }

  // scalar tail: n not a multiple of four
  for (; q<n; q++) {
    for (int nl=0; nl<nl_bins; nl++) {
      out[nl][q] = pcm_1h_richness(k[q], a[q], nl);
    }
  }
#else
  // the reference: the scalar read at every point
  for (int q=0; q<n; q++) {
    for (int nl=0; nl<nl_bins; nl++) {
      out[nl][q] = pcm_1h_richness(k[q], a[q], nl);
    }
  }
#endif
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
  // the one-halo table only enters cluster lensing (C_cs); its work
  // function warms pcm_1h_richness serially itself before its threaded
  // loops, so a run without cluster lensing skips this refill
  if (1 == cluster.probe_cs) {
    cluster_p1h_table();
  }

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
