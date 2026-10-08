#include <assert.h>
#include <gsl/gsl_sf.h>
#include <complex.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_integration.h>

#include "halo.h"
#include "basics.h"
#include "cosmo3D.h"
#include "redshift_spline.h"
#include "structs.h"

#include "log.c/src/log.h"

// Halo mass-node loops use SIMDe in both optimized and debug builds.
// Scalar kernels remain for individual-node calls and incomplete vectors.

// SIMDe vectors (simde/x86/avx2.h and fma.h, included by basics.h).
//
// A v4d holds four doubles side by side, its "lanes" 0, 1, 2, 3, and one
// simde_mm256_* call applies the same operation to all four lanes at
// once (one AVX2 register on x86-64, two NEON registers on arm64). A v2d
// holds two doubles: one half of a v4d, lanes 0,1 (the low half) or
// lanes 2,3 (the high half). Vector variables carry a v prefix. In the
// halo spectra the four lanes are four consecutive mass nodes q, q+1,
// q+2, q+3 of one quadrature sum, so a v4d line does what the scalar
// path's line does for one node, four nodes at a time.
typedef simde__m256d v4d;
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Halo model: peak-background split, halo and galaxy profiles, and the
// galaxy power spectra. The gas (electron-pressure) profiles and the
// spectra built from them (p_my, p_yy), and the halo-model matter spectrum
// (p_mm), are kept, not compiled, in future_port_unfinished/ (halo_tsz.c,
// halo_pmm.c).
//
// Units, shared by every routine in this file:
//
//   k        = comoving wavenumber in (c/H0)^-1
//   r, R     = comoving lengths in c/H0
//   M, m     = halo masses in M_sun/h
//   rho_crit = 3 H0^2/(8 pi G) = cosmology.rho_crit
//            = 7.4775e21 M_sun/h per (c/H0)^3
//              (2.775e11 h^2 M_sun/Mpc^3 times 2997.92^3)
//   P(k)     = (c/H0)^3
//
// A halo of mass M is labeled by its peak height
//
//   nu(M, a) = delta_c/sigma_cb(M, a).
//
// sigma2(M,a) integrates P_cb at that scale factor, with Lagrangian
// radius R = (3M/(4 pi rho_cb))^(1/3). No single growth factor can
// evolve all halo masses when neutrinos free-stream. This is the nu
// of Tinker et al. 2010 (1001.3162 sec. 2), not the squared peak height
// of Cooray & Sheth 2002 (astro-ph/0206508 Eq. 57).
//
// The two constants below:
//
//   delta_c          = 1.686, the linear collapse threshold of
//                      spherical collapse, (3/20)(12 pi)^(2/3); the
//                      value the Tinker fits assume (1001.3162 sec. 2)
//   Delta            = 200, the halo overdensity with respect to the
//                      mean matter density,
//                      M = (4 pi/3) R_Delta^3 Delta rho_m
//                      (1001.3162 Eq. 1). With the comoving mean
//                      density rho_m = rho_crit Omega_m, R_Delta is
//                      comoving and a-independent. The mass function,
//                      the bias and the concentration below all use
//                      this halo definition.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Glossary: the function names of this file are terse. Each one maps to
// a halo-model quantity:
//
// Halo demographics (bias and mass function: Tinker et al. 2010,
// 1001.3162; concentration: Bhattacharya et al. 2013, 1112.5479; these
// are the defaults, and like.halo_model selects the alternatives listed
// in the hb1nu, fnu and conc headers):
//
//   hb1nu        = b_1(nu), the linear (first-order) halo bias as a
//                  function of the peak height nu: "h" halo, "b1" bias of
//                  first order, "nu" its variable
//   fnu          = f(nu), the halo multiplicity function: the mass
//                  function per unit nu
//   tinker_alpha = alpha(a), the amplitude of f(nu), fixed at every a
//                  by int b_1 f dnu = 1 (a table in a)
//   conc         = c(M), the NFW concentration r_Delta/r_s
//   dlognudlogm  = d ln nu/d ln M, the Jacobian from nu to halo mass
//   bias_norm    = int b_1 f dnu over the tabulated mass range;
//                  1 - bias_norm is the share of the matter that range
//                  misses (a table in a)
//   *_params_at,
//   *_core       = the two halves of hb1nu and fnu: the nu-independent
//                  coefficients and the nu-dependent remainder
//
// Profiles, in Fourier space ("u" = a profile transform):
//
//   u_nfw_c      = u(k|M) of the NFW profile, given its concentration c
//   u_g          = u_g(k|M), the satellite-galaxy profile: u_nfw at
//                  c_g = gc[ni] c(M) (computed inline by p_gm, p_gg)
//   (the gas profiles u_KS, frac_bnd, frac_ejc, u_y_ejc and the window
//   W_p are in future_port_unfinished/halo_tsz.c)
//
// Halo occupation distribution, HOD (Zehavi et al. 2011, 1005.2413):
//
//   HOD_nc       = <N_c|M>, mean number of central galaxies
//   HOD_ns       = <N_s|M>, mean number of satellite galaxies
//   HOD_fc       = f_c, completeness factor of the centrals
//   ngal         = n_g, comoving galaxy number density
//   bgal         = b_g, number-weighted mean galaxy bias
//   set_HOD      = built-in HOD values for lens bin ni: fills
//                  nuisance.hod[ni][0..5], sets nuisance.gc[ni] = 1 and
//                  stores b_g in nuisance.gb[0][ni]
//
// Halo-model intrinsic alignment (Fortuna et al. 2021, 2003.02700;
// HALO-MODEL INTRINSIC ALIGNMENT banner), one IA population over the
// source redshift range:
//
//   ia_f_red_central = f_rc(a), the red-central fraction that scales the
//                      NLA (2-halo) alignment of the centrals
//   ia_p1h_dI        = a_1h f_1h(k) S_dI(k, a), the satellites' 1-halo
//                      matter-intrinsic power (signed with a_1h)
//   ia_p1h_II        = a_1h^2 f_1h(k) S_II(k, a), their 1-halo
//                      intrinsic-intrinsic power
//   ia_window_2h     = f_2h(k), the window on the 2-halo IA power
//
// Halo-model integrals, named after the I^beta_mu of Cooray & Hu 2001
// (astro-ph/0012087 Eq. 12, whose delta_halo(k, M) is (M/rho_m) u(k|M);
// Cooray & Sheth 2002, astro-ph/0206508 sec. 4.2, write the same
// integral as M_ij):
//
//   I^beta_mu(k_1 .. k_mu) = int dM n(M) b_beta(M) (M/rho_m)^mu
//                                 u(k_1|M) ... u(k_mu|M)
//
//   beta = order of the halo bias (0 = none, 1 = linear)
//   mu   = number of profiles in the integrand
//
// so that P(k) = I^0_2(k, k) + [I^1_1(k)]^2 P_lin(k) (astro-ph/0012087
// Eqs. 14-15):
//
//   I02          = I^0_2: no bias, two profiles  -> the 1-halo term of
//                  P_XY
//   I11          = I^1_1: linear bias, one profile -> the 2-halo
//                  amplitude, P_2h = I11_X I11_Y P_lin, plus the HMx
//                  term for the halos below M_min (POWER SPECTRA banner)
//   (the 1-halo sums of p_gm and p_gg: the galaxy-matter and the
//   galaxy-galaxy pairs of one halo, divided by n_g and n_g^2)
//
// Spectra: p_gm(k, a) and p_gg(k, a) (m = matter, g = galaxies). The
// halo-model matter spectrum p_mm and the y = electron-pressure spectra
// p_my, p_yy are kept, not compiled, in future_port_unfinished/
// (halo_pmm.c, halo_tsz.c).
//
// Suffixes:
//
//   *_work       = batched computation over many inputs
//   int_for_*,
//   int_*        = integrand of a mass or radius integral
//   (no suffix)  = the cached table, read by interpolation
// ---------------------------------------------------------------------------

#define delta_c 1.686
#define Delta 200



// ============================================================================
// [SECTION] BASIC PEAK BACKGROUND SPLIT ROUTINES
// ============================================================================
//
// The halo model needs three functions of the halo mass: how many halos
// there are (the mass function), how they cluster on large scales (the
// linear bias) and how concentrated they are. In the peak height nu the
// first two are nearly universal (Tinker et al. 2010, 1001.3162):
//
//   f(nu) dnu = fraction of all matter in halos with peak height in
//               [nu, nu + dnu]                          -> fnu
//   b(nu)     = large-scale linear bias of those halos  -> hb1nu
//
// and the mass function follows by the change of variable nu -> M:
//
//   dn/dlnM = (rho_cb/M) nu f(nu) dln nu/dln M           -> dlognudlogm
//
// bias_norm measures how much of the consistency relation int b f dnu = 1
// the finite mass range of the halo-model integrals covers; a halo-model
// matter spectrum adds the rest, 1 - bias_norm, back as halos of mass M_min
// (future_port_unfinished/halo_pmm.c). conc gives the NFW concentration,
// a function of nu for the default fit (of M and z for Duffy et al. 2008).


// ---------------------------------------------------------------------------
// hb1nu and fnu are each split in two, so that a loop over many nu at
// one scale factor computes the nu-independent part once:
//
//   *_params_at(a)   = the fit coefficients at a (dispatch on
//                      like.halo_model)
//   *_core(nu, par)  = the nu-dependent remainder, Eq. 6 or Eq. 8
//
// hb1nu(nu, a) and fnu(nu, a) are params_at followed by core; a batched
// caller and a scalar caller get the same values bitwise.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// hb1nu = b_1(nu): the linear (first-order) halo bias as a function
// of the peak height nu, from Tinker et al. 2010 (1001.3162 Eq. 6):
//
//   b(nu) = 1 - A nu^a/(nu^a + delta_c^a) + B nu^b + C nu^c
//
// 1001.3162 Table 2 gives the six coefficients as functions of
// y = log10(Delta); at Delta = 200 (mean density, see the file header):
//
//   A = 1 + 0.24 y exp[-(4/y)^4]              = 1.00006
//   a = 0.44 y - 0.88                         = 0.13245
//   B = 0.183,  b = 1.5
//   C = 0.019 + 0.107 y + 0.19 exp[-(4/y)^4]  = 0.26523
//   c = 2.4
//
// (hb1nu_params holds A, a, delta_c^a, B, C; b and c are literals in
// hb1nu_core.) b -> 1 as nu -> 0, slowly because a is small (b ~ 0.6
// at nu = 0.2); b ~ 1 near nu = 1; the C nu^2.4 term takes over for
// rare, massive halos (b ~ 5 at nu = 3). The fit has no redshift
// dependence: it combines all outputs 0 <= z <= 2.5, over which the
// evolution at fixed nu is at most very weak (1001.3162 sec. 3.1). The
// scale factor stays in the signature so that a z-dependent fit could
// use it.
//
// Parameters:
//   nu - peak height delta_c/sigma(M, a)
//   a  - scale factor (unused by the Tinker fit)
//
// Returns:
//   b(nu), dimensionless. like.halo_model[1] selects the fit:
//   HALO_BIAS_TINKER_2010 (the default) or
//   HALO_BIAS_SHETH_MO_TORMEN_2001; other values abort.
//
// HALO_BIAS_SHETH_MO_TORMEN_2001: Sheth, Mo & Tormen 2001
// (astro-ph/9907024, Eq. 8) with their a = 0.707, b = 0.5, c = 0.6.
// Writing x = a nu^2,
//
//   b(nu) = 1 + [ sqrt(a) x + sqrt(a) b x^(1-c)
//                 - x^c / (x^c + b (1-c)(1-c/2)) ] / (sqrt(a) delta_c).
//
// Like the Tinker fit it has no redshift dependence at fixed nu. It was
// calibrated on virial spherical-overdensity halos, so at this file's
// Delta = 200 rho_mean definition it is an approximation, provided for
// cross-code comparison (CCL applies it at this definition only when its
// strict mass-definition check is relaxed, as TJPCov does); it does not
// replace the Delta-matched Tinker default.
// ---------------------------------------------------------------------------
typedef struct {
  int model;    // like.halo_model[1]: selects which fields below apply
  // HALO_BIAS_TINKER_2010 (Eq. 6 of 1001.3162):
  double ALPHA; // A of Eq. 6 (1001.3162)
  double pa;    // exponent a = 0.44 y - 0.88, y = log10(Delta)
  double dca;   // delta_c^a
  double BETA;  // B = 0.183
  double GAMMA; // C
  // HALO_BIAS_SHETH_MO_TORMEN_2001 (Eq. 8 of astro-ph/9907024):
  double SMT_A;   // a = 0.707, multiplying nu^2 in x = a nu^2
  double SMT_SA;  // sqrt(a)
  double SMT_SAB; // sqrt(a) b, the x^(1-c) coefficient
  double SMT_BC;  // b (1-c)(1-c/2), the denominator offset
  double SMT_INV; // 1/(sqrt(a) delta_c), the overall bracket factor
} hb1nu_params;

#if Delta != 200
#error "hb1nu_params_at: the Tinker bias literals assume Delta = 200"
#endif
static inline hb1nu_params hb1nu_params_at(
    const double a __attribute__((unused)) // scale factor (unused: the
                                           // Delta = 200 fit is z-free)
  )
{
  hb1nu_params p;
  switch(like.halo_model[1])
  {
    case HALO_BIAS_TINKER_2010:
    {
      // A, a, delta_c^a, B, C of the header (Table 2 of 1001.3162 at
      // y = log10(200)), as literals so nothing is recomputed per call.
      p.model = HALO_BIAS_TINKER_2010;
      p.ALPHA = 1.00005974393421592059;
      p.pa    = 0.132453198092151725894;
      p.dca   = 1.07163776686581864305;
      p.BETA  = 0.183;
      p.GAMMA = 0.265230764366423426079;
      break;
    }
    case HALO_BIAS_SHETH_MO_TORMEN_2001:
    {
      // a, sqrt(a), sqrt(a) b, b(1-c)(1-c/2) and 1/(sqrt(a) delta_c)
      // of the header, as literals (mpmath, 21 digits) so nothing is
      // recomputed per call; delta_c = 1.686 as everywhere in this file.
      p.model   = HALO_BIAS_SHETH_MO_TORMEN_2001;
      p.SMT_A   = 0.707;
      p.SMT_SA  = 0.840832920383116303209;
      p.SMT_SAB = 0.420416460191558151604;
      p.SMT_BC  = 0.14;
      p.SMT_INV = 0.705395561738249015697;
      break;
    }
    default:
    {
      log_fatal("like.halo_model[1] = %d not supported", like.halo_model[1]);
      exit(1);
    }
  }
  return p;
}


static inline double hb1nu_core(
    const double nu,        // peak height delta_c/sigma_cb(M,a)
    const hb1nu_params* p   // coefficients from hb1nu_params_at
  )
{
  if (p->model == HALO_BIAS_SHETH_MO_TORMEN_2001)
  {
    // Eq. 8 of the header with x = a nu^2; 0.6 and 0.4 = 1 - c are the
    // fit's own exponents, independent of the halo-mass definition.
    const double x  = p->SMT_A*nu*nu;
    const double xc = pow(x, 0.6);
    return 1.0 + p->SMT_INV*(p->SMT_SA*x + p->SMT_SAB*pow(x, 0.4)
                             - xc/(xc + p->SMT_BC));
  }
  // Eq. 6 with nu_alpha = nu^a, nu_beta = nu^b (b = 1.5) and
  // nu_gamma = nu^c (c = 2.4).
  const double nu_alpha = pow(nu, p->pa);
  const double nu_beta  = pow(nu, 1.5);
  const double nu_gamma = pow(nu, 2.4);
  return 1.0 - p->ALPHA * nu_alpha / (nu_alpha + p->dca)
             + p->BETA * nu_beta + p->GAMMA * nu_gamma;
}


double hb1nu(
    const double nu, // peak height delta_c/sigma_cb(M,a)
    const double a   // scale factor (unused by the Tinker bias fit)
  )
{
  const hb1nu_params p = hb1nu_params_at(a);
  return hb1nu_core(nu, &p);
}


// ---------------------------------------------------------------------------
// fnu = f(nu): the halo multiplicity function, i.e. the mass function
// per unit peak height nu, of Tinker et al. 2010 (1001.3162 Eq. 8):
//
//   f(nu) = alpha [1 + (beta nu)^(-2 phi)] nu^(2 eta) exp(-gamma nu^2/2)
//
// It turns into the halo mass function through
//
//   dn/dM = f(nu) (rho_cb/M) dnu/dM
//     ->  dn/dlnM = (rho_cb/M) nu f(nu) dln nu/dln M,
//
// so nu f(nu) is the g(sigma) of Tinker et al. 2008 (1001.3162 sec. 4):
// the normalized form of their App. C, not the Eq. 3 f(sigma) that the
// HMF_TINKER_2008 option below evaluates.
//
// Shape parameters (fnu_params.beta, .gamma, .phi, .eta). 1001.3162
// Table 4 gives the z = 0 values at Delta = 200 (mean density),
//
//   beta_0 = 0.589, gamma_0 = 0.864, phi_0 = -0.729, eta_0 = -0.243,
//
// and Eqs. 9-12 evolve them, with 1 + z = 1/a:
//
//   beta  = beta_0  (1+z)^0.20   = 0.589  a^-0.20
//   phi   = phi_0   (1+z)^-0.08  = -0.729 a^0.08
//   eta   = eta_0   (1+z)^0.27   = -0.243 a^-0.27
//   gamma = gamma_0 (1+z)^-0.01  = 0.864  a^0.01
//
// At small nu the bracket tends to 1 (-2 phi = 1.46 > 0) and f is a
// power law that grows toward light halos, f -> alpha nu^(2 eta) =
// alpha nu^-0.49 at z = 0; at large nu the Gaussian exp(-gamma nu^2/2)
// makes massive halos (nu > 2) exponentially few. Beyond z = 3 the
// paper recommends the z = 3 parameters (text after Eq. 12):
// everything is evaluated at aa = max(a, 0.25).
//
// Amplitude (fnu_params.alpha). alpha is not fitted: the peak-background
// split fixes it at each z through the consistency relation
// (1001.3162 Eq. 7)
//
//   int_0^inf b(nu) f(nu) dnu = 1,
//
// with b(nu) the linear halo bias of hb1nu: the matter-weighted mean
// bias of all halos is the bias of matter with respect to itself, 1.
// This is what makes the 2-halo term P_2h = I11_m^2 P_lin tend to
// P_lin as k -> 0. Since alpha factors out of f, Eq. 7 gives it
// directly,
//
//   alpha(a) = 1 / int_0^inf b(nu) ftilde(nu; a) dnu,
//   ftilde   = f with alpha = 1,
//
// which tinker_alpha tabulates in aa: alpha = 0.3684 at z = 0 (Table 4
// lists 0.368), falling to 0.2520 for z >= 3. The other natural
// normalization, int f dnu = 1 (all matter in halos), is not imposed
// and holds only approximately.
//
// Parameters:
//   nu - peak height delta_c/sigma(M, a)
//   a  - scale factor, 0 < a < 1 (aborts otherwise)
//
// Returns:
//   f(nu), dimensionless, per unit nu. like.halo_model[0] selects the
//   fit: HMF_TINKER_2010 (the default) or HMF_TINKER_2008; other
//   values abort.
//
// HMF_TINKER_2008: Tinker et al. 2008 (0803.2706, Eqs. 3 and 5-8,
// Table 2 at Delta = 200 rho_mean, this file's halo definition),
//
//   f_T08(sigma) = A [ (sigma/b)^-q + 1 ] exp(-c/sigma^2),
//   A(z) = 0.186 (1+z)^-0.14,   q(z) = 1.47 (1+z)^-0.06,
//   b(z) = 2.57 (1+z)^-alpha,   log10 alpha = -(0.75/log10(200/75))^1.2,
//   c = 1.19,
//
// written with q for the paper's exponent a, to keep a for the scale
// factor; (1+z)^-x = a^x. The paper's own redshift scaling is used at
// every a, as CCL and TJPCov do; the fit was calibrated at z <= 2.5.
//
// Convention bridge: this file's f(nu) is per unit nu, with
// dn/dlnM = (rho/M) nu f(nu) dln(nu)/dlnM, while Tinker 2008 defines
// dn/dlnM = (rho/M) f_T08(sigma) dln(1/sigma)/dlnM. Since
// dln(1/sigma) = dln(nu) at fixed delta_c, the two agree exactly when
//
//   f(nu) = f_T08(delta_c/nu) / nu,
//
// which is what fnu_core returns for this model. The amplitude A is
// the paper's own: no tinker_alpha table and no bias-weighted Eq.-7
// normalization enter this option.
// ---------------------------------------------------------------------------
typedef struct {
  int model;    // like.halo_model[0]: selects which fields below apply
  // HMF_TINKER_2010 (Eq. 8 of 1001.3162):
  double alpha; // amplitude: 1 from fnu_shape, Eq. 7 from tinker_alpha
  double beta;  // the four shape parameters, Eqs. 9-12 + Table 4 of
  double gamma; //   1001.3162 at Delta = 200
  double phi;
  double eta;
  // HMF_TINKER_2008 (Eqs. 3, 5-8 of 0803.2706 at Delta = 200), at the
  // requested scale factor:
  double t08_amp; // A(z)
  double t08_q;   // the sigma exponent q(z) (the paper's a)
  double t08_b;   // b(z)
  double t08_c;   // c = 1.19
} fnu_params;

// The four shape parameters of Eq. 8 at aa (Eqs. 9-12) with alpha = 1:
// the ftilde of the Eq. 7 integral. No clamp on aa: tinker_alpha calls
// this beyond [0.25, 1] at its padding nodes; fnu_params_at clamps.
static inline fnu_params fnu_shape(
    const double aa  // scale factor of the Tinker evolution (no clamp)
  )
{
  fnu_params p;
  p.model = HMF_TINKER_2010; // fnu_shape is the Tinker 2010 shape
  p.alpha = 1.0;
  p.beta  = 0.589 * pow(aa, -0.2);    // Eq. 9:  beta_0  (1+z)^0.20
  p.gamma = 0.864 * pow(aa, 0.01);    // Eq. 12: gamma_0 (1+z)^-0.01
  p.phi   = -0.729 * pow(aa, .08);    // Eq. 10: phi_0   (1+z)^-0.08
  p.eta   = -0.243 * pow(aa, -0.27);  // Eq. 11: eta_0   (1+z)^0.27
  return p;
}


// Eq. 8 itself, the nu-dependent remainder:
//
//   f(nu) = alpha [1 + (beta nu)^(-2 phi)] nu^(2 eta) exp(-gamma nu^2/2)
static inline double fnu_core(
    const double nu,      // peak height delta_c/sigma_cb(M,a)
    const fnu_params* p   // parameters from fnu_params_at or fnu_shape
  )
{
  if (p->model == HMF_TINKER_2008)
  {
    // The header's convention bridge: evaluate the sigma-form fit at
    // sigma = delta_c/nu and divide by nu, so nu f(nu) = f_T08(sigma).
    const double sigma = delta_c/nu;
    return p->t08_amp*(pow(sigma/p->t08_b, -p->t08_q) + 1.0)*
           exp(-p->t08_c/(sigma*sigma))/nu;
  }
  return p->alpha*(1. + pow(p->beta*nu,-2*p->phi))*pow(nu,2*p->eta)*
         exp(-p->gamma*nu*nu/2.);
}


// ---------------------------------------------------------------------------
// tinker_alpha = alpha(aa): the amplitude of the Tinker et al. 2010
// multiplicity function that satisfies 1001.3162 Eq. 7 at the scale
// factor aa (fnu header),
//
//   alpha(aa) = 1/I(aa),   I(aa) = int_0^inf b(nu) ftilde(nu; aa) dnu,
//
// with b the linear bias of hb1nu_core (the Tinker bias by default,
// the like.halo_model[1] fit otherwise) and ftilde the Eq. 8 shape at
// alpha = 1 (fnu_shape). The bias option therefore also sets the
// amplitude of this mass function: with HALO_BIAS_SHETH_MO_TORMEN_2001,
// alpha = 0.321 at z = 0 instead of 0.368. alpha depends on aa alone,
// so it is tabulated once and read by linear interpolation.
//
// The integral, in s = ln nu (dnu = nu ds), as a trapezoid sum:
//
//   I(aa) = int b ftilde nu ds ~ sum_q bias_weight[q] ftilde(nu_q; aa)
//
// The integrand decays exponentially at both ends (nu^(1 + 2 eta) with
// 1 + 2 eta > 0 toward small nu, exp(-gamma nu^2/2) toward large nu),
// so the trapezoid rule on s = -90 .. 3.5 in steps DS = 0.1 (nu from
// 8e-40 to 33) converges far faster than its nominal h^2. Only ftilde
// depends on aa; the rest is folded into one weight per node.
//
// Map of the build:
//
//   nu_node[q]      = nu_q = e^(s_q)              (NS trapezoid nodes)
//   bias_weight[q]  = w_q nu_q b(nu_q), w_q = DS (DS/2 at the two ends)
//   alpha_coarse[i] = alpha(aa_i) = 1/I(aa_i) at the coarse nodes
//                     aa_i = aa0 + i hc: NC on [0.25, 1] plus PAD
//                     beyond each end (NE = NC + 2 PAD)
//   spline_curv[j]  = S''(aa_j)/2 of the natural cubic spline S
//                     through the coarse alpha values
//   table[i]        = S(aa) at the ND dense nodes aa = 0.25 + i lim[2]
//
// The PAD nodes put the spline's S'' = 0 end condition, wrong for the
// curved alpha(aa), six intervals outside [0.25, 1], where its effect
// has died out; this is why fnu_shape is called outside [0.25, 1] and
// carries no clamp.
//
// Accuracy: alpha(aa) to 5e-8 relative over [0.25, 1].
//
// Cache invalidation:
//   rebuilt when like.halo_model[0] or [1] (the mass-function and the
//   bias fit) differ from the pair the table holds, or when
//   Ntable.random changes (the node counts Ntable.halo_hmf_n and
//   Ntable.halo_spline_pad). The cosmology does not enter (nu is the
//   integration variable).
//
// Thread safety: the first call builds the table and must run outside
// any parallel region.
//
// Parameters:
//   aa - scale factor of the Tinker evolution, max(a, 0.25), in
//        [0.25, 1)
//
// Returns:
//   alpha(aa), dimensionless; the end values for aa outside [0.25, 1]
// ---------------------------------------------------------------------------
static double tinker_alpha(
    const double aa  // max(a, 0.25), in [0.25, 1)
  )
{
  // The table and the pair of fits it holds; NULL and {-1, -1} make
  // the first call build.
  static int key[2] = {-1, -1};  // like.halo_model[0..1] of the table
  static uint64_t ntable_key = 0; // Ntable.random of the table (sizes)
  static double* table = NULL;   // [ND] alpha on the dense aa grid
  static double lim[3];          // aa_min, aa_max, dense spacing
  static int ND = 0;             // dense lookup nodes on [0.25, 1]

  // Build block: the first call, or a change of the fits in use.
  if (NULL == table ||
      key[0] != like.halo_model[0] ||
      key[1] != like.halo_model[1] ||
      fdiff2(ntable_key, Ntable.random))
  {
    ND = Ntable.halo_hmf_n[NODES_DENSE][like.halo_model[0]];

    /* PHYSICAL DERIVATION & LOGIC FLOW
       1. trapezoid in s = ln nu: bias_weight[q] = w_q nu_q b(nu_q)
       2. alpha_coarse[i] = 1/sum_q bias_weight[q] ftilde(nu_q; aa_i)
       3. natural cubic spline through alpha_coarse -> table[] on the
          dense aa grid (full derivation: the header above) */

    // --- 1. COARSE PADDED aa NODES ---
    // aa_i = aa0 + i hc: NC exact nodes on [0.25, 1] plus PAD beyond
    // each end; 0.75 below is the width of the [0.25, 1] range.
    const int NC  = Ntable.halo_hmf_n[NODES_COARSE][like.halo_model[0]]; // exact nodes
    const int PAD = Ntable.halo_spline_pad; // exact nodes beyond each end
    const int NE  = NC + 2*PAD;
    const double hc  = 0.75/((double) NC - 1.0);
    const double aa0 = 0.25 - PAD*hc;

    // --- 2. TRAPEZOID RULE IN s = ln nu ---
    // Nodes s_q = SMIN + q DS (NS = 936). bias_weight[q] folds the
    // trapezoid weight w_q, the Jacobian of dnu = nu ds and the bias
    // b(nu_q) of hb1nu_core: only ftilde still depends on aa. The bias
    // fit does not evolve, so hb1nu_params_at takes any a; 1.0 is a
    // placeholder.
    const double SMIN = -90.0;  // trapezoid range in s = ln nu
    const double SMAX = 3.5;
    const double DS   = 0.1;    // trapezoid step
    const int NS = (int) lround((SMAX - SMIN)/DS) + 1;

    double* nu_node     = (double*) malloc(sizeof(double)*NS);
    double* bias_weight = (double*) malloc(sizeof(double)*NS);
    const hb1nu_params bias_par = hb1nu_params_at(1.0);

    for (int q=0; q<NS; q++) {
      nu_node[q] = exp(SMIN + q*DS);

      // trapezoid weight: DS inside the range, DS/2 at the two ends
      double wtrap = DS;
      if (0 == q || NS - 1 == q) {
        wtrap = 0.5*DS;
      }
      bias_weight[q] = wtrap*nu_node[q]*hb1nu_core(nu_node[q], &bias_par);
    }

    // --- 3. EXACT ALPHA AT EACH COARSE NODE ---
    // alpha_coarse[i] = 1/I(aa_i), I = sum_q bias_weight[q] ftilde(nu_q)
    double* alpha_coarse = (double*) malloc(sizeof(double)*NE);
    double* spline_curv  = (double*) malloc(sizeof(double)*NE);

    const double* restrict nu_q = nu_node;
    const double* restrict bw_q = bias_weight;

    #pragma omp parallel for schedule(static)
    for (int i=0; i<NE; i++) {
      const fnu_params shape_par = fnu_shape(aa0 + i*hc);

      double sum = 0.0;
      for (int q=0; q<NS; q++) {
        sum += bw_q[q]*fnu_core(nu_q[q], &shape_par);
      }
      alpha_coarse[i] = 1.0/sum;
    }

    // --- 4. NATURAL CUBIC SPLINE THROUGH THE COARSE VALUES ---
    // spline_curv[j] = S''(aa_j)/2, zero at the two padding ends
    spline_coeffs_uniform(alpha_coarse, NE, hc, spline_curv);

    // --- 5. DENSE LOOKUP TABLE ---
    // ND nodes uniform on [0.25, 1], both ends included.
    if (table != NULL) {
      free(table);
    }
    table = (double*) malloc(sizeof(double)*ND);
    lim[0] = 0.25;
    lim[1] = 1.0;
    lim[2] = (lim[1] - lim[0])/((double) ND - 1.0);

    // table[i] = S(aa) at dense node aa = 0.25 + i lim[2], which lies
    // r coarse spacings from aa0: coarse interval j, offset t (in aa)
    // from coarse node j. The cubic on interval j, in Horner form:
    //
    //   S(aa_j + t) = y_j + t (b + t (c_j + t d)),
    //   b = (y_{j+1} - y_j)/hc - hc (c_{j+1} + 2 c_j)/3,
    //   d = (c_{j+1} - c_j)/(3 hc),
    //
    // with y = alpha_coarse and c = spline_curv. (The clamp on j is a
    // guard only: the last dense node has r = 133.)
    for (int i=0; i<ND; i++) {
      const double r = (lim[0] + i*lim[2] - aa0)/hc;

      int j = (int) r;
      if (j > NE - 2) {
        j = NE - 2;
      }

      const double t = (r - j)*hc;
      const double b = (alpha_coarse[j+1] - alpha_coarse[j])/hc
                       - hc*(spline_curv[j+1] + 2.0*spline_curv[j])/3.0;
      const double d = (spline_curv[j+1] - spline_curv[j])/(3.0*hc);
      table[i] = alpha_coarse[j] + t*(b + t*(spline_curv[j] + t*d));
    }

    free(nu_node);
    free(bias_weight);
    free(alpha_coarse);
    free(spline_curv);

    // --- 6. RECORD THE FITS THE TABLE HOLDS ---
    key[0] = like.halo_model[0];
    key[1] = like.halo_model[1];
    ntable_key = Ntable.random;
  }

  return interpol1d(table, ND, lim[0], lim[1], lim[2], aa);
}


static inline fnu_params fnu_params_at(
    const double a  // scale factor, 0 < a < 1 (aborts otherwise)
  )
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true");
    exit(1);
  }

  fnu_params p;
  switch(like.halo_model[0])
  {
    case HMF_TINKER_2010:
    {
      // Eqs. 8-12 + Table 4 of 1001.3162 at aa = max(a, 0.25): the
      // evolution is frozen at z = 3, as the paper recommends, and shape
      // and amplitude are both taken at aa, so Eq. 7 keeps holding.
      const double aa = fmax(0.25, a);
      p = fnu_shape(aa);          // beta, gamma, phi, eta; alpha = 1
      p.alpha = tinker_alpha(aa); // alpha from Eq. 7 (a table read)
      break;
    }
    case HMF_TINKER_2008:
    {
      // A(z), q(z), b(z) and c of the header, with (1+z)^-x = a^x and
      // the Delta = 200 exponent alpha = 0.0106756286522959060767
      // (its log10 is -1.97160654139105438701; mpmath, 21 digits).
      // The paper's own amplitude: nothing reads tinker_alpha here.
      p.model   = HMF_TINKER_2008;
      p.t08_amp = 0.186*pow(a, 0.14);
      p.t08_q   = 1.47*pow(a, 0.06);
      p.t08_b   = 2.57*pow(a, 0.0106756286522959060767);
      p.t08_c   = 1.19;
      break;
    }
    default:
    {
      log_fatal("like.halo_model[0] = %d not supported", like.halo_model[0]);
      exit(1);
    }
  }
  return p;
}


double fnu(
    const double nu, // peak height delta_c/sigma_cb(M,a)
    const double a   // scale factor, 0 < a < 1 (aborts otherwise)
  )
{
  const fnu_params p = fnu_params_at(a);
  return fnu_core(nu, &p);
}


// ---------------------------------------------------------------------------
// Halo concentration c = r_Delta/r_s of the NFW profile, Bhattacharya et
// al. 2013 (1112.5479 Table 2, full halo sample, Delta = 200 times the
// mean density: the halo definition of this file):
//
//   c(M,a) = 9.0 nu^-0.29 D_cb(M,a)^1.15,
//   nu = delta_c/sigma_cb(M,a), D_cb = sigma_cb(M,a)/sigma_cb(M,1).
//
// The same cold variance sets both the peak height and its growth.
// This extends the massless fit consistently to scale-dependent growth;
// it is not a separate calibration of concentration with neutrinos.
//
// The fit is calibrated at z = 0-2 and group-to-cluster masses, with
// delta_c = 1.673 where this file uses 1.686 (a -0.2% shift in c). The
// halo model evaluates it over all of [limits.halo_m[RANGE_MIN],
// limits.halo_m[RANGE_MAX]] and at every z, so also in extrapolation.
//
// Parameters:
//   m         - halo mass in M_sun/h
//   a - scale factor, limits.a_min <= a <= 1
//
// Returns:
//   c, dimensionless. like.halo_model[2] selects the fit:
//   CONCENTRATION_BHATTACHARYA_2013 (the default) or
//   CONCENTRATION_DUFFY_2008; other values abort.
//
// CONCENTRATION_DUFFY_2008: Duffy et al. 2008 (0804.2486 Table 1, FULL
// halo sample, mean-200 row: the same Delta = 200 rho_mean definition),
//
//   c(M,a) = 10.14 (M/M_piv)^-0.081 a^1.01,   M_piv = 2e12 Msun/h,
//
// with a^1.01 = (1+z)^-1.01. A pure (M, z) fit from massless N-body
// runs at z = 0-2: no growth factor enters, so with massive neutrinos
// it has no cb extension (unlike the Bhattacharya default's D_cb^1.15).
// OneCov and TJPCov (through CCL) use exactly this relation.
// ---------------------------------------------------------------------------
double conc(
    const double m,         // halo mass in M_sun/h
    const double a          // scale factor
  )
{
  double c;
  switch(like.halo_model[2])
  {
    case CONCENTRATION_BHATTACHARYA_2013:
    {
      // Bhattacharya et al. 2013, Delta = 200 rho_{mean} (Table 2, full
      // halo sample): c = 9.0 nu^-0.29 D^1.15
      const double variance = sigma2(m, a);
      const double growth_cb = sqrt(variance/sigma2(m, 1.0));
      const double nu = delta_c/sqrt(variance);
      c = 9.0*pow(nu, -0.29)*pow(growth_cb, 1.15);
      break;
    }
    case CONCENTRATION_DUFFY_2008:
    {
      // Duffy et al. 2008, Delta = 200 rho_mean (Table 1, full sample):
      // c = 10.14 (M/2e12)^-0.081 (1+z)^-1.01, and (1+z)^-1.01 = a^1.01.
      c = 10.14*pow(m/2.0e12, -0.081)*pow(a, 1.01);
      break;
    }
    default:
    {
      log_fatal("like.halo_model[2] = %d not supported", like.halo_model[2]);
      exit(1);
    }
  }
  return c;
}


// ---------------------------------------------------------------------------
// Cached bias_norm(a): the bias-weighted multiplicity integral over the
// tabulated mass range,
//
//   bias_norm(a) = int_{nu_min(a)}^{nu_max(a)} b(nu) f(nu, a) dnu,
//   nu(M, a)     = delta_c / sigma_cb(M,a),
//
// with b the linear halo bias (hb1nu), f the multiplicity function
// (fnu), and nu_min, nu_max the cold-field peak heights of
// limits.halo_m[RANGE_MIN] and limits.halo_m[RANGE_MAX].
//
// Why the 2-halo term needs it: matter is unbiased with respect to
// itself, int b f dnu = 1 over all nu, so P_2h = I11_m^2 P_lin (file
// glossary) tends to P_lin as k -> 0. The mass integrals of this file
// stop at M_min, and f grows toward light halos. Below M_min = 1e4
// M_sun/h, halos hold about 0.17 of the integral at z = 0 (0.28 at
// z = 1) for a Planck-like cosmology, so I11_m(k -> 0) would be ~0.83
// and P_2h -> 0.68 P_lin. The I11 sum of a halo-model matter spectrum
// (future_port_unfinished/halo_pmm.c) adds the missing 1 - bias_norm(a)
// back as halos of mass M_min; this function measures the shortfall.
//
// At each a, the lower and upper peak heights are read from the cb
// variance at M_min and M_max. Growth is mass dependent, so these two
// endpoints cannot be obtained by dividing z=0 values by one D(a).
// Map a Gauss-Legendre rule x_q, w_q on [-1,1] to the current interval:
//
//   mid  = (nu_max + nu_min)/2,   half = (nu_max - nu_min)/2,
//   nu_q = mid + half x_q,
//   bias_norm(a) = half sum_q w_q b(nu_q) f(nu_q,a).
//
// Code map: agrid[i] is the scale factor; numin, numax, numid and
// nuhalf are these endpoints, midpoint and half-width; x[q], w[q]
// are the fixed Gauss-Legendre nodes and weights. Only the interval
// changes with a; the same quadrature rule is reused for every row.
//
// Cache invalidation:
//   the a grid and the Gauss-Legendre nodes: rebuilt when Ntable.random
//     changes
//   table refill: cosmology.random (cache[0]) or Ntable.random (cache[1])
//
// Parameters:
//   a - scale factor
//
// Returns:
//   bias_norm(a), dimensionless, linearly interpolated in a from the
//   table; constant outside [limits.a_min, 0.9999999]
// ---------------------------------------------------------------------------
double bias_norm(
    const double a  // scale factor
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS]; // [0] cosmology, [1] Ntable tag
  static double* table = NULL;     // [N_a] bias_norm on the a grid
  static double* agrid = NULL;     // [N_a] the a nodes
  static double lim[3];            // a_min, 0.9999999, spacing in a
  static int n_gauss = 0;          // number of Gauss-Legendre nodes
  static double* gl_node = NULL;   // [n_gauss] nodes x_q on [-1, 1]
  static double* gl_weight = NULL; // [n_gauss] weights w_q on [-1, 1]

  // --- 1. NTABLE REBUILD: THE a GRID AND THE GAUSS-LEGENDRE RULE ---
  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    // a_i = a_min + i da, both ends included; the top stays below 1
    // because fnu_params_at needs 0 < a < 1
    if (table != NULL) {
      free(table);
    }
    if (agrid != NULL) {
      free(agrid);
    }
    table = (double*) malloc(sizeof(double)*Ntable.N_a);
    agrid = (double*) malloc(sizeof(double)*Ntable.N_a);
    lim[0] = limits.a_min;
    lim[1] = 0.9999999;  // just below 1: fnu_params_at aborts at a = 1
    lim[2] = (lim[1] - lim[0]) / ((double) Ntable.N_a - 1.0);
    for (int i=0; i<Ntable.N_a; i++) {
      agrid[i] = lim[0] + i*lim[2];
    }

    // x_q, w_q on [-1, 1]; the refill maps them onto [nu_min, nu_max].
    // The node count grows with the accuracy switch.
    if (gl_node != NULL) {
      free(gl_node);
      free(gl_weight);
    }
    const int level = abs(Ntable.high_def_integration);
    if (0 == level) {
      n_gauss = 128;
    } else if (1 == level) {
      n_gauss = 256;
    } else {
      n_gauss = 512;
    }
    gl_node   = (double*) malloc(sizeof(double)*n_gauss);
    gl_weight = (double*) malloc(sizeof(double)*n_gauss);
    gsl_integration_glfixed_table* gauss_table =
        malloc_gslint_glfixed(n_gauss);
    for (int q=0; q<n_gauss; q++) {
      gsl_integration_glfixed_point(-1.0, 1.0, q,
                                    &gl_node[q], &gl_weight[q], gauss_table);
    }
    gsl_integration_glfixed_table_free(gauss_table);
  }

  // --- 2. TABLE REFILL: THE QUADRATURE AT EVERY a NODE ---
  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random)) {
    // Warm both lazy tables before the row loop. The nu bounds depend
    // on a because neutrino growth depends on halo mass.
    (void) fnu(1.0, agrid[0]);
    (void) sigma2(limits.halo_m[RANGE_MIN], agrid[0]);

    const double* restrict x = gl_node;
    const double* restrict w = gl_weight;
    const int n = n_gauss;

    #pragma omp parallel for schedule(static)
    for (int i=0; i<Ntable.N_a; i++) {
      // Peak-height endpoints and nu-independent coefficients at a_i
      const double numin = delta_c/sqrt(sigma2(limits.halo_m[RANGE_MIN], agrid[i]));
      const double numax = delta_c/sqrt(sigma2(limits.halo_m[RANGE_MAX], agrid[i]));
      const double numid = 0.5*(numax+numin);
      const double nuhalf = 0.5*(numax-numin);
      const hb1nu_params bias_par = hb1nu_params_at(agrid[i]);
      const fnu_params fnu_par = fnu_params_at(agrid[i]);

      // sum_q w_q b(nu_q) f(nu_q, a_i),   nu_q = numid + nuhalf x_q
      double sum = 0.0;
      for (int q=0; q<n; q++) {
        const double nu = numid + nuhalf*x[q];
        sum += w[q]*hb1nu_core(nu, &bias_par)*fnu_core(nu, &fnu_par);
      }

      // bias_norm(a_i): multiply by nuhalf because dnu = nuhalf dx
      table[i] = sum*nuhalf;
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }

  return interpol1d(table, Ntable.N_a, lim[0], lim[1], lim[2], a);
}


// ---------------------------------------------------------------------------
// Logarithmic slope d ln nu/d ln M of the peak height: the Jacobian that
// turns f(nu) dnu into dn/dlnM (section banner).
//
// dln nu/dlnM = -dln sigma_cb/dlnM at fixed a. The FFTLog derivative
// kernel supplies this slope on the same (a, lnM) grid as sigma2;
// finite differencing the interpolated variance would add grid noise.
// Cache invalidation:
// follows the variance tables (cosmology.random and Ntable.random).
// ---------------------------------------------------------------------------
double dlognudlogm(
    const double M, // halo mass in M_sun/h
    const double a  // scale factor
  )
{
  return -dlnsigma_dlnm_field(M, a, HALO_FIELD_CB);
}



// ============================================================================
// [SECTION] HALO PROFILES
// ============================================================================
//
// A profile enters the halo model through its Fourier transform. For
// the matter it is normalized by the halo mass,
//
//   u(k|M) = int_0^{r_Delta} 4 pi r^2 [sin(kr)/(kr)] rho(r|M) dr / M
//
// (astro-ph/0206508 Eq. 80), so u -> 1 as k -> 0 (on scales much larger
// than the halo it is a point mass) and u falls off once k r_s ~ 1 (the
// halo is resolved). Truncating the profile at r_Delta makes its mass
// exactly the M of the mass function.


// ---------------------------------------------------------------------------
// The NFW transform table nfw_, shared by u_nfw_c and the spectrum table
// builders: the two smooth functions f, g of Abramowitz & Stegun
// 5.2.6-5.2.7 that carry the non-oscillating part of the sine and
// cosine integrals (u_nfw_c header),
//
//   Si(t) = pi/2 - f(t) cos t - g(t) sin t,   Ci(t) = f(t) sin t - g(t) cos t
//   ->  f = Ci sin t + (pi/2 - Si) cos t,   g = -Ci cos t + (pi/2 - Si) sin t
//
// stored as f(t) and G(t) = g(t) + ln t (g ~ -ln t at t -> 0; G stays
// finite, -gamma_E there) at Ntable.halo_nfw_n nodes uniform in ln t
// over [NFW_TMIN, NFW_TASY]:
//
//   tab[0][i] = f(t_i),   tab[1][i] = G(t_i),   ln t_i = lim[0] + i lim[2]
//
// Reads (nfw_um) interpolate linearly in ln t between nodes, clamp
// below NFW_TMIN and switch to the asymptotic series above NFW_TASY.
// ---------------------------------------------------------------------------
static const double NFW_TMIN = 1e-10; // reads clamp below NFW_TMIN
static const double NFW_TASY = 50.0;  // asymptotic series above NFW_TASY

static struct {
  uint64_t cache;      // Ntable.random of the table
  int n_nodes;         // number of ln t nodes
  double lim[3];       // ln t axis: first, last, spacing
  double inv_spacing;  // 1/spacing (nfw_pos multiplies by it)
  double** tab;        // [2][n_nodes] f(t), G(t) = g(t) + ln t
} nfw_ = {0};


static void nfw_table(void)
{
  // f(t_i), G(t_i) from GSL Si, Ci (formulas in the nfw_ header);
  // rebuilt when Ntable.random changes
  if (NULL == nfw_.tab || fdiff2(nfw_.cache, Ntable.random)) {
    if (nfw_.tab != NULL) {
      free(nfw_.tab);
    }

    // --- 1. AXIS SETUP ---
    const int n_nodes = Ntable.halo_nfw_n;
    double** tab = (double**) malloc2d(2, n_nodes);
    double* lim  = nfw_.lim;
    lim[0] = log(NFW_TMIN);
    lim[1] = log(NFW_TASY);
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
    nfw_.n_nodes     = n_nodes;
    nfw_.inv_spacing = 1.0/lim[2];
    nfw_.tab         = tab;
    nfw_.cache       = Ntable.random;
  }
}


// Position of ln t on the nfw_ grid: node index i and the fraction frac
// of the interval [i, i + 1], so a read is
// tab[i] + frac*(tab[i + 1] - tab[i]); f and G share the grid. Below
// NFW_TMIN the read returns tab[0]; i is clamped onto the last
// interval. nfw_table must have run.
static inline int nfw_pos(
    const double lnt, // ln t
    double* frac      // output: fraction of the interval [i, i + 1]
  )
{
  const double pos = (fmax(lnt, nfw_.lim[0]) - nfw_.lim[0])*nfw_.inv_spacing;

  int i = (int) pos;
  if (i > nfw_.n_nodes - 2) {
    i = nfw_.n_nodes - 2;
  }

  *frac = pos - i;
  return i;
}


// u m(c) of the NFW transform for one halo at one k (u_nfw_c header),
//
//   u m(c) = [g(x) - g(xu)] + 2 g(xu) sin^2(c x/2) + [f(xu) - 1/xu] sin(c x)
//   x = k r_s,   xu = (1 + c) x
//
// with f, g read from the nfw_ table as f and G = g + ln t:
//
//   fu = f(xu),   Gx = G(x),   Gu = G(xu),   gu = g(xu) = Gu - ln xu,
//   g(x) - g(xu) = Gx - Gu + ln(1 + c)
//
// The caller passes lnx = ln x and ln1c = ln(1 + c) (the table axis is
// ln t) and divides by m(c). nfw_table must have run (the build is not
// thread-safe; this read is).
static inline double nfw_um(
    const double c,    // concentration r_Delta/r_s
    const double x,    // k r_s
    const double lnx,  // ln x
    const double ln1c  // ln(1 + c)
  )
{
  const double* restrict tab_f = nfw_.tab[0];  // f(t) on the ln t grid
  const double* restrict tab_G = nfw_.tab[1];  // G(t) = g(t) + ln t
  const double lnxu = lnx + ln1c;       // ln xu, xu = (1 + c) x
  const double xu = (1.0 + c)*x;

  // f(xu), G(x), G(xu): table reads up to NFW_TASY, the asymptotic
  // series (A&S 5.2.34-35) above it, nested,
  //   f(t) ~ (1 - 2!/t^2 + 4!/t^4 - 6!/t^6 + 8!/t^8)/t
  //   g(t) ~ (1 - 3!/t^2 + 5!/t^4 - 7!/t^6 + 9!/t^8)/t^2
  // (2, 12, 30, 56 and 6, 20, 42, 72 are the ratios of consecutive
  // coefficients). x < xu, so x may sit in the table when xu does not.
  double fu, Gx, Gu;
  if (xu <= NFW_TASY) {
    double frac_u, frac_x;
    const int iu = nfw_pos(lnxu, &frac_u); // f and G share the grid
    const int ix = nfw_pos(lnx, &frac_x);
    Gu = frac_u*(tab_G[iu + 1] - tab_G[iu]) + tab_G[iu];
    fu = frac_u*(tab_f[iu + 1] - tab_f[iu]) + tab_f[iu];
    Gx = frac_x*(tab_G[ix + 1] - tab_G[ix]) + tab_G[ix];
  }
  else {
    const double v = 1.0/(xu*xu); // series variable v = 1/xu^2
    fu = (1.0 - 2.0*v*(1.0 - 12.0*v*(1.0 - 30.0*v*(1.0 - 56.0*v))))/xu;
    Gu = v*(1.0 - 6.0*v*(1.0 - 20.0*v*(1.0 - 42.0*v*(1.0 - 72.0*v)))) + lnxu;
    if (x <= NFW_TASY) {
      double frac_x;
      const int ix = nfw_pos(lnx, &frac_x);
      Gx = frac_x*(tab_G[ix + 1] - tab_G[ix]) + tab_G[ix];
    }
    else {
      const double w = 1.0/(x*x); // series variable w = 1/x^2
      Gx = w*(1.0 - 6.0*w*(1.0 - 20.0*w*(1.0 - 42.0*w*(1.0 - 72.0*w)))) + lnx;
    }
  }

  // u m(c) = [g(x) - g(xu)] + 2 g(xu) sin^2(c x/2) + [f(xu) - 1/xu] sin(c x),
  // with g(x) - g(xu) = Gx - Gu + ln1c
  const double gu = Gu - lnxu;          // g(xu)
  const double sin_half = sin(0.5*c*x); // sin(c x/2)
  return (Gx - Gu + ln1c) + 2.0*gu*sin_half*sin_half + (fu - 1.0/xu)*sin(c*x);
}


// ---------------------------------------------------------------------------
// SIMD path of the halo spectra: nfw_um on four mass nodes at once.
//
// The mass-node sums of p_gm and p_gg call nfw_um4, which
// is nfw_um with each of its four arguments carrying four nodes (one
// per lane of a v4d) and each lane of its result bitwise the scalar
// nfw_um of that node. The helpers below build it, in reading order:
//
//   nfw_fmadd4, nfw_fnmadd4 - a*b + c and c - a*b, rounded once where
//                             the scalar path fuses them
//   nfw_pos4                - position (node index, fraction) of ln t
//                             on the table grid
//   nfw_read4               - the linear table read at that position
//   nfw_sin4                - libm sin on each lane
//   nfw_series_step4        - one bracket of the asymptotic series
//   nfw_G_asym4             - the asymptotic series of G(t)
//   nfw_um4                 - the kernel itself
//
// Each header says what its function computes, how the lanes map to
// nodes and why its result is bitwise the scalar path's, under the
// strict IEEE flags (-frounding-math -ftrapping-math): fused
// multiply-adds exactly where the scalar path fuses, sign-bit masks
// instead of FP compares, table indices truncated and clamped in
// double, and every helper always inlined (on arm64 a v4d is a union of
// two NEON registers and a real call would pass it through memory).
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// nfw_fmadd4: a*b + c on four lanes with one rounding, the vector form of
// the scalar path's fused multiply-add.
//
// A fused multiply-add computes a*b + c as one operation: the product
// a*b is kept exact and only the final sum is rounded to double (one
// rounding). A separate multiply and add rounds twice, and the two
// results can differ in the last bit. The compiler fuses the scalar
// path's a*b + c (nfw_um, the mass-node sums), so the vector path must
// fuse the same products at the same places to stay bitwise equal to
// it; nfw_fmadd4 is the one place where it does.
//
// With native x86 FMA, simde_mm256_fmadd_pd is one 256-bit FMA3
// instruction (vfmadd...pd, one rounding).
// Without it (arm64), that same call is a multiply and then an add, two
// roundings: SIMDe has no NEON branch at 256 bits. The two-lane
// simde_mm_fmadd_pd is a real fused NEON instruction (vfmaq_f64), so the
// v4d is split into its two v2d halves (lanes 0,1 = the low half, lanes
// 2,3 = the high half), each half is fused, and the halves are joined
// again into a v4d. Lane l of the result is a[l]*b[l] + c[l] either
// way. An x86 build without FMA also takes this split branch; there the
// halves round twice, as the scalar path does, so the two still match.
//
// Parameters:
//   va - the four multiplicands a
//   vb - the four multipliers b
//   vc - the four addends c
//
// Returns:
//   a*b + c on each lane, one rounding (two on x86 built without FMA)
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d nfw_fmadd4(
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
// nfw_fnmadd4: c - a*b on four lanes with one rounding, the vector form
// of the scalar path's fused negative multiply-add.
//
// As nfw_fmadd4 with the product negated: the exact a*b is subtracted
// from c and only the difference is rounded (one rounding). The
// asymptotic series of nfw_um is a chain of 1 - k v (...) steps that
// the compiler fuses this way in the scalar path, so nfw_series_step4
// and nfw_G_asym4 fuse them through this function to stay bitwise
// equal.
//
// One 256-bit FMA3 instruction with native x86 FMA; otherwise the
// 128-bit split of nfw_fmadd4 (lanes 0,1 low, lanes 2,3 high), each half
// fused on arm64 by vfmsq_f64, joined again (on x86 without FMA the
// halves round twice, as the scalar path does). Lane l of the result is
// c[l] - a[l]*b[l] either way.
//
// Parameters:
//   va - the four multiplicands a
//   vb - the four multipliers b
//   vc - the four minuends c
//
// Returns:
//   c - a*b on each lane, one rounding (two on x86 built without FMA)
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d nfw_fnmadd4(
    const v4d va,   // a on four lanes
    const v4d vb,   // b on four lanes
    const v4d vc    // c on four lanes
  )
{
#ifdef SIMDE_X86_FMA_NATIVE
  // c - a*b on all four lanes, one fused instruction
  return simde_mm256_fnmadd_pd(va, vb, vc);
#else
  // the low half of each input, as in nfw_fmadd4
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
// nfw_pos4: nfw_pos on four lanes, the position of ln t on the nfw_ grid
// for four values of t at once.
//
// The nfw_ table stores f and G at uniform nodes ln t_i = ln t_min +
// i spacing. A "position" is the pair (i, frac) that places ln t in the
// grid: i is the index of the node at or below ln t, and frac in [0, 1)
// is how far ln t sits into the cell [ln t_i, ln t_{i+1}], in cells:
//
//   pos  = (max(ln t, ln t_min) - ln t_min)/spacing
//   i    = min(trunc(pos), n_nodes - 2)      (clamped onto the last cell)
//   frac = pos - i
//
// Lane l of vlnt is one ln t; index[l] and lane l of the result are its
// i and frac. Below ln t_min the clamp puts the read at the first node
// (f and G are flat there, see NFW_TMIN).
//
// Why it is bitwise nfw_pos: the scalar path computes pos in double,
// converts it to int (which truncates) and clamps. Here trunc and min
// are taken in double, and both are exact for the non-negative
// positions of the grid (an integer-valued double rounds to nothing),
// so pos - i is the same double and the int conversion, done lane by
// lane, yields the same i. Compares are avoided on purpose: the strict
// IEEE flags split a vector FP compare into scalar compares per lane.
//
// Parameters:
//   vlnt  - ln t on four lanes
//   index - output: the node index i of each lane
//
// Returns:
//   frac, the fraction of the cell [i, i + 1], on each lane
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d nfw_pos4(
    const v4d vlnt,   // ln t on four lanes
    int index[4]      // output: node index i of each lane
  )
{
  // ln t_min, the first grid node, in all four lanes (set1 copies one
  // scalar into every lane)
  const v4d vlnt_min = simde_mm256_set1_pd(nfw_.lim[0]);

  // 1/spacing of the grid in all four lanes
  const v4d vinv_spacing = simde_mm256_set1_pd(nfw_.inv_spacing);

  // n_nodes - 2, the index of the last cell, in all four lanes
  const v4d vlast_cell = simde_mm256_set1_pd((double) (nfw_.n_nodes - 2));

  // scalar (nfw_pos): pos = (max(ln t, ln t_min) - ln t_min)/spacing,
  // i = min(trunc(pos), n - 2), each step below on all four lanes

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
// nfw_read4: the linear table read of nfw_um on four lanes.
//
// A read of the nfw_ table at position (i, frac) is the straight line
// between the two nodes that bracket ln t,
//
//   tab[i] + frac*(tab[i + 1] - tab[i]),
//
// frac = 0 at node i, frac = 1 at node i + 1 (nfw_pos4 gives i and
// frac). Lane l reads tab at index[l] with lane l of vfrac.
//
// Memory access: each lane needs the two neighbours tab[i], tab[i + 1],
// which sit side by side, so one 16-byte load per lane fetches both
// (a v2d pair). The four pairs are then regrouped into a v4d of left
// nodes and a v4d of right nodes. A gather instruction (four scattered
// loads in one call) would do the same, but it is slow on several x86
// cores and is lane-by-lane loads on NEON anyway.
//
// Why it is bitwise nfw_um: the scalar read is
// frac*(tab[i + 1] - tab[i]) + tab[i], which the compiler fuses into
// one multiply-add; the vector read fuses the same product through
// nfw_fmadd4.
//
// Parameters:
//   tab   - the table row to read, f (nfw_.tab[0]) or G (nfw_.tab[1])
//   index - the node index i of each lane (from nfw_pos4)
//   vfrac - the fraction of the cell [i, i + 1] of each lane
//
// Returns:
//   the interpolated table value on each lane
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d nfw_read4(
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
  return nfw_fmadd4(vfrac, vrise, vleft);
}


// ---------------------------------------------------------------------------
// nfw_sin4: sin on each of four lanes, with the libm sin of the scalar
// path.
//
// nfw_um needs sin(c x/2) and sin(c x) per node, and they are about
// half of the kernel's cost. There is no vector sine here by maintainer
// decision: a vector math library (SLEEF and the like) would give a
// different last bit from libm, and the SIMD path must be bitwise
// nfw_um. So the
// four angles are written out of the v4d into a plain double[4]
// (storeu), sin is called on each one exactly as the scalar path does,
// and the four sines are read back into a v4d (loadu). Lane l of the
// result is sin of lane l of vangle.
//
// Parameters:
//   vangle - the four angles, in radians
//
// Returns:
//   sin(angle) on each lane
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d nfw_sin4(
    const v4d vangle   // the four angles
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
// nfw_series_step4: one step of the nested asymptotic series of nfw_um,
//
//   poly -> 1 - k v poly,
//
// on four lanes, fused as the scalar 1.0 - k*v*(...) (nfw_fnmadd4).
//
// Above the table's top (t > NFW_TASY) nfw_um evaluates f and g by their
// asymptotic series in v = 1/t^2 (A&S 5.2.34-35),
//
//   f(t) ~ (1 - 2!/t^2 + 4!/t^4 - 6!/t^6 + 8!/t^8)/t
//   g(t) ~ (1 - 3!/t^2 + 5!/t^4 - 7!/t^6 + 9!/t^8)/t^2
//
// written in nested (Horner) form, innermost bracket first:
//
//   f(t) = (1 - 2v(1 - 12v(1 - 30v(1 - 56v))))/t
//   g(t) = v(1 - 6v(1 - 20v(1 - 42v(1 - 72v))))
//
// where 2, 12, 30, 56 and 6, 20, 42, 72 are the ratios of consecutive
// coefficients (2!/0!, 4!/2!, ...). Starting from the innermost bracket
// (1 - 56v or 1 - 72v), each call of this function wraps the series in
// one more bracket: poly = 1 - 56v becomes 1 - 30v(1 - 56v), and so on
// outward, k taking the next ratio each time. nfw_um4 chains the f
// steps (30, 12, 2) and nfw_G_asym4 the g steps (42, 20, 6).
//
// Parameters:
//   k     - the coefficient ratio of this bracket
//   vv    - the series variable v = 1/t^2 on four lanes
//   vpoly - the bracket built so far on four lanes
//
// Returns:
//   1 - k v poly on each lane, rounded as nfw_fnmadd4 rounds it
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d nfw_series_step4(
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
  return nfw_fnmadd4(vkv, vpoly, vone);
}


// ---------------------------------------------------------------------------
// nfw_G_asym4: the asymptotic series of G(t) = g(t) + ln t on four lanes,
//
//   G(t) = v(1 - 6v(1 - 20v(1 - 42v(1 - 72v)))) + ln t,   v = 1/t^2,
//
// the nested form of g(t) ~ (1 - 3!/t^2 + 5!/t^4 - 7!/t^6 + 9!/t^8)/t^2
// (nfw_series_step4 explains the nesting and the ratios 6, 20, 42, 72).
//
// The nfw_ table stores G = g + ln t up to t = NFW_TASY; above it the
// series replaces the table, since there the dropped terms (11!/t^10
// and beyond) are below the table's accuracy while the table would need
// ever more nodes for g's slow 1/t^2 fall-off. nfw_um4 calls this for
// G(xu) and G(x) on the lanes past NFW_TASY (the other lanes get a
// finite stand-in that the caller discards).
//
// Bitwise nfw_um: the same brackets in the same order, each 1 - k v (..)
// fused (nfw_fnmadd4), and the final v poly + ln t fused (nfw_fmadd4),
// as the compiler fuses the scalar expression.
//
// Parameters:
//   vv   - the series variable v = 1/t^2 on four lanes
//   vlnt - ln t on four lanes
//
// Returns:
//   G(t) on each lane
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d nfw_G_asym4(
    const v4d vv,    // series variable v = 1/t^2 on four lanes
    const v4d vlnt   // ln t on four lanes
  )
{
  // 1 in all four lanes
  const v4d vone = simde_mm256_set1_pd(1.0);

  // the innermost coefficient ratio 72 in all four lanes
  const v4d vratio72 = simde_mm256_set1_pd(72.0);

  // scalar: v*(1 - 6v(1 - 20v(1 - 42v(1 - 72v)))) + ln t, innermost first

  // 1 - 72v
  v4d vpoly = nfw_fnmadd4(vratio72, vv, vone);

  // the outer brackets, one per step
  vpoly = nfw_series_step4(42.0, vv, vpoly);  // 1 - 42v(1 - 72v)
  vpoly = nfw_series_step4(20.0, vv, vpoly);  // 1 - 20v(...)
  vpoly = nfw_series_step4(6.0, vv, vpoly);   // 1 - 6v(...)

  // v poly + ln t, fused
  return nfw_fmadd4(vv, vpoly, vlnt);
}


// ---------------------------------------------------------------------------
// nfw_um4: nfw_um on four mass nodes at once, the NFW transform u m(c)
// of four halos at one k (u_nfw_c header),
//
//   u m(c) = [g(x) - g(xu)] + 2 g(xu) sin^2(c x/2) + [f(xu) - 1/xu] sin(c x)
//   x = k r_s,   xu = (1 + c) x,
//
// with f and g read from the nfw_ table as f and G = g + ln t:
//
//   fu = f(xu),   Gx = G(x),   Gu = G(xu),   g(xu) = Gu - ln xu,
//   g(x) - g(xu) = Gx - Gu + ln(1 + c).
//
// Lanes: lane l of every argument belongs to one mass node (the callers
// pass nodes q, q+1, q+2, q+3 of a quadrature sum), and lane l of the
// result is u m(c) of that node. The callers divide by m(c) (folded into
// their weights).
//
// Algorithm, in order (the section banners in the body):
//   1. branch masks: per lane, table (t <= NFW_TASY) or asymptotic
//      series (t > NFW_TASY) for t = xu and for t = x, from the sign
//      bit of NFW_TASY - t (movemask), no FP compare;
//   2. table reads: the position (i, frac) of ln xu and of ln x on the
//      ln t grid (nfw_pos4), then f(xu), G(xu), G(x) by linear
//      interpolation (nfw_read4), skipped when every lane is past
//      NFW_TASY;
//   3. asymptotic series: f(xu), G(xu), G(x) in v = 1/t^2
//      (nfw_series_step4, nfw_G_asym4), skipped when no lane needs it;
//      per lane, blendv keeps the table value or takes the series;
//   4. the two sines sin(c x/2) and sin(c x) (nfw_sin4), then the
//      combination above.
//
// Why it is bitwise nfw_um: the same operations in the same order on
// every lane, a fused multiply-add exactly where the compiler fuses the
// scalar a*b + c (nfw_fmadd4 / nfw_fnmadd4), and libm sin on every lane
// (nfw_sin4). The branch is taken per lane by masks, so a lane past
// NFW_TASY gets the series value and a lane below it the table value,
// as the scalar if/else would give. Every function here is always
// inlined: on arm64 a v4d is a union of two NEON registers and a real
// call would pass it through memory.
//
// nfw_table must have run (the build is not thread-safe; this read is).
//
// Parameters:
//   vc    - concentration c = r_Delta/r_s of the four nodes
//   vx    - x = k r_s of the four nodes
//   vlnx  - ln x of the four nodes
//   vln1c - ln(1 + c) of the four nodes
//
// Returns:
//   u m(c) of each node on its lane, dimensionless
// ---------------------------------------------------------------------------
static inline __attribute__((always_inline)) v4d nfw_um4(
    const v4d vc,     // concentration r_Delta/r_s on four lanes
    const v4d vx,     // k r_s on four lanes
    const v4d vlnx,   // ln x on four lanes
    const v4d vln1c   // ln(1 + c) on four lanes
  )
{
  const double* restrict tab_f = nfw_.tab[0];  // f(t) on the ln t grid
  const double* restrict tab_G = nfw_.tab[1];  // G(t) = g(t) + ln t

  // 1 in all four lanes
  const v4d vone = simde_mm256_set1_pd(1.0);

  // NFW_TASY, the top of the table, in all four lanes
  const v4d vtasy = simde_mm256_set1_pd(NFW_TASY);

  // scalar: lnxu = lnx + ln1c;  xu = (1.0 + c)*x

  // ln xu = ln x + ln(1 + c)
  const v4d vlnxu = simde_mm256_add_pd(vlnx, vln1c);

  // 1 + c
  const v4d vone_plus_c = simde_mm256_add_pd(vone, vc);

  // xu = (1 + c) x
  const v4d vxu = simde_mm256_mul_pd(vone_plus_c, vx);

  // --- 1. BRANCH MASKS ---
  // scalar: if (xu <= NFW_TASY) read the table, else the series; the
  // same for x. NFW_TASY - t has its sign bit set exactly where
  // t > NFW_TASY (the asymptotic series) and clear where the table is
  // read (+0 at t = NFW_TASY). The masks must see the rounded xu of the
  // scalar compare: xu = (1 + c) x stays unfused because it has other
  // uses (1/xu, the blend), which GCC's -ffp-contract=fast needs to
  // leave the product alone

  // NFW_TASY - xu: negative (sign bit set) on the series lanes
  const v4d vasym_u = simde_mm256_sub_pd(vtasy, vxu);

  // NFW_TASY - x
  const v4d vasym_x = simde_mm256_sub_pd(vtasy, vx);

  // movemask collects the sign bit of each lane into bit l of an int:
  // 0 = every lane reads the table, 0xF = every lane takes the series
  const int asym_u = simde_mm256_movemask_pd(vasym_u);  // for t = xu
  const int asym_x = simde_mm256_movemask_pd(vasym_x);  // for t = x

  // --- 2. TABLE READS: f(xu), G(xu), G(x) ---
  // lanes past NFW_TASY read the clamped last interval; the blend below
  // replaces them

  // (0, 0, 0, 0) until a branch fills them
  v4d vfu = simde_mm256_setzero_pd();  // f(xu)
  v4d vGu = simde_mm256_setzero_pd();  // G(xu)
  v4d vGx = simde_mm256_setzero_pd();  // G(x)

  if (asym_u != 0xF) {
    int index_u[4];

    // scalar: iu = nfw_pos(lnxu, &frac_u), f and G share the grid
    const v4d vfrac_u = nfw_pos4(vlnxu, index_u);

    // scalar: Gu = frac_u*(tab_G[iu + 1] - tab_G[iu]) + tab_G[iu]
    vGu = nfw_read4(tab_G, index_u, vfrac_u);

    // scalar: fu = frac_u*(tab_f[iu + 1] - tab_f[iu]) + tab_f[iu]
    vfu = nfw_read4(tab_f, index_u, vfrac_u);
  }
  if (asym_x != 0xF) {
    int index_x[4];

    // scalar: ix = nfw_pos(lnx, &frac_x)
    const v4d vfrac_x = nfw_pos4(vlnx, index_x);

    // scalar: Gx = frac_x*(tab_G[ix + 1] - tab_G[ix]) + tab_G[ix]
    vGx = nfw_read4(tab_G, index_x, vfrac_x);
  }

  // --- 3. ASYMPTOTIC SERIES (A&S 5.2.34-35, nfw_um) ---
  // lanes that read the table evaluate the series at t = NFW_TASY, a
  // finite stand-in (no 1/t^2 overflow at tiny t) that the blend discards
  if (asym_u != 0) {
    // t = xu on the series lanes, NFW_TASY on the table lanes (blendv
    // takes lane l from its second argument where bit l of the mask is
    // set, from its first argument otherwise)
    const v4d vt = simde_mm256_blendv_pd(vtasy, vxu, vasym_u);

    // t^2
    const v4d vt2 = simde_mm256_mul_pd(vt, vt);

    // scalar: v = 1.0/(xu*xu), the series variable
    const v4d vv = simde_mm256_div_pd(vone, vt2);

    // scalar: fu = (1 - 2v(1 - 12v(1 - 30v(1 - 56v))))/xu, innermost first

    // the innermost coefficient ratio 56 in all four lanes
    const v4d vratio56 = simde_mm256_set1_pd(56.0);

    // 1 - 56v
    v4d vpoly = nfw_fnmadd4(vratio56, vv, vone);

    // the outer brackets, one per step
    vpoly = nfw_series_step4(30.0, vv, vpoly);  // 1 - 30v(1 - 56v)
    vpoly = nfw_series_step4(12.0, vv, vpoly);  // 1 - 12v(...)
    vpoly = nfw_series_step4(2.0, vv, vpoly);   // 1 - 2v(...)

    // poly/t
    const v4d vfu_asym = simde_mm256_div_pd(vpoly, vt);

    // f(xu): the series on the series lanes, the table value elsewhere
    vfu = simde_mm256_blendv_pd(vfu, vfu_asym, vasym_u);

    // scalar: Gu = v*(1 - 6v(1 - 20v(1 - 42v(1 - 72v)))) + lnxu
    const v4d vGu_asym = nfw_G_asym4(vv, vlnxu);

    // G(xu): the series on the series lanes, the table value elsewhere
    vGu = simde_mm256_blendv_pd(vGu, vGu_asym, vasym_u);
  }
  if (asym_x != 0) {
    // t = x on the series lanes, NFW_TASY on the table lanes
    const v4d vt = simde_mm256_blendv_pd(vtasy, vx, vasym_x);

    // t^2
    const v4d vt2 = simde_mm256_mul_pd(vt, vt);

    // scalar: w = 1.0/(x*x), the series variable
    const v4d vw = simde_mm256_div_pd(vone, vt2);

    // scalar: Gx = w*(1 - 6w(1 - 20w(1 - 42w(1 - 72w)))) + lnx
    const v4d vGx_asym = nfw_G_asym4(vw, vlnx);

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
  const v4d vsin_half = nfw_sin4(vhalf_cx);

  // c x, the full angle
  const v4d vcx = simde_mm256_mul_pd(vc, vx);

  // sin(c x)
  const v4d vsin_full = nfw_sin4(vcx);

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
  const v4d vsum = nfw_fmadd4(vtwo_gu_sin, vsin_half, vg_diff);

  // (fu - 1/xu) sin(c x) + the rest, fused: u m(c) on the four lanes
  return nfw_fmadd4(vf_term, vsin_full, vsum);
}


// ---------------------------------------------------------------------------
// Normalized Fourier transform u(k|M) of the NFW profile truncated at
// r_Delta (astro-ph/0206508 Eq. 81).
//
// The NFW profile (astro-ph/9611107) rho(r) = rho_s/[(r/r_s)(1 + r/r_s)^2],
// r_s = r_Delta/c, holds M = 4 pi rho_s r_s^3 m(c) inside r_Delta, with
// m(c) = ln(1+c) - c/(1+c) (astro-ph/0206508 Eq. 76). With
//
//   r_Delta = (3M/(4 pi Delta rho_m))^(1/3)   (comoving, c/H0)
//   x       = k r_Delta/c = k r_s,   xu = (1 + c) x
//
// Eq. 81 reads
//
//   u = { sin x [Si(xu) - Si(x)] - sin(c x)/xu
//         + cos x [Ci(xu) - Ci(x)] } / m(c)
//
// with Si, Ci the sine and cosine integrals. At k -> 0 the three terms
// tend to 0, -c/(1+c) and ln(1+c), so u -> 1. r_Delta and k are
// comoving (rho_m = rho_crit Omega_m): the scale factor never enters.
//
// Si, Ci oscillate. Abramowitz & Stegun 5.2.6-5.2.7 split them into the
// explicit sin t, cos t and two smooth functions f, g (nfw_ header):
//
//   Si(t) = pi/2 - f(t) cos t - g(t) sin t,   Ci(t) = f(t) sin t - g(t) cos t
//
// In Eq. 81 the sin x, cos x factors then collapse (xu - x = c x); with
// cos(c x) = 1 - 2 sin^2(c x/2) (no cancellation at small c x), exactly
//
//   u m(c) = [g(x) - g(xu)] + 2 g(xu) sin^2(c x/2) + [f(xu) - 1/xu] sin(c x)
//
// Code map: nfw_um reads f, g from the nfw_ table (as f and G = g + ln t,
// linear in ln t) and assembles u m(c); this function supplies r_Delta,
// x, ln x, ln(1 + c) and divides by m(c). Accurate to 6e-7 relative for
// c in [0.05, 100].
//
// Cache invalidation:
//   f, G depend on no parameter: built on the first call, rebuilt when
//   Ntable.random changes (halo_nfw_n). The build is not thread-safe:
//   the first call is halo_warmup's nfw_table(), single-threaded.
//
// Parameters:
//   c - concentration r_Delta/r_s, c > 0 (m(0) = 0)
//   k - wavenumber in (c/H0)^-1, k > 0
//   m - halo mass in M_sun/h
//   a - scale factor (unused)
//
// Returns:
//   u(k|M), dimensionless; 1 at k -> 0
// ---------------------------------------------------------------------------
double u_nfw_c(
    const double c, // concentration r_Delta/r_s
    const double k, // wavenumber in (c/H0)^-1
    const double m, // halo mass in M_sun/h
    const double a  // scale factor (unused: r_Delta and k are comoving)
  )
{
  nfw_table();

  // r_Delta = (3M/(4 pi Delta rho_m))^(1/3),  x = k r_s = k r_Delta/c
  const double rho_delta = Delta * cosmology.rho_crit * cosmology.Omega_m;
  const double r_delta   = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
  const double x         = k * r_delta / c;

  const double ln1c = log1p(c); // ln(1 + c)
  const double lnx  = log(x);   // ln x

  return nfw_um(c, x, lnx, ln1c)/(ln1c - c/(1.0 + c)); // u m(c) / m(c)
}


// ============================================================================
// [SECTION] GALAXY PROFILES
// ============================================================================
//
// The halo occupation distribution (HOD) gives the mean number of
// galaxies of lens bin ni in a halo of mass M: one central galaxy at the
// halo center plus satellites that follow a scaled NFW profile (Zehavi
// et al. 2011, 1005.2413 Eq. 7; Coupon et al. 2012, 1107.0616
// sec. 4.1):
//
//   <N|M>  = f_c N_c(M) + N_s(M)
//   N_c(M) = (1/2) [1 + erf((log10 M - log10 M_min)/sigma_lgM)]
//   N_s(M) = N_c(M) [(M - M_0)/M_1]^alpha
//
// The six parameters per bin, nuisance.hod[ni][0..5]:
//
//   [0] = log10 M_min  mass at which half the halos host a central
//   [1] = sigma_lgM    width of that step in log10 M
//   [2] = log10 M_1    mass scale of the satellite power law (M_1' in
//                      1005.2413)
//   [3] = log10 M_0    satellite cutoff mass
//   [4] = alpha        satellite power-law slope
//   [5] = f_c          fraction of centrals in the sample (0 = unset,
//                      read as 1 by HOD_fc)
//
// Masses are in M_sun/h in this file's halo definition (Delta = 200
// times the mean density); HOD fits quoted for another halo definition
// (the virial masses of 1107.0616, for example) carry that difference.


// ---------------------------------------------------------------------------
// Mean number of central galaxies of lens bin ni in a halo of mass M
// (1005.2413 Eq. 7, central factor):
//
//   N_c(M) = (1/2) [1 + erf((log10 M - log10 M_min)/sigma_lgM)]
//
// A smoothed step: N_c = 1/2 at M = M_min, and sigma_lgM is the scatter
// between galaxy luminosity and halo mass, seen as a width in log10 M
// (the erf argument has no sqrt(2)). In the code (GALAXY PROFILES
// banner), nuisance.hod[ni][0] = log10 M_min and [1] = sigma_lgM.
//
// Parameters:
//   m  - halo mass in M_sun/h
//   a  - scale factor, 0 < a < 1 (checked; the HOD does not evolve)
//   ni - lens bin, 0 <= ni < redshift.clustering_nbin
//
// Returns:
//   N_c in [0, 1]. Aborts when log10 M_min lies outside [10, 16], the
//   sign that the bin's HOD is not set.
// ---------------------------------------------------------------------------
double HOD_nc(
    const double m, // halo mass in M_sun/h
    const double a, // scale factor, 0 < a < 1 (checked; HOD is z-free)
    const int ni    // lens bin, 0 <= ni < redshift.clustering_nbin
  )
{
  if (!(a > 0) || !(a < 1)) {
    log_fatal("a>0 and a<1 not true");
    exit(1);
  }
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }

  // log10 M_min outside this range flags a bin whose HOD was never set
  const double lgm_min_lo = 10.0;
  const double lgm_min_hi = 16.0;
  if (nuisance.hod[ni][0] < lgm_min_lo || nuisance.hod[ni][0] > lgm_min_hi) {
    log_fatal("HOD parameters in redshift bin %d not set", ni);
    exit(1);
  }

  // erf argument x = (log10 M - log10 M_min)/sigma_lgM
  const double x = (log10(m) - nuisance.hod[ni][0])/nuisance.hod[ni][1];

  gsl_sf_result erf_result;
  {
    int status = gsl_sf_erf_e(x, &erf_result);
    if (status) {
      log_fatal(gsl_strerror(status));
      exit(1);
    }
  }

  return 0.5*(1.0 + erf_result.val); // N_c = (1 + erf x)/2
}


// ---------------------------------------------------------------------------
// Mean number of satellite galaxies of lens bin ni in a halo of mass M
// (1005.2413 Eq. 7; 1107.0616 sec. 4.1):
//
//   N_s(M) = N_c(M) [(M - M_0)/M_1]^alpha
//
// The factor N_c makes satellites need a central: a halo too light to
// host a central hosts no satellites either. Above M_0 the count grows
// as a power law of slope alpha, and M_1 sets its amplitude (N_s ~ 1 at
// M = M_1 when M_0 and M_min are well below M_1). In the code (GALAXY
// PROFILES banner), nuisance.hod[ni][2] = log10 M_1, [3] = log10 M_0
// and [4] = alpha.
//
// M <= M_0 returns 1e-15 (the power law has no real value there for
// non-integer alpha), and so does an N_s that underflows to 0: N_s
// stays strictly positive, and the floor is negligible in every
// integral.
//
// Parameters:
//   m  - halo mass in M_sun/h
//   a  - scale factor (passed on to HOD_nc)
//   ni - lens bin, 0 <= ni < redshift.clustering_nbin
//
// Returns:
//   N_s(M) >= 1e-15
// ---------------------------------------------------------------------------
double HOD_ns(
    const double m, // halo mass in M_sun/h
    const double a, // scale factor (passed on to HOD_nc)
    const int ni    // lens bin, 0 <= ni < redshift.clustering_nbin
  )
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }

  // floor keeping N_s strictly positive (negligible in every integral)
  const double n_sat_floor = 1.e-15;

  const double m0 = pow(10., nuisance.hod[ni][3]); // M_0
  if (!(m > m0)) {
    return n_sat_floor; // no satellites at or below M_0
  }

  const double x = (m - m0)/pow(10., nuisance.hod[ni][2]); // (M - M_0)/M_1

  // N_s = N_c(M) x^alpha, floored at n_sat_floor if it underflows to 0
  const double n_sat = HOD_nc(m, a, ni)*pow(x, nuisance.hod[ni][4]);
  if (n_sat > 0) {
    return n_sat;
  }
  return n_sat_floor;
}


// ---------------------------------------------------------------------------
// Central fraction f_c of lens bin ni: of the halos that host a central
// above the threshold, the fraction whose central belongs to the sample
// (a completeness factor on centrals only, beyond the five-parameter
// form of 1005.2413). It multiplies N_c in the occupation,
// <N|M> = f_c N_c + N_s, and in the central-satellite pair count
// 2 f_c N_c N_s of the 1-halo term; satellites do not carry it.
//
// Parameters:
//   ni - lens bin, 0 <= ni < redshift.clustering_nbin
//
// Returns:
//   nuisance.hod[ni][5] = f_c, or 1.0 when that slot is 0 (unset)
// ---------------------------------------------------------------------------
double HOD_fc(
    const int ni  // lens bin, 0 <= ni < redshift.clustering_nbin
  )
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }

  // slot [5] left at 0 means unset: read it as f_c = 1
  const double f_c = nuisance.hod[ni][5];
  if (f_c != 0.0) {
    return f_c;
  }
  return 1.0;
}



// ============================================================================
// [SECTION] HALO MODEL ROUTINES
// ============================================================================


// ---------------------------------------------------------------------------
// Tables of ngal and bgal, built and refilled by hod_tables (its header
// maps each array to the integrals); zero at program start, so the
// first call builds everything.
// ---------------------------------------------------------------------------
static struct {
  uint64_t cache[MAX_SIZE_ARRAYS]; // [0] cosmology, [1] Ntable, [2] HOD,
                                   //   [3] clustering n(z), [4] lens
                                   //   photo-z, [5] source n(z) (the a
                                   //   grid) tags
  int nbin;                 // lens bins of the allocation
  int n_nodes;              // Gauss-Legendre mass nodes per lens bin
  double lim[3];            // a grid: min, max, step
  double*** tab;            // [2][nbin][N_a] ngal(a_j) (0), bgal(a_j) (1)
  double*** node_data;      // [2][nbin][n_nodes] per mass node q:
                            //   M_q (0), P_q (1), see hod_tables header
  double** gauss_legendre;  // [2][n_nodes] Gauss-Legendre nodes x_q (0)
                            //   and weights w_q (1) on [-1, 1]
  double* a_nodes;          // [N_a] scale factors of the variance reads
  hb1nu_params* bias_pars;  // [N_a] Tinker bias b(nu) parameters at a_j
  fnu_params* mult_pars;    // [N_a] Tinker multiplicity f(nu) parameters
                            //   at a_j
} hod_ = {0};


// ---------------------------------------------------------------------------
// Fills hod_: number density and mean halo bias of the galaxies of every
// lens bin on one grid in a, read by ngal and bgal:
//
//   ngal(a) = int dlnM (dn/dlnM) <N|M>                       (c/H0)^-3
//   bgal(a) = int dlnM (dn/dlnM) <N|M> b(nu) / ngal(a)
//
//   dn/dlnM = (rho_cb/M) nu f(nu) dln nu/dln M    halo mass function
//   nu      = delta_c/sigma_cb(M,a)            peak height
//   <N|M>   = f_c N_c(M) + N_s(M)                HOD occupation
//   b(nu)                                        linear halo bias
//
// f is the Tinker multiplicity function (fnu), b the halo bias (hb1nu),
// <N|M> the occupation of the GALAXY PROFILES banner (HOD_fc, HOD_nc,
// HOD_ns); the POWER SPECTRA banner explains dn/dlnM.
//
// Quadrature: Gauss-Legendre in ln M (the node count follows an
// Ntable.high_def_integration ladder, set in the rebuild block) over
// [ln 10^(lg M_min - 2), ln limits.halo_m[RANGE_MAX]] of each bin (N_c is an
// erf tail below). Nodes x_q and weights w_q on [-1, 1] map to
// ln M_q = mid + half_width x_q with weight half_width w_q.
//
// The variance, its mass slope, f and b depend on a. The occupation
// and quadrature weight are independent of a and are stored per mass
// node; the remaining factors are evaluated in each a row.
//
// Mapping these factors to the arrays of hod_:
//
//   gauss_legendre[0][q], [1][q]  x_q, w_q
//   node_data[0][b][q]    M_q, the quadrature mass
//   node_data[1][b][q]    P_q = half_width w_q (rho_cb/M_q)
//                           <N|M_q>
//   a_nodes[j], mult_pars[j], bias_pars[j]
//                         a_j, fit parameters of f and b at a_j
//   tab[0][b][j]          ngal(a_j) = sum_q P_q dlnnu/dlnM nu f(nu),
//                         with nu = delta_c/sigma_cb(M_q,a_j)
//   tab[1][b][j]          bgal(a_j) = sum_q P_q dlnnu/dlnM nu f(nu) b(nu)
//                                     / ngal(a_j)
//
// Thus P_q (dlnnu/dlnM) nu f(nu) equals the quadrature weight times
// (dn/dlnM) <N|M> at mass node q.
//
// The a grid: Ntable.N_a nodes uniform in a over [1/(1 + z_max),
// 1/(1 + z_min)] of the clustering n(z), all bins, widened to every lens
// bin's [amin_lens, amax_lens] when one reaches past it (the refill
// explains why); ngal and bgal return 0 outside it and interpolate
// linearly inside, accurate to about 1e-5 (the quadrature itself is
// converged below 1e-6).
//
// Aborts: HOD_nc, on a lens bin whose HOD is not set; the tables cover
// all bins at once.
//
// Cache invalidation:
//   rebuild (sizes, GL nodes, every allocation): Ntable.random
//     or redshift.random_clustering
//   refill (and the a grid): those two, cosmology.random,
//     nuisance.random_galaxy_bias, nuisance.random_photoz_clustering or
//     redshift.random_shear (amax_lens reads the source n(z)'s lower
//     edge when magnification bias is on)
// ---------------------------------------------------------------------------
static void hod_tables(void)
{
  // --- 1. ALLOCATION ---
  // Rebuild: sizes, allocations, the GL nodes x_q, w_q and the a grid
  // (first call, or Ntable or the clustering n(z) changed).
  if (NULL == hod_.tab ||
      fdiff2(hod_.cache[1], Ntable.random) ||
      fdiff2(hod_.cache[3], redshift.random_clustering))
  {
    if (hod_.tab != NULL) {
      free(hod_.tab);
      free(hod_.node_data);
      free(hod_.gauss_legendre);
      free(hod_.a_nodes);
      free(hod_.bias_pars);
      free(hod_.mult_pars);
    }

    // Sizes: lens bins, and the mass-node count of the quadrature
    // ladder (n_nodes follows Ntable.high_def_integration, header).
    const int accuracy = abs(Ntable.high_def_integration);
    hod_.nbin = redshift.clustering_nbin;
    if (0 == accuracy) {
      hod_.n_nodes = 128;
    }
    else if (1 == accuracy) {
      hod_.n_nodes = 256;
    }
    else if (2 == accuracy) {
      hod_.n_nodes = 512;
    }
    else {
      hod_.n_nodes = 1024;
    }

    hod_.tab            = (double***) malloc3d(2, hod_.nbin, Ntable.N_a);
    hod_.node_data      = (double***) malloc3d(2, hod_.nbin, hod_.n_nodes);
    hod_.gauss_legendre = (double**) malloc2d(2, hod_.n_nodes);
    hod_.a_nodes         = (double*) malloc1d(Ntable.N_a);
    hod_.bias_pars      = (hb1nu_params*)
                          malloc(sizeof(hb1nu_params)*Ntable.N_a);
    hod_.mult_pars      = (fnu_params*)
                          malloc(sizeof(fnu_params)*Ntable.N_a);

    // x_q, w_q on [-1, 1]; the refill maps them onto each bin's ln M range.
    gsl_integration_glfixed_table* gauss_table =
        malloc_gslint_glfixed(hod_.n_nodes);
    for (int q=0; q<hod_.n_nodes; q++) {
      gsl_integration_glfixed_point(-1.0, 1.0, q,
                                    &hod_.gauss_legendre[0][q],
                                    &hod_.gauss_legendre[1][q], gauss_table);
    }
    gsl_integration_glfixed_table_free(gauss_table);
  }

  // Refill: cosmology, Ntable, the HOD, the clustering n(z), the lens
  // photo-z or the source n(z) changed. The source n(z) enters through
  // the a grid only: with magnification bias on, amax_lens
  // (redshift_spline.c) is the source edge 1/(1 + max(shear zmin_all,
  // 0.001)), so a new source n(z) moves the range p_gm and p_gg read
  // ngal and bgal on. The source photo-z shifts do not enter (amin_lens
  // and amax_lens never read them).
  if (fdiff2(hod_.cache[0], cosmology.random) ||
      fdiff2(hod_.cache[1], Ntable.random) ||
      fdiff2(hod_.cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(hod_.cache[3], redshift.random_clustering) ||
      fdiff2(hod_.cache[4], nuisance.random_photoz_clustering) ||
      fdiff2(hod_.cache[5], redshift.random_shear))
  {
    // The a grid: min, max, step. It must cover every a at which p_gm and
    // p_gg read ngal and bgal, i.e. every lens bin's [amin_lens, amax_lens]
    // (the rows of their per-bin grids): outside the grid ngal and bgal
    // are 0, the row's P = bgal P_delta + GM02/ngal is GM02/0 = inf, and
    // W_gal p_gm (cosmo2D.c) is 0 x inf = NaN in every gammat and w entry.
    // Two effects move the lens ranges (redshift_spline.c) past the
    // clustering table's [1/(1 + zmax_all), 1/(1 + zmin_all)]:
    //   - magnification bias (gbmag != 0) widens amax_lens to the source
    //     edge 1/(1 + max(shear zmin_all, 0.001)). DES Y3: the lens n(z)
    //     starts at z = 0.005 and the source n(z) at z = 0, so every
    //     bin's range ends at a = 1/1.001 = 0.999, past 1/1.005 = 0.995;
    //   - the lens photo-z shift and stretch move both ends: a bin whose
    //     n(z) reaches the table's top extends past 1/(1 + zmax_all) once
    //     its stretch or its 2|shift| padding exceeds the last z step.
    //     desy1xplanck MagLim: every bin reaches z = 1.58, zmax_all =
    //     1.59, and bins 1 and 4 (stretch 1.31 and 1.08) start at
    //     a = 0.333 and 0.379, below 1/2.59 = 0.386.
    // So the grid is the union of the table's range and every bin's.
    // When no bin reaches past the table, the grid (and every result) is
    // exactly the previous one. When a bin extends the top, the top gets
    // a 1e-12 relative margin: the last row amin + (na - 1) step of p_gm
    // and p_gg can round a few ulps past amax_lens (the first row is amin
    // exactly, so the bottom needs none).
    const double a_table_min = 1.0/(redshift.clustering_zdist_zall[RANGE_MAX] + 1.0);
    const double a_table_max = 1.0/(redshift.clustering_zdist_zall[RANGE_MIN] + 1.0);
    double a_lower = a_table_min;
    double a_upper = a_table_max;
    for (int l=0; l<redshift.clustering_nbin; l++) {
      a_lower = fmin(a_lower, amin_lens(l));
      a_upper = fmax(a_upper, amax_lens(l));
    }
    if (a_upper > a_table_max) {
      a_upper *= 1.0 + 1.e-12; // rounding margin (see above)
    }
    hod_.lim[0] = a_lower;
    hod_.lim[1] = a_upper;
    hod_.lim[2] = (hod_.lim[1] - hod_.lim[0])/((double) Ntable.N_a - 1.0);

    const int nbin    = hod_.nbin;
    const int n_a     = Ntable.N_a;
    const int n_nodes = hod_.n_nodes;

    // cold matter density of the mass function (POWER SPECTRA banner)
    const double rho_hmf = cosmology.rho_crit * omega_halo_field();

    (void) sigma2(limits.halo_m[RANGE_MIN], hod_.lim[0]);

    // --- 2. PER-REFILL PRECOMPUTATION ---
    // P_q = half_width w_q (rho_cb/M_q) <N|M_q>: the factors that
    // do not evolve with a. The peak height and dlnnu/dlnM are read
    // later at each (M_q,a). The HOD itself does not evolve; hod_.lim[0]
    // supplies an a inside the allowed range for HOD_nc's range check.
    for (int b=0; b<nbin; b++) {
      // lower bound 2 dex below the HOD lg M_min (N_c is an erf tail)
      const double lnMmin = log(10.0)*(nuisance.hod[b][0] - 2.);
      const double lnMmax = log(limits.halo_m[RANGE_MAX]);

      // GL map onto [lnMmin, lnMmax]: ln M_q = mid + half_width x_q
      const double half_width = 0.5*(lnMmax - lnMmin);
      const double mid        = 0.5*(lnMmax + lnMmin);

      const double fc = HOD_fc(b); // f_c of <N|M> = f_c N_c + N_s

      for (int q=0; q<n_nodes; q++) {
        const double lnM = mid + half_width*hod_.gauss_legendre[0][q];
        const double m   = exp(lnM);

        // <N|M_q>, the mean HOD occupation of a halo of mass M_q
        const double occupation = fc*HOD_nc(m, hod_.lim[0], b) +
                                  HOD_ns(m, hod_.lim[0], b);

        hod_.node_data[0][b][q] = m; // mass, read at every a node
        hod_.node_data[1][b][q] = half_width*hod_.gauss_legendre[1][q]
                                  *(rho_hmf/m)*occupation; // a-independent weight
      }
    }

    // The scale factors and the fit parameters of f and b at each node.
    // Serial: fnu_params_at builds its normalization table here.
    for (int j=0; j<n_a; j++) {
      const double a = hod_.lim[0] + j*hod_.lim[2];
      hod_.a_nodes[j]    = a; // the variance is evaluated at this scale factor
      hod_.bias_pars[j] = hb1nu_params_at(a);
      hod_.mult_pars[j] = fnu_params_at(a);
    }

    // --- 3. TABLE FILL: THE (BIN, a) OPENMP LOOP ---
    /* PHYSICAL DERIVATION & LOGIC FLOW
       1. nu = delta_c/sigma_cb(M_q,a_j)    peak height at mass node q
       2. P_q dlnnu/dlnM nu f(nu) = half_width w_q (dn/dlnM) <N|M_q>
       3. ngal(a_j) = sum_q P_q dlnnu/dlnM nu f(nu)
                    = int dlnM (dn/dlnM) <N|M>
       4. bgal(a_j) = sum_q P_q dlnnu/dlnM nu f(nu) b(nu) / ngal(a_j)
                    = (1/ngal) int dlnM (dn/dlnM) b(nu) <N|M>            */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b=0; b<nbin; b++) {
      for (int j=0; j<n_a; j++) {
        const double* restrict mass = hod_.node_data[0][b]; // M_q
        const double* restrict Pq  = hod_.node_data[1][b]; // P_q

        const double a               = hod_.a_nodes[j];
        const hb1nu_params* tinker_b = &hod_.bias_pars[j];
        const fnu_params* tinker_f   = &hod_.mult_pars[j];

        double n_gal  = 0.0; // sum_q P_q dlnnu/dlnM nu f(nu)       -> ngal(a_j)
        double bn_gal = 0.0; // sum_q P_q dlnnu/dlnM nu f(nu) b(nu) -> bgal x ngal

        for (int q=0; q<n_nodes; q++) {
          const double nu = delta_c/sqrt(sigma2(mass[q], a));
          // dn_gal = half_width w_q (dn/dlnM) <N|M>: node q's share
          const double dn_gal = Pq[q]*dlognudlogm(mass[q], a)*fnu_core(nu, tinker_f)*nu;
          n_gal  += dn_gal;
          bn_gal += dn_gal*hb1nu_core(nu, tinker_b); // times b(nu)
        }

        hod_.tab[0][b][j] = n_gal;        // ngal(a_j)
        hod_.tab[1][b][j] = bn_gal/n_gal; // bgal(a_j)
      }
    }

    // --- 4. CACHE TAGS: THE INPUTS THE TABLES NOW HOLD ---
    hod_.cache[0] = cosmology.random;
    hod_.cache[1] = Ntable.random;
    hod_.cache[2] = nuisance.random_galaxy_bias;
    hod_.cache[3] = redshift.random_clustering;
    hod_.cache[4] = nuisance.random_photoz_clustering;
    hod_.cache[5] = redshift.random_shear;
  }
}


// ---------------------------------------------------------------------------
// Comoving number density of the galaxies of lens bin ni at scale factor
// a: the halo mass function times the mean HOD occupation, summed over
// halo mass,
//
//   ngal(a) = int dlnM (dn/dlnM) [f_c N_c(M) + N_s(M)],
//
// tabulated by hod_tables (its header) and interpolated linearly on its
// a grid; 0 outside that grid ([1/(1 + z_max), 1/(1 + z_min)] of the
// clustering n(z), widened to every lens bin's [amin_lens, amax_lens]).
//
// Parameters:
//   ni - lens bin, 0 <= ni < redshift.clustering_nbin (aborts otherwise)
//   a  - scale factor
//
// Returns:
//   ngal in (c/H0)^-3; 0 outside the a grid
// ---------------------------------------------------------------------------
double ngal(const int ni, const double a)
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }

  hod_tables();

  // 0 outside the a grid of hod_ (header); linear in a inside
  if ((a < hod_.lim[0]) || (a > hod_.lim[1])) {
    return 0.0;
  }

  return interpol1d(hod_.tab[0][ni], Ntable.N_a, hod_.lim[0], hod_.lim[1],
                    hod_.lim[2], a);
}


// ---------------------------------------------------------------------------
// Mean halo bias of the galaxies of lens bin ni at scale factor a: the
// halo bias b(nu) (hb1nu) of the host halos, weighted by how many galaxies
// each halo mass contributes,
//
//   bgal(a) = int dlnM (dn/dlnM) [f_c N_c(M) + N_s(M)] b(nu) / ngal(a),
//
// tabulated by hod_tables (its header) and interpolated linearly on its
// a grid; 0 outside that grid (ngal's header). The large-scale galaxy
// bias of p_gm and p_gg.
//
// Parameters:
//   ni - lens bin, 0 <= ni < redshift.clustering_nbin (aborts otherwise)
//   a  - scale factor
//
// Returns:
//   bgal, dimensionless; 0 outside the a grid
// ---------------------------------------------------------------------------
double bgal(const int ni, const double a)
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }

  hod_tables();

  // 0 outside the a grid of hod_ (header); linear in a inside
  if ((a < hod_.lim[0]) || (a > hod_.lim[1])) {
    return 0.0;
  }

  return interpol1d(hod_.tab[1][ni], Ntable.N_a, hod_.lim[0], hod_.lim[1],
                    hod_.lim[2], a);
}



// ============================================================================
// [SECTION] HALO MODEL POWER SPECTRA
// ============================================================================
//
//   P_XY(k, a) = I02_XY(k, a) + I11_X(k, a) I11_Y(k, a) P_lin(k, a)
//                                                  (2005.00009 Eqs. 1-2)
//   I02_XY = int dlnM dn/dlnM W_X(M, k) W_Y(M, k)              1-halo
//   I11_X  = int dlnM dn/dlnM b(nu) W_X(M, k)
//            + A(a) W_X(M_min, k)/(M_min/rho_m)                2-halo
//
// (the I^0_2 and I^1_1 of the file glossary), with
//
//   dn/dlnM = (rho_cb/M) nu f(nu) dlnnu/dlnM,  nu = delta_c/sigma_cb(M,a)
//
// the mass function (fnu, dlognudlogm; sigma2 in cosmo3D.c), b the halo
// bias (hb1nu) and the windows W_X the Fourier transforms of the profiles,
// in the units of the field times a volume.
//
// Halos are assigned to cold dark matter plus baryons: free-streaming
// neutrinos are not included in their mass. Thus sigma2 is sigma_cb^2
// at the requested a, and the mass-function density is
//
//   rho_hmf = rho_cb = rho_crit (Omega_m - Omega_nu).
//
// The code obtains this density through omega_halo_field(). In contrast,
// M/rho_m in the lensing matter window and r_200m's overdensity
// definition refer to total matter. The galaxy spectra use b_gal times
// the total nonlinear spectrum for their 2-halo term (DES convention).
// The uncompiled halo-model matter spectrum in future_port_unfinished/
// still needs its separate massive-neutrino terms before it can be used.
//
// The matter window:
//
//   matter    W_m = (M/rho_m) u(k|M),  u -> 1 at k -> 0   (c/H0)^3
//
// (the pressure windows of p_my and p_yy: the header of
// future_port_unfinished/halo_tsz.c).
//
// A(a) = 1 - bias_norm(a) is the HMx correction (2005.00009 App. A) for
// the halos below M_min, which hold about 17% of the bias-weighted matter
// at z = 0 (bias_norm header): their share is put back as halos of mass
// exactly M_min, n(M) -> n(M) + A delta_D(M - M_min)/[b(M_min)
// M_min/rho_m] (Eq. A7), so that I11_m -> 1 and P_2h -> P_lin at k -> 0
// at every a, while at high k the added halos stay point-like as the
// real light halos are (r_Delta = 0.5 kpc/h at M_min = 1e4 M_sun/h).
//
// The galaxy spectra take Pdelta b_gal as their 2-halo term instead of
// I11 (p_gm, p_gg headers).
//
// Each builder tabulates ln P on a uniform (a, ln k) grid with a
// Gauss-Legendre rule in ln M (the node count follows an
// Ntable.high_def_integration ladder, set in each rebuild block) and
// reads it bilinearly; the first call of a refill is halo_warmup.
//
// The structure the builders share, written for the template
//
//   P_mm = I02 + I11^2 P_lin                          (2005.00009 Eqs. 1-2)
//   I02  = int dlnM dn/dlnM (M/rho_m)^2 u(k|M)^2
//   I11  = int dlnM dn/dlnM b(nu) (M/rho_m) u(k|M) + A(a) u(k|M_min)
//
// (the halo-model matter spectrum p_mm itself is kept, not compiled, in
// future_port_unfinished/halo_pmm.c):
//
// 1. Quadrature: the n-point Gauss-Legendre rule in ln M (exact for
// polynomials of degree 2n - 1; n follows an
// Ntable.high_def_integration ladder); nodes M_q and weights w_q on
// [ln M_min, ln M_max] are mapped once in the rebuild block
// (gsl_integration_glfixed_point). High k converges slowest: the
// profile's ringing is sampled in ln M.
//
// 2. Loop levels: each factor is computed at the outermost level it
// depends on, so the innermost loop is the NFW kernel alone (nfw_um:
// three table reads and two sines; no pow, exp or log):
//
//   per refill, per node q       M_q, w_q;
//     (mass_node)                w_q (rho_cb/M_q); M_q/rho_m;
//                                r_Delta(M_q)
//   per a row i, threaded        f, b fit parameters; A(a); c(M_min)
//   per (i, q) (a_node[i])       nu = delta_c/sigma_cb(M,a); c = conc(M, a);
//                                ln(1+c); m(c) = ln(1+c) - c/(1+c);
//                                r_s = r_Delta/c, ln r_s;
//                                w1h_q = dn (M/rho_m)^2/m(c)^2;
//                                w2h_q = dn b(nu) (M/rho_m)/m(c), with
//                                dn = w (rho_cb/M) dlnnu/dlnM f(nu) nu
//   per (i, k), sum over q       x = k r_s, ln x = ln k + ln r_s,
//                                um = u m(c) = nfw_um(c, x, ln x, ln(1+c));
//                                I02 = sum w1h_q um^2,
//                                I11 = sum w2h_q um + A u(k|M_min); ln P
//
// The 1/m(c) of u = um/m(c) lives in w1h_q and w2h_q. The rows read the
// NFW kernel directly, so like.halo_model[3] must be HALO_PROFILE_NFW;
// anything else aborts.


// ---------------------------------------------------------------------------
// Warm-up of the spectrum table builders: one single-threaded call to
// each function with lazily built static state that the threaded loops
// of p_gm and p_gg read, so that inside the loops
// those functions only read (the warm-up rule of the cosmo2D.c _work
// functions). Each call is one table read once its build has run; the
// values are thrown away, which the (void) casts say. The builds
// themselves thread (sigma2, dlognudlogm, bias_norm, hod_tables,
// tinker_alpha and nfw_table each own a parallel loop): one more
// reason they must start outside a parallel region.
//
//   sigma2, dlognudlogm   the ln M tables (cosmo3D.c; above); keys
//                         cosmology.random, Ntable.random
//   fnu_params_at         the tinker_alpha table (HMF_TINKER_2010); keys
//                         like.halo_model[0..1], Ntable.random
//   nfw_table             the NFW f, G table nfw_, read by the rows
//                         through nfw_um; key Ntable.random
//   bias_norm             the HMx term of the I11 2-halo spectra
//                         (matter); keys cosmology, Ntable
//   ngal                  the hod_ tables of ngal and bgal (galaxies);
//                         keys cosmology, Ntable, the HOD tag, the
//                         clustering n(z), lens photo-z and source n(z)
//                         tags
//   Pdelta                its run-mode latch, a static set on the
//                         first call (galaxies: the 2-halo term)
//
// Static-free, so absent: growfac, p_lin, p_nonlin, PkRatio_baryons
// (the CAMB-fed cosmology tables); hb1nu_params_at and the *_core
// kernels; conc (a sigma2 read); HOD_nc, HOD_ns, HOD_fc; nfw_um.
//
// Preconditions, checked by the callees: 0 < a < 1 (fnu_params_at);
// hod = 1 needs the HOD of every lens bin set (HOD_nc aborts inside
// hod_tables); gas = 1 aborts (the gas tables are not compiled).
//
// Parameters:
//   a   - a scale factor of the builder's a grid, 0 < a < 1
//   k   - a wavenumber of the builder's k grid, (c/H0)^-1
//   gas - must be 0: the gas tables (u_KS) and the y spectra that set 1
//         are in future_port_unfinished/halo_tsz.c; 1 aborts
//   hod - 1: the galaxy spectra (p_gm, p_gg) read ngal, bgal and
//         Pdelta; 0: the I11 spectra (future_port_unfinished/) read
//         bias_norm. p_gm and p_gg pass 1; ia_tables passes 0 (it reads
//         neither table, and 1 would build the lens HOD tables, which
//         abort when a lens bin's HOD is not set)
//
// Returns:
//   nothing
// ---------------------------------------------------------------------------
static void halo_warmup(
    const double a,  // scale factor of the builder's a grid, 0 < a < 1
    const double k,  // wavenumber of the builder's k grid, (c/H0)^-1
    const int gas,   // must be 0 (1 aborts: gas tables not compiled)
    const int hod    // 1 = the galaxy spectra read ngal, bgal, Pdelta
  )
{
  const double mmin = limits.halo_m[RANGE_MIN];

  (void) sigma2(mmin, a);
  (void) dlognudlogm(mmin, a);
  (void) fnu_params_at(a);
  nfw_table();

  if (1 == hod) {
    // 2-halo term Pdelta bgal: the hod_ tables and the Pdelta latch
    (void) ngal(0, a);
    (void) Pdelta(k, a);
  }
  else {
    // 2-halo term I11^2 P_lin, I11 with the HMx share 1 - bias_norm
    (void) bias_norm(a);
  }

  if (1 == gas) {
    // the gas tables (u_KS) are not compiled: future_port_unfinished/
    // halo_tsz.c holds them and this branch (its header)
    log_fatal("halo_warmup: gas = 1, but the gas profiles are not compiled "
              "(future_port_unfinished/halo_tsz.c)");
    exit(1);
  }
}


// ---------------------------------------------------------------------------
// Dense ln k values of a smooth function known exactly on a coarse ln k
// grid: the 1-halo sums of p_gm and p_gg are evaluated at every
// k_step-th ln k node only (plus pad nodes beyond each end, which keep
// the natural spline's end conditions away from the table's range) and
// filled in between by a natural cubic spline of their logarithm.
//
//   coarse node c sits at ln k = lnk_first + (c - pad) k_step dlnk,
//   dense node j at ln k = lnk_first + j dlnk
//
// The spline follows spline_coeffs_uniform (basics.h): curvatures from
// the tridiagonal system [1 4 1] c = (3/h^2) (second differences),
// c = 0 at both ends, solved by the Thomas algorithm; each interval is
// then y_q + b t + c_q t^2 + d t^3 in Horner form. The Thomas
// multipliers depend only on the node count, so the caller computes
// them once per rebuild (ln_k_spline_multipliers); all other storage is
// the caller's thread-private scratch, so nothing is allocated here.
//
// Parameters:
//   ln_coarse - [n_coarse] the log of the exact coarse values
//   n_coarse  - coarse nodes, pads included
//   k_step    - dense nodes per coarse interval
//   pad       - pad nodes before the first dense node
//   dlnk      - dense ln k spacing
//   n_dense   - dense nodes
//   mult      - [n_coarse] the Thomas multipliers
//   curv      - [n_coarse] scratch: the spline curvatures c_q
//   ln_dense  - [n_dense] output: the spline at the dense nodes
// ---------------------------------------------------------------------------
static void ln_k_spline_multipliers(
    const int n_coarse,
    double* restrict mult
  )
{
  mult[0] = 0.0;
  for (int q=1; q<n_coarse-1; q++) {
    mult[q] = 1.0/(4.0 - mult[q-1]);
  }
  mult[n_coarse-1] = 0.0;
}


static void ln_k_spline_upsample(
    const double* restrict ln_coarse,
    const int n_coarse,
    const int k_step,
    const int pad,
    const double dlnk,
    const int n_dense,
    const double* restrict mult,
    double* restrict curv,
    double* restrict ln_dense
  )
{
  const double h         = k_step*dlnk; // coarse spacing in ln k
  const double inv_h     = 1.0/h;
  const double inv_3h    = 1.0/(3.0*h);
  const double h_third   = h/3.0;
  const double inv_h2x3  = 3.0/(h*h);

  // --- 1. CURVATURES: THOMAS SOLVE OF THE NATURAL-SPLINE SYSTEM ---

  curv[0] = 0.0;
  for (int q=1; q<n_coarse-1; q++) {
    const double rhs =
        inv_h2x3*(ln_coarse[q-1] - 2.0*ln_coarse[q] + ln_coarse[q+1]);
    curv[q] = (rhs - curv[q-1])*mult[q];
  }
  curv[n_coarse-1] = 0.0;
  for (int q=n_coarse-2; q>0; q--) {
    curv[q] -= mult[q]*curv[q+1];
  }

  // --- 2. HORNER EVALUATION, ONE COARSE INTERVAL AT A TIME ---

  // b and d once per interval; its k_step dense nodes sit at the
  // offsets t = r dlnk, r = 0 .. k_step - 1
  int j = 0;
  for (int q=pad; j<n_dense; q++) {
    const double y = ln_coarse[q];
    const double c = curv[q];
    const double b = (ln_coarse[q+1] - y)*inv_h
                     - h_third*(curv[q+1] + 2.0*c);
    const double d = (curv[q+1] - c)*inv_3h;

    for (int r=0; r<k_step && j<n_dense; r++) {
      const double t = r*dlnk;
      ln_dense[j] = y + t*(b + t*(c + t*d));
      j++;
    }
  }
}



// ---------------------------------------------------------------------------
// P_gm(k, a, ni), the halo-model galaxy-matter power spectrum of lens bin
// ni, from a table of ln P per bin on na x Ntable.N_k_nlin nodes, na =
// Ntable.halo_na_lens, uniform in a over the bin's range [amin_lens,
// amax_lens]
// and in ln k, read bilinearly (interpol2d) and exponentiated:
//
//   P_gm = P_delta b_gal + GM02/n_gal
//   GM02 = int dlnM dn/dlnM (M/rho_m) u_m(k|M) [N_s u_g(k|M) + f_c N_c]
//
// 2-halo: the nonlinear matter spectrum (Pdelta) times the mean galaxy
// bias (bgal). 1-halo: the satellite-matter and central-matter pairs of
// one halo per galaxy (ngal): satellites follow u_g, the NFW profile at
// c_g = gc c, gc = nuisance.gc[ni] (file glossary); the central sits at the
// center (window 1). dn/dlnM, (M/rho_m) u_m as in the section banner;
// N_c, N_s, f_c the occupation of the GALAXY PROFILES banner.
//
// 1. Quadrature: the Gauss-Legendre rule of the section banner
// (item 1) over ln M from ln 10^(lg M_min - 1) of the bin to
// ln limits.halo_m[RANGE_MAX]: nodes x_q, weights w_q on [-1, 1] (gl) are
// mapped per bin in the refill, ln M_q = mid + half_width x_q, weight
// half_width w_q.
//
// 2. Loop levels as in the section banner (item 2): the innermost loop is
// the NFW kernel nfw_um alone. The occupation does not depend on a
// (HOD_nc, HOD_ns only range-check it: amin_lens is a placeholder):
//
//   per refill, per (bin, node) -> bin_tab:
//     M, half_width w (rho_cb/M), r_Delta, N_s, f_c N_c
//   per bin, per a row, threaded:
//     a; Tinker f parameters; ngal, bgal
//   per (a, node) -> a_tab:
//     c = conc(M, a) and c_g = gc c, each with ln(1+c), m(c), r_s, ln r_s;
//     w_matter = dn_halo (M/rho_m)/m(c);
//     W1 = w_matter N_s/m(c_g), W0 = w_matter f_c N_c
//   per (a, k), sum over nodes:
//     um = u_m m(c), ug = u_g m(c_g); GM02 = sum um (W1 ug + W0); ln P
//
// w_matter carries the 1/m(c) of the matter leg, W1 the 1/m(c_g) of the
// satellite leg (the central has no profile). gc = 1 makes c_g = c
// exactly, so ug = um bitwise and one kernel call serves both legs (the
// same_conc branch): half the kernel calls. like.halo_model[3] must be
// HALO_PROFILE_NFW (the rows read nfw_um directly) and nuisance.gc[l] > 0
// in every bin (u_g's condition, checked in the refill); else abort.
//
// Thread safety: the single-threaded halo_warmup call before the
// threaded loops builds every lazy table the rows read (its header). One
// parallel region per bin.
//
// Cache invalidation:
//   rebuild block (table, lim, gl, bin_tab, a_tab, k_tab, k_mult; every
//     allocation lives here, one block each from malloc1d/malloc2d/
//     malloc3d): Ntable.random or redshift.random_clustering (bin count:
//     clustering n(z))
//   refill: cosmology.random, Ntable.random, nuisance.random_galaxy_bias
//     (HOD, gc, and the magnification bias that widens amax_lens),
//     redshift.random_clustering, nuisance.random_photoz_clustering or
//     redshift.random_shear (the source n(z) edge amax_lens widens to);
//     the per-bin a ranges (amin_lens, amax_lens: they move with the
//     lens photo-z shift and stretch) are set at every refill
//
// Parameters:
//   k  - wavenumber in (c/H0)^-1
//   a  - scale factor
//   ni - lens bin, 0 <= ni < redshift.clustering_nbin (aborts otherwise)
//
// Returns:
//   P_gm(k, a, ni) in (c/H0)^3; 0 outside [amin_lens(ni), amax_lens(ni)];
//   ln P continued with unit slope outside [ln k_min, ln k_max] (interpol2d)
// ---------------------------------------------------------------------------
double p_gm(
    const double k,
    const double a,
    const int ni
  )
{
  static uint64_t  cache[MAX_SIZE_ARRAYS];
  static double*** table   = NULL;
  static double**  lim     = NULL; // [nbin+1][3]: row l < nbin the a grid
                                   // of bin l (min, max, step); row nbin
                                   // the ln k grid (min, max, step)
  static int       nbin    = 0;    // lens bins of the allocation
  static int       na      = 0;    // a nodes per bin
  static int       nnode   = 0;    // Gauss-Legendre mass nodes in ln M
  static double**  gl      = NULL; // [2][nnode] GL nodes (0), weights (1)
                                   // on [-1, 1]
  static double*** bin_tab = NULL; // [nbin][5][nnode] per (bin, mass
                                   // node): M, half_width w (rho_cb/M)
                                   // r_Delta,
                                   // N_s, f_c N_c
  static double*** a_tab   = NULL; // [n_threads][10][nnode]: one (bin,
                                   // a-row) iteration's scratch: c,
                                   // ln(1+c), r_s, ln r_s and the same
                                   // for c_g = gc c, then W1, W0
  static double*** k_tab   = NULL; // [n_threads][3][n_dense + pads]:
                                   // the coarse ln k scratch (coarse
                                   // ln GM02, curvatures,
                                   // dense ln GM02)
  static int       k_step  = 0;    // dense ln k nodes per coarse one
  static double*   k_mult  = NULL; // [n_coarse] Thomas multipliers
  static int       n_coarse = 0;   // coarse ln k nodes, pads included
  const int        K_PAD   = Ntable.halo_spline_pad; // pads per end

  // --- 1. REBUILD: SIZES, ALLOCATIONS, GL RULE, TABLE AXES ---

  // first call, or the Ntable or clustering-n(z) tag differs from the
  // allocation's; every allocation is one block from malloc1d/malloc2d/
  // malloc3d, so one free each (header, Cache invalidation)
  if (NULL == table ||
      fdiff2(cache[1], Ntable.random) ||
      fdiff2(cache[3], redshift.random_clustering))
  {
    if (table != NULL) {
      free(table);
      free(lim);
      free(gl);
      free(bin_tab);
      free(a_tab);
      free(k_tab);
      free(k_mult);
    }

    nbin  = redshift.clustering_nbin;
    na    = Ntable.halo_na_lens; // a nodes per lens bin
    // mass-node ladder: the default already lands far inside the
    // code's chi2 error budget (measured ladder: the skill file's
    // halo.c numerics); high_def_integration steps toward the largest
    // GSL rule
    if (0 == abs(Ntable.high_def_integration)) {
      nnode = Ntable.halo_nm;
    }
    else if (1 == abs(Ntable.high_def_integration)) {
      nnode = 2*Ntable.halo_nm;
    }
    else if (2 == abs(Ntable.high_def_integration)) {
      nnode = 4*Ntable.halo_nm;
    }
    else {
      nnode = 1024;
    }

    // coarse ln k step of the 1-halo sums (ln_k_spline_upsample): its
    // ladder, like the mass nodes', lands inside the chi2 error budget
    // at the default and becomes exact with high_def_integration
    if (0 == abs(Ntable.high_def_integration)) {
      k_step = Ntable.halo_nk_step;
    }
    else if (1 == abs(Ntable.high_def_integration)) {
      k_step = Ntable.halo_nk_step/2;
    }
    else {
      k_step = 1;
    }
    if (k_step < 1) {
      k_step = 1;
    }
    n_coarse = (Ntable.N_k_nlin - 1)/k_step + 2 + 2*K_PAD;

    table   = (double***) malloc3d(nbin, na, Ntable.N_k_nlin);
    lim     = (double**) malloc2d(nbin+1, 3);
    gl      = (double**) malloc2d(2, nnode);
    bin_tab = (double***) malloc3d(nbin, 5, nnode);
    // one scratch block per thread (the thread count of this rebuild;
    // raising OMP_NUM_THREADS afterwards requires an Ntable bump)
    a_tab   = (double***) malloc3d(omp_get_max_threads(), 10, nnode);
    k_tab   = (double***) malloc3d(omp_get_max_threads(), 3,
                                   Ntable.N_k_nlin + n_coarse);
    k_mult   = (double*) malloc1d(n_coarse);
    ln_k_spline_multipliers(n_coarse, k_mult);

    // gsl_integration_glfixed_point(lo, hi, q, &x, &w, t): node q of the
    // rule t mapped onto [lo, hi], and its weight; kept on [-1, 1] here
    gsl_integration_glfixed_table* t = malloc_gslint_glfixed(nnode);
    for (int q=0; q<nnode; q++) {
      gsl_integration_glfixed_point(-1.0, 1.0, q, &gl[0][q], &gl[1][q], t);
    }
    gsl_integration_glfixed_table_free(t);

    // ln k grid, shared by all bins
    lim[nbin][0] = log(limits.k_cH0[RANGE_MIN]);
    lim[nbin][1] = log(limits.k_cH0[RANGE_MAX]);
    lim[nbin][2] = (lim[nbin][1]-lim[nbin][0])
                   /((double) Ntable.N_k_nlin - 1.0);
  }

  // --- 2. REFILL: THE HOD-WEIGHTED ln P TABLE ---

  // any of the six tags differs from the table's
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], Ntable.random)    ||
      fdiff2(cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(cache[3], redshift.random_clustering)  ||
      fdiff2(cache[4], nuisance.random_photoz_clustering) ||
      fdiff2(cache[5], redshift.random_shear))
  {
    // a grid of bin l over its lens range, node i at lim[l][0] + i
    // lim[l][2], both ends included. Set at every refill, not in the
    // rebuild block: the range moves with the lens photo-z shift and
    // stretch, and amax_lens widens to the source n(z)'s lower edge
    // when magnification bias is on (the redshift.random_shear tag)
    for (int l=0; l<nbin; l++) {
      lim[l][0] = amin_lens(l);
      lim[l][1] = amax_lens(l);
      lim[l][2] = (lim[l][1] - lim[l][0])/((double) na - 1.0);
    }

    // --- 2a. GUARDS AND WARM-UP ---

    // the k rows read the NFW kernel directly (header, item 2)
    if (like.halo_model[3] != HALO_PROFILE_NFW) {
      log_fatal("like.halo_model[3] = %d not supported", like.halo_model[3]);
      exit(1);
    }

    // every lazy table the threaded loops read is built here, on one
    // thread (header, Thread safety)
    halo_warmup(lim[0][0], exp(lim[nbin][0]), 0, 1);

    /* PHYSICAL DERIVATION & LOGIC FLOW (P_gm and GM02: header above)
       1. bin_tab, per (bin, mass node): M, weighted dn/dlnM factors,
          r_Delta, occupation N_s and f_c N_c
       2. per a row, threaded: sigma_cb(M,a), its mass slope, the
          nu-independent f(nu) half, n_gal, b_gal
       3. thread scratch a_tab, per node of one a row: c, c_g = gc c,
          scale radii, logs, leg weights W1 (satellite), W0 (central)
       4. per k: GM02 = sum_q um (W1 ug + W0); table = ln P_gm */

    // --- 2b. PER (BIN, MASS NODE), SERIAL ---

    // the GL nodes mapped onto the bin's ln M range, the a-independent
    // factors, and the occupation at the placeholder a = amin
    const double rho_m     = cosmology.rho_crit * cosmology.Omega_m;
    const double rho_delta = Delta * rho_m;
    // The cb density normalizes dn/dlnM (POWER SPECTRA banner).
    const double rho_hmf   = cosmology.rho_crit * omega_halo_field();

    for (int l=0; l<nbin; l++) {
      // u_g's condition, checked for every bin at once
      if (!(nuisance.gc[l] > 0)) {
        log_fatal("galaxy concentration factor gc[%d] = %g must be > 0",
                  l, nuisance.gc[l]);
        exit(1);
      }

      // GL map [-1, 1] -> [ln 10^(lg M_min - 1), ln M_max]: one decade
      // below the bin's minimum HOD mass up to the global maximum
      const double lnMmin     = log(10.)*(nuisance.hod[l][0] - 1.0);
      const double lnMmax     = log(limits.halo_m[RANGE_MAX]);
      const double half_width = 0.5*(lnMmax - lnMmin);
      const double mid        = 0.5*(lnMmax + lnMmin);

      const double fc = HOD_fc(l);

      // bin_tab rows: M | half_width w_q (rho_hmf/M) |
      // r_Delta from M = (4 pi/3) Delta rho_m r_Delta^3 |
      // N_s | f_c N_c
      for (int q=0; q<nnode; q++) {
        const double m = exp(mid + half_width*gl[0][q]);
        bin_tab[l][0][q] = m;
        bin_tab[l][1][q] = half_width*gl[1][q]*(rho_hmf/m);
        bin_tab[l][2][q] = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
        bin_tab[l][3][q] = HOD_ns(m, lim[l][0], l);
        bin_tab[l][4][q] = fc*HOD_nc(m, lim[l][0], l);
      }
    }

    // --- 2c. (BIN, a ROW) PAIRS COLLAPSED AND THREADED ---
    #pragma omp parallel for collapse(2) schedule(static)
    for (int l=0; l<nbin; l++) {
      for (int i=0; i<na; i++) {
        const double gc = nuisance.gc[l];

        // gc = 1 makes c_g = c bitwise: one kernel call serves both
        const int same_conc = (1.0 == gc);

        const double ai = lim[l][0] + i*lim[l][2];

        // The nu-independent Tinker f(nu) half, and the mean
        // galaxy density and bias of this a row
        const fnu_params fnu_pars = fnu_params_at(ai);
        const double     n_gal    = ngal(l, ai);
        const double     b_gal    = bgal(l, ai);

        // thread-private scratch: this (bin, a-row) iteration fills
        // it and consumes it in its own k loop; restrict: each row is
        // reached only through its pointer, no reload after libm calls
        double** const wsp   = a_tab[omp_get_thread_num()];
        double** const k_wsp = k_tab[omp_get_thread_num()];
        double* restrict conc_halo = wsp[0];
        double* restrict ln1c_halo = wsp[1];
        double* restrict r_s       = wsp[2];
        double* restrict lnrs      = wsp[3];
        double* restrict conc_gal  = wsp[4];
        double* restrict ln1c_gal  = wsp[5];
        double* restrict r_sg      = wsp[6];
        double* restrict lnrsg     = wsp[7];
        double* restrict w1        = wsp[8];
        double* restrict w0        = wsp[9];

        // per (a, node): both concentrations, scale radii, the logs of
        // each, and the weights W1, W0 with 1/m(c), 1/m(c_g) folded in
        for (int q=0; q<nnode; q++) {
          const double m  = bin_tab[l][0][q];
          const double nu = delta_c/sqrt(sigma2(m, ai));

          // c and c_g = gc c, with m(c) = ln(1+c) - c/(1+c) for each
          const double c     = conc(m, ai);
          const double cg    = c*gc;
          const double ln1c  = log1p(c);
          const double ln1cg = log1p(cg);
          const double mc    = ln1c - c/(1.0 + c);
          const double mcg   = ln1cg - cg/(1.0 + cg);

          // dn_halo = quadrature weight x dn/dlnM (Tinker);
          // w_matter carries the matter leg's (M/rho_m) and 1/m(c)
          const double dn_halo  = bin_tab[l][1][q]*dlognudlogm(m, ai)*fnu_core(nu, &fnu_pars)*nu;
          const double w_matter = dn_halo*(m/rho_m)/mc;

          conc_halo[q] = c;
          ln1c_halo[q] = ln1c;
          r_s[q]       = bin_tab[l][2][q]/c;
          lnrs[q]      = log(r_s[q]);
          conc_gal[q]  = cg;
          ln1c_gal[q]  = ln1cg;
          r_sg[q]      = bin_tab[l][2][q]/cg;
          lnrsg[q]     = log(r_sg[q]);
          w1[q]        = w_matter*bin_tab[l][3][q]/mcg;
          w0[q]        = w_matter*bin_tab[l][4][q];
        }

        // per k: the 1-halo sum on the coarse ln k grid only (the
        // expensive part: nnode NFW kernels per k), its log splined to
        // the dense grid, then ln P with the 2-halo term read exactly
        // at every dense node
        double* restrict ln_coarse = k_wsp[0];
        double* restrict curv      = k_wsp[1];
        double* restrict ln_dense  = k_wsp[2];

        const double dlnk      = lim[nbin][2];
        const double lnk_first = lim[nbin][0] - K_PAD*k_step*dlnk;

        for (int c=0; c<n_coarse; c++) {
          const double lnk = lnk_first + c*k_step*dlnk;
          const double kj  = exp(lnk);

          double gm02 = 0.0;
          // Sum um*(w1*ug + w0), using ug=um for equal concentrations.
          // Evaluate four nodes q, q+1, q+2, q+3 per step
          // (one per lane; nfw_um4 = nfw_um on each lane,
          // bitwise) into four-lane partial sums, added in a fixed lane
          // order (simd_horizontal_sum), then a scalar tail. The
          // summation order differs from the scalar loop's (last digits
          // of GM02) but never depends on the thread count.

          // k and ln k of this column in all four lanes
          const v4d vk   = simde_mm256_set1_pd(kj);   // k
          const v4d vlnk = simde_mm256_set1_pd(lnk);  // ln k

          // the four-lane partial sums of gm02, from zero
          v4d vgm02 = simde_mm256_setzero_pd();

          int q = 0;
          if (same_conc) {
            for (; q<=nnode-4; q+=4) {
              // the four arguments of nfw_um at nodes q..q+3; scalar:
              //   nfw_um(conc_halo[q], kj*r_s[q], lnk + lnrs[q],
              //          ln1c_halo[q])

              // c, the halo concentrations of nodes q..q+3
              const v4d vconc_halo = simde_mm256_loadu_pd(conc_halo + q);

              // r_s of nodes q..q+3
              const v4d vrs_halo = simde_mm256_loadu_pd(r_s + q);

              // x = k r_s
              const v4d vkrs_halo = simde_mm256_mul_pd(vk, vrs_halo);

              // ln r_s of nodes q..q+3
              const v4d vlnrs_halo = simde_mm256_loadu_pd(lnrs + q);

              // ln x = ln k + ln r_s
              const v4d vlnkrs_halo = simde_mm256_add_pd(vlnk, vlnrs_halo);

              // ln(1 + c) of nodes q..q+3
              const v4d vln1c_halo = simde_mm256_loadu_pd(ln1c_halo + q);

              // um = u m(c) at nodes q..q+3
              const v4d vum = nfw_um4(vconc_halo, vkrs_halo, vlnkrs_halo,
                                      vln1c_halo);

              // the weights of nodes q..q+3
              const v4d vw1 = simde_mm256_loadu_pd(w1 + q);  // W1
              const v4d vw0 = simde_mm256_loadu_pd(w0 + q);  // W0

              // scalar: gm02 += um*(w1[q]*um + w0[q])

              // W1 um + W0, fused: satellites at u_g = u, the central at 1
              const v4d vgal_weight = nfw_fmadd4(vw1, vum, vw0);

              // um (W1 um + W0) + gm02, lane by lane, fused
              vgm02 = nfw_fmadd4(vum, vgal_weight, vgm02);
            }
          }
          else {
            for (; q<=nnode-4; q+=4) {
              // the four arguments of nfw_um at nodes q..q+3, the halo
              // profile; scalar:
              //   nfw_um(conc_halo[q], kj*r_s[q], lnk + lnrs[q],
              //          ln1c_halo[q])

              // c, the halo concentrations of nodes q..q+3
              const v4d vconc_halo = simde_mm256_loadu_pd(conc_halo + q);

              // r_s of nodes q..q+3
              const v4d vrs_halo = simde_mm256_loadu_pd(r_s + q);

              // x = k r_s
              const v4d vkrs_halo = simde_mm256_mul_pd(vk, vrs_halo);

              // ln r_s of nodes q..q+3
              const v4d vlnrs_halo = simde_mm256_loadu_pd(lnrs + q);

              // ln x = ln k + ln r_s
              const v4d vlnkrs_halo = simde_mm256_add_pd(vlnk, vlnrs_halo);

              // ln(1 + c) of nodes q..q+3
              const v4d vln1c_halo = simde_mm256_loadu_pd(ln1c_halo + q);

              // um = u m(c) at nodes q..q+3
              const v4d vum = nfw_um4(vconc_halo, vkrs_halo, vlnkrs_halo,
                                      vln1c_halo);

              // the same four arguments for the galaxy profile; scalar:
              //   nfw_um(conc_gal[q], kj*r_sg[q], lnk + lnrsg[q],
              //          ln1c_gal[q])

              // c_g, the galaxy concentrations of nodes q..q+3
              const v4d vconc_gal = simde_mm256_loadu_pd(conc_gal + q);

              // r_s,g of nodes q..q+3
              const v4d vrs_gal = simde_mm256_loadu_pd(r_sg + q);

              // k r_s,g
              const v4d vkrs_gal = simde_mm256_mul_pd(vk, vrs_gal);

              // ln r_s,g of nodes q..q+3
              const v4d vlnrs_gal = simde_mm256_loadu_pd(lnrsg + q);

              // ln(k r_s,g) = ln k + ln r_s,g
              const v4d vlnkrs_gal = simde_mm256_add_pd(vlnk, vlnrs_gal);

              // ln(1 + c_g) of nodes q..q+3
              const v4d vln1c_gal = simde_mm256_loadu_pd(ln1c_gal + q);

              // ug = u_g m(c_g) at nodes q..q+3
              const v4d vug = nfw_um4(vconc_gal, vkrs_gal, vlnkrs_gal,
                                      vln1c_gal);

              // the weights of nodes q..q+3
              const v4d vw1 = simde_mm256_loadu_pd(w1 + q);  // W1
              const v4d vw0 = simde_mm256_loadu_pd(w0 + q);  // W0

              // scalar: gm02 += um*(w1[q]*ug + w0[q])

              // W1 ug + W0, fused: satellites at u_g, the central at 1
              const v4d vgal_weight = nfw_fmadd4(vw1, vug, vw0);

              // um (W1 ug + W0) + gm02, lane by lane, fused
              vgm02 = nfw_fmadd4(vum, vgal_weight, vgm02);
            }
          }

          // lane 0 + lane 1 + lane 2 + lane 3 of the partial sums
          gm02 = simd_horizontal_sum(vgm02);

          // scalar tail: nnode not a multiple of four (ug = um when
          // same_conc, as in the scalar path)
          for (; q<nnode; q++) {
            const double um = nfw_um(conc_halo[q], kj*r_s[q],
                                     lnk + lnrs[q], ln1c_halo[q]);
            double ug = um;
            if (!same_conc) {
              ug = nfw_um(conc_gal[q], kj*r_sg[q],
                          lnk + lnrsg[q], ln1c_gal[q]);
            }
            gm02 += um*(w1[q]*ug + w0[q]);
          }

          // a 1-halo sum is a sum of positive terms; its log is splined
          if (!(gm02 > 0)) {
            log_fatal("non-positive 1-halo sum at ln k = %g", lnk);
            exit(1);
          }
          ln_coarse[c] = log(gm02);
        }

        ln_k_spline_upsample(ln_coarse, n_coarse, k_step, K_PAD, dlnk,
                             Ntable.N_k_nlin, k_mult, curv, ln_dense);

        for (int j=0; j<Ntable.N_k_nlin; j++) {
          const double kj = exp(lim[nbin][0] + j*dlnk);
          const double gm02 = exp(ln_dense[j]);

          table[l][i][j] = log(Pdelta(kj, ai)*b_gal + gm02/n_gal);
        }
      }
    }

    // record the tags this table was built from
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_galaxy_bias;
    cache[3] = redshift.random_clustering;
    cache[4] = nuisance.random_photoz_clustering;
    cache[5] = redshift.random_shear;
  }

  // --- 3. BILINEAR TABLE READ ---

  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }

  // bin ni's ln P, read bilinearly and exponentiated; 0 outside the
  // bin's a range
  if (a < lim[ni][0] || a > lim[ni][1]) {
    return 0.0;
  }

  return exp(interpol2d(table[ni],
                        na, lim[ni][0], lim[ni][1], lim[ni][2], a,
                        Ntable.N_k_nlin, lim[nbin][0], lim[nbin][1],
                        lim[nbin][2], log(k)));
}


// ---------------------------------------------------------------------------
// P_gg(k, a, ni, nj), the halo-model galaxy power spectrum of lens bin ni
// (auto-spectra only: nj must equal ni), from a table of ln P per bin as
// in p_gm (its header: na x Ntable.N_k_nlin nodes uniform in a over the
// bin's range and in ln k, read bilinearly and exponentiated):
//
//   P_gg = P_delta b_gal^2 + G02/n_gal^2
//   G02  = int dlnM dn/dlnM [N_s^2 u_g(k|M)^2 + 2 f_c N_c N_s u_g(k|M)]
//
// 2-halo: the nonlinear matter spectrum (Pdelta) times the mean galaxy
// bias (bgal) squared. 1-halo: the satellite-satellite and
// central-satellite pairs of one halo per galaxy pair (ngal^2);
// satellites follow u_g, the NFW profile at c_g = gc c (file glossary), the
// central sits at the center (window 1). dn/dlnM as in the section
// banner; N_c(M), N_s(M), f_c the occupation of the GALAXY PROFILES
// banner.
//
// 1. Quadrature: the Gauss-Legendre rule of the section banner
// (item 1) over [ln limits.halo_m[RANGE_MIN], ln limits.halo_m[RANGE_MAX]],
// the same for every bin: nodes and weights are mapped once, in the
// rebuild block.
//
// 2. Loop levels as in the section banner (item 2): the innermost loop is
// the NFW kernel nfw_um alone. The occupation does not depend on a
// (HOD_nc, HOD_ns only range-check it: amin_lens is a placeholder):
//
//   per refill, per node -> mass_tab:
//     M, w, r_Delta, w (rho_cb/M)
//   per refill, per (bin, node) -> occ_tab:
//     N_s, f_c N_c
//   per bin, per a row, threaded:
//     a; Tinker f parameters; ngal, bgal
//   per (a, node) -> a_tab:
//     c_g = gc conc(M, a), ln(1+c_g), m(c_g), r_s,g = r_Delta/c_g,
//     ln r_s,g; W2 = dn_halo (N_s/m(c_g))^2,
//     W1 = 2 dn_halo (N_s/m(c_g)) f_c N_c
//   per (a, k), sum over nodes:
//     ug = u_g m(c_g) (nfw_um); G02 = sum ug (W2 ug + W1); ln P
//
// W2 and W1 carry the 1/m(c_g) of each u_g (the central has no profile).
// like.halo_model[3] must be HALO_PROFILE_NFW (the rows read nfw_um
// directly) and nuisance.gc[l] > 0 in every bin (u_g's condition,
// checked in the refill); else abort.
//
// Thread safety: as p_gm (its header).
//
// Cache invalidation:
//   as p_gm (its header); the rebuild block here holds table, lim,
//     mass_tab (with the mapped GL nodes), occ_tab, a_tab, k_tab and
//     k_mult
//
// Parameters:
//   k      - wavenumber in (c/H0)^-1
//   a      - scale factor
//   ni, nj - lens bins, 0 <= ni < redshift.clustering_nbin and nj = ni
//            (aborts otherwise)
//
// Returns:
//   P_gg(k, a, ni) in (c/H0)^3; 0 outside [amin_lens(ni), amax_lens(ni)];
//   ln P continued with unit slope outside [ln k_min, ln k_max] (interpol2d)
// ---------------------------------------------------------------------------
double p_gg(
    const double k,
    const double a,
    const int ni,
    const int nj
  )
{
  static uint64_t  cache[MAX_SIZE_ARRAYS];
  static double*** table    = NULL;
  static double**  lim      = NULL; // [nbin+1][3]: row l < nbin the a
                                    // grid of bin l (min, max, step);
                                    // row nbin the ln k grid
                                    // (min, max, step)
  static int       nbin     = 0;    // lens bins of the allocation
  static int       na       = 0;    // a nodes per bin
  static int       nnode    = 0;    // Gauss-Legendre mass nodes in ln M
  static double**  mass_tab = NULL; // [4][nnode] per mass node: M,
                                    // weight, r_Delta,
                                    // weight x (rho_cb/M)
  static double*** occ_tab  = NULL; // [nbin][2][nnode] per (bin, mass
                                    // node): N_s, f_c N_c
  static double*** a_tab    = NULL; // [n_threads][6][nnode]: one
                                    // (bin, a-row) iteration's scratch:
                                    // c_g, ln(1+c_g), r_s,g, ln r_s,g,
                                    // W2, W1
  static double*** k_tab    = NULL; // [n_threads][3][n_dense + pads]:
                                    // the coarse ln k scratch (coarse
                                    // ln G02, curvatures,
                                    // dense ln G02)
  static int       k_step   = 0;    // dense ln k nodes per coarse one
  static double*   k_mult   = NULL; // [n_coarse] Thomas multipliers
  static int       n_coarse = 0;    // coarse ln k nodes, pads included
  const int        K_PAD    = Ntable.halo_spline_pad; // pads per end

  // --- 1. REBUILD: SIZES, ALLOCATIONS, MAPPED GL RULE, TABLE AXES ---

  // first call, or the Ntable or clustering-n(z) tag differs from the
  // allocation's; every allocation is one block from malloc1d/malloc2d/
  // malloc3d, so one free each (header, Cache invalidation)
  if (NULL == table ||
      fdiff2(cache[1], Ntable.random) ||
      fdiff2(cache[3], redshift.random_clustering))
  {
    if (table != NULL) {
      free(table);
      free(lim);
      free(mass_tab);
      free(occ_tab);
      free(a_tab);
      free(k_tab);
      free(k_mult);
    }

    nbin  = redshift.clustering_nbin;
    na    = Ntable.halo_na_lens; // a nodes per lens bin
    // mass-node ladder: the default already lands far inside the
    // code's chi2 error budget (measured ladder: the skill file's
    // halo.c numerics); high_def_integration steps toward the largest
    // GSL rule
    if (0 == abs(Ntable.high_def_integration)) {
      nnode = Ntable.halo_nm;
    }
    else if (1 == abs(Ntable.high_def_integration)) {
      nnode = 2*Ntable.halo_nm;
    }
    else if (2 == abs(Ntable.high_def_integration)) {
      nnode = 4*Ntable.halo_nm;
    }
    else {
      nnode = 1024;
    }

    // coarse ln k step of the 1-halo sums (ln_k_spline_upsample): its
    // ladder, like the mass nodes', lands inside the chi2 error budget
    // at the default and becomes exact with high_def_integration
    if (0 == abs(Ntable.high_def_integration)) {
      k_step = Ntable.halo_nk_step;
    }
    else if (1 == abs(Ntable.high_def_integration)) {
      k_step = Ntable.halo_nk_step/2;
    }
    else {
      k_step = 1;
    }
    if (k_step < 1) {
      k_step = 1;
    }
    n_coarse = (Ntable.N_k_nlin - 1)/k_step + 2 + 2*K_PAD;

    table    = (double***) malloc3d(nbin, na, Ntable.N_k_nlin);
    lim      = (double**) malloc2d(nbin+1, 3);
    mass_tab = (double**) malloc2d(4, nnode);
    occ_tab  = (double***) malloc3d(nbin, 2, nnode);
    // one scratch block per thread (the thread count of this rebuild;
    // raising OMP_NUM_THREADS afterwards requires an Ntable bump)
    a_tab    = (double***) malloc3d(omp_get_max_threads(), 6, nnode);
    k_tab    = (double***) malloc3d(omp_get_max_threads(), 3,
                                    Ntable.N_k_nlin + n_coarse);
    k_mult   = (double*) malloc1d(n_coarse);
    ln_k_spline_multipliers(n_coarse, k_mult);

    // GL rule mapped once onto [ln M_min, ln M_max], shared by all bins
    const double lnMmin = log(limits.halo_m[RANGE_MIN]);
    const double lnMmax = log(limits.halo_m[RANGE_MAX]);

    // gsl_integration_glfixed_point(lo, hi, q, &x, &w, t): node q of the
    // rule t mapped onto [lo, hi], and its weight
    gsl_integration_glfixed_table* t = malloc_gslint_glfixed(nnode);
    for (int q=0; q<nnode; q++) {
      double lnM;
      gsl_integration_glfixed_point(lnMmin, lnMmax, q,
                                    &lnM, &mass_tab[1][q], t);
      mass_tab[0][q] = exp(lnM);
    }
    gsl_integration_glfixed_table_free(t);

    // ln k grid, shared by all bins
    lim[nbin][0] = log(limits.k_cH0[RANGE_MIN]);
    lim[nbin][1] = log(limits.k_cH0[RANGE_MAX]);
    lim[nbin][2] = (lim[nbin][1]-lim[nbin][0])
                   /((double) Ntable.N_k_nlin - 1.);
  }

  // --- 2. REFILL: THE HOD-WEIGHTED ln P TABLE ---

  // any of the six tags differs from the table's
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], Ntable.random)    ||
      fdiff2(cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(cache[3], redshift.random_clustering)  ||
      fdiff2(cache[4], nuisance.random_photoz_clustering) ||
      fdiff2(cache[5], redshift.random_shear))
  {
    // a grid of bin l over its lens range, node i at lim[l][0] + i
    // lim[l][2], both ends included. Set at every refill, not in the
    // rebuild block: the range moves with the lens photo-z shift and
    // stretch, and amax_lens widens to the source n(z)'s lower edge
    // when magnification bias is on (the redshift.random_shear tag)
    for (int l=0; l<nbin; l++) {
      lim[l][0] = amin_lens(l);
      lim[l][1] = amax_lens(l);
      lim[l][2] = (lim[l][1] - lim[l][0])/((double) na - 1.);
    }

    // --- 2a. GUARDS AND WARM-UP ---

    // the k rows read the NFW kernel directly (header, item 2)
    if (like.halo_model[3] != HALO_PROFILE_NFW) {
      log_fatal("like.halo_model[3] = %d not supported", like.halo_model[3]);
      exit(1);
    }

    // every lazy table the threaded loops read is built here, on one
    // thread (header, Thread safety)
    halo_warmup(lim[0][0], exp(lim[nbin][0]), 0, 1);

    /* PHYSICAL DERIVATION & LOGIC FLOW (P_gg and G02: header above)
       1. mass_tab, per mass node: M, weight, r_Delta,
          weighted dn/dlnM factor
       2. occ_tab, per (bin, mass node): occupation N_s and f_c N_c
       3. per a row, threaded: sigma_cb(M,a), its mass slope, the
          nu-independent f(nu) half, n_gal, b_gal
       4. thread scratch a_tab, per node of one a row: c_g, r_s,g,
          logs, weights W2, W1
       5. per k: G02 = sum_q ug (W2 ug + W1); table = ln P_gg */

    // --- 2b. PER MASS NODE, SERIAL ---

    // The factors independent of scale factor and lens bin.
    const double rho_m     = cosmology.rho_crit * cosmology.Omega_m;
    const double rho_delta = Delta * rho_m;
    // The cb density normalizes dn/dlnM (POWER SPECTRA banner).
    const double rho_hmf   = cosmology.rho_crit * omega_halo_field();

    // mass_tab rows 2-3: r_Delta from M = (4 pi/3) Delta rho_m r_Delta^3
    // and weight (rho_cb/M). The mass slope is evaluated at each a.
    for (int q=0; q<nnode; q++) {
      const double m = mass_tab[0][q];
      mass_tab[2][q] = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
      mass_tab[3][q] = mass_tab[1][q]*(rho_hmf/m);
    }

    // --- 2c. PER (BIN, MASS NODE): OCCUPATION, SERIAL ---

    // the occupation at the placeholder a = amin (HOD_nc, HOD_ns only
    // range-check it)
    for (int l=0; l<nbin; l++) {
      // u_g's condition, checked for every bin at once
      if (!(nuisance.gc[l] > 0)) {
        log_fatal("galaxy concentration factor gc[%d] = %g must be > 0",
                  l, nuisance.gc[l]);
        exit(1);
      }

      const double fc = HOD_fc(l);

      for (int q=0; q<nnode; q++) {
        const double m = mass_tab[0][q];
        occ_tab[l][0][q] = HOD_ns(m, lim[l][0], l);    // N_s
        occ_tab[l][1][q] = fc*HOD_nc(m, lim[l][0], l); // f_c N_c
      }
    }

    // --- 2d. (BIN, a ROW) PAIRS COLLAPSED AND THREADED ---
    #pragma omp parallel for collapse(2) schedule(static)
    for (int l=0; l<nbin; l++) {
      for (int i=0; i<na; i++) {
        const double gc = nuisance.gc[l];

        const double ai = lim[l][0] + i*lim[l][2];

        // The nu-independent Tinker f(nu) half, and the mean
        // galaxy density and bias of this a row
        const fnu_params fnu_pars = fnu_params_at(ai);
        const double     n_gal    = ngal(l, ai);
        const double     b_gal    = bgal(l, ai);

        // thread-private scratch: this (bin, a-row) iteration fills
        // it and consumes it in its own k loop; restrict: each row is
        // reached only through its pointer, no reload after libm calls
        double** const wsp   = a_tab[omp_get_thread_num()];
        double** const k_wsp = k_tab[omp_get_thread_num()];
        double* restrict conc_gal = wsp[0];
        double* restrict ln1c_gal = wsp[1];
        double* restrict r_sg     = wsp[2];
        double* restrict lnrsg    = wsp[3];
        double* restrict w2       = wsp[4];
        double* restrict w1       = wsp[5];

        // per (a, node): c_g, r_s,g, their logs, and the weights W2, W1
        // with 1/m(c_g) folded in
        for (int q=0; q<nnode; q++) {
          const double m  = mass_tab[0][q];
          const double nu = delta_c/sqrt(sigma2(m, ai));

          // c_g = gc c, with m(c_g) = ln(1+c_g) - c_g/(1+c_g)
          const double cg    = conc(m, ai)*gc;
          const double ln1cg = log1p(cg);
          const double mcg   = ln1cg - cg/(1.0 + cg);

          // dn_halo = quadrature weight x dn/dlnM (Tinker); n_sat = N_s
          const double dn_halo = mass_tab[3][q]*dlognudlogm(m, ai)*fnu_core(nu, &fnu_pars)*nu;
          const double n_sat   = occ_tab[l][0][q];

          conc_gal[q] = cg;
          ln1c_gal[q] = ln1cg;
          r_sg[q]     = mass_tab[2][q]/cg;
          lnrsg[q]    = log(r_sg[q]);
          w2[q]       = dn_halo*(n_sat/mcg)*(n_sat/mcg);
          w1[q]       = 2.0*dn_halo*(n_sat/mcg)*occ_tab[l][1][q];
        }

        // per k: the 1-halo sum on the coarse ln k grid only (the
        // expensive part: nnode NFW kernels per k), its log splined to
        // the dense grid, then ln P with the 2-halo term read exactly
        // at every dense node
        double* restrict ln_coarse = k_wsp[0];
        double* restrict curv      = k_wsp[1];
        double* restrict ln_dense  = k_wsp[2];

        const double dlnk      = lim[nbin][2];
        const double lnk_first = lim[nbin][0] - K_PAD*k_step*dlnk;

        for (int c=0; c<n_coarse; c++) {
          const double lnk = lnk_first + c*k_step*dlnk;
          const double kj  = exp(lnk);

          double g02 = 0.0;
          // Sum ug*(w2*ug + w1), four nodes q, q+1, q+2, q+3 per step
          // (one per lane; nfw_um4 = nfw_um on each lane,
          // bitwise) into four-lane partial sums, added in a fixed lane
          // order (simd_horizontal_sum), then a scalar tail (summation
          // order as in p_gm)

          // k and ln k of this column in all four lanes
          const v4d vk   = simde_mm256_set1_pd(kj);   // k
          const v4d vlnk = simde_mm256_set1_pd(lnk);  // ln k

          // the four-lane partial sums of g02, from zero
          v4d vg02 = simde_mm256_setzero_pd();

          int q = 0;
          for (; q<=nnode-4; q+=4) {
            // the four arguments of nfw_um at nodes q..q+3; scalar:
            //   nfw_um(conc_gal[q], kj*r_sg[q], lnk + lnrsg[q], ln1c_gal[q])

            // c_g, the galaxy concentrations of nodes q..q+3
            const v4d vconc_gal = simde_mm256_loadu_pd(conc_gal + q);

            // r_s,g of nodes q..q+3
            const v4d vrs_gal = simde_mm256_loadu_pd(r_sg + q);

            // k r_s,g
            const v4d vkrs_gal = simde_mm256_mul_pd(vk, vrs_gal);

            // ln r_s,g of nodes q..q+3
            const v4d vlnrs_gal = simde_mm256_loadu_pd(lnrsg + q);

            // ln(k r_s,g) = ln k + ln r_s,g
            const v4d vlnkrs_gal = simde_mm256_add_pd(vlnk, vlnrs_gal);

            // ln(1 + c_g) of nodes q..q+3
            const v4d vln1c_gal = simde_mm256_loadu_pd(ln1c_gal + q);

            // ug = u_g m(c_g) at nodes q..q+3
            const v4d vug = nfw_um4(vconc_gal, vkrs_gal, vlnkrs_gal,
                                    vln1c_gal);

            // the weights of nodes q..q+3
            const v4d vw2 = simde_mm256_loadu_pd(w2 + q);  // W2
            const v4d vw1 = simde_mm256_loadu_pd(w1 + q);  // W1

            // scalar: g02 += ug*(w2[q]*ug + w1[q])

            // W2 ug + W1, fused: satellite pairs and central-satellite
            // pairs
            const v4d vpair_weight = nfw_fmadd4(vw2, vug, vw1);

            // ug (W2 ug + W1) + g02, lane by lane, fused
            vg02 = nfw_fmadd4(vug, vpair_weight, vg02);
          }

          // lane 0 + lane 1 + lane 2 + lane 3 of the partial sums
          g02 = simd_horizontal_sum(vg02);

          // scalar tail: nnode not a multiple of four
          for (; q<nnode; q++) {
            const double ug = nfw_um(conc_gal[q], kj*r_sg[q],
                                     lnk + lnrsg[q], ln1c_gal[q]);
            g02 += ug*(w2[q]*ug + w1[q]);
          }

          // a 1-halo sum is a sum of positive terms; its log is splined
          if (!(g02 > 0)) {
            log_fatal("non-positive 1-halo sum at ln k = %g", lnk);
            exit(1);
          }
          ln_coarse[c] = log(g02);
        }

        ln_k_spline_upsample(ln_coarse, n_coarse, k_step, K_PAD, dlnk,
                             Ntable.N_k_nlin, k_mult, curv, ln_dense);

        for (int j=0; j<Ntable.N_k_nlin; j++) {
          const double kj = exp(lim[nbin][0] + j*dlnk);
          const double g02 = exp(ln_dense[j]);

          table[l][i][j] = log(Pdelta(kj, ai)*b_gal*b_gal
                               + g02/(n_gal*n_gal));
        }
      }
    }

    // record the tags this table was built from
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_galaxy_bias;
    cache[3] = redshift.random_clustering;
    cache[4] = nuisance.random_photoz_clustering;
    cache[5] = redshift.random_shear;
  }

  // --- 3. BILINEAR TABLE READ ---

  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }

  // auto-spectra only (header)
  if (ni != nj) {
    log_fatal("cross-tomography (ni,nj) = (%d,%d) bins not supported", ni, nj);
    exit(1);
  }

  // bin ni's ln P, read bilinearly and exponentiated; 0 outside the
  // bin's a range
  if (a < lim[ni][0] || a > lim[ni][1]) {
    return 0.0;
  }

  return exp(interpol2d(table[ni],
                        na, lim[ni][0], lim[ni][1], lim[ni][2], a,
                        Ntable.N_k_nlin, lim[nbin][0], lim[nbin][1],
                        lim[nbin][2], log(k)));
}



// ============================================================================
// [SECTION] HALO-MODEL INTRINSIC ALIGNMENT
// ============================================================================
//
// Halo-model intrinsic alignments of Fortuna et al. 2021 (F21,
// 2003.02700): red central galaxies align with the large-scale tidal
// field (the NLA 2-halo term, built in cosmo2D.c from Pdelta and the
// red-central fraction read here), and satellite galaxies point
// radially at the center of their host halo (the 1-halo term, tabulated
// here).
//
// One IA population, the shape (source) sample, with an a-free HOD and
// red fractions per halo mass (nuisance.ia_hod, nuisance.ia_red; their
// layouts: the guards of ia_tables):
//
//   N_tot(M) = f_c N_c(M) + N_s(M)          all galaxies of the sample
//   N_rc(M)  = f_c N_c(M) f_red,cen(M)      red centrals
//   N_rs(M)  = N_s(M) f_red,sat(M)          red satellites
//   n_g(a)   = int dlnM dn/dlnM N_tot       number density
//   f_rc(a)  = int dlnM dn/dlnM N_rc / n_g  red-central fraction
//
// The satellite alignment (F21 Eq. 20 with the 0.3 ceiling of F21
// sec. 4.2): the shear of a satellite at distance r from the center is
//
//   gamma_bar(r) = sign(a_1h) min{ |a_1h| [max(r, r_floor)/r_vir]^-2,
//                                  gamma_max }
//
// with r_vir = r_Delta (Delta = 200 times the mean density: F21's
// halo definition and this file's), r_floor = 0.06 Mpc/h and
// gamma_max = 0.3. Both limits are one radius r_e below which
// gamma_bar is flat:
//
//   r_e = max( r_floor, r_vir (|a_1h|/gamma_max)^(1/2) )
//   gamma_bar(r) = a_1h g(r),   g(r) = [max(r, r_e)/r_vir]^-2
//
// The alignment amplitude a_1h(a) = a_1h ((1+z)/(1+z_pivot))^eta_1h
// (nuisance.ia_halo[0..2]) is global; its sign multiplies the dI
// spectrum, its square the II spectrum, and |a_1h(a)| moves r_e.
//
// The Fourier transform of the density-weighted alignment field, per
// unit a_1h (F21 sec. 4.1 and App. C, Eqs. C1-C9; Schneider & Bridle
// 2010, 0903.3870, App. B), at theta_k = pi/2 (F21's choice):
//
//   gamma_hat(k|M) = sum_{l = 2, 4, 6} P_l K_l(k r_s) / m(c)
//   K_l(t)         = int_0^c g(x) x (1 + x)^-2 j_l(t x) dx,   x = r/r_s
//   P_l            = i^l (2l + 1) A_l / (4 pi)
//   m(c)           = ln(1 + c) - c/(1 + c)
//
// j_l the spherical Bessel function and A_l the angular integral of
// sin^-2(theta) e^(2 i phi) against P_l: F21's alignment depends on the
// projected radius, (r sin theta/r_vir)^-2 = (r/r_vir)^-2 sin^-2 theta,
// and e^(2 i phi) is the spin-2 phase of the shear. One angular set
// serves every radius: A_2 = 3 pi/2, A_4 = -5 pi/6, A_6 = 21 pi/32, so
// P_2 = P_4 = -15/8 and P_6 = -273/128.
// Ntable.halo_ia_lmax (2, 4 or 6) truncates the multipole sum; F21 use
// l <= 6. gamma_hat -> 0 as k^2 at k -> 0 (j_l ~ (tx)^l).
//
// The 1-halo spectra (F21 Eqs. 17-18 with f_s <N_s>/n_s = N_rs/n_g, a
// Poisson satellite count, and F21's |gamma_hat| in the dI term):
//
//   P_dI^1h = a_1h(a) f_1h(k) S_dI(k, a)
//   P_II^1h = a_1h(a)^2 f_1h(k) S_II(k, a)
//   S_dI    = int dlnM dn/dlnM (M/rho_m) u(k|M) (N_rs/n_g) |gamma_hat|
//   S_II    = int dlnM dn/dlnM (N_rs/n_g)^2 gamma_hat^2
//
// |gamma_hat| is F21's convention (Eq. 17); the signed halo-model
// product gamma_hat u differs from it only where gamma_hat rings,
// k r_s >~ 3. u(k|M) is the NFW transform (nfw_um); the F21 windows
// (App. B, Eqs. B1-B2) that split the scales between the two terms:
//
//   f_1h(k) = 1 - exp[-(k/k_1h)^2],   f_2h(k) = exp[-(k/k_2h)^2]
//   k_1h = 4 h/Mpc,   k_2h = 6 h/Mpc
//
// Sign convention: the readers return P_dI^1h with the sign of a_1h;
// the C_l cores of cosmo2D.c subtract it, as they subtract the NLA term
// (P_dI^phys = -[f_rc C_1 P_delta f_2h + P_dI^1h], C_1(a) the NLA
// amplitude IA_A1_Z1 of cosmo2D.c, > 0 for A_IA > 0). Radial alignment,
// a_1h > 0, is a negative dI correlation, the same sense as A_IA > 0.
//
// Scope: the matter-intrinsic and intrinsic-intrinsic 1-halo terms. The
// lens-galaxy x satellite 1-halo term S_gI of the HOD galaxy-shear core
// is not tabulated here (that combination aborts in cosmo2D.c).
//
// Glossary of this section:
//
//   ia_a1h            = a_1h(a), the satellite alignment amplitude
//   ia_window_1h,
//   ia_window_2h      = f_1h(k), f_2h(k)
//   ia_fg_read        = f(t), g(t) of the nfw_ table, cubic Hermite read
//   ia_moments        = M_p(x) = int_0^x y^p (1 + y)^-2 dy
//   ia_edge_setup     = the per-halo constants of one integration edge
//   ia_gamma_hat_m    = m(c) gamma_hat(k|M), the satellite kernel
//   ia_tables         = the owner of the S_dI, S_II and f_rc tables
//   ia_f_red_central  = f_rc(a)
//   ia_p1h_dI,
//   ia_p1h_II         = P_dI^1h, P_II^1h
// ---------------------------------------------------------------------------

// r_floor, gamma_max: the F21 alignment profile (F21 sec. 4.2)
static const double IA_R_FLOOR_MPCH = 0.06; // Mpc/h, comoving
static const double IA_GAMMA_MAX    = 0.3;

// k_1h, k_2h of the F21 windows (F21 App. B, Eqs. B1-B2), h/Mpc
static const double IA_K1H_HMPC = 4.0;
static const double IA_K2H_HMPC = 6.0;

// P_l = i^l (2l + 1) A_l/(4 pi) for l = 2, 4, 6 (section banner)
#define IA_NL 3
static const double IA_MULTIPOLE_WEIGHT[IA_NL] = {-1.875, -1.875, -2.1328125};


// ---------------------------------------------------------------------------
// The satellite kernel K_l(t) in closed form. With x_e = r_e/r_s,
// g = (c/x_e)^2 is flat inside x_e and g = c^2 x^-2 outside, so K_l is
// two radial integrals of one family,
//
//   K_l^beta(x0, x1) = int_{x0}^{x1} y^beta (1 + y)^-2 j_l(t y) dy
//
//   K_l(t) = (c/x_e)^2 K_l^+1(0, x_in) + c^2 K_l^-1(x_in, c)
//   x_in   = min(x_e, c)       (the second piece is absent if x_e >= c)
//
// Each integral is E(x1) - E(x0), E(x) = int_0^x, evaluated at each
// edge x by one of two exact formulas:
//
// 1. Closed form (large t x). j_l is a finite sum of sin u/u^n and
//    cos u/u^n (u = t y); partial fractions in y and integration by
//    parts leave the sine and cosine integrals at u = t x and at
//    z = t (1 + x), which the auxiliary functions f, g of A&S 5.2.6-7
//    (the nfw_ table functions) turn into
//
//      G(x) = R_s sin u + R_c cos u - S_u [f(u) cos u + g(u) sin u]
//             - S_z [f(z) cos u + g(z) sin u] + C_z [f(z) sin u - g(z) cos u]
//      E(x) = G(x) - G(0)
//
//    R_s, R_c are polynomials in 1/t whose coefficients are rational
//    functions of the edge x (per-halo constants: ia_edge_setup);
//    S_u, S_z, C_z and G(0) (a combination of f(t), g(t) and pi) are
//    polynomials in 1/t alone. The tables below hold their coefficients,
//    split by parity in 1/t. G is the antiderivative with
//    G(infinity) = 0, so E(x) = G(x) - G(0) and
//    G(0) = -int_0^infinity y^beta (1 + y)^-2 j_l(t y) dy.
//
// 2. Taylor series (small t x). The closed form cancels there (its
//    terms grow like (t x)^-(l+1) while the result is ~ (t x)^l: a
//    cancellation of (t x)^-(2l+1)), while the power series of j_l,
//    j_l(u) = sum_k a_k u^(l+2k), a_k = -a_(k-1)/(2k (2l + 2k + 1)),
//    a_0 = 1/(2l + 1)!!, integrates term by term:
//
//      E(x) = t^l sum_{k < K} a_k M_(beta+l+2k)(x) t^(2k),
//      M_p(x) = int_0^x y^p (1 + y)^-2 dy          (ia_moments)
//
//    a_k M_p(x) is a per-halo constant, so per k the series is one
//    Horner polynomial in t^2.
//
// Switch rule, per edge x and multipole l: the series below
// u = t x < s_l(x), the closed form at and above it, with
//
//   s_l(x) = min( max(U_MIN_l, T_MIN_l x), U_MAX ),   K = ceil(K0 + K1 s_l)
//
// The closed form is accurate once both u and t are large enough
// (U_MIN, T_MIN); the series loses digits once its alternating terms
// (~ e^u) grow past the result (U_MAX), and needs K0 + K1 u terms.
// ---------------------------------------------------------------------------
static const double IA_SWITCH_U_MIN[IA_NL] = {3.0, 4.4, 5.0};
static const double IA_SWITCH_T_MIN[IA_NL] = {0.6, 2.0, 3.2};
static const double IA_SWITCH_U_MAX        = 24.0;
static const double IA_SERIES_K0           = 12.0;
static const double IA_SERIES_K1           = 1.2;

// sizes: terms of the series at s_l = U_MAX, ceil(K0 + K1 U_MAX); the
// highest moment it reads, M_p with p = 1 + 6 + 2 (IA_KMAX - 1); the
// coefficients in 1/t of the l = 6 polynomials (up to t^-7), also the
// number of x-coefficients (x^0 .. x^7) of the r_s,j, r_c,j numerators;
// and the coefficients of one parity
#define IA_KMAX 41
#define IA_PMAX 87
#define IA_NCOEF 8
#define IA_NHALF 4

// Coefficient tables, index [l/2 - 1][0: beta = +1, 1: beta = -1]:
//
//   S_u, S_z (odd in 1/t)  = (1/t) sum_j IA_S_U[j] t^-2j (same for S_Z)
//   C_z (even in 1/t)      = sum_j IA_C_Z[j] t^-2j
//   G(0) = f(t) (1/t) sum_j IA_G0_F[j] t^-2j + g(t) sum_j IA_G0_G[j] t^-2j
//          + sum_n IA_G0_C[n] t^-n
//   R_s = (1/t) sum_j r_s,j t^-2j,   R_c = sum_j r_c,j t^-2j, with
//   r_s,j = [sum_i IA_R_S_NUM[j][i] x^i] / [x^IA_R_S_XPOW[j] (1 + x)]
//   (same for r_c,j with IA_R_C_NUM, IA_R_C_XPOW)
static const double IA_S_U[IA_NL][2][IA_NHALF] = {
  { // l = 2
    {0.0, -6.0, 0.0, 0.0},  // beta = +1
    {-1.0, -12.0, 0.0, 0.0}  // beta = -1
  },
  { // l = 4
    {0.0, -15.0, -420.0, 0.0},  // beta = +1
    {-0.75, -30.0, -630.0, 0.0}  // beta = -1
  },
  { // l = 6
    {0.0, -26.25, -1890.0, -62370.0},  // beta = +1
    {-0.625, -52.5, -2835.0, -83160.0}  // beta = -1
  }
};

static const double IA_S_Z[IA_NL][2][IA_NHALF] = {
  { // l = 2
    {-3.0, 6.0, 0.0, 0.0},  // beta = +1
    {-5.0, 12.0, 0.0, 0.0}  // beta = -1
  },
  { // l = 4
    {10.0, -195.0, 420.0, 0.0},  // beta = +1
    {12.0, -285.0, 630.0, 0.0}  // beta = -1
  },
  { // l = 6
    {-21.0, 1680.0, -29295.0, 62370.0},  // beta = +1
    {-23.0, 2100.0, -38745.0, 83160.0}  // beta = -1
  }
};

static const double IA_C_Z[IA_NL][2][IA_NHALF] = {
  { // l = 2
    {-1.0, 6.0, 0.0, 0.0},  // beta = +1
    {-1.0, 12.0, 0.0, 0.0}  // beta = -1
  },
  { // l = 4
    {1.0, -55.0, 420.0, 0.0},  // beta = +1
    {1.0, -75.0, 630.0, 0.0}  // beta = -1
  },
  { // l = 6
    {-1.0, 231.0, -8505.0, 62370.0},  // beta = +1
    {-1.0, 273.0, -11025.0, 83160.0}  // beta = -1
  }
};

static const double IA_G0_F[IA_NL][2][IA_NHALF] = {
  { // l = 2
    {3.0, -6.0, 0.0, 0.0},  // beta = +1
    {5.0, -12.0, 0.0, 0.0}  // beta = -1
  },
  { // l = 4
    {-10.0, 195.0, -420.0, 0.0},  // beta = +1
    {-12.0, 285.0, -630.0, 0.0}  // beta = -1
  },
  { // l = 6
    {21.0, -1680.0, 29295.0, -62370.0},  // beta = +1
    {23.0, -2100.0, 38745.0, -83160.0}  // beta = -1
  }
};

static const double IA_G0_G[IA_NL][2][IA_NHALF] = {
  { // l = 2
    {1.0, -6.0, 0.0, 0.0},  // beta = +1
    {1.0, -12.0, 0.0, 0.0}  // beta = -1
  },
  { // l = 4
    {-1.0, 55.0, -420.0, 0.0},  // beta = +1
    {-1.0, 75.0, -630.0, 0.0}  // beta = -1
  },
  { // l = 6
    {1.0, -231.0, 8505.0, -62370.0},  // beta = +1
    {1.0, -273.0, 11025.0, -83160.0}  // beta = -1
  }
};

static const double IA_G0_C[IA_NL][2][IA_NCOEF] = {
  { // l = 2
    {0.0, 0.0, -6.0, 9.4247779607693793, 0.0, 0.0, 0.0, 0.0},  // beta = +1
    {-0.33333333333333331, 1.5707963267948966, -12.0, 18.849555921538759,
     0.0, 0.0, 0.0, 0.0}  // beta = -1
  },
  { // l = 4
    {0.0, 0.0, 8.3333333333333339, 23.56194490192345, -420.0,
     659.73445725385659, 0.0, 0.0},  // beta = +1
    {-0.13333333333333333, 1.1780972450961724, 5.0, 47.1238898038469,
     -630.0, 989.60168588078488, 0.0, 0.0}  // beta = -1
  },
  { // l = 6
    {0.0, 0.0, -25.199999999999999, 41.233403578366037, 1575.0,
     2968.8050576423548, -62370.0, 97970.566902197708},  // beta = +1
    {-0.076190476190476197, 0.98174770424681035, -33.600000000000001,
     82.466807156732074, 1785.0, 4453.2075864635317, -83160.0,
     130627.42253626361}  // beta = -1
  }
};

static const double IA_R_S_NUM[IA_NL][2][IA_NHALF][IA_NCOEF] = {
  { // l = 2
    { // beta = +1
      {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {-3.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    },
    { // beta = -1
      {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {-1.0, 2.0, -6.0, -12.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    }
  },
  { // l = 4
    { // beta = +1
      {-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {10.0, 55.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {-35.0, 70.0, -210.0, -420.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    },
    { // beta = -1
      {-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {8.0, -10.75, 11.25, 75.0, 0.0, 0.0, 0.0, 0.0},
      {-21.0, 31.5, -52.5, 105.0, -315.0, -630.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    }
  },
  { // l = 6
    { // beta = +1
      {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {-21.0, -231.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {882.0, -1244.25, 1653.75, 8505.0, 0.0, 0.0, 0.0, 0.0},
      {-2079.0, 3118.5, -5197.5, 10395.0, -31185.0, -62370.0, 0.0, 0.0}
    },
    { // beta = -1
      {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {-19.0, 20.375, -23.625, -273.0, 0.0, 0.0, 0.0, 0.0},
      {648.0, -848.25, 1149.75, -1606.5, 2047.5, 11025.0, 0.0, 0.0},
      {-1485.0, 1980.0, -2772.0, 4158.0, -6930.0, 13860.0, -41580.0,
       -83160.0}
    }
  }
};

static const int IA_R_S_XPOW[IA_NL][2][IA_NHALF] = {
  {{0, 1, 0, 0}, {0, 3, 0, 0}},  // l = 2
  {{0, 1, 3, 0}, {0, 3, 5, 0}},  // l = 4
  {{0, 1, 3, 5}, {0, 3, 5, 7}}  // l = 6
};

static const double IA_R_C_NUM[IA_NL][2][IA_NHALF][IA_NCOEF] = {
  { // l = 2
    { // beta = +1
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {-3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    },
    { // beta = -1
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {1.0, -2.0, -6.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    }
  },
  { // l = 4
    { // beta = +1
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {35.0, -70.0, -210.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    },
    { // beta = -1
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {-1.0, 0.25, 11.25, 0.0, 0.0, 0.0, 0.0, 0.0},
      {21.0, -31.5, 52.5, -105.0, -315.0, 0.0, 0.0, 0.0},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
    }
  },
  { // l = 6
    { // beta = +1
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {-21.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {-189.0, 204.75, 1653.75, 0.0, 0.0, 0.0, 0.0, 0.0},
      {2079.0, -3118.5, 5197.5, -10395.0, -31185.0, 0.0, 0.0, 0.0}
    },
    { // beta = -1
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {1.0, -1.625, -23.625, 0.0, 0.0, 0.0, 0.0, 0.0},
      {-153.0, 188.25, -225.75, 220.5, 2047.5, 0.0, 0.0, 0.0},
      {1485.0, -1980.0, 2772.0, -4158.0, 6930.0, -13860.0, -41580.0, 0.0}
    }
  }
};

static const int IA_R_C_XPOW[IA_NL][2][IA_NHALF] = {
  {{0, 0, 0, 0}, {0, 2, 0, 0}},  // l = 2
  {{0, 0, 2, 0}, {0, 2, 4, 0}},  // l = 4
  {{0, 0, 2, 4}, {0, 2, 4, 6}}  // l = 6
};



// ---------------------------------------------------------------------------
// Per-halo constants of the satellite kernel. One ia_edge per edge x of
// the radial integrals (x_in, and c when the power-law piece exists);
// every array is indexed [l/2 - 1][beta: 0 = +1, 1 = -1].
// ---------------------------------------------------------------------------
typedef struct {
  double x;                         // edge radius in units of r_s
  double lnx;                       // ln x
  double ln1x;                      // ln(1 + x)
  double switch_u[IA_NL];           // s_l(x): series below u = t x < s_l
  int    n_series[IA_NL];           // K: series terms at this edge
  double series[IA_NL][2][IA_KMAX]; // a_k M_(beta+l+2k)(x)
  double r_sin[IA_NL][2][IA_NHALF]; // r_s,j(x): R_s = (1/t) sum r_s,j t^-2j
  double r_cos[IA_NL][2][IA_NHALF]; // r_c,j(x): R_c = sum r_c,j t^-2j
} ia_edge;


typedef struct {
  double conc;          // c(M, a)
  double ln1c;          // ln(1 + c)
  double r_s;           // r_s = r_Delta/c, c/H0
  double lnrs;          // ln r_s
  double w_inner;       // (c/x_e)^2: g inside x_e
  double conc2;         // c^2: g = c^2 x^-2 outside x_e
  double w_dI;          // dn (M/rho_m) (N_rs/n_g)/m(c)^2
  double w_II;          // dn (N_rs/n_g)^2/m(c)^2
  int    has_power;     // 1: x_e < c, the power-law piece exists
  ia_edge edge[2];      // [0] x_in = min(x_e, c), [1] c
} ia_halo;


// ---------------------------------------------------------------------------
// Tables of the halo-model IA, built and refilled by ia_tables (its
// header maps each array to the integrals); zero at program start, so
// the first call builds everything.
// ---------------------------------------------------------------------------
static struct {
  uint64_t cache[MAX_SIZE_ARRAYS]; // [0] cosmology, [1] Ntable,
                                   // [2] nuisance.random_ia_halo,
                                   // [3] shear n(z), [4] source photo-z
  int n_a;               // a nodes
  int n_nodes;           // Gauss-Legendre mass nodes in ln M
  int n_active;          // mass nodes with red satellites, N_rs > 0
  int n_l;               // multipoles in the kernel, halo_ia_lmax/2
  int k_step;            // dense ln k nodes per coarse one
  int n_coarse;          // coarse ln k nodes, pads included
  int block_rows;        // a rows per block of the halo scratch
  double lim[2][3];      // [0] a grid, [1] ln k grid: min, max, step
  double*** tab;         // [2][n_a][N_k_nlin] ln S_dI (0), ln S_II (1)
  double* f_red_cen;     // [n_a] f_rc(a_i)
  double** mass_node;    // [8][n_nodes] per mass node q: M, GL weight,
                         //   GL weight x (rho_cb/M),
                         //   r_Delta, M/rho_m, N_tot, N_rc, N_rs
  int* active;           // [n_nodes] mass-node index of each active node
  double** row;          // [2][n_a] per a row: a_1h(a), n_g(a)
  fnu_params* fnu_pars;  // [n_a] Tinker f(nu) parameters per a row
  double** dn;           // [n_a][n_nodes] GL weight x dn/dlnM
  ia_halo* halo;         // [block_rows x n_nodes] kernel constants
  double*** ln_coarse;   // [2][n_a][n_coarse] ln S_dI, ln S_II, coarse
  double** curv;         // [n_a][n_coarse] spline curvatures (scratch)
  double* k_mult;        // [n_coarse] Thomas multipliers of the spline
  double** fg_slope;     // [2][nfw_.n_nodes] Hermite slopes of f, G
} ia_ = {0};


// ---------------------------------------------------------------------------
// a_1h(a), the satellite alignment amplitude at scale factor a:
//
//   a_1h(a) = a_1h [(1 + z)/(1 + z_pivot)]^eta_1h,   1 + z = 1/a
//
// nuisance.ia_halo[0] = a_1h, [1] = eta_1h, [2] = z_pivot.
//
// Parameters:
//   a - scale factor
//
// Returns:
//   a_1h(a), dimensionless, with the sign of a_1h
// ---------------------------------------------------------------------------
static inline double ia_a1h(
    const double a  // scale factor
  )
{
  const double a1h     = nuisance.ia_halo[0];
  const double eta_1h  = nuisance.ia_halo[1];
  const double z_pivot = nuisance.ia_halo[2];

  if (0.0 == eta_1h) {
    return a1h;
  }

  // (1 + z)/(1 + z_pivot), with 1 + z = 1/a
  const double redshift_ratio = 1.0/(a*(1.0 + z_pivot));
  return a1h*pow(redshift_ratio, eta_1h);
}


// ---------------------------------------------------------------------------
// The F21 windows (F21 App. B, Eqs. B1-B2), k in (c/H0)^-1: the 1-halo term
// switched on above k_1h and the 2-halo term switched off above k_2h,
//
//   f_1h(k) = 1 - exp[-(k/k_1h)^2],   f_2h(k) = exp[-(k/k_2h)^2],
//
// k_1h, k_2h in h/Mpc times cosmology.coverH0 (k in (c/H0)^-1 is k in
// h/Mpc times c/H0 in Mpc/h). f_1h is -expm1: no cancellation at
// k << k_1h, where f_1h ~ (k/k_1h)^2.
//
// Returns:
//   f_1h(k), f_2h(k), dimensionless, in [0, 1]
// ---------------------------------------------------------------------------
static inline double ia_window_1h(
    const double k  // wavenumber in (c/H0)^-1
  )
{
  const double k_1h  = IA_K1H_HMPC*cosmology.coverH0;
  const double ratio = k/k_1h;
  return -expm1(-ratio*ratio);
}


// f_2h(k) of the header above, exported for the NLA 2-halo leg of
// cosmo2D.c
double ia_window_2h(
    const double k  // wavenumber in (c/H0)^-1
  )
{
  const double k_2h  = IA_K2H_HMPC*cosmology.coverH0;
  const double ratio = k/k_2h;
  return exp(-ratio*ratio);
}


// ---------------------------------------------------------------------------
// f(t) and g(t), the auxiliary functions of the sine and cosine
// integrals (the nfw_ header), read from the nfw_ table by cubic
// Hermite interpolation in ln t. The IA kernel only; nfw_um keeps its
// linear read.
//
// What cubic Hermite interpolation is: between two neighbouring nodes
// it uses the cubic that matches the tabulated value and the tabulated
// slope at both nodes. With s in [0, 1] the fraction of the interval,
//
//   y(s) = h00(s) y_i + h10(s) h y'_i + h01(s) y_(i+1) + h11(s) h y'_(i+1)
//   h00 = 2s^3 - 3s^2 + 1,   h10 = s^3 - 2s^2 + s,
//   h01 = -2s^3 + 3s^2,      h11 = s^3 - s^2
//
// (h the node spacing, ' = d/dln t). Its error falls as h^4 (about
// h^4 y''''/384), where the linear read's falls as h^2 (h^2 y''/8), on
// the same table: at the table's spacing it is several orders of
// magnitude more accurate.
//
// Why the slopes are exact and free: f and g obey f' = -g and
// g' = f - 1/t (A&S 5.2.6-7), so in the table variables
//
//   df/dln t = -t g(t),   dG/dln t = t f(t)     (G = g + ln t)
//
// follow from the stored values themselves: fg_slope (built with the
// table, ia_tables' rebuild block) holds h df/dln t and h dG/dln t at
// every node, no derivative is approximated.
//
// Why the IA kernel needs it: the closed form of the kernel (the
// section's kernel header) subtracts terms that can be about a million
// times larger than their sum (near the series switch, large c, l = 6).
// The linear read's relative error of f, g is amplified by that factor
// into percent-level errors of gamma_hat; the Hermite read keeps the
// kernel near its floating-point limit. nfw_um has no such cancellation
// and keeps the cheaper linear read.
//
// Above NFW_TASY: the asymptotic series (A&S 5.2.34-35), nested,
//
//   f(t) ~ (1 - 2!/t^2 + 4!/t^4 - ... - 14!/t^14)/t
//   g(t) ~ (1 - 3!/t^2 + 5!/t^4 - ... - 15!/t^14)/t^2
//
// three terms longer than nfw_um's, which this accuracy needs (2, 12,
// ..., 182 and 6, 20, ..., 210 are the ratios of consecutive
// coefficients). Below NFW_TMIN the read returns the first node
// (nfw_pos).
//
// Cache invalidation:
//   nfw_.tab: nfw_table (Ntable.random); fg_slope: the rebuild block of
//   ia_tables (Ntable.random)
//
// Parameters:
//   t, lnt - the argument and its log (the caller has ln t)
//   f, g   - output: f(t), g(t)
// ---------------------------------------------------------------------------
static inline void ia_fg_read(
    const double t,    // argument t > 0
    const double lnt,  // ln t
    double* f,         // output: f(t)
    double* g          // output: g(t)
  )
{
  if (t <= NFW_TASY) {
    const double* restrict tab_f   = nfw_.tab[0];      // f(t_i)
    const double* restrict tab_G   = nfw_.tab[1];      // G(t_i)
    const double* restrict slope_f = ia_.fg_slope[0];  // h df/dln t
    const double* restrict slope_G = ia_.fg_slope[1];  // h dG/dln t

    double s;
    const int i = nfw_pos(lnt, &s);  // node i, fraction s of [i, i + 1]

    // the four Hermite basis polynomials at s
    const double s2  = s*s;
    const double s3  = s2*s;
    const double h00 = 2.0*s3 - 3.0*s2 + 1.0;
    const double h10 = s3 - 2.0*s2 + s;
    const double h01 = -2.0*s3 + 3.0*s2;
    const double h11 = s3 - s2;

    *f = h00*tab_f[i] + h10*slope_f[i] + h01*tab_f[i + 1] + h11*slope_f[i + 1];

    const double G = h00*tab_G[i] + h10*slope_G[i]
                     + h01*tab_G[i + 1] + h11*slope_G[i + 1];
    *g = G - lnt;  // g = G - ln t
  }
  else {
    const double v = 1.0/(t*t);  // series variable v = 1/t^2

    *f = (1.0 - 2.0*v*(1.0 - 12.0*v*(1.0 - 30.0*v*(1.0 - 56.0*v
         *(1.0 - 90.0*v*(1.0 - 132.0*v*(1.0 - 182.0*v)))))))/t;

    *g = v*(1.0 - 6.0*v*(1.0 - 20.0*v*(1.0 - 42.0*v*(1.0 - 72.0*v
         *(1.0 - 110.0*v*(1.0 - 156.0*v*(1.0 - 210.0*v)))))));
  }
}


// ---------------------------------------------------------------------------
// Moments of the kernel's radial weight,
//
//   M_p(x) = int_0^x y^p (1 + y)^-2 dy,   p = 0 .. IA_PMAX,
//
// the coefficients of the series branch (kernel header, item 2). From
// y^(p-2) = [y^p + 2 y^(p-1) + y^(p-2)]/(1 + y)^2, integrated:
//
//   M_p + 2 M_(p-1) + M_(p-2) = x^(p-1)/(p - 1)
//   M_0 = x/(1 + x),   M_1 = ln(1 + x) - x/(1 + x)
//
// x >= 1: the recursion runs forward from M_0, M_1 (stable for x > 1:
// M_p grows like x^p/p, faster than the recursion's own solutions
// (-1)^p, p (-1)^p; at x = 1 the loss is ~p^2 eps, negligible).
// x < 1: M_p decays like x^(p+1)/(p+1), so the recursion runs backward
// from the two highest moments, each from the convergent series of
// (1 + y)^-2 about y = x (w = x/(1 + x) <= 1/2):
//
//   M_p(x) = x^(p+1)/(1 + x)^2 sum_j T_j,
//   T_0 = 1/(p + 1),   T_(j+1) = T_j w (j + 2)/(p + j + 2)
//
// The powers x^p are built upward by multiplication, so a tiny x only
// underflows the highest powers (whose moments are negligible).
//
// Parameters:
//   x      - the edge, x > 0
//   moment - output: [IA_PMAX + 1] M_0 .. M_IA_PMAX
// ---------------------------------------------------------------------------
static void ia_moments(
    const double x,  // edge x > 0
    double* moment   // output: [IA_PMAX + 1] M_p(x)
  )
{
  // relative size of the last series term kept
  const double series_tol = 1.0e-17;

  // x^p, p = 0 .. IA_PMAX + 1, upward
  double x_pow[IA_PMAX + 2];
  x_pow[0] = 1.0;
  for (int p=1; p<IA_PMAX+2; p++) {
    x_pow[p] = x_pow[p - 1]*x;
  }

  if (x >= 1.0) {
    // --- FORWARD RECURSION ---
    moment[0] = x/(1.0 + x);
    moment[1] = log1p(x) - x/(1.0 + x);
    for (int p=2; p<=IA_PMAX; p++) {
      moment[p] = x_pow[p - 1]/(p - 1) - 2.0*moment[p - 1] - moment[p - 2];
    }
    return;
  }

  // --- TOP TWO MOMENTS FROM THE SERIES ABOUT y = x ---
  const double w = x/(1.0 + x);
  const double inv_one_plus_x2 = 1.0/((1.0 + x)*(1.0 + x));

  for (int p=IA_PMAX-1; p<=IA_PMAX; p++) {
    double term = 1.0/(p + 1);
    double sum  = term;
    for (int j=0; term > series_tol*sum; j++) {
      term *= w*(j + 2)/(p + j + 2);
      sum  += term;
    }
    moment[p] = x_pow[p + 1]*inv_one_plus_x2*sum;
  }

  // --- BACKWARD RECURSION ---
  for (int p=IA_PMAX; p>1; p--) {
    moment[p - 2] = x_pow[p - 1]/(p - 1) - 2.0*moment[p - 1] - moment[p];
  }
}


// Horner evaluation of sum_{n < count} coef[n] y^n
static inline double ia_horner(
    const double* coef,  // [count] coefficients, lowest power first
    const int count,     // number of coefficients
    const double y       // the variable
  )
{
  double acc = 0.0;
  for (int n=count-1; n>=0; n--) {
    acc = acc*y + coef[n];
  }
  return acc;
}


// ---------------------------------------------------------------------------
// The per-halo constants of one edge x of the kernel's radial integrals
// (kernel header): for every multipole l <= 2 n_l and both slopes beta,
//
//   switch_u[l]   s_l(x), the series/closed-form switch in u = t x
//   n_series[l]   K = ceil(K0 + K1 s_l), the series terms
//   series        a_k M_(beta+l+2k)(x), k < K        (series branch)
//   r_sin, r_cos  the x-dependent coefficients r_s,j, r_c,j of R_s, R_c
//                 (closed form; the tables' header)
//
// Parameters:
//   x    - the edge, x > 0
//   n_l  - multipoles l = 2 .. 2 n_l
//   edge - output
// ---------------------------------------------------------------------------
static void ia_edge_setup(
    const double x,  // edge x > 0, in units of r_s
    const int n_l,   // multipoles l = 2 .. 2 n_l
    ia_edge* edge    // output
  )
{
  edge->x    = x;
  edge->lnx  = log(x);
  edge->ln1x = log1p(x);

  double moment[IA_PMAX + 1];
  ia_moments(x, moment);

  // x^p for the denominators x^p (1 + x) of r_s,j, r_c,j
  double x_pow[IA_NCOEF];
  x_pow[0] = 1.0;
  for (int p=1; p<IA_NCOEF; p++) {
    x_pow[p] = x_pow[p - 1]*x;
  }
  const double one_plus_x = 1.0 + x;

  for (int li=0; li<n_l; li++) {
    const int l      = 2*li + 2;
    const int n_half = li + 2;  // coefficients of one parity, (l + 2)/2

    // --- 1. SWITCH AND SERIES LENGTH ---
    const double switch_u = fmin(fmax(IA_SWITCH_U_MIN[li],
                                      IA_SWITCH_T_MIN[li]*x),
                                 IA_SWITCH_U_MAX);
    const int n_terms = (int) ceil(IA_SERIES_K0 + IA_SERIES_K1*switch_u);
    edge->switch_u[li] = switch_u;
    edge->n_series[li] = n_terms;

    // --- 2. SERIES COEFFICIENTS a_k M_(beta+l+2k)(x) ---
    // a_0 = 1/(2l + 1)!!, a_k = -a_(k-1)/(2k (2l + 2k + 1))
    double a_k = 1.0;
    for (int n=2*l+1; n>1; n-=2) {
      a_k /= n;
    }
    for (int k=0; k<n_terms; k++) {
      if (k > 0) {
        a_k *= -1.0/(2.0*k*(2*l + 2*k + 1));
      }
      edge->series[li][0][k] = a_k*moment[1 + l + 2*k];   // beta = +1
      edge->series[li][1][k] = a_k*moment[-1 + l + 2*k];  // beta = -1
    }

    // --- 3. CLOSED-FORM COEFFICIENTS r_s,j(x), r_c,j(x) ---
    for (int bi=0; bi<2; bi++) {
      for (int j=0; j<n_half; j++) {
        const double num_sin = ia_horner(IA_R_S_NUM[li][bi][j], IA_NCOEF, x);
        const double num_cos = ia_horner(IA_R_C_NUM[li][bi][j], IA_NCOEF, x);
        const double den_sin = x_pow[IA_R_S_XPOW[li][bi][j]]*one_plus_x;
        const double den_cos = x_pow[IA_R_C_XPOW[li][bi][j]]*one_plus_x;
        edge->r_sin[li][bi][j] = num_sin/den_sin;
        edge->r_cos[li][bi][j] = num_cos/den_cos;
      }
    }
  }
}


// Per-k quantities of one edge that every multipole shares: the phase
// u = t x and the three f, g combinations of the closed form G(x)
typedef struct {
  double sin_u;    // sin u
  double cos_u;    // cos u
  double comb_u;   // f(u) cos u + g(u) sin u   (multiplies S_u)
  double comb_z1;  // f(z) cos u + g(z) sin u   (multiplies S_z)
  double comb_z2;  // f(z) sin u - g(z) cos u   (multiplies C_z)
} ia_phase;


// Per-k quantities of one multipole l and slope beta: the polynomials in
// 1/t of the closed form and its value at 0 (tables' header)
typedef struct {
  double S_u;
  double S_z;
  double C_z;
  double G_zero;  // G(0)
} ia_poly;


// The closed-form antiderivative G(x) at one edge (kernel header,
// item 1), for multipole index li and slope index bi (0: beta = +1,
// 1: beta = -1); R_s, R_c from the edge's coefficients, Horner in 1/t^2
static inline double ia_closed_G(
    const ia_edge* edge,   // per-halo constants of the edge
    const ia_phase* phase, // per-k phase and f, g combinations
    const ia_poly* poly,   // per-k polynomials of (l, beta)
    const int li,          // l/2 - 1
    const int bi,          // 0: beta = +1, 1: beta = -1
    const double inv_t,    // 1/t
    const double inv_t2    // 1/t^2
  )
{
  const int n_half = li + 2;
  const double R_s = inv_t*ia_horner(edge->r_sin[li][bi], n_half, inv_t2);
  const double R_c = ia_horner(edge->r_cos[li][bi], n_half, inv_t2);

  return R_s*phase->sin_u + R_c*phase->cos_u
         - poly->S_u*phase->comb_u
         - poly->S_z*phase->comb_z1
         + poly->C_z*phase->comb_z2;
}


// The series E(x) = t^l sum_k a_k M_(beta+l+2k)(x) t^(2k) at one edge
// (kernel header, item 2), Horner in t^2
static inline double ia_series_E(
    const ia_edge* edge,   // per-halo constants of the edge
    const int li,          // l/2 - 1
    const int bi,          // 0: beta = +1, 1: beta = -1
    const double t2,       // t^2
    const double t_pow_l   // t^l
  )
{
  const double sum = ia_horner(edge->series[li][bi], edge->n_series[li], t2);
  return t_pow_l*sum;
}


// ---------------------------------------------------------------------------
// m(c) gamma_hat(k|M) of one halo at one k: the satellite kernel of the
// section banner, in the form the table builder sums,
//
//   m(c) gamma_hat = sum_{l = 2 .. 2 n_l} P_l K_l(t),   t = k r_s
//   K_l(t) = (c/x_e)^2 E_l^+1(x_in) + c^2 [E_l^-1(c) - E_l^-1(x_in)]
//
// (the second piece only when x_e < c). Each E(x) is the series or
// G(x) - G(0) (kernel header). When both edges of the power-law piece
// use the closed form, its integral is G(c) - G(x_in) directly: G(0)
// cancels, and leaving it out avoids its rounding error, which would
// otherwise dominate at large t (where the piece is small and G(0) is
// not).
//
// Work per call: at each edge that reaches the closed form for some l,
// one sin/cos pair and two f, g reads (at u = t x and z = t (1 + x));
// one more f, g read at t for G(0); then short Horner polynomials. The
// caller passes ln t (the reads' axis); the 1/m(c) is folded into the
// table weights w_dI, w_II (ia_tables).
//
// Cache invalidation:
//   ia_fg_read (its header)
//
// Parameters:
//   halo - the halo's per-halo constants (ia_tables, halo block)
//   t    - k r_s
//   lnt  - ln t
//   n_l  - multipoles l = 2 .. 2 n_l
//
// Returns:
//   m(c) gamma_hat(k|M) per unit a_1h, dimensionless
// ---------------------------------------------------------------------------
static inline double ia_gamma_hat_m(
    const ia_halo* halo,  // per-halo constants
    const double t,       // k r_s
    const double lnt,     // ln t
    const int n_l         // multipoles l = 2 .. 2 n_l
  )
{
  const double inv_t   = 1.0/t;
  const double inv_t2  = inv_t*inv_t;
  const double t2      = t*t;
  const int    n_edges = 1 + halo->has_power;

  // --- 1. PER EDGE: PHASE, AND f, g WHERE THE CLOSED FORM IS USED ---
  ia_phase phase[2] = {{0.0, 0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0, 0.0}};
  double   u_edge[2] = {0.0, 0.0};
  int      any_closed = 0;

  for (int e=0; e<n_edges; e++) {
    const ia_edge* edge = &halo->edge[e];
    const double u = t*edge->x;
    u_edge[e] = u;

    // the switch grows with l, so l = 2 reaches the closed form first
    if (u >= edge->switch_u[0]) {
      double f_u;
      double g_u;
      double f_z;
      double g_z;
      ia_fg_read(u, lnt + edge->lnx, &f_u, &g_u);        // at u = t x
      ia_fg_read(t + u, lnt + edge->ln1x, &f_z, &g_z);   // z = t (1 + x)

      const double sin_u = sin(u);
      const double cos_u = cos(u);
      phase[e].sin_u   = sin_u;
      phase[e].cos_u   = cos_u;
      phase[e].comb_u  = f_u*cos_u + g_u*sin_u;
      phase[e].comb_z1 = f_z*cos_u + g_z*sin_u;
      phase[e].comb_z2 = f_z*sin_u - g_z*cos_u;
      any_closed = 1;
    }
  }

  // f(t), g(t) for G(0)
  double f_t = 0.0;
  double g_t = 0.0;
  if (any_closed) {
    ia_fg_read(t, lnt, &f_t, &g_t);
  }

  // --- 2. SUM OVER MULTIPOLES ---
  const ia_edge* inner = &halo->edge[0];  // x_in
  const ia_edge* outer = &halo->edge[1];  // c

  double gamma_m = 0.0;  // m(c) gamma_hat
  double t_pow_l = 1.0;  // t^l

  for (int li=0; li<n_l; li++) {
    const int l      = 2*li + 2;
    const int n_half = li + 2;
    t_pow_l *= t2;

    const int inner_closed = (u_edge[0] >= inner->switch_u[li]);
    int outer_closed = 0;
    if (halo->has_power) {
      outer_closed = (u_edge[1] >= outer->switch_u[li]);
    }

    // the polynomials in 1/t of this l, both slopes (closed form only)
    ia_poly poly[2] = {{0.0, 0.0, 0.0, 0.0}, {0.0, 0.0, 0.0, 0.0}};
    if (inner_closed || outer_closed) {
      for (int bi=0; bi<2; bi++) {
        poly[bi].S_u = inv_t*ia_horner(IA_S_U[li][bi], n_half, inv_t2);
        poly[bi].S_z = inv_t*ia_horner(IA_S_Z[li][bi], n_half, inv_t2);
        poly[bi].C_z = ia_horner(IA_C_Z[li][bi], n_half, inv_t2);

        const double G0_f = inv_t*ia_horner(IA_G0_F[li][bi], n_half, inv_t2);
        const double G0_g = ia_horner(IA_G0_G[li][bi], n_half, inv_t2);
        const double G0_c = ia_horner(IA_G0_C[li][bi], l + 2, inv_t);
        poly[bi].G_zero = f_t*G0_f + g_t*G0_g + G0_c;
      }
    }

    // inner piece: g = (c/x_e)^2 flat, beta = +1 on [0, x_in]
    double E_inner;
    if (inner_closed) {
      E_inner = ia_closed_G(inner, &phase[0], &poly[0], li, 0, inv_t, inv_t2)
                - poly[0].G_zero;
    }
    else {
      E_inner = ia_series_E(inner, li, 0, t2, t_pow_l);
    }
    double K_l = halo->w_inner*E_inner;

    // power-law piece: g = c^2 x^-2, beta = -1 on [x_in, c]
    if (halo->has_power) {
      double K_power;
      if (inner_closed && outer_closed) {
        // G(c) - G(x_in): G(0) cancels
        const double G_hi = ia_closed_G(outer, &phase[1], &poly[1], li, 1,
                                        inv_t, inv_t2);
        const double G_lo = ia_closed_G(inner, &phase[0], &poly[1], li, 1,
                                        inv_t, inv_t2);
        K_power = G_hi - G_lo;
      }
      else {
        // E(c) - E(x_in), each edge by its own branch
        double E_hi;
        double E_lo;
        if (outer_closed) {
          E_hi = ia_closed_G(outer, &phase[1], &poly[1], li, 1, inv_t, inv_t2)
                 - poly[1].G_zero;
        }
        else {
          E_hi = ia_series_E(outer, li, 1, t2, t_pow_l);
        }
        if (inner_closed) {
          E_lo = ia_closed_G(inner, &phase[0], &poly[1], li, 1, inv_t, inv_t2)
                 - poly[1].G_zero;
        }
        else {
          E_lo = ia_series_E(inner, li, 1, t2, t_pow_l);
        }
        K_power = E_hi - E_lo;
      }
      K_l += halo->conc2*K_power;
    }

    gamma_m += IA_MULTIPOLE_WEIGHT[li]*K_l;
  }

  return gamma_m;
}


// ---------------------------------------------------------------------------
// Fills ia_: the 1-halo IA sums S_dI, S_II and the red-central fraction
// f_rc on one (a, ln k) grid over the source redshift range (section
// banner for the physics):
//
//   S_dI(k, a) = int dlnM dn/dlnM (M/rho_m) u(k|M) (N_rs/n_g) |gamma_hat|
//   S_II(k, a) = int dlnM dn/dlnM (N_rs/n_g)^2 gamma_hat^2
//   f_rc(a)    = int dlnM dn/dlnM N_rc / n_g(a)
//   n_g(a)     = int dlnM dn/dlnM N_tot
//
//   dn/dlnM = (rho_cb/M) nu f(nu) dlnnu/dlnM,  nu = delta_c/sigma_cb(M,a)
//
// with the occupations of the section banner (nuisance.ia_hod, the
// Zheng et al. form of HOD_nc, HOD_ns with the IA population's own
// parameters; nuisance.ia_red, the red fractions
// f_red = (1/2) [1 + tanh((log10 M - log10 M_red)/width)]), u(k|M) the
// NFW transform and gamma_hat the satellite kernel (ia_gamma_hat_m).
//
// 1. Quadrature: the Gauss-Legendre rule of the POWER SPECTRA banner
// (item 1: the Ntable.halo_nm ladder on Ntable.high_def_integration)
// over [ln limits.halo_m[RANGE_MIN], ln limits.halo_m[RANGE_MAX]]. Only
// the nodes with red satellites (N_rs > 0: above the satellite cutoff
// M_0) enter the S sums ("active" nodes); every node enters n_g and
// f_rc.
//
// 2. Grids: Ntable.halo_ia_na nodes uniform in a over the source range
// [min_i amin_source(i), max_i amax_source(i)], set at every refill
// (it moves with the source photo-z parameters); the ln k grid of the
// spectra (Ntable.N_k_nlin nodes on [ln k_min, ln k_max]). The S sums are
// computed exactly on the coarse ln k grid of p_gm (every k_step-th
// node, plus pads) and their logs splined to the dense grid
// (ln_k_spline_upsample).
//
// 3. Loop nests (each a separate OpenMP loop, so every expensive
// per-halo or per-k quantity is spread over all threads):
//
//   per refill, per mass node q (serial)
//     -> mass_node: M, w (rho_cb/M),
//        r_Delta, M/rho_m, N_tot, N_rc, N_rs; the active node list
//   per a row i (threaded)
//     -> row: a_1h(a); fnu_pars; dn[i][q] = w (rho_cb/M)
//        dlnnu/dlnM f(nu) nu; n_g(a_i); f_red_cen[i]
//   per row block, collapse(2) over (row, active node) (threaded)
//     -> halo: c = conc(M, a), ln(1+c), r_s, ln r_s, x_e, the weights
//        w_dI, w_II and the kernel's edge constants (ia_edge_setup)
//   per row block, collapse(2) over (row, coarse k) (threaded)
//     -> ln_coarse: ln sum_q w_dI um |gamma_m|, ln sum_q w_II gamma_m^2,
//        um = u m(c) (nfw_um), gamma_m = m(c) gamma_hat (ia_gamma_hat_m)
//   per a row i (threaded)
//     -> tab: the two coarse ln k rows splined to the dense ln k grid
//
// The halo constants of all (row, active node) pairs are held at once
// when they fit IA_HALO_SCRATCH halos; otherwise the rows are processed
// in blocks of block_rows (a scratch of block_rows x n_nodes halos).
// The 1/m(c)^2 of um and gamma_m lives in w_dI and w_II.
//
// The logs: u m(c) > 0 on the table's (c, k r_s) range (checked
// numerically for the truncated NFW transform) and |gamma_hat| >= 0, so
// both sums are positive; a non-positive sum aborts. f_1h(k)
// and the amplitude a_1h(a) multiply at read (ia_p1h_dI, ia_p1h_II), so
// the tables keep the smooth k^2 (S_dI) and k^4 (S_II) rise at low k.
//
// Thread safety: the single-threaded halo_warmup call before the
// threaded loops builds every lazy table the rows read (its header);
// the Hermite slopes of the nfw_ table are built in the rebuild block.
// Every table value is one serial sum over mass nodes; threads only
// split the table nodes, so no table depends on the thread count.
//
// Aborts: like.halo_model[3] not HALO_PROFILE_NFW; Ntable.halo_ia_lmax
// not 2, 4 or 6; an IA HOD that is not set (log10 M_min outside
// [10, 16]) or has sigma_lgM <= 0; a red-fraction width <= 0; no mass
// node with red satellites; n_g = 0 at some a.
//
// Cache invalidation:
//   rebuild (sizes, allocations, the GL rule, the ln k axes, the
//     Hermite slopes): Ntable.random
//   refill: cosmology.random, Ntable.random, nuisance.random_ia_halo,
//     redshift.random_shear, nuisance.random_photoz_shear
// ---------------------------------------------------------------------------
static const int IA_HALO_SCRATCH = 4096; // halo constants held at once

static void ia_tables(void)
{
  const int K_PAD = Ntable.halo_spline_pad; // pads per end, coarse ln k

  // --- 1. NTABLE REBUILD: SIZES, ALLOCATIONS, GL RULE, AXES ---
  if (NULL == ia_.tab || fdiff2(ia_.cache[1], Ntable.random)) {
    if (ia_.tab != NULL) {
      free(ia_.tab);
      free(ia_.f_red_cen);
      free(ia_.mass_node);
      free(ia_.active);
      free(ia_.row);
      free(ia_.fnu_pars);
      free(ia_.dn);
      free(ia_.halo);
      free(ia_.ln_coarse);
      free(ia_.curv);
      free(ia_.k_mult);
      free(ia_.fg_slope);
    }

    const int accuracy = abs(Ntable.high_def_integration);

    // mass-node ladder of the spectra (POWER SPECTRA banner, item 1)
    if (0 == accuracy) {
      ia_.n_nodes = Ntable.halo_nm;
    }
    else if (1 == accuracy) {
      ia_.n_nodes = 2*Ntable.halo_nm;
    }
    else if (2 == accuracy) {
      ia_.n_nodes = 4*Ntable.halo_nm;
    }
    else {
      ia_.n_nodes = 1024;
    }

    // coarse ln k step of the S sums: the p_gm ladder
    if (0 == accuracy) {
      ia_.k_step = Ntable.halo_nk_step;
    }
    else if (1 == accuracy) {
      ia_.k_step = Ntable.halo_nk_step/2;
    }
    else {
      ia_.k_step = 1;
    }
    if (ia_.k_step < 1) {
      ia_.k_step = 1;
    }

    ia_.n_a      = Ntable.halo_ia_na;
    ia_.n_coarse = (Ntable.N_k_nlin - 1)/ia_.k_step + 2 + 2*K_PAD;

    // a rows per block of the halo scratch (header, item 3)
    ia_.block_rows = IA_HALO_SCRATCH/ia_.n_nodes;
    if (ia_.block_rows < 1) {
      ia_.block_rows = 1;
    }
    if (ia_.block_rows > ia_.n_a) {
      ia_.block_rows = ia_.n_a;
    }

    const int n_a      = ia_.n_a;
    const int n_nodes  = ia_.n_nodes;
    const int n_coarse = ia_.n_coarse;

    ia_.tab       = (double***) malloc3d(2, n_a, Ntable.N_k_nlin);
    ia_.f_red_cen = (double*) malloc1d(n_a);
    ia_.mass_node = (double**) malloc2d(8, n_nodes);
    ia_.active    = (int*) malloc1d_int(n_nodes);
    ia_.row       = (double**) malloc2d(2, n_a);
    ia_.fnu_pars  = (fnu_params*) malloc(sizeof(fnu_params)*n_a);
    ia_.dn        = (double**) malloc2d(n_a, n_nodes);
    ia_.halo      = (ia_halo*)
                    malloc(sizeof(ia_halo)*ia_.block_rows*n_nodes);
    ia_.ln_coarse = (double***) malloc3d(2, n_a, n_coarse);
    ia_.curv      = (double**) malloc2d(n_a, n_coarse);
    ia_.k_mult    = (double*) malloc1d(n_coarse);
    ln_k_spline_multipliers(n_coarse, ia_.k_mult);

    // GL nodes M_q and weights w_q on [ln M_min, ln M_max]
    const double lnMmin = log(limits.halo_m[RANGE_MIN]);
    const double lnMmax = log(limits.halo_m[RANGE_MAX]);
    gsl_integration_glfixed_table* gl_table = malloc_gslint_glfixed(n_nodes);
    for (int q=0; q<n_nodes; q++) {
      double lnM;
      gsl_integration_glfixed_point(lnMmin, lnMmax, q, &lnM,
                                    &ia_.mass_node[1][q], gl_table);
      ia_.mass_node[0][q] = exp(lnM);
    }
    gsl_integration_glfixed_table_free(gl_table);

    // the ln k axis of the spectra
    ia_.lim[1][0] = log(limits.k_cH0[RANGE_MIN]);
    ia_.lim[1][1] = log(limits.k_cH0[RANGE_MAX]);
    ia_.lim[1][2] = (ia_.lim[1][1] - ia_.lim[1][0])
                    /((double) Ntable.N_k_nlin - 1.0);

    // Hermite slopes of the nfw_ table (ia_fg_read header): at node
    // ln t_i, h df/dln t = -h t g(t) and h dG/dln t = h t f(t), with
    // g = G - ln t and h the node spacing
    nfw_table();
    ia_.fg_slope = (double**) malloc2d(2, nfw_.n_nodes);
    for (int i=0; i<nfw_.n_nodes; i++) {
      const double lnt = nfw_.lim[0] + i*nfw_.lim[2];
      const double t   = exp(lnt);
      const double f_i = nfw_.tab[0][i];
      const double g_i = nfw_.tab[1][i] - lnt;
      ia_.fg_slope[0][i] = -t*g_i*nfw_.lim[2];
      ia_.fg_slope[1][i] = t*f_i*nfw_.lim[2];
    }
  }

  // --- 2. REFILL ---
  if (fdiff2(ia_.cache[0], cosmology.random) ||
      fdiff2(ia_.cache[1], Ntable.random) ||
      fdiff2(ia_.cache[2], nuisance.random_ia_halo) ||
      fdiff2(ia_.cache[3], redshift.random_shear) ||
      fdiff2(ia_.cache[4], nuisance.random_photoz_shear))
  {
    // --- 2a. GUARDS, THE a GRID AND THE WARM-UP ---

    // the rows read the NFW kernel directly (POWER SPECTRA banner, item 2)
    if (like.halo_model[3] != HALO_PROFILE_NFW) {
      log_fatal("like.halo_model[3] = %d not supported", like.halo_model[3]);
      exit(1);
    }

    const int lmax = Ntable.halo_ia_lmax;
    if (lmax != 2 && lmax != 4 && lmax != 6) {
      log_fatal("Ntable.halo_ia_lmax = %d must be 2, 4 or 6", lmax);
      exit(1);
    }
    ia_.n_l = lmax/2;

    // the IA HOD, {lg M_min, sigma_lgM, lg M_1, lg M_0, alpha, f_c}:
    // lg M_min outside this range flags an HOD that was never set
    const double lgM_min_lo = 10.0;
    const double lgM_min_hi = 16.0;
    const double lgM_min    = nuisance.ia_hod[0];
    const double sigma_lgM  = nuisance.ia_hod[1];

    const int hod_unset   = (lgM_min < lgM_min_lo || lgM_min > lgM_min_hi);
    const int sigma_unset = !(sigma_lgM > 0);
    if (hod_unset || sigma_unset) {
      log_fatal("IA HOD not set (lgMmin = %g, sigma_lgM = %g)",
                lgM_min, sigma_lgM);
      exit(1);
    }

    // red fractions: centers and widths in log10 M
    const double lgM_red_cen = nuisance.ia_red[0];
    const double width_cen   = nuisance.ia_red[1];
    const double lgM_red_sat = nuisance.ia_red[2];
    const double width_sat   = nuisance.ia_red[3];
    if (!(width_cen > 0) || !(width_sat > 0)) {
      log_fatal("red-fraction widths must be > 0 (%g, %g)",
                width_cen, width_sat);
      exit(1);
    }

    // a grid over the source range, both ends included; it moves with
    // the source photo-z parameters, so it is set at every refill
    double amin = amin_source(0);
    double amax = amax_source(0);
    for (int ns=1; ns<redshift.shear_nbin; ns++) {
      amin = fmin(amin, amin_source(ns));
      amax = fmax(amax, amax_source(ns));
    }
    ia_.lim[0][0] = amin;
    ia_.lim[0][1] = amax;
    ia_.lim[0][2] = (amax - amin)/((double) ia_.n_a - 1.0);

    // every lazy table the threaded loops read is built here, on one
    // thread (header, Thread safety)
    halo_warmup(ia_.lim[0][0], exp(ia_.lim[1][0]), 0, 0);

    /* PHYSICAL DERIVATION & LOGIC FLOW (equations: header above)
       1. node q: M, w (rho_cb/M), r_Delta, occupations
          N_tot = f_c N_c + N_s, N_rc = f_c N_c f_red,cen,
          N_rs = N_s f_red,sat                          (section banner)
       2. row i: a_1h(a), dn = w (rho_hmf/M) dlnnu/dlnM f(nu) nu,
          n_g = sum dn N_tot, f_rc = sum dn N_rc/n_g
       3. (row, active node): c(M, a), r_s, r_e = max(r_floor,
          r_Delta (|a_1h|/gamma_max)^(1/2)), x_e = r_e/r_s,
          w_dI = dn (M/rho_m)(N_rs/n_g)/m^2, w_II = dn (N_rs/n_g)^2/m^2,
          edge constants at x_in = min(x_e, c) and c   (F21 Eq. 20)
       4. (row, coarse k): S_dI = sum w_dI um |gamma_m|     (F21 Eq. 17)
                           S_II = sum w_II gamma_m^2        (F21 Eq. 18)
       5. row i: ln S splined from the coarse to the dense ln k grid */

    const int n_a      = ia_.n_a;
    const int n_nodes  = ia_.n_nodes;
    const int n_coarse = ia_.n_coarse;
    const int n_l      = ia_.n_l;

    const double rho_m     = cosmology.rho_crit*cosmology.Omega_m;
    const double rho_delta = Delta*rho_m;
    // The cb density normalizes dn/dlnM (POWER SPECTRA banner).
    const double rho_hmf   = cosmology.rho_crit*omega_halo_field();

    // --- 2b. PER MASS NODE, SERIAL: WEIGHTS AND OCCUPATIONS ---
    double f_c = nuisance.ia_hod[5];  // 0 = unset: read as 1 (HOD_fc)
    if (0.0 == f_c) {
      f_c = 1.0;
    }
    const double M_1   = pow(10.0, nuisance.ia_hod[2]);
    const double M_0   = pow(10.0, nuisance.ia_hod[3]);
    const double alpha = nuisance.ia_hod[4];

    double** const mass_node = ia_.mass_node;
    ia_.n_active = 0;

    for (int q=0; q<n_nodes; q++) {
      const double m   = mass_node[0][q];
      const double lgM = log10(m);

      // N_c = (1 + erf[(lg M - lg M_min)/sigma_lgM])/2,
      // N_s = N_c [(M - M_0)/M_1]^alpha above M_0, 0 below
      const double n_cen = 0.5*(1.0 + erf((lgM - lgM_min)/sigma_lgM));
      double n_sat = 0.0;
      if (m > M_0) {
        n_sat = n_cen*pow((m - M_0)/M_1, alpha);
      }

      // red fractions f_red = (1 + tanh[(lg M - lg M_red)/width])/2
      const double f_red_cen = 0.5*(1.0 + tanh((lgM - lgM_red_cen)/width_cen));
      const double f_red_sat = 0.5*(1.0 + tanh((lgM - lgM_red_sat)/width_sat));

      mass_node[2][q] = mass_node[1][q]*(rho_hmf/m);
      mass_node[3][q] = pow(3.0/(4.0*M_PI)*(m/rho_delta), 1.0/3.0); // r_Delta
      mass_node[4][q] = m/rho_m;
      mass_node[5][q] = f_c*n_cen + n_sat;                      // N_tot
      mass_node[6][q] = f_c*n_cen*f_red_cen;                    // N_rc
      mass_node[7][q] = n_sat*f_red_sat;                        // N_rs

      if (mass_node[7][q] > 0) {
        ia_.active[ia_.n_active] = q;
        ia_.n_active++;
      }
    }
    if (0 == ia_.n_active) {
      log_fatal("IA HOD: no mass node hosts red satellites");
      exit(1);
    }
    const int n_active = ia_.n_active;

    // --- 2c. PER a ROW, THREADED: a_1h, dn, n_g, f_rc ---
    #pragma omp parallel for schedule(static)
    for (int i=0; i<n_a; i++) {
      const double ai = ia_.lim[0][0] + i*ia_.lim[0][2];
      ia_.fnu_pars[i] = fnu_params_at(ai);

      double* restrict dn_row = ia_.dn[i];
      double n_gal = 0.0;       // sum_q dn N_tot -> n_g(a_i)
      double n_red_cen = 0.0;   // sum_q dn N_rc

      for (int q=0; q<n_nodes; q++) {
        const double m = mass_node[0][q];
        const double nu = delta_c/sqrt(sigma2(m, ai));
        const double dn = mass_node[2][q]*dlognudlogm(m, ai)*fnu_core(nu, &ia_.fnu_pars[i])*nu;
        dn_row[q] = dn;
        n_gal     += dn*mass_node[5][q];
        n_red_cen += dn*mass_node[6][q];
      }

      if (!(n_gal > 0)) {
        log_fatal("IA HOD: n_g = %g at a = %g", n_gal, ai);
        exit(1);
      }

      ia_.row[0][i]     = ia_a1h(ai);
      ia_.row[1][i]     = n_gal;
      ia_.f_red_cen[i]  = n_red_cen/n_gal;
    }

    // --- 2d. ROW BLOCKS: HALO CONSTANTS, THEN THE COARSE SUMS ---
    const double r_floor   = IA_R_FLOOR_MPCH/cosmology.coverH0; // c/H0
    const double dlnk      = ia_.lim[1][2];
    const double lnk_first = ia_.lim[1][0] - K_PAD*ia_.k_step*dlnk;

    for (int i0=0; i0<n_a; i0+=ia_.block_rows) {
      int n_rows = ia_.block_rows;
      if (i0 + n_rows > n_a) {
        n_rows = n_a - i0;
      }

      // (row, active node): the kernel's per-halo constants
      #pragma omp parallel for collapse(2) schedule(static)
      for (int b=0; b<n_rows; b++) {
        for (int p=0; p<n_active; p++) {
          const int i = i0 + b;
          const int q = ia_.active[p];
          ia_halo* halo = &ia_.halo[b*n_nodes + p];

          const double m       = mass_node[0][q];
          const double ai      = ia_.lim[0][0] + i*ia_.lim[0][2];
          const double a1h_abs = fabs(ia_.row[0][i]);
          const double n_gal   = ia_.row[1][i];
          const double r_delta = mass_node[3][q];

          // NFW: c(M, a), m(c) = ln(1 + c) - c/(1 + c), r_s = r_Delta/c
          const double c    = conc(m, ai);
          const double ln1c = log1p(c);
          const double mc   = ln1c - c/(1.0 + c);
          const double r_s  = r_delta/c;

          // r_e = max(r_floor, r_Delta (|a_1h|/gamma_max)^(1/2)): g is
          // flat inside x_e = r_e/r_s (section banner)
          const double r_cap = r_delta*sqrt(a1h_abs/IA_GAMMA_MAX);
          const double r_e   = fmax(r_floor, r_cap);
          const double x_e   = r_e/r_s;
          const double x_in  = fmin(x_e, c);

          // red satellites per galaxy of the sample, N_rs/n_g
          const double sat_per_gal = mass_node[7][q]/n_gal;
          const double dn          = ia_.dn[i][q];

          halo->conc      = c;
          halo->ln1c      = ln1c;
          halo->r_s       = r_s;
          halo->lnrs      = log(r_s);
          halo->w_inner   = (c/x_e)*(c/x_e);
          halo->conc2     = c*c;
          halo->w_dI      = dn*mass_node[4][q]*sat_per_gal/(mc*mc);
          halo->w_II      = dn*sat_per_gal*sat_per_gal/(mc*mc);
          halo->has_power = (x_e < c);

          ia_edge_setup(x_in, n_l, &halo->edge[0]);
          if (halo->has_power) {
            ia_edge_setup(c, n_l, &halo->edge[1]);
          }
        }
      }

      // (row, coarse k): the two sums over the active nodes
      #pragma omp parallel for collapse(2) schedule(static)
      for (int b=0; b<n_rows; b++) {
        for (int ck=0; ck<n_coarse; ck++) {
          const int i = i0 + b;
          const double lnk = lnk_first + ck*ia_.k_step*dlnk;
          const double k_coarse = exp(lnk);
          const ia_halo* row_halo = &ia_.halo[b*n_nodes];

          double sum_dI = 0.0;  // sum_q w_dI um |gamma_m|
          double sum_II = 0.0;  // sum_q w_II gamma_m^2

          for (int p=0; p<n_active; p++) {
            const ia_halo* halo = &row_halo[p];
            const double t   = k_coarse*halo->r_s; // k r_s
            const double lnt = lnk + halo->lnrs; // ln(k r_s)

            const double um      = nfw_um(halo->conc, t, lnt, halo->ln1c);
            const double gamma_m = ia_gamma_hat_m(halo, t, lnt, n_l);

            sum_dI += halo->w_dI*um*fabs(gamma_m);
            sum_II += halo->w_II*gamma_m*gamma_m;
          }

          if (!(sum_dI > 0) || !(sum_II > 0)) {
            log_fatal("non-positive IA 1-halo sum at ln k = %g", lnk);
            exit(1);
          }
          ia_.ln_coarse[0][i][ck] = log(sum_dI);
          ia_.ln_coarse[1][i][ck] = log(sum_II);
        }
      }
    }

    // --- 2e. PER a ROW, THREADED: SPLINE TO THE DENSE ln k GRID ---
    #pragma omp parallel for schedule(static)
    for (int i=0; i<n_a; i++) {
      for (int s=0; s<2; s++) {
        ln_k_spline_upsample(ia_.ln_coarse[s][i], n_coarse, ia_.k_step,
                             K_PAD, dlnk, Ntable.N_k_nlin, ia_.k_mult,
                             ia_.curv[i], ia_.tab[s][i]);
      }
    }

    // --- 2f. CACHE TAGS: THE INPUTS THE TABLES NOW HOLD ---
    ia_.cache[0] = cosmology.random;
    ia_.cache[1] = Ntable.random;
    ia_.cache[2] = nuisance.random_ia_halo;
    ia_.cache[3] = redshift.random_shear;
    ia_.cache[4] = nuisance.random_photoz_shear;
  }
}


// ---------------------------------------------------------------------------
// f_rc(a), the fraction of the IA (source) sample that are red centrals,
//
//   f_rc(a) = int dlnM dn/dlnM f_c N_c(M) f_red,cen(M) / n_g(a)
//
// (F21's f_cen^red, the weight of the NLA 2-halo term), tabulated by
// ia_tables (its header) and interpolated linearly in a.
//
// Cache invalidation:
//   ia_tables (its header)
//
// Parameters:
//   a - scale factor
//
// Returns:
//   f_rc(a) in [0, 1]; 0 outside the source a range
// ---------------------------------------------------------------------------
double ia_f_red_central(
    const double a  // scale factor
  )
{
  ia_tables();

  if (a < ia_.lim[0][0] || a > ia_.lim[0][1]) {
    return 0.0;
  }

  return interpol1d(ia_.f_red_cen, ia_.n_a, ia_.lim[0][0], ia_.lim[0][1],
                    ia_.lim[0][2], a);
}


// ---------------------------------------------------------------------------
// P_dI^1h(k, a) = a_1h(a) f_1h(k) S_dI(k, a), the satellite (1-halo)
// part of the matter-intrinsic spectrum (F21 Eq. 17; section banner),
// with the sign of a_1h(a): the C_l cores of cosmo2D.c subtract it,
// P_dI^phys = -[f_rc C_1 P_delta f_2h + P_dI^1h]. ln S_dI is read
// bilinearly in (a, ln k) from ia_tables and exponentiated.
//
// Cache invalidation:
//   ia_tables (its header)
//
// Parameters:
//   k - wavenumber in (c/H0)^-1
//   a - scale factor
//
// Returns:
//   P_dI^1h in (c/H0)^3, signed; 0 outside the source a range; ln S_dI
//   continued with unit slope outside [ln k_min, ln k_max] (interpol2d)
// ---------------------------------------------------------------------------
double ia_p1h_dI(
    const double k,  // wavenumber in (c/H0)^-1
    const double a   // scale factor
  )
{
  ia_tables();

  if (a < ia_.lim[0][0] || a > ia_.lim[0][1]) {
    return 0.0;
  }

  const double a1h = ia_a1h(a);
  if (0.0 == a1h) {
    return 0.0;
  }

  const double ln_S = interpol2d(ia_.tab[0],
                                 ia_.n_a, ia_.lim[0][0], ia_.lim[0][1],
                                 ia_.lim[0][2], a,
                                 Ntable.N_k_nlin, ia_.lim[1][0],
                                 ia_.lim[1][1], ia_.lim[1][2], log(k));

  return a1h*ia_window_1h(k)*exp(ln_S);
}


// ---------------------------------------------------------------------------
// P_II^1h(k, a) = a_1h(a)^2 f_1h(k) S_II(k, a), the satellite (1-halo)
// part of the intrinsic-intrinsic E-mode spectrum (F21 Eq. 18; section
// banner). ln S_II is read bilinearly in (a, ln k) from ia_tables and
// exponentiated. The B mode of radial alignment vanishes by symmetry (a
// radial pattern has no 45-degree component); F21 sec. 4.1 likewise
// keeps only the II and dI satellite terms.
//
// Cache invalidation:
//   ia_tables (its header)
//
// Parameters:
//   k - wavenumber in (c/H0)^-1
//   a - scale factor
//
// Returns:
//   P_II^1h in (c/H0)^3, >= 0; 0 outside the source a range; ln S_II
//   continued with unit slope outside [ln k_min, ln k_max] (interpol2d)
// ---------------------------------------------------------------------------
double ia_p1h_II(
    const double k,  // wavenumber in (c/H0)^-1
    const double a   // scale factor
  )
{
  ia_tables();

  if (a < ia_.lim[0][0] || a > ia_.lim[0][1]) {
    return 0.0;
  }

  const double a1h = ia_a1h(a);
  if (0.0 == a1h) {
    return 0.0;
  }

  const double ln_S = interpol2d(ia_.tab[1],
                                 ia_.n_a, ia_.lim[0][0], ia_.lim[0][1],
                                 ia_.lim[0][2], a,
                                 Ntable.N_k_nlin, ia_.lim[1][0],
                                 ia_.lim[1][1], ia_.lim[1][2], log(k));

  return a1h*a1h*ia_window_1h(k)*exp(ln_S);
}



// ============================================================================
// [SECTION] MISCELLANEOUS
// ============================================================================

// b_gal of lens bin ni at one scale factor, integrated directly: the
// bgal integral of the hod_tables header with the same Gauss-Legendre
// ladder, n_gal and the bias-weighted sum from one node loop. set_HOD
// runs bin by bin, before the other bins' HOD may be set, so it cannot
// read the all-bin hod_ tables.
static double hod_bgal_direct(
    const int ni,   // lens bin
    const double a  // scale factor, 0 < a < 1
  )
{
  /* PHYSICAL DERIVATION & LOGIC FLOW
     b_gal = (1/n_gal) int dlnM dn/dlnM b(nu) <N|M>, one Gauss-Legendre
     sum over ln M:
     1. nu = delta_c/sigma_cb(M,a); occupation <N|M> = f_c N_c + N_s
     2. dn_gal = weight (rho_hmf/M) dlnnu/dlnM <N|M> f(nu) nu, the node's
        share of the galaxy number density (Tinker dn/dlnM)
     3. n_gal = sum_q dn_gal; bn_gal = sum_q b(nu) dn_gal
     4. return b_gal = bn_gal/n_gal */

  // --- 1. CONFIGURATION: NODE COUNT AND MASS RANGE ---

  // Gauss-Legendre node count: one of the predefined GSL rules, stepped
  // up by Ntable.high_def_integration
  const int high_def = abs(Ntable.high_def_integration);

  int nnode;
  if (0 == high_def) {
    nnode = 128;
  }
  else if (1 == high_def) {
    nnode = 256;
  }
  else if (2 == high_def) {
    nnode = 512;
  }
  else {
    nnode = 1024;
  }

  // ln M range: two decades below the bin's lg M_min up to the global
  // maximum halo mass
  const double lnMmin = log(10.0)*(nuisance.hod[ni][0] - 2.);
  const double lnMmax = log(limits.halo_m[RANGE_MAX]);

  // --- 2. COSMOLOGY AND HOD FACTORS AT THIS SCALE FACTOR ---

  // Cold matter plus baryons supply the mass-function density rho_cb.
  // The total-matter density belongs to the lensing weights instead.
  const double rho_hmf = cosmology.rho_crit * omega_halo_field();

  // the nu-independent halves of the Tinker mass function f(nu) and the
  // halo bias b(nu), frozen at this scale factor
  const fnu_params   fnu_pars   = fnu_params_at(a);
  const hb1nu_params hb1nu_pars = hb1nu_params_at(a);

  const double fc = HOD_fc(ni);

  // --- 3. GAUSS-LEGENDRE SUM OVER MASS NODES ---

  gsl_integration_glfixed_table* t = malloc_gslint_glfixed(nnode);

  double n_gal  = 0.0; // sum of dn_gal: the galaxy number density
  double bn_gal = 0.0; // sum of b(nu) dn_gal: the bias-weighted density

  for (int q=0; q<nnode; q++) {
    // node q of the rule mapped onto [lnMmin, lnMmax], and its weight
    double lnM;
    double weight;
    gsl_integration_glfixed_point(lnMmin, lnMmax, q, &lnM, &weight, t);

    // peak height and mean occupation <N|M> at this node's mass
    const double m          = exp(lnM);
    const double nu         = delta_c/sqrt(sigma2(m, a));
    const double occupation = fc*HOD_nc(m, a, ni) + HOD_ns(m, a, ni);

    // the node's share of the galaxy number density:
    // dn_gal = weight x dn/dlnM x <N|M>
    const double dn_gal =
        weight*(rho_hmf/m)*dlognudlogm(m, a)*occupation*fnu_core(nu, &fnu_pars)*nu;

    n_gal  += dn_gal;
    bn_gal += dn_gal*hb1nu_core(nu, &hb1nu_pars);
  }

  gsl_integration_glfixed_table_free(t);

  // --- 4. MEAN BIAS: WEIGHTED SUM OVER NUMBER SUM ---

  return bn_gal/n_gal;
}


// HOD parameters of lens bin ni, fixed to the Coupon et al. 2012 fits
// below, and the bin's mean galaxy bias <b_g> at its mean redshift
void set_HOD(const int ni)
{
  const double z = zmean(ni);
  const double a = 1.0/(z + 1.0);

  // Five-parameter HOD of Zehavi et al. 2011 (1005.2413 Eq. 7) plus f_c:
  // hod[ni][] = {lg M_min, sigma_lgM, lg M_1, lg M_0, alpha, f_c}
  // nuisance.gc[ni] = f_g, the galaxy concentration factor of u_g:
  // c_g(M) = f_g c(M)

  // Values from Coupon et al. 2012 (1107.0616), Table B.1: all galaxies
  // with M_g - 5 log h < -21.8, one row per redshift slice (0.2-0.4,
  // 0.4-0.6, 0.6-0.8, 0.8-1.0, 1.0-1.2 for bins 0-4). The table's
  // columns run lg M_min, lg M_1, lg M_0, sigma_lgM, alpha; masses in
  // M_sun/h.
  switch (ni)
  {
    case 0:
    {
      nuisance.hod[0][0] = 13.17;
      nuisance.hod[0][1] = 0.39;
      nuisance.hod[0][2] = 14.53;
      nuisance.hod[0][3] = 11.09;
      nuisance.hod[0][4] = 1.27;
      nuisance.hod[0][5] = 1.00;
      nuisance.gb[0][ni] = hod_bgal_direct(ni, a);
      break;
    }
    case 1:
    {
      nuisance.hod[1][0] = 13.18;
      nuisance.hod[1][1] = 0.30;
      nuisance.hod[1][2] = 14.47;
      nuisance.hod[1][3] = 10.93;
      nuisance.hod[1][4] = 1.36;
      nuisance.hod[1][5] = 1.00;
      nuisance.gb[0][ni] = hod_bgal_direct(ni, a);
      break;
    }
    case 2:
    {
      nuisance.hod[2][0] = 12.96;
      nuisance.hod[2][1] = 0.38;
      nuisance.hod[2][2] = 14.10;
      nuisance.hod[2][3] = 12.47;
      nuisance.hod[2][4] = 1.28;
      nuisance.hod[2][5] = 1.00;
      nuisance.gb[0][ni] = hod_bgal_direct(ni, a);
      break;
    }
    case 3:
    {
      nuisance.hod[3][0] = 12.80;
      nuisance.hod[3][1] = 0.33;
      nuisance.hod[3][2] = 13.94;
      nuisance.hod[3][3] = 12.15;
      nuisance.hod[3][4] = 1.52;
      nuisance.hod[3][5] = 1.00;
      nuisance.gb[0][ni] = hod_bgal_direct(ni, a);
      break;
    }
    case 4:
    { // the 1.0 < z < 1.2 row
      nuisance.hod[4][0] = 12.62;
      nuisance.hod[4][1] = 0.30;
      nuisance.hod[4][2] = 13.79;
      nuisance.hod[4][3] = 8.67;
      nuisance.hod[4][4] = 1.50;
      nuisance.hod[4][5] = 1.00;
      nuisance.gb[0][ni] = hod_bgal_direct(ni, a);
      break;
    }
    default:
    {
      log_fatal("no HOD parameters specified to initialize bin %d\n", ni);
      exit(1);
    }
  }

  // galaxies trace the halo concentration by default: u_g evaluates the
  // NFW transform at c_g = gc[ni] * c(M) and aborts unless gc[ni] > 0
  nuisance.gc[ni] = 1.0;

  log_debug("HOD: bin %d; <z> %.2f; <b_g> %.2f", ni, z, nuisance.gb[0][ni]);
}
