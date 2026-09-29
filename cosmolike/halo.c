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

#ifndef COSMO2D_NOT_USE_SIMD
// SIMDe vector of four doubles (simde/x86/avx2.h, included by basics.h):
// one AVX2 register on x86-64, two NEON registers on arm64.
typedef simde__m256d v4d;
#endif

// ---------------------------------------------------------------------------
// Halo model: peak-background split, halo and galaxy profiles, gas
// (electron-pressure) profiles, and the power spectra built from them.
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
//   nu(M, a) = delta_c/sigma(M, a),   sigma(M, a) = sqrt(sigma2(M)) D(a)
//
// with sigma2(M) the a = 1 variance of cosmo3D.c (top hat of Lagrangian
// radius R = (3M/(4 pi rho_crit Omega_m))^(1/3)) and D(a) the growth
// factor, D(1) = 1. This is the nu of Tinker et al. 2010 (1001.3162
// sec. 2), not the nu = delta_c^2/sigma^2 of Cooray & Sheth 2002
// (astro-ph/0206508 Eq. 57). Rare, massive halos have nu >> 1.
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
// 1001.3162; concentration: Bhattacharya et al. 2013, 1112.5479):
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
//   u_c          = u(k|M) of the halo matter profile (selects u_nfw_c)
//   u_g          = u_g(k|M), the satellite-galaxy profile: u_nfw at
//                  c_g = gc[ni] c(M) (computed inline by p_gm, p_gg)
//   u_KS         = F/F0, the bound-gas pressure shape factor: F0 the
//                  mass integral of the Komatsu-Seljak ("KS") bound-gas
//                  density profile ("0": the k = 0 integral), F the
//                  Fourier integral of the KS electron pressure
//   ks_ctheta,
//   ks_cg        = theta(x) = ln(1 + x)/x, the KS profile shape, and
//                  g(x) = x theta(x)^p, the integrand of F, at complex
//                  x (the contour integrals of the u_KS header)
//   ks_upsample1d = the 1D cubic upsampling of the u_KS tables
//   frac_bnd     = f_bnd(M), fraction of the halo mass in bound gas
//   frac_ejc     = f_ejc(M), fraction of the halo mass in ejected gas
//   W_p          = the bound-gas electron-pressure window, Y(a) B(M)
//                  u_KS (GAS PROFILES banner; "y": the Compton-y,
//                  thermal-SZ, field it sources)
//   u_y_ejc      = the ejected-gas electron-pressure window
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
// Spectra: p_XY(k, a) with X, Y in {m = matter, y = electron pressure,
// g = galaxies}: p_mm, p_my, p_yy, p_gm, p_gg.
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
//   dn/dlnM = (rho_m/M) nu f(nu) dln nu/dln M           -> dlognudlogm
//
// bias_norm measures how much of the consistency relation int b f dnu = 1
// the finite mass range of the halo-model integrals covers; the 2-halo
// sums of p_mm, p_my and p_yy add the rest, 1 - bias_norm, back as halos of
// mass M_min. conc gives the NFW concentration as a function of nu.


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
// rare, massive halos (b ~ 5 at nu = 3). There is no redshift
// dependence: the fit combines all outputs 0 <= z <= 2.5 and finds
// none at fixed nu (1001.3162 sec. 3.1). The scale factor stays in the
// signature so that a z-dependent fit could use it.
//
// Parameters:
//   nu - peak height delta_c/sigma(M, a)
//   a  - scale factor (unused by the Tinker fit)
//
// Returns:
//   b(nu), dimensionless. like.halo_model[1] selects the fit;
//   HALO_BIAS_TINKER_2010 is the only option, other values abort.
// ---------------------------------------------------------------------------
typedef struct {
  double ALPHA; // A of Eq. 6 (1001.3162)
  double pa;    // exponent a = 0.44 y - 0.88, y = log10(Delta)
  double dca;   // delta_c^a
  double BETA;  // B = 0.183
  double GAMMA; // C
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
      p.ALPHA = 1.00005974393421592059;
      p.pa    = 0.132453198092151725894;
      p.dca   = 1.07163776686581864305;
      p.BETA  = 0.183;
      p.GAMMA = 0.265230764366423426079;
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
    const double nu,        // peak height delta_c/(sigma(M) D(a))
    const hb1nu_params* p   // coefficients from hb1nu_params_at
  )
{
  // Eq. 6 with nu_alpha = nu^a, nu_beta = nu^b (b = 1.5) and
  // nu_gamma = nu^c (c = 2.4).
  const double nu_alpha = pow(nu, p->pa);
  const double nu_beta  = pow(nu, 1.5);
  const double nu_gamma = pow(nu, 2.4);
  return 1.0 - p->ALPHA * nu_alpha / (nu_alpha + p->dca)
             + p->BETA * nu_beta + p->GAMMA * nu_gamma;
}


double hb1nu(
    const double nu, // peak height delta_c/(sigma(M) D(a))
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
//   dn/dM = f(nu) (rho_m/M) dnu/dM
//     ->  dn/dlnM = (rho_m/M) nu f(nu) dln nu/dln M,
//
// so nu f(nu) is the g(sigma) of Tinker et al. 2008 (1001.3162 sec. 4).
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
//   fit; HMF_TINKER_2010 is the only option, other values abort.
// ---------------------------------------------------------------------------
typedef struct {
  double alpha; // amplitude: 1 from fnu_shape, Eq. 7 from tinker_alpha
  double beta;  // the four shape parameters, Eqs. 9-12 + Table 4 of
  double gamma; //   1001.3162 at Delta = 200
  double phi;
  double eta;
} fnu_params;

// The four shape parameters of Eq. 8 at aa (Eqs. 9-12) with alpha = 1:
// the ftilde of the Eq. 7 integral. No clamp on aa: tinker_alpha calls
// this beyond [0.25, 1] at its padding nodes; fnu_params_at clamps.
static inline fnu_params fnu_shape(
    const double aa  // scale factor of the Tinker evolution (no clamp)
  )
{
  fnu_params p;
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
    const double nu,      // peak height delta_c/(sigma(M) D(a))
    const fnu_params* p   // parameters from fnu_params_at or fnu_shape
  )
{
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
// with b the Tinker bias (hb1nu_core) and ftilde the Eq. 8 shape at
// alpha = 1 (fnu_shape). alpha depends on aa alone, so it is tabulated
// once and read by linear interpolation.
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
//   bias fit) differ from the pair the table holds. Nothing else
//   enters: not the cosmology (nu is the integration variable) and not
//   Ntable.
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
  static double* table = NULL;   // [ND] alpha on the dense aa grid
  static double lim[3];          // aa_min, aa_max, dense spacing
  const int ND = 4096;           // dense lookup nodes on [0.25, 1]

  // Build block: the first call, or a change of the fits in use.
  if (NULL == table ||
      key[0] != like.halo_model[0] ||
      key[1] != like.halo_model[1])
  {
    /* PHYSICAL DERIVATION & LOGIC FLOW
       1. trapezoid in s = ln nu: bias_weight[q] = w_q nu_q b(nu_q)
       2. alpha_coarse[i] = 1/sum_q bias_weight[q] ftilde(nu_q; aa_i)
       3. natural cubic spline through alpha_coarse -> table[] on the
          dense aa grid (full derivation: the header above) */

    // --- 1. COARSE PADDED aa NODES ---
    // aa_i = aa0 + i hc: NC exact nodes on [0.25, 1] plus PAD beyond
    // each end; 0.75 below is the width of the [0.25, 1] range.
    const int NC  = 128;  // exact nodes on [0.25, 1]
    const int PAD = 6;    // exact padding nodes beyond each end
    const int NE  = NC + 2*PAD;
    const double hc  = 0.75/((double) NC - 1.0);
    const double aa0 = 0.25 - PAD*hc;

    // --- 2. TRAPEZOID RULE IN s = ln nu ---
    // Nodes s_q = SMIN + q DS (NS = 936). bias_weight[q] folds the
    // trapezoid weight w_q, the Jacobian of dnu = nu ds and the Tinker
    // bias b(nu_q): only ftilde still depends on aa. The bias fit does
    // not evolve, so hb1nu_params_at takes any a; 1.0 is a placeholder.
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
    default:
    {
      log_fatal("like.halo_model[0] = %d not supported", like.halo_model[0]);
      exit(1);
    }
  }
  return p;
}


double fnu(
    const double nu, // peak height delta_c/(sigma(M) D(a))
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
//   c(M, z) = 9.0 nu^-0.29 D(z)^1.15,   nu = delta_c/(sigma(M) D(z))
//
// with sigma(M) the a = 1 rms of the linear density field (sigma2 in
// cosmo3D.c, sigma2 = sigma^2) and D the growth factor, D(1) = 1. At
// fixed nu the amplitude falls with D, so at fixed mass the c-M
// relation flattens toward high z (1112.5479 sec. 4.1-4.2).
//
// The fit is calibrated at z = 0-2 and group-to-cluster masses, with
// delta_c = 1.673 where this file uses 1.686 (a -0.2% shift in c). The
// halo model evaluates it over all of [limits.halo_m_min,
// limits.halo_m_max] and at every z, so also in extrapolation.
//
// Parameters:
//   m         - halo mass in M_sun/h
//   growfac_a - growth factor D(a), D(1) = 1 (not the scale factor)
//
// Returns:
//   c, dimensionless. like.halo_model[2] selects the fit;
//   CONCENTRATION_BHATTACHARYA_2013 is the only option, other values
//   abort.
// ---------------------------------------------------------------------------
double conc(
    const double m,         // halo mass in M_sun/h
    const double growfac_a  // growth factor D(a), not a itself
  )
{
  double c;
  switch(like.halo_model[2])
  {
    case CONCENTRATION_BHATTACHARYA_2013:
    {
      // Bhattacharya et al. 2013, Delta = 200 rho_{mean} (Table 2, full
      // halo sample): c = 9.0 nu^-0.29 D^1.15
      const double nu = delta_c/(sqrt(sigma2(m))*growfac_a); // nu(M, z)
      c = 9.0*pow(nu, -0.29)*pow(growfac_a, 1.15);
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
// Cached bias_norm(a): the Tinker bias integral over the tabulated mass
// range,
//
//   bias_norm(a) = int_{nu_min(a)}^{nu_max(a)} b(nu) f(nu, a) dnu,
//   nu(M, a)     = delta_c / (sigma(M) D(a)),
//
// with b the linear halo bias (hb1nu), f the multiplicity function
// (fnu), sigma(M) the a = 1 value (sigma2 in cosmo3D.c), D the growth
// factor with D(1) = 1, and nu_min, nu_max the peak heights of
// limits.halo_m_min and limits.halo_m_max.
//
// Why the 2-halo term needs it: matter is unbiased with respect to
// itself, int b f dnu = 1 over all nu, so P_2h = I11_m^2 P_lin (file
// glossary) tends to P_lin as k -> 0. The mass integrals of this file
// stop at M_min, and f grows toward light halos: the halos below
// M_min = 1e6 M_sun/h hold about 0.2 of the integral at z = 0 for a
// Planck-like cosmology, so I11_m(k -> 0) would be ~0.8 and
// P_2h -> 0.64 P_lin. The I11 sums of p_mm, p_my and p_yy add the
// missing 1 - bias_norm(a) back as halos of mass M_min; this function
// measures the shortfall.
//
// One quadrature for every a: both limits scale as 1/D(a), so in
// t = nu D(a), the peak height the same halo has at a = 1,
//
//   bias_norm(a) = (1/D) int_{t_min}^{t_max} b(t/D) f(t/D, a) dt,
//   t_min = delta_c/sigma(M_min),   t_max = delta_c/sigma(M_max),
//
// and a Gauss-Legendre rule on [t_min, t_max], nodes x_q and weights w_q
// on [-1, 1], gives
//
//   bias_norm(a) = (h/D) sum_q w_q b(nu_q) f(nu_q, a),
//   nu_q = (m + h x_q)/D,   m = (t_max + t_min)/2,   h = (t_max - t_min)/2.
//
// Code map (refill block): agrid[i] = a_i, Ntable.N_a nodes uniform in
// a on [limits.a_min, 0.9999999];
//   tmin, tmax, tmid, thalf = t_min, t_max, m, h
//   x[q], w[q] = x_q, w_q on [-1, 1]
//   D = D(a_i)                     hb1nu_core(nu, &bias_par) = b(nu)
//   table[i] = bias_norm(a_i)      fnu_core(nu, &fnu_par) = f(nu, a_i)
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

    // x_q, w_q on [-1, 1]; the refill maps them onto [t_min, t_max].
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
    /* PHYSICAL DERIVATION & LOGIC FLOW
       1. t = nu D(a), the a = 1 peak height: [tmin, tmax] is the same
          range for every a
       2. nu_q = (tmid + thalf x_q)/D(a_i), Gauss-Legendre on [-1, 1]
       3. table[i] = (thalf/D) sum_q w_q b(nu_q) f(nu_q, a_i) */

    // builds the fnu and sigma2 tables before the threads start
    (void) fnu(1.0, agrid[0]);

    // tmin, tmax: a = 1 peak heights of M_min, M_max (heavy halos:
    // small sigma, large t); tmid, thalf: midpoint and half-width
    const double tmin  = delta_c/sqrt(sigma2(limits.halo_m_min));
    const double tmax  = delta_c/sqrt(sigma2(limits.halo_m_max));
    const double tmid  = 0.5*(tmax + tmin);
    const double thalf = 0.5*(tmax - tmin);

    const double* restrict x = gl_node;
    const double* restrict w = gl_weight;
    const int n = n_gauss;

    #pragma omp parallel for schedule(static)
    for (int i=0; i<Ntable.N_a; i++) {
      // D(a_i) and the nu-independent coefficients of b and f(., a_i)
      const double D = growfac(agrid[i]);
      const hb1nu_params bias_par = hb1nu_params_at(agrid[i]);
      const fnu_params fnu_par = fnu_params_at(agrid[i]);

      // sum_q w_q b(nu_q) f(nu_q, a_i),   nu_q = (tmid + thalf x_q)/D
      double sum = 0.0;
      for (int q=0; q<n; q++) {
        const double nu = (tmid + thalf*x[q])/D;
        sum += w[q]*hb1nu_core(nu, &bias_par)*fnu_core(nu, &fnu_par);
      }

      // bias_norm(a_i): thalf from dt = thalf dx, 1/D from dnu = dt/D
      table[i] = sum*thalf/D;
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
// From nu = delta_c/(sqrt(sigma2(M)) D(a)),
//
//   ln nu = ln delta_c - ln D(a) - (1/2) ln sigma2(M)
//     ->  d ln nu/d ln M = -(1/2) d ln sigma2/d ln M
//
// delta_c and D(a) drop out, so one table at a = 1 serves every
// redshift (sigma(M, a) = sigma(M) D(a)). sigma falls with M, so the
// slope is positive; for a local power law P ~ k^n_eff it is
// (n_eff + 3)/6: about 0.05 for the lightest halos (n_eff near -3) and
// 0.3 for clusters.
//
// Code map: table[i] is the slope at ln M_i = lim[0] + i lim[2],
// Ntable.N_M nodes uniform in ln M over [ln limits.halo_m_min,
// ln limits.halo_m_max], as the symmetric difference
//
//   -(1/2) [ln sigma2(M_hi) - ln sigma2(M_lo)] / (ln M_hi - ln M_lo),
//   ln M_lo, ln M_hi = ln M_i -+ 0.05   (lnMlo, lnMhi in the code)
//
// pulled inside the mass range at the two edges (one-sided there): the
// sigma2 table clamps outside it and would flatten the slope.
//
// Cache invalidation:
//   allocation and ln M limits: rebuilt when Ntable.random changes
//   table refill: cosmology.random (cache[0]) or Ntable.random (cache[1])
//
// Parameters:
//   M - halo mass in M_sun/h
//
// Returns:
//   d ln nu/d ln M, dimensionless and positive, linearly interpolated
//   in ln M; constant outside [limits.halo_m_min, limits.halo_m_max]
// ---------------------------------------------------------------------------
double dlognudlogm(
    const double M  // halo mass in M_sun/h
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* table = NULL;
  static double lim[3];

  // --- 1. NTABLE REBUILD: ALLOCATION AND THE ln M LIMITS ---
  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) {
      free(table);
    }
    table = (double*) malloc(sizeof(double)*Ntable.N_M);
    lim[0] = log(limits.halo_m_min);
    lim[1] = log(limits.halo_m_max);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_M - 1.0);
  }

  // --- 2. TABLE REFILL: SYMMETRIC DIFFERENCE AT EVERY MASS NODE ---
  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random)) {
    /* PHYSICAL DERIVATION & LOGIC FLOW
       1. ln nu = ln delta_c - ln D(a) - (1/2) ln sigma2(M)
       2. table[i] = -(1/2) dln sigma2/dln M at ln M_i, the symmetric
          difference over [lnMlo, lnMhi] = ln M_i -+ half_step */

    (void) sigma2(exp(lim[0])); // builds the sigma2 table before the threads

    // the difference is pulled inside the mass range at the two edges
    // (the sigma2 table clamps outside it and would flatten the slope)
    #pragma omp parallel for schedule(static,1)
    for (int i=0; i<Ntable.N_M; i++) {
      const double half_step = 0.05;  // half-width in ln M
      const double lnMlo = fmax(lim[0] + i*lim[2] - half_step, lim[0]);
      const double lnMhi = fmin(lim[0] + i*lim[2] + half_step, lim[1]);
      table[i] = -0.5*(log(sigma2(exp(lnMhi))) - log(sigma2(exp(lnMlo))))
                 /(lnMhi - lnMlo);
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }

  return interpol1d(table, Ntable.N_M, lim[0], lim[1], lim[2], log(M));
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
// The NFW transform table nfw_, shared by u_nfw_c and the p_mm table
// builder: the two smooth functions f, g of Abramowitz & Stegun
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


// ---------------------------------------------------------------------------
// Normalized Fourier transform u(k|M) of the halo matter profile, with
// the profile chosen by like.halo_model[3]: HALO_PROFILE_NFW (u_nfw_c)
// is the only option, other values abort.
//
// Parameters:
//   c - concentration r_Delta/r_s
//   k - wavenumber in (c/H0)^-1
//   m - halo mass in M_sun/h
//   a - scale factor
//
// Returns:
//   u(k|M), dimensionless; 1 at k -> 0
// ---------------------------------------------------------------------------
double u_c(
    const double c, // concentration r_Delta/r_s
    const double k, // wavenumber in (c/H0)^-1
    const double m, // halo mass in M_sun/h
    const double a  // scale factor (passed to the selected profile)
  )
{
  double ans;

  switch (like.halo_model[3])
  {
    case HALO_PROFILE_NFW:
    {
      ans = u_nfw_c(c, k, m, a);
      break;
    }
    default:
    {
      log_fatal("like.halo_model[3] = %d not supported", like.halo_model[3]);
      exit(1);
    }
  }

  return ans;
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
// [SECTION] GAS PROFILES
// ============================================================================
//
// The electron-pressure (thermal SZ) side of the halo model, after the
// HMx model of Mead et al. 2020 (2005.00009 secs. 3.2-3.3). The baryons
// that belong to a halo of mass M split into three parts:
//
//   f_bnd(M) = gas bound inside r_Delta, in hydrostatic
//              equilibrium, Komatsu-Seljak profile        -> frac_bnd
//   f_*(M)   = stars (a local of frac_ejc)
//   f_ejc(M) = gas ejected beyond r_Delta,
//              Omega_b/Omega_m - f_bnd - f_*               -> frac_ejc
//
// The bound gas enters both halo terms through its pressure window
// W_p; the ejected gas is a smooth, warm component that enters
// the 2-halo term only (u_y_ejc).
//
// Both windows are volume integrals of the electron pressure, i.e.
// energies, in units of U = G (M_sun/h)^2/(c/H0). No factor
// sigma_T/(m_e c^2) is applied: the "y" functions below return pressure
// windows, not Compton-y.
//
// The gas parameters, nuisance.gas[0..10] (structs.h):
//
//   [0]  = Gamma       polytropic index of the bound gas, > 1
//   [1]  = beta        mass slope of f_bnd
//   [2]  = log10 M_0   mass at which halos keep half their gas bound
//   [3]  = eps1        (not read in this file)
//   [4]  = eps2        (not read in this file)
//   [5]  = alpha       bound-gas temperature in units of T_v
//   [6]  = A_*         peak stellar fraction
//   [7]  = log10 M_*   mass of that peak
//   [8]  = sigma_*     width of the stellar peak in log10 M
//   [9]  = log10 T_w   temperature of the ejected gas in K
//   [10] = f_H         hydrogen mass fraction
//
// Electron-pressure window of the bound gas: the Fourier-weighted volume
// integral of the pressure (2005.00009 Eq. 4 with the pressure profile),
//
//   W_p(M, k) = int_0^{r_Delta} 4 pi r^2 [sin(kr)/(kr)] P_e(r) dr
//
// Derivation, chaining 2005.00009 Eqs. 40, 38, 39 and 13:
//
//   P_e = n_e k_B T_g,  n_e = rho_bnd/(m_p mu_e),  T_g = T_v theta
//     ->  W_p = [k_B T_v/(m_p mu_e)] f_bnd M u_KS
//   (3/2) k_B T_v = alpha G M m_p mu_p/(a r_v)
//     ->  W_p = (2 alpha/(3a)) (mu_p/mu_e) f_bnd (G M^2/r_v) u_KS
//
// The value comes back with G left out, i.e. in units of
// U = G (M_sun/h)^2/(c/H0): an energy (pressure times volume). The a
// turns the comoving r_v into the physical radius. At k -> 0,
// W_p ~ f_bnd M^(5/3) (2005.00009 Eq. 41): gas mass times a virial
// temperature ~ M/r_v ~ M^(2/3).
//
// Mean particle masses of a fully ionized hydrogen-helium gas with
// hydrogen mass fraction f_H (2005.00009, footnote to Eq. 40): per
// proton mass there are 2 f_H + 3(1 - f_H)/4 particles and
// f_H + (1 - f_H)/2 electrons, hence
//
//   mu_p = 4/(3 + 5 f_H),   mu_e = 2/(1 + f_H)
//
// r_v is r_Delta of this file (Delta = 200 times the mean density);
// 2005.00009 uses the virial radius (its Eq. 22), and the free alpha
// absorbs the difference, so its fitted alpha does not carry over.
//
// p_my and p_yy evaluate W_p as Y(a) B(M) u_KS(c, k, r_Delta), with
// Y = (2 alpha/(3a)) mu_p/mu_e and B = f_bnd M^2/r_Delta.
//
// Masses in M_sun/h.
// ============================================================================


// ---------------------------------------------------------------------------
// Komatsu-Seljak profile shape and the integrand of the u_KS contour
// integrals (u_KS header) at complex radius x = r/r_s:
//
//   theta(x) = ln(1 + x)/x,   g(x) = x theta(x)^p,   p = Gamma/(Gamma - 1).
//
// theta^p is the KS pressure profile. u_KS integrates g(x) e^{iyx}
// along the rays x = i tau and x = c + i tau (tau >= 0), where
// Re x >= 0: the branch cut of ln(1 + x), the real axis left of
// x = -1, is never approached.
// ---------------------------------------------------------------------------
static inline double complex ks_ctheta(
    const double complex x  // complex radius r/r_s, Re x > -1
  )
{
  /* PHYSICAL DERIVATION & LOGIC FLOW
     1. theta(x) = ln(1 + x)/x, a 0/0 at x = 0
     2. |x| < X_TAYLOR: the Taylor series of theta takes over (the
        dropped term, |x|^5/6, is ~1e-21 there)
     3. else ln1p = ln(1 + x), built from its two parts:
        Re ln(1 + x) = ln|1 + x| = (1/2) log1p(2 Re x + |x|^2)
        Im ln(1 + x) = arg(1 + x) = atan2(Im x, 1 + Re x)  in (-pi, pi] */

  // below this |x| the series replaces the 0/0 form ln(1 + x)/x
  const double X_TAYLOR = 1e-4;

  if (cabs(x) < X_TAYLOR) {
    return 1.0 - x/2.0 + x*x/3.0 - x*x*x/4.0 + x*x*x*x/5.0;
  }

  const double x_re = creal(x);
  const double x_im = cimag(x);

  const double complex ln1p = 0.5*log1p(2.0*x_re + x_re*x_re + x_im*x_im) +
                              I*atan2(x_im, 1.0 + x_re);
  return ln1p/x;
}


static inline double complex ks_cg(
    const double complex x, // complex radius r/r_s
    const double p          // Gamma/(Gamma - 1)
  )
{
  /* PHYSICAL DERIVATION & LOGIC FLOW
     1. g(x) = x theta(x)^p, the power taken with the principal log:
        |arg theta| < pi/2 on the contour rays, so g is one continuous
        analytic function there */
  return x*cexp(p*clog(ks_ctheta(x)));
}


// ---------------------------------------------------------------------------
// Natural cubic spline through the n_coarse coarse values y_coarse
// (uniform nodes, spacing h_coarse), evaluated on the dense grid that
// splits every interval into m: (n_coarse - 1) m + 1 nodes sharing both
// ends with the coarse grid. On interval j, for 0 <= t <= h_coarse,
//
//   S(x_j + t) = y_j + t (b + t (c_j + t d)),   c_j = S''(x_j)/2,
//   b = (y_{j+1} - y_j)/h_coarse - h_coarse (c_{j+1} + 2 c_j)/3,
//   d = (c_{j+1} - c_j)/(3 h_coarse),
//
// with c = 0 at both ends (natural). In the code c_coef[j] = c_j and
// y_dense[j m + r] = S(x_j + r h_coarse/m); r = 0 returns y_j exactly.
// u_KS upsamples its three 1D tables (ln P, ln F0, ln g) with it.
// ---------------------------------------------------------------------------
static void ks_upsample1d(
    const double* y_coarse, // coarse values
    const int n_coarse,     // coarse nodes
    const double h_coarse,  // coarse spacing
    double* c_coef,         // workspace [n_coarse]: spline c coefficients
    double* y_dense,        // output [(n_coarse - 1) m + 1]
    const int m             // refinement factor
  )
{
  /* PHYSICAL DERIVATION & LOGIC FLOW
     1. c_j = S''(x_j)/2 at the coarse nodes (natural: c = 0 at ends)
     2. interval j: b and d of the header, then
        S(x_j + t) = y_j + t (b + t (c_j + t d)) at t = r h_coarse/m */

  spline_coeffs_uniform(y_coarse, n_coarse, h_coarse, c_coef);

  for (int j=0; j<n_coarse-1; j++) {
    // linear (b) and cubic (d) coefficients of interval j
    const double b = (y_coarse[j+1] - y_coarse[j])/h_coarse -
                     h_coarse*(c_coef[j+1] + 2.0*c_coef[j])/3.0;
    const double d = (c_coef[j+1] - c_coef[j])/(3.0*h_coarse);

    // S at the m dense nodes t = r h_coarse/m of this interval
    for (int r=0; r<m; r++) {
      const double t = h_coarse*((double) r)/((double) m);
      y_dense[j*m + r] = y_coarse[j] + t*(b + t*(c_coef[j] + t*d));
    }
  }

  // last coarse node: no interval starts there
  y_dense[(n_coarse - 1)*m] = y_coarse[n_coarse - 1];
}


// ---------------------------------------------------------------------------
// Shape factor of the bound-gas pressure window:
//
//   u_KS(c, k, r_v) = F(c, y)/F0(c),   y = k r_v/c = k r_s,
//
//   F0(c)    = int_0^c x^2 theta(x)^q dx             (bound-gas mass)
//   F(c, y)  = int_0^c x sin(y x)/y theta(x)^p dx    (pressure transform)
//   theta(x) = ln(1 + x)/x,   p = Gamma/(Gamma - 1),   q = 1/(Gamma - 1),
//
// with x = r/r_s the radius in units of the NFW scale radius. theta^q
// is the Komatsu-Seljak ("KS") density profile of gas in hydrostatic
// equilibrium inside an NFW halo, theta^p its pressure profile and
// Gamma = nuisance.gas[0] its polytropic index (2005.00009 sec. 3.2,
// the rho_bnd equation). The full window W_p is
//
//   W_p(M, k) = [k_B T_v f_bnd M/(m_p mu_e)] u_KS.
//
// Unlike the matter u(k|M), u_KS does not tend to 1 at k -> 0:
// u_KS(c, 0) = <T_g>/T_v < 1, the mass-weighted gas temperature in units
// of the central one (theta(0) = 1), and |u_KS(c, k)| <= u_KS(c, 0).
//
// Two phases appear below: y = k r_s, the argument of the integral, and
// z = y c = k r_v, the phase at the outer edge x = c. The sin(y x) makes
// u ring like cos z under a slowly falling envelope out to z ~ 1e4, far
// too many zero crossings for a table of u itself. So u is tabulated
// directly only at small z (item 3); above, the oscillation is taken out
// of the integral analytically and put back exactly at lookup (items 1
// and 2).
//
// 1. The contour formula (z >= ZSW). With g(x) = x theta(x)^p,
//
//   F = Im J/y,    J(c, y) = int_0^c g(x) e^{i y x} dx.
//
// g is analytic in the upper half plane (the branch cut of ln(1 + x)
// runs along x < -1), so by Cauchy's theorem the path 0 -> c can be
// traded for the two vertical rays x = i tau and x = c + i tau, tau in
// [0, inf) (the edge closing them at Im x -> inf drops out, e^{iyx} -> 0
// there). On the rays e^{iyx} is the real, decaying e^{-y tau}: nothing
// oscillates, and the whole phase sits in the one factor e^{iz} of the
// second ray. With tau = c t on that ray and g(c) pulled out,
//
//   J = i I0(y) - e^{iz} (i g(c)/y) Q(c, z),
//
//   P(y)    = Re I0(y) = Re int_0^inf g(i tau) e^{-y tau} dtau,
//   Q(c, z) = z int_0^inf [g(c + i c t)/g(c)] e^{-z t} dt,
//
//   u = [P(y) - (g(c)/y) (cos z Re Q - sin z Im Q)]/(y F0(c)).
//
// P, Q, g and F0 are smooth and tabulated; cos z and sin z are exact at
// lookup. The weight z e^{-zt} of Q has unit integral and averages the
// ratio over t up to ~1/z, where the ratio is near 1, so Q -> 1 as
// z -> inf; above ZHI = 2.5e5, Q is held at its ZHI value.
//
// 2. Q and P by the trapezoid rule in s = ln t (t = e^s, dt = t ds, and
// tau = t/y in P):
//
//   Q(c, z) = z int [g(c + i c t)/g(c)] t e^{-z t} ds,
//   P(y)    = (1/y) int Re g(i t/y) t e^{-t} ds.
//
// Each integrand is one smooth bump (like e^s toward s -> -inf, like
// exp(-e^s) toward +inf), on which the trapezoid rule converges
// exponentially in the step h. The bump of P sits at the same s for
// every y; with the ln y spacing hy = h/rP, rP an integer, every
// t_k/y_j is a node of one grid in ln tau, so g(i tau) is evaluated
// once per node rather than once per (k, j) pair.
//
// 3. Small z (z < ZSW = 3): u tabulated directly on (ln c, w = z^2).
// With x = c s (the c^3 of both integrals cancels),
//
//   u(c, z) = int_0^1 s sin(z s)/z theta(c s)^p ds
//             / int_0^1 s^2 theta(c s)^q ds,
//
// two Gauss-Legendre integrals on [0, 1]. u is even in z, so it is a
// straight line in w near z = 0 (u0 - a w + ...). At the padding nodes
// w < 0, z = i kappa with kappa = sqrt(-w), and sin(z s)/z =
// sinh(kappa s)/kappa.
//
// 4. Tables. Each smooth ingredient is computed exactly on a coarse
// uniform grid, upsampled by a natural cubic spline onto a dense grid
// sharing its ends, and read from the dense grid by linear interpolation:
//
//   quantity        axes          coarse               dense
//   u (item 3)      ln c, w       u_coarse[i][j]       u_dense
//   Re Q, Im Q      ln c, ln z    Q_coarse[0|1][i][j]  Q_dense[0|1]
//   ln P            ln y          lnP_coarse[j]        lnP_dense
//   ln F0, ln g     ln c (1D)     lnF0g_coarse[0|1]    lnF0g_dense[0|1]
//
// Used ranges: ln c in [ln limits.halo_uks_cmin, ln limits.halo_uks_cmax]
// (a query outside is clamped to the edge), w in [0, ZSW^2], ln z in
// [ln ZSW, ln ZHI], ln y in [ln(ZSW/cmax), ln(ZHI/cmin)]. The coarse
// node counts of ln c and ln z are Ntable.halo_uks_nc and
// Ntable.halo_uks_nz (scaled by init_accuracy_boost); the others follow.
// Each used end gets PAD extra coarse nodes (a natural spline sets
// S'' = 0 at its ends; the lookups clamp to the used range, so the
// padding is never read), except the top of ln z, where Q is flat to
// O(1/ZHI). Dense counts are (coarse - 1) m + 1, m the refinement factor.
//
// Accuracy: u to 5e-6 of its local envelope, and to 2e-5 relative where
// |u| > 1e-2, at init_accuracy_boost = 1.
//
// Cache invalidation:
//   Ntable.random rebuilds the allocation, axes, nodes and weights;
//   nuisance.random_gas (Gamma) or Ntable.random refills the tables.
//
// Parameters:
//   c  - concentration r_Delta/r_s
//   k  - wavenumber in (c/H0)^-1
//   rv - halo radius r_Delta in c/H0 (comoving)
//
// Returns:
//   u_KS, dimensionless, with u_KS(c, 0) < 1
// ---------------------------------------------------------------------------
double u_KS(
    double c,        // concentration r_Delta/r_s
    double k,        // wavenumber in (c/H0)^-1
    const double rv  // halo radius r_Delta in c/H0 (comoving)
  )
{
  // --- 1. CONFIGURATION ---
  // ZSW, ZHI and PAD of the header; MC..M1 are the refinement factors m
  // of header item 4 for the five axes ln c, w, ln z, ln y, ln c (1D).
  const double ZSW = 3.0;    // z = k r_v below: table of u(ln c, z^2)
  const double ZHI = 2.5e5;  // top of the ln z axis of Q
  enum {
    PAD = 6,   // coarse padding nodes beyond used ends
    MC  = 12,  // dense refinement factors
    MW  = 32,
    MZ  = 16,
    MY  = 115,
    M1  = 70
  };

  // --- 2. STATIC STATE ---
  // Built on the first call (u_dense == NULL). Suffix "p" = padded
  // coarse count, "d" = dense count; lim[a] = {first node, last node,
  // spacing} of dense axis a (coarse spacing = lim[a][2] times m).
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static int ncp, nwp, nzp, nyp, n1p;  // padded coarse sizes
  static int ncd, nwd, nzd, nyd, n1d;  // dense sizes (shared endpoints)
  static int ngl;   // Gauss-Legendre nodes s_q on [0, 1]
  static int nt;    // trapezoid nodes t_k in s = ln t
  static int rP;    // hs/hy: trapezoid steps per ln y step
  static int ntau;  // nodes tau_m of the shared ln tau grid
  static double** u_dense = NULL;      // [ncd][nwd] u(ln c, w)
  static double*** Q_dense = NULL;     // [2][ncd][nzd] Re Q, Im Q
                                       // (ln c, ln z)
  static double* lnP_dense = NULL;     // [nyd] ln P(ln y)
  static double** lnF0g_dense = NULL;  // [2][n1d] ln F0, ln g (ln c)
  static double lim[5][3];             // dense axes (padded extents):
                                       // ln c, w, ln z, ln y, ln c (1D)
  static double** u_coarse = NULL;     // coarse exact values, same layouts
  static double*** Q_coarse = NULL;
  static double* lnP_coarse = NULL;
  static double** lnF0g_coarse = NULL;
  static double** spline_ws = NULL;    // [3][max(nyp, n1p)] workspaces
  static double** gl = NULL;           // [2][ngl] GL nodes s_q, weights w_q
  static double** sin_kern = NULL;     // [ngl][nwp] s_q sin(z_j s_q)/z_j
                                       // (sinh at w < 0)
  static double** Qwgt = NULL;         // [nt][nzp] z_j h t_k e^{-z_j t_k}
  static double** trap = NULL;         // [2][nt] t_k = e^{s_k},
                                       // h t_k e^{-t_k}
  static double** tau_g = NULL;        // [2][ntau] tau_m, Re g(i tau_m)

  // --- 3. NTABLE REBUILD ---
  // Sizes, dense axes, allocation, and the Gamma-independent quadrature
  // nodes and weights.
  if (NULL == u_dense || fdiff2(cache[1], Ntable.random)) {
    if (u_dense != NULL) {
      free(u_dense);
      free(Q_dense);
      free(lnP_dense);
      free(lnF0g_dense);
      free(u_coarse);
      free(Q_coarse);
      free(lnP_coarse);
      free(lnF0g_coarse);
      free(spline_ws);
      free(gl);
      free(sin_kern);
      free(trap);
      free(Qwgt);
      free(tau_g);
    }

    // Used ranges of ln c and ln y (header, item 4): y = z/c is smallest
    // at z = ZSW, c = cmax and largest at z = ZHI, c = cmin.
    const double lnc0 = log(limits.halo_uks_cmin);
    const double lnc1 = log(limits.halo_uks_cmax);
    const double lny0 = log(ZSW/limits.halo_uks_cmax);
    const double lny1 = log(ZHI/limits.halo_uks_cmin);

    // Coarse spacings hc (ln c) and hz (ln z) from the two knobs. hs is
    // the trapezoid step h of header item 2, halved for high-def
    // integration, and hy = hs/rP with rP an integer puts every t_k/y_j
    // on one ln tau grid. NW, NY and N1 are the coarse counts of w,
    // ln y and the 1D ln c axis.
    const int NC = Ntable.halo_uks_nc;
    const int NZ = Ntable.halo_uks_nz;
    const double hc = (lnc1 - lnc0)/((double) NC - 1.0);
    const double hz = (log(ZHI) - log(ZSW))/((double) NZ - 1.0);
    const int hdi = abs(Ntable.high_def_integration);  // accuracy knob

    double hs = 0.2;  // trapezoid step
    if (hdi >= 2) {
      hs = 0.1;
    }
    rP = (int) ceil(hs/hz);
    const double hy = hs/rP;

    const int NW = (int) ceil(6.0*NZ/64.0);  // w axis scales with ln z
    const int NY = (int) ceil((lny1 - lny0)/hy) + 1;
    const int N1 = (int) ceil(1.5*NC);       // denser 1D ln c axis
    const double hw = ZSW*ZSW/((double) NW - 1.0);
    const double h1 = (lnc1 - lnc0)/((double) N1 - 1.0);

    // Padded coarse counts (PAD nodes beyond each used end) and dense
    // counts sharing their endpoints: m dense steps per coarse interval.
    ncp = NC + 2*PAD;
    nwp = NW + 2*PAD;
    nzp = NZ + PAD;  // ln z is padded below only
    nyp = NY + 2*PAD;
    n1p = N1 + 2*PAD;
    ncd = (ncp - 1)*MC + 1;
    nwd = (nwp - 1)*MW + 1;
    nzd = (nzp - 1)*MZ + 1;
    nyd = (nyp - 1)*MY + 1;
    n1d = (n1p - 1)*M1 + 1;

    // Quadrature sizes from hdi: ngl Gauss-Legendre nodes for the [0, 1]
    // integrals of u and F0 (header, item 3); nt trapezoid nodes at the
    // step hs on the window [smin, smax] in s = ln t for Q and P (item
    // 2); ntau nodes of the shared ln tau grid of P.
    switch (hdi) {  // predefined GSL table sizes
      case 0:
        ngl = 96;
        break;
      case 1:
        ngl = 128;
        break;
      case 2:
        ngl = 256;
        break;
      case 3:
        ngl = 512;
        break;
      default:
        ngl = 1024;
        break;
    }

    double smin = -32.0;  // s = ln tau window
    if (hdi != 0) {
      smin = -40.0;
    }
    const double smax = 4.0;
    nt = (int) lround((smax - smin)/hs) + 1;
    ntau = (nt - 1)*rP + nyp;

    // Allocation; spline_ws holds one workspace per 1D upsampling job,
    // sized for the longer of the ln y and 1D ln c axes.
    int n_work = n1p;
    if (nyp > n1p) {
      n_work = nyp;
    }
    u_dense      = (double**) malloc2d(ncd, nwd);
    Q_dense      = (double***) malloc3d(2, ncd, nzd);
    lnP_dense    = (double*) malloc1d(nyd);
    lnF0g_dense  = (double**) malloc2d(2, n1d);
    u_coarse     = (double**) malloc2d(ncp, nwp);
    Q_coarse     = (double***) malloc3d(2, ncp, nzp);
    lnP_coarse   = (double*) malloc1d(nyp);
    lnF0g_coarse = (double**) malloc2d(2, n1p);
    spline_ws    = (double**) malloc2d(3, n_work);
    gl           = (double**) malloc2d(2, ngl);
    sin_kern     = (double**) malloc2d(ngl, nwp);
    trap         = (double**) malloc2d(2, nt);
    Qwgt         = (double**) malloc2d(nt, nzp);
    tau_g        = (double**) malloc2d(2, ntau);

    // Dense axes: first node = used start minus PAD coarse spacings,
    // spacing = coarse spacing/m. The w axis starts below w = 0
    // (header, item 3).
    lim[0][0] = lnc0 - PAD*hc;      // axis 0: ln c
    lim[0][2] = hc/MC;
    lim[1][0] = -PAD*hw;            // axis 1: w
    lim[1][2] = hw/MW;
    lim[2][0] = log(ZSW) - PAD*hz;  // axis 2: ln z
    lim[2][2] = hz/MZ;
    lim[3][0] = lny0 - PAD*hy;      // axis 3: ln y
    lim[3][2] = hy/MY;
    lim[4][0] = lnc0 - PAD*h1;      // axis 4: ln c (1D)
    lim[4][2] = h1/M1;

    // last node = first + (count - 1) spacings
    lim[0][1] = lim[0][0] + (ncd - 1)*lim[0][2];
    lim[1][1] = lim[1][0] + (nwd - 1)*lim[1][2];
    lim[2][1] = lim[2][0] + (nzd - 1)*lim[2][2];
    lim[3][1] = lim[3][0] + (nyd - 1)*lim[3][2];
    lim[4][1] = lim[4][0] + (n1d - 1)*lim[4][2];

    // Gauss-Legendre nodes gl[0][q] = s_q and weights gl[1][q] = w_q on
    // [0, 1] (header, item 3).
    gsl_integration_glfixed_table* gl_table = malloc_gslint_glfixed(ngl);
    for (int q=0; q<ngl; q++) {
      gsl_integration_glfixed_point(0.0, 1.0, q, &gl[0][q], &gl[1][q],
                                    gl_table);
    }
    gsl_integration_glfixed_table_free(gl_table);

    // sin_kern[q][j] = s_q sin(z_j s_q)/z_j, z_j = sqrt(w_j): the
    // z-dependent factor of the numerator integrand of header item 3 at
    // GL node q and w node w_j = (j - PAD) hw. At w < 0 it is
    // s sinh(kappa s)/kappa, kappa = sqrt(-w), and s^2 at w = 0: one
    // analytic function of w.
    for (int j=0; j<nwp; j++) {
      const double w = -PAD*hw + j*hw;
      for (int q=0; q<ngl; q++) {
        if (w > 0) {
          const double z = sqrt(w);
          sin_kern[q][j] = gl[0][q]*sin(z*gl[0][q])/z;
        }
        else if (w < 0) {
          const double kappa = sqrt(-w);
          sin_kern[q][j] = gl[0][q]*sinh(kappa*gl[0][q])/kappa;
        }
        else {
          sin_kern[q][j] = gl[0][q]*gl[0][q];
        }
      }
    }

    // Trapezoid nodes of header item 2: trap[0][k] = t_k = e^{s_k},
    // s_k = smin + k hs, and trap[1][k] = h t_k e^{-t_k}, the weight of
    // the P sum. End weights are not halved: the integrand vanishes at
    // both ends.
    for (int k=0; k<nt; k++) {
      trap[0][k] = exp(smin + k*hs);
      trap[1][k] = hs*trap[0][k]*exp(-trap[0][k]);
    }

    // Qwgt[k][j] = z_j h t_k e^{-z_j t_k}, the weight of the Q sum
    // (header, item 2; leading z included) at node j of the padded ln z
    // axis.
    for (int j=0; j<nzp; j++) {
      const double z = exp(lim[2][0] + j*hz);
      for (int k=0; k<nt; k++) {
        Qwgt[k][j] = z*hs*trap[0][k]*exp(-z*trap[0][k]);
      }
    }

    // Shared ln tau grid of P (header, item 2). With s_k = smin + k hs =
    // smin + k rP hy and ln y_j = lim[3][0] + j hy,
    //
    //   ln(t_k/y_j) = (smin - lim[3][0]) + (k rP - j) hy,
    //
    // so tau = t_k/y_j is node m = k rP - j + (nyp - 1) of the grid
    // tau_g[0][m] = tau_m (m = 0 at (k, j) = (0, nyp - 1), m = ntau - 1
    // at (nt - 1, 0)). The refill fills tau_g[1][m] = Re g(i tau_m).
    for (int m=0; m<ntau; m++) {
      tau_g[0][m] = exp(smin - lim[3][0] + (m - (nyp - 1))*hy);
    }
  }

  // --- 4. GAMMA REFILL ---
  // Everything Gamma enters, rebuilt from the nodes and weights above.
  if (fdiff2(cache[0], nuisance.random_gas) || fdiff2(cache[1], Ntable.random))
  {
    // The KS exponents p (pressure) and q (density) of the header; the
    // coarse spacings come back from the dense ones.
    const double p  = nuisance.gas[0]/(nuisance.gas[0] - 1.0);
    const double q  = 1.0/(nuisance.gas[0] - 1.0);
    const double hc = lim[0][2]*MC;
    const double hw = lim[1][2]*MW;
    const double hz = lim[2][2]*MZ;
    const double hy = lim[3][2]*MY;
    const double h1 = lim[4][2]*M1;

    /* PHYSICAL DERIVATION & LOGIC FLOW (one c node per iteration)
       1. GL pass at x = c s_q, theta = log1p(x)/x:
          f0      = sum_q w_q s_q^2 theta^q      (denominator, item 3)
          wthp[q] = w_q theta^p                  (numerator, z-free part)
       2. u(c_i, w_j) = sum_q wthp[q] sin_kern[q][j]/f0, all w at once
       3. g_re[k] + i g_im[k] = g(c + i c t_k)/g(c) at the trapezoid
          nodes, with g_at_c = g(c) = c theta(c)^p
       4. Q(c_i, z_j) = sum_k (g_re[k] + i g_im[k]) Qwgt[k][j] (item 2) */
    #pragma omp parallel for schedule(static)
    for (int i=0; i<ncp; i++) {
      const double c = exp(lim[0][0] + i*hc);

      // GL pass: mass norm f0 and the z-free pressure weights wthp[q]
      double wthp[ngl];
      double f0 = 0.0;
      for (int k=0; k<ngl; k++) {
        const double x = c*gl[0][k];
        const double theta = log1p(x)/x;
        f0 += gl[1][k]*gl[0][k]*gl[0][k]*pow(theta, q);
        wthp[k] = gl[1][k]*pow(theta, p);
      }

      // u(c_i, w_j) = sum_q wthp[q] sin_kern[q][j]/f0, all w nodes at
      // once (four per SIMDe step, then a scalar tail;
      // COSMO2D_NOT_USE_SIMD selects the plain loop)
      double* restrict u_row = u_coarse[i];
      for (int j=0; j<nwp; j++) {
        u_row[j] = 0.0;
      }
      for (int k=0; k<ngl; k++) {
        const double wthp_k = wthp[k];
        const double* restrict kern_k = sin_kern[k];
#ifdef COSMO2D_NOT_USE_SIMD
        for (int j=0; j<nwp; j++) {
          u_row[j] += wthp_k*kern_k[j];
        }
#else
        const v4d vt = simde_mm256_set1_pd(wthp_k);
        int j = 0;
        for (; j <= nwp - 4; j += 4) {
          const v4d prod =
              simde_mm256_mul_pd(vt, simde_mm256_loadu_pd(kern_k + j));
          simde_mm256_storeu_pd(u_row + j,
            simde_mm256_add_pd(simde_mm256_loadu_pd(u_row + j), prod));
        }
        for (; j < nwp; j++) {
          u_row[j] += wthp_k*kern_k[j];
        }
#endif
      }
      for (int j=0; j<nwp; j++) {
        u_row[j] /= f0;
      }

      // g ratio at the trapezoid nodes (derivation step 3)
      double g_re[nt];
      double g_im[nt];
      const double g_at_c = c*pow(log1p(c)/c, p);
      for (int k=0; k<nt; k++) {
        const double complex g_ratio = ks_cg(c + I*c*trap[0][k], p)/g_at_c;
        g_re[k] = creal(g_ratio);
        g_im[k] = cimag(g_ratio);
      }

      // Q(c_i, z_j) = sum_k (g_re[k] + i g_im[k]) Qwgt[k][j], all z
      // nodes at once; Re Q goes to Q_coarse[0][i], Im Q to
      // Q_coarse[1][i], as for u above
      double* restrict Qre_row = Q_coarse[0][i];
      double* restrict Qim_row = Q_coarse[1][i];
      for (int j=0; j<nzp; j++) {
        Qre_row[j] = 0.0;
        Qim_row[j] = 0.0;
      }
      for (int k=0; k<nt; k++) {
        const double g_re_k = g_re[k];
        const double g_im_k = g_im[k];
        const double* restrict Qwgt_k = Qwgt[k];
#ifdef COSMO2D_NOT_USE_SIMD
        for (int j=0; j<nzp; j++) {
          Qre_row[j] += Qwgt_k[j]*g_re_k;
          Qim_row[j] += Qwgt_k[j]*g_im_k;
        }
#else
        const v4d var = simde_mm256_set1_pd(g_re_k);
        const v4d vai = simde_mm256_set1_pd(g_im_k);
        int j = 0;
        for (; j <= nzp - 4; j += 4) {
          const v4d vw = simde_mm256_loadu_pd(Qwgt_k + j);
          simde_mm256_storeu_pd(Qre_row + j, simde_mm256_add_pd(
            simde_mm256_loadu_pd(Qre_row + j), simde_mm256_mul_pd(vw, var)));
          simde_mm256_storeu_pd(Qim_row + j, simde_mm256_add_pd(
            simde_mm256_loadu_pd(Qim_row + j), simde_mm256_mul_pd(vw, vai)));
        }
        for (; j < nzp; j++) {
          Qre_row[j] += Qwgt_k[j]*g_re_k;
          Qim_row[j] += Qwgt_k[j]*g_im_k;
        }
#endif
      }
    }

    /* PHYSICAL DERIVATION & LOGIC FLOW (header, item 2)
       1. tau_g[1][m] = Re g(i tau_m), once per shared ln tau node
       2. P(y_j) = (1/y_j) sum_k trap[1][k] tau_g[1][k rP - j + nyp - 1]
          (g_row = tau_g[1] + nyp - 1 - j, read at g_row[k rP])
       3. stored as ln P: P -> p/y^3 at large y, so ln P is close to a
          straight line in ln y */
    #pragma omp parallel for schedule(static)
    for (int m=0; m<ntau; m++) {
      tau_g[1][m] = creal(ks_cg(I*tau_g[0][m], p));
    }
    #pragma omp parallel for schedule(static)
    for (int j=0; j<nyp; j++) {
      const double y = exp(lim[3][0] + j*hy);
      const double* restrict g_row = tau_g[1] + (nyp - 1 - j);
      const double* restrict wgt = trap[1];
      double sum = 0.0;
      for (int k=0; k<nt; k++) {
        sum += wgt[k]*g_row[k*rP];
      }
      lnP_coarse[j] = log(sum/y);
    }

    /* PHYSICAL DERIVATION & LOGIC FLOW (1D ln c axis)
       1. F0 = c^3 sum_q w_q s_q^2 theta(c s_q)^q  (the c^3 from x = c s)
       2. ln g = ln c + p ln theta(c) */
    for (int j=0; j<n1p; j++) {
      const double c = exp(lim[4][0] + j*h1);
      double f0 = 0.0;
      for (int k=0; k<ngl; k++) {
        const double x = c*gl[0][k];
        f0 += gl[1][k]*gl[0][k]*gl[0][k]*pow(log1p(x)/x, q);
      }
      lnF0g_coarse[0][j] = log(c*c*c*f0);
      lnF0g_coarse[1][j] = log(c) + p*log(log1p(c)/c);
    }

    // Upsampling (header, item 4): six independent jobs, each a natural
    // cubic spline from a padded coarse grid to the dense grid sharing
    // its ends. spline2d_upsample_uniform (basics.c) splines along each
    // axis in turn; ks_upsample1d is the 1D form.
    #pragma omp parallel for schedule(dynamic, 1)
    for (int job=0; job<6; job++) {
      switch (job) {
        case 0:
          spline2d_upsample_uniform(Q_coarse[0], ncp, nzp, hc, hz,
                                    Q_dense[0], ncd, nzd);
          break;
        case 1:
          spline2d_upsample_uniform(Q_coarse[1], ncp, nzp, hc, hz,
                                    Q_dense[1], ncd, nzd);
          break;
        case 2:
          spline2d_upsample_uniform(u_coarse, ncp, nwp, hc, hw,
                                    u_dense, ncd, nwd);
          break;
        case 3:
          ks_upsample1d(lnP_coarse, nyp, hy, spline_ws[0], lnP_dense, MY);
          break;
        case 4:
          ks_upsample1d(lnF0g_coarse[0], n1p, h1, spline_ws[1],
                        lnF0g_dense[0], M1);
          break;
        default:
          ks_upsample1d(lnF0g_coarse[1], n1p, h1, spline_ws[2],
                        lnF0g_dense[1], M1);
          break;
      }
    }

    cache[0] = nuisance.random_gas;
    cache[1] = Ntable.random;
  }

  // --- 5. LOOKUP ---
  // c is clamped to the tabulated range; z = k r_v is formed from the
  // arguments as given, so a clamped query returns u at the edge
  // concentration and the true phase.
  const double c_clamped = fmin(fmax(c, limits.halo_uks_cmin),
                                limits.halo_uks_cmax);
  const double lnc = log(c_clamped);
  const double z   = k*rv;  // the phase z = k r_v

  // z < ZSW: one bilinear read of u(ln c, w) at w = z^2 (header, item 3)
  if (z < ZSW) {
    return interpol2d(u_dense, ncd, lim[0][0], lim[0][1], lim[0][2], lnc,
                      nwd, lim[1][0], lim[1][1], lim[1][2], z*z);
  }

  // z >= ZSW: the contour formula of header item 1 at y = z/c. lnz
  // clamps z to ZHI (Q held at its ZHI value); lny clamps ln y to its
  // axis, which acts only above ZHI, where the P term (order 1/y^4) is
  // negligible against the Q term (order 1/y^2). P, g and F0 come back
  // from their logs.
  const double y   = z/c_clamped;
  const double lnz = log(fmin(z, ZHI));
  const double lny = fmin(fmax(log(y), log(ZSW/limits.halo_uks_cmax)),
                          log(ZHI/limits.halo_uks_cmin));

  const double P = exp(interpol1d(lnP_dense, nyd, lim[3][0], lim[3][1],
                                  lim[3][2], lny));
  const double Q_re = interpol2d(Q_dense[0], ncd, lim[0][0], lim[0][1],
                                 lim[0][2], lnc, nzd, lim[2][0], lim[2][1],
                                 lim[2][2], lnz);
  const double Q_im = interpol2d(Q_dense[1], ncd, lim[0][0], lim[0][1],
                                 lim[0][2], lnc, nzd, lim[2][0], lim[2][1],
                                 lim[2][2], lnz);
  const double g  = exp(interpol1d(lnF0g_dense[1], n1d, lim[4][0],
                                   lim[4][1], lim[4][2], lnc));
  const double F0 = exp(interpol1d(lnF0g_dense[0], n1d, lim[4][0],
                                   lim[4][1], lim[4][2], lnc));

  // u = [P - (g/y)(cos z Re Q - sin z Im Q)]/(y F0), header item 1: the
  // oscillation enters only through the exact cos z and sin z
  return (P - g/y*(cos(z)*Q_re - sin(z)*Q_im))/(y*F0);
}


// ---------------------------------------------------------------------------
// Fraction of the halo mass in bound gas (2005.00009 Eq. 25, from
// 1510.06034 Eq. 2.19):
//
//   f_bnd(M) = (Omega_b/Omega_m) / [1 + (M_0/M)^beta]
//
// Massive halos keep their cosmic share of baryons as hot bound gas
// (M >> M_0: f_bnd -> Omega_b/Omega_m); feedback empties light halos
// (M << M_0: f_bnd ~ (Omega_b/Omega_m)(M/M_0)^beta -> 0). A halo of
// mass M_0 keeps half; beta sets how sharp the transition is (HMx
// defaults, 2005.00009 sec. 3.2: M_0 = 1e14 M_sun, beta = 0.6).
//
// Parameters:
//   M - halo mass in M_sun/h (M_0 = 10^nuisance.gas[2] M_sun/h,
//       beta = nuisance.gas[1])
//
// Returns:
//   f_bnd in [0, Omega_b/Omega_m]
// ---------------------------------------------------------------------------
double frac_bnd(
    double M  // halo mass in M_sun/h
  )
{
  const double M0   = pow(10.0, nuisance.gas[2]);  // half-bound mass
  const double beta = nuisance.gas[1];             // mass slope

  // f_bnd = (Omega_b/Omega_m)/[1 + (M_0/M)^beta]
  const double suppression = pow(M0/M, beta);
  return cosmology.Omega_b/(cosmology.Omega_m*(1.0 + suppression));
}


// ---------------------------------------------------------------------------
// Fraction of the halo mass in ejected gas (2005.00009 Eq. 26):
//
//   f_ejc(M) = Omega_b/Omega_m - f_bnd(M) - f_*(M)
//
// The baryons of the halo's initial overdensity that are neither bound
// gas nor stars have been pushed beyond r_Delta by feedback. The
// stellar fraction (2005.00009 Eq. 27, from 1401.2997 sec. 3.3)
//
//   f_*(M) = A_* exp[-log10^2(M/M_*)/(2 sigma_*^2)]
//
// peaks at M_* with height A_* and width sigma_* in dex. Above M_* it
// is floored at A_*/3, the high-mass saturation of the stellar-to-halo
// mass relation (2005.00009 sec. 3.2):
//
//   f_*(M > M_*) = max(f_*(M), A_*/3)
//
// Clip: f_ejc is floored at 0 where f_bnd + f_* would exceed
// Omega_b/Omega_m (the heaviest halos). 2005.00009 (footnote in
// sec. 3.2) takes the excess out of the stars instead; the gas to eject
// is zero either way, and f_* is used nowhere else.
//
// Parameters:
//   M - halo mass in M_sun/h (A_* = nuisance.gas[6],
//       log10 M_* = nuisance.gas[7], sigma_* = nuisance.gas[8])
//
// Returns:
//   f_ejc in [0, Omega_b/Omega_m]
// ---------------------------------------------------------------------------
double frac_ejc(
    double M  // halo mass in M_sun/h
  )
{
  // stellar fraction: Gaussian peak in log10 M, delta in units of
  // sigma_*
  const double log10M = log10(M);
  const double delta  = (log10M - nuisance.gas[7])/nuisance.gas[8];
  const double f_star_gauss = nuisance.gas[6]*exp(-0.5*delta*delta);

  // above M_*, f_* saturates at the floor A_*/3 (header)
  const double f_star_floor = nuisance.gas[6]/3.0;
  double frac_star = f_star_gauss;
  if ((log10M > nuisance.gas[7]) && (f_star_gauss < f_star_floor)) {
    frac_star = f_star_floor;
  }

  // clip at 0: no gas to eject once f_bnd + f_* fill the baryon budget
  return fmax(0.0,
              cosmology.Omega_b/cosmology.Omega_m - frac_bnd(M) - frac_star);
}


// ---------------------------------------------------------------------------
// Electron-pressure window of the ejected gas: its electron count times
// k_B T_w, at the warm temperature T_w,
//
//   W_ejc(M) = N_e k_B T_w,   N_e = f_ejc M/(mu_e m_p)
//
// The ejected gas follows the linear density field outside halos, so it
// has no 1-halo term and a k-independent (point-like) window in the
// 2-halo term (2005.00009 sec. 3.3 and Eq. 36).
//
// Unit chain, landing on the units of W_p so the two windows add:
//
//   num_p = M_sun/m_p = 1.1892e57           (M_sun = 1.989e30 kg)
//   num_p m[M_sun/h]  = h N_p
//   k_B T_w [eV]      = 8.6173e-5 T_w[K]
//   1 eV              = 5.616e-44 h U,   U = G (M_sun/h)^2/(c/H0)
//     ->  E_w = 8.6173e-5 T_w 5.616e-44 = (k_B T_w in U)/h
//     ->  num_p m f_ejc E_w/mu_e = N_e k_B T_w in U   (h cancels)
//
// W_ejc >= 0, since frac_ejc clips f_ejc at 0.
//
// Parameters:
//   m - halo mass in M_sun/h (T_w = 10^nuisance.gas[9] K,
//       f_H = nuisance.gas[10])
//
// Returns:
//   W_ejc(M) in U = G (M_sun/h)^2/(c/H0)
// ---------------------------------------------------------------------------
double u_y_ejc(
    double m  // halo mass in M_sun/h
  )
{
  // the unit chain of the header, one named factor per step
  const double num_p    = 1.1892e57;  // M_sun/m_p: protons per M_sun
  const double kB_eV_K  = 8.6173e-5;  // k_B in eV per K
  const double eV_to_hU = 5.616e-44;  // 1 eV in h U

  // k_B T_w: K -> eV -> U
  const double E_w = pow(10, nuisance.gas[9])*kB_eV_K*eV_to_hU;

  // mu_e = 2/(1 + f_H): proton masses per electron of the ionized gas
  const double mu_e = 2./(1. + nuisance.gas[10]);

  // W_ejc = N_e k_B T_w
  return (num_p * m * frac_ejc(m) / mu_e) * E_w;
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
                                   //   [3] clustering n(z) tags
  int nbin;                 // lens bins of the allocation
  int n_nodes;              // Gauss-Legendre mass nodes per lens bin
  double lim[3];            // a grid: min, max, step
  double*** tab;            // [2][nbin][N_a] ngal(a_j) (0), bgal(a_j) (1)
  double*** node_data;      // [2][nbin][n_nodes] per mass node q:
                            //   nu0_q (0), P_q (1), see hod_tables header
  double** gauss_legendre;  // [2][n_nodes] Gauss-Legendre nodes x_q (0)
                            //   and weights w_q (1) on [-1, 1]
  double* growth;           // [N_a] growth factor D(a_j)
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
//   dn/dlnM = (rho_m/M) nu f(nu) dln nu/dln M    halo mass function
//   nu      = delta_c/(sigma(M) D(a))            peak height
//   <N|M>   = f_c N_c(M) + N_s(M)                HOD occupation
//   b(nu)                                        Tinker halo bias
//
// f is the Tinker multiplicity function (fnu), b the Tinker bias (hb1nu),
// <N|M> the occupation of the GALAXY PROFILES banner (HOD_fc, HOD_nc,
// HOD_ns); the POWER SPECTRA banner explains dn/dlnM.
//
// Quadrature: Gauss-Legendre in ln M with n_nodes = 128, 256, 512 or
// 1024 nodes for abs(Ntable.high_def_integration) = 0, 1, 2, >= 3, over
// [ln 10^(lg M_min - 2), ln limits.halo_m_max] of each bin (N_c is an
// erf tail below). Nodes x_q and weights w_q on [-1, 1] map to
// ln M_q = mid + half_width x_q with weight half_width w_q.
//
// Only nu, f and b depend on a (through D(a) and the Tinker
// parameters), so the integrand splits into an a-free factor per mass
// node and a sum per a node. Equations to the arrays of hod_:
//
//   gauss_legendre[0][q], [1][q]  x_q, w_q
//   node_data[0][b][q]    nu0_q = delta_c/sigma(M_q), nu at D = 1
//   node_data[1][b][q]    P_q = half_width w_q (rho_m/M_q)
//                           (dln nu/dln M) <N|M_q>
//   growth[j], mult_pars[j], bias_pars[j]
//                         D(a_j), Tinker parameters of f and b at a_j
//   tab[0][b][j]          ngal(a_j) = sum_q P_q nu f(nu),  nu = nu0_q/D_j
//   tab[1][b][j]          bgal(a_j) = sum_q P_q nu f(nu) b(nu) / ngal(a_j)
//
// so that P_q nu f(nu) is half_width w_q (dn/dlnM) <N|M> at mass node q.
//
// The a grid: Ntable.N_a nodes uniform in a over [1/(1 + z_max),
// 1/(1 + z_min)] of the clustering n(z), all bins; ngal and bgal return
// 0 outside it and interpolate linearly inside, accurate to about 1e-5
// (the quadrature itself is converged below 1e-6).
//
// Aborts: HOD_nc, on a lens bin whose HOD is not set; the tables cover
// all bins at once.
//
// Cache invalidation:
//   rebuild (sizes, GL nodes, a grid, every allocation): Ntable.random
//     or redshift.random_clustering
//   refill: those two, cosmology.random or nuisance.random_galaxy_bias
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
      free(hod_.growth);
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
    hod_.growth         = (double*) malloc1d(Ntable.N_a);
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

    // The a grid: min, max, step.
    hod_.lim[0] = 1.0/(redshift.clustering_zdist_zmax_all + 1.0);
    hod_.lim[1] = 1.0/(redshift.clustering_zdist_zmin_all + 1.0);
    hod_.lim[2] = (hod_.lim[1] - hod_.lim[0])/((double) Ntable.N_a - 1.0);
  }

  // Refill: cosmology, Ntable, the HOD or the clustering n(z) changed.
  if (fdiff2(hod_.cache[0], cosmology.random) ||
      fdiff2(hod_.cache[1], Ntable.random) ||
      fdiff2(hod_.cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(hod_.cache[3], redshift.random_clustering))
  {
    const int nbin    = hod_.nbin;
    const int n_a     = Ntable.N_a;
    const int n_nodes = hod_.n_nodes;

    const double rho_m = cosmology.rho_crit * cosmology.Omega_m;

    // --- 2. PER-REFILL PRECOMPUTATION ---
    // nu0_q = delta_c/sigma(M_q) and P_q = half_width w_q (rho_m/M_q)
    // (dln nu/dln M) <N|M_q> at the mass nodes
    // ln M_q = mid + half_width x_q of each bin: the a-free part of the
    // integrand. The HOD does not evolve; hod_.lim[0] is an a inside
    // the grid for the range check of HOD_nc. Serial: sigma2 and
    // dlognudlogm build their tables here.
    for (int b=0; b<nbin; b++) {
      // lower bound 2 dex below the HOD lg M_min (N_c is an erf tail)
      const double lnMmin = log(10.0)*(nuisance.hod[b][0] - 2.);
      const double lnMmax = log(limits.halo_m_max);

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

        hod_.node_data[0][b][q] = delta_c/sqrt(sigma2(m)); // nu0_q
        hod_.node_data[1][b][q] = half_width*hod_.gauss_legendre[1][q]
                                  *(rho_m/m)*dlognudlogm(m)*occupation; // P_q
      }
    }

    // D(a_j) and the Tinker parameters of f and b at each a node.
    // Serial: growfac and fnu_params_at build their tables here.
    for (int j=0; j<n_a; j++) {
      const double a = hod_.lim[0] + j*hod_.lim[2];
      hod_.growth[j]    = growfac(a);
      hod_.bias_pars[j] = hb1nu_params_at(a);
      hod_.mult_pars[j] = fnu_params_at(a);
    }

    // --- 3. TABLE FILL: THE (BIN, a) OPENMP LOOP ---
    /* PHYSICAL DERIVATION & LOGIC FLOW
       1. nu = nu0_q/D(a_j)                    peak height at mass node q
       2. P_q nu f(nu) = half_width w_q (dn/dlnM) <N|M_q>
       3. ngal(a_j) = sum_q P_q nu f(nu)
                    = int dlnM (dn/dlnM) <N|M>
       4. bgal(a_j) = sum_q P_q nu f(nu) b(nu) / ngal(a_j)
                    = (1/ngal) int dlnM (dn/dlnM) b(nu) <N|M>            */
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b=0; b<nbin; b++) {
      for (int j=0; j<n_a; j++) {
        const double* restrict nu0 = hod_.node_data[0][b]; // nu0_q
        const double* restrict Pq  = hod_.node_data[1][b]; // P_q

        const double D               = hod_.growth[j];
        const hb1nu_params* tinker_b = &hod_.bias_pars[j];
        const fnu_params* tinker_f   = &hod_.mult_pars[j];

        double n_gal  = 0.0; // sum_q P_q nu f(nu)       -> ngal(a_j)
        double bn_gal = 0.0; // sum_q P_q nu f(nu) b(nu) -> bgal x ngal

        for (int q=0; q<n_nodes; q++) {
          const double nu = nu0[q]/D; // nu0_q/D(a_j)
          // dn_gal = half_width w_q (dn/dlnM) <N|M>: node q's share
          const double dn_gal = Pq[q]*fnu_core(nu, tinker_f)*nu;
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
// a grid; 0 outside [1/(1 + z_max), 1/(1 + z_min)] of the clustering
// n(z).
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
// Tinker bias b(nu) of the host halos, weighted by how many galaxies
// each halo mass contributes,
//
//   bgal(a) = int dlnM (dn/dlnM) [f_c N_c(M) + N_s(M)] b(nu) / ngal(a),
//
// tabulated by hod_tables (its header) and interpolated linearly on its
// a grid; 0 outside [1/(1 + z_max), 1/(1 + z_min)] of the clustering
// n(z). The large-scale galaxy bias of p_gm and p_gg.
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
//   dn/dlnM = (rho_m/M) nu f(nu) dlnnu/dlnM,  nu = delta_c/(sigma(M) D(a))
//
// the mass function (fnu, dlognudlogm; sigma2 in cosmo3D.c), b the Tinker
// bias (hb1nu) and the windows W_X the Fourier transforms of the profiles,
// in the units of the field times a volume:
//
//   matter    W_m = (M/rho_m) u(k|M),  u -> 1 at k -> 0   (c/H0)^3
//   pressure  W_y = W_p (1-halo); + u_y_ejc in I11_y       U (energy)
//
// One M/rho_m per matter leg, none on the pressure: W_p is already the
// volume integral of the pressure (~ M^(5/3) at k -> 0, 2005.00009
// Eq. 41); with an extra M/rho_m a halo's pressure would scale as M^(8/3).
// The ejected gas follows the linear field outside halos, so it enters
// the 2-halo term only, as a k-independent window (2005.00009 sec. 3.3).
//
// A(a) = 1 - bias_norm(a) is the HMx correction (2005.00009 App. A) for
// the halos below M_min, which hold about 20% of the bias-weighted matter
// at z = 0 (bias_norm header, item 1): their share is put back as halos
// of mass exactly M_min, n(M) -> n(M) + A delta_D(M - M_min)/[b(M_min)
// M_min/rho_m] (Eq. A7), so that I11_m -> 1 and P_2h -> P_lin at k -> 0
// at every a, while at high k the added halos stay point-like as the
// real light halos are (r_Delta = 2.4 kpc/h at 1e6 M_sun/h).
//
// The y spectra damp their 1-halo term at low k, I02 -> I02 x/(1 + x),
// x = (k/k_s)^4, k_s = 0.05618 (sigma8 a)^-1.013 h/Mpc (2009.01858
// Eq. 17 and Table 2); the galaxy spectra take Pdelta b_gal as their
// 2-halo term instead of I11 (p_gm, p_gg headers).
//
// Each builder tabulates ln P on a uniform (a, ln k) grid with the
// 1024-node Gauss-Legendre rule in ln M and reads it bilinearly; the
// first call of a refill is halo_warmup.


// ---------------------------------------------------------------------------
// Warm-up of the spectrum table builders: one single-threaded call to
// each function with lazily built static state that the threaded loops
// of p_mm, p_my, p_yy, p_gm and p_gg read, so that inside the loops
// those functions only read (the warm-up rule of the cosmo2D.c _work
// functions). Each call is one table read once its build has run; the
// values are thrown away, which the (void) casts say. The builds
// themselves thread (sigma2, dlognudlogm, bias_norm, hod_tables,
// tinker_alpha, nfw_table and u_KS each own a parallel loop): one more
// reason they must start outside a parallel region.
//
//   sigma2, dlognudlogm   the ln M tables (cosmo3D.c; above); keys
//                         cosmology.random, Ntable.random
//   fnu_params_at         the tinker_alpha table; key like.halo_model
//   nfw_table             the NFW f, G table nfw_, read by the rows
//                         through nfw_um; key Ntable.random
//   bias_norm             the HMx term of the I11 2-halo spectra
//                         (matter, y); keys cosmology, Ntable
//   ngal                  the hod_ tables of ngal and bgal (galaxies);
//                         keys cosmology, Ntable, the HOD tag, the
//                         clustering n(z) tag
//   Pdelta                its run-mode latch, a static set on the
//                         first call (galaxies: the 2-halo term)
//   u_KS                  the gas tables (y); keys nuisance.random_gas,
//                         Ntable.random
//
// Static-free, so absent: growfac, p_lin, p_nonlin, PkRatio_baryons
// (the CAMB-fed cosmology tables); hb1nu_params_at and the *_core
// kernels; conc (a sigma2 read); HOD_nc, HOD_ns, HOD_fc; frac_bnd,
// frac_ejc, u_y_ejc; nfw_um.
//
// Preconditions, checked by the callees: 0 < a < 1 (fnu_params_at);
// hod = 1 needs the HOD of every lens bin set (HOD_nc aborts inside
// hod_tables); gas = 1 needs cosmology.Omega_b > 0 and
// nuisance.gas[0] > 1, which the y builders check in their refill
// blocks before calling here.
//
// Parameters:
//   a   - a scale factor of the builder's a grid, 0 < a < 1
//   k   - a wavenumber of the builder's k grid, (c/H0)^-1
//   gas - 1: the y spectra (p_my, p_yy) read u_KS
//   hod - 1: the galaxy spectra (p_gm, p_gg) read ngal, bgal and
//         Pdelta; 0: the I11 spectra (p_mm, p_my, p_yy) read bias_norm
//
// Returns:
//   nothing
// ---------------------------------------------------------------------------
static void halo_warmup(
    const double a,  // scale factor of the builder's a grid, 0 < a < 1
    const double k,  // wavenumber of the builder's k grid, (c/H0)^-1
    const int gas,   // 1 = the y spectra read u_KS
    const int hod    // 1 = the galaxy spectra read ngal, bgal, Pdelta
  )
{
  const double mmin = limits.halo_m_min;

  (void) sigma2(mmin);
  (void) dlognudlogm(mmin);
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
    // one read at the lightest halo: c(M_min) is a sigma2 read (built
    // above); r_v = r_Delta(M_min) as in W_p (GAS PROFILES banner),
    // from M_min = (4 pi/3) rho_Delta r_v^3
    const double rho_delta = Delta*cosmology.rho_crit*cosmology.Omega_m;
    const double rv = pow(3./(4.0*M_PI)*(mmin/rho_delta), 1./3.);
    (void) u_KS(conc(mmin, growfac(a)), k, rv);
  }
}


// ---------------------------------------------------------------------------
// P_mm(k, a), the halo-model matter power spectrum, from a table of ln P
// on Ntable.N_a x Ntable.N_k_nlin nodes uniform in (a, ln k), read
// bilinearly (interpol2d) and exponentiated:
//
//   P_mm = I02 + I11^2 P_lin                          (2005.00009 Eqs. 1-2)
//   I02  = int dlnM dn/dlnM (M/rho_m)^2 u(k|M)^2
//   I11  = int dlnM dn/dlnM b(nu) (M/rho_m) u(k|M) + A(a) u(k|M_min)
//
// dn/dlnM = (rho_m/M) nu f(nu) dlnnu/dlnM is the mass function, (M/rho_m) u
// the matter window, b the Tinker bias, A(a) = 1 - bias_norm(a) the HMx
// share of matter below M_min put back as halos of mass M_min (section
// banner).
//
// 1. Quadrature: the n-point Gauss-Legendre rule in ln M (exact for
// polynomials of degree 2n - 1), n = 1024, the largest size GSL
// tabulates; nodes M_q and weights w_q on [ln M_min, ln M_max] are mapped
// once in the rebuild block (gsl_integration_glfixed_point). High k
// converges slowest: the profile's ringing is sampled in ln M.
//
// 2. Loop levels: each factor is computed at the outermost level it
// depends on, so the innermost loop is the NFW kernel alone (nfw_um:
// three table reads and two sines; no pow, exp or log):
//
//   per refill, per node q       M_q, w_q; nu0_q = delta_c/sigma(M_q);
//     (mass_node)                w_q (rho_m/M_q) dlnnu/dlnM; M_q/rho_m;
//                                r_Delta(M_q)
//   per a row i, threaded        D(a); Tinker f, b parameters; A(a); c(M_min)
//   per (i, q) (a_node[i])       nu = nu0/D; c = conc(M, D); ln(1+c);
//                                m(c) = ln(1+c) - c/(1+c); r_s = r_Delta/c,
//                                ln r_s; w1h_q = dn (M/rho_m)^2/m(c)^2;
//                                w2h_q = dn b(nu) (M/rho_m)/m(c), with
//                                dn = w (rho_m/M) dlnnu/dlnM f(nu) nu
//   per (i, k), sum over q       x = k r_s, ln x = ln k + ln r_s,
//                                um = u m(c) = nfw_um(c, x, ln x, ln(1+c));
//                                I02 = sum w1h_q um^2,
//                                I11 = sum w2h_q um + A u_c(k|M_min); ln P
//
// The 1/m(c) of u = um/m(c) lives in w1h_q and w2h_q. The rows read the
// NFW kernel directly, so like.halo_model[3] must be HALO_PROFILE_NFW (the
// only option of u_c); anything else aborts.
//
// Thread safety: the single-threaded halo_warmup call before the
// threaded loop builds every lazy table the rows read (its header).
//
// Cache invalidation:
//   rebuild block (table, mass_node, a_node, GL nodes, both grids; every
//     allocation lives here, one block each from malloc2d/malloc3d):
//     Ntable.random
//   refill: cosmology.random or Ntable.random
//
// Parameters:
//   k - wavenumber in (c/H0)^-1
//   a - scale factor
//
// Returns:
//   P_mm(k, a) in (c/H0)^3; 0 outside [limits.a_min, 0.9999999]; ln P
//   continued with unit slope outside [ln k_min, ln k_max] (interpol2d)
// ---------------------------------------------------------------------------
double p_mm(
    const double k,
    const double a
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];  // tags the table was built from
  static double** table = NULL;            // ln P on the (a, ln k) grid
  static double lim[2][3];           // [0] a grid: min, max, step;
                                     // [1] ln k grid: min, max, step
  static int n_nodes = 0;            // Gauss-Legendre nodes in ln M
  static double** mass_node = NULL;  // [6][n_nodes] per mass node q:
                                     // 0 M, 1 GL weight w, 2 nu0 = nu(D=1),
                                     // 3 w (rho_m/M) dlnnu/dlnM,
                                     // 4 M/rho_m, 5 r_Delta
  static double*** a_node = NULL;    // [N_a][6][n_nodes] per (a row, node):
                                     // 0 c, 1 ln(1+c), 2 r_s, 3 ln r_s,
                                     // 4 w1h (1-halo), 5 w2h (2-halo)

  // --- 1. NTABLE REBUILD: TABLE, NODE ARRAYS, GL RULE, GRIDS ---
  // the table, the per-node and per-(a, node) arrays (one block each from
  // malloc2d/malloc3d, so one free each), the GL nodes mapped onto
  // [ln M_min, ln M_max], both grids (header, item 1)
  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) {
      free(table);
      free(mass_node);
      free(a_node);
    }

    table     = (double**) malloc2d(Ntable.N_a, Ntable.N_k_nlin);
    n_nodes   = 1024;  // the largest rule GSL tabulates (header, item 1)
    mass_node = (double**) malloc2d(6, n_nodes);
    a_node    = (double***) malloc3d(Ntable.N_a, 6, n_nodes);

    // gsl_integration_glfixed_point(lo, hi, q, &x, &w, t): node q of the
    // rule t mapped onto [lo, hi], and its weight
    const double lnMmin = log(limits.halo_m_min);
    const double lnMmax = log(limits.halo_m_max);
    gsl_integration_glfixed_table* gl_table = malloc_gslint_glfixed(n_nodes);
    for (int q=0; q<n_nodes; q++) {
      double lnM;
      gsl_integration_glfixed_point(lnMmin, lnMmax, q, &lnM,
                                    &mass_node[1][q], gl_table);
      mass_node[0][q] = exp(lnM);
    }
    gsl_integration_glfixed_table_free(gl_table);

    // the uniform table axes: [0] a, [1] ln k (min, max, step)
    lim[0][0] = limits.a_min;
    lim[0][1] = 0.9999999;  // a_max, just below a = 1 (today)
    lim[0][2] = (lim[0][1] - lim[0][0]) / ((double) Ntable.N_a - 1.0);
    lim[1][0] = log(limits.k_min_cH0);
    lim[1][1] = log(limits.k_max_cH0);
    lim[1][2] = (lim[1][1] - lim[1][0]) / ((double) Ntable.N_k_nlin - 1.0);
  }

  // --- 2. REFILL GUARD AND SINGLE-THREADED WARM-UP ---
  // refill when the cosmology or Ntable tag differs from the table's
  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random)) {
    // the k columns read the NFW kernel directly (header, item 2)
    if (like.halo_model[3] != HALO_PROFILE_NFW) {
      log_fatal("like.halo_model[3] = %d not supported", like.halo_model[3]);
      exit(1);
    }

    // warm-up (header, Thread safety): every lazy table the threaded
    // loop reads is built here, on one thread
    halo_warmup(lim[0][0], exp(lim[1][0]), 0, 0);

    /* PHYSICAL DERIVATION & LOGIC FLOW (full derivation: the header)
       1. node q: M_q, GL weight w_q; nu0_q = delta_c/sigma(M_q)
       2. row i (threaded): nu = nu0_q/D(a); c = conc(M_q, D);
          w1h_q = dn (M/rho_m)^2/m(c)^2; w2h_q = dn b(nu) (M/rho_m)/m(c);
          dn = w_q (rho_m/M) dlnnu/dlnM f(nu) nu
       3. column j: um = nfw_um(c, k r_s); I02 = sum_q w1h_q um^2;
          I11 = sum_q w2h_q um + A u_c(k|M_min)
       4. table = ln(I02 + I11^2 P_lin) */

    // --- 3. PER MASS NODE ---
    // quantities that depend on M alone (header, item 2, first row)
    const double rho_m     = cosmology.rho_crit * cosmology.Omega_m;
    const double rho_delta = Delta * rho_m;  // Delta x mean matter density

    for (int q=0; q<n_nodes; q++) {
      const double m = mass_node[0][q];
      mass_node[2][q] = delta_c/sqrt(sigma2(m));  // nu0 = delta_c/sigma(M)
      // GL weight x the (rho_m/M) dlnnu/dlnM of the mass function
      mass_node[3][q] = mass_node[1][q]*(rho_m/m)*dlognudlogm(m);
      mass_node[4][q] = m/rho_m;  // the matter window amplitude
      // r_Delta = (3 M/(4 pi rho_Delta))^(1/3), the halo edge
      mass_node[5][q] = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
    }

    // --- 4. PER a ROW, THREADED ---
    // each row: D(a), the Tinker f and b parameters (*_params_at, the
    // nu-independent halves), A(a), c(M_min) (header, item 2, second row)
    const double m_min = limits.halo_m_min;  // M_min of A u_c(k|M_min)

    #pragma omp parallel for schedule(static)
    for (int i=0; i<Ntable.N_a; i++) {
      const double ai = lim[0][0] + i*lim[0][2];
      const double D  = growfac(ai);

      // the nu-independent halves of Tinker f(nu) and b(nu) at this a
      const fnu_params   f_params = fnu_params_at(ai);
      const hb1nu_params b_params = hb1nu_params_at(ai);

      // A(a) = 1 - bias_norm(a): the HMx share of matter below M_min
      const double A_hmx    = 1.0 - bias_norm(ai);
      const double conc_min = conc(m_min, D);  // its c(M_min, a)

      // per-node rows of this a row (restrict: distinct rows)
      double* restrict conc_q = a_node[i][0];  // c(M_q, a)
      double* restrict ln1c_q = a_node[i][1];  // ln(1 + c)
      double* restrict rs_q   = a_node[i][2];  // r_s = r_Delta/c
      double* restrict lnrs_q = a_node[i][3];  // ln r_s
      double* restrict w1h_q  = a_node[i][4];  // 1-halo weight
      double* restrict w2h_q  = a_node[i][5];  // 2-halo weight

      // per (a, node): c(M, a), r_s and their logs, and the weights
      // w1h_q, w2h_q with the 1/m(c) of u = um/m(c) folded in (header,
      // item 2, third row); dn = w (rho_m/M) dlnnu/dlnM f(nu) nu
      for (int q=0; q<n_nodes; q++) {
        const double nu   = mass_node[2][q]/D;  // nu = nu0/D(a)
        const double c    = conc(mass_node[0][q], D);
        const double ln1c = log1p(c);
        const double mc   = ln1c - c/(1.0 + c);  // NFW norm m(c)
        const double dn   = mass_node[3][q]*fnu_core(nu, &f_params)*nu;
        conc_q[q] = c;
        ln1c_q[q] = ln1c;
        rs_q[q]   = mass_node[5][q]/c;  // r_s = r_Delta/c
        lnrs_q[q] = log(rs_q[q]);
        w1h_q[q]  = dn*(mass_node[4][q]/mc)*(mass_node[4][q]/mc);
        w2h_q[q]  = dn*hb1nu_core(nu, &b_params)*(mass_node[4][q]/mc);
      }

      // per k column: I02 and I11 as sums of the NFW kernel over the
      // nodes, the HMx term A u_c(k|M_min), then ln P (header, item 2,
      // last row)
      for (int j=0; j<Ntable.N_k_nlin; j++) {
        const double lnk = lim[1][0] + j*lim[1][2];
        const double kj  = exp(lnk);

        double sum_I02 = 0.0;  // 1-halo: sum_q w1h_q um^2
        double sum_I11 = 0.0;  // 2-halo: sum_q w2h_q um

        for (int q=0; q<n_nodes; q++) {
          const double um = nfw_um(conc_q[q], kj*rs_q[q],
                                   lnk + lnrs_q[q], ln1c_q[q]);
          sum_I02 += w1h_q[q]*um*um;
          sum_I11 += w2h_q[q]*um;
        }

        const double I11 = sum_I11 + A_hmx*u_c(conc_min, kj, m_min, ai);
        table[i][j] = log(sum_I02 + I11*I11*p_lin(kj, ai));
      }
    }

    // stamp the table with the tags it was built from
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }

  // --- 5. TABLE READ ---
  // bilinear read of ln P; 0 outside the a range
  if ((a < lim[0][0]) || (a > lim[0][1])) {
    return 0.0;
  }

  return exp(interpol2d(table,
                        Ntable.N_a, lim[0][0], lim[0][1], lim[0][2], a,
                        Ntable.N_k_nlin, lim[1][0], lim[1][1], lim[1][2],
                        log(k)));
}


// ---------------------------------------------------------------------------
// P_my(k, a), the halo-model matter-pressure cross spectrum, from a table
// of ln P on Ntable.N_a x Ntable.N_k_nlin nodes uniform in (a, ln k), read
// bilinearly (interpol2d) and exponentiated (section banner):
//
//   P_my   = I02_my S(k, a) + I11_m I11_y P_lin        (2005.00009 Eqs. 1-2)
//   I02_my = int dlnM dn/dlnM (M/rho_m) u(k|M) W_y(M, k)
//   I11_m  = int dlnM dn/dlnM b(nu) (M/rho_m) u(k|M) + A(a) u(k|M_min)
//   I11_y  = int dlnM dn/dlnM b(nu) [W_y(M, k) + W_ejc(M)]
//            + A(a) [W_y(M_min, k) + W_ejc(M_min)]/(M_min/rho_m)
//   S      = x/(1 + x),  x = (k/k_s)^4                   (2009.01858 Eq. 17)
//
// dn/dlnM, (M/rho_m) u, b and A(a) = 1 - bias_norm(a) as in p_mm (its
// header). W_y = Y(a) B(M) u_KS(c, k, r_Delta) is the bound-gas pressure
// window (GAS PROFILES banner), Y(a) = (2 alpha/(3a)) mu_p/mu_e and
// B(M) = f_bnd(M) M^2/r_Delta; W_ejc = u_y_ejc(M) the ejected gas, in the
// 2-halo term only and k-independent. k_s(a) = 0.05618 (sigma8 a)^-1.013
// h/Mpc (2009.01858 Table 2) with sigma8 = sigma(M8) read from the sigma2
// table, M8 = (4 pi/3) rho_m (8 Mpc/h)^3.
//
// 1. Quadrature: the 1024-node Gauss-Legendre rule of p_mm (its header,
// item 1) over [ln M_min, ln M_max], mapped once in the rebuild block.
//
// 2. Loop levels as in p_mm (its header, item 2), two kernels per node:
// nfw_um for the matter leg, u_KS for the pressure leg:
//
//   per refill, per node q       M, w; nu0 = delta_c/sigma(M);
//     (mass_node)                w (rho_m/M) dlnnu/dlnM; M/rho_m; r_Delta;
//                                B(M); W_ejc(M)
//   per refill                   mu_p/mu_e; sigma8; the M_min pieces of the
//                                HMx terms
//   per a row i, threaded        D(a); Tinker f, b parameters; A(a); c(M_min);
//                                Y(a); k_s(a)
//   per (i, q) (a_node[i])       nu = nu0/D; c = conc(M, D); ln(1+c);
//                                m(c); r_s = r_Delta/c, ln r_s;
//                                w1h_q = dn (M/rho_m)/m(c) Y B,
//                                w2hm_q = dn b (M/rho_m)/m(c),
//                                w2hy_q = dn b Y B, with
//                                dn = w (rho_m/M) dlnnu/dlnM f(nu) nu;
//                                sum_ejc = sum dn b W_ejc
//   per (i, k), sum over q       um = nfw_um(c, k r_s, ln k + ln r_s, ln(1+c))
//                                uy = u_KS(c, k, r_Delta);
//                                I02 = sum w1h_q uy um;
//                                I11_m = sum w2hm_q um + HMx;
//                                I11_y = sum w2hy_q uy + sum_ejc + HMx; ln P
//
// The 1/m(c) of u = um/m(c) lives in w1h_q and w2hm_q. The rows read the
// NFW kernel directly, so like.halo_model[3] must be HALO_PROFILE_NFW; the
// gas needs cosmology.Omega_b > 0 (f_bnd) and a polytropic index
// nuisance.gas[0] > 1 (u_KS); anything else aborts.
//
// Thread safety: the single-threaded halo_warmup(a_min, k_min, 1, 0) call
// before the threaded loop builds every lazy table the rows read (its
// header), so inside the loop they are only read.
//
// Cache invalidation:
//   rebuild block (table, mass_node, a_node, GL nodes, both grids; every
//     allocation lives here, one block each from malloc2d/malloc3d):
//     Ntable.random
//   refill: cosmology.random, Ntable.random or nuisance.random_gas
//
// Parameters:
//   k - wavenumber in (c/H0)^-1
//   a - scale factor
//
// Returns:
//   P_my(k, a) in U = G (M_sun/h)^2/(c/H0) (GAS PROFILES banner); 0 outside
//   [limits.a_min, 0.9999999]; ln P continued with unit slope outside
//   [ln k_min, ln k_max] (interpol2d)
// ---------------------------------------------------------------------------
double p_my(
    const double k,
    const double a
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];  // tags the table was built from
  static double** table = NULL;            // ln P on the (a, ln k) grid
  static double lim[2][3];           // [0] a grid: min, max, step;
                                     // [1] ln k grid: min, max, step
  static int n_nodes = 0;            // Gauss-Legendre nodes in ln M
  static double** mass_node = NULL;  // [8][n_nodes] per mass node q:
                                     // 0 M, 1 GL weight w, 2 nu0 = nu(D=1),
                                     // 3 w (rho_m/M) dlnnu/dlnM, 4 M/rho_m,
                                     // 5 r_Delta, 6 B = f_bnd M^2/r_Delta,
                                     // 7 W_ejc
  static double*** a_node = NULL;    // [N_a][7][n_nodes] per (a row, node):
                                     // 0 c, 1 ln(1+c), 2 r_s, 3 ln r_s,
                                     // 4 w1h, 5 w2hm, 6 w2hy

  // --- 1. NTABLE REBUILD: TABLE, NODE ARRAYS, GL RULE, GRIDS ---
  // the table, the per-node and per-(a, node) arrays (one block each from
  // malloc2d/malloc3d, so one free each), the GL nodes mapped onto
  // [ln M_min, ln M_max], both grids (header, item 1)
  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) {
      free(table);
      free(mass_node);
      free(a_node);
    }

    table     = (double**) malloc2d(Ntable.N_a, Ntable.N_k_nlin);
    n_nodes   = 1024;  // the largest rule GSL tabulates (header, item 1)
    mass_node = (double**) malloc2d(8, n_nodes);
    a_node    = (double***) malloc3d(Ntable.N_a, 7, n_nodes);

    // gsl_integration_glfixed_point(lo, hi, q, &x, &w, t): node q of the
    // rule t mapped onto [lo, hi], and its weight
    const double lnMmin = log(limits.halo_m_min);
    const double lnMmax = log(limits.halo_m_max);
    gsl_integration_glfixed_table* gl_table = malloc_gslint_glfixed(n_nodes);
    for (int q=0; q<n_nodes; q++) {
      double lnM;
      gsl_integration_glfixed_point(lnMmin, lnMmax, q, &lnM,
                                    &mass_node[1][q], gl_table);
      mass_node[0][q] = exp(lnM);
    }
    gsl_integration_glfixed_table_free(gl_table);

    // the uniform table axes: [0] a, [1] ln k (min, max, step)
    lim[0][0] = limits.a_min;
    lim[0][1] = 0.9999999;  // a_max, just below a = 1 (today)
    lim[0][2] = (lim[0][1] - lim[0][0]) / ((double) Ntable.N_a - 1.0);
    lim[1][0] = log(limits.k_min_cH0);
    lim[1][1] = log(limits.k_max_cH0);
    lim[1][2] = (lim[1][1] - lim[1][0]) / ((double) Ntable.N_k_nlin - 1.0);
  }

  // --- 2. REFILL GUARD AND SINGLE-THREADED WARM-UP ---
  // refill when the cosmology, Ntable or gas tag differs from the table's
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], Ntable.random) ||
      fdiff2(cache[2], nuisance.random_gas))
  {
    // the k columns read the NFW kernel directly (header, item 2)
    if (like.halo_model[3] != HALO_PROFILE_NFW) {
      log_fatal("like.halo_model[3] = %d not supported", like.halo_model[3]);
      exit(1);
    }

    // the gas: f_bnd carries Omega_b/Omega_m, u_KS the exponents
    // Gamma/(Gamma - 1) and 1/(Gamma - 1)
    if (!(cosmology.Omega_b > 0)) {
      log_fatal("Compton-y spectra need cosmology.Omega_b > 0 "
                "(set_cosmological_parameters)");
      exit(1);
    }

    if (!(nuisance.gas[0] > 1)) {
      log_fatal("Compton-y spectra need a polytropic index gas[0] = %g > 1",
                nuisance.gas[0]);
      exit(1);
    }

    // warm-up (header, Thread safety): every lazy table the threaded
    // loop reads is built here, on one thread
    halo_warmup(lim[0][0], exp(lim[1][0]), 1, 0);

    /* PHYSICAL DERIVATION & LOGIC FLOW (full derivation: the header)
       1. node q: M_q, w_q, nu0_q; B = f_bnd M^2/r_Delta; W_ejc
       2. row i (threaded): nu = nu0_q/D(a); c = conc(M_q, D); Y(a); k_s(a);
          w1h_q = dn (M/rho_m)/m(c) Y B; w2hm_q = dn b (M/rho_m)/m(c);
          w2hy_q = dn b Y B; sum_ejc = sum_q dn b W_ejc
       3. column j: I02 = sum_q w1h_q uy um; I11_m = sum_q w2hm_q um + HMx;
          I11_y = sum_q w2hy_q uy + sum_ejc + HMx
       4. table = ln(I02 S + I11_m I11_y P_lin),
          S = x/(1 + x), x = (k/k_s)^4 */

    // --- 3. PER MASS NODE ---
    // quantities that depend on M alone (header, item 2, first row)
    const double rho_m     = cosmology.rho_crit * cosmology.Omega_m;
    const double rho_delta = Delta * rho_m;  // Delta x mean matter density

    for (int q=0; q<n_nodes; q++) {
      const double m = mass_node[0][q];
      mass_node[2][q] = delta_c/sqrt(sigma2(m));  // nu0 = delta_c/sigma(M)
      // GL weight x the (rho_m/M) dlnnu/dlnM of the mass function
      mass_node[3][q] = mass_node[1][q]*(rho_m/m)*dlognudlogm(m);
      mass_node[4][q] = m/rho_m;  // the matter window amplitude
      // r_Delta = (3 M/(4 pi rho_Delta))^(1/3), the halo edge
      mass_node[5][q] = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
      mass_node[6][q] = frac_bnd(m)*m*(m/mass_node[5][q]);  // B(M)
      mass_node[7][q] = u_y_ejc(m);  // W_ejc(M), the ejected gas
    }

    // --- 4. GAS AND M_MIN CONSTANTS ---
    // mu_p, mu_e of the ionized gas (GAS PROFILES banner), sigma8 from
    // the sigma2 table at M8 = (4 pi/3) rho_m (8 Mpc/h)^3, and the M_min
    // pieces of the HMx terms (header, item 2, second row)
    const double mu_p = 4.0/(3.0 + 5*nuisance.gas[10]);  // 4/(3 + 5 f_H)
    const double mu_e = 2.0/(1.0 + nuisance.gas[10]);    // 2/(1 + f_H)

    const double R8     = 8.0/cosmology.coverH0;  // 8 Mpc/h in c/H0 units
    const double sigma8 = sqrt(sigma2(4.0*M_PI/3.0*rho_m*R8*R8*R8));

    // the M_min pieces: M_min/rho_m, r_Delta, B(M_min), W_ejc(M_min)
    const double m_min      = limits.halo_m_min;
    const double vol_min    = m_min/rho_m;
    const double rdelta_min = pow(3./(4.0*M_PI)*(m_min/rho_delta), 1./3.);
    const double B_min      = frac_bnd(m_min)*m_min*(m_min/rdelta_min);
    const double Wejc_min   = u_y_ejc(m_min);

    // --- 5. PER a ROW, THREADED ---
    // each row: D(a), the Tinker f and b parameters, A(a), c(M_min),
    // Y(a) of the bound-gas window, k_s(a) of the damping (header,
    // item 2, third row)
    #pragma omp parallel for schedule(static)
    for (int i=0; i<Ntable.N_a; i++) {
      const double ai = lim[0][0] + i*lim[0][2];
      const double D  = growfac(ai);

      // the nu-independent halves of Tinker f(nu) and b(nu) at this a
      const fnu_params   f_params = fnu_params_at(ai);
      const hb1nu_params b_params = hb1nu_params_at(ai);

      // A(a) = 1 - bias_norm(a): the HMx share of matter below M_min
      const double A_hmx    = 1.0 - bias_norm(ai);
      const double conc_min = conc(m_min, D);  // its c(M_min, a)

      // Y(a) = (2 alpha/(3 a)) mu_p/mu_e, alpha = nuisance.gas[5] (header)
      const double Y_gas = (2.0*nuisance.gas[5]/(3.0*ai))*(mu_p/mu_e);
      // k_s(a) = 0.05618 (sigma8 a)^-1.013 h/Mpc in (c/H0)^-1 (header)
      const double k_s = 0.05618/pow(sigma8*ai, 1.013)*cosmology.coverH0;

      // per-node rows of this a row (restrict: distinct rows)
      const double* restrict rdelta_q = mass_node[5];  // r_Delta(M_q)
      double* restrict conc_q = a_node[i][0];  // c(M_q, a)
      double* restrict ln1c_q = a_node[i][1];  // ln(1 + c)
      double* restrict rs_q   = a_node[i][2];  // r_s = r_Delta/c
      double* restrict lnrs_q = a_node[i][3];  // ln r_s
      double* restrict w1h_q  = a_node[i][4];  // 1-halo weight
      double* restrict w2hm_q = a_node[i][5];  // 2-halo matter weight
      double* restrict w2hy_q = a_node[i][6];  // 2-halo pressure weight

      // per (a, node): c(M, a), r_s and their logs, and the weights with
      // the 1/m(c) of u = um/m(c) folded into the matter legs (header,
      // item 2, fourth row); dn = w (rho_m/M) dlnnu/dlnM f(nu) nu
      double sum_ejc = 0.0;  // sum_q dn b W_ejc: the k-independent
                             // ejected-gas share of I11_y
      for (int q=0; q<n_nodes; q++) {
        const double nu   = mass_node[2][q]/D;  // nu = nu0/D(a)
        const double c    = conc(mass_node[0][q], D);
        const double dn   = mass_node[3][q]*fnu_core(nu, &f_params)*nu;
        const double bias = hb1nu_core(nu, &b_params);
        const double YB   = Y_gas*mass_node[6][q];  // W_y without u_KS
        conc_q[q] = c;
        ln1c_q[q] = log1p(c);
        rs_q[q]   = rdelta_q[q]/c;  // r_s = r_Delta/c
        lnrs_q[q] = log(rs_q[q]);
        const double mc = ln1c_q[q] - c/(1.0 + c);  // NFW norm m(c)
        w1h_q[q]  = dn*(mass_node[4][q]/mc)*YB;
        w2hm_q[q] = dn*bias*(mass_node[4][q]/mc);
        w2hy_q[q] = dn*bias*YB;
        sum_ejc += dn*bias*mass_node[7][q];
      }

      // per k column: I02 and the two I11 as sums of the two kernels
      // over the nodes, the HMx terms at M_min, the damping S, then ln P
      // (header, item 2, last row)
      for (int j=0; j<Ntable.N_k_nlin; j++) {
        const double lnk = lim[1][0] + j*lim[1][2];
        const double kj  = exp(lnk);

        double sum_I02  = 0.0;  // 1-halo: sum_q w1h_q uy um
        double sum_I11m = 0.0;  // 2-halo matter leg: sum_q w2hm_q um
        double sum_I11y = 0.0;  // 2-halo pressure leg: sum_q w2hy_q uy

        for (int q=0; q<n_nodes; q++) {
          const double um = nfw_um(conc_q[q], kj*rs_q[q],
                                   lnk + lnrs_q[q], ln1c_q[q]);
          const double uy = u_KS(conc_q[q], kj, rdelta_q[q]);
          sum_I02  += w1h_q[q]*uy*um;
          sum_I11m += w2hm_q[q]*um;
          sum_I11y += w2hy_q[q]*uy;
        }

        // the damping S = x/(1 + x), x = (k/k_s)^4 (header)
        const double x4  = (kj/k_s)*(kj/k_s)*(kj/k_s)*(kj/k_s);
        const double P1H = sum_I02*(x4/(x4 + 1.0));

        const double I11m = sum_I11m + A_hmx*u_c(conc_min, kj, m_min, ai);

        // W_y + W_ejc at M_min, the window of the HMx term of I11_y
        const double W_min =
            Y_gas*B_min*u_KS(conc_min, kj, rdelta_min) + Wejc_min;
        const double I11y = sum_I11y + sum_ejc + A_hmx*W_min/vol_min;

        table[i][j] = log(P1H + I11m*I11y*p_lin(kj, ai));
      }
    }

    // stamp the table with the tags it was built from
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_gas;
  }

  // --- 6. TABLE READ ---
  // bilinear read of ln P; 0 outside the a range
  if ((a < lim[0][0]) || (a > lim[0][1])) {
    return 0.0;
  }

  return exp(interpol2d(table,
                        Ntable.N_a, lim[0][0], lim[0][1], lim[0][2], a,
                        Ntable.N_k_nlin, lim[1][0], lim[1][1], lim[1][2],
                        log(k)));
}


// ---------------------------------------------------------------------------
// P_yy(k, a), the halo-model pressure auto spectrum, from a table of ln P
// on Ntable.N_a x Ntable.N_k_nlin nodes uniform in (a, ln k), read
// bilinearly (interpol2d) and exponentiated (section banner):
//
//   P_yy   = I02_yy S(k, a) + I11_y^2 P_lin              (2005.00009 Eqs. 1-2)
//   I02_yy = int dlnM dn/dlnM W_y(M, k)^2
//   I11_y  = int dlnM dn/dlnM b(nu) [W_y(M, k) + W_ejc(M)]
//            + A(a) [W_y(M_min, k) + W_ejc(M_min)]/(M_min/rho_m)
//   S      = x/(1 + x),  x = (k/k_s)^4                   (2009.01858 Eq. 17)
//
// W_y = Y(a) B(M) u_KS, W_ejc, k_s(a), dn/dlnM, b and A(a) as in p_my (its
// header). No matter leg: the only kernel is u_KS, and no M/rho_m enters
// (the pressure window is the full volume integral, GAS PROFILES banner).
//
// 1. Quadrature: the 1024-node Gauss-Legendre rule of p_mm (its header,
// item 1) over [ln M_min, ln M_max], mapped once in the rebuild block.
//
// 2. Loop levels as in p_my (its header, item 2) without the matter leg:
//
//   per refill, per node q       M, w; nu0; w (rho_m/M) dlnnu/dlnM; r_Delta;
//     (mass_node)                B(M); W_ejc(M)
//   per refill                   mu_p/mu_e; sigma8; the M_min pieces
//   per a row i, threaded        D(a); Tinker f, b parameters; A(a); c(M_min);
//                                Y(a); k_s(a)
//   per (i, q) (a_node[i])       nu = nu0/D; c = conc(M, D);
//                                w1h_q = dn (Y B)^2, w2hy_q = dn b Y B;
//                                sum_ejc = sum dn b W_ejc
//   per (i, k), sum over q       uy = u_KS(c, k, r_Delta);
//                                I02 = sum w1h_q uy^2; I11_y = sum w2hy_q uy
//                                + sum_ejc + HMx; ln P
//
// The gas needs cosmology.Omega_b > 0 (f_bnd) and a polytropic index
// nuisance.gas[0] > 1 (u_KS); anything else aborts.
//
// Thread safety: the single-threaded halo_warmup(a_min, k_min, 1, 0) call
// before the threaded loop builds every lazy table the rows read (its
// header), so inside the loop they are only read.
//
// Cache invalidation:
//   rebuild block (table, mass_node, a_node, GL nodes, both grids; every
//     allocation lives here, one block each from malloc2d/malloc3d):
//     Ntable.random
//   refill: cosmology.random, Ntable.random or nuisance.random_gas
//
// Parameters:
//   k - wavenumber in (c/H0)^-1
//   a - scale factor
//
// Returns:
//   P_yy(k, a) in U^2 (c/H0)^-3, U = G (M_sun/h)^2/(c/H0) (GAS PROFILES
//   banner); 0 outside [limits.a_min, 0.9999999]; ln P continued with unit
//   slope outside [ln k_min, ln k_max] (interpol2d)
// ---------------------------------------------------------------------------
double p_yy(
    const double k,
    const double a
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];  // tags the table was built from
  static double** table = NULL;            // ln P on the (a, ln k) grid
  static double lim[2][3];           // [0] a grid: min, max, step;
                                     // [1] ln k grid: min, max, step
  static int n_nodes = 0;            // Gauss-Legendre nodes in ln M
  static double** mass_node = NULL;  // [7][n_nodes] per mass node q:
                                     // 0 M, 1 GL weight w, 2 nu0 = nu(D=1),
                                     // 3 w (rho_m/M) dlnnu/dlnM, 4 r_Delta,
                                     // 5 B = f_bnd M^2/r_Delta, 6 W_ejc
  static double*** a_node = NULL;    // [N_a][3][n_nodes] per (a row, node):
                                     // 0 c, 1 w1h, 2 w2hy

  // --- 1. NTABLE REBUILD: TABLE, NODE ARRAYS, GL RULE, GRIDS ---
  // the table, the per-node and per-(a, node) arrays (one block each from
  // malloc2d/malloc3d, so one free each), the GL nodes mapped onto
  // [ln M_min, ln M_max], both grids (header, item 1)
  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) {
      free(table);
      free(mass_node);
      free(a_node);
    }

    table     = (double**) malloc2d(Ntable.N_a, Ntable.N_k_nlin);
    n_nodes   = 1024;  // the largest rule GSL tabulates (header, item 1)
    mass_node = (double**) malloc2d(7, n_nodes);
    a_node    = (double***) malloc3d(Ntable.N_a, 3, n_nodes);

    // gsl_integration_glfixed_point(lo, hi, q, &x, &w, t): node q of the
    // rule t mapped onto [lo, hi], and its weight
    const double lnMmin = log(limits.halo_m_min);
    const double lnMmax = log(limits.halo_m_max);
    gsl_integration_glfixed_table* gl_table = malloc_gslint_glfixed(n_nodes);
    for (int q=0; q<n_nodes; q++) {
      double lnM;
      gsl_integration_glfixed_point(lnMmin, lnMmax, q, &lnM,
                                    &mass_node[1][q], gl_table);
      mass_node[0][q] = exp(lnM);
    }
    gsl_integration_glfixed_table_free(gl_table);

    // the uniform table axes: [0] a, [1] ln k (min, max, step)
    lim[0][0] = limits.a_min;
    lim[0][1] = 0.9999999;  // a_max, just below a = 1 (today)
    lim[0][2] = (lim[0][1] - lim[0][0]) / ((double) Ntable.N_a - 1.0);
    lim[1][0] = log(limits.k_min_cH0);
    lim[1][1] = log(limits.k_max_cH0);
    lim[1][2] = (lim[1][1] - lim[1][0]) / ((double) Ntable.N_k_nlin - 1.0);
  }

  // --- 2. REFILL GUARD AND SINGLE-THREADED WARM-UP ---
  // refill when the cosmology, Ntable or gas tag differs from the table's
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], Ntable.random) ||
      fdiff2(cache[2], nuisance.random_gas))
  {
    // the gas: f_bnd carries Omega_b/Omega_m, u_KS the exponents
    // Gamma/(Gamma - 1) and 1/(Gamma - 1)
    if (!(cosmology.Omega_b > 0)) {
      log_fatal("Compton-y spectra need cosmology.Omega_b > 0 "
                "(set_cosmological_parameters)");
      exit(1);
    }

    if (!(nuisance.gas[0] > 1)) {
      log_fatal("Compton-y spectra need a polytropic index gas[0] = %g > 1",
                nuisance.gas[0]);
      exit(1);
    }

    // warm-up (header, Thread safety): every lazy table the threaded
    // loop reads is built here, on one thread
    halo_warmup(lim[0][0], exp(lim[1][0]), 1, 0);

    /* PHYSICAL DERIVATION & LOGIC FLOW (full derivation: the header)
       1. node q: M_q, w_q, nu0_q; B = f_bnd M^2/r_Delta; W_ejc
       2. row i (threaded): nu = nu0_q/D(a); c = conc(M_q, D); Y(a); k_s(a);
          w1h_q = dn (Y B)^2; w2hy_q = dn b Y B; sum_ejc = sum_q dn b W_ejc
       3. column j: I02 = sum_q w1h_q uy^2;
          I11_y = sum_q w2hy_q uy + sum_ejc + HMx
       4. table = ln(I02 S + I11_y^2 P_lin),
          S = x/(1 + x), x = (k/k_s)^4 */

    // --- 3. PER MASS NODE ---
    // quantities that depend on M alone (header, item 2, first row)
    const double rho_m     = cosmology.rho_crit * cosmology.Omega_m;
    const double rho_delta = Delta * rho_m;  // Delta x mean matter density

    for (int q=0; q<n_nodes; q++) {
      const double m = mass_node[0][q];
      mass_node[2][q] = delta_c/sqrt(sigma2(m));  // nu0 = delta_c/sigma(M)
      // GL weight x the (rho_m/M) dlnnu/dlnM of the mass function
      mass_node[3][q] = mass_node[1][q]*(rho_m/m)*dlognudlogm(m);
      // r_Delta = (3 M/(4 pi rho_Delta))^(1/3), the halo edge
      mass_node[4][q] = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
      mass_node[5][q] = frac_bnd(m)*m*(m/mass_node[4][q]);  // B(M)
      mass_node[6][q] = u_y_ejc(m);  // W_ejc(M), the ejected gas
    }

    // --- 4. GAS AND M_MIN CONSTANTS ---
    // mu_p, mu_e of the ionized gas (GAS PROFILES banner), sigma8 from
    // the sigma2 table at M8 = (4 pi/3) rho_m (8 Mpc/h)^3, and the M_min
    // pieces of the HMx term (header, item 2, second row)
    const double mu_p = 4.0/(3.0 + 5*nuisance.gas[10]);  // 4/(3 + 5 f_H)
    const double mu_e = 2.0/(1.0 + nuisance.gas[10]);    // 2/(1 + f_H)

    const double R8     = 8.0/cosmology.coverH0;  // 8 Mpc/h in c/H0 units
    const double sigma8 = sqrt(sigma2(4.0*M_PI/3.0*rho_m*R8*R8*R8));

    // the M_min pieces: M_min/rho_m, r_Delta, B(M_min), W_ejc(M_min)
    const double m_min      = limits.halo_m_min;
    const double vol_min    = m_min/rho_m;
    const double rdelta_min = pow(3./(4.0*M_PI)*(m_min/rho_delta), 1./3.);
    const double B_min      = frac_bnd(m_min)*m_min*(m_min/rdelta_min);
    const double Wejc_min   = u_y_ejc(m_min);

    // --- 5. PER a ROW, THREADED ---
    // each row: D(a), the Tinker f and b parameters, A(a), c(M_min),
    // Y(a) of the bound-gas window, k_s(a) of the damping (header,
    // item 2, third row)
    #pragma omp parallel for schedule(static)
    for (int i=0; i<Ntable.N_a; i++) {
      const double ai = lim[0][0] + i*lim[0][2];
      const double D  = growfac(ai);

      // the nu-independent halves of Tinker f(nu) and b(nu) at this a
      const fnu_params   f_params = fnu_params_at(ai);
      const hb1nu_params b_params = hb1nu_params_at(ai);

      // A(a) = 1 - bias_norm(a): the HMx share of matter below M_min
      const double A_hmx    = 1.0 - bias_norm(ai);
      const double conc_min = conc(m_min, D);  // its c(M_min, a)

      // Y(a) = (2 alpha/(3 a)) mu_p/mu_e, alpha = nuisance.gas[5] (header)
      const double Y_gas = (2.0*nuisance.gas[5]/(3.0*ai))*(mu_p/mu_e);
      // k_s(a) = 0.05618 (sigma8 a)^-1.013 h/Mpc in (c/H0)^-1 (header)
      const double k_s = 0.05618/pow(sigma8*ai, 1.013)*cosmology.coverH0;

      // per-node rows of this a row (restrict: distinct rows)
      const double* restrict rdelta_q = mass_node[4];  // r_Delta(M_q)
      double* restrict conc_q = a_node[i][0];  // c(M_q, a)
      double* restrict w1h_q  = a_node[i][1];  // 1-halo weight
      double* restrict w2hy_q = a_node[i][2];  // 2-halo pressure weight

      // per (a, node): c(M, a) and the weights (header, item 2, fourth
      // row); dn = w (rho_m/M) dlnnu/dlnM f(nu) nu
      double sum_ejc = 0.0;  // sum_q dn b W_ejc: the k-independent
                             // ejected-gas share of I11_y
      for (int q=0; q<n_nodes; q++) {
        const double nu   = mass_node[2][q]/D;  // nu = nu0/D(a)
        const double dn   = mass_node[3][q]*fnu_core(nu, &f_params)*nu;
        const double bias = hb1nu_core(nu, &b_params);
        const double YB   = Y_gas*mass_node[5][q];  // W_y without u_KS
        conc_q[q] = conc(mass_node[0][q], D);
        w1h_q[q]  = dn*YB*YB;
        w2hy_q[q] = dn*bias*YB;
        sum_ejc += dn*bias*mass_node[6][q];
      }

      // per k column: I02 and I11_y as sums of the kernel over the
      // nodes, the HMx term at M_min, the damping S, then ln P (header,
      // item 2, last row)
      for (int j=0; j<Ntable.N_k_nlin; j++) {
        const double lnk = lim[1][0] + j*lim[1][2];
        const double kj  = exp(lnk);

        double sum_I02  = 0.0;  // 1-halo: sum_q w1h_q uy^2
        double sum_I11y = 0.0;  // 2-halo: sum_q w2hy_q uy

        for (int q=0; q<n_nodes; q++) {
          const double uy = u_KS(conc_q[q], kj, rdelta_q[q]);
          sum_I02  += w1h_q[q]*uy*uy;
          sum_I11y += w2hy_q[q]*uy;
        }

        // the damping S = x/(1 + x), x = (k/k_s)^4 (header)
        const double x4  = (kj/k_s)*(kj/k_s)*(kj/k_s)*(kj/k_s);
        const double P1H = sum_I02*(x4/(x4 + 1.0));

        // W_y + W_ejc at M_min, the window of the HMx term of I11_y
        const double W_min =
            Y_gas*B_min*u_KS(conc_min, kj, rdelta_min) + Wejc_min;
        const double I11y = sum_I11y + sum_ejc + A_hmx*W_min/vol_min;

        table[i][j] = log(P1H + I11y*I11y*p_lin(kj, ai));
      }
    }

    // stamp the table with the tags it was built from
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_gas;
  }

  // --- 6. TABLE READ ---
  // bilinear read of ln P; 0 outside the a range
  if ((a < lim[0][0]) || (a > lim[0][1])) {
    return 0.0;
  }

  return exp(interpol2d(table,
                        Ntable.N_a, lim[0][0], lim[0][1], lim[0][2], a,
                        Ntable.N_k_nlin, lim[1][0], lim[1][1], lim[1][2],
                        log(k)));
}


// ---------------------------------------------------------------------------
// P_gm(k, a, ni), the halo-model galaxy-matter power spectrum of lens bin
// ni, from a table of ln P per bin on na x Ntable.N_k_nlin nodes, na =
// Ntable.N_a/5, uniform in a over the bin's range [amin_lens, amax_lens]
// and in ln k, read bilinearly (interpol2d) and exponentiated:
//
//   P_gm = P_delta b_gal + GM02/n_gal
//   GM02 = int dlnM dn/dlnM (M/rho_m) u_m(k|M) [N_s u_g(k|M) + f_c N_c]
//
// 2-halo: the nonlinear matter spectrum (Pdelta) times the mean galaxy
// bias (bgal). 1-halo: the satellite-matter and central-matter pairs of
// one halo per galaxy (ngal): satellites follow u_g, the NFW profile at
// c_g = gc c, gc = nuisance.gc[ni] (u_g header); the central sits at the
// center (window 1). dn/dlnM, (M/rho_m) u_m as in p_mm (its header);
// N_c, N_s, f_c the occupation of the GALAXY PROFILES banner.
//
// 1. Quadrature: the 1024-node Gauss-Legendre rule of p_mm (its header,
// item 1) over ln M from ln 10^(lg M_min - 1) of the bin to
// ln limits.halo_m_max: nodes x_q, weights w_q on [-1, 1] (gl) are
// mapped per bin in the refill, ln M_q = mid + half_width x_q, weight
// half_width w_q.
//
// 2. Loop levels as in p_mm (its header, item 2): the innermost loop is
// the NFW kernel nfw_um alone. The occupation does not depend on a
// (HOD_nc, HOD_ns only range-check it: amin_lens is a placeholder):
//
//   per refill, per (bin, node) -> bin_tab:
//     M, half_width w (rho_m/M) dlnnu/dlnM, nu0, r_Delta, N_s, f_c N_c
//   per bin, per a row, threaded:
//     D(a); Tinker f parameters; ngal, bgal
//   per (a, node) -> a_tab:
//     c = conc(M, D) and c_g = gc c, each with ln(1+c), m(c), r_s, ln r_s;
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
//   rebuild block (table, lim, gl, bin_tab, a_tab; every allocation
//     lives here, one block each from malloc2d/malloc3d): Ntable.random
//     or redshift.random_clustering (bin count and a ranges: clustering
//     n(z))
//   refill: cosmology.random, Ntable.random, nuisance.random_galaxy_bias
//     (HOD and gc) or redshift.random_clustering
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
  static double*** bin_tab = NULL; // [nbin][6][nnode] per (bin, mass
                                   // node): M, half_width w (rho_m/M)
                                   // dlnnu/dlnM, nu at D = 1, r_Delta,
                                   // N_s, f_c N_c
  static double*** a_tab   = NULL; // [n_threads][10][nnode]: one (bin,
                                   //   a-row) iteration's scratch
                                   // of one bin: c, ln(1+c), r_s, ln r_s
                                   // and the same for c_g = gc c, then
                                   // the weights W1, W0

  // --- 1. REBUILD: SIZES, ALLOCATIONS, GL RULE, TABLE AXES ---

  // first call, or the Ntable or clustering-n(z) tag differs from the
  // allocation's; every allocation is one block from malloc2d/malloc3d,
  // so one free each (header, item 1)
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
    }

    nbin  = redshift.clustering_nbin;
    na    = (int) Ntable.N_a/5.0; // a fifth of p_mm's a grid: each bin's
                                  // a range is a slice of the full range
    nnode = 1024;                 // largest predefined GSL rule

    table   = (double***) malloc3d(nbin, na, Ntable.N_k_nlin);
    lim     = (double**) malloc2d(nbin+1, 3);
    gl      = (double**) malloc2d(2, nnode);
    bin_tab = (double***) malloc3d(nbin, 6, nnode);
    // one scratch block per thread (the thread count of this rebuild;
    // raising OMP_NUM_THREADS afterwards requires an Ntable bump)
    a_tab   = (double***) malloc3d(omp_get_max_threads(), 10, nnode);

    // gsl_integration_glfixed_point(lo, hi, q, &x, &w, t): node q of the
    // rule t mapped onto [lo, hi], and its weight; kept on [-1, 1] here
    gsl_integration_glfixed_table* t = malloc_gslint_glfixed(nnode);
    for (int q=0; q<nnode; q++) {
      gsl_integration_glfixed_point(-1.0, 1.0, q, &gl[0][q], &gl[1][q], t);
    }
    gsl_integration_glfixed_table_free(t);

    // a grid of bin l over its lens range; node i sits at
    // lim[l][0] + i lim[l][2], both ends included
    for (int l=0; l<nbin; l++) {
      lim[l][0] = amin_lens(l);
      lim[l][1] = amax_lens(l);
      lim[l][2] = (lim[l][1] - lim[l][0])/((double) na - 1.0);
    }

    // ln k grid, shared by all bins
    lim[nbin][0] = log(limits.k_min_cH0);
    lim[nbin][1] = log(limits.k_max_cH0);
    lim[nbin][2] = (lim[nbin][1]-lim[nbin][0])
                   /((double) Ntable.N_k_nlin - 1.0);
  }

  // --- 2. REFILL: THE HOD-WEIGHTED ln P TABLE ---

  // any of the four tags differs from the table's
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], Ntable.random)    ||
      fdiff2(cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(cache[3], redshift.random_clustering))
  {
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
          nu at D = 1, r_Delta, occupation N_s and f_c N_c
       2. per a row, threaded: D(a), Tinker f(nu) half, n_gal, b_gal
       3. thread scratch a_tab, per node of one a row: c, c_g = gc c,
          scale radii, logs, leg weights W1 (satellite), W0 (central)
       4. per k: GM02 = sum_q um (W1 ug + W0); table = ln P_gm */

    // --- 2b. PER (BIN, MASS NODE), SERIAL ---

    // the GL nodes mapped onto the bin's ln M range, the a-independent
    // factors, and the occupation at the placeholder a = amin; the
    // sigma2 and dlognudlogm reads happen here, before the threads
    const double rho_m     = cosmology.rho_crit * cosmology.Omega_m;
    const double rho_delta = Delta * rho_m;

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
      const double lnMmax     = log(limits.halo_m_max);
      const double half_width = 0.5*(lnMmax - lnMmin);
      const double mid        = 0.5*(lnMmax + lnMmin);

      const double fc = HOD_fc(l);

      // bin_tab rows: M | half_width w_q (rho_m/M) dlnnu/dlnM | nu at
      // D = 1 | r_Delta from M = (4 pi/3) Delta rho_m r_Delta^3 |
      // N_s | f_c N_c
      for (int q=0; q<nnode; q++) {
        const double m = exp(mid + half_width*gl[0][q]);
        bin_tab[l][0][q] = m;
        bin_tab[l][1][q] = half_width*gl[1][q]*(rho_m/m)*dlognudlogm(m);
        bin_tab[l][2][q] = delta_c/sqrt(sigma2(m));
        bin_tab[l][3][q] = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
        bin_tab[l][4][q] = HOD_ns(m, lim[l][0], l);
        bin_tab[l][5][q] = fc*HOD_nc(m, lim[l][0], l);
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

        // growth, the nu-independent Tinker f(nu) half, and the mean
        // galaxy density and bias of this a row
        const double     D        = growfac(ai);
        const fnu_params fnu_pars = fnu_params_at(ai);
        const double     n_gal    = ngal(l, ai);
        const double     b_gal    = bgal(l, ai);

        // thread-private scratch: this (bin, a-row) iteration fills
        // it and consumes it in its own k loop; restrict: each row is
        // reached only through its pointer, no reload after libm calls
        double** const wsp = a_tab[omp_get_thread_num()];
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
          const double nu = bin_tab[l][2][q]/D;

          // c and c_g = gc c, with m(c) = ln(1+c) - c/(1+c) for each
          const double c     = conc(m, D);
          const double cg    = c*gc;
          const double ln1c  = log1p(c);
          const double ln1cg = log1p(cg);
          const double mc    = ln1c - c/(1.0 + c);
          const double mcg   = ln1cg - cg/(1.0 + cg);

          // dn_halo = quadrature weight x dn/dlnM (Tinker);
          // w_matter carries the matter leg's (M/rho_m) and 1/m(c)
          const double dn_halo  = bin_tab[l][1][q]*fnu_core(nu, &fnu_pars)*nu;
          const double w_matter = dn_halo*(m/rho_m)/mc;

          conc_halo[q] = c;
          ln1c_halo[q] = ln1c;
          r_s[q]       = bin_tab[l][3][q]/c;
          lnrs[q]      = log(r_s[q]);
          conc_gal[q]  = cg;
          ln1c_gal[q]  = ln1cg;
          r_sg[q]      = bin_tab[l][3][q]/cg;
          lnrsg[q]     = log(r_sg[q]);
          w1[q]        = w_matter*bin_tab[l][4][q]/mcg;
          w0[q]        = w_matter*bin_tab[l][5][q];
        }

        // per k: GM02 as a sum of the NFW kernel over the nodes, one
        // call per leg or one for both (same_conc), then ln P
        for (int j=0; j<Ntable.N_k_nlin; j++) {
          const double lnk = lim[nbin][0] + j*lim[nbin][2];
          const double kj  = exp(lnk);

          double gm02 = 0.0;
          if (same_conc) {
            for (int q=0; q<nnode; q++) {
              const double um = nfw_um(conc_halo[q], kj*r_s[q],
                                       lnk + lnrs[q], ln1c_halo[q]);
              gm02 += um*(w1[q]*um + w0[q]);
            }
          }
          else {
            for (int q=0; q<nnode; q++) {
              const double um = nfw_um(conc_halo[q], kj*r_s[q],
                                       lnk + lnrs[q], ln1c_halo[q]);
              const double ug = nfw_um(conc_gal[q], kj*r_sg[q],
                                       lnk + lnrsg[q], ln1c_gal[q]);
              gm02 += um*(w1[q]*ug + w0[q]);
            }
          }

          table[l][i][j] = log(Pdelta(kj, ai)*b_gal + gm02/n_gal);
        }
      }
    }

    // record the tags this table was built from
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_galaxy_bias;
    cache[3] = redshift.random_clustering;
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
// satellites follow u_g, the NFW profile at c_g = gc c (u_g header), the
// central sits at the center (window 1). dn/dlnM as in p_mm (its
// header); N_c(M), N_s(M), f_c the occupation of the GALAXY PROFILES
// banner.
//
// 1. Quadrature: the 1024-node Gauss-Legendre rule of p_mm (its header,
// item 1) over [ln limits.halo_m_min, ln limits.halo_m_max], the same for
// every bin: nodes and weights are mapped once, in the rebuild block.
//
// 2. Loop levels as in p_mm (its header, item 2): the innermost loop is
// the NFW kernel nfw_um alone. The occupation does not depend on a
// (HOD_nc, HOD_ns only range-check it: amin_lens is a placeholder):
//
//   per refill, per node -> mass_tab:
//     M, w, nu0, r_Delta, w (rho_m/M) dlnnu/dlnM
//   per refill, per (bin, node) -> occ_tab:
//     N_s, f_c N_c
//   per bin, per a row, threaded:
//     D(a); Tinker f parameters; ngal, bgal
//   per (a, node) -> a_tab:
//     c_g = gc conc(M, D), ln(1+c_g), m(c_g), r_s,g = r_Delta/c_g,
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
//     mass_tab (with the mapped GL nodes), occ_tab and a_tab
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
  static double**  mass_tab = NULL; // [5][nnode] per mass node: M,
                                    // weight, nu at D = 1, r_Delta,
                                    // weight x (rho_m/M) dlnnu/dlnM
  static double*** occ_tab  = NULL; // [nbin][2][nnode] per (bin, mass
                                    // node): N_s, f_c N_c
  static double*** a_tab    = NULL; // [n_threads][6][nnode]: one
                                    //   (bin, a-row) scratch
                                    // of one bin: c_g, ln(1+c_g), r_s,g,
                                    // ln r_s,g, W2, W1

  // --- 1. REBUILD: SIZES, ALLOCATIONS, MAPPED GL RULE, TABLE AXES ---

  // first call, or the Ntable or clustering-n(z) tag differs from the
  // allocation's; every allocation is one block from malloc2d/malloc3d,
  // so one free each (header, item 1)
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
    }

    nbin  = redshift.clustering_nbin;
    na    = (int) Ntable.N_a/5.0; // a fifth of p_mm's a grid: each bin's
                                  // a range is a slice of the full range
    nnode = 1024;                 // largest predefined GSL rule

    table    = (double***) malloc3d(nbin, na, Ntable.N_k_nlin);
    lim      = (double**) malloc2d(nbin+1, 3);
    mass_tab = (double**) malloc2d(5, nnode);
    occ_tab  = (double***) malloc3d(nbin, 2, nnode);
    // one scratch block per thread (the thread count of this rebuild;
    // raising OMP_NUM_THREADS afterwards requires an Ntable bump)
    a_tab    = (double***) malloc3d(omp_get_max_threads(), 6, nnode);

    // GL rule mapped once onto [ln M_min, ln M_max], shared by all bins
    const double lnMmin = log(limits.halo_m_min);
    const double lnMmax = log(limits.halo_m_max);

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

    // a grid of bin l over its lens range; node i sits at
    // lim[l][0] + i lim[l][2], both ends included
    for (int l=0; l<nbin; l++) {
      lim[l][0] = amin_lens(l);
      lim[l][1] = amax_lens(l);
      lim[l][2] = (lim[l][1] - lim[l][0])/((double) na - 1.);
    }

    // ln k grid, shared by all bins
    lim[nbin][0] = log(limits.k_min_cH0);
    lim[nbin][1] = log(limits.k_max_cH0);
    lim[nbin][2] = (lim[nbin][1]-lim[nbin][0])
                   /((double) Ntable.N_k_nlin - 1.);
  }

  // --- 2. REFILL: THE HOD-WEIGHTED ln P TABLE ---

  // any of the four tags differs from the table's
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], Ntable.random)    ||
      fdiff2(cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(cache[3], redshift.random_clustering))
  {
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
       1. mass_tab, per mass node: M, weight, nu at D = 1, r_Delta,
          weighted dn/dlnM factor
       2. occ_tab, per (bin, mass node): occupation N_s and f_c N_c
       3. per a row, threaded: D(a), Tinker f(nu) half, n_gal, b_gal
       4. thread scratch a_tab, per node of one a row: c_g, r_s,g,
          logs, weights W2, W1
       5. per k: G02 = sum_q ug (W2 ug + W1); table = ln P_gg */

    // --- 2b. PER MASS NODE, SERIAL ---

    // the a- and bin-independent factors; the sigma2 and dlognudlogm
    // reads happen here, before the threads
    const double rho_m     = cosmology.rho_crit * cosmology.Omega_m;
    const double rho_delta = Delta * rho_m;

    // mass_tab rows 2-4: nu at D = 1 | r_Delta from
    // M = (4 pi/3) Delta rho_m r_Delta^3 | weight (rho_m/M) dlnnu/dlnM
    for (int q=0; q<nnode; q++) {
      const double m = mass_tab[0][q];
      mass_tab[2][q] = delta_c/sqrt(sigma2(m));
      mass_tab[3][q] = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
      mass_tab[4][q] = mass_tab[1][q]*(rho_m/m)*dlognudlogm(m);
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

        // growth, the nu-independent Tinker f(nu) half, and the mean
        // galaxy density and bias of this a row
        const double     D        = growfac(ai);
        const fnu_params fnu_pars = fnu_params_at(ai);
        const double     n_gal    = ngal(l, ai);
        const double     b_gal    = bgal(l, ai);

        // thread-private scratch: this (bin, a-row) iteration fills
        // it and consumes it in its own k loop; restrict: each row is
        // reached only through its pointer, no reload after libm calls
        double** const wsp = a_tab[omp_get_thread_num()];
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
          const double nu = mass_tab[2][q]/D;

          // c_g = gc c, with m(c_g) = ln(1+c_g) - c_g/(1+c_g)
          const double cg    = conc(m, D)*gc;
          const double ln1cg = log1p(cg);
          const double mcg   = ln1cg - cg/(1.0 + cg);

          // dn_halo = quadrature weight x dn/dlnM (Tinker); n_sat = N_s
          const double dn_halo = mass_tab[4][q]*fnu_core(nu, &fnu_pars)*nu;
          const double n_sat   = occ_tab[l][0][q];

          conc_gal[q] = cg;
          ln1c_gal[q] = ln1cg;
          r_sg[q]     = mass_tab[3][q]/cg;
          lnrsg[q]    = log(r_sg[q]);
          w2[q]       = dn_halo*(n_sat/mcg)*(n_sat/mcg);
          w1[q]       = 2.0*dn_halo*(n_sat/mcg)*occ_tab[l][1][q];
        }

        // per k: G02 as a sum of the NFW kernel over the nodes, then
        // ln P
        for (int j=0; j<Ntable.N_k_nlin; j++) {
          const double lnk = lim[nbin][0] + j*lim[nbin][2];
          const double kj  = exp(lnk);

          double g02 = 0.0;
          for (int q=0; q<nnode; q++) {
            const double ug = nfw_um(conc_gal[q], kj*r_sg[q],
                                     lnk + lnrsg[q], ln1c_gal[q]);
            g02 += ug*(w2[q]*ug + w1[q]);
          }

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
     1. nu = delta_c/(sigma(M) D(a)); occupation <N|M> = f_c N_c + N_s
     2. dn_gal = weight (rho_m/M) dlnnu/dlnM <N|M> f(nu) nu, the node's
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
  const double lnMmax = log(limits.halo_m_max);

  // --- 2. COSMOLOGY AND HOD FACTORS AT THIS SCALE FACTOR ---

  const double rho_m = cosmology.rho_crit * cosmology.Omega_m;
  const double D     = growfac(a);

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
    const double nu         = delta_c/(sqrt(sigma2(m))*D);
    const double occupation = fc*HOD_nc(m, a, ni) + HOD_ns(m, a, ni);

    // the node's share of the galaxy number density:
    // dn_gal = weight x dn/dlnM x <N|M>
    const double dn_gal =
        weight*(rho_m/m)*dlognudlogm(m)*occupation*fnu_core(nu, &fnu_pars)*nu;

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
