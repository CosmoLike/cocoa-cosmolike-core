#include <assert.h>
#include <gsl/gsl_sf.h>
#include <complex.h>
#include <math.h>
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
//
// Quadrature: Gauss-Legendre rules only in sizes GSL tabulates (the
// "predefined GSL tables" of cosmo2D.c; GSL computes any other size on
// the fly, with weights good to only ~5e-7). The gas integrals of u_KS
// ladder 96/128/256/512/1024 with Ntable.high_def_integration and
// bias_norm 128/256/512; both are converged at their base size. The
// halo-model mass integrals (ngal, hm_funcs, I02_XY, I11_X, G02,
// GM02) run at 1024, the largest tabulated size, at every hdi: their
// integrands read sigma2 and dlognudlogm by linear interpolation in
// ln M, and GL converges only algebraically across those kinks (at
// 256 nodes p_gg moves by up to 2.4e-3 from its 1024-node value).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Glossary: the function names of this file follow the original
// cosmolike and are terse. Each one maps to a halo-model quantity:
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
//                  coefficients (once per a) and the nu-dependent
//                  remainder (once per node)
//
// Profiles, in Fourier space ("u" = a profile transform):
//
//   u_nfw_c      = u(k|M) of the NFW profile, given its concentration c
//   u_c          = u(k|M) of the halo matter profile (selects u_nfw_c)
//   u_g          = u_g(k|M), the satellite-galaxy profile
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
//   u_y_bnd      = W_p(M, k), the bound-gas electron-pressure window
//                  ("y": the Compton-y, thermal-SZ, field it sources)
//   u_y_ejc      = the ejected-gas electron-pressure window
//   n_s_cmv      = comoving number density of source galaxies (no
//                  caller; aborts if run, see its header)
//
// Halo occupation distribution, HOD (Zehavi et al. 2011, 1005.2413):
//
//   HOD_nc       = <N_c|M>, mean number of central galaxies
//   HOD_ns       = <N_s|M>, mean number of satellite galaxies
//   HOD_fc       = f_c, completeness factor of the centrals
//   ngal         = n_g, comoving galaxy number density
//   bgal         = b_g, number-weighted mean galaxy bias
//   mmean        = <M>, mean halo mass of the galaxies
//   fsat         = f_sat, satellite fraction
//   int_hm_funcs = the integrand ngal, mmean, fsat and bgal share
//                  ("hm": halo model)
//   hm_funcs     = the dispatcher behind ngal, mmean, fsat and bgal:
//                  func = 0, 1, 2, 3 in that order, the last three
//                  divided by n_g
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
//   I02_XY       = I^0_2: no bias, two profiles  -> the 1-halo term of
//                  P_XY
//   I11_X        = I^1_1: linear bias, one profile -> the 2-halo
//                  amplitude, P_2h = I11_X I11_Y P_lin, plus the HMx
//                  term for the halos below M_min (I11_X_nointerp)
//   G02          = the galaxy-galaxy 1-halo integral (central-satellite
//                  and satellite-satellite pairs); p_gg divides it by
//                  n_g^2
//   GM02         = the galaxy-matter 1-halo integral; p_gm divides it
//                  by n_g
//
// Spectra: p_XY(k, a) with X, Y in {m = matter, y = electron pressure,
// g = galaxies}: p_mm, p_my, p_yy, p_gm, p_gg.
//
// Suffixes:
//
//   *_nointerp   = direct computation at one point (no table)
//   *_work       = batched computation over many inputs
//   int_for_*,
//   int_*        = integrand of a mass or radius integral
//   (no suffix)  = the cached table, read by interpolation
// ---------------------------------------------------------------------------

#define delta_c 1.686
#define Delta 200

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// BASIC PEAK BACKGROUND SPLIT ROUTINES
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
// integral I11_X_nointerp adds the rest, 1 - bias_norm, back as halos of
// mass M_min (its header, item 2). conc gives the NFW concentration as a
// function of nu.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tinker et al. 2010 kernels, split for batched evaluation.
//
// hb1nu and fnu run at every node of every halo-model mass integral,
// yet most of their arithmetic does not depend on nu: the bias
// constants depend only on Delta = 200, the mass-function parameters
// only on a. Each kernel is therefore split in two:
//
//   *_params(a)      = everything independent of nu (once per call, or
//                      once per scale factor inside a batched loop)
//   *_core(nu, par)  = the nu-dependent remainder (once per node)
//
// hb1nu(nu, a) and fnu(nu, a) below are exactly params + core, with the
// same arithmetic in the same order, so a batched caller and a scalar
// caller get bitwise-identical values. The *_params functions dispatch
// on like.halo_model; the cores are the Tinker 2010 forms, the only
// models implemented.
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
// Shape: b -> 1 as nu -> 0, but slowly because a is small (b ~ 0.6 at
// nu = 0.2); b ~ 1 near nu = 1; the C nu^2.4 term takes over for rare,
// massive halos (b ~ 5 at nu = 3).
//
// No redshift dependence: the fit combines all outputs 0 <= z <= 2.5
// and finds no significant evolution at a given nu (1001.3162
// sec. 3.1). The scale factor stays in the signature so that a
// z-dependent fit could use it.
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
      // Table 2 of 1001.3162 at y = log10(Delta) = log10(200)
      // = 2.30102999566398119521. Every coefficient is a function of
      // that constant, so each is a constant too, written to 21
      // significant digits (mpmath at 40 digits):
      //
      //   ALPHA = A = 1 + 0.24 y exp[-(4/y)^4]
      //   pa    = a = 0.44 y - 0.88
      //   dca   = delta_c^a, delta_c = 1.686
      //   GAMMA = C = 0.019 + 0.107 y + 0.19 exp[-(4/y)^4]
      //
      // Literals, not expressions: the default build compiles with
      // -frounding-math, which forbids the compiler from folding an
      // inexact constant expression (its value would depend on the
      // run-time rounding mode), so log10/exp/pow of constants would
      // run on every call.
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
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// fnu = f(nu): the halo multiplicity function, i.e. the mass function
// per unit peak height nu, of Tinker et al. 2010 (1001.3162 Eq. 8):
//
//   f(nu) = alpha [1 + (beta nu)^(-2 phi)] nu^(2 eta) exp(-gamma nu^2/2)
//
// It turns into the halo mass function through
//
//   dn/dM = f(nu) (rho_m/M) dnu/dM
//     ->  dn/dlnM = (rho_m/M) nu f(nu) dln nu/dln M
//
// so nu f(nu) is the g(sigma) of Tinker et al. 2008 (1001.3162 sec. 4).
//
// 1. The four shape parameters
//
// 1001.3162 Table 4 gives the z = 0 values at Delta = 200 (mean
// density),
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
// Shape of f. At small nu the bracket tends to 1 (-2 phi = 1.46 > 0)
// and f is a power law that grows toward light halos,
//
//   f -> alpha nu^(2 eta) = alpha nu^-0.49   (z = 0);
//
// at large nu the Gaussian exp(-gamma nu^2/2) makes massive halos
// (nu > 2) exponentially few.
//
// The paper recommends the z = 3 parameters beyond z = 3 (text after
// Eq. 12): fnu_params_at evaluates everything at aa = max(a, 0.25).
//
// 2. The amplitude alpha
//
// alpha is not fitted. The paper fixes it at each z through the
// consistency relation of the peak-background split (1001.3162 Eq. 7),
//
//   int_0^inf b(nu) f(nu) dnu = 1,
//
// with b(nu) the linear halo bias of hb1nu. In words: f(nu) dnu is the
// fraction of matter in halos of peak height nu, b(nu) is how strongly
// they cluster, and the matter-weighted bias of all halos is the bias
// of matter with respect to itself, 1.
//
// This is what makes the 2-halo term P_2h = I11_m^2 P_lin tend to P_lin
// as k -> 0 (bias_norm header, item 1).
//
// Since alpha factors out of f, Eq. 7 gives it directly:
//
//   alpha(a) = 1 / int_0^inf b(nu) ftilde(nu; a) dnu,
//   ftilde   = f with alpha = 1 (the four parameters of item 1).
//
// tinker_alpha tabulates this integral in aa (its header covers the
// numerics) and fnu_params_at reads the table:
//
//   z        0         0.5       1         2         >= 3
//   alpha    0.36840   0.33951   0.31722   0.28171   0.25197
//
// Table 4 lists 0.368 at z = 0. The other natural normalization,
// int f dnu = 1 (all matter in halos), is not imposed and holds only
// approximately: 1.0012 at z = 0, 1.048 at z = 1, 1.118 for z >= 3.
// Nothing in this file divides by it.
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
  double alpha; // amplitude: 1 in fnu_shape, Eq. 7 via tinker_alpha
  double beta;  // Eqs. 9-12 + Table 4 of 1001.3162 at Delta = 200
  double gamma; //   (fnu header, item 1)
  double phi;
  double eta;
} fnu_params;

// The four shape parameters of Eq. 8 at aa (Eqs. 9-12) with alpha = 1:
// the ftilde of the Eq. 7 integral. No clamp on aa: tinker_alpha calls
// this a little outside [0.25, 1] while it builds its table (its
// header, item 4); fnu_params_at clamps.
static inline fnu_params fnu_shape(
    const double aa  // scale factor of the Tinker evolution (no clamp)
  )
{
  fnu_params p;
  p.alpha = 1.0;
  p.beta  = 0.589 * pow(aa, -0.2);
  p.gamma = 0.864 * pow(aa, 0.01);
  p.phi   = -0.729 * pow(aa, .08);
  p.eta   = -0.243 * pow(aa, -0.27);
  return p;
}

// Eq. 8 itself: the nu-dependent remainder, one evaluation per
// quadrature node. p comes from fnu_params_at (alpha from the table)
// or, inside the tinker_alpha build, from fnu_shape (alpha = 1).
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
// factor aa (fnu header, item 2),
//
//   alpha(aa) = 1/I(aa),   I(aa) = int_0^inf b(nu) ftilde(nu; aa) dnu,
//
// with b the Tinker bias (hb1nu_core) and ftilde the Eq. 8 shape at
// alpha = 1 (fnu_shape). The caller gets alpha by linear interpolation
// from a table on ND = 4096 nodes uniform in aa over [0.25, 1], built
// once per process.
//
// 1. Why a table
//
// alpha is an integral (a 936-term sum, item 2), and fnu needs it at
// every node of every halo-model mass integral: 1024 nodes per
// integral, several integrals per k and a. alpha depends on aa alone,
// so the sum is done once on a grid in aa and every call is a lookup.
//
// 2. The exact integral: a trapezoid rule in ln nu
//
// The trapezoid rule joins the integrand values at equally spaced
// nodes s_q = s_0 + q h with straight lines and adds the areas:
//
//   int_{s_0}^{s_N} g(s) ds  ~  h [g_0/2 + g_1 + ... + g_{N-1} + g_N/2].
//
// Its error is h^2 in general, but the error formula is built from the
// integrand's derivatives at the two ends alone: when those are
// negligible, the rule is far more accurate than h^2 suggests.
//
// The integrand has that property in s = ln nu,
//
//   nu = e^s,   dnu = nu ds   ->   I = int b(nu) ftilde(nu) nu ds.
//
// Toward small nu, b -> 1 and
//
//   nu ftilde ~ nu^(1 + 2 eta),   1 + 2 eta = 0.29 (aa = 0.25)
//                                            to 0.51 (aa = 1),
//
// so the integrand falls as e^(0.29 s) or faster as s -> -inf. Toward
// large nu the exp(-gamma nu^2/2) of ftilde kills it.
//
// The grid
//
//   s = SMIN .. SMAX = -90 .. 3.5 in steps DS = 0.1:  NS = 936 nodes
//
// covers nu = 8e-40 to 33. The tail dropped below s = -90 is
//
//   int_{-inf}^{-90} e^(0.29 s) ds = e^-26/0.29 = 1e-11,
//
// 3e-12 of I ~ 4 at aa = 0.25 and less at larger aa, where the power
// is steeper; at s = 3.5 the Gaussian factor is e^-474.
//
// Check against the same sum at DS = 0.01 over [-200, 5]: the 936-node
// sum is exact to 2.9e-12 for every aa in [0.25, 1] (worst at
// aa = 0.25, all of it the dropped tail); DS = 0.05 reproduces it to
// 2e-15.
//
// Everything except ftilde is the same for every aa, so the build
// folds it into one weight per node,
//
//   bw[q] = w_q nu_q b(nu_q),   w_q = DS (DS/2 at the two ends),
//
// with nu_q the Jacobian of dnu = nu ds. One aa node then costs one
// plain sum, I = sum_q bw[q] ftilde(nu_q; aa).
//
// 3. Coarse exact nodes, cubic upsampling, dense linear reads
//
// The house pattern for a smooth function of one variable (sigma2 in
// cosmo3D.c does the same in ln M):
//
//   exact nodes   NC = 128 uniform in aa on [0.25, 1], spacing
//                 hc = 0.75/127 = 0.0059, plus PAD = 6 nodes beyond
//                 each end: NE = 140 exact integrals, aa from 0.2146
//                 (aa0) to 1.0354
//   dense nodes   ND = 4096 uniform on [0.25, 1], spacing 1.8e-4
//   read          interpol1d, linear between the two dense nodes
//                 around aa
//
// Linear reads, because every table in this code base is read that
// way (one index computation, one multiply-add). A spline rather than
// 4096 exact integrals, because a cubic through exact values is far
// more accurate than the linear reads it feeds (item 5).
//
// The spline. A natural cubic spline is a chain of cubics, one per
// interval, with value, first and second derivative continuous at
// every node and S'' = 0 at the two end nodes. spline_coeffs_uniform
// (basics.c) solves the tridiagonal system for c_j = S''(x_j)/2; the
// piece on the interval starting at node j, in Horner form, is
//
//   S(x_j + t) = y_j + t (b + t (c_j + t d)),       0 <= t <= hc,
//   b = (y_{j+1} - y_j)/hc - hc (c_{j+1} + 2 c_j)/3,
//   d = (c_{j+1} - c_j)/(3 hc),
//
// as in sigma2. The spline runs through alpha = 1/I, the quantity read
// back, not through I.
//
// 4. Why the padding
//
// S'' = 0 at the end nodes is wrong for alpha, which is curved there:
// alpha''(0.25) = -2.9. A spline pinned at aa = 0.25 misses alpha by
// 2e-5 (relative) in the first interval.
//
// The tridiagonal rows
//
//   c_{j-1} + 4 c_j + c_{j+1} = rhs_j
//
// damp a disturbance by 2 - sqrt(3) = 0.268 per interval, so with
// PAD = 6 extra nodes on each side only 0.268^6 = 4e-4 of that error
// reaches aa = 0.25. This is why the build calls fnu_shape at
// aa0 = 0.2146 and up to 1.0354, and why fnu_shape carries no clamp.
//
// 5. Cost and accuracy
//
// Build: 140 sums of 936 terms, threaded over the nodes, then one
// tridiagonal solve and 4096 cubic evaluations: 0.9 ms with 4 threads.
// Read: one interpol1d.
//
// Accuracy against the exact Eq. 7 integral at 997 values of a:
// maximum relative error 4.7e-8 (at a = 0.2545), median 1.4e-9. The
// linear read sets this floor: it misses a curved function by up to
//
//   alpha'' dx^2/8 = 2.9 x (1.8e-4)^2/8 = 1.2e-8   near aa = 0.25,
//
// 5e-8 of alpha = 0.25. Exact values on the same 4096 nodes, read
// linearly, give 4.6e-8; the spline alone is good to 8.5e-9.
//
// Cache invalidation:
//   rebuilt when like.halo_model[0] or [1] (the mass-function and the
//   bias fit, whose forms and constants the integrand is made of)
//   differ from the pair the table holds: once per process in practice.
//   Nothing else enters: not the cosmology (nu is the integration
//   variable, sigma2 never appears) and not Ntable.
//
// Thread safety: the first call builds the table and must run outside
// any parallel region (the warm-up rule of the cosmo2D.c _work
// functions). bias_norm's refill calls fnu(1.0, agrid[0]) serially
// before its loop; the mass integrals reach fnu through their init = 1
// calls.
//
// Parameters:
//   aa - scale factor of the Tinker evolution, max(a, 0.25), in
//        [0.25, 1)
//
// Returns:
//   alpha(aa), dimensionless; interpol1d returns the end values for aa
//   outside [0.25, 1]
// ---------------------------------------------------------------------------
static double tinker_alpha(
    const double aa  // max(a, 0.25), in [0.25, 1)
  )
{
  // Static state, kept between calls; table NULL and key {-1, -1} make
  // the first call build.
  static int key[2] = {-1, -1};  // like.halo_model[0..1] of the table
  static double* table = NULL;   // [ND] alpha on the dense aa grid
  static double lim[3];          // aa_min, aa_max, dense spacing
  const int ND = 4096;           // dense lookup nodes on [0.25, 1]

  // Build block (header, items 2-4): the first call, or a change of the
  // fits in use. Each like.halo_model entry has one accepted value, so
  // in practice once per process.
  if (NULL == table ||
      key[0] != like.halo_model[0] ||
      key[1] != like.halo_model[1])
  {
    // Coarse exact grid: NC nodes on [0.25, 1] plus PAD beyond each
    // end; node i sits at aa0 + i hc (header, items 3-4).
    const int NC = 128;          // exact nodes on [0.25, 1]
    const int PAD = 6;           // exact padding nodes beyond each end
    const int NE = NC + 2*PAD;
    const double hc = 0.75/((double) NC - 1.0);
    const double aa0 = 0.25 - PAD*hc;

    // Trapezoid nodes in s = ln nu (header, item 2); lround keeps a
    // last-digit rounding of the division from losing a node: NS = 936.
    const double SMIN = -90.0;   // trapezoid in s = ln nu
    const double SMAX = 3.5;
    const double DS = 0.1;
    const int NS = (int) lround((SMAX - SMIN)/DS) + 1;
    double* nus = (double*) malloc(sizeof(double)*NS);
    double* bw  = (double*) malloc(sizeof(double)*NS);
    // bw = w_q nu_q b(nu_q): trapezoid weight, Jacobian of dnu = nu ds,
    // Tinker bias. The bias fit does not evolve, so hb1nu_params_at
    // takes any a; 1.0 is a placeholder.
    const hb1nu_params pb = hb1nu_params_at(1.0);
    for (int q=0; q<NS; q++) {
      nus[q] = exp(SMIN + q*DS);
      const double wq = (0 == q || NS - 1 == q) ? 0.5*DS : DS;
      bw[q] = wq*nus[q]*hb1nu_core(nus[q], &pb);
    }
    // Exact alpha at the NE coarse nodes, ye[i] = 1/I(aa_i). restrict:
    // nus and bw are reached only through nq and wb, so the compiler
    // need not reload them after each store to ye[i]. One serial sum
    // per node inside one thread: the values do not depend on the
    // thread count.
    double* ye = (double*) malloc(sizeof(double)*NE);
    double* ce = (double*) malloc(sizeof(double)*NE);
    const double* restrict nq = nus;
    const double* restrict wb = bw;
    #pragma omp parallel for schedule(static)
    for (int i=0; i<NE; i++) {
      const fnu_params p = fnu_shape(aa0 + i*hc);
      double sum = 0.0;
      for (int q=0; q<NS; q++) {
        sum += wb[q]*fnu_core(nq[q], &p);
      }
      ye[i] = 1.0/sum;
    }
    // Natural cubic spline through the NE exact values (header, item
    // 3): ce[j] = S''(aa_j)/2, zero at the two padding ends.
    spline_coeffs_uniform(ye, NE, hc, ce);

    // Dense grid: ND nodes uniform on [0.25, 1], both ends included,
    // spacing lim[2] = 0.75/4095 = 1.8e-4.
    if (table != NULL) free(table);
    table = (double*) malloc(sizeof(double)*ND);
    lim[0] = 0.25;
    lim[1] = 1.0;
    lim[2] = (lim[1] - lim[0])/((double) ND - 1.0);
    // Upsampling: r = distance of dense node i from aa0 in coarse
    // spacings, j = its coarse interval, t = offset from coarse node j
    // in aa; then the Horner form of header item 3. The last dense node
    // has r = 133 < NE - 2, so the clamp on j is a guard only.
    for (int i=0; i<ND; i++) {
      const double r = (lim[0] + i*lim[2] - aa0)/hc;
      int j = (int) r;
      if (j > NE - 2) {
        j = NE - 2;
      }
      const double t = (r - j)*hc;
      const double b = (ye[j+1] - ye[j])/hc - hc*(ce[j+1] + 2.0*ce[j])/3.0;
      const double d = (ce[j+1] - ce[j])/(3.0*hc);
      table[i] = ye[j] + t*(b + t*(ce[j] + t*d));
    }
    free(nus);
    free(bw);
    free(ye);
    free(ce);
    key[0] = like.halo_model[0];
    key[1] = like.halo_model[1];
  }
  // Read-out: interpol1d is the house linear interpolation on a uniform
  // grid (index by arithmetic, no search); it returns the end values
  // outside [lim[0], lim[1]]. Example: z = 0.7, aa = 1/1.7 = 0.5882
  // gives r = 0.3382/1.8315e-4 = 1846.76, read 76% of the way from
  // node 1846 to node 1847.
  return interpol1d(table, ND, lim[0], lim[1], lim[2], aa);
}

static inline fnu_params fnu_params_at(
    const double a  // scale factor, 0 < a < 1 (aborts otherwise)
  )
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  fnu_params p;
  switch(like.halo_model[0])
  {
    case HMF_TINKER_2010:
    { // Eqs. 8-12 + Table 4 of 1001.3162 (fnu header, item 1).
      // aa = max(a, 0.25) freezes the evolution at z = 3 (a = 0.25), as
      // the paper recommends beyond z = 3. Shape and amplitude are both
      // taken at aa, so Eq. 7 keeps holding at z > 3.
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
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Halo concentration c = r_Delta/r_s of the NFW profile, Bhattacharya et
// al. 2013 (1112.5479 Table 2: full halo sample, column Delta = 200
// rho_b, i.e. 200 times the mean density - the halo definition of this
// file):
//
//   c(M, z) = 9.0 nu^-0.29 D(z)^1.15,   nu = delta_c/(sigma(M) D(z))
//
// In nu the relation keeps its shape at all redshifts (1112.5479
// sec. 4.2); at a given nu its amplitude falls with the growth factor,
// and at a given mass the c-M relation flattens toward high z
// (1112.5479 sec. 4.1).
//
// Calibration: z = 0-2 and group-to-cluster masses, with the paper's nu
// built on delta_c = 1.673 (its reference cosmology) where this file
// uses 1.686, a -0.2% shift in c. The halo model evaluates the fit over
// the whole [limits.halo_m_min, limits.halo_m_max] and at every z, so
// also in extrapolation.
//
// Parameters:
//   m         - halo mass in M_sun/h
//   growfac_a - linear growth factor D(a), D(1) = 1 (not the scale
//               factor itself)
//
// Returns:
//   c, dimensionless. like.halo_model[2] selects the fit;
//   CONCENTRATION_BHATTACHARYA_2013 is the only option.
// ---------------------------------------------------------------------------
double conc(
    const double m,         // halo mass in M_sun/h
    const double growfac_a  // growth factor D(a), not a itself
  )
{
  double ans;
  switch(like.halo_model[2])
  {
    case CONCENTRATION_BHATTACHARYA_2013:
    { // Bhattacharya et al. 2013, Delta = 200 rho_{mean} (Table 2)
      const double nu = delta_c/(sqrt(sigma2(m))*growfac_a);
      ans =  9.0*pow(nu, -0.29)*pow(growfac_a, 1.15); 
      break;
    }
    default:
    {
      log_fatal("like.halo_model[2] = %d not supported", like.halo_model[2]);
      exit(1);  
    }
  }
  return ans;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

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
// limits.halo_m_min and limits.halo_m_max (1e6 and 1e17 M_sun/h by
// default).
//
// The caller gets it by linear interpolation from a table in a
// (item 4), refilled once per cosmology.
//
// 1. Why the 2-halo term needs it
//
// The 2-halo term of the matter power spectrum, P_2h = I11_m^2 P_lin
// (file glossary), tends to P_lin as k -> 0 because matter is unbiased
// with respect to itself: int b f dnu = 1 over all nu (fnu header,
// item 2). The mass integrals of this file, however, run only over
//
//   M in [limits.halo_m_min, limits.halo_m_max] = [1e6, 1e17] M_sun/h
//
// by default, and the two ends of the range cost very different
// amounts.
//
// The heavy end loses very little. At large nu
//
//   f ~ exp(-gamma nu^2/2),   gamma = 0.864 at z = 0,
//
// and 1e17 M_sun/h is far above the most massive clusters.
//
// The light end loses a lot. At small nu (fnu header, item 1)
//
//   f ~ nu^(2 eta),   eta = -0.243 at z = 0   ->   f ~ nu^-0.49,
//
// so f grows toward light halos, and the halos below 1e6 M_sun/h hold
// about 0.2 of the integral at z = 0 for a Planck-like cosmology.
//
// The table therefore holds (default limits, Planck-like cosmology)
//
//   z            0      0.5    1      2      3
//   bias_norm    0.80   0.80   0.79   0.75   0.70
//
// and without a correction the 2-halo term would miss its limit:
//
//   I11_m(k -> 0) = bias_norm = 0.8,   P_2h -> 0.64 P_lin   (z = 0).
//
// The fix lives in I11_X_nointerp, not here: it adds the missing
// 1 - bias_norm(a) back as halos of mass exactly M_min (its header,
// item 2). This function measures the shortfall and nothing else; the
// integrand of int_for_I11_X carries no bias_norm factor.
//
// 2. One integration domain for every a
//
// Both limits depend on a through the same factor 1/D(a). With
//
//   t = nu D(a),   dnu = dt/D
//
// (t is the peak height the same halo has at a = 1) the domain loses
// its a-dependence:
//
//   bias_norm(a) = (1/D) int_{t_min}^{t_max} b(t/D) f(t/D, a) dt,
//
//   t_min = delta_c/sigma(M_min)   (light halos: sigma large, t small)
//   t_max = delta_c/sigma(M_max)   (heavy halos: sigma small, t large)
//
// Example: a halo with t = 3 has nu = 3 at a = 1 and, at a = 0.5 where
// D = 0.61 (flat, Omega_m = 0.3), nu = 3/0.61 = 4.9: the field was
// smoother then, so the same mass was a rarer peak.
//
// The map nu = t/D is linear, so one Gauss-Legendre rule on
// [t_min, t_max] serves every a, node by node: nu_q(a) = t_q/D. sigma2
// is read twice per refill (t_min, t_max), never inside the node loop,
// and the Jacobian is the exact constant 1/D.
//
// 3. The quadrature
//
// Gauss-Legendre (GL) with n nodes, int g(t) dt ~ sum_q w_q g(t_q), is
// exact for every polynomial of degree <= 2n - 1 and converges
// exponentially on a smooth integrand (sigma2 header in cosmo3D.c,
// item 3). GSL stores x_q, w_q on [-1, 1]; stretched onto
// [t_min, t_max] with dt = h dx,
//
//   bias_norm(a) = (h/D) sum_q w_q b(nu_q) f(nu_q, a),
//
//   nu_q = (m + h x_q)/D    (the node mapped to [t_min, t_max], then /D)
//   m    = (t_max + t_min)/2
//   h    = (t_max - t_min)/2
//
// The integrand is powers and one exponential, smooth over the whole
// interval (sigma2 enters only through t_min and t_max), so GL
// converges fast here, unlike the mass integrals of this file, whose
// integrands read tables with kinks (file header). The node count
// ladders with hdi = abs(Ntable.high_def_integration):
//
//   hdi      0     1     >= 2
//   nodes    128   256   512
//
// 128 nodes are converged to 3e-15 relative, about a dozen units in
// the last place of a double.
//
// 4. The table
//
// I11_X_nointerp calls bias_norm(a) once per (k, a) node of every
// halo-model spectrum table, inside its threaded fill loop (256 x 512
// calls per refill of p_mm alone). bias_norm depends on a alone, so
// the sum is done once per cosmology on a grid in a:
//
//   Ntable.N_a nodes uniform in a from limits.a_min to 0.9999999, both
//   ends included; by default 256 nodes from 1/41 = 0.02439 (z = 40)
//   in steps of 0.003826.
//
// The top stays below 1 because fnu_params_at aborts unless 0 < a < 1.
// interpol1d returns the end values outside the grid, so a = 1 gets
// the value at 0.9999999.
//
// Thread safety: the sigma2 and tinker_alpha tables are built lazily
// by their first caller, so the refill touches both serially before
// the parallel loop (the warm-up rule of the cosmo2D.c _work
// functions; growfac has no static state). Each entry is its own sum
// inside one thread, so the result does not depend on the thread count.
//
// Cache invalidation:
//   allocation, the a-grid and the Gauss-Legendre nodes: rebuilt when
//     Ntable.random changes (hdi enters the node count)
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
  // Static state, kept between calls and zeroed at program start, so
  // the first call builds everything. Two blocks write it:
  //   Ntable rebuild block   table/agrid/lim and the node cache
  //                          nq/xg/wg (every malloc lives there)
  //   refill block           the values in table, then the tags cache[]
  static uint64_t cache[MAX_SIZE_ARRAYS]; // [0] cosmology, [1] Ntable tag
  static double* table = NULL;  // [N_a] bias_norm on the a grid
  static double* agrid = NULL;  // [N_a] the a nodes
  static double lim[3];         // a_min, 0.9999999, spacing in a
  static int nq = 0;            // number of Gauss-Legendre nodes
  static double* xg = NULL;     // [nq] Gauss-Legendre nodes on [-1, 1]
  static double* wg = NULL;     // [nq] Gauss-Legendre weights on [-1, 1]

  // Ntable rebuild block: the first call, or Ntable.random (the tag of
  // the Ntable settings) differs from the one the table was built with.
  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    // The a grid (header, item 4): N_a nodes uniform in a from
    // limits.a_min to 0.9999999, both ends included.
    if (table != NULL) free(table);
    if (agrid != NULL) free(agrid);
    table = (double*) malloc(sizeof(double)*Ntable.N_a);
    agrid = (double*) malloc(sizeof(double)*Ntable.N_a);
    lim[0] = limits.a_min;
    lim[1] = 0.9999999;
    lim[2] = (lim[1] - lim[0]) / ((double) Ntable.N_a - 1.0);
    for (int i=0; i<Ntable.N_a; i++) {
      agrid[i] = lim[0] + i*lim[2];
    }

    // Node cache: Gauss-Legendre nodes and weights on [-1, 1] (header,
    // item 3). They depend on hdi only, so they live here and the
    // refill only reads them; the stretch onto [t_min, t_max] happens
    // there. malloc_gslint_glfixed(n) wraps the GSL table of n nodes;
    // gsl_integration_glfixed_point copies node q and its weight out.
    if (xg != NULL) {
      free(xg);
      free(wg);
    }
    const int hdi = abs(Ntable.high_def_integration);
    nq = (0 == hdi) ? 128 :
         (1 == hdi) ? 256 : 512; // predefined GSL tables
    xg = (double*) malloc(sizeof(double)*nq);
    wg = (double*) malloc(sizeof(double)*nq);
    gsl_integration_glfixed_table* t = malloc_gslint_glfixed(nq);
    for (int q=0; q<nq; q++) {
      gsl_integration_glfixed_point(-1.0, 1.0, q, &xg[q], &wg[q], t);
    }
    gsl_integration_glfixed_table_free(t);
  }
  // Refill block: the cosmology tag or the Ntable tag differs from the
  // one the table holds.
  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random)) {
    // Warm-up (header, Thread safety): the first fnu call builds the
    // tinker_alpha table and the first sigma2 call (below) builds the
    // sigma2 table, so both happen here, on one thread, before the
    // parallel loop. agrid[0] = 1/41 is clamped to aa = 0.25 inside;
    // the value f(1) is thrown away, which the (void) cast says.
    (void) fnu(1.0, agrid[0]);

    // t_min, t_max, m, h of header items 2-3: the a = 1 peak heights of
    // the two mass limits, then the midpoint and half-width of
    // [t_min, t_max]. Heavy halos have small sigma and large t, so
    // tmax > tmin and h > 0.
    const double tmin = delta_c/sqrt(sigma2(limits.halo_m_min));
    const double tmax = delta_c/sqrt(sigma2(limits.halo_m_max));
    const double m = 0.5*(tmax + tmin);
    const double h = 0.5*(tmax - tmin);

    // restrict: promises the compiler that xg and wg are reached only
    // through x and w, so it need not reload a node after each store to
    // table[i] or libm call (the loop body must index x and w for
    // this). n is a plain local copy of the node count.
    const double* restrict x = xg;
    const double* restrict w = wg;
    const int n = nq;

    // schedule(static): contiguous chunks of i per thread. Each entry's
    // sum is a serial loop inside one thread, so the result does not
    // depend on the thread count; table[i] is the only shared write,
    // one slot per iteration, and the body's locals are private.
    #pragma omp parallel for schedule(static)
    for (int i=0; i<Ntable.N_a; i++) {
      // Once per scale factor: D(a) and the nu-independent halves of
      // the two Tinker kernels (the params/core split of the section
      // banner), hoisted so that the node loop does only nu arithmetic.
      const double D = growfac(agrid[i]);
      const hb1nu_params pb = hb1nu_params_at(agrid[i]);
      const fnu_params pf = fnu_params_at(agrid[i]);
      // The node sum of header item 3: nu_q = (m + h x_q)/D, summand
      // w_q b(nu_q) f(nu_q, a).
      double sum = 0.0;
      for (int q=0; q<n; q++) {
        const double nu = (m + h*x[q])/D;
        sum += w[q]*hb1nu_core(nu, &pb)*fnu_core(nu, &pf);
      }
      // Jacobians: h for dt = h dx, 1/D for dnu = dt/D (header, item 2).
      table[i] = sum*h/D;
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
  // Read-out: interpol1d is the house linear interpolation on a uniform
  // grid; it returns the end values outside [lim[0], lim[1]], so a = 1
  // gets the value at 0.9999999. Example with the defaults, a = 0.5:
  // r = (0.5 - 0.02439)/0.003826 = 124.31, read 31% of the way from
  // node 124 to node 125.
  return interpol1d(table, Ntable.N_a, lim[0], lim[1], lim[2], a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Logarithmic slope d ln nu/d ln M of the peak height: the Jacobian that
// turns f(nu) dnu into dn/dlnM (see the section banner).
//
// Taking the log of nu = delta_c/(sqrt(sigma2(M)) D(a)):
//
//   ln nu = ln delta_c - ln D(a) - (1/2) ln sigma2(M)
//     ->  d ln nu/d ln M = -(1/2) d ln sigma2/d ln M
//
// delta_c and D(a) drop out, so one table at a = 1 serves every
// redshift (this assumes scale-independent growth, sigma(M, a) =
// sigma(M) D(a)). sigma falls with M, so the slope is positive; for a
// local power law P ~ k^n_eff it is (n_eff + 3)/6: about 0.05 for the
// lightest halos (n_eff near -3) and 0.3 for clusters.
//
// Table: Ntable.N_M nodes uniform in ln M over [ln limits.halo_m_min,
// ln limits.halo_m_max], each a symmetric difference of ln sigma2 over
// h = 0.05 in ln M. Near the edges the stencil is clipped to the mass
// range (one-sided there), because the sigma2 table clamps outside it
// and would flatten the slope. Read back with linear interpol1d;
// constant extrapolation outside the range.
//
// Cache invalidation:
//   allocation and ln M limits: rebuilt when Ntable.random changes
//   table refill: cosmology.random (cache[0]) or Ntable.random (cache[1])
//
// Parameters:
//   M - halo mass in M_sun/h
//
// Returns:
//   d ln nu/d ln M, dimensionless and positive
// ---------------------------------------------------------------------------
double dlognudlogm(
    const double M  // halo mass in M_sun/h
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double* table = NULL;
  static double lim[3];

  if (NULL == table || fdiff2(cache[1], Ntable.random))
  {
    if (table != NULL) free(table);
    table = (double*) malloc(sizeof(double) * Ntable.N_M);
    lim[0] = log(limits.halo_m_min);
    lim[1] = log(limits.halo_m_max);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_M - 1.0);
  }

  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random))
  {
    (void) sigma2(exp(lim[0])); // build the sigma2 table before the threads
    // d ln nu/d ln M = -(1/2) d ln sigma2/d ln M: nu = delta_c/sigma,
    // and delta_c (with the growth factor) drops out of the log
    // derivative. Symmetric difference over h = 0.05 in ln M; at the
    // mass range's edges the stencil is pulled inside
    // [ln m_min, ln m_max], where the sigma2 table would otherwise
    // clamp to a constant and flatten the slope.
    #pragma omp parallel for schedule(static,1)
    for (int i=0; i <Ntable.N_M; i++)
    {
      const double h = 0.05;
      const double lo = fmax(lim[0] + i*lim[2] - h, lim[0]);
      const double hi = fmin(lim[0] + i*lim[2] + h, lim[1]);
      table[i] = -0.5*(log(sigma2(exp(hi))) - log(sigma2(exp(lo))))/(hi - lo);
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }  
  return interpol1d(table, Ntable.N_M, lim[0], lim[1], lim[2], log(M));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO PROFILES
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
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The NFW transform table (nfw_) and its kernel (nfw_um), at file scope so
// that two callers share them:
//
//   u_nfw_c   u(k|M) for one (c, k, M): computes r_Delta (a pow), ln(1+c)
//             (log1p) and ln x (log), calls nfw_um, divides by m(c)
//   p_mm      its rows call nfw_um directly with ln(1+c), r_s and ln r_s
//             computed once per (a, mass node) and reused over k, so the
//             innermost loop of the P_mm table does no pow, log1p or log
//
// The table holds f(t) and G(t) = g(t) + ln t of the u_nfw_c header at n
// nodes uniform in ln t over [NFW_TMIN, NFW_TASY]; reads clamp below
// NFW_TMIN and switch to the asymptotic series above NFW_TASY (nfw_um).
// ---------------------------------------------------------------------------
static const double NFW_TMIN = 1e-10; // reads clamp below NFW_TMIN
static const double NFW_TASY = 50.0;  // asymptotic series above NFW_TASY
static struct {
  uint64_t cache;        // Ntable.random of the table
  int n;                 // ln t nodes
  double lim[3];         // ln t axis: first, last, spacing
  double** tab;          // [2][n] f(t), G(t) = g(t) + ln t
} nfw_ = {0};

static void nfw_table(void)
{
  // Table build, once per Ntable setting: f, G at n nodes uniform in ln t,
  // exact from GSL Si, Ci (A&S 5.2.6-5.2.7 solved for f, g); threaded loop
  //   f = Ci sin t + (pi/2 - Si) cos t,   g = -Ci cos t + (pi/2 - Si) sin t
  if (NULL == nfw_.tab || fdiff2(nfw_.cache, Ntable.random)) {
    if (nfw_.tab != NULL) {
      free(nfw_.tab);
    }
    const int n = Ntable.halo_nfw_n;
    double** tab = (double**) malloc2d(2, n);
    double* lim = nfw_.lim;
    lim[0] = log(NFW_TMIN);
    lim[1] = log(NFW_TASY);
    lim[2] = (lim[1] - lim[0])/((double) n - 1.0);
    #pragma omp parallel for schedule(static)
    for (int i=0; i<n; i++) {
      const double s = lim[0] + i*lim[2];
      const double t = exp(s);
      const double si = gsl_sf_Si(t);
      const double ci = gsl_sf_Ci(t);
      tab[0][i] = ci*sin(t) + (M_PI_2 - si)*cos(t);      // f(t)
      tab[1][i] = -ci*cos(t) + (M_PI_2 - si)*sin(t) + s; // G(t) = g(t) + ln t
    }
    nfw_.n = n;
    nfw_.tab = tab;
    nfw_.cache = Ntable.random;
  }
}

// u m(c) of the NFW transform (u_nfw_c header) for one halo at one k.
// Inputs: c, x = k r_s, and the two logs the table axis needs, lx = ln x
// and l1c = ln(1 + c), supplied by the caller (an outer loop may hold
// them). Output: u m(c), dimensionless; the caller divides by m(c).
// nfw_table must have run (the build is not thread-safe; this read is).
// u_nfw_c and the p_mm rows both evaluate u through this one arithmetic.
static inline double nfw_um(
    const double c,   // concentration r_Delta/r_s
    const double x,   // k r_s
    const double lx,  // ln x
    const double l1c  // ln(1 + c)
  )
{
  const int n = nfw_.n;
  const double* lim = nfw_.lim;
  double** tab = nfw_.tab;
  const double lxu = lx + l1c;          // ln xu, xu = (1 + c) x
  const double xu = (1.0 + c)*x;

  // f(xu), G(x), G(xu): table reads up to NFW_TASY (clamped at NFW_TMIN);
  // above NFW_TASY the asymptotic series (A&S 5.2.34-35), in nested form,
  //   f(t) ~ (1 - 2!/t^2 + 4!/t^4 - 6!/t^6 + 8!/t^8)/t
  //   g(t) ~ (1 - 3!/t^2 + 5!/t^4 - 7!/t^6 + 9!/t^8)/t^2
  // (the factors 2, 12, 30, 56 and 6, 20, 42, 72 are ratios of consecutive
  // factorials; the first omitted terms 10!/t^10, 11!/t^10 are 4e-11 and
  // 4e-10 at t = 50). x < xu, so x may sit in the table when xu does not.
  double fu, Gx, Gu;
  if (xu <= NFW_TASY) {
    Gx = interpol1d(tab[1], n, lim[0], lim[1], lim[2], fmax(lx, lim[0]));
    Gu = interpol1d(tab[1], n, lim[0], lim[1], lim[2], fmax(lxu, lim[0]));
    fu = interpol1d(tab[0], n, lim[0], lim[1], lim[2], fmax(lxu, lim[0]));
  }
  else {
    const double v = 1.0/(xu*xu);
    fu = (1.0 - 2.0*v*(1.0 - 12.0*v*(1.0 - 30.0*v*(1.0 - 56.0*v))))/xu;
    Gu = v*(1.0 - 6.0*v*(1.0 - 20.0*v*(1.0 - 42.0*v*(1.0 - 72.0*v)))) + lxu;
    if (x <= NFW_TASY) {
      Gx = interpol1d(tab[1], n, lim[0], lim[1], lim[2], fmax(lx, lim[0]));
    }
    else {
      const double w = 1.0/(x*x);
      Gx = w*(1.0 - 6.0*w*(1.0 - 20.0*w*(1.0 - 42.0*w*(1.0 - 72.0*w)))) + lx;
    }
  }

  // Assembly of u m(c) = [g(x) - g(xu)] + 2 g(xu) sin^2(c x/2)
  //                      + [f(xu) - 1/xu] sin(c x);
  // Gx - Gu + l1c is g(x) - g(xu)
  const double gu = Gu - lxu;           // g(xu)
  const double sh = sin(0.5*c*x);
  return (Gx - Gu + l1c) + 2.0*gu*sh*sh + (fu - 1.0/xu)*sin(c*x);
}

// ---------------------------------------------------------------------------
// Normalized Fourier transform u(k|M) of the NFW profile truncated at
// r_Delta (astro-ph/0206508 Eq. 81), from a table of two smooth functions.
//
// The NFW profile (astro-ph/9611107) rho(r) = rho_s/[(r/r_s)(1 + r/r_s)^2],
// r_s = r_Delta/c, holds M = 4 pi rho_s r_s^3 m(c) inside r_Delta with
// m(c) = ln(1+c) - c/(1+c) (astro-ph/0206508 Eq. 76): the transform
// carries the prefactor 1/m(c). With
//
//   r_Delta = (3M/(4 pi Delta rho_m))^(1/3)   (comoving, c/H0)
//   x       = k r_Delta/c = k r_s,   xu = (1 + c) x
//
// Eq. 81 reads
//
//   u = { sin x [Si(xu) - Si(x)] - sin(c x)/xu
//         + cos x [Ci(xu) - Ci(x)] } / m(c)
//
// with Si, Ci the sine and cosine integrals. Check at k -> 0: the three
// terms tend to 0, -c/(1+c) and ln(1+c), so u -> 1. r_Delta and k are
// comoving (rho_m = rho_crit Omega_m): the scale factor never enters.
//
// Si, Ci oscillate. Abramowitz & Stegun 5.2.6-5.2.7 split them into the
// explicit sin t, cos t and two smooth, non-oscillating functions f, g:
//
//   Si(t) = pi/2 - f(t) cos t - g(t) sin t,   Ci(t) = f(t) sin t - g(t) cos t
//
// In Eq. 81 the sin x, cos x factors then collapse (xu - x = c x); with
// cos(c x) = 1 - 2 sin^2(c x/2) (no cancellation at small c x), exactly
//
//   u m(c) = [g(x) - g(xu)] + 2 g(xu) sin^2(c x/2) + [f(xu) - 1/xu] sin(c x)
//
// g ~ -ln t at t -> 0, so the table stores G(t) = g(t) + ln t (finite,
// -gamma_E at 0); since ln xu - ln x = ln(1+c), exactly,
//
//   g(x) - g(xu) = G(x) - G(xu) + ln(1+c),   g(xu) = G(xu) - ln xu
//
// Table (nfw_, built by nfw_table): f and G at Ntable.halo_nfw_n nodes
// uniform in ln t over [NFW_TMIN, NFW_TASY] = [1e-10, 50], from GSL Si,
// Ci; read by linear interpolation in ln t (interpol1d), clamped below
// NFW_TMIN (f, G flat there to 3e-9); asymptotic series above NFW_TASY.
// nfw_um does the reads and assembles u m(c) from (c, x, ln x, ln(1+c));
// this function supplies r_Delta, x and the two logs and divides by m(c).
// A call costs at most three table reads plus log, log1p, pow and two
// sines: no special function. The p_mm table builder reads nfw_um
// directly, with ln(1+c), r_s and ln r_s computed once per (a, mass node)
// and reused across its k loop (its header, item 2).
//
// Measured 2026-09-29 (3000 random (c, k, m), c in [0.05, 100], vs Eq. 81
// in 30-digit arithmetic, mpmath): max relative error 6.1e-7 (at c < 0.1,
// where m(c) ~ c^2/2 amplifies the table's absolute error), median 1e-10.
//
// Cache invalidation:
//   f, G depend on no parameter: nfw_table builds them on the first call
//   and rebuilds when Ntable.random changes (halo_nfw_n is boosted by
//   accuracy_boost in init_accuracy_boost); nfw_.cache holds the tag. The
//   build is not thread-safe and the halo integrands call u_nfw_c (and
//   the p_mm rows nfw_um) inside OpenMP loops: the first call is the
//   single-threaded init = 1 warm-up (halo_wrapper.hpp, "The init flag").
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
  // Geometry: x = k r_s and the logs the table axis needs
  const double rho_delta = Delta * cosmology.rho_crit * cosmology.Omega_m;
  const double r_delta = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
  const double x = k * r_delta / c;
  const double l1c = log1p(c);          // ln(1 + c)
  const double lx = log(x);
  return nfw_um(c, x, lx, l1c)/(l1c - c/(1.0 + c));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

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
  switch(like.halo_model[3])
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

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GALAXY PROFILES
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
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Normalized Fourier transform of the satellite-galaxy profile of lens
// bin ni: an NFW profile with the halo's truncation radius r_Delta and
// the concentration scaled by f_g = nuisance.gc[ni],
//
//   c_g(M) = f_g c(M),   r_s,g = r_Delta/c_g
//
// f_g = 1 puts the satellites on the dark matter profile (the
// assumption of 1005.2413 sec. 2.3); f_g < 1 spreads them out, f_g > 1
// concentrates them. f_g must be positive: c_g = 0 makes the NFW
// normalization m(0) vanish, so the function aborts unless gc[ni] > 0.
//
// Parameters:
//   c  - halo concentration c(M)
//   k  - wavenumber in (c/H0)^-1
//   m  - halo mass in M_sun/h
//   a  - scale factor (unused by the NFW form)
//   ni - lens bin (indexes nuisance.gc)
//
// Returns:
//   u_g(k|M), dimensionless; 1 at k -> 0. Aborts unless nuisance.gc[ni]
//   is positive.
// ---------------------------------------------------------------------------
double u_g(
    const double c, // halo concentration c(M)
    const double k, // wavenumber in (c/H0)^-1
    const double m, // halo mass in M_sun/h
    const double a, // scale factor (unused by the NFW form)
    const int ni    // lens bin: selects the factor nuisance.gc[ni]
  )
{
  if (!(nuisance.gc[ni] > 0)) {
    log_fatal("galaxy concentration factor gc[%d] = %g must be > 0",
              ni, nuisance.gc[ni]);
    exit(1);
  }
  return u_nfw_c(c*nuisance.gc[ni], k, m, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Mean number of central galaxies of lens bin ni in a halo of mass M
// (1005.2413 Eq. 7, central factor):
//
//   N_c(M) = (1/2) [1 + erf((log10 M - log10 M_min)/sigma_lgM)]
//
// A smoothed step: N_c = 1/2 at M = M_min, and sigma_lgM is the scatter
// between galaxy luminosity and halo mass, seen as a width in log10 M
// (the erf argument has no sqrt(2)).
//
// Parameters:
//   m  - halo mass in M_sun/h
//   a  - scale factor, 0 < a < 1 (checked; the HOD does not evolve)
//   ni - lens bin, 0 <= ni < redshift.clustering_nbin
//
// Returns:
//   N_c in [0, 1]. Aborts when nuisance.hod[ni][0] = log10 M_min lies
//   outside [10, 16], the sign that the bin's HOD is not set.
// ---------------------------------------------------------------------------
double HOD_nc(
    const double m, // halo mass in M_sun/h
    const double a, // scale factor, 0 < a < 1 (checked; HOD is z-free)
    const int ni    // lens bin, 0 <= ni < redshift.clustering_nbin
  )
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  if (ni < 0 || ni > redshift.clustering_nbin - 1) { 
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  if (nuisance.hod[ni][0] < 10 || nuisance.hod[ni][0] > 16) {
    log_fatal("HOD parameters in redshift bin %d not set", ni); exit(1);
  }

  const double x = (log10(m) - nuisance.hod[ni][0])/nuisance.hod[ni][1];
  
  gsl_sf_result ERF;
  {
    int status = gsl_sf_erf_e(x, &ERF);
    if (status) {
      log_fatal(gsl_strerror(status)); exit(1);
    }
  }
  return 0.5*(1.0 + ERF.val);
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
// M = M_1 when M_0 and M_min are well below M_1).
//
// Floor: the power law needs M > M_0 (a negative base has no real
// power for non-integer alpha), so M <= M_0 is tested explicitly and
// returns 1e-15; an ns that underflows to 0 gets the same floor. It
// keeps N_s strictly positive, and its contribution to every integral
// is negligible.
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
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  const double m0 = pow(10., nuisance.hod[ni][3]);
  if (!(m > m0)) {
    return 1.e-15; // no satellites at or below M_0
  }
  const double x = (m - m0)/pow(10., nuisance.hod[ni][2]);
  const double ns = HOD_nc(m, a, ni)*pow(x, nuisance.hod[ni][4]);
  return (ns > 0) ? ns : 1.e-15;
}

// ---------------------------------------------------------------------------
// Central fraction f_c of lens bin ni: of the halos that host a central
// above the threshold, the fraction whose central belongs to the sample
// (a completeness factor on centrals only; not part of the
// five-parameter form of 1005.2413). It multiplies N_c in the
// occupation, <N|M> = f_c N_c + N_s, and in the central-satellite pair
// count 2 f_c N_c N_s of the 1-halo term; satellites do not carry it.
//
// Parameters:
//   ni - lens bin, 0 <= ni < redshift.clustering_nbin
//
// Returns:
//   nuisance.hod[ni][5], or 1.0 when that slot is 0 (unset)
// ---------------------------------------------------------------------------
double HOD_fc(
    const int ni  // lens bin, 0 <= ni < redshift.clustering_nbin
  )
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  return (nuisance.hod[ni][5]) ? nuisance.hod[ni][5] : 1.0;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// GAS PROFILES
//
// The electron-pressure (thermal SZ) side of the halo model, after the
// HMx model of Mead et al. 2020 (2005.00009 secs. 3.2-3.3). The baryons
// that belong to a halo of mass M split into three parts:
//
//   f_bnd(M) = gas bound inside r_Delta, in hydrostatic
//              equilibrium, Komatsu-Seljak profile        -> frac_bnd
//   f_*(M)   = stars                                       -> frac_ejc
//   f_ejc(M) = gas ejected beyond r_Delta,
//              Omega_b/Omega_m - f_bnd - f_*               -> frac_ejc
//
// The bound gas enters both halo terms through its pressure window
// (u_y_bnd); the ejected gas is a smooth, warm component that enters
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
// Masses in M_sun/h.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Komatsu-Seljak ingredients of u_KS at complex radius (u_KS header,
// items 2-3): the profile shape theta and the integrand g of the two
// contour rays. C99 <complex.h>: double complex is a pair of doubles,
// I the imaginary unit, creal/cimag/cabs the real part, imaginary part
// and modulus, clog and cexp the complex log (principal branch) and exp.
// ---------------------------------------------------------------------------
static inline double complex ks_ctheta(
    const double complex x  // complex radius r/r_s, Re x > -1
  )
{ // theta(x) = ln(1 + x)/x, with ln(1 + x) accurate near x = 0
  // The quotient is 0/0 at x = 0, so below |x| = 1e-4 the Taylor series
  // of theta takes over (five terms; the dropped one is |x|^5/6 ~ 1e-21
  // at the switch). Elsewhere ln(1 + x) is assembled from its two parts:
  //
  //   Re ln(1 + x) = ln|1 + x| = (1/2) ln(1 + 2 Re x + |x|^2),
  //   Im ln(1 + x) = arg(1 + x) = atan2(Im x, 1 + Re x)  in (-pi, pi].
  //
  // log1p(v) is ln(1 + v) computed without forming 1 + v, so the real
  // part keeps its digits when x is small; atan2 gives the principal
  // argument, whose cut runs along the real axis left of x = -1, where
  // no contour point lies.
  if (cabs(x) < 1e-4) {
    return 1.0 - x/2.0 + x*x/3.0 - x*x*x/4.0 + x*x*x*x/5.0;
  }
  const double xr = creal(x);
  const double xi = cimag(x);
  const double complex l1p = 0.5*log1p(2.0*xr + xr*xr + xi*xi) +
                             I*atan2(xi, 1.0 + xr);
  return l1p/x;
}

static inline double complex ks_cg(
    const double complex x, // complex radius r/r_s
    const double p          // Gamma/(Gamma - 1)
  )
{ // g(x) = x theta(x)^p
  // A complex number to a real power is exp(p log theta) with the
  // principal log. On the rays and inside the contour of the u_KS
  // header, |arg theta| < pi/2 (checked over c in [0.05, 100]), so
  // theta never reaches the cut of clog on the negative real axis and
  // g is one continuous, analytic function there.
  return x*cexp(p*clog(ks_ctheta(x)));
}

// natural cubic spline of yc (nc uniform nodes, spacing hc) evaluated at
// the (nc - 1) m + 1 nodes of the grid that refines each interval m times
static void ks_upsample1d(
    const double* yc, // coarse values
    const int nc,     // coarse nodes
    const double hc,  // coarse spacing
    double* cs,       // workspace [nc]: spline c coefficients
    double* yf,       // output [(nc - 1) m + 1]
    const int m       // refinement factor
  )
{
  // The house 1D upsampling (tinker_alpha header, item 3, and the
  // sigma2 refill block in cosmo3D.c). spline_coeffs_uniform solves the
  // tridiagonal system for c_j = S''(x_j)/2 with c = 0 at both ends;
  // on interval j the cubic in Horner form is
  //
  //   S(x_j + t) = y_j + t (b + t (c_j + t d)),      0 <= t <= hc,
  //   b = (y_{j+1} - y_j)/hc - hc (c_{j+1} + 2 c_j)/3,
  //   d = (c_{j+1} - c_j)/(3 hc).
  //
  // Dense node j m + r sits at offset t = r hc/m inside interval j: no
  // search and no clamp, because the dense grid is built from the
  // coarse one. r = 0 reproduces y_j exactly; the last coarse node is
  // written on its own, as no interval starts there.
  spline_coeffs_uniform(yc, nc, hc, cs);
  for (int j=0; j<nc-1; j++) {
    const double b = (yc[j+1] - yc[j])/hc - hc*(cs[j+1] + 2.0*cs[j])/3.0;
    const double d = (cs[j+1] - cs[j])/(3.0*hc);
    for (int r=0; r<m; r++) {
      const double t = hc*((double) r)/((double) m);
      yf[j*m + r] = yc[j] + t*(b + t*(cs[j] + t*d));
    }
  }
  yf[(nc - 1)*m] = yc[nc - 1];
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
// Gamma its polytropic index (2005.00009 sec. 3.2, the rho_bnd
// equation). Of the gas parameters only Gamma = nuisance.gas[0]
// enters. The full window (u_y_bnd) is
//
//   W_p(M, k) = [k_B T_v f_bnd M/(m_p mu_e)] u_KS.
//
// Unlike the matter u(k|M), u_KS does not tend to 1 at k -> 0:
//
//   u_KS(c, 0) = F(c, 0)/F0(c) = mass-weighted mean of theta
//              = <T_g>/T_v < 1
//
// the mass-weighted gas temperature in units of the central temperature
// T_v (theta(0) = 1). For every k, |u_KS(c, k)| <= u_KS(c, 0).
//
// Two phases appear below: y = k r_s, the argument of the integral, and
// z = y c = k r_v, the phase at the outer edge x = c.
//
// 1. Why u cannot be tabulated directly
//
// The sin(y x) inside F makes u oscillate in z. Past its first zero
// (z = 4.5 at the earliest, over all c and Gamma) u rings like cos z
// under a slowly falling envelope, crossing zero hundreds of times
// before |u| drops below 1e-6. No table follows that ringing up to
// z ~ 1e4, and any interpolation across a zero crossing is a large
// relative error. The cure: take the oscillation out of the integral
// analytically and put it back, exactly, at lookup time.
//
// 2. The oscillation becomes a factor
//
// Write F through a complex integral. With g(x) = x theta(x)^p,
//
//   J(c, y) = int_0^c g(x) e^{i y x} dx,    F = Im J/y,
//
// since Im e^{iyx} = sin(yx). g is analytic (complex differentiable)
// off the real axis: the only singularity of ln(1 + x) is the branch
// point at x = -1, and its cut runs along the real axis to the left of
// it. The integral of an analytic function around a closed loop is zero
// (Cauchy's theorem), so the path from 0 to c can be traded for any
// other path in the upper half plane. Take the rectangle
//
//   0 -> c -> c + i T -> i T -> 0,    T -> infinity.
//
// On the top edge |e^{i y x}| = e^{-y T} -> 0 while g grows at most
// like a power, so that edge drops out and J equals the two vertical
// rays, x = i tau and x = c + i tau with tau from 0 to infinity:
//
//   J = i int_0^inf g(i tau) e^{-y tau} dtau
//       - e^{i z} i int_0^inf g(c + i tau) e^{-y tau} dtau.
//
// On both rays e^{iyx} has turned into the real, decaying e^{-y tau}:
// neither integrand oscillates. The whole phase sits in the one factor
// e^{iz} of the second ray, z = y c. Substituting tau = c t there and
// pulling out g(c),
//
//   J = i I0(y) - e^{iz} (i g(c)/y) Q(c, z),
//
//   I0(y)   = int_0^inf g(i tau) e^{-y tau} dtau,
//   Q(c, z) = z int_0^inf [g(c + i c t)/g(c)] e^{-z t} dt.
//
// Taking the imaginary part (Im(i w) = Re w) and dividing by y F0,
//
//   u = [P(y) - (g(c)/y) (cos z Re Q - sin z Im Q)]/(y F0(c)),
//
//   P(y) = Re I0(y) = Re int_0^inf g(i tau) e^{-y tau} dtau.
//
// Every oscillation now lives in cos z and sin z, evaluated exactly at
// lookup. The four other ingredients are smooth: P depends on y alone,
// F0 and g(c) on c alone, Q on (c, z) and is O(1). The weight z e^{-zt}
// in Q has unit integral and width 1/z, so Q averages the ratio
// g(c + ict)/g(c) over t up to about 1/z; the ratio is 1 at t = 0,
// hence Q -> 1 as z -> infinity, with corrections in powers of 1/z
// (expanding such an integral from the Taylor series of its integrand
// at t = 0 is Watson's lemma). Above the top of the ln z axis,
// ZHI = 2.5e5, Q is held at its ZHI value.
//
// The contour method is standard complex analysis; 2005.00009 is cited
// here for the profile only.
//
// 3. The two smooth integrals: a trapezoid rule in s = ln t
//
// Q and P are half-line integrals whose integrands decay exponentially
// at large t and vanish as a power at t -> 0. In s = ln t (t = e^s,
// dt = t ds), and with tau = t/y in P,
//
//   Q = z int [g(c + i c t)/g(c)] t e^{-z t} ds,
//   P = (1/y) int Re g(i t/y) t e^{-t} ds,
//
// each integrand is one smooth bump: it falls like e^s or faster toward
// s = -infinity and like exp(-e^s) toward +infinity. On such a bump the
// plain trapezoid rule converges exponentially in the step (tinker_alpha
// header, item 2: its error is made of the integrand's derivatives at
// the two ends, all negligible here). The scaling tau = t/y puts the
// bump of P at the same s for every y, so one window serves the whole
// ln y axis; and with the ln y spacing set to hy = h/rP, rP an integer,
// every t_k/y_j lies on one grid of ln tau (body, the tg loops), so
// g(i tau) is evaluated once per grid point rather than once per pair.
// Window and step ladder with hdi = abs(Ntable.high_def_integration):
//
//   hdi          0           1           >= 2
//   s window     [-32, 4]    [-40, 4]    [-40, 4]
//   step h       0.2         0.2         0.1
//   nodes (nt)   181         221         441
//
// The dropped left tail of Q is about z e^{smin}: 3e-9 at z = ZHI for
// hdi = 0 and 1e-12 for hdi >= 1 (measured against a finer rule:
// |dQ| <= 2.9e-9 over c in [0.05, 100], z in [1, 2.5e5] and Gamma in
// [1.05, 1.35] at hdi = 0). At s = 4 the factor exp(-z e^4) is e^{-164}
// already at z = 3.
//
// The weights z h t e^{-zt} of Q depend on z alone and h t e^{-t} of P
// on nothing, so both are built once per Ntable rebuild; a change of
// Gamma recomputes only g at the nodes (ks_cg).
//
// 4. Small z: a table of u itself
//
// Below ZSW = 3, u has not yet crossed zero (item 1), and the two terms
// of item 2 each grow like 1/y as y -> 0 while their difference stays
// finite, so the formula would lose digits there. For z < ZSW the code
// tabulates u directly, on (ln c, w = z^2). With x = c s,
//
//   u(c, z) = int_0^1 s sin(z s)/z theta(c s)^p ds
//             / int_0^1 s^2 theta(c s)^q ds
//
// (the c^3 of both integrals cancels): two integrals on [0, 1] of
// smooth integrands, done by Gauss-Legendre quadrature (sigma2 header
// in cosmo3D.c, item 3) with the ladder
//
//   hdi      0     1     2     3     >= 4
//   nodes    96    128   256   512   1024    (predefined GSL tables)
//
// converged to 2e-14 in F0 at 96 nodes. w = z^2 rather than z as the
// table variable: u is even in z, so near z = 0 it is a straight line
// in w (u0 - a w + ...) and the plateau costs the interpolation
// nothing. The padding nodes at w < 0 (item 5) hold the same function
// continued to imaginary z: with z = i kappa, kappa = sqrt(-w),
// sin(z s)/z = sinh(kappa s)/kappa.
//
// 5. Tables: coarse exact values, cubic upsampling, linear reads
//
// The house pattern (tinker_alpha header, item 3): exact values on a
// coarse grid, a natural cubic spline through them evaluated on a
// dense grid, and linear reads from the dense grid (interpol1d and
// interpol2d, index by arithmetic, no search). Six tables:
//
//   table          axes          coarse nodes   dense nodes
//   S = u          ln c, w       NC x NW        613 x 545
//   Re Q, Im Q     ln c, ln z    NC x NZ        613 x 1105  (each)
//   ln P           ln y          NY             23231
//   ln F0, ln g    ln c          N1             4971        (each)
//
// (dense sizes at init_accuracy_boost = 1: about 13.8 MB in all, the
// two Q tables 10.8 MB of it). NC = Ntable.halo_uks_nc (40) and
// NZ = Ntable.halo_uks_nz (64), both scaled by init_accuracy_boost;
// NW = ceil(6 NZ/64) (6); NY covers the ln y range at the spacing
// hy = h/rP of item 3 (191); N1 = 1.5 NC (60). Used ranges:
//
//   ln c    [ln limits.halo_uks_cmin, ln limits.halo_uks_cmax]
//           = [ln 0.05, ln 100]
//   w       [0, ZSW^2] = [0, 9]
//   ln z    [ln ZSW, ln ZHI] = [ln 3, ln 2.5e5]
//   ln y    [ln(ZSW/cmax), ln(ZHI/cmin)] = [ln 0.03, ln 5e6]
//
// ZHI covers the largest phase the halo model asks for, z = k r_v up
// to k_max r_v(M_max) = 3e6 x 3.8e-3 ~ 1.1e4. A query with c outside
// [cmin, cmax] is clamped to the edge (c ~ 0.16 occurs at a = 1/41).
// ln c as the axis: at equal node count it is 1e2 to 1e4 times more
// accurate than an axis uniform in c, the profile integrals varying on
// a logarithmic scale in c.
//
// Padding. A natural spline sets S'' = 0 at its two end nodes, wrong
// wherever the function is curved there (tinker_alpha header, item 4).
// Every used boundary therefore gets PAD = 6 extra coarse nodes beyond
// it, and the lookups clamp their arguments to the used ranges, so the
// padding is never read (20 to 200 times less error at the used edges).
// ln z is padded below only: at ZHI, Q is within O(1/z) of its constant
// limit and S'' = 0 is nearly exact.
//
// Refinement. The dense grids share both endpoints with the padded
// coarse grids (the contract of spline2d_upsample_uniform, basics.c),
// so each dense count is (coarse - 1) m + 1 with an integer m:
//
//   axis      ln c    w       ln z     ln y      ln c (1D)
//   m         12      32      16       115       70
//   spacing   0.016   0.056   0.011    0.00087   0.0018
//
// The 1D tables are cheap, so their grids are the finest.
//
// 6. Cost and accuracy
//
// Build, once per Gamma change: the coarse S and Q values run threaded
// over the c nodes, g(i tau) over the shared tau grid and P over the y
// nodes, F0 and g serially; then the six upsamplings run as six
// independent jobs, one per thread. The upsamplings dominate (2.9 ms
// per Q table, 1.3 ms for S, on one thread): 3.0 ms in all with 4
// threads, the two Q jobs running side by side and setting the wall
// time. Read: a cos, a sin, three log, three exp and five table reads:
// 46 ns.
//
// Accuracy against an independent evaluator (verified with mpmath) at
// 20000 random (c, z), c in [0.05, 100], z in [1e-6, 1.2e4],
// Gamma = 1.17: maximum error 4.8e-6 relative to the local envelope of
// u, median 8e-7; maximum relative error 1.8e-5 where |u| > 1e-2.
// init_accuracy_boost = 2 gives 1.1e-6; hdi = 1 or 2 changes nothing
// (the quadratures are converged; the table reads set the floor).
//
// Cache invalidation:
//   allocation, limits, quadrature nodes, weights and tau grid: rebuilt
//     when Ntable.random changes (hdi and the node counts enter)
//   table refill: nuisance.random_gas (cache[0]) or Ntable.random
//                 (cache[1])
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
  // Design constants (header, items 2, 4 and 5). ZSW splits the two
  // methods at z = k r_v: the table of u below it, the contour formula
  // above; ZHI tops the ln z axis of Q. The enum holds the integer
  // constants, PAD and the refinement factors m of the five axes (an
  // enum name is a compile-time constant, usable wherever an int
  // literal is).
  const double ZSW = 3.0;     // z = k r_v below: table of u(ln c, z^2)
  const double ZHI = 2.5e5;   // top of the ln z axis of Q
  enum {
    PAD = 6,                  // coarse padding nodes beyond used ends
    MC  = 12,                 // dense refinement factors
    MW  = 32,
    MZ  = 16,
    MY  = 115,
    M1  = 70
  };

  // Static state, kept between calls and zeroed at program start, so
  // the first call (Sd == NULL) builds everything. Two blocks write it:
  //   Ntable rebuild block   the sizes (rP and ntau among them), the
  //                          dense axes lim, every allocation, the
  //                          Gamma-independent quadrature nodes and
  //                          weights gl, kw, ns, wz and the tau grid
  //                          tg[0] of the P sums
  //   refill block           g on the tau grid tg[1], the coarse values
  //                          Sc, Qc, lnPc, c1, the dense tables Sd, Qd,
  //                          lnPd, d1, then the tags cache[]
  // Naming: a "p" suffix is a padded coarse count, a "d" suffix a dense
  // count; lim[a] = {first node, last node, spacing} of dense axis a,
  // and the coarse spacing is lim[a][2] times the refinement factor.
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static int ncp, nwp, nzp, nyp, n1p;  // padded coarse sizes
  static int ncd, nwd, nzd, nyd, n1d;  // dense sizes (shared endpoints)
  static int ngl;              // Gauss-Legendre nodes on s in [0, 1]
  static int nt;               // contour trapezoid nodes in s = ln tau
  static int rP;               // hs/hy: trapezoid steps per ln y step
  static int ntau;             // distinct tau = t_k/y_j of the P sums
  static double** Sd = NULL;  // [ncd][nwd] u(ln c, w)
  static double*** Qd = NULL; // [2][ncd][nzd] Re Q, Im Q (ln c, ln z)
  static double* lnPd = NULL; // [nyd] ln P(ln y)
  static double** d1 = NULL;  // [2][n1d] ln F0, ln g (ln c)
  static double lim[5][3];    // dense axes (padded extents): ln c, w,
                              // ln z, ln y, ln c (1D)
  static double** Sc = NULL;  // coarse exact values, same layouts
  static double*** Qc = NULL;
  static double* lnPc = NULL;
  static double** c1 = NULL;
  static double** cs = NULL;  // [3][max(nyp, n1p)] spline workspaces
  static double** gl = NULL;  // [2][ngl] GL nodes, weights on [0, 1]
  static double** kw = NULL;  // [ngl][nwp] s sin(z s)/z (sinh for w < 0)
  static double** wz = NULL;  // [nt][nzp] z h tau e^{-z tau}
  static double** ns = NULL;  // [2][nt] tau_k = e^{s_k}, h t e^{-t}
  static double** tg = NULL;  // [2][ntau] tau_m, Re g(i tau_m)

  // Ntable rebuild block: the first call, or Ntable.random (the tag of
  // the Ntable settings) differs from the tag the tables hold. malloc1d,
  // malloc2d and malloc3d (basics.c) each return one contiguous block,
  // row pointers first and data after, so one free releases each.
  if (NULL == Sd || fdiff2(cache[1], Ntable.random)) {
    if (Sd != NULL) {
      free(Sd); free(Qd); free(lnPd); free(d1); free(Sc); free(Qc);
      free(lnPc); free(c1); free(cs); free(gl); free(kw); free(ns); free(wz);
      free(tg);
    }
    // Used ranges of ln c and ln y (header, item 5); y = z/c is
    // smallest at z = ZSW, c = cmax and largest at z = ZHI, c = cmin.
    const double lc0 = log(limits.halo_uks_cmin);
    const double lc1 = log(limits.halo_uks_cmax);
    const double y0 = log(ZSW/limits.halo_uks_cmax);
    const double y1 = log(ZHI/limits.halo_uks_cmin);
    // used-range node counts: ln c and ln z from the knobs; w, ln y and
    // the 1D ln c axis follow them
    //
    // hc and hz are the coarse spacings of the two knob axes and hs the
    // trapezoid step of the hdi ladder (header, item 3). The ln y
    // spacing hy = hs/rP with rP = ceil(hs/hz) is the largest step at
    // or below hz of which hs is an integer multiple: then every
    // t_k/y_j of the P sums lies on one grid of ln tau (the tg loop
    // below). NW keeps 6 nodes on w in [0, 9] at NZ = 64 and grows with
    // NZ; NY is the count that covers [y0, y1] at the spacing hy (the
    // ceil rounds the interval count up, the + 1 counts nodes); N1 gives
    // the 1D ln c tables 1.5 times the nodes of the 2D ones. Defaults:
    // NC = 40, NZ = 64, hz = 0.18, hs = 0.2, rP = 2, hy = 0.1, NW = 6,
    // NY = 191, N1 = 60.
    const int NC = Ntable.halo_uks_nc;
    const int NZ = Ntable.halo_uks_nz;
    const double hc = (lc1 - lc0)/((double) NC - 1.0);
    const double hz = (log(ZHI) - log(ZSW))/((double) NZ - 1.0);
    const int hdi = abs(Ntable.high_def_integration);
    const double hs = (hdi < 2) ? 0.2 : 0.1;        // trapezoid step
    rP = (int) ceil(hs/hz);
    const double hy = hs/rP;
    const int NW = (int) ceil(6.0*NZ/64.0);
    const int NY = (int) ceil((y1 - y0)/hy) + 1;
    const int N1 = (int) ceil(1.5*NC);
    const double hw = ZSW*ZSW/((double) NW - 1.0);
    const double h1 = (lc1 - lc0)/((double) N1 - 1.0);
    // Padded coarse counts (header, item 5: PAD nodes beyond each used
    // end) and the dense counts that share their endpoints: with
    // (coarse - 1) m + 1 dense nodes every coarse node is a dense node
    // and every coarse interval holds exactly m dense steps.
    ncp = NC + 2*PAD;
    nwp = NW + 2*PAD;
    nzp = NZ + PAD;   // ln z is padded below only
    nyp = NY + 2*PAD;
    n1p = N1 + 2*PAD;
    ncd = (ncp - 1)*MC + 1;
    nwd = (nwp - 1)*MW + 1;
    nzd = (nzp - 1)*MZ + 1;
    nyd = (nyp - 1)*MY + 1;
    n1d = (n1p - 1)*M1 + 1;
    // Quadrature sizes from the two hdi ladders (header, items 3-4): ngl
    // Gauss-Legendre nodes for the [0, 1] integrals of S, F0 and g; a
    // trapezoid window [smin, smax] in s = ln t for Q and P, nt nodes at
    // the step hs set above. lround keeps a last-digit rounding of the
    // division from losing a node. ntau counts the distinct tau = t_k/y_j
    // of the P sums (derived at the tg loop below).
    ngl = (0 == hdi) ? 96 :
          (1 == hdi) ? 128 :
          (2 == hdi) ? 256 :
          (3 == hdi) ? 512 : 1024; // predefined GSL tables
    const double smin = (0 == hdi) ? -32.0 : -40.0; // s = ln tau window
    const double smax = 4.0;
    nt = (int) lround((smax - smin)/hs) + 1;
    ntau = (nt - 1)*rP + nyp;

    // Allocation, all of it in this block. cs holds three separate
    // spline workspaces because the three 1D upsamplings of the refill
    // block run at the same time, one per thread.
    Sd = (double**) malloc2d(ncd, nwd);
    Qd = (double***) malloc3d(2, ncd, nzd);
    lnPd = (double*) malloc1d(nyd);
    d1 = (double**) malloc2d(2, n1d);
    Sc = (double**) malloc2d(ncp, nwp);
    Qc = (double***) malloc3d(2, ncp, nzp);
    lnPc = (double*) malloc1d(nyp);
    c1 = (double**) malloc2d(2, n1p);
    cs = (double**) malloc2d(3, (nyp > n1p) ? nyp : n1p);
    gl = (double**) malloc2d(2, ngl);
    kw = (double**) malloc2d(ngl, nwp);
    ns = (double**) malloc2d(2, nt);
    wz = (double**) malloc2d(nt, nzp);
    tg = (double**) malloc2d(2, ntau);

    // padded coarse extents; the dense grids share them
    //
    // Each dense axis: first node = used start minus PAD coarse
    // spacings, spacing = coarse spacing over m, last node = first +
    // (dense count - 1) spacings. This is the (first, last, spacing)
    // triple interpol1d and interpol2d take. The w axis starts at
    // -PAD hw = -10.8, below w = 0 (header, item 4).
    lim[0][0] = lc0 - PAD*hc;          lim[0][2] = hc/MC;
    lim[1][0] = -PAD*hw;               lim[1][2] = hw/MW;
    lim[2][0] = log(ZSW) - PAD*hz;     lim[2][2] = hz/MZ;
    lim[3][0] = y0 - PAD*hy;           lim[3][2] = hy/MY;
    lim[4][0] = lc0 - PAD*h1;          lim[4][2] = h1/M1;
    lim[0][1] = lim[0][0] + (ncd - 1)*lim[0][2];
    lim[1][1] = lim[1][0] + (nwd - 1)*lim[1][2];
    lim[2][1] = lim[2][0] + (nzd - 1)*lim[2][2];
    lim[3][1] = lim[3][0] + (nyd - 1)*lim[3][2];
    lim[4][1] = lim[4][0] + (n1d - 1)*lim[4][2];

    // Gauss-Legendre nodes gl[0][q] and weights gl[1][q] on [0, 1]
    // (header, item 4). malloc_gslint_glfixed(n) wraps the GSL table of
    // n nodes; gsl_integration_glfixed_point copies node q mapped onto
    // [0, 1], and its weight scaled for that interval, out of it.
    gsl_integration_glfixed_table* t = malloc_gslint_glfixed(ngl);
    for (int q=0; q<ngl; q++) {
      gsl_integration_glfixed_point(0.0, 1.0, q, &gl[0][q], &gl[1][q], t);
    }
    gsl_integration_glfixed_table_free(t);
    // kw[q][j] = s_q sin(z s_q)/z at w node j, z = sqrt(w): the
    // z-dependent half of the numerator integrand of header item 4.
    // Gauss-Legendre node q is the row index, so the factors of all w
    // nodes at one node form one contiguous row, which the S sums of
    // the refill block add in one sweep. The w nodes run over the
    // padded axis, -PAD hw + j hw, and the three branches are one
    // analytic function of w: sinh(kappa s)/kappa with kappa = sqrt(-w)
    // at w < 0, and the limit s^2 at w = 0.
    for (int j=0; j<nwp; j++) {
      const double w = -PAD*hw + j*hw;
      for (int q=0; q<ngl; q++) {
        if (w > 0) {
          const double z = sqrt(w);
          kw[q][j] = gl[0][q]*sin(z*gl[0][q])/z;
        }
        else if (w < 0) {
          const double kk = sqrt(-w);
          kw[q][j] = gl[0][q]*sinh(kk*gl[0][q])/kk;
        }
        else {
          kw[q][j] = gl[0][q]*gl[0][q];
        }
      }
    }
    // Trapezoid nodes in s (header, item 3): ns[0][k] = t_k = e^{s_k}
    // and ns[1][k] = h t_k e^{-t_k}, the weight of P (h from the
    // trapezoid rule, t_k from dt = t ds, e^{-t_k} from the integrand).
    // The end weights are not halved: the integrand is negligible at
    // both ends, which is the property the rule relies on.
    for (int k=0; k<nt; k++) {
      ns[0][k] = exp(smin + k*hs);
      ns[1][k] = hs*ns[0][k]*exp(-ns[0][k]);
    }
    // wz[k][j] = z h t_k e^{-z t_k}, the weight of Q at ln z node j
    // (the leading z of Q included); z runs over the padded ln z axis
    // from its first node lim[2][0]. Trapezoid node k is the row index,
    // so the weights of all z nodes at one node form one contiguous row;
    // the Q sums of the refill block add each row, times the g ratio at
    // its node, to the row of Q values.
    for (int j=0; j<nzp; j++) {
      const double z = exp(lim[2][0] + j*hz);
      for (int k=0; k<nt; k++) {
        wz[k][j] = z*hs*ns[0][k]*exp(-z*ns[0][k]);
      }
    }
    // Shared tau grid of the P sums (header, item 3). P(y_j) needs
    // Re g(i tau) at tau = t_k/y_j for every pair (j, k). With
    // s_k = smin + k hs = smin + k rP hy and ln y_j = lim[3][0] + j hy,
    //
    //   ln(t_k/y_j) = s_k - ln y_j = (smin - lim[3][0]) + (k rP - j) hy,
    //
    // so every pair lands on one grid of ln tau with spacing hy, at the
    // index m = k rP - j + (nyp - 1): m = 0 for (k, j) = (0, nyp - 1),
    // m = ntau - 1 for (nt - 1, 0). tg[0][m] = tau_m is Gamma
    // independent and built here; the refill block fills tg[1][m] =
    // Re g(i tau_m) once per m and each P sum reads it at stride rP (a
    // shift in ln y is a shift of index on a log grid; the FFTLog of
    // cosmo2D.c uses the same fact as a phase factor).
    for (int m=0; m<ntau; m++) {
      tg[0][m] = exp(smin - lim[3][0] + (m - (nyp - 1))*hy);
    }
  }

  // Refill block: the gas tag (Gamma changed) or the Ntable tag differs
  // from the ones the tables hold. Everything Gamma enters is recomputed
  // here; the nodes and weights above are reused.
  if (fdiff2(cache[0], nuisance.random_gas) || fdiff2(cache[1], Ntable.random))
  {
    // The two KS exponents (header): p for the pressure profile, q for
    // the density profile. The coarse spacings come back from the dense
    // ones, the only spacings kept in static storage.
    const double p = nuisance.gas[0]/(nuisance.gas[0] - 1.0);
    const double q = 1.0/(nuisance.gas[0] - 1.0);
    const double hc = lim[0][2]*MC;
    const double hw = lim[1][2]*MW;
    const double hz = lim[2][2]*MZ;
    const double hy = lim[3][2]*MY;
    const double h1 = lim[4][2]*M1;

    // Coarse S and Q, one c node per iteration (header, items 2-4).
    // schedule(static): contiguous chunks of i per thread; every entry
    // is a serial sum inside one thread, so the values do not depend on
    // the thread count. thp, gr and gi are variable-length arrays on the
    // stack, declared inside the loop body and hence private to each
    // iteration.
    #pragma omp parallel for schedule(static)
    for (int i=0; i<ncp; i++) {
      const double c = exp(lim[0][0] + i*hc);
      // Gauss-Legendre pass over s in [0, 1] at x = c s. f0 accumulates
      // int s^2 theta(cs)^q ds, the denominator of S; thp[k] keeps the
      // weight times theta(cs)^p, the z-independent half of the
      // numerator's integrand, and kw supplies the other half per w
      // node. log1p(x)/x is theta at real x.
      double thp[ngl];
      double f0 = 0.0;
      for (int k=0; k<ngl; k++) {
        const double x = c*gl[0][k];
        const double th = log1p(x)/x;
        f0 += gl[1][k]*gl[0][k]*gl[0][k]*pow(th, q);
        thp[k] = gl[1][k]*pow(th, p);
      }
      // S at every w node: the ratio of the two integrals of header
      // item 4. The numerator runs node-outer, w-inner: sr = Sc[i] is
      // zeroed, then each Gauss-Legendre node k adds thp[k] times its
      // row kw[k] to all of sr, so each w node adds its terms in the
      // order k = 0, 1, ..., a serial sum; the division by f0 comes
      // last. restrict on sr and kk promises the two rows do not overlap.
      //
      // The w loop is SIMDe intrinsics (v4d, top of file; the pattern of
      // cosmo2D.c): under the strict IEEE flags of the default build,
      // -frounding-math and -fno-associative-math, clang emits each
      // floating-point operation as a constrained call and vectorizes no
      // such loop, while intrinsics compile to vector instructions
      // directly. simde_mm256_set1_pd copies tk into all four lanes;
      // loadu_pd and storeu_pd move four consecutive doubles to and from
      // any address, aligned or not; mul_pd then add_pd round separately
      // (no fused multiply-add); the last nwp % 4 nodes take the scalar
      // tail. Both paths agree to rounding (below 1e-12 relative), not
      // bit for bit, and these sums are a small part of the build
      // (header, item 6). COSMO2D_NOT_USE_SIMD, defined by the debug
      // build (MakefileCosmolike), selects the scalar loop.
      double* restrict sr = Sc[i];
      for (int j=0; j<nwp; j++) {
        sr[j] = 0.0;
      }
      for (int k=0; k<ngl; k++) {
        const double tk = thp[k];
        const double* restrict kk = kw[k];
#ifdef COSMO2D_NOT_USE_SIMD
        for (int j=0; j<nwp; j++) {
          sr[j] += tk*kk[j];
        }
#else
        const v4d vt = simde_mm256_set1_pd(tk);
        int j = 0;
        for (; j <= nwp - 4; j += 4) {
          const v4d prod = simde_mm256_mul_pd(vt, simde_mm256_loadu_pd(kk + j));
          simde_mm256_storeu_pd(sr + j,
            simde_mm256_add_pd(simde_mm256_loadu_pd(sr + j), prod));
        }
        for (; j < nwp; j++) {
          sr[j] += tk*kk[j];
        }
#endif
      }
      for (int j=0; j<nwp; j++) {
        sr[j] /= f0;
      }
      // Q at every ln z node (header, items 2-3). gr[k] + i gi[k] is
      // the ratio g(c + i c t_k)/g(c) at the trapezoid nodes, computed
      // once per c because only the weights wz depend on z; gcr = g(c)
      // at real c in real arithmetic. The ratio is kept as two double
      // arrays, the form the v4d lanes take: Re Q and Im Q are two real
      // sums with the same weights and go to separate tables, Qc[0] and
      // Qc[1]. The sums run as for S above, node-outer and z-inner: the
      // rows qr and qi are zeroed, then each trapezoid node k adds gr[k]
      // and gi[k] times its row wz[k], four z nodes per SIMDe step with
      // a scalar tail for the last nzp % 4.
      double gr[nt];
      double gi[nt];
      const double gcr = c*pow(log1p(c)/c, p);
      for (int k=0; k<nt; k++) {
        const double complex gk = ks_cg(c + I*c*ns[0][k], p)/gcr;
        gr[k] = creal(gk);
        gi[k] = cimag(gk);
      }
      double* restrict qr = Qc[0][i];
      double* restrict qi = Qc[1][i];
      for (int j=0; j<nzp; j++) {
        qr[j] = 0.0;
        qi[j] = 0.0;
      }
      for (int k=0; k<nt; k++) {
        const double ar = gr[k];
        const double ai = gi[k];
        const double* restrict wk = wz[k];
#ifdef COSMO2D_NOT_USE_SIMD
        for (int j=0; j<nzp; j++) {
          qr[j] += wk[j]*ar;
          qi[j] += wk[j]*ai;
        }
#else
        const v4d var = simde_mm256_set1_pd(ar);
        const v4d vai = simde_mm256_set1_pd(ai);
        int j = 0;
        for (; j <= nzp - 4; j += 4) {
          const v4d vw = simde_mm256_loadu_pd(wk + j);
          simde_mm256_storeu_pd(qr + j, simde_mm256_add_pd(
            simde_mm256_loadu_pd(qr + j), simde_mm256_mul_pd(vw, var)));
          simde_mm256_storeu_pd(qi + j, simde_mm256_add_pd(
            simde_mm256_loadu_pd(qi + j), simde_mm256_mul_pd(vw, vai)));
        }
        for (; j < nzp; j++) {
          qr[j] += wk[j]*ar;
          qi[j] += wk[j]*ai;
        }
#endif
      }
    }
    // Coarse ln P (header, item 3): P(y_j) = (1/y_j) sum_k w_k
    // Re g(i t_k/y_j), w_k = ns[1][k] = h t_k e^{-t_k}. The first loop
    // evaluates Re g once per node of the shared tau grid (I*tg[0][m] is
    // the point i tau_m on the imaginary axis, creal the real part of g
    // there): ntau = 563 calls of ks_cg, each a clog and a cexp, in
    // place of one per (j, k) pair, nyp x nt = 36743 at the defaults.
    // The second loop is the strided sum of the tg comment above,
    //
    //   P(y_j) = (1/y_j) sum_k w_k tg[1][k rP - j + nyp - 1],
    //
    // gm pointing at tg[1] + (nyp - 1 - j) and read at gm[k rP]
    // (restrict: gm and w do not overlap). Both loops are threaded over
    // their index; the sum of each y node is serial inside one thread.
    // The table holds ln P: P falls as p/y^3 over the upper part of the
    // ln y range, so its log is close to a straight line, the
    // friendliest shape for a cubic.
    #pragma omp parallel for schedule(static)
    for (int m=0; m<ntau; m++) {
      tg[1][m] = creal(ks_cg(I*tg[0][m], p));
    }
    #pragma omp parallel for schedule(static)
    for (int j=0; j<nyp; j++) {
      const double y = exp(lim[3][0] + j*hy);
      const double* restrict gm = tg[1] + (nyp - 1 - j);
      const double* restrict w = ns[1];
      double sum = 0.0;
      for (int k=0; k<nt; k++) {
        sum += w[k]*gm[k*rP];
      }
      lnPc[j] = log(sum/y);
    }
    // Coarse ln F0 and ln g on the 1D ln c axis, serial (n1p sums of
    // ngl terms). F0 = c^3 int_0^1 s^2 theta(cs)^q ds, the c^3 from
    // x = c s (x^2 dx = c^3 s^2 ds); g(c) = c theta(c)^p, so
    // ln g = ln c + p ln theta(c). Both logs are near straight lines in
    // ln c.
    for (int j=0; j<n1p; j++) {
      const double c = exp(lim[4][0] + j*h1);
      double f0 = 0.0;
      for (int k=0; k<ngl; k++) {
        const double x = c*gl[0][k];
        f0 += gl[1][k]*gl[0][k]*gl[0][k]*pow(log1p(x)/x, q);
      }
      c1[0][j] = log(c*c*c*f0);
      c1[1][j] = log(c) + p*log(log1p(c)/c);
    }

    // Upsampling (header, item 5): six independent jobs, one per loop
    // iteration. schedule(dynamic, 1) hands each job to the next free
    // thread, so the two large Q jobs run side by side while the small
    // ones fill the other threads. spline2d_upsample_uniform (basics.c)
    // is the 2D pattern, a natural cubic spline along each axis in turn
    // from the padded coarse grid to the dense one sharing its ends;
    // ks_upsample1d is the 1D one, each job with its own workspace
    // cs[0..2].
    #pragma omp parallel for schedule(dynamic, 1)
    for (int job=0; job<6; job++) {
      switch (job) {
        case 0:
          spline2d_upsample_uniform(Qc[0], ncp, nzp, hc, hz, Qd[0], ncd, nzd);
          break;
        case 1:
          spline2d_upsample_uniform(Qc[1], ncp, nzp, hc, hz, Qd[1], ncd, nzd);
          break;
        case 2:
          spline2d_upsample_uniform(Sc, ncp, nwp, hc, hw, Sd, ncd, nwd);
          break;
        case 3:
          ks_upsample1d(lnPc, nyp, hy, cs[0], lnPd, MY);
          break;
        case 4:
          ks_upsample1d(c1[0], n1p, h1, cs[1], d1[0], M1);
          break;
        default:
          ks_upsample1d(c1[1], n1p, h1, cs[2], d1[1], M1);
          break;
      }
    }
    cache[0] = nuisance.random_gas;
    cache[1] = Ntable.random;
  }

  // Lookup (header, items 2 and 4). c is clamped to the tabulated range
  // and z = k r_v, the physical phase, is formed from the arguments as
  // given: a clamped query returns u at the edge concentration and the
  // true phase.
  const double cc = fmin(fmax(c, limits.halo_uks_cmin), limits.halo_uks_cmax);
  const double lc = log(cc);
  const double z = k*rv; // the phase z = k r_v, formed directly
  // Small z: one bilinear read of u(ln c, w) at w = z^2; z < ZSW keeps
  // w inside the used range [0, ZSW^2), so the padding is never read.
  if (z < ZSW) {
    return interpol2d(Sd, ncd, lim[0][0], lim[0][1], lim[0][2], lc,
                      nwd, lim[1][0], lim[1][1], lim[1][2], z*z);
  }
  // Contour formula. y = z/c pairs the true phase with the clamped
  // concentration. lz clamps z to ZHI (Q held at its ZHI value above);
  // ly clamps ln y to its axis, which only acts above ZHI as well, where
  // the P term (order 1/y^4) is negligible against the Q term (order
  // 1/y^2). Every read lands inside a used range, with padded dense
  // nodes on both sides, so the out-of-range paths of interpol1d and
  // interpol2d never run. P, g and F0 come back exponentiated from
  // their logs.
  const double y = z/cc;
  const double lz = log(fmin(z, ZHI));
  const double ly = fmin(fmax(log(y), log(ZSW/limits.halo_uks_cmax)),
                         log(ZHI/limits.halo_uks_cmin));
  const double P = exp(interpol1d(lnPd, nyd, lim[3][0], lim[3][1],
                                  lim[3][2], ly));
  const double RQ = interpol2d(Qd[0], ncd, lim[0][0], lim[0][1], lim[0][2], lc,
                               nzd, lim[2][0], lim[2][1], lim[2][2], lz);
  const double IQ = interpol2d(Qd[1], ncd, lim[0][0], lim[0][1], lim[0][2], lc,
                               nzd, lim[2][0], lim[2][1], lim[2][2], lz);
  const double g = exp(interpol1d(d1[1], n1d, lim[4][0], lim[4][1],
                                  lim[4][2], lc));
  const double F0 = exp(interpol1d(d1[0], n1d, lim[4][0], lim[4][1],
                                   lim[4][2], lc));
  // u = [P - (g/y)(cos z Re Q - sin z Im Q)]/(y F0), header item 2: the
  // oscillation enters only through the exact cos z and sin z.
  return (P - g/y*(cos(z)*RQ - sin(z)*IQ))/(y*F0);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fraction of the halo mass in bound gas (2005.00009 Eq. 25, from
// 1510.06034 Eq. 2.19):
//
//   f_bnd(M) = (Omega_b/Omega_m) / [1 + (M_0/M)^beta]
//
// Massive halos keep their cosmic share of baryons as hot bound gas
// (M >> M_0: f_bnd -> Omega_b/Omega_m); feedback empties light halos
// (M << M_0: f_bnd ~ (Omega_b/Omega_m)(M/M_0)^beta -> 0). A halo of
// mass M_0 keeps half; beta sets how sharp the transition is. HMx
// defaults (2005.00009 sec. 3.2): M_0 = 1e14 M_sun, beta = 0.6;
// 1510.06034 sec. 2.4 fits M_c = 1.2e14 M_sun/h and beta = 0.6 to X-ray
// gas fractions, with masses defined at 200 times the critical density.
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
  const double M0 = pow(10.0, nuisance.gas[2]);
  return cosmology.Omega_b/(cosmology.Omega_m*(1.0+ pow(M0/M, nuisance.gas[1])));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

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
// Clip: where f_bnd + f_* would exceed Omega_b/Omega_m (above about
// 10^15.9 M_sun/h for M_0 = 1e14, beta = 0.6, A_* = 0.03 and
// Omega_b/Omega_m = 0.156) f_ejc is set to 0. The footnote to the f_ejc
// equation of 2005.00009 (sec. 3.2) takes the excess out of the stars,
// which leaves no gas to eject. f_* is a local here, so nothing else
// sees the reduced stellar fraction.
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
  const double logM = log10(M);
  const double delta = (logM - nuisance.gas[7])/nuisance.gas[8];
  
  const double tmp = nuisance.gas[6] * exp(-0.5*delta*delta);  
  const double frac_star = ((logM > nuisance.gas[7]) && 
                           (tmp < nuisance.gas[6]/3.0)) ? nuisance.gas[6]/3.0 : tmp; 
  
  // clip at 0: the 2005.00009 sec. 3.2 footnote takes the excess from
  // the stars, not from the ejected gas
  return fmax(0.0,
              cosmology.Omega_b/cosmology.Omega_m - frac_bnd(M) - frac_star);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
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
// r_v here is r_Delta (Delta = 200 times the mean density), where
// 2005.00009 uses the virial overdensity (its Eq. 22); the free alpha
// absorbs the difference, so alpha values fitted there do not carry
// over one to one.
//
// Parameters:
//   c - concentration r_Delta/r_s
//   k - wavenumber in (c/H0)^-1
//   m - halo mass in M_sun/h
//   a - scale factor
//   (alpha = nuisance.gas[5], f_H = nuisance.gas[10])
//
// Returns:
//   W_p(M, k) in U = G (M_sun/h)^2/(c/H0): the full window, not a
//   profile normalized to 1. In 2005.00009 Eqs. 1-2 it stands where the
//   matter field has W_m = (M/rho_m) u(k|M).
// ---------------------------------------------------------------------------
double u_y_bnd(
    double c, // concentration r_Delta/r_s
    double k, // wavenumber in (c/H0)^-1
    double m, // halo mass in M_sun/h
    double a  // scale factor (comoving r_v -> physical, in T_v)
  )
{
  
  const double rho_delta = Delta * cosmology.rho_crit * cosmology.Omega_m;
  const double r_delta = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
  const double rv = r_delta;

  const double mu_p = 4.0/(3.0 + 5*nuisance.gas[10]);
  const double mu_e = 2.0/(1.0 + nuisance.gas[10]);
  
  return (2.0*nuisance.gas[5]/(3.0*a))*(mu_p/mu_e)*frac_bnd(m)*m*(m/rv)*u_KS(c, k, rv);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

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
// Unit chain, landing on the units of u_y_bnd so the two windows add:
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
  const double num_p = 1.1892e57;
  
  const double E_w = pow(10,nuisance.gas[9]) * 8.6173e-5 * 5.616e-44;
  const double mu_e = 2./(1.+nuisance.gas[10]);
  
  return (num_p * m * frac_ejc(m) / mu_e) * E_w;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Comoving number density of source galaxies at scale factor a: the
// angular density per unit redshift divided by the comoving volume per
// unit redshift and solid angle,
//
//   n(a) = n_gal n_src(z) / [dV/(dz dOmega)],   z = 1/a - 1
//
//   n_gal          = survey.n_gal times survey.n_gal_conversion_factor
//                    (arcmin^-2 -> sr^-1)
//   n_src(z)       = nz_source_photoz(z, -1), intended as the all-bin
//                    source redshift distribution with unit integral
//   dV/(dz dOmega) = f_K(chi)^2 dchi/dz = f_K(chi)^2/(H/H0),
//                    in (c/H0)^3
//
// nz_source_photoz aborts for nj < 0 (redshift_spline.c), so this
// function cannot run as written; it has no caller and no declaration
// in halo.h.
//
// Parameters:
//   a - scale factor
//
// Returns:
//   n in (c/H0)^-3
// ---------------------------------------------------------------------------
double n_s_cmv(
    double a  // scale factor
  )
{ 
  double dV_dz = pow(f_K(chi(a)), 2.0) / hoverh0(a);
  return nz_source_photoz(1.0/a - 1., -1) * survey.n_gal * 
    survey.n_gal_conversion_factor / dV_dz;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO MODEL ROUTINES
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_hm_funcs(double lnM, void* params)
{ // 0 = ngal, 1 = m_mean, 2 = fsat, 3 = bgal 
  double* ar = (double*) params;
  
  const double a = ar[0];
  const int ni = (int) ar[1];
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  const int func = (int) ar[2];
  const double growfac_a = (double) ar[3];
  const double m = exp(lnM);
  
  const double nu = delta_c/(sqrt(sigma2(m))*growfac_a);
  const double gnu  = fnu(nu, a) * nu; 
  const double rhom = cosmology.rho_crit * cosmology.Omega_m;
  const double dNdlnM = gnu * (rhom/m) * dlognudlogm(m);
  
  const double nc = HOD_fc(ni)*HOD_nc(m, a, ni);
  const double ns = HOD_ns(m, a, ni);
  
  double res;
  switch(func)
  {
    case 0:
    { // N_gal = \int dM n(M)*(nc + ns) = \int dlnM M n(M)*(nc + ns) 
      res = dNdlnM*(nc + ns);
      break;
    }
    case 1:
    { // <M> = \int dM M*n(M)*(nc + ns) = \int dlnM M^2 n(M)*(nc + ns) 
      res = m*(dNdlnM*(nc + ns));
      break;
    }
    case 2:
    {
      res = dNdlnM*ns;
      break;
    }
    case 3:
    {
      res = hb1nu(nu, a)*(dNdlnM*(nc + ns));
      break;
    }
    default:
    {
      log_fatal("option not supported");
      exit(1);
    }
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double ngal_nointerp(
    const int ni, 
    const double a, 
    const int init
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = 1024; // largest predefined GSL table
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[4] = {a, (double) ni, (double) 0, growfac(a)};
  const double lnMmin = log(10.0)*(nuisance.hod[ni][0] - 2.);
  const double lnMmax = log(limits.halo_m_max);

  double res = 0.0;
  if (1 == init) {
    res = int_hm_funcs((lnMmin + lnMmax)/2.0, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_hm_funcs;
    res = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// Table owner of ngal and bgal: static state, zeroed at program start so
// the first hod_tables call builds everything. Two blocks of hod_tables
// write it (its header, Cache invalidation):
//   rebuild block   nbin, nq, lim, gl and every allocation
//   refill block    nd, D, pb, pf, tab, then the tags cache[]
// ---------------------------------------------------------------------------
static struct {
  uint64_t cache[MAX_SIZE_ARRAYS]; // [0] cosmology, [1] Ntable, [2] HOD,
                                   //   [3] clustering n(z) tags
  int nbin;               // lens bins of the allocation
  int nq;                 // Gauss-Legendre nodes per lens bin
  double lim[3];          // a grid: min, max, step
  double*** tab;          // [2][nbin][N_a] ngal (0), bgal (1)
  double*** nd;           // [2][nbin][nq] per node: nu at D = 1 (0),
                          //   hw w_q (rho_m/M) dlnnu/dlnM (f_c N_c + N_s) (1)
  double** gl;            // [2][nq] Gauss-Legendre nodes (0), weights (1)
                          //   on [-1, 1]
  double* D;              // [N_a] growth factor D(a)
  hb1nu_params* pb;       // [N_a] Tinker bias parameters (hb1nu_params_at)
  fnu_params* pf;         // [N_a] Tinker multiplicity parameters
                          //   (fnu_params_at)
} hod_ = {0};

// ---------------------------------------------------------------------------
// Fills hod_: number density and mean halo bias of the galaxies of every
// lens bin on one grid in a, read by ngal and bgal:
//
//   ngal(a) = int dlnM dn/dlnM [f_c N_c(M) + N_s(M)]              (c/H0)^-3
//   bgal(a) = int dlnM dn/dlnM [f_c N_c(M) + N_s(M)] b(nu) / ngal(a)
//
// dn/dlnM = (rho_m/M) nu f(nu) dln nu/dln M, nu = delta_c/(sigma(M) D(a)),
// is the mass function of int_for_I02_XY (its header, item 1); f_c, N_c,
// N_s the occupation of the GALAXY PROFILES banner (HOD_fc, HOD_nc,
// HOD_ns); b(nu) the Tinker bias (hb1nu). ln M runs from two decades
// below the bin's M_min (N_c is an erf tail there) to ln M_max.
// ngal_nointerp and bgal_nointerp are the same integrals done directly
// at 1024 nodes (Python-facing diagnostics); the tables do not call them.
//
// 1. Quadrature: the n-point Gauss-Legendre rule in ln M (exact for
// polynomials of degree 2n - 1); nodes x_q and weights w_q on [-1, 1]
// (gl) are mapped per bin as ln M_q = mid + hw x_q, weight hw w_q.
// n = 128 / 256 / 512 / 1024 for hdi = abs(Ntable.high_def_integration)
// = 0 / 1 / 2 / >= 3, sizes GSL tabulates.
//
// - Measured 2026-09-29 (5 lens bins x a = 0.5, 0.75, 0.95, vs 32-node
//   panels 0.05 wide in ln M): 5e-7 / 1.3e-7 / 3e-8 / 5e-9 relative at
//   128 / 256 / 512 / 1024 nodes. Splitting the range at M_min and M_0
//   does not help: the floor is the linear read of the sigma2 and
//   dlognudlogm tables (a kink per cell; file header), not the HOD shape;
//   the linear read in a (item 3), up to 1.4e-5, dominates at 128 nodes.
//
// 2. Loop levels: a enters only through nu = nu0(M)/D(a) and the Tinker
// parameters of f and b; the occupation does not depend on a (HOD_nc and
// HOD_ns only range-check it: the placeholder hod_.lim[0]). Each factor
// is computed at the outermost level it depends on:
//
//   per refill, per (bin, node q)  nu0_q = delta_c/sigma(M_q)          nd[0]
//                                  P_q = hw w_q (rho_m/M_q) dlnnu/dlnM
//                                        (f_c N_c + N_s)               nd[1]
//   per a node j                   D_j, Tinker f and b parameters   D, pf, pb
//   per (bin, j), threaded         nu = nu0_q/D_j, t_q = P_q f(nu) nu:
//                                  ngal = sum t_q                      tab[0]
//                                  bgal = sum t_q b(nu) / ngal         tab[1]
//
// 3. The a grid: Ntable.N_a nodes uniform in a over [1/(1 + z_max),
// 1/(1 + z_min)] of the clustering n(z), all bins; the readers return 0
// outside and read linearly inside (interpol1d: uniform grid, direct
// index).
//
// - Measured 2026-09-29 (4 threads, 10 lens bins x N_a = 256): one
//   refill 3.9 ms.
//
// Thread safety: the refill calls sigma2, dlognudlogm, growfac and
// fnu_params_at (the tinker_alpha table) serially before the threaded
// loop, so their lazy tables are built outside the parallel region (the
// warm-up rule of the cosmo2D.c _work functions). The first call is the
// single-threaded init = 1 pass of p_gm_nointerp and p_gg_nointerp.
//
// Cache invalidation:
//   rebuild block (sizes, GL nodes, a grid; every allocation lives here):
//     Ntable.random (cache[1]) or redshift.random_clustering (cache[3])
//   refill: those two, cosmology.random (cache[0]) or the HOD tag
//     nuisance.random_galaxy_bias (cache[2])
// ---------------------------------------------------------------------------
static void hod_tables(void)
{
  // Rebuild block: the first call (tab is NULL from the = {0}), or the
  // Ntable or clustering-n(z) tag differs from the allocation's. Every
  // malloc lives here; malloc1d/2d/3d return one block each, pointer rows
  // included, so one free releases a table.
  if (NULL == hod_.tab ||
      fdiff2(hod_.cache[1], Ntable.random) ||
      fdiff2(hod_.cache[3], redshift.random_clustering))
  {
    if (hod_.tab != NULL) {
      free(hod_.tab);
      free(hod_.nd);
      free(hod_.gl);
      free(hod_.D);
      free(hod_.pb);
      free(hod_.pf);
    }
    // Node count n (header, item 1) and the sizes of the allocation.
    const int hdi = abs(Ntable.high_def_integration);
    hod_.nbin = redshift.clustering_nbin;
    hod_.nq = (0 == hdi) ? 128 :
              (1 == hdi) ? 256 :
              (2 == hdi) ? 512 : 1024; // predefined GSL tables
    hod_.tab = (double***) malloc3d(2, hod_.nbin, Ntable.N_a);
    hod_.nd = (double***) malloc3d(2, hod_.nbin, hod_.nq);
    hod_.gl = (double**) malloc2d(2, hod_.nq);
    hod_.D = (double*) malloc1d(Ntable.N_a);
    hod_.pb = (hb1nu_params*) malloc(sizeof(hb1nu_params)*Ntable.N_a);
    hod_.pf = (fnu_params*) malloc(sizeof(fnu_params)*Ntable.N_a);
    // Gauss-Legendre nodes and weights on [-1, 1] (header, item 1):
    // malloc_gslint_glfixed(n) wraps the GSL table of n nodes and
    // gsl_integration_glfixed_point copies node q and its weight out. The
    // stretch onto each bin's ln M range happens in the refill.
    gsl_integration_glfixed_table* t = malloc_gslint_glfixed(hod_.nq);
    for (int q=0; q<hod_.nq; q++) {
      gsl_integration_glfixed_point(-1.0, 1.0, q, &hod_.gl[0][q],
                                    &hod_.gl[1][q], t);
    }
    gsl_integration_glfixed_table_free(t);
    // The a grid (header, item 3): min, max, step; node j sits at
    // lim[0] + j lim[2], both ends included.
    hod_.lim[0] = 1.0/(redshift.clustering_zdist_zmax_all + 1.0);
    hod_.lim[1] = 1.0/(redshift.clustering_zdist_zmin_all + 1.0);
    hod_.lim[2] = (hod_.lim[1] - hod_.lim[0])/((double) Ntable.N_a - 1.0);
  }
  // Refill block: any of the four tags differs from the one the tables
  // hold.
  if (fdiff2(hod_.cache[0], cosmology.random) ||
      fdiff2(hod_.cache[1], Ntable.random) ||
      fdiff2(hod_.cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(hod_.cache[3], redshift.random_clustering))
  {
    const int nbin = hod_.nbin;
    const int na = Ntable.N_a;
    const int nq = hod_.nq;
    const double rhom = cosmology.rho_crit * cosmology.Omega_m;

    // Per (bin, node), serially: the a-independent factors of header
    // item 2. ln M_q = mid + hw x_q covers [ln 10^(lg M_min - 2),
    // ln limits.halo_m_max]; hod_.lim[0] is an a in (0, 1) for the range
    // check of HOD_nc, which aborts on a bin whose HOD is not set
    // (lg M_min outside [10, 16]): the tables cover all bins at once. The
    // sigma2 and dlognudlogm reads happen here, before the threads
    // (header, Thread safety).
    for (int b=0; b<nbin; b++) {
      const double lnMmin = log(10.0)*(nuisance.hod[b][0] - 2.);
      const double lnMmax = log(limits.halo_m_max);
      const double hw = 0.5*(lnMmax - lnMmin);
      const double mid = 0.5*(lnMmax + lnMmin);
      const double fc = HOD_fc(b);
      for (int q=0; q<nq; q++) {
        const double lnM = mid + hw*hod_.gl[0][q];
        const double m = exp(lnM);
        const double occ = fc*HOD_nc(m, hod_.lim[0], b) +
                           HOD_ns(m, hod_.lim[0], b);
        hod_.nd[0][b][q] = delta_c/sqrt(sigma2(m));
        hod_.nd[1][b][q] = hw*hod_.gl[1][q]*(rhom/m)*dlognudlogm(m)*occ;
      }
    }

    // Per a node, serially: D(a) and the nu-independent halves of the
    // two Tinker kernels (the params/core split of the section banner);
    // growfac and fnu_params_at run here, before the threads.
    for (int j=0; j<na; j++) {
      const double a = hod_.lim[0] + j*hod_.lim[2];
      hod_.D[j] = growfac(a);
      hod_.pb[j] = hb1nu_params_at(a);
      hod_.pf[j] = fnu_params_at(a);
    }

    // Per (bin, a), threaded: the node sum of header item 2. collapse(2)
    // makes the (b, j) pairs one iteration space, cut into contiguous
    // static chunks; each entry is its own serial sum, so the result does
    // not depend on the thread count. restrict: the node arrays are
    // reached only through v0 and p, so no node is reloaded after each
    // libm call inside the two kernels.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int b=0; b<nbin; b++) {
      for (int j=0; j<na; j++) {
        const double* restrict v0 = hod_.nd[0][b];
        const double* restrict p = hod_.nd[1][b];
        const double D = hod_.D[j];
        const hb1nu_params* pb = &hod_.pb[j];
        const fnu_params* pf = &hod_.pf[j];
        double sn = 0.0;
        double sb = 0.0;
        for (int q=0; q<nq; q++) {
          const double nu = v0[q]/D;
          const double tq = p[q]*fnu_core(nu, pf)*nu; // hw w_q dn/dlnM <N|M>
          sn += tq;
          sb += tq*hb1nu_core(nu, pb);
        }
        hod_.tab[0][b][j] = sn;     // ngal
        hod_.tab[1][b][j] = sb/sn;  // bgal
      }
    }
    // Tags of the inputs the tables hold.
    hod_.cache[0] = cosmology.random;
    hod_.cache[1] = Ntable.random;
    hod_.cache[2] = nuisance.random_galaxy_bias;
    hod_.cache[3] = redshift.random_clustering;
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Comoving number density of the galaxies of lens bin ni at scale factor
// a: the ngal(a) integral of the hod_tables header, read linearly from
// hod_.tab[0][ni] on its a grid (interpol1d); 0 outside [1/(1 + z_max),
// 1/(1 + z_min)] of the clustering n(z).
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
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  hod_tables();
  return ((a < hod_.lim[0]) || (a > hod_.lim[1])) ? 0.0 :
    interpol1d(hod_.tab[0][ni], Ntable.N_a, hod_.lim[0], hod_.lim[1],
               hod_.lim[2], a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double hm_funcs_nointerp(
    const int ni, 
    const double a, 
    const int func,
    const int init
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }

  if (w == NULL || fdiff2(cache[0], Ntable.random))
  {
    const size_t szint = 1024; // largest predefined GSL table
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[4] = {a, (double) ni, (double) func, growfac(a)}; 
  const double lnMmin = log(10.0)*(nuisance.hod[ni][0] - 2.);
  const double lnMmax = log(limits.halo_m_max);

  double res = 0.0;
  if (init == 1)
    res = int_hm_funcs((lnMmin + lnMmax)/2.0, (void*) ar);
  else
  {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_hm_funcs;
    res = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
  }
  // funcs 1-3 are number-weighted means (mean mass, satellite fraction,
  // mean bias): divide by the same bin's number density, integrated
  // directly so one bin never forces the all-bin ngal table build
  return (func == 1 || func == 2 || func == 3) ?
    res/ngal_nointerp(ni, a, 0) : res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double mmean_nointerp(
    const int ni, 
    const double a, 
    const int init
  )
{
  return hm_funcs_nointerp(ni, a, 1, init);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double fsat_nointerp(
    const int ni, 
    const double a, 
    const int init
  )
{
  return hm_funcs_nointerp(ni, a, 2, init);
} 

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double bgal_nointerp(
    const int ni, 
    const double a, 
    const int init
  )
{
  return hm_funcs_nointerp(ni, a, 3, init);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Mean halo bias of the galaxies of lens bin ni at scale factor a: the
// number-weighted bgal(a) of the hod_tables header, read linearly from
// hod_.tab[1][ni] on its a grid (interpol1d); 0 outside [1/(1 + z_max),
// 1/(1 + z_min)] of the clustering n(z). The large-scale galaxy bias of
// p_gm and p_gg.
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
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  hod_tables();
  return ((a < hod_.lim[0]) || (a > hod_.lim[1])) ? 0.0 :
    interpol1d(hod_.tab[1][ni], Ntable.N_a, hod_.lim[0], hod_.lim[1],
               hod_.lim[2], a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Integrand of I02_XY_nointerp, the 1-halo integral I^0_2 of the file
// glossary, in the variable ln M:
//
//   int_for_I02_XY(ln M) = dn/dlnM  W_X(M, k1)  W_Y(M, k2),
//
// so that its integral over ln M is int dM n(M) W_X(M, k1) W_Y(M, k2),
// the 1-halo term of P_XY (2005.00009 Eq. 2; astro-ph/0012087 Eq. 14
// with delta_halo = W): a halo of mass M contributes the product of its
// two windows, weighted by how many such halos there are per volume.
//
// 1. The mass function
//
// dn/dlnM is the comoving number of halos per unit ln M per (c/H0)^3,
// built from the multiplicity function of the section banner:
//
//   dn/dlnM = (rho_m/M) nu f(nu) (dln nu/dln M),
//   nu      = delta_c/(sigma(M) D(a)).
//
// Right to left:
//
//   f(nu) dnu       fraction of all matter in halos of peak height
//                   [nu, nu + dnu] (fnu)
//   nu              dnu = nu dln nu: "per unit nu" -> "per unit ln nu";
//                   gnu = nu f(nu) in the code
//   dln nu/dln M    "per unit ln nu" -> "per unit ln M" (dlognudlogm,
//                   an a = 1 table: its header)
//   rho_m/M         a mass fraction over the mass of one halo is a
//                   number density
//
// growfac_a arrives in params so that D(a) is not recomputed at each of
// the 1024 nodes.
//
// 2. The windows: one M/rho_m per matter leg
//
// W_X(M, k) is the Fourier transform of the profile of field X in a
// halo of mass M (2005.00009 Eq. 4), in the units of the field times a
// volume:
//
//   matter    W_m(M, k) = (M/rho_m) u_c(k|M)    a volume, (c/H0)^3
//   pressure  W_y(M, k) = u_y_bnd(M, k)         an energy, in U
//
// Matter: the profile is the overdensity rho(r)/rho_m, whose volume
// integral is M/rho_m, and u_c is that profile normalized to 1 at
// k = 0, so W_m -> M/rho_m as k -> 0.
//
// Pressure: the profile is the pressure itself and u_y_bnd is its full
// volume integral (u_y_bnd header), gas mass f_bnd M times a virial
// temperature ~ M^(2/3):
//
//   W_y ~ M^(5/3)   as k -> 0   (2005.00009 Eq. 41).
//
// No M/rho_m belongs in front of it; with one, a halo's pressure would
// scale as M^(8/3) and P_my, P_yy would be weighted toward even heavier
// halos than they are. The factor vol below encodes this:
//
//   XY = 0  mm   u = u_c(k1) u_c(k2)           vol = (M/rho_m)^2
//   XY = 1  my   u = u_y_bnd(k1) u_c(k2)       vol = M/rho_m
//   XY = 2  yy   u = u_y_bnd(k1) u_y_bnd(k2)   vol = 1
//
// The ejected gas has no 1-halo term (u_y_ejc header; 2005.00009
// sec. 3.3), so only the bound gas appears here.
//
// Parameters:
//   lnM    - ln of the halo mass in M_sun/h (the GSL node)
//   params - double[5] {a, k1, k2, XY, D(a)}, packed by I02_XY_nointerp
//            and handed over by GSL as an untyped void*
//
// Returns:
//   dn/dlnM W_X(M, k1) W_Y(M, k2): (c/H0)^3 for mm, U for my,
//   U^2 (c/H0)^-3 for yy, with U = G (M_sun/h)^2/(c/H0)
// ---------------------------------------------------------------------------
double int_for_I02_XY(double lnM, void* params)
{
  double* ar = (double*) params;
  const double a = ar[0];
  const double k1 = ar[1];
  const double k2 = ar[2];
  const int XY = (int) ar[3];
  const double growfac_a = ar[4];
  const double m = exp(lnM);
  
  const double nu = delta_c/(sqrt(sigma2(m))*growfac_a);
  const double gnu  = fnu(nu, a) * nu; 
  const double rhom = cosmology.rho_crit * cosmology.Omega_m;
  const double dNdlnM = gnu * (rhom/m) * dlognudlogm(m); // mass function

  const double c = conc(m, growfac_a);

  double u;   // product of the two profiles
  double vol; // one m/rho_m per matter leg (header, item 2)
  switch(XY)
  {
    case 0:
    { // matter-matter
      u = u_c(c, k1, m, a) * u_c(c, k2, m, a);
      vol = (m/rhom) * (m/rhom);
      break;
    }
    case 1:
    { // matter-y
      u = u_y_bnd(c, k1, m, a) * u_c(c, k2, m, a);
      vol = m/rhom;
      break;
    }
    case 2:
    { // y-y 
      u = u_y_bnd(c, k1, m, a) * u_y_bnd(c, k2, m, a);
      vol = 1.0;
      break;
    }
    default:
    {
      log_fatal("option not supported"); exit(1);
    }
  }
  return dNdlnM * u * vol;
}  

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// I02_XY(k1, k2, a) = I^0_2, the 1-halo integral of the file glossary,
//
//   I02_XY = int_{ln M_min}^{ln M_max} dlnM  dn/dlnM W_X(M, k1) W_Y(M, k2)
//
// over M in [limits.halo_m_min, limits.halo_m_max] (1e6 to 1e17 M_sun/h
// by default) with the integrand of int_for_I02_XY: the 1-halo term of
// P_XY (2005.00009 Eq. 2). p_xy_nointerp reads it at k1 = k2 = k.
//
// Quadrature: Gauss-Legendre with 1024 nodes in ln M, the largest size
// GSL tabulates, at every hdi (file header), because the integrand
// reads sigma2 and dlognudlogm by linear interpolation in ln M and GL
// converges only algebraically across those kinks.
//
// gsl_integration_glfixed(&F, lo, hi, w) evaluates F.function at the
// 1024 nodes of table w mapped onto [lo, hi] and returns the weighted
// sum; w holds nodes and weights on [-1, 1] and is built once per
// Ntable.random.
//
// Why the finite mass range is harmless here and not in I11_X: the two
// halo terms weight a halo differently,
//
//   1-halo   W_X W_Y   ~ M^2 for matter   (this integral)
//   2-halo   W_X       ~ M   for matter   (I11_X_nointerp)
//
// The halos below M_min hold about 20% of the bias-weighted matter at
// z = 0 (bias_norm header, item 1). Weighted by M^2 they add a
// negligible share of the 1-halo term, which is carried by the mass a
// typical unit of matter sits in, ~1e13.5 M_sun at z = 0 (2005.00009
// sec. 2.3 and App. A), well inside the range. Weighted by M they do
// matter: I11_X_nointerp header, item 1.
//
// init = 1: one evaluation of the integrand at the midpoint of ln M,
// single-threaded, value thrown away. It triggers the lazy builds the
// integrand reaches,
//
//   sigma2, dlognudlogm    their tables
//   fnu                    the tinker_alpha table
//   u_c                    the NFW f, G table (u_nfw_c)
//   u_y_bnd                the u_KS table
//
// before p_mm, p_my and p_yy fill their tables in parallel: the
// warm-up rule of the cosmo2D.c _work functions (halo_wrapper.hpp,
// "The init flag").
//
// Cache invalidation:
//   Gauss-Legendre table: rebuilt when Ntable.random changes. No value
//   is cached here; every init = 0 call integrates.
//
// Parameters:
//   k1, k2 - wavenumbers in (c/H0)^-1 of the X and Y legs
//   a      - scale factor, 0 < a < 1 (fnu aborts otherwise)
//   func   - 0 = mm, 1 = my (X = y at k1, Y = m at k2), 2 = yy;
//            other values abort inside the integrand
//   init   - 1 = warm-up evaluation (above), 0 = the integral
//
// Returns:
//   I02_XY: (c/H0)^3 for mm, U for my, U^2 (c/H0)^-3 for yy
// ---------------------------------------------------------------------------
double I02_XY_nointerp(
    const double k1,
    const double k2,
    const double a,
    const int func, // 0 = MM, 1 = MY, 2 = YY
    const int init
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = 1024; // largest predefined GSL table
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[5] = {a, k1, k2, func, growfac(a)};
  const double lnMmin = log(limits.halo_m_min);
  const double lnMmax = log(limits.halo_m_max);

  double res;
  if (1 == init) {
    res = int_for_I02_XY((lnMmin + lnMmax)/2.0, (void*) ar);
  }
  else
  {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_I02_XY;
    res = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Integrand of I11_X_nointerp, the 2-halo integral I^1_1 of the file
// glossary, in the variable ln M:
//
//   int_for_I11_X(ln M) = dn/dlnM  b(nu)  W_X(M, k),
//
// so that its integral over ln M is int dM n(M) b(M) W_X(M, k): the
// window of field X in a halo of mass M, weighted by how many such
// halos there are and by how strongly they cluster (b, the linear halo
// bias of hb1nu).
//
// Two of these and P_lin make the 2-halo term (2005.00009 Eq. 1;
// astro-ph/0012087 Eq. 15),
//
//   P_2h = I11_X I11_Y P_lin:
//
// the two points sit in two different halos, each halo stands in for
// its field, and the pair is correlated through the linear density
// field both halos trace.
//
// dn/dlnM is the mass function of int_for_I02_XY (its header, item 1)
// and the windows follow the same convention (its item 2): one M/rho_m
// on the matter leg, none on the pressure leg.
//
//   func = 0  matter    W_m = (M/rho_m) u_c(k|M)
//   func = 1  pressure  W_y = u_y_bnd(M, k) + u_y_ejc(M)
//
// The ejected gas appears here and not in the 1-halo term: it follows
// the linear field outside halos, so it is a point-like, k-independent
// window that clusters with the halo it came from (u_y_ejc header;
// 2005.00009 sec. 3.3 and Eq. 36).
//
// The integrand carries no normalization: the finite mass range misses
// the halos below M_min, and I11_X_nointerp adds their share after the
// sum (its header, item 2).
//
// Parameters:
//   lnM    - ln of the halo mass in M_sun/h (the GSL node)
//   params - double[4] {a, k, func, D(a)}, packed by I11_X_nointerp
//
// Returns:
//   dn/dlnM b(nu) W_X(M, k): dimensionless for matter, U (c/H0)^-3 for
//   the pressure
// ---------------------------------------------------------------------------
double int_for_I11_X(double lnM, void* params)
{
  const double* ar = (double*) params;  
  const double a = ar[0];
  const double k = ar[1];
  const int func = (int) ar[2];
  const double growfac_a = ar[3];
  const double m = exp(lnM);
  
  const double nu = delta_c/(sqrt(sigma2(m))*growfac_a);
  const double gnu = fnu(nu, a) * nu; 
  const double rhom = cosmology.rho_crit * cosmology.Omega_m;
  const double dNdlnM = gnu * (rhom/m) * dlognudlogm(m);

  const double c = conc(m, growfac_a);

  double w; // the window W_X(m, k)
  switch(func)
  {
    case 0:
    { // matter: W_m = (m/rho_m) u_c
      w = u_c(c, k, m, a) * (m/rhom);
      break;
    }
    case 1:
    { // electron pressure: W_y = u_y_bnd + u_y_ejc, no m/rho_m
      w = u_y_bnd(c, k, m, a) + u_y_ejc(m);
      break;
    }
    default:
    {
      log_fatal("option not supported"); exit(1);
    }
  }
  return dNdlnM * w * hb1nu(nu, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// I11_X(k, a) = I^1_1, the 2-halo amplitude of field X (file glossary):
//
//   I11_X(k) = int_{M_min}^{M_max} dM n(M) b(M) W_X(M, k)
//            + A(a) W_X(M_min, k)/(M_min/rho_m),    A(a) = 1 - bias_norm(a)
//
// The first line runs over [limits.halo_m_min, limits.halo_m_max] with
// the integrand of int_for_I11_X (Gauss-Legendre in ln M, as in
// I02_XY_nointerp); the second is the correction of Mead et al. 2020
// (HMx, 2005.00009 App. A) for the halos the integral does not reach.
// p_xy_nointerp multiplies two of these with P_lin (2005.00009 Eq. 1).
//
// 1. Why the matter integral must tend to 1 as k -> 0
//
// On scales larger than any halo u_c(k|M) -> 1 and W_m -> M/rho_m
// (int_for_I02_XY header, item 2), so I11_m(k -> 0) = int b f dnu,
// which is 1 over all nu because matter is unbiased with respect to
// itself (fnu header, item 2): P_2h -> P_lin as k -> 0.
//
// 2005.00009 Eq. 1 integrates over all M; this file starts at
// M_min = 1e6 M_sun/h and the Tinker f grows toward light halos, so the
// covered share is (bias_norm header, item 1)
//
//   z            0      0.5    1      2      3
//   bias_norm    0.80   0.80   0.79   0.75   0.70
//
// and the integral alone would give P_2h -> 0.64 P_lin at z = 0; App. A
// finds 0.67 for a standard mass function integrated from 1e10 M_sun.
//
// 2. The correction: the missing matter as halos of mass M_min
//
// What not to do (2005.00009 App. A): multiply n(M) above M_min by
// 1/bias_norm = 1.25. Massive halos have resolved profiles (u < 1) at k
// where light halos are still points, so moving the light halos' matter
// into them changes the k-dependence of both halo terms. Dividing the
// 2-halo integrand by bias_norm does the same to the 2-halo term alone.
//
// HMx's second option keeps n(M) and adds the missing share A as halos
// of mass exactly M_min (its Eq. A7),
//
//   n(M) -> n(M) + A delta_D(M - M_min)/[b(M_min) M_min/rho_m],
//
// the denominator making the bias-weighted mass of the added term
// exactly A. In the I11 integral b(M_min) cancels (Eq. A8), leaving
// the second line of the formula at the top.
//
// For matter W_m(M_min, k)/(M_min/rho_m) = u_c(k|M_min): at k -> 0 the
// correction is A and I11_m = 1 at every a; at high k it fades like
// u_c(k|M_min), slowly, because the lightest halos are also the
// smallest:
//
//   r_Delta = 2.4 kpc/h   at 1e6 M_sun/h, Omega_m = 0.3,
//   k r_Delta = 0.24      at k = 100 h/Mpc,
//
// so the added halos stay point-like over the k range of the spectra,
// as the halos below M_min are (App. A: delta-function profiles).
//
// For the pressure, W_y = u_y_bnd + u_y_ejc at M_min. The bound part is
// negligible, f_bnd(1e6 M_sun/h) = 2.5e-6 with the HMx defaults (App. A
// judges it ignorable); the ejected part, u_y_ejc(M)/(M/rho_m) = num_p
// rho_m f_ejc(M) E_w/mu_e, does not depend on M, so the light halos put
// their ejected gas back as the resolved halos do (u_y_ejc header).
//
// Size of the effect on p_mm, additive form against the multiplicative
// alternative (integrand divided by bias_norm):
//
//   k [h/Mpc]    1e-3    0.3      100
//   a = 0.3      0       +0.1%    +1.3%
//   a = 0.99     0       +0.2%    +0.4%
//
// At k = 1e-3 both give I11_m = 1 (p_mm/p_lin agrees to 1e-6); the
// additive form keeps the missing matter unresolved (u ~ 1) to high k,
// where the multiplicative form has spread it over resolved profiles.
//
// 3. The pieces in the code
//
//   wmin            W_X at M_min with conc(M_min, D(a)), the same
//                   expression int_for_I11_X evaluates at a node
//   mmin/rhom       M_min/rho_m, so wmin/(mmin/rhom) is u_c(k|M_min)
//                   for matter
//
// The correction is added on the init = 1 path as well, so the
// single-threaded warm-up call of the table builders (halo_wrapper.hpp,
// "The init flag") builds bias_norm's table before the parallel fills.
//
// Cache invalidation:
//   Gauss-Legendre table: rebuilt when Ntable.random changes. No value
//   is cached here; bias_norm keeps its own table.
//
// Parameters:
//   k    - wavenumber in (c/H0)^-1
//   a    - scale factor, 0 < a < 1 (fnu aborts otherwise)
//   func - 0 = matter, 1 = electron pressure; other values abort
//   init - 1 = warm-up evaluation (integrand at the midpoint of ln M
//          plus the correction), 0 = the integral plus the correction
//
// Returns:
//   I11_X(k, a): dimensionless for matter, 1 as k -> 0; in U for the
//   pressure
// ---------------------------------------------------------------------------
double I11_X_nointerp(
    const double k,
    const double a,
    const int func,
    const int init
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = 1024; // largest predefined GSL table
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[4] = {a, k, func, growfac(a)};
  const double lnMmin = log(limits.halo_m_min);
  const double lnMmax = log(limits.halo_m_max);
  
  double res;
  if (1 == init) {
    res = int_for_I11_X((lnMmin + lnMmax)/2.0, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_I11_X;
    res = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
  }

  // The HMx correction (header, item 2): A W_X(M_min, k)/(M_min/rho_m)
  // with A = 1 - bias_norm(a); wmin is W_X(M_min, k) with the window
  // convention of int_for_I11_X. Added on the init = 1 path too, so the
  // single-threaded warm-up builds bias_norm's table first (item 3).
  const double mmin = limits.halo_m_min;
  const double rhom = cosmology.rho_crit * cosmology.Omega_m;
  const double cmin = conc(mmin, ar[3]);
  double wmin;
  switch(func)
  {
    case 0:
    { // matter: W_m = (M_min/rho_m) u_c
      wmin = u_c(cmin, k, mmin, a) * (mmin/rhom);
      break;
    }
    case 1:
    { // electron pressure: W_y = u_y_bnd + u_y_ejc
      wmin = u_y_bnd(cmin, k, mmin, a) + u_y_ejc(mmin);
      break;
    }
    default:
    {
      log_fatal("option not supported"); exit(1);
    }
  }
  return res + (1.0 - bias_norm(a)) * wmin/(mmin/rhom);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_for_G02(double lnM, void* param)
{
  double* ar = (double*) param;
  
  const double k = ar[0];
  const double a = ar[1];
  const int ni = (int) ar[2];
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  const double growfac_a = ar[3];
  const double m  = exp(lnM);

  const double nu = delta_c/(sqrt(sigma2(m))*growfac_a);
  const double gnu = fnu(nu, a) * nu; 
  const double rhom = cosmology.rho_crit * cosmology.Omega_m;
  const double dNdlnM = gnu * (rhom/m) * dlognudlogm(m);

  const double c  = conc(m, growfac_a);
  const double u  = u_g(c, k, m, a, ni);
  const double ns = HOD_ns(m, a, ni);
  const double nc = HOD_nc(m, a, ni);
  const double fc = HOD_fc(ni);

  return dNdlnM*(u*u*ns*ns + 2.0*u*ns*nc*fc);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double G02_nointerp(
    double k, 
    double a, 
    int ni, 
    const int init
  )
{ //needs to be divided by ngal^2
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = 1024; // largest predefined GSL table
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[4] = {k, a, (double) ni, growfac(a)};
  const double lnMmin = log(limits.halo_m_min);
  const double lnMmax = log(limits.halo_m_max);

  double res;
  if (1 == init) {
    res = int_for_G02((lnMmin + lnMmax)/2.0, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_for_G02;
    res = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double int_GM02(double lnM, void* params)
{ // 1-halo galaxy-matter spectrum
  double* ar = (double*) params;
  
  const double k = ar[0];
  const double a = ar[1];
  const int ni = (int) ar[2];
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  const double growfac_a = ar[3];
  const double m = exp(lnM);

  const double nu = delta_c/(sqrt(sigma2(m))*growfac_a);
  const double gnu  = fnu(nu, a) * nu; 
  const double rhom = cosmology.rho_crit * cosmology.Omega_m;
  const double dNdlnM = gnu * (rhom/m) * dlognudlogm(m);

  const double c = conc(m, growfac_a);
  const double ns = HOD_ns(m, a, ni);
  const double nc = HOD_nc(m, a, ni);
  const double fc = HOD_fc(ni);

  return dNdlnM*(m/rhom)*u_c(c,k,m,a)*(u_g(c,k,m,a,ni)*ns + nc*fc);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double GM02_nointerp(
    double k, 
    double a, 
    int ni, 
    const int init
  )
{ // needs to be divided by ngal
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni);
    exit(1);
  }

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const size_t szint = 1024; // largest predefined GSL table
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[4] = {k, a, (double) ni, growfac(a)};
  const double lnMmin = log(10.)*(nuisance.hod[ni][0] - 1.0);
  const double lnMmax = log(limits.halo_m_max);

  double res;
  if (1 == init) {
    res = int_GM02((lnMmin + lnMmax)/2.0, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_GM02;
    res = gsl_integration_glfixed(&F, lnMmin, lnMmax, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO MODEL POWER SPECTRA
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double p_xy_nointerp(
    const double k, 
    const double a,
    const int func,
    const int init
  ) 
{
  const double I02 = I02_XY_nointerp(k, k, a, func, init);

  double P1H, I11X, I11Y;

  switch(func)
  {
    case 0:
    { // PMM
      P1H  = I02;
      I11X = I11_X_nointerp(k, a, func, init);
      I11Y = I11X;
      break;
    }
    case 1:
    { // PMY
      if (!(cosmology.Omega_b > 0)) {
        log_fatal("Compton-y spectra need cosmology.Omega_b > 0 "
                  "(set_cosmological_parameters)");
        exit(1);
      }
      // sigma8 from the same sigma2(M) table the halo model uses:
      // sigma8 = sigma(R = 8 Mpc/h), the rms fluctuation in a top-hat
      // holding the mass M8 = (4 pi/3) rho_m R8^3 (lengths in c/H0
      // units, so R8 = 8/coverH0); sigma2's own cosmology cache key
      // keeps the value current
      const double R8 = 8.0/cosmology.coverH0;
      const double rhom = cosmology.rho_crit * cosmology.Omega_m;
      const double s8 = sqrt(sigma2(4.0*M_PI/3.0*rhom*R8*R8*R8));
      // convert to code unit, Table 2, 2009.01858
      const double ks = 0.05618/pow(s8*a,1.013)*cosmology.coverH0;
      // suppress low k (Eq17;2009.01858): P1H -> P1H (k/ks)^4/(1+(k/ks)^4),
      // which -> 0 for k << ks and -> 1 for k >> ks
      const double x = (k/ks)*(k/ks)*(k/ks)*(k/ks);
      P1H  = I02*(x/(x + 1.0));
      I11X = I11_X_nointerp(k, a, 0, init);
      I11Y = I11_X_nointerp(k, a, 1, init);
      break;
    }
    case 2:
    { // PYY
      if (!(cosmology.Omega_b > 0)) {
        log_fatal("Compton-y spectra need cosmology.Omega_b > 0 "
                  "(set_cosmological_parameters)");
        exit(1);
      }
      // sigma8 recomputed as in PMY above (one sigma2 table lookup)
      const double R8 = 8.0/cosmology.coverH0;
      const double rhom = cosmology.rho_crit * cosmology.Omega_m;
      const double s8 = sqrt(sigma2(4.0*M_PI/3.0*rhom*R8*R8*R8));
      // convert to code unit, Table 2, 2009.01858
      const double ks = 0.05618/pow(s8*a,1.013)*cosmology.coverH0;
      // suppress low k (Eq17;2009.01858): P1H -> P1H (k/ks)^4/(1+(k/ks)^4),
      // which -> 0 for k << ks and -> 1 for k >> ks
      const double x = (k/ks)*(k/ks)*(k/ks)*(k/ks);
      P1H  = I02*(x/(x + 1.0));
      I11X = I11_X_nointerp(k, a, 1, init);
      I11Y = I11X;
      break;
    }
    default:
    {
      log_fatal("option not supported");
      exit(1);
    }
  }

  return P1H + (I11X * I11Y * p_lin(k, a));;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

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
// the matter window (int_for_I02_XY header, items 1-2), b the Tinker bias,
// A(a) = 1 - bias_norm(a) the HMx share of matter below M_min put back as
// halos of mass M_min (I11_X_nointerp header, item 2). p_xy_nointerp
// evaluates the same P_mm directly (I02_XY_nointerp and I11_X_nointerp,
// GSL fixed rules: the route of p_my and p_yy); the rows here call it
// only as the warm-up below.
//
// 1. Quadrature: the n-point Gauss-Legendre rule in ln M (exact for
// polynomials of degree 2n - 1), n = 1024, the largest size GSL
// tabulates; nodes M_q and weights w_q on [ln M_min, ln M_max] are mapped
// once in the rebuild block (gsl_integration_glfixed_point).
//
// - Measured 2026-09-29 (a = 0.3, 0.6, 0.95; k = 0.05 to 1e6 (c/H0)^-1;
//   vs composite GL panels 0.05 wide in ln M), worst over k: I02 relative
//   error 6e-6 / 1e-4 / 8e-4, I11 2e-7 / 4e-6 / 2e-5 at 1024 / 512 / 256
//   nodes. High k converges slowly: the profile's ringing is sampled in ln M.
//
// 2. Loop levels: each factor is computed at the outermost level it
// depends on, so the innermost loop is the NFW kernel alone (nfw_um:
// three table reads and two sines; no pow, exp or log):
//
//   per refill, per node q (mq)  M_q, w_q; nu0_q = delta_c/sigma(M_q);
//                                w_q (rho_m/M_q) dlnnu/dlnM; M_q/rho_m;
//                                r_Delta(M_q)
//   per a row i, threaded        D(a); Tinker f, b parameters; A(a); c(M_min)
//   per (i, q) (aq[i])           nu = nu0/D; c = conc(M, D); ln(1+c);
//                                m(c) = ln(1+c) - c/(1+c); r_s = r_Delta/c,
//                                ln r_s; W2 = dn (M/rho_m)^2/m(c)^2;
//                                B1 = dn b(nu) (M/rho_m)/m(c), with
//                                dn = w (rho_m/M) dlnnu/dlnM f(nu) nu
//   per (i, k), sum over q       x = k r_s, ln x = ln k + ln r_s,
//                                um = u m(c) = nfw_um(c, x, ln x, ln(1+c));
//                                I02 = sum W2 um^2,
//                                I11 = sum B1 um + A u_c(k|M_min); ln P
//
// The 1/m(c) of u = um/m(c) lives in W2 and B1. The rows read the NFW
// kernel directly, so like.halo_model[3] must be HALO_PROFILE_NFW (the
// only option of u_c); anything else aborts.
//
// - Measured 2026-09-29 (4 threads; N_a = 256, N_k = 512, 1024 nodes:
//   1.3e8 kernel calls): one refill 0.5 s.
//
// Thread safety: the single-threaded p_xy_nointerp(k_min, a_min, 0, 1)
// call before the threaded loop builds every lazy table the rows read
// (sigma2, dlognudlogm, the tinker_alpha table of fnu_params_at,
// bias_norm, the NFW table nfw_), so inside the loop they are only read:
// the warm-up rule of the cosmo2D.c _work functions (halo_wrapper.hpp,
// "The init flag"). growfac and p_lin read the CAMB-fed cosmology tables
// and hold no static state.
//
// Cache invalidation:
//   rebuild block (table, mq, aq, GL nodes, both grids; every allocation
//     lives here, one block each from malloc2d/malloc3d): Ntable.random
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
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double lim[2][3];    // [0] a grid: min, max, step;
                              // [1] ln k grid: min, max, step
  static int nq = 0;          // Gauss-Legendre nodes in ln M
  static double** mq = NULL;  // [6][nq] per mass node: M, weight, nu at
                              // D = 1, weight x (rho_m/M) dlnnu/dlnM,
                              // M/rho_m, r_Delta
  static double*** aq = NULL; // [N_a][6][nq] per (a, mass node): c,
                              // ln(1+c), r_s, ln r_s, W2, B1

  // Ntable rebuild block: the table, the per-node and per-(a, node) arrays
  // (one block each from malloc2d/malloc3d, so one free each), the GL
  // nodes mapped onto [ln M_min, ln M_max], both grids (header, item 1)
  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) {
      free(table);
      free(mq);
      free(aq);
    }
    table = (double**) malloc2d(Ntable.N_a, Ntable.N_k_nlin);
    nq = 1024; // largest predefined GSL table
    mq = (double**) malloc2d(6, nq);
    aq = (double***) malloc3d(Ntable.N_a, 6, nq);
    const double lnMmin = log(limits.halo_m_min);
    const double lnMmax = log(limits.halo_m_max);
    // gsl_integration_glfixed_point(lo, hi, q, &x, &w, t): node q of the
    // rule t mapped onto [lo, hi], and its weight
    gsl_integration_glfixed_table* t = malloc_gslint_glfixed(nq);
    for (int q=0; q<nq; q++) {
      double lnM;
      gsl_integration_glfixed_point(lnMmin, lnMmax, q, &lnM, &mq[1][q], t);
      mq[0][q] = exp(lnM);
    }
    gsl_integration_glfixed_table_free(t);
    lim[0][0] = limits.a_min;
    lim[0][1] = 0.9999999;
    lim[0][2] = (lim[0][1] - lim[0][0]) / ((double) Ntable.N_a - 1.0);
    lim[1][0] = log(limits.k_min_cH0);
    lim[1][1] = log(limits.k_max_cH0);
    lim[1][2] = (lim[1][1] - lim[1][0]) / ((double) Ntable.N_k_nlin - 1.0);
  }
  // Refill: the cosmology or Ntable tag differs from the table's
  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random)) {
    // the rows read the NFW kernel directly (header, item 2)
    if (like.halo_model[3] != HALO_PROFILE_NFW) {
      log_fatal("like.halo_model[3] = %d not supported", like.halo_model[3]);
      exit(1);
    }
    // Warm-up: builds every lazy table the threaded loop reads (header,
    // Thread safety); the value is thrown away
    (void) p_xy_nointerp(exp(lim[1][0]), lim[0][0], 0, 1);
    // Per mass node (header, item 2, first row)
    const double rhom = cosmology.rho_crit * cosmology.Omega_m;
    const double rho_delta = Delta * rhom;
    for (int q=0; q<nq; q++) {
      const double m = mq[0][q];
      mq[2][q] = delta_c/sqrt(sigma2(m));
      mq[3][q] = mq[1][q]*(rhom/m)*dlognudlogm(m);
      mq[4][q] = m/rhom;
      mq[5][q] = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
    }
    // Per a row, threaded: D(a), the Tinker f and b parameters (the
    // nu-independent halves, *_params_at), A(a), c(M_min)
    const double mmin = limits.halo_m_min;
    #pragma omp parallel for schedule(static)
    for (int i=0; i<Ntable.N_a; i++) {
      const double ai = lim[0][0] + i*lim[0][2];
      const double D = growfac(ai);
      const fnu_params pf = fnu_params_at(ai);
      const hb1nu_params pb = hb1nu_params_at(ai);
      const double A = 1.0 - bias_norm(ai);
      const double cmin = conc(mmin, D);
      double* restrict cq = aq[i][0];
      double* restrict l1 = aq[i][1];
      double* restrict rs = aq[i][2];
      double* restrict lrs = aq[i][3];
      double* restrict w2 = aq[i][4];
      double* restrict b1 = aq[i][5];
      // Per (a, node): concentration, r_s, their logs, and the weights
      // W2, B1 with 1/m(c) folded in (header, item 2, third row)
      for (int q=0; q<nq; q++) {
        const double nu = mq[2][q]/D;
        const double c = conc(mq[0][q], D);
        const double l1c = log1p(c);
        const double mc = l1c - c/(1.0 + c);
        const double dn = mq[3][q]*fnu_core(nu, &pf)*nu;
        cq[q] = c;
        l1[q] = l1c;
        rs[q] = mq[5][q]/c;
        lrs[q] = log(rs[q]);
        w2[q] = dn*(mq[4][q]/mc)*(mq[4][q]/mc);
        b1[q] = dn*hb1nu_core(nu, &pb)*(mq[4][q]/mc);
      }
      // Per k: I02 and I11 as sums of the NFW kernel over the nodes, the
      // HMx term A u_c(k|M_min), then ln P (header, item 2, last row)
      for (int j=0; j<Ntable.N_k_nlin; j++) {
        const double lk = lim[1][0] + j*lim[1][2];
        const double kj = exp(lk);
        double s02 = 0.0;
        double s11 = 0.0;
        for (int q=0; q<nq; q++) {
          const double um = nfw_um(cq[q], kj*rs[q], lk + lrs[q], l1[q]);
          s02 += w2[q]*um*um;
          s11 += b1[q]*um;
        }
        const double I11 = s11 + A*u_c(cmin, kj, mmin, ai);
        table[i][j] = log(s02 + I11*I11*p_lin(kj, ai));
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
  }
  // bilinear read of ln P; 0 outside the a range
  return ((a < lim[0][0]) || (a > lim[0][1])) ? 0.0 :
    exp(interpol2d(table,
                   Ntable.N_a, lim[0][0], lim[0][1], lim[0][2], a,
                   Ntable.N_k_nlin, lim[1][0], lim[1][1], lim[1][2], log(k)));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double p_my(
    const double k, 
    const double a
  )
{ 
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = 0;
  static double lim[2][3]; // lim[0][0] = amin, lim[0][1] = amax, lim[0][2] = da 
                           // lim[1][0] = lnkmin, lim[1][1] = lnkmax, lim[1][2] = dlnk

  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double**) malloc2d(Ntable.N_a, Ntable.N_k_nlin); 
    lim[0][0] = limits.a_min;
    lim[0][1] = 0.9999999;
    lim[0][2] = (lim[0][1] - lim[0][0]) / ((double) Ntable.N_a - 1.0);
    lim[1][0] = log(limits.k_min_cH0);
    lim[1][1] = log(limits.k_max_cH0);
    lim[1][2] = (lim[1][1] - lim[1][0]) / ((double) Ntable.N_k_nlin - 1.0);
  }
  if (fdiff2(cache[0], cosmology.random) || 
      fdiff2(cache[1], Ntable.random) ||
      fdiff2(cache[2], nuisance.random_gas))
  {
    (void) p_xy_nointerp(exp(lim[1][0]), lim[0][0], 1, 1); // init static vars
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.N_a; i++) {
      for (int j=0; j<Ntable.N_k_nlin; j++) {
        table[i][j] = log(p_xy_nointerp(exp(lim[1][0] + j*lim[1][2]), 
                                            lim[0][0] + i*lim[0][2], 1, 0));
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_gas;
  }
  return ((a < lim[0][0]) || (a > lim[0][1])) ? 0.0 :
    exp(interpol2d(table,
                   Ntable.N_a, lim[0][0], lim[0][1], lim[0][2], a,
                   Ntable.N_k_nlin, lim[1][0], lim[1][1], lim[1][2], log(k)));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double p_yy(
    const double k, 
    const double a
  )
{ 
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = 0;
  static double lim[2][3]; // lim[0][0]=amin, lim[0][1]=amax, lim[0][2]=da 
                           // lim[1][0]=lnkmin, lim[1][1]=lnkmax, lim[1][2]=dlnk

  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double**) malloc2d(Ntable.N_a, Ntable.N_k_nlin);
    lim[0][0] = limits.a_min;
    lim[0][1] = 0.9999999;
    lim[0][2] = (lim[0][1] - lim[0][0]) / ((double) Ntable.N_a - 1.0);
    lim[1][0] = log(limits.k_min_cH0);
    lim[1][1] = log(limits.k_max_cH0);
    lim[1][2] = (lim[1][1] - lim[1][0]) / ((double) Ntable.N_k_nlin - 1.0);
  }
  if (fdiff2(cache[0], cosmology.random) || 
      fdiff2(cache[1], Ntable.random) ||
      fdiff2(cache[2], nuisance.random_gas))
  { 
    (void) p_xy_nointerp(exp(lim[1][0]), lim[0][0], 2, 1); // init static vars
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.N_a; i++) {
      for (int j=0; j<Ntable.N_k_nlin; j++) {
        table[i][j] = log(p_xy_nointerp(exp(lim[1][0] + j*lim[1][2]), 
                                            lim[0][0] + i*lim[0][2], 2, 0));
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_gas;
  }
  return ((a < lim[0][0]) || (a > lim[0][1])) ? 0.0 :
    exp(interpol2d(table,
                   Ntable.N_a, lim[0][0], lim[0][1], lim[0][2], a,
                   Ntable.N_k_nlin, lim[1][0], lim[1][1], lim[1][2], log(k)));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double p_gm_nointerp(
    const double k, 
    const double a, 
    const int ni,
    const int init
  )
{
  return Pdelta(k, a)*bgal(ni, a) + GM02_nointerp(k, a, ni, init)/ngal(ni, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

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
// N_c, N_s, f_c the occupation of the GALAXY PROFILES banner. The same
// P_gm done directly: p_gm_nointerp (GM02_nointerp, GSL fixed rule),
// called by the rows here only as the warm-up below.
//
// 1. Quadrature: the 1024-node Gauss-Legendre rule of p_mm (its header,
// item 1) over ln M from ln 10^(lg M_min - 1) of the bin to
// ln limits.halo_m_max: nodes x_q, weights w_q on [-1, 1] (gl) are mapped
// per bin in the refill, ln M_q = mid + hw x_q, weight hw w_q.
//
// 2. Loop levels as in p_mm (its header, item 2): the innermost loop is
// the NFW kernel nfw_um alone. The occupation does not depend on a
// (HOD_nc, HOD_ns only range-check it: amin_lens is a placeholder):
//
//   per refill, per (bin, node) (bq)  M, hw w (rho_m/M) dlnnu/dlnM, nu0,
//                                     r_Delta, N_s, f_c N_c
//   per bin, per a row, threaded      D(a); Tinker f parameters; ngal, bgal
//   per (a, node) (aq)                c = conc(M, D) and c_g = gc c, each
//                                     with ln(1+c), m(c), r_s, ln r_s;
//                                     vm = dn (M/rho_m)/m(c);
//                                     W1 = vm N_s/m(c_g), W0 = vm f_c N_c
//   per (a, k), sum over nodes        um = u_m m(c), ug = u_g m(c_g);
//                                     GM02 = sum um (W1 ug + W0); ln P
//
// vm carries the 1/m(c) of the matter leg, W1 the 1/m(c_g) of the
// satellite leg (the central has no profile). gc = 1 makes c_g = c
// exactly, so ug = um bitwise and one kernel call serves both legs (the
// same branch): half the kernel calls. like.halo_model[3] must be
// HALO_PROFILE_NFW (the rows read nfw_um directly) and nuisance.gc[l] > 0
// in every bin (u_g's condition, checked in the refill); else abort.
//
// - Measured 2026-09-29 (4 threads; 10 lens bins, na = 51, N_k = 512,
//   1024 nodes): one refill 1.0 s.
//
// Thread safety: the single-threaded p_gm_nointerp(k_min, a_min, 0, 1)
// call before the threaded loops (p_mm's warm-up rule) builds every lazy
// table the rows read (sigma2, dlognudlogm, tinker_alpha of fnu_params_at,
// the NFW table nfw_, the hod_ tables of ngal, bgal) and latches Pdelta's
// run mode, a static set on its first call; growfac, p_lin, p_nonlin and
// PkRatio_baryons hold no static state. One parallel region per bin.
//
// Cache invalidation:
//   rebuild block (table, lim, gl, bq, aq; every allocation lives here,
//     one block each from malloc2d/malloc3d): Ntable.random or
//     redshift.random_clustering (bin count and a ranges: clustering n(z))
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
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double*** table = NULL;
  static double** lim = NULL; // [nbin+1][3]: row l < nbin the a grid of
                              // bin l (min, max, step); row nbin the
                              // ln k grid (min, max, step)
  static int nbin = 0;        // lens bins of the allocation
  static int na = 0;          // a nodes per bin
  static int nq = 0;          // Gauss-Legendre nodes in ln M
  static double** gl = NULL;  // [2][nq] GL nodes (0), weights (1) on [-1, 1]
  static double*** bq = NULL; // [nbin][6][nq] per (bin, mass node): M,
                              // hw w (rho_m/M) dlnnu/dlnM, nu at D = 1,
                              // r_Delta, N_s, f_c N_c
  static double*** aq = NULL; // [na][10][nq] per (a, mass node) of one bin:
                              // c, ln(1+c), r_s, ln r_s and the same for
                              // c_g = gc c, then the weights W1, W0

  // Rebuild block: the first call, or the Ntable or clustering-n(z) tag
  // differs from the allocation's: sizes, every allocation (one block
  // each from malloc2d/malloc3d, so one free each), the GL rule on
  // [-1, 1], the a grid of each bin and the ln k grid (header, item 1)
  if (NULL == table ||
      fdiff2(cache[1], Ntable.random) ||
      fdiff2(cache[3], redshift.random_clustering))
  {
    if (table != NULL) {
      free(table);
      free(lim);
      free(gl);
      free(bq);
      free(aq);
    }
    nbin = redshift.clustering_nbin;
    na = (int) Ntable.N_a/5.0; // a bin's a range is a slice of p_mm's
    nq = 1024; // largest predefined GSL table
    table = (double***) malloc3d(nbin, na, Ntable.N_k_nlin);
    lim = (double**) malloc2d(nbin+1, 3);
    gl = (double**) malloc2d(2, nq);
    bq = (double***) malloc3d(nbin, 6, nq);
    aq = (double***) malloc3d(na, 10, nq);
    // gsl_integration_glfixed_point(lo, hi, q, &x, &w, t): node q of the
    // rule t mapped onto [lo, hi], and its weight; kept on [-1, 1] here
    gsl_integration_glfixed_table* t = malloc_gslint_glfixed(nq);
    for (int q=0; q<nq; q++) {
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
    lim[nbin][2] = (lim[nbin][1]-lim[nbin][0])/((double) Ntable.N_k_nlin - 1.0);
  }

  // Refill: any of the four tags differs from the table's
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], Ntable.random)    ||
      fdiff2(cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(cache[3], redshift.random_clustering))
  {
    // the rows read the NFW kernel directly (header, item 2)
    if (like.halo_model[3] != HALO_PROFILE_NFW) {
      log_fatal("like.halo_model[3] = %d not supported", like.halo_model[3]);
      exit(1);
    }
    // Warm-up: builds every lazy table the threaded loops read (header,
    // Thread safety); the value is thrown away
    (void) p_gm_nointerp(exp(lim[nbin][0]), lim[0][0], 0, 1);
    // Per (bin, node), serially (header, item 2, first row): the GL nodes
    // mapped onto [ln 10^(lg M_min - 1), ln M_max] of the bin, the
    // a-independent factors, the occupation at the placeholder a = amin;
    // the sigma2 and dlognudlogm reads happen here, before the threads
    const double rhom = cosmology.rho_crit * cosmology.Omega_m;
    const double rho_delta = Delta * rhom;
    for (int l=0; l<nbin; l++) {
      // u_g's condition, checked for every bin at once
      if (!(nuisance.gc[l] > 0)) {
        log_fatal("galaxy concentration factor gc[%d] = %g must be > 0",
                  l, nuisance.gc[l]);
        exit(1);
      }
      const double lnMmin = log(10.)*(nuisance.hod[l][0] - 1.0);
      const double lnMmax = log(limits.halo_m_max);
      const double hw = 0.5*(lnMmax - lnMmin);
      const double mid = 0.5*(lnMmax + lnMmin);
      const double fc = HOD_fc(l);
      for (int q=0; q<nq; q++) {
        const double m = exp(mid + hw*gl[0][q]);
        bq[l][0][q] = m;
        bq[l][1][q] = hw*gl[1][q]*(rhom/m)*dlognudlogm(m);
        bq[l][2][q] = delta_c/sqrt(sigma2(m));
        bq[l][3][q] = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
        bq[l][4][q] = HOD_ns(m, lim[l][0], l);
        bq[l][5][q] = fc*HOD_nc(m, lim[l][0], l);
      }
    }
    // Per bin, its rows over a threaded: D(a), the Tinker f parameters
    // (the nu-independent half, fnu_params_at), ngal and bgal of the row
    // (header, item 2, second row)
    for (int l=0; l<nbin; l++) {
      const double gc = nuisance.gc[l];
      const int same = (1.0 == gc); // c_g = c: one kernel serves both legs
      #pragma omp parallel for schedule(static)
      for (int i=0; i<na; i++) {
        const double ai = lim[l][0] + i*lim[l][2];
        const double D = growfac(ai);
        const fnu_params pf = fnu_params_at(ai);
        const double ng = ngal(l, ai);
        const double bg = bgal(l, ai);
        double* restrict cq = aq[i][0];
        double* restrict l1 = aq[i][1];
        double* restrict rs = aq[i][2];
        double* restrict lrs = aq[i][3];
        double* restrict cgq = aq[i][4];
        double* restrict l1g = aq[i][5];
        double* restrict rsg = aq[i][6];
        double* restrict lrsg = aq[i][7];
        double* restrict w1 = aq[i][8];
        double* restrict w0 = aq[i][9];
        // Per (a, node): both concentrations, r_s and the logs of each,
        // and the weights W1, W0 with 1/m(c), 1/m(c_g) folded in (header,
        // item 2, third row). restrict: each row is reached only through
        // its pointer, so no reload after the libm calls
        for (int q=0; q<nq; q++) {
          const double m = bq[l][0][q];
          const double nu = bq[l][2][q]/D;
          const double c = conc(m, D);
          const double cg = c*gc;
          const double l1c = log1p(c);
          const double l1cg = log1p(cg);
          const double mc = l1c - c/(1.0 + c);
          const double mcg = l1cg - cg/(1.0 + cg);
          const double dn = bq[l][1][q]*fnu_core(nu, &pf)*nu;
          const double vm = dn*(m/rhom)/mc;
          cq[q] = c;
          l1[q] = l1c;
          rs[q] = bq[l][3][q]/c;
          lrs[q] = log(rs[q]);
          cgq[q] = cg;
          l1g[q] = l1cg;
          rsg[q] = bq[l][3][q]/cg;
          lrsg[q] = log(rsg[q]);
          w1[q] = vm*bq[l][4][q]/mcg;
          w0[q] = vm*bq[l][5][q];
        }
        // Per k: GM02 as a sum of the NFW kernel over the nodes, one call
        // per leg or one for both (same), then ln P (header, item 2,
        // last row)
        for (int j=0; j<Ntable.N_k_nlin; j++) {
          const double lk = lim[nbin][0] + j*lim[nbin][2];
          const double kj = exp(lk);
          double sum = 0.0;
          if (same) {
            for (int q=0; q<nq; q++) {
              const double um = nfw_um(cq[q], kj*rs[q], lk + lrs[q], l1[q]);
              sum += um*(w1[q]*um + w0[q]);
            }
          }
          else {
            for (int q=0; q<nq; q++) {
              const double um = nfw_um(cq[q], kj*rs[q], lk + lrs[q], l1[q]);
              const double ug = nfw_um(cgq[q], kj*rsg[q], lk + lrsg[q], l1g[q]);
              sum += um*(w1[q]*ug + w0[q]);
            }
          }
          table[l][i][j] = log(Pdelta(kj, ai)*bg + sum/ng);
        }
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_galaxy_bias;
    cache[3] = redshift.random_clustering;
  }
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  // bilinear read of bin ni's ln P; 0 outside its a range
  return (a < lim[ni][0] || a > lim[ni][1]) ? 0.0 : exp(interpol2d(table[ni],
    na, lim[ni][0], lim[ni][1], lim[ni][2], a,
    Ntable.N_k_nlin, lim[nbin][0], lim[nbin][1], lim[nbin][2], log(k)));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double p_gg_nointerp(
    const double k, 
    const double a, 
    const int ni, 
    const int nj,
    const int init
  )
{
  if (ni != nj) {
    log_fatal("cross-tomography (ni,nj) = (%d,%d) bins not supported", ni, nj);
    exit(1);
  }
  const double bg = bgal(ni, a);
  const double ng = ngal(ni, a);
  return Pdelta(k, a)*bg*bg + G02_nointerp(k, a, ni, init)/(ng*ng);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

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
// banner. p_gg_nointerp is the same P_gg done directly (G02_nointerp, GSL
// fixed rule); the rows here call it only as the warm-up below.
//
// 1. Quadrature: the 1024-node Gauss-Legendre rule of p_mm (its header,
// item 1) over [ln limits.halo_m_min, ln limits.halo_m_max], the same for
// every bin: nodes and weights are mapped once, in the rebuild block.
//
// 2. Loop levels as in p_mm (its header, item 2): the innermost loop is
// the NFW kernel nfw_um alone. The occupation does not depend on a
// (HOD_nc, HOD_ns only range-check it: amin_lens is a placeholder):
//
//   per refill, per node (mq)         M, w, nu0, r_Delta,
//                                     w (rho_m/M) dlnnu/dlnM
//   per refill, per (bin, node) (hq)  N_s, f_c N_c
//   per bin, per a row, threaded      D(a); Tinker f parameters; ngal, bgal
//   per (a, node) (aq)                c_g = gc conc(M, D), ln(1+c_g),
//                                     m(c_g), r_s,g = r_Delta/c_g, ln r_s,g;
//                                     W2 = dn (N_s/m(c_g))^2,
//                                     W1 = 2 dn (N_s/m(c_g)) f_c N_c
//   per (a, k), sum over nodes        ug = u_g m(c_g) (nfw_um);
//                                     G02 = sum ug (W2 ug + W1); ln P
//
// W2 and W1 carry the 1/m(c_g) of each u_g (the central has no profile).
// like.halo_model[3] must be HALO_PROFILE_NFW (the rows read nfw_um
// directly) and nuisance.gc[l] > 0 in every bin (u_g's condition,
// checked in the refill); else abort.
//
// - Measured 2026-09-29 (4 threads; 10 lens bins, na = 51, N_k = 512,
//   1024 nodes: 2.7e8 kernel calls): one refill 1.0 s.
//
// Thread safety: as p_gm (its header), with the single-threaded
// p_gg_nointerp(k_min, a_min, 0, 0, 1) call before the threaded loops as
// the warm-up that builds every lazy table the rows read.
//
// Cache invalidation:
//   as p_gm (its header); the rebuild block here holds table, lim, mq
//     (with the mapped GL nodes), hq and aq
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
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double*** table = NULL;
  static double** lim = NULL; // [nbin+1][3]: row l < nbin the a grid of
                              // bin l (min, max, step); row nbin the
                              // ln k grid (min, max, step)
  static int nbin = 0;        // lens bins of the allocation
  static int na = 0;          // a nodes per bin
  static int nq = 0;          // Gauss-Legendre nodes in ln M
  static double** mq = NULL;  // [5][nq] per mass node: M, weight, nu at
                              // D = 1, r_Delta, weight x (rho_m/M) dlnnu/dlnM
  static double*** hq = NULL; // [nbin][2][nq] per (bin, mass node): N_s,
                              // f_c N_c
  static double*** aq = NULL; // [na][6][nq] per (a, mass node) of one bin:
                              // c_g, ln(1+c_g), r_s,g, ln r_s,g, W2, W1

  // Rebuild block: the first call, or the Ntable or clustering-n(z) tag
  // differs from the allocation's: sizes, every allocation (one block
  // each from malloc2d/malloc3d, so one free each), the GL nodes mapped
  // onto [ln M_min, ln M_max], the a grid of each bin and the ln k grid
  // (header, item 1)
  if (NULL == table ||
      fdiff2(cache[1], Ntable.random) ||
      fdiff2(cache[3], redshift.random_clustering))
  {
    if (table != NULL) {
      free(table);
      free(lim);
      free(mq);
      free(hq);
      free(aq);
    }
    nbin = redshift.clustering_nbin;
    na = (int) Ntable.N_a/5.0; // a bin's a range is a slice of p_mm's
    nq = 1024; // largest predefined GSL table
    table = (double***) malloc3d(nbin, na, Ntable.N_k_nlin);
    lim = (double**) malloc2d(nbin+1, 3);
    mq = (double**) malloc2d(5, nq);
    hq = (double***) malloc3d(nbin, 2, nq);
    aq = (double***) malloc3d(na, 6, nq);
    const double lnMmin = log(limits.halo_m_min);
    const double lnMmax = log(limits.halo_m_max);
    // gsl_integration_glfixed_point(lo, hi, q, &x, &w, t): node q of the
    // rule t mapped onto [lo, hi], and its weight
    gsl_integration_glfixed_table* t = malloc_gslint_glfixed(nq);
    for (int q=0; q<nq; q++) {
      double lnM;
      gsl_integration_glfixed_point(lnMmin, lnMmax, q, &lnM, &mq[1][q], t);
      mq[0][q] = exp(lnM);
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
    lim[nbin][2] = (lim[nbin][1]-lim[nbin][0])/((double) Ntable.N_k_nlin - 1.);
  }
  // Refill: any of the four tags differs from the table's
  if (fdiff2(cache[0], cosmology.random) ||
      fdiff2(cache[1], Ntable.random)    ||
      fdiff2(cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(cache[3], redshift.random_clustering))
  {
    // the rows read the NFW kernel directly (header, item 2)
    if (like.halo_model[3] != HALO_PROFILE_NFW) {
      log_fatal("like.halo_model[3] = %d not supported", like.halo_model[3]);
      exit(1);
    }
    // Warm-up: builds every lazy table the threaded loops read (header,
    // Thread safety); the value is thrown away
    (void) p_gg_nointerp(exp(lim[nbin][0]), lim[0][0], 0, 0, 1);
    // Per mass node, serially (header, item 2, first row); the sigma2 and
    // dlognudlogm reads happen here, before the threads
    const double rhom = cosmology.rho_crit * cosmology.Omega_m;
    const double rho_delta = Delta * rhom;
    for (int q=0; q<nq; q++) {
      const double m = mq[0][q];
      mq[2][q] = delta_c/sqrt(sigma2(m));
      mq[3][q] = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
      mq[4][q] = mq[1][q]*(rhom/m)*dlognudlogm(m);
    }
    // Per (bin, node), serially: the occupation at the placeholder
    // a = amin (header, item 2, second row)
    for (int l=0; l<nbin; l++) {
      // u_g's condition, checked for every bin at once
      if (!(nuisance.gc[l] > 0)) {
        log_fatal("galaxy concentration factor gc[%d] = %g must be > 0",
                  l, nuisance.gc[l]);
        exit(1);
      }
      const double fc = HOD_fc(l);
      for (int q=0; q<nq; q++) {
        const double m = mq[0][q];
        hq[l][0][q] = HOD_ns(m, lim[l][0], l);
        hq[l][1][q] = fc*HOD_nc(m, lim[l][0], l);
      }
    }
    // Per bin, its rows over a threaded: D(a), the Tinker f parameters
    // (the nu-independent half, fnu_params_at), ngal and bgal of the row
    // (header, item 2, third row)
    for (int l=0; l<nbin; l++) {
      const double gc = nuisance.gc[l];
      #pragma omp parallel for schedule(static)
      for (int i=0; i<na; i++) {
        const double ai = lim[l][0] + i*lim[l][2];
        const double D = growfac(ai);
        const fnu_params pf = fnu_params_at(ai);
        const double ng = ngal(l, ai);
        const double bg = bgal(l, ai);
        double* restrict cgq = aq[i][0];
        double* restrict l1g = aq[i][1];
        double* restrict rsg = aq[i][2];
        double* restrict lrsg = aq[i][3];
        double* restrict w2 = aq[i][4];
        double* restrict w1 = aq[i][5];
        // Per (a, node): c_g, r_s,g, their logs, and the weights W2, W1
        // with 1/m(c_g) folded in (header, item 2, fourth row). restrict:
        // each row is reached only through its pointer, so no reload
        // after the libm calls
        for (int q=0; q<nq; q++) {
          const double m = mq[0][q];
          const double nu = mq[2][q]/D;
          const double cg = conc(m, D)*gc;
          const double l1cg = log1p(cg);
          const double mcg = l1cg - cg/(1.0 + cg);
          const double dn = mq[4][q]*fnu_core(nu, &pf)*nu;
          const double ns = hq[l][0][q];
          cgq[q] = cg;
          l1g[q] = l1cg;
          rsg[q] = mq[3][q]/cg;
          lrsg[q] = log(rsg[q]);
          w2[q] = dn*(ns/mcg)*(ns/mcg);
          w1[q] = 2.0*dn*(ns/mcg)*hq[l][1][q];
        }
        // Per k: G02 as a sum of the NFW kernel over the nodes, then ln P
        // (header, item 2, last row)
        for (int j=0; j<Ntable.N_k_nlin; j++) {
          const double lk = lim[nbin][0] + j*lim[nbin][2];
          const double kj = exp(lk);
          double sum = 0.0;
          for (int q=0; q<nq; q++) {
            const double ug = nfw_um(cgq[q], kj*rsg[q], lk + lrsg[q], l1g[q]);
            sum += ug*(w2[q]*ug + w1[q]);
          }
          table[l][i][j] = log(Pdelta(kj, ai)*bg*bg + sum/(ng*ng));
        }
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_galaxy_bias;
    cache[3] = redshift.random_clustering;
  }
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("error in selecting bin number ni = %d", ni); exit(1);
  }
  // auto-spectra only (header)
  if (ni != nj) {
    log_fatal("cross-tomography (ni,nj) = (%d,%d) bins not supported", ni, nj);
    exit(1);
  }
  // bilinear read of bin ni's ln P; 0 outside its a range
  return (a < lim[ni][0] || a > lim[ni][1]) ? 0.0 : exp(
    interpol2d(table[ni],
               na, lim[ni][0], lim[ni][1], lim[ni][2], a,
               Ntable.N_k_nlin, lim[nbin][0], lim[nbin][1], lim[nbin][2], log(k)));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// MISCELLANEOUS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

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
      nuisance.gb[0][ni] = hm_funcs_nointerp(ni, a, 3, 0);
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
      nuisance.gb[0][ni] = hm_funcs_nointerp(ni, a, 3, 0);
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
      nuisance.gb[0][ni] = hm_funcs_nointerp(ni, a, 3, 0);
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
      nuisance.gb[0][ni] = hm_funcs_nointerp(ni, a, 3, 0);
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
      nuisance.gb[0][ni] = hm_funcs_nointerp(ni, a, 3, 0);
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

  log_debug("HOD: bin %d; <z> %.2f; <n_g> %e(h/Mpc)^3", ni, z, 
    ngal_nointerp(ni, a, 0)*pow(cosmology.coverH0, -3.0));
  
  log_debug("HOD: bin %d; <z> %.2f; <M> h/Msun %.4e", ni, z, mmean_nointerp(ni,a,0));
  
  log_debug("HOD: bin %d; <z> %.2f; f_sat %.3f", ni, z, fsat_nointerp(ni,a,0));
  
  log_debug("HOD: bin %d; <z> %.2f; <b_g> %.2f", ni, z, nuisance.gb[0][ni]);
}
