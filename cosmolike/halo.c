#include <assert.h>
#include <gsl/gsl_sf.h>
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
// the fly, with weights good to only ~5e-7). The gas integrals (F0_KS,
// F_KS) ladder 256/512/1024 with Ntable.high_def_integration and
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
//   F0_KS        = gas-mass integral of the Komatsu-Seljak ("KS")
//                  bound-gas profile ("0": the k = 0, mass integral)
//   F_KS         = Fourier integral of the KS electron pressure
//   u_KS         = F_KS/F0_KS, the bound-gas pressure shape factor
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
// Normalized Fourier transform u(k|M) of the NFW profile truncated at
// r_Delta, in closed form (astro-ph/0206508 Eq. 81).
//
// The NFW profile (astro-ph/9611107), with scale radius r_s = r_Delta/c:
//
//   rho(r) = rho_s / [(r/r_s) (1 + r/r_s)^2]
//
// Its mass inside r_Delta is M = 4 pi rho_s r_s^3 m(c), with
// m(c) = ln(1+c) - c/(1+c) (astro-ph/0206508 Eq. 76), which sets the
// prefactor of the transform to 1/m(c). With
//
//   r_Delta = (3M/(4 pi Delta rho_m))^(1/3)   (comoving, c/H0)
//   x       = k r_Delta/c = k r_s
//   xu      = (1 + c) x
//
// Eq. 81 reads
//
//   u = { sin x [Si(xu) - Si(x)] - sin(c x)/xu
//         + cos x [Ci(xu) - Ci(x)] } / m(c)
//
// with Si, Ci the sine and cosine integrals (GSL). Check at k -> 0: the
// three terms tend to 0, -c/(1+c) and ln(1+c), so u -> 1.
//
// r_Delta is comoving (rho_m = rho_crit Omega_m is the comoving mean
// density), and so is k: the scale factor never enters.
//
// Parameters:
//   c - concentration r_Delta/r_s, c > 0 (m(0) = 0)
//   k - wavenumber in (c/H0)^-1, k > 0 (Ci diverges at 0; the GSL
//       domain error aborts)
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
  const double rho_delta = Delta * cosmology.rho_crit * cosmology.Omega_m;
  const double r_delta = pow(3./(4.0*M_PI)*(m/rho_delta), 1./3.);
  const double x = k * r_delta / c;
  const double xu = (1. + c) * x;

  gsl_sf_result SI_XU;
  int status = gsl_sf_Si_e(xu, &SI_XU);
  if (status) {
    log_fatal(gsl_strerror(status)); exit(1);
  }

  gsl_sf_result SI_X;
  {
    int status = gsl_sf_Si_e(x, &SI_X);
    if (status) {
      log_fatal(gsl_strerror(status)); exit(1);
    }
  }

  gsl_sf_result CI_XU;
  {
    int status = gsl_sf_Ci_e(xu, &CI_XU);
    if (status) {
      log_fatal(gsl_strerror(status)); exit(1);
    }
  }

  gsl_sf_result CI_X;
  {
    int status = gsl_sf_Ci_e(x, &CI_X);
    if (status) {
      log_fatal(gsl_strerror(status)); exit(1);
    }
  }
  return (sin(x)*(SI_XU.val - SI_X.val) 
          - sinl(c*x)/xu 
          + cos(x)*(CI_XU.val - CI_X.val))/(log(1. + c) - c/(1. + c));
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
// normalization m(0) vanish.
//
// Parameters:
//   c  - halo concentration c(M)
//   k  - wavenumber in (c/H0)^-1
//   m  - halo mass in M_sun/h
//   a  - scale factor (unused by the NFW form)
//   ni - lens bin (indexes nuisance.gc)
//
// Returns:
//   u_g(k|M), dimensionless; 1 at k -> 0
// ---------------------------------------------------------------------------
double u_g(
    const double c, // halo concentration c(M)
    const double k, // wavenumber in (c/H0)^-1
    const double m, // halo mass in M_sun/h
    const double a, // scale factor (unused by the NFW form)
    const int ni    // lens bin: selects the factor nuisance.gc[ni]
  )
{
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
// Floor: for M <= M_0 the base (M - M_0)/M_1 is not positive: at
// M = M_0 the power is 0, below it pow returns NaN for a non-integer
// alpha. Either way the test ns > 0 fails and the function returns
// 1e-15, which keeps N_s strictly positive; the floor's contribution to
// every integral is negligible.
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
  const double x = (m - pow(10.,nuisance.hod[ni][3]))/pow(10., nuisance.hod[ni][2]);
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
// Integrand of the bound-gas mass normalization F0 (F0_KS_nointerp):
//
//   x^2 theta(x)^(1/(Gamma - 1)),   theta(x) = ln(1 + x)/x,  x = r/r_s
//
// theta^(1/(Gamma-1)) is the Komatsu-Seljak gas density profile in the
// form of 2005.00009 Eq. 35 (also 1510.06034 Eq. 2.10). It follows
// from the general solution for a polytrope, P ~ rho^Gamma, in
// hydrostatic equilibrium in an NFW potential (astro-ph/0106151
// Eq. 19, with the NFW integral of its Eq. 9):
//
//   y_gas^(Gamma-1) = 1 - B int_0^x m(u)/u^2 du = 1 - B (1 - theta)
//
// with B a constant set by the central temperature. The condition that
// the gas temperature, T ~ y_gas^(Gamma-1), vanishes as r -> infinity
// (theta -> 0) forces B = 1: y_gas = theta^(1/(Gamma-1)) and
// T_g = T_v theta.
//
// Parameters:
//   x      - radius in units of r_s
//   params - unused (GSL signature)
//
// Returns:
//   the integrand, dimensionless; Gamma = nuisance.gas[0] > 1
// ---------------------------------------------------------------------------
double int_F0_KS(
    double x,                             // radius r/r_s
    void* params __attribute__((unused))  // unused (GSL signature)
  )
{
  return x*x*pow(log(1.0 + x)/x, 1.0/(nuisance.gas[0] - 1.0));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Bound-gas mass normalization of the Komatsu-Seljak profile:
//
//   F0(c) = int_0^c x^2 theta(x)^(1/(Gamma - 1)) dx
//
// the dimensionless gas mass inside r_Delta = c r_s: the bound gas holds
// M_bnd = 4 pi rho_bnd(0) r_s^3 F0(c) = f_bnd M (2005.00009 Eq. 13), so
// dividing by F0 turns the pressure transform into a window per unit
// bound-gas mass.
//
// Numerics:
//   Gauss-Legendre on [0, c] with 256/512/1024 nodes at
//   Ntable.high_def_integration = 0/1/>=2 (predefined GSL tables); init = 1 returns the integrand at
//   the midpoint instead of the integral (its only role is to build the
//   static GL table before a parallel region).
//
// Cache invalidation:
// the static GL table rebuilds when Ntable.random changes.
//
// Parameters:
//   c    - concentration (the upper limit, in units of r_s)
//   init - 1 = build the static table only, 0 = integrate
//
// Returns:
//   F0(c), dimensionless
// ---------------------------------------------------------------------------
double F0_KS_nointerp(
    double c,       // concentration: upper limit, in units of r_s
    const int init  // 1 = build the static GL table only, 0 = integrate
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 256 :
                         (1 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[1] = {0.0};
  const double xmin = 0.0;
  const double xmax = c;
  
  double res = 0.0;
  if (1 == init) {
    res = int_F0_KS((xmin + xmax)/2.0, (void*) ar);
  }
  else
  {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_F0_KS;
    res = gsl_integration_glfixed(&F, xmin, xmax, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Integrand of the bound-gas pressure transform F (F_KS_nointerp):
//
//   x^2 [sin(y x)/(y x)] theta(x)^(Gamma/(Gamma - 1))
//     = x sin(y x)/y theta(x)^(Gamma/(Gamma - 1)),   y = k r_s
//
// The electron pressure is density times temperature (2005.00009
// Eqs. 38, 40): P_e ~ rho_bnd T_g ~ theta^(1/(Gamma-1)) theta =
// theta^(Gamma/(Gamma-1)). The kernel sin(y x)/(y x) is the spherical
// Fourier transform of 2005.00009 Eq. 4.
//
// Parameters:
//   x      - radius in units of r_s
//   params - params[0] = y = k r_s
//
// Returns:
//   the integrand, dimensionless
// ---------------------------------------------------------------------------
double int_F_KS(
    double x,     // radius r/r_s
    void* params  // params[0] = y = k r_s
  )
{
  double* ar = (double*) params;
  const double y = ar[0];  
  return (x*sinl(y*x)/y)*
         pow(log(1.0 + x)/x, nuisance.gas[0]/(nuisance.gas[0] - 1.0));
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fourier transform of the Komatsu-Seljak electron-pressure profile,
// truncated at r_Delta = c r_s, in units of 4 pi r_s^3 P_e(0):
//
//   F(c, y) = int_0^c x^2 [sin(y x)/(y x)] theta(x)^(Gamma/(Gamma-1)) dx
//
// Limits: F(c, 0) = int_0^c x^2 theta^(Gamma/(Gamma-1)) dx < F0(c),
// since theta < 1 for x > 0; and |F(c, y)| <= F(c, 0) for every y.
//
// Numerics:
//   Gauss-Legendre on [0, c]; init as in F0_KS_nointerp.
//
// Cache invalidation:
// the static GL table rebuilds when Ntable.random changes.
//
// Parameters:
//   c    - concentration (the upper limit, in units of r_s)
//   krs  - y = k r_s, dimensionless
//   init - 1 = build the static table only, 0 = integrate
//
// Returns:
//   F(c, y), dimensionless
// ---------------------------------------------------------------------------
double F_KS_nointerp(
    double c,       // concentration: upper limit, in units of r_s
    double krs,     // y = k r_s, dimensionless
    const int init  // 1 = build the static GL table only, 0 = integrate
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static gsl_integration_glfixed_table* w = NULL;

  if (NULL == w || fdiff2(cache[0], Ntable.random)) {
    const int hdi = abs(Ntable.high_def_integration);
    const size_t szint = (0 == hdi) ? 256 :
                         (1 == hdi) ? 512 : 1024; // predefined GSL tables
    if (w != NULL)  gsl_integration_glfixed_table_free(w);
    w = malloc_gslint_glfixed(szint);
    cache[0] = Ntable.random;
  }

  double ar[1] = {krs};
  const double cmin = 0.0;
  const double cmax = c;

  double res = 0.0;
  if (1 == init) {
    res = int_F_KS((cmin + cmax)/2.0, (void*) ar);
  }
  else {
    gsl_function F;
    F.params = (void*) ar;
    F.function = int_F_KS;
    res = gsl_integration_glfixed(&F, cmin, cmax, w);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shape factor of the bound-gas pressure window:
//
//   u_KS(c, k, r_v) = F(c, y)/F0(c),   y = k r_v/c = k r_s
//
// the Fourier transform of the pressure profile per unit bound-gas mass
// and per unit central temperature. The full window (u_y_bnd) is
//
//   W_p(M, k) = [k_B T_v f_bnd M/(m_p mu_e)] u_KS
//
// Unlike the matter u(k|M), u_KS does not tend to 1 at k -> 0:
//
//   u_KS(c, 0) = F(c, 0)/F0(c) = mass-weighted mean of theta
//              = <T_g>/T_v < 1
//
// the mass-weighted gas temperature in units of the central temperature
// T_v (theta(0) = 1). For every k, |u_KS(c, k)| <= u_KS(c, 0).
//
// Of the gas parameters, only Gamma = nuisance.gas[0] enters.
//
// Numerics:
//   table in (c, ln y) on Ntable.halo_uks_nc x Ntable.halo_uks_nx nodes
//   over [limits.halo_uks_cmin, limits.halo_uks_cmax] x
//   [ln limits.halo_uks_xmin, ln limits.halo_uks_xmax], read with
//   interpol2d. Off the table, interpol2d returns 0 for c outside its
//   range and, for ln y beyond an edge, the edge value plus the signed
//   overshoot (ln y - edge); neither is a clamp.
//
// Cache invalidation:
//   allocation and limits: rebuilt when Ntable.random changes
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
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = 0;
  static double* norm = 0;
  static double  lim[2][3]; // lim[0][0] = cmin;  lim[1][0] = lnxmin;
                            // lim[0][1] = cmax;  lim[1][1] = lnxmax;
                            // lim[0][2] = dc;    lim[1][2] = dlnx; 
 
  if (NULL == table || fdiff2(cache[1], Ntable.random)) {   
    if (table != NULL) free(table); 
    table = (double**) malloc2d(Ntable.halo_uks_nc, Ntable.halo_uks_nx);
    if (norm != NULL) free(norm); 
    norm = (double*) malloc1d(Ntable.halo_uks_nc);

    lim[0][0] = limits.halo_uks_cmin; 
    lim[0][1] = limits.halo_uks_cmax;
    lim[0][2] = (lim[0][1] - lim[0][0])/((double) Ntable.halo_uks_nc - 1.);
    // ln y range: limits.halo_uks_xmin .. xmax bracket the k r_Delta/c
    // the code can ask for
    lim[1][0] = log(limits.halo_uks_xmin);
    lim[1][1] = log(limits.halo_uks_xmax); 
    lim[1][2] = (lim[1][1] - lim[1][0])/((double) Ntable.halo_uks_nx - 1.); 
  }

  if (fdiff2(cache[0], nuisance.random_gas) || fdiff2(cache[1], Ntable.random)) 
  { 
    (void) F0_KS_nointerp(lim[0][0], 1);                 // init static vars
    (void) F_KS_nointerp(lim[0][0],exp(lim[1][0]), 1);   // init static vars
    #pragma omp parallel for schedule(static,1)
    for (int i=0; i<Ntable.halo_uks_nc; i++) {
      norm[i] = F0_KS_nointerp(lim[0][0] + i*lim[0][2], 0);
    }
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.halo_uks_nc; i++) {
      for (int j=0; j<Ntable.halo_uks_nx; j++) {
        table[i][j] = F_KS_nointerp(lim[0][0]+i*lim[0][2], 
                                    exp(lim[1][0]+j*lim[1][2]), 0)/norm[i];
      }
    }
    cache[0] = nuisance.random_gas; 
    cache[1] = Ntable.random;
  }
  return interpol2d(table, 
    Ntable.halo_uks_nc, lim[0][0], lim[0][1], lim[0][2], c, 
    Ntable.halo_uks_nx, lim[1][0], lim[1][1], lim[1][2], log(k * rv/c));
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
// Sign: no clipping. Where f_bnd + f_* exceeds Omega_b/Omega_m, f_ejc
// is negative: with M_0 = 1e14, beta = 0.6, A_* = 0.03 and
// Omega_b/Omega_m = 0.156 this happens above about 10^15.9 M_sun/h.
//
// Parameters:
//   M - halo mass in M_sun/h (A_* = nuisance.gas[6],
//       log10 M_* = nuisance.gas[7], sigma_* = nuisance.gas[8])
//
// Returns:
//   f_ejc, dimensionless, at most Omega_b/Omega_m
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
  
  return cosmology.Omega_b/cosmology.Omega_m - frac_bnd(M) - frac_star;
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
// The sign follows f_ejc (frac_ejc).
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
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double ngal(const int ni, const double a)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double lim[3]; // [0] = amin; [1] = amax; [2] = da

  if (table == NULL || 
      fdiff2(cache[1], Ntable.random) ||
      fdiff2(cache[3], redshift.random_clustering)) 
  { 
    if (table != NULL) free(table);
    table = (double**) malloc2d(redshift.clustering_nbin, Ntable.N_a);

    lim[0] = 1.0/(redshift.clustering_zdist_zmax_all + 1.0);
    lim[1] = 1.0/(redshift.clustering_zdist_zmin_all + 1.0);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_a - 1.0);
  }
  
  if (fdiff2(cache[0], cosmology.random) || 
      fdiff2(cache[1], Ntable.random)    ||
      fdiff2(cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(cache[3], redshift.random_clustering))
  {
    (void) ngal_nointerp(0, lim[0], 1);    
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<redshift.clustering_nbin; i++) {
      for (int j=0; j<Ntable.N_a; j++) {
        table[i][j] = ngal_nointerp(i, lim[0] + j*lim[2], 0);
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_galaxy_bias;
    cache[3] = redshift.random_clustering;
  }
  return ((a < lim[0]) || (a > lim[1]))? 0.0 :
    interpol1d(table[ni], Ntable.N_a, lim[0], lim[1], lim[2], a);
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

double bgal(const int ni, const double a)
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
  static double lim[3]; // [0] = amin; [1] = amax; [2] = da

  if (NULL == table || 
      fdiff2(cache[1], Ntable.random) ||
      fdiff2(cache[3], redshift.random_clustering)) 
  {  
    if (table != NULL) free(table); 
    table = (double**) malloc2d(redshift.clustering_nbin, Ntable.N_a);
    lim[0] = 1.0/(redshift.clustering_zdist_zmax_all + 1.0);
    lim[1] = 1.0/(redshift.clustering_zdist_zmin_all + 1.0);
    lim[2] = (lim[1] - lim[0])/((double) Ntable.N_a - 1.0);
  }
  if (fdiff2(cache[0], cosmology.random) || 
      fdiff2(cache[1], Ntable.random)    ||
      fdiff2(cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(cache[3], redshift.random_clustering)) 
  {
    (void) bgal_nointerp(0, lim[0], 1); // init static vars  
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<redshift.clustering_nbin; i++) {
      for (int j=0; j<Ntable.N_a; j++) {
        table[i][j] = bgal_nointerp(i, lim[0] + j*lim[2], 0);
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
    cache[2] = nuisance.random_galaxy_bias;
    cache[3] = redshift.random_clustering;
  }  
  return (a < lim[0]) || (a > lim[1]) ? 0.0 : 
    interpol1d(table[ni], Ntable.N_a, lim[0], lim[1], lim[2], a);
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

double p_mm(
    const double k, 
    const double a
  )
{ 
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double** table = NULL;
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
  if (fdiff2(cache[0], cosmology.random) || fdiff2(cache[1], Ntable.random)) {
    (void) p_xy_nointerp(exp(lim[1][0]), lim[0][0], 0, 1); 
    #pragma omp parallel for collapse(2) schedule(static,1)
    for (int i=0; i<Ntable.N_a; i++) {
      for (int j=0; j<Ntable.N_k_nlin; j++) { 
        table[i][j] = log(p_xy_nointerp(exp(lim[1][0] + j*lim[1][2]), 
                                        lim[0][0] + i*lim[0][2], 0, 0));
      }
    }
    cache[0] = cosmology.random;
    cache[1] = Ntable.random;
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

double p_gm(
    const double k, 
    const double a, 
    const int ni
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double*** table = NULL;
  static double** lim = NULL; //lim[:,0] = amin; lim[:,1] = amax; lim[:,2] = da; 
                              //lim[redshift.clustering_nbin][0] = lnkmin; 
                              //lim[redshift.clustering_nbin][1] = lnkmax; 
                              //lim[redshift.clustering_nbin][2] = dlnk; 

  const int nbin = redshift.clustering_nbin;
  const int na = (int) Ntable.N_a/5.0; // range is the (\delta a) of a single bin
  
  if (NULL == table || fdiff2(cache[1], Ntable.random))
  {
    if (table != NULL) free(table);
    table = (double***) malloc3d(nbin, na, Ntable.N_k_nlin);
    if (lim != NULL) free(lim);
    lim = (double**) malloc2d(nbin+1, 3);
    for (int l=0; l<redshift.clustering_nbin; l++) {
      lim[l][0] = amin_lens(l);
      lim[l][1] = amax_lens(l);
      lim[l][2] = (lim[l][1] - lim[l][0])/((double) na - 1.0);
    }
    lim[nbin][0] = log(limits.k_min_cH0);
    lim[nbin][1] = log(limits.k_max_cH0);
    lim[nbin][2] = (lim[nbin][1]-lim[nbin][0])/((double) Ntable.N_k_nlin - 1.0);
  }

  if (fdiff2(cache[0], cosmology.random) || 
      fdiff2(cache[1], Ntable.random)    ||
      fdiff2(cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(cache[3], redshift.random_clustering))
  { 
    (void) p_gm_nointerp(exp(lim[nbin][0]), lim[0][0], 0, 1); // init static vars
    #pragma omp parallel for collapse(3) schedule(static,1)
    for (int l=0; l<redshift.clustering_nbin; l++) {
      for (int i=0; i<na; i++) {
        for (int j=0; j<Ntable.N_k_nlin; j++) {
          table[l][i][j] = log(p_gm_nointerp(exp(lim[nbin][0] + j*lim[nbin][2]), 
                                             lim[l][0] + i*lim[l][2], l, 0));
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

double p_gg(
    const double k, 
    const double a, 
    const int ni, 
    const int nj
  )
{
  static uint64_t cache[MAX_SIZE_ARRAYS];
  static double*** table = NULL;
  static double** lim = NULL; //lim[0,:] = amin; lim[:,1] = amax; lim[:,2] = da; 
                              //lim[redshift.clustering_nbin] = lnkmin; 
                              //lim[redshift.clustering_nbin] = lnkmax; 
                              //lim[redshift.clustering_nbin] = dlnk; 
  const int nbin = redshift.clustering_nbin;
  const int na = (int) Ntable.N_a/5.0;

  if (NULL == table || fdiff2(cache[1], Ntable.random)) {
    if (table != NULL) free(table);
    table = (double***) malloc3d(nbin, na, Ntable.N_k_nlin);
    if (lim != NULL) free(lim);
    lim = (double**) malloc2d(nbin+1, 3);
    for (int l=0; l<redshift.clustering_nbin; l++) {
      lim[l][0] = amin_lens(l);
      lim[l][1] = amax_lens(l);
      lim[l][2] = (lim[l][1] - lim[l][0])/((double) na - 1.);
    }
    lim[nbin][0] = log(limits.k_min_cH0);
    lim[nbin][1] = log(limits.k_max_cH0);
    lim[nbin][2] = (lim[nbin][1]-lim[nbin][0])/((double) Ntable.N_k_nlin - 1.);
  }
  if (fdiff2(cache[0], cosmology.random) || 
      fdiff2(cache[1], Ntable.random)    ||
      fdiff2(cache[2], nuisance.random_galaxy_bias) ||
      fdiff2(cache[3], redshift.random_clustering))
  { 
    (void) p_gg_nointerp(exp(lim[nbin][0]), lim[0][0], 0, 0, 1); // init static vars
    #pragma omp parallel for collapse(3) schedule(static,1)
    for (int l=0; l<nbin; l++) {
      for (int i=0; i<na; i++) {
        for (int j=0; j<Ntable.N_k_nlin; j++) {
          table[l][i][j] = log(p_gg_nointerp(exp(lim[nbin][0]+j*lim[nbin][2]),
                                             lim[l][0]+i*lim[l][2], l, l, 0));
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
  if (ni != nj) {
    log_fatal("cross-tomography (ni,nj) = (%d,%d) bins not supported", ni, nj);
    exit(1);
  }  
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
  
  // Parameterization of Zehavi et al. 
  // hod[zi][] = {lg(M_min), sigma_{lg M}, lg M_1, lg M_0, alpha, f_c}
  // gbias.gc[] = {f_g} (shift of concentration parameter: c_g(M) = f_g c(M))
  
  // Values from Coupon etal. (2012) for red gals with M_r < -21.8 (Table B.2)
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
      nuisance.hod[3][1] = 0.35;
      nuisance.hod[3][2] = 13.94;
      nuisance.hod[3][3] = 12.15;
      nuisance.hod[3][4] = 1.52;
      nuisance.hod[3][5] = 1.00;
      nuisance.gb[0][ni] = hm_funcs_nointerp(ni, a, 3, 0);
      break;
    }
    case 4:
    { // no information for higher redshift populations - copy 1<z<1.2 values
      nuisance.hod[4][0] = 12.80;
      nuisance.hod[4][1] = 0.35;
      nuisance.hod[4][2] = 13.94;
      nuisance.hod[4][3] = 12.15;
      nuisance.hod[4][4] = 1.52;
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
  // NFW transform at c_g = gc[ni] * c(M), and gc = 0 would collapse the
  // galaxy profile
  nuisance.gc[ni] = 1.0;

  log_debug("HOD: bin %d; <z> %.2f; <n_g> %e(h/Mpc)^3", ni, z, 
    ngal_nointerp(ni, a, 0)*pow(cosmology.coverH0, -3.0));
  
  log_debug("HOD: bin %d; <z> %.2f; <M> h/Msun %.4e", ni, z, mmean_nointerp(ni,a,0));
  
  log_debug("HOD: bin %d; <z> %.2f; f_sat %.3f", ni, z, fsat_nointerp(ni,a,0));
  
  log_debug("HOD: bin %d; <z> %.2f; <b_g> %.2f", ni, z, nuisance.gb[0][ni]);
}
