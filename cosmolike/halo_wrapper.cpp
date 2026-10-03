#include <string>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <cmath>
#include <stdexcept>
#include <array>
#include <random>
#include <variant>
#include <cmath>

// SPDLOG
#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_DEBUG
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/cfg/env.h>

// ARMADILLO LIB AND PYBIND WRAPPER (CARMA)
#include <carma.h>
#include <armadillo>

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
namespace py = pybind11;

// cosmolike
#include "cosmolike/basics.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/halo.h"
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"
#include "cosmolike/generic_interface.hpp"
#include "cosmolike/halo_wrapper.hpp"

// ---------------------------------------------------------------------------
// Pybind wrappers of the halo model (halo.c). The header
// (halo_wrapper.hpp) documents the call chain, the units and the
// warm-up once; each function below says which C function it calls and
// what comes back.
//
// Conventions shared by every wrapper here:
//
//   scalar overload = one C call, one number back
//   array overload  = the same C call in a serial loop over the k array,
//                     one number per k, returned as arma::Col<double>
//   bad input       = spdlog::critical + exit(1), the same way the C
//                     code itself fails (a bad bin or k <= 0 would
//                     otherwise index a table out of bounds or take
//                     log(0))
//
// carma converts at the pybind11 boundary: a numpy array handed to an
// arma::Col<double> parameter arrives as an armadillo column (a copy),
// and a returned arma::Col<double> reaches Python as a numpy array of
// shape (n, 1) - np.ravel gives the flat (n,) array.
// ---------------------------------------------------------------------------
namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// INPUT CHECKS (PRIVATE)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Abort unless ni names a lens (clustering) tomographic bin.
//
// The HOD tables are indexed [ni][...], so a bin outside
// [0, clustering_nbin) would read unrelated memory rather than fail.
//
// Parameters:
//   fname - name of the calling wrapper, for the message
//   ni    - lens bin to check
//
// Returns:
//   nothing; an invalid bin aborts (spdlog::critical + exit)
// ---------------------------------------------------------------------------
static void check_lens_bin(const char* fname, const int ni)
{
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    spdlog::critical("{}: invalid bin input ni = {} (clustering_nbin = {})",
                     fname, ni, redshift.clustering_nbin);
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Abort unless k is a positive wavenumber (the spectra tables are read
// at ln k).
//
// Parameters:
//   fname - name of the calling wrapper, for the message
//   k     - wavenumber in (c/H0)^-1
//
// Returns:
//   nothing; k <= 0 (or NaN) aborts (spdlog::critical + exit)
// ---------------------------------------------------------------------------
static void check_wavenumber(const char* fname, const double k)
{
  if (!(k > 0)) {
    spdlog::critical("{}: k = {} not positive", fname, k);
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Abort unless the k array is non-empty and every entry is positive.
//
// Parameters:
//   fname - name of the calling wrapper, for the message
//   k     - wavenumbers in (c/H0)^-1
//
// Returns:
//   nothing; an empty array or any k(i) <= 0 aborts (spdlog::critical +
//   exit)
// ---------------------------------------------------------------------------
static void check_wavenumbers(const char* fname, const arma::Col<double>& k)
{
  if (0 == k.n_elem) {
    spdlog::critical("{}: k array size = 0", fname);
    exit(1);
  }
  for (arma::uword i=0; i<k.n_elem; i++) {
    if (!(k(i) > 0)) {
      spdlog::critical("{}: k({}) = {} not positive", fname, i, k(i));
      exit(1);
    }
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Abort unless a is a scale factor strictly inside (0, 1).
//
// The IA readers index their tables at a and evaluate (1 + z) = 1/a, so
// a <= 0 would divide by zero or read a negative redshift, and a >= 1
// lies outside every source redshift range.
//
// Parameters:
//   fname - name of the calling wrapper, for the message
//   a     - scale factor
//
// Returns:
//   nothing; a outside (0, 1) (or NaN) aborts (spdlog::critical + exit)
// ---------------------------------------------------------------------------
static void check_scale_factor(const char* fname, const double a)
{
  if (!(a > 0 && a < 1)) {
    spdlog::critical("{}: a = {} outside (0, 1)", fname, a);
    exit(1);
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// PEAK-BACKGROUND SPLIT KERNELS (Tinker et al. 2010) AND CONCENTRATION
//
// The halo model labels a halo of mass M by its peak height
//
//   nu = delta_c / sigma(M, a),   sigma(M, a) = sqrt(sigma2(M,a))
//
// (delta_c = 1.686 the collapse threshold, sigma the rms linear density
// fluctuation in a sphere holding mass M, D the growth factor). Rare,
// massive halos have nu >> 1. In this variable the Tinker et al. 2010
// fits are nearly universal:
//
//   f(nu) dnu = fraction of all matter in halos with peak height in
//               [nu, nu + dnu]            (Eqs. 8-12, Table 4)
//   b(nu)     = linear bias of those halos (Eq. 6, Table 2)
//
// and the mass function follows as dn/dlnM = (rho_m/M) f(nu) nu
// dln nu/dln M - which is where dlognudlogm enters.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Halo bias b(nu) of the peak-background split, Tinker et al. 2010 Eq. 6
// with the Table 2 coefficients at Delta = 200.
//
// Calls halo.c hb1nu (fit selected by like.halo_model[1]). A closed form,
// no table.
//
// Parameters:
//   nu - peak height delta_c/sigma(M, a)
//   a  - scale factor (the Delta = 200 fit does not evolve; kept for the
//        halo.c signature)
//
// Returns:
//   b(nu), dimensionless
// ---------------------------------------------------------------------------
double hb1nu_cpp(
    const double nu,  // peak height
    const double a    // scale factor
  )
{
  return hb1nu(nu, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Multiplicity function f(nu) of the Tinker et al. 2010 mass function,
// Eqs. 8-12 with the Table 4 parameters at Delta = 200; the redshift
// evolution of the parameters is frozen beyond z = 3 (the fit's range).
// The amplitude alpha of f(nu) is set at every a by Eq. 7 of the paper,
// int b f dnu = 1 over all nu (matter is unbiased with respect to
// itself).
//
// Calls halo.c fnu (fit selected by like.halo_model[0]): a closed form
// in nu whose alpha is read from a table in a (halo.c tinker_alpha,
// built once per process).
//
// Parameters:
//   nu - peak height delta_c/sigma(M, a)
//   a  - scale factor, 0 < a < 1 (halo.c aborts otherwise)
//
// Returns:
//   f(nu), dimensionless (per unit nu)
// ---------------------------------------------------------------------------
double fnu_cpp(
    const double nu,  // peak height
    const double a    // scale factor
  )
{
  return fnu(nu, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Halo concentration c = r_Delta/r_s, Bhattacharya et al. 2013 Table 2
// (Delta = 200 times the mean matter density):
//
//   c = 9.0 nu^-0.29 D^1.15,   nu = delta_c/sigma_cb(m,a)
//
// Calls halo.c conc (fit selected by like.halo_model[2]), which reads the
// cached sigma2(m,a) table of cosmo3D.c; D = sigma_cb(m,a)/sigma_cb(m,1).
//
// Parameters:
//   m         - halo mass in M_sun/h
//   a         - scale factor
//
// Returns:
//   c(m), dimensionless
// ---------------------------------------------------------------------------
double conc_cpp(
    const double m,          // halo mass in M_sun/h
    const double a          // scale factor
  )
{
  return conc(m, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Logarithmic slope d ln nu / d ln M of the peak height.
//
// Massive neutrinos give mass-dependent growth, so the cb slope must
// be evaluated at the requested scale factor:
//
//   d ln nu/d ln M = -(1/2) d ln sigma2(M,a)/d ln M at fixed a
//
// Calls halo.c dlognudlogm, which reads the FFTLog slope table by
// bilinear interpolation in ln M and a. The mass grid contains
// Ntable.N_M[NODES_DENSE] nodes between limits.halo_m's endpoints.
//
// Parameters:
//   M - halo mass in M_sun/h
//   a - scale factor at which the mass slope is evaluated
//
// Returns:
//   d ln nu/d ln M at fixed a, dimensionless
// ---------------------------------------------------------------------------
double dlognudlogm_cpp(
    const double M,  // halo mass in M_sun/h
    const double a   // scale factor
  )
{
  return dlognudlogm(M, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Share of the halo-bias consistency relation that the tabulated mass
// range covers:
//
//   bias_norm(a) = int_{nu(M_min)}^{nu(M_max)} b(nu) f(nu) dnu
//
// with M_min, M_max = limits.halo_m[RANGE_MIN], limits.halo_m[RANGE_MAX]. Over all
// nu the integral is 1 by construction: the amplitude alpha of f(nu) is
// set at every a by Tinker et al. 2010 Eq. 7, int b f dnu = 1 (matter
// is unbiased with respect to itself; halo.c tinker_alpha). Over the
// tabulated mass range it is below 1, 0.80 at z = 0 and 0.79 at z = 1
// with the defaults, because the light halos under M_min hold a sizable
// share of the matter. The 2-halo sum of a halo-model matter spectrum
// (future_port_unfinished/halo_pmm.c) runs over the tabulated range and
// adds the missing 1 - bias_norm(a)
// back as halos of mass exactly M_min, the additive correction of Mead
// et al. 2020 (2005.00009 App. A), so that P_2h -> P_lin as k -> 0
// with the mass function left as fitted.
//
// Calls halo.c bias_norm: a cached table on Ntable.N_a nodes in a over
// [limits.a_min, 0.9999999], filled by one threaded Gauss-Legendre pass
// and read by linear interpolation (constant extrapolation past the
// last node).
//
// Parameters:
//   a - scale factor
//
// Returns:
//   bias_norm(a), dimensionless
// ---------------------------------------------------------------------------
double bias_norm_cpp(
    const double a   // scale factor
  )
{
  return bias_norm(a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO AND GAS PROFILES
//
// A profile enters the halo model through its Fourier transform
// normalized by the enclosed mass, u(k|m): u -> 1 as k -> 0 (on scales
// much larger than the halo it is a point mass) and u falls off once
// k r_Delta/c ~ 1 (the halo is resolved).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fourier transform of the NFW density profile truncated at r_Delta,
// normalized to 1 at k = 0 (Cooray & Sheth 2002, analytic form in the
// sine and cosine integrals Si, Ci):
//
//   r_Delta = (3 m/(4 pi Delta rho_m))^(1/3)   (Delta = 200)
//   x       = k r_Delta/c                      (k times the scale radius)
//
// Calls halo.c u_nfw_c: a cached table in ln t of the smooth auxiliary
// functions f, g of Si, Ci (Ntable.halo_nfw_n nodes), built once and
// rebuilt only when Ntable changes.
//
// Parameters:
//   c - concentration r_Delta/r_s
//   k - wavenumber in (c/H0)^-1
//   m - halo mass in M_sun/h
//   a - scale factor (unused by the NFW form; kept for the halo.c
//       signature)
//
// Returns:
//   u(k|m), dimensionless; equals 1 at k = 0
// ---------------------------------------------------------------------------
double u_nfw_c_cpp(
    const double c,   // concentration
    const double k,   // wavenumber in (c/H0)^-1
    const double m,   // halo mass in M_sun/h
    const double a    // scale factor
  )
{
  return u_nfw_c(c, k, m, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HOD INTEGRALS
//
// The halo occupation distribution (Zehavi et al. 2011 form) gives the
// mean number of galaxies of lens bin ni in a halo of mass M,
//
//   <N|M> = f_c N_c(M) + N_s(M)   (centrals + satellites)
//
// with the six parameters nuisance.hod[ni][0..5] = {lg M_min, sigma_lgM,
// lg M_1, lg M_0, alpha, f_c}. Each quantity below integrates a weight
// times <N|M> over the mass function dn/dlnM, from
// 10^(nuisance.hod[ni][0] - 2) to limits.halo_m[RANGE_MAX] in M_sun/h:
//
//   ngal  = int dlnM dn/dlnM <N|M>                 (number density)
//   bgal  = int dlnM dn/dlnM <N|M> b(M) / ngal     (mean galaxy bias)
//
// halo.c aborts on a bin whose lg M_min lies outside [10, 16]. The
// ngal/bgal tables build all bins at once, so table reads (ngal, bgal,
// p_gm, p_gg) need every lens bin set first.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Galaxy number density of lens bin ni.
//
// Calls halo.c ngal: a cached table over (bin, a) on Ntable.N_a nodes in
// a across the whole lens redshift range, rebuilt when the cosmology,
// the HOD (nuisance.random_galaxy_bias) or the lens n(z) change; 0
// outside that range.
//
// Parameters:
//   ni - lens bin
//   a  - scale factor
//
// Returns:
//   ngal in (c/H0)^-3 (multiply by coverH0^-3 for (h/Mpc)^3)
// ---------------------------------------------------------------------------
double ngal_cpp(
    const int ni,     // lens bin
    const double a    // scale factor
  )
{
  check_lens_bin("ngal_cpp", ni);
  return ngal(ni, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Number-weighted mean galaxy bias of lens bin ni (the bgal line of
// the banner above).
//
// Calls halo.c bgal: a cached (bin, a) table like ngal's, rebuilt on the
// same keys; 0 outside the lens redshift range.
//
// Parameters:
//   ni - lens bin
//   a  - scale factor
//
// Returns:
//   bgal, dimensionless (order unity)
// ---------------------------------------------------------------------------
double bgal_cpp(
    const int ni,     // lens bin
    const double a    // scale factor
  )
{
  check_lens_bin("bgal_cpp", ni);
  return bgal(ni, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO-MODEL POWER SPECTRA
//
// Every spectrum is a 1-halo term (both points in the same halo) plus a
// 2-halo term (points in two different halos, correlated through the
// linear bias):
//
//   P_XY(k) = int dn u_X u_Y           (1-halo)
//           + I_X(k) I_Y(k) P(k)       (2-halo)
//
//   u_X = the profile of field X in one halo (matter, galaxies through
//         the HOD)
//   I_X = int dn b u_X, the bias-weighted mean profile, plus the HMx
//         term that stands in for the halos below limits.halo_m[RANGE_MIN]
//         (halo.c POWER SPECTRA banner)
//
// halo.c tabulates ln P on a uniform (a, ln k) grid - Ntable.N_a x
// Ntable.N_k_nlin nodes over [limits.k_cH0[RANGE_MIN],
// limits.k_cH0[RANGE_MAX]] in k, per lens bin over that bin's a-range for
// p_gm, p_gg - and interpolates bilinearly. The first call pays the whole table build (a mass
// integral per node); later calls are lookups.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Galaxy-matter power spectrum of lens bin ni at one (k, a):
//
//   P_gm = bgal Pdelta + (1-halo HOD integral)/ngal
//
// Calls halo.c p_gm: one table per lens bin over that bin's a-range
// [amin_lens(ni), amax_lens(ni)] (0 outside it), rebuilt when the
// cosmology, the HOD or the lens n(z) change.
//
// Parameters:
//   k  - wavenumber in (c/H0)^-1; k <= 0 aborts (spdlog::critical + exit)
//   a  - scale factor
//   ni - lens bin; outside [0, clustering_nbin) aborts
//
// Returns:
//   P_gm(k, a) in (c/H0)^3
// ---------------------------------------------------------------------------
double p_gm_cpp(
    const double k,   // wavenumber in (c/H0)^-1
    const double a,   // scale factor
    const int ni      // lens bin
  )
{
  check_wavenumber("p_gm_cpp", k);
  check_lens_bin("p_gm_cpp", ni);
  return p_gm(k, a, ni);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Galaxy-matter power spectrum of lens bin ni at many k, one a (serial
// loop over the scalar call).
//
// Parameters:
//   k  - wavenumbers in (c/H0)^-1; an empty array or any k(i) <= 0
//        aborts
//   a  - scale factor
//   ni - lens bin; outside [0, clustering_nbin) aborts
//
// Returns:
//   arma::Col of P_gm(k(i), a) in (c/H0)^3, same length and order as k
// ---------------------------------------------------------------------------
arma::Col<double> p_gm_cpp(
    const arma::Col<double> k,   // wavenumbers in (c/H0)^-1
    const double a,              // scale factor
    const int ni                 // lens bin
  )
{
  check_wavenumbers("p_gm_cpp", k);
  check_lens_bin("p_gm_cpp", ni);
  arma::Col<double> res(k.n_elem, arma::fill::zeros);
  for (arma::uword i=0; i<k.n_elem; i++) {
    res(i) = p_gm(k(i), a, ni);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Galaxy-galaxy power spectrum of lens bin ni at one (k, a):
//
//   P_gg = bgal^2 Pdelta + (1-halo HOD integral)/ngal^2
//
// Calls halo.c p_gg, tabulated like p_gm. Only the auto spectrum exists:
// halo.c aborts when ni != nj.
//
// Parameters:
//   k  - wavenumber in (c/H0)^-1; k <= 0 aborts (spdlog::critical + exit)
//   a  - scale factor
//   ni - lens bin; outside [0, clustering_nbin) aborts
//   nj - second lens bin; must equal ni
//
// Returns:
//   P_gg(k, a) in (c/H0)^3
// ---------------------------------------------------------------------------
double p_gg_cpp(
    const double k,   // wavenumber in (c/H0)^-1
    const double a,   // scale factor
    const int ni,     // lens bin
    const int nj      // second lens bin (= ni)
  )
{
  check_wavenumber("p_gg_cpp", k);
  check_lens_bin("p_gg_cpp", ni);
  check_lens_bin("p_gg_cpp", nj);
  return p_gg(k, a, ni, nj);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Galaxy-galaxy power spectrum of lens bin ni at many k, one a (serial
// loop over the scalar call).
//
// Parameters:
//   k  - wavenumbers in (c/H0)^-1; an empty array or any k(i) <= 0
//        aborts
//   a  - scale factor
//   ni - lens bin; outside [0, clustering_nbin) aborts
//   nj - second lens bin; must equal ni
//
// Returns:
//   arma::Col of P_gg(k(i), a) in (c/H0)^3, same length and order as k
// ---------------------------------------------------------------------------
arma::Col<double> p_gg_cpp(
    const arma::Col<double> k,   // wavenumbers in (c/H0)^-1
    const double a,              // scale factor
    const int ni,                // lens bin
    const int nj                 // second lens bin (= ni)
  )
{
  check_wavenumbers("p_gg_cpp", k);
  check_lens_bin("p_gg_cpp", ni);
  check_lens_bin("p_gg_cpp", nj);
  arma::Col<double> res(k.n_elem, arma::fill::zeros);
  for (arma::uword i=0; i<k.n_elem; i++) {
    res(i) = p_gg(k(i), a, ni, nj);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO-MODEL INPUTS FROM cosmo3D.c
//
// The halo model is built on top of these; the unit tests compare the
// halo-model spectra against them on large scales, where the 2-halo
// term must reduce to them.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Linear growth factor D(a), normalized to D(1) = 1.
//
// Calls cosmo3D.c growfac (interpolated from the growth table the
// likelihood hands over through set_cosmology).
//
// Parameters:
//   a - scale factor
//
// Returns:
//   D(a), dimensionless
// ---------------------------------------------------------------------------
double growfac_cpp(
    const double a   // scale factor
  )
{
  return growfac(a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Linear matter power spectrum at one (k, a).
//
// Calls cosmo3D.c p_lin (interpolated from the linear P(k, z) table the
// likelihood hands over).
//
// Parameters:
//   k - wavenumber in (c/H0)^-1; k <= 0 aborts (spdlog::critical + exit)
//   a - scale factor
//
// Returns:
//   P_lin(k, a) in (c/H0)^3
// ---------------------------------------------------------------------------
double p_lin_cpp(
    const double k,   // wavenumber in (c/H0)^-1
    const double a    // scale factor
  )
{
  check_wavenumber("p_lin_cpp", k);
  return p_lin(k, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Matter power spectrum of the run mode at one (k, a): nonlinear
// (Halofit or emulator table) unless the run mode is linear.
//
// Calls cosmo3D.c Pdelta.
//
// Parameters:
//   k - wavenumber in (c/H0)^-1; k <= 0 aborts (spdlog::critical + exit)
//   a - scale factor
//
// Returns:
//   P(k, a) in (c/H0)^3
// ---------------------------------------------------------------------------
double Pdelta_cpp(
    const double k,   // wavenumber in (c/H0)^-1
    const double a    // scale factor
  )
{
  check_wavenumber("Pdelta_cpp", k);
  return Pdelta(k, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HOD PARAMETER SETTERS
//
// halo.c reads its galaxy parameters from the nuisance struct, and its
// tables remember which parameters they were built with through a cache
// key:
//
//   nuisance.random_galaxy_bias -> ngal, bgal, p_gm, p_gg tables
//
// A setter that changes a parameter must therefore draw a new key
// (RandomNumber, generic_interface.hpp), or the next call would return
// a table built for the old values.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Load halo.c's built-in HOD for lens bin ni: the Coupon et al. 2012
// fits for all galaxies with M_g - 5 log h < -21.8 (1107.0616 Table
// B.1) that halo.c set_HOD hard-codes for bins 0-4, one redshift slice
// of width 0.2 each from z = 0.2 to 1.2. set_HOD also sets the galaxy
// concentration factor nuisance.gc[ni] = 1 and stores the resulting
// mean galaxy bias in nuisance.gb[0][ni].
//
// Cache invalidation:
// draws a new nuisance.random_galaxy_bias before calling set_HOD.
// set_HOD integrates the HOD directly (no table reads), but it writes
// nuisance.hod, gc and gb, so the fresh key makes every HOD-keyed
// table rebuild on its next read and the galaxy-bias caches of
// cosmo2D.c see a changed key too.
//
// Parameters:
//   ni - lens bin; outside [0, clustering_nbin) aborts (halo.c itself
//        has values for bins 0-4 only)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_HOD_cpp(
    const int ni   // lens bin
  )
{
  check_lens_bin("set_HOD_cpp", ni);
  nuisance.random_galaxy_bias = RandomNumber::get_instance().get();
  set_HOD(ni);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Set the HOD of lens bin ni from explicit values (unit tests and
// notebooks use this instead of set_HOD's built-in table):
//
//   hod(0) = lg M_min    (M_sun/h; centrals switch on around M_min)
//   hod(1) = sigma_lgM   (width of that switch)
//   hod(2) = lg M_1      (mass scale of the satellite power law)
//   hod(3) = lg M_0      (satellite cutoff mass)
//   hod(4) = alpha       (satellite power-law slope)
//   hod(5) = f_c         (central fraction; 0 is read as 1)
//   gc     = f_g         (galaxy concentration = f_g x halo
//                         concentration)
//
// written into nuisance.hod[ni][0..5] and nuisance.gc[ni].
//
// Cache invalidation:
// draws a new nuisance.random_galaxy_bias when any value changed (fdiff);
// unchanged input leaves the key alone.
//
// Parameters:
//   ni  - lens bin; outside [0, clustering_nbin) aborts
//   hod - the six HOD parameters above; any other length or a NaN
//         entry aborts (spdlog::critical + exit)
//   gc  - galaxy concentration factor f_g
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_nuisance_hod_cpp(
    const int ni,                  // lens bin
    const arma::Col<double> hod,   // {lgMmin, sigma_lgM, lgM1, lgM0,
                                   //  alpha, f_c}
    const double gc                // galaxy concentration factor
  )
{
  check_lens_bin("set_nuisance_hod_cpp", ni);
  if (6 != hod.n_elem) {
    spdlog::critical("{}: hod array size = {} (!= 6)",
                     "set_nuisance_hod_cpp", hod.n_elem);
    exit(1);
  }
  int cache_update = 0;
  for (int j=0; j<6; j++) {
    if (std::isnan(hod(j))) {
      spdlog::critical("{}: NaN found on index {}",
                       "set_nuisance_hod_cpp", j);
      exit(1);
    }
    if (fdiff(nuisance.hod[ni][j], hod(j))) {
      cache_update = 1;
      nuisance.hod[ni][j] = hod(j);
    }
  }
  if (std::isnan(gc)) {
    spdlog::critical("{}: gc is NaN", "set_nuisance_hod_cpp");
    exit(1);
  }
  if (fdiff(nuisance.gc[ni], gc)) {
    cache_update = 1;
    nuisance.gc[ni] = gc;
  }
  if (1 == cache_update) {
    nuisance.random_galaxy_bias = RandomNumber::get_instance().get();
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO-MODEL INTRINSIC ALIGNMENT (Fortuna et al. 2021, 2003.02700)
//
// One IA population, the shape (source) sample. Red central galaxies
// align with the large-scale tidal field (the NLA 2-halo term, built in
// cosmo2D.c), and satellite galaxies point radially at the center of
// their host halo (the 1-halo term, tabulated in halo.c):
//
//   P_dI^1h(k, a) = a_1h(a)   f_1h(k) S_dI(k, a)     (signed with a_1h)
//   P_II^1h(k, a) = a_1h(a)^2 f_1h(k) S_II(k, a)     (>= 0)
//
//   a_1h(a) = a_1h [(1 + z)/(1 + z_pivot)]^eta_1h    (nuisance.ia_halo)
//   f_1h(k) = 1 - exp[-(k/k_1h)^2],   k_1h = 4 h/Mpc
//   f_2h(k) = exp[-(k/k_2h)^2],       k_2h = 6 h/Mpc
//
// S_dI, S_II are the halo-mass integrals of the satellite alignment
// (halo.c HALO-MODEL INTRINSIC ALIGNMENT banner). The red-central
// fraction f_rc(a) weights the NLA 2-halo term, and f_2h switches that
// term off above k_2h.
//
// Sign convention: the C_l cores of cosmo2D.c SUBTRACT the dI spectrum,
//
//   P_dI^phys = -[f_rc C_1 P_delta f_2h + P_dI^1h]
//
// (C_1 the NLA amplitude, > 0 for A_IA > 0), so radial alignment,
// a_1h > 0, returns a positive ia_p1h_dI and gives a negative physical
// dI correlation - the same sense as A_IA > 0.
//
// halo.c tabulates ln S_dI and ln S_II on a uniform (a, ln k) grid -
// Ntable.halo_ia_na x Ntable.N_k_nlin nodes over the source a range
// [min_i amin_source(i), max_i amax_source(i)] x [limits.k_cH0[RANGE_MIN],
// limits.k_cH0[RANGE_MAX]] - and f_rc on the same a nodes. The readers return
// 0 outside the source a range. The tables are refilled when the
// cosmology, Ntable, the IA parameters (nuisance.random_ia_halo), the
// source n(z) or the source photo-z shifts change; the first call pays
// the build (a mass integral per node), later calls are lookups.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Red-central fraction of the IA (source) sample,
//
//   f_rc(a) = int dlnM dn/dlnM f_c N_c(M) f_red,cen(M) / n_g(a)
//
// (F21's f_cen^red, the weight of the NLA 2-halo term).
//
// Calls halo.c ia_f_red_central: a cached table on Ntable.halo_ia_na
// nodes in a over the source range, read by linear interpolation.
//
// Parameters:
//   a - scale factor; outside (0, 1) aborts (spdlog::critical + exit)
//
// Returns:
//   f_rc(a), dimensionless, in [0, 1]; 0 outside the source a range
// ---------------------------------------------------------------------------
double ia_f_red_central_cpp(
    const double a   // scale factor
  )
{
  check_scale_factor("ia_f_red_central_cpp", a);
  return ia_f_red_central(a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Window of the NLA 2-halo term (F21 Eq. 31),
//
//   f_2h(k) = exp[-(k/k_2h)^2],   k_2h = 6 h/Mpc x coverH0
//
// Calls halo.c ia_window_2h. A closed form, no table.
//
// Parameters:
//   k - wavenumber in (c/H0)^-1; k <= 0 aborts (spdlog::critical + exit)
//
// Returns:
//   f_2h(k), dimensionless, in (0, 1]
// ---------------------------------------------------------------------------
double ia_window_2h_cpp(
    const double k   // wavenumber in (c/H0)^-1
  )
{
  check_wavenumber("ia_window_2h_cpp", k);
  return ia_window_2h(k);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Window of the NLA 2-halo term at many k (serial loop over the scalar
// call).
//
// Parameters:
//   k - wavenumbers in (c/H0)^-1; an empty array or any k(i) <= 0 aborts
//
// Returns:
//   arma::Col of f_2h(k(i)), dimensionless, same length and order as k
// ---------------------------------------------------------------------------
arma::Col<double> ia_window_2h_cpp(
    const arma::Col<double> k   // wavenumbers in (c/H0)^-1
  )
{
  check_wavenumbers("ia_window_2h_cpp", k);
  arma::Col<double> res(k.n_elem, arma::fill::zeros);
  for (arma::uword i=0; i<k.n_elem; i++) {
    res(i) = ia_window_2h(k(i));
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Satellite (1-halo) part of the matter-intrinsic spectrum at one (k, a)
// (F21 Eq. 17):
//
//   P_dI^1h = a_1h(a) f_1h(k) S_dI(k, a)
//
// SIGNED with a_1h: the C_l cores of cosmo2D.c subtract it (section
// banner), so a_1h > 0 returns a positive value.
//
// Calls halo.c ia_p1h_dI: ln S_dI read bilinearly in (a, ln k) from the
// cached table and exponentiated (ln S_dI continued with unit slope
// outside [ln k_min, ln k_max]).
//
// Parameters:
//   k - wavenumber in (c/H0)^-1; k <= 0 aborts (spdlog::critical + exit)
//   a - scale factor; outside (0, 1) aborts
//
// Returns:
//   P_dI^1h(k, a) in (c/H0)^3, signed; 0 outside the source a range or
//   for a_1h = 0
// ---------------------------------------------------------------------------
double ia_p1h_dI_cpp(
    const double k,   // wavenumber in (c/H0)^-1
    const double a    // scale factor
  )
{
  check_wavenumber("ia_p1h_dI_cpp", k);
  check_scale_factor("ia_p1h_dI_cpp", a);
  return ia_p1h_dI(k, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Satellite (1-halo) part of the matter-intrinsic spectrum at many k,
// one a (serial loop over the scalar call; signed with a_1h, as above).
//
// Parameters:
//   k - wavenumbers in (c/H0)^-1; an empty array or any k(i) <= 0 aborts
//   a - scale factor; outside (0, 1) aborts
//
// Returns:
//   arma::Col of P_dI^1h(k(i), a) in (c/H0)^3, same length and order as k
// ---------------------------------------------------------------------------
arma::Col<double> ia_p1h_dI_cpp(
    const arma::Col<double> k,   // wavenumbers in (c/H0)^-1
    const double a               // scale factor
  )
{
  check_wavenumbers("ia_p1h_dI_cpp", k);
  check_scale_factor("ia_p1h_dI_cpp", a);
  arma::Col<double> res(k.n_elem, arma::fill::zeros);
  for (arma::uword i=0; i<k.n_elem; i++) {
    res(i) = ia_p1h_dI(k(i), a);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Satellite (1-halo) part of the intrinsic-intrinsic E-mode spectrum at
// one (k, a) (F21 Eq. 18):
//
//   P_II^1h = a_1h(a)^2 f_1h(k) S_II(k, a)
//
// The B mode of radial alignment vanishes (F21 sec. 4.1).
//
// Calls halo.c ia_p1h_II: ln S_II read bilinearly in (a, ln k) from the
// cached table and exponentiated (ln S_II continued with unit slope
// outside [ln k_min, ln k_max]).
//
// Parameters:
//   k - wavenumber in (c/H0)^-1; k <= 0 aborts (spdlog::critical + exit)
//   a - scale factor; outside (0, 1) aborts
//
// Returns:
//   P_II^1h(k, a) in (c/H0)^3, >= 0; 0 outside the source a range or for
//   a_1h = 0
// ---------------------------------------------------------------------------
double ia_p1h_II_cpp(
    const double k,   // wavenumber in (c/H0)^-1
    const double a    // scale factor
  )
{
  check_wavenumber("ia_p1h_II_cpp", k);
  check_scale_factor("ia_p1h_II_cpp", a);
  return ia_p1h_II(k, a);
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Satellite (1-halo) part of the intrinsic-intrinsic spectrum at many k,
// one a (serial loop over the scalar call).
//
// Parameters:
//   k - wavenumbers in (c/H0)^-1; an empty array or any k(i) <= 0 aborts
//   a - scale factor; outside (0, 1) aborts
//
// Returns:
//   arma::Col of P_II^1h(k(i), a) in (c/H0)^3, same length and order as k
// ---------------------------------------------------------------------------
arma::Col<double> ia_p1h_II_cpp(
    const arma::Col<double> k,   // wavenumbers in (c/H0)^-1
    const double a               // scale factor
  )
{
  check_wavenumbers("ia_p1h_II_cpp", k);
  check_scale_factor("ia_p1h_II_cpp", a);
  arma::Col<double> res(k.n_elem, arma::fill::zeros);
  for (arma::uword i=0; i<k.n_elem; i++) {
    res(i) = ia_p1h_II(k(i), a);
  }
  return res;
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Set the halo-model intrinsic-alignment parameters (Fortuna et al.
// 2021; halo.c's IA tables, cosmo2D.c's include_halo_IA mode):
//
//   ia_halo(0) = a_1h     satellite radial alignment amplitude; |a_1h|
//                         must stay below 0.3 (the profile saturates
//                         at the 0.3 cap there - F21 Eq. 20)
//   ia_halo(1) = eta_1h   a_1h (1+z)^eta_1h/(1+z_pivot)^eta_1h
//   ia_halo(2) = z_pivot
//   ia_red(0..3)          red fractions of centrals and satellites,
//                         sigmoids in log10 M: lg M_c,cen, width_cen,
//                         lg M_c,sat, width_sat
//   ia_hod(0..5)          HOD of the IA (source) population, {lg M_min,
//                         sigma_lgM, lg M_1, lg M_0, alpha, f_c}
//
// Cache invalidation:
// draws new nuisance.random_ia_halo (the halo IA tables) and
// nuisance.random_ia (every IA consumer downstream) when any value
// changed (fdiff); unchanged input leaves both keys alone.
//
// Parameters:
//   ia_halo, ia_red, ia_hod - as above; wrong sizes, NaN entries,
//                             |a_1h| >= 0.3 or a red-fraction width
//                             <= 0 abort (spdlog::critical + exit)
//
// Returns:
//   void
// ---------------------------------------------------------------------------
void set_nuisance_ia_halo_cpp(
    const arma::Col<double> ia_halo, // a_1h, eta_1h, z_pivot
    const arma::Col<double> ia_red,  // the four red-fraction sigmoid params
    const arma::Col<double> ia_hod   // the six IA-population HOD params
  )
{
  static constexpr const char* fname = "set_nuisance_ia_halo_cpp";

  // --- 1. SIZES AND VALUES ---

  if (ia_halo.n_elem != 3 || ia_red.n_elem != 4 || ia_hod.n_elem != 6) {
    spdlog::critical("{}: sizes (ia_halo, ia_red, ia_hod) = ({}, {}, {}); "
                     "expected (3, 4, 6)", fname, ia_halo.n_elem,
                     ia_red.n_elem, ia_hod.n_elem);
    exit(1);
  }
  if (!(ia_red(1) > 0) || !(ia_red(3) > 0)) {
    spdlog::critical("{}: red-fraction widths ({}, {}) must be > 0",
                     fname, ia_red(1), ia_red(3));
    exit(1);
  }
  if (!(std::fabs(ia_halo(0)) < 0.3)) {
    spdlog::critical("{}: |a_1h| = {} must be < 0.3 (the alignment "
                     "profile saturates at the cap)", fname, ia_halo(0));
    exit(1);
  }

  // --- 2. WRITE, NOTING ANY CHANGE ---

  int cache_update = 0;
  auto write = [&](const arma::Col<double>& v, double* dst) {
    for (int j=0; j<static_cast<int>(v.n_elem); j++) {
      if (std::isnan(v(j))) {
        spdlog::critical("{}: NaN found on index {}", fname, j);
        exit(1);
      }
      if (fdiff(dst[j], v(j))) {
        cache_update = 1;
        dst[j] = v(j);
      }
    }
  };
  write(ia_halo, nuisance.ia_halo);
  write(ia_red, nuisance.ia_red);
  write(ia_hod, nuisance.ia_hod);

  // the halo IA tables key on random_ia_halo; every IA consumer
  // downstream (C_ss, C_gs, xi_pm, gamma_t) keys on random_ia
  if (1 == cache_update) {
    nuisance.random_ia_halo = RandomNumber::get_instance().get();
    nuisance.random_ia = RandomNumber::get_instance().get();
  }
}

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

}  // namespace cosmolike_interface
