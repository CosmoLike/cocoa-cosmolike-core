#include <carma.h>
#include <armadillo>
#include <map>

// Python Binding
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#ifndef __COSMOLIKE_HALO_WRAPPER_HPP
#define __COSMOLIKE_HALO_WRAPPER_HPP

namespace cosmolike_interface
{

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// Halo-model bindings (halo.c), implemented in halo_wrapper.cpp.
//
// halo.c computes the halo model: the halo mass function and halo bias
// (Tinker et al. 2010 fits), the halo concentration and density profile
// (NFW), the gas pressure profile (Komatsu-Seljak), the HOD galaxy
// counts, and the power spectra assembled from them. Its functions are
// plain C. This layer makes them callable from Python, so the unit
// tests (projects/roman_real/tests/test_halo.py) and notebooks can
// evaluate them one number at a time.
//
// One Python call travels
//
//   ci.p_mm(k, a)                         (Python)
//     -> m.def("p_mm", ...)               (project interface.cpp)
//     -> p_mm_cpp(k, a)                   (this layer: checks the input,
//                                          loops over arrays)
//     -> p_mm(k, a)                       (halo.c: reads a cached table)
//     -> on first use, or after a cache key changed: the table is
//        rebuilt from p_xy_nointerp on the (a, ln k) grid
//
// Names: each function below is the C function's name plus _cpp, and
// its Python name is the C name itself (as for the sigma2 and
// scale-cut bindings): halo.c p_mm -> p_mm_cpp -> ci.p_mm.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Units: halo.c works in cosmolike code units and this layer passes them
// through unconverted (unlike the scale-cut wrappers, whose k is in
// (Mpc/h)^-1):
//
//   k      = wavenumber in (c/H0)^-1: k = k[h/Mpc] * coverH0, with
//            coverH0 = c/H0 = 2997.92458 Mpc/h
//   P      = power spectra in (c/H0)^3: P = P[(Mpc/h)^3] / coverH0^3
//   m, M   = halo mass in M_sun/h
//   rv     = halo radius in c/H0
//   ngal   = comoving galaxy number density in (c/H0)^-3
//   a      = scale factor; wherever the Tinker fits or the HOD enter,
//            halo.c requires 0 < a < 1 (a = 1 aborts)
//   ni, nj = lens (clustering) tomographic bins, counted from 0
//
// Dimensionless: u_nfw_c, u_KS, conc, hb1nu, fnu, dlognudlogm,
// bias_norm, fsat.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// The init flag. Several halo.c integrators take an int init argument:
//
//   init = 1 -> evaluate the integrand once, at the midpoint, and return
//               that throwaway number; the only purpose of the call is
//               to build the function's static state (its Gauss-Legendre
//               table and every cached table the integrand reads)
//   init = 0 -> the actual integral
//
// halo.c's table builders make one init = 1 call right before their
// OpenMP fill loops: a static table built lazily from inside a threaded
// loop would be built by several threads at once (a data race). The
// wrappers here always pass init = 0. Python calls arrive one at a time,
// so any lazy build a wrapper triggers is serial already, and the
// throwaway init = 1 value must never reach Python.
//
// Threading: the array overloads loop serially over their inputs. The
// parallelism lives inside halo.c, in the table builders a first call
// triggers (each with its own serial priming call, as above).
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// PEAK-BACKGROUND SPLIT KERNELS (Tinker et al. 2010) AND CONCENTRATION
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// halo bias b(nu) at peak height nu = delta_c/(sigma(M) D(a))
double hb1nu_cpp(const double nu, const double a);

// multiplicity function f(nu) of the halo mass function
double fnu_cpp(const double nu, const double a);

// halo concentration c(m) (Bhattacharya et al. 2013, Delta = 200 mean)
double conc_cpp(const double m, const double growfac_a);

// d ln nu / d ln M at a = 1 (cached table)
double dlognudlogm_cpp(const double M);

// -----------------------------------------------------------------------------

// integral of b(nu) f(nu) over the tabulated mass range: the 2-halo
// renormalization (table in a)
double bias_norm_cpp(const double a);

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO AND GAS PROFILES
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// Fourier transform of the NFW density profile, normalized to 1 at k = 0
double u_nfw_c_cpp(const double c, const double k, const double m,
                   const double a);

// Fourier transform of the Komatsu-Seljak gas pressure profile (table)
double u_KS_cpp(const double c, const double k, const double rv);

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HOD INTEGRALS (galaxies in lens bin ni)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// galaxy number density (table / direct integral)
double ngal_cpp(const int ni, const double a);

double ngal_nointerp_cpp(const int ni, const double a);

// number-weighted mean galaxy bias (table / direct integral)
double bgal_cpp(const int ni, const double a);

double bgal_nointerp_cpp(const int ni, const double a);

// mean halo mass and satellite fraction of the galaxies (direct integrals)
double mmean_nointerp_cpp(const int ni, const double a);

double fsat_nointerp_cpp(const int ni, const double a);

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO-MODEL POWER SPECTRA (cached 2D tables in (a, ln k))
//
// m = matter, y = Compton-y (thermal SZ), g = galaxies. Scalar overloads
// return one value; array overloads batch over k at one a.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

double p_mm_cpp(const double k, const double a);

arma::Col<double> p_mm_cpp(const arma::Col<double> k, const double a);

// -----------------------------------------------------------------------------

double p_my_cpp(const double k, const double a);

arma::Col<double> p_my_cpp(const arma::Col<double> k, const double a);

// -----------------------------------------------------------------------------

double p_yy_cpp(const double k, const double a);

arma::Col<double> p_yy_cpp(const arma::Col<double> k, const double a);

// -----------------------------------------------------------------------------

double p_gm_cpp(const double k, const double a, const int ni);

arma::Col<double> p_gm_cpp(
    const arma::Col<double> k,
    const double a,
    const int ni
  );

// -----------------------------------------------------------------------------

double p_gg_cpp(const double k, const double a, const int ni, const int nj);

arma::Col<double> p_gg_cpp(
    const arma::Col<double> k,
    const double a,
    const int ni,
    const int nj
  );

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO-MODEL INPUTS FROM cosmo3D.c (the reference values of the unit
// tests' large-scale limits)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// linear growth factor D(a), D(1) = 1
double growfac_cpp(const double a);

// linear matter power spectrum (the 2-halo term of p_mm multiplies it)
double p_lin_cpp(const double k, const double a);

// nonlinear matter power spectrum (the 2-halo term of p_gm/p_gg uses it)
double Pdelta_cpp(const double k, const double a);

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HOD AND GAS PARAMETER SETTERS
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// halo.c's built-in HOD table (Coupon et al. 2012) for lens bin ni
void set_HOD_cpp(const int ni);

// HOD parameters and galaxy concentration factor of lens bin ni
void set_nuisance_hod_cpp(
    const int ni,
    const arma::Col<double> hod,
    const double gc
  );

// gas (Compton-y) parameters nuisance.gas[0..n-1]
void set_nuisance_gas_cpp(const arma::Col<double> gas);

// -----------------------------------------------------------------------------

}  // namespace cosmolike_interface
#endif // HEADER GUARD
