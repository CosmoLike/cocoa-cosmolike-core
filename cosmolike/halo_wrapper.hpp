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
// (NFW), the HOD galaxy counts, and the power spectra assembled from
// them. Its functions are
// plain C. This layer makes them callable from Python, so the unit
// tests (projects/roman_real/tests/test_halo.py) and notebooks can
// evaluate them one number at a time.
//
// One Python call travels
//
//   ci.p_gm(k, a, ni)                     (Python)
//     -> m.def("p_gm", ...)               (project interface.cpp)
//     -> p_gm_cpp(k, a, ni)               (this layer: checks the input,
//                                          loops over arrays)
//     -> p_gm(k, a, ni)                   (halo.c: reads a cached table)
//     -> on first use, or after a cache key changed: the table is
//        refilled (the halo-model mass integrals at every (a, ln k) node)
//
// Names: each function below is the C function's name plus _cpp, and
// its Python name is the C name itself (as for the sigma2 and
// scale-cut bindings): halo.c p_gm -> p_gm_cpp -> ci.p_gm.
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
// Dimensionless: u_nfw_c, conc, hb1nu, fnu, dlognudlogm,
// bias_norm, bgal, ia_f_red_central, ia_window_2h.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Warm-up: halo.c's spectrum builders call halo_warmup before their
// OpenMP loops, so every lazily built table is built on one thread.
// Python calls arrive one at a time, so any lazy build a wrapper
// triggers is serial already.
//
// Threading: the array overloads loop serially over their inputs. The
// parallelism lives inside halo.c, in the table builders a first call
// triggers.
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
double conc_cpp(const double m, const double a);

// d ln nu / d ln M at a = 1 (cached table)
double dlognudlogm_cpp(const double M, const double a);

// -----------------------------------------------------------------------------

// integral of b(nu) f(nu) over the tabulated mass range; 1 - bias_norm is
// the HMx additive 2-halo correction of a halo-model I11 sum
// (future_port_unfinished/halo_pmm.c; table in a)
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

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HOD INTEGRALS (galaxies in lens bin ni)
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// galaxy number density (table)
double ngal_cpp(const int ni, const double a);

// number-weighted mean galaxy bias (table)
double bgal_cpp(const int ni, const double a);

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO-MODEL POWER SPECTRA (cached 2D tables in (a, ln k))
//
// m = matter, g = galaxies. Scalar overloads
// return one value; array overloads batch over k at one a.
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

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

// linear matter power spectrum
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

// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// HALO-MODEL INTRINSIC ALIGNMENT (Fortuna et al. 2021; cached tables over
// the source a range, 0 outside it)
//
// ia_p1h_dI is signed with a_1h: the C_l cores of cosmo2D.c subtract it,
// P_dI^phys = -[f_rc C_1 P_delta f_2h + P_dI^1h]. Scalar overloads return
// one value; array overloads batch over k at one a. The IA wrappers
// abort for a outside (0, 1).
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------
// ---------------------------------------------------------------------------

// red-central fraction f_rc(a) of the IA (source) sample (table in a)
double ia_f_red_central_cpp(const double a);

// -----------------------------------------------------------------------------

// f_2h(k) = exp[-(k/k_2h)^2], the window of the NLA 2-halo term
double ia_window_2h_cpp(const double k);

arma::Col<double> ia_window_2h_cpp(const arma::Col<double> k);

// -----------------------------------------------------------------------------

// P_dI^1h = a_1h(a) f_1h(k) S_dI(k, a), signed with a_1h
double ia_p1h_dI_cpp(const double k, const double a);

arma::Col<double> ia_p1h_dI_cpp(const arma::Col<double> k, const double a);

// -----------------------------------------------------------------------------

// P_II^1h = a_1h(a)^2 f_1h(k) S_II(k, a), >= 0
double ia_p1h_II_cpp(const double k, const double a);

arma::Col<double> ia_p1h_II_cpp(const arma::Col<double> k, const double a);

// halo-model IA parameters: a_1h, eta_1h, z_pivot; red-fraction sigmoids;
// IA-population HOD (halo_wrapper.cpp header)
void set_nuisance_ia_halo_cpp(
    const arma::Col<double> ia_halo,
    const arma::Col<double> ia_red,
    const arma::Col<double> ia_hod
  );

// -----------------------------------------------------------------------------

}  // namespace cosmolike_interface
#endif // HEADER GUARD
