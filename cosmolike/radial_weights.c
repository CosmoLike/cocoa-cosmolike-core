#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "log.c/src/log.h"

#include "basics.h"
#include "bias.h"
#include "cosmo3D.h"
#include "redshift_spline.h"
#include "structs.h"

// ---------------------------------------------------------------------------
// Lensing convergence projection kernel for source tomographic bin nz.
//
//   W_kappa(a, nz) = 1.5 * Omega_m * (H0/c)^2 * fK(chi(a))/a * g_tomo(a, nz)
//
// where g_tomo is the cumulative lensing efficiency of the source n(z)
// (see redshift_spline.c). Distances are in c/H0 units, so (H0/c)^2 = 1
// and the code prefactor reduces to 1.5*Omega_m.
//
// Parameters:
//   a  - scale factor, 0 < a < 1
//   fK - comoving angular diameter distance f_K(chi(a)) in c/H0 units
//   nz - source tomographic bin index (0 .. shear_nbin-1)
//
// Returns:
//   convergence kernel value at (a, nz).
// ---------------------------------------------------------------------------
double W_kappa(const double a, const double fK, const int nz) 
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true");
    exit(1);
  }
  if (nz < 0 || nz > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input ni = %d", nz);
    exit(1);
  }
  return (1.5*cosmology.Omega_m*fK/a) * g_tomo(a, nz);
}

// ---------------------------------------------------------------------------
// Squared-efficiency convergence kernel for source tomographic bin nz.
//
//   W2_kappa(a, nz) = (1.5 * Omega_m * fK(chi(a))/a)^2 * g2_tomo(a, nz)
//
// g2_tomo is the integral of the squared lensing efficiency over the
// source n(z), not the square of the integral, so W2_kappa differs from
// W_kappa^2; it is used where two lensing factors share the same
// line-of-sight integration variable. Distances in c/H0 units.
//
// Parameters:
//   a  - scale factor, 0 < a < 1
//   fK - comoving angular diameter distance f_K(chi(a)) in c/H0 units
//   nz - source tomographic bin index (0 .. shear_nbin-1)
//
// Returns:
//   squared-kernel weight at (a, nz).
// ---------------------------------------------------------------------------
double W2_kappa(double a, double fK, int nz) 
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true");
    exit(1);
  }
  if (nz < 0 || nz > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input ni = %d", nz);
    exit(1);
  }
  const double tmp = (1.5*cosmology.Omega_m*fK/a);
  return tmp*tmp*g2_tomo(a, nz);
}

// ---------------------------------------------------------------------------
// Magnification convergence kernel for lens tomographic bin nz.
//
//   W_mag(a, nz) = 1.5 * Omega_m * fK(chi(a))/a * g_lens(a, nz)
//
// Same convergence prefactor as W_kappa, with the lensing efficiency
// g_lens integrated over the lens-sample n(z): the magnified galaxies are
// the lenses of bin nz. The magnification bias amplitude (gbmag) is
// applied by the callers, not here. Distances in c/H0 units.
//
// Parameters:
//   a  - scale factor, 0 < a < 1
//   fK - comoving angular diameter distance f_K(chi(a)) in c/H0 units
//   nz - lens tomographic bin index (0 .. clustering_nbin-1)
//
// Returns:
//   magnification kernel value at (a, nz).
// ---------------------------------------------------------------------------
double W_mag(double a, double fK, int nz) 
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true");
    exit(1);
  }
  if (nz < 0 || nz > redshift.clustering_nbin - 1) {
    log_fatal("invalid bin input ni = %d", nz);
    exit(1);
  }
  return (1.5 * cosmology.Omega_m * fK / a) * g_lens(a, nz);
}

// ---------------------------------------------------------------------------
// Radial galaxy density kernel for lens tomographic bin ni.
//
//   W_gal(a, ni) = n_i(z(a)) * dz/dchi = nz_lens_photoz(1/a - 1, ni) * H/H0
//
// The hoverh0 factor converts the normalized redshift distribution into a
// distribution in comoving distance (chi in c/H0 units gives dz/dchi =
// H(z)/H0). Galaxy bias is applied by the callers, not here.
//
// Parameters:
//   a       - scale factor, 0 < a < 1
//   ni      - lens tomographic bin index (0 .. clustering_nbin-1)
//   hoverh0 - H(a)/H0, supplied by the caller (avoids recomputation)
//
// Returns:
//   n_i(chi(a)), the lens density per unit comoving distance.
// ---------------------------------------------------------------------------
double W_gal(double a, int ni, double hoverh0) {
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni); exit(1);
  }
  const double z = 1. / a - 1;
  return nz_lens_photoz(z, ni) * hoverh0;
}

// ---------------------------------------------------------------------------
// Radial galaxy density kernel for source tomographic bin ni.
//
//   W_source(a, ni) = n_i(z(a)) * dz/dchi
//                   = nz_source_photoz(1/a - 1, ni) * H/H0
//
// Same construction as W_gal with the source photo-z distribution; it
// enters the intrinsic alignment and source-clustering terms.
//
// Parameters:
//   a       - scale factor, 0 < a < 1
//   ni      - source tomographic bin index (0 .. shear_nbin-1)
//   hoverh0 - H(a)/H0, supplied by the caller (avoids recomputation)
//
// Returns:
//   n_i(chi(a)), the source density per unit comoving distance.
// ---------------------------------------------------------------------------
double W_source(double a, int ni, double hoverh0)
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  if (ni < 0 || ni > redshift.shear_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni); exit(1);
  }
  const double z = 1.0 / a - 1.0;
  return nz_source_photoz(z, ni) * hoverh0;
}

// ---------------------------------------------------------------------------
// Logarithmic growth rate at scale factor a.
//
//   f_rsd(a) = f_growth(z = 1/a - 1) = dln D / dln a
//
// Thin wrapper converting a to redshift for the RSD kernel below.
//
// Parameters:
//   a - scale factor, 0 < a < 1
//
// Returns:
//   linear growth rate f(z(a)) (dimensionless).
// ---------------------------------------------------------------------------
double f_rsd(double a) 
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  const double z = 1.0 / a - 1.0;
  return f_growth(z);
}

// ---------------------------------------------------------------------------
// Linear redshift-space distortion kernel for lens tomographic bin ni.
//
// Limber RSD correction to the galaxy density kernel, combining the
// growth-weighted density f * n_i * H/H0 at two line-of-sight points:
//
//   W_RSD = (1 + 8*l)/(2*l + 1)^2 * n_i(z(a0)) * (H/H0)(a0) * f_rsd(a0)
//         - 4/(2*l + 3) * sqrt((2*l + 1)/(2*l + 3))
//              * n_i(z(a1)) * (H/H0)(a1) * f_rsd(a1)
//
// with n_i = nz_lens_photoz. The Limber callers pass l = ell + 0.5 (a
// half-integer) and the scale factors of the two Bessel-peak points,
// a0 = a(chi) at chi = l/k and a1 = a(chi') at chi' = (l + 1)/k.
//
// Parameters:
//   l  - multipole argument of the RSD prefactors (callers pass ell + 0.5)
//   a0 - scale factor of the first evaluation point, 0 < a0 < 1
//   a1 - scale factor of the second evaluation point, 0 < a1 < 1
//   ni - lens tomographic bin index; the guard admits -1, but
//        nz_lens_photoz only accepts 0 .. clustering_nbin-1
//
// Returns:
//   RSD kernel value.
// ---------------------------------------------------------------------------
double W_RSD(double l, double a0, double a1, int ni) 
{
  if (!(a0>0) || !(a0<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  if (!(a1>0) || !(a1<1)) {
    log_fatal("a>0 and a<1 not true"); exit(1);
  }
  if (ni < 0 || ni > redshift.clustering_nbin - 1) {
    log_fatal("invalid bin input ni = %d", ni); exit(1);
  }
  double w = (1 + 8. * l) / ((2. * l + 1.) * (2. * l + 1.)) *
    nz_lens_photoz(1. / a0 - 1., ni) * hoverh0(a0) * f_rsd(a0);
  
  w -= 4. / (2 * l + 3.) * sqrt((2 * l + 1.) / (2 * l + 3.)) *
    nz_lens_photoz(1./a1 - 1., ni) * hoverh0(a1) * f_rsd(a1);
  
  return w;
}

// ---------------------------------------------------------------------------
// CMB lensing convergence kernel.
//
//   W_k(a) = 1.5 * Omega_m * fK(chi(a))/a * g_cmb(a)
//
// with g_cmb(a) = f_K(chi_cmb - chi(a)) / f_K(chi_cmb), the lensing
// efficiency of the CMB source plane (see redshift_spline.c). Distances
// in c/H0 units, so the (H0/c)^2 prefactor is unity.
//
// Parameters:
//   a  - scale factor, 0 < a < 1
//   fK - comoving angular diameter distance f_K(chi(a)) in c/H0 units
//
// Returns:
//   CMB convergence kernel value at a.
// ---------------------------------------------------------------------------
double W_k(double a, double fK) 
{
  if (!(a>0) || !(a<1)) {
    log_fatal("a>0 and a<1 not true");
    exit(1);
  }
  return (1.5*cosmology.Omega_m*fK/a)*g_cmb(a);
}

// ---------------------------------------------------------------------------
// Compton-y projection kernel (thermal Sunyaev-Zeldovich).
//
//   W_y(a) = sigma_Th / (m_e c^2 * a^2)      (Eq. D9 of 2005.00009)
//
// evaluated in code units: sigma_Th converted from Mpc^2 to (c/H0)^2 and
// the electron rest energy from MeV to G*(M_solar/h)^2/(c/H0) (the body
// comments give the conversion factors). No range check is applied to a.
//
// Parameters:
//   a - scale factor
//
// Returns:
//   kernel value; units [c/H0]^2 / [G*(M_solar/h)^2/(c/H0)].
// ---------------------------------------------------------------------------
double W_y(double a) // efficiency weight function for Compton-y
{ // sigma_Th /(m_e*c^2) / a^2 , see Eq.D9 of 2005.00009.
  const double real_coverH0 = cosmology.coverH0 / cosmology.h0; // unit Mpc
  const double sigma_Th = 7.012e-74 / (real_coverH0*real_coverH0); // from Mpc^2 to (c/H0)^2
  const double E_e = 0.511*cosmology.h0*5.6131e-38;  // from MeV to [G(M_solar/h)^2/(c/H0)]
  return sigma_Th/(E_e*a*a); //  dim = [comoving L]^2 / [Energy], 
                             // units = [c/H0]^2 / [G(M_solar/h)^2/(c/H0)]
}
