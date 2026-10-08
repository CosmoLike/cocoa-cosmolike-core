#ifndef COSMOLIKE_NONLIMBER_COV_H
#define COSMOLIKE_NONLIMBER_COV_H

#include "spectra_cov.h"

#ifdef __cplusplus
extern "C" {
#endif

// Return the exact separable-linear spectrum and its matching Limber
// approximation for all pairs, including unmeasured crosses. Subtract
// the latter from the former before adding a nonlinear Limber spectrum.
// Field order and units match spectra_cov.h. No shot/shape noise is added.
//
// Both describe the same linear field D(a) delta(k,1): for l = 2..lmax,
//   exact[p][l-2]   = (2/pi) integral dlnk k^3 P_lin(k,1) F_A F_B,
//   matched[p][l-2] = the Limber projection of D(a)^2 P_lin(k,1),
// with F the spherical-Bessel transfer of each field of pair p.
// amin, nwindow and include_ia must equal the values used to build
// radial, so that both terms see the same windows. Requires flat
// geometry and massless neutrinos; call outside OpenMP regions.
void nonlimber_spectra_cov(
    const struct radial_cov* radial, // existing common quadrature snapshot
    const double amin,              // far boundary enclosing both catalogs
    const int nwindow,               // cumulative lensing-efficiency samples
    const int include_ia,            // signed linear alignment contribution
    const int lmax,                  // integer multipoles 2..lmax
    const int nchi,                  // log-distance samples, 2^n+1 >= 65
    const double chi_min,            // positive near distance in c/H0
    double* const* exact,            // [triangular pair][lmax-1]
    double* const* matched           // same shape, matched Limber spectrum
  );

// Add exact-minus-matched only to galaxy-containing rows at integer ell.
// Multipoles outside [2,lmax] and all source-source rows remain unchanged.
// The result is the hybrid spectrum
//   nonlinear Limber + exact separable linear - matched linear Limber.
// Every ell inside [2,lmax] must be an integer (the C++ entry points
// reject fractional values there); amin, nwindow and include_ia must
// match radial, as for nonlimber_spectra_cov.
void apply_nonlimber_cov(
    const struct radial_cov* radial, // common Limber rule and windows
    const double amin,              // far boundary in scale factor
    const int nwindow,               // efficiency interpolation samples
    const int include_ia,            // signed linear alignment
    const int lmax,                  // last corrected integer multipole
    const int nchi,                  // logarithmic radial samples
    const double chi_min,            // near distance in c/H0
    const int nell,                  // requested output multipoles
    const double* ell,               // [nell], same order as spectra
    double* const* spectra           // [triangular pair][nell], in/out
  );

#ifdef __cplusplus
}
#endif
#endif
