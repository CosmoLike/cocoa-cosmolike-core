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
void nonlimber_spectra_cov(
    const struct radial_cov* radial, // existing common quadrature snapshot
    const double amin,              // far boundary enclosing both catalogs
    const int nwindow,               // cumulative lensing-efficiency samples
    const int include_ia,            // signed linear alignment contribution
    const int lmax,                  // integer multipoles 2..lmax
    const int nchi,                  // log-distance samples, 2^n+1
    const double chi_min,            // positive near distance in c/H0
    double* const* exact,            // [triangular pair][lmax-1]
    double* const* matched           // same shape, matched Limber spectrum
  );

// Add exact-minus-matched only to galaxy-containing rows at integer ell.
// Multipoles outside [2,lmax] and all source-source rows remain unchanged.
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
