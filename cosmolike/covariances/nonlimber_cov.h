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
    const struct radial_cov* radial,
    const double amin,
    const int nwindow,
    const int include_ia,
    const int lmax,
    const int nchi,
    const double chi_min,
    const int nell,
    const double* ell,
    double* const* spectra
  );

#ifdef __cplusplus
}
#endif
#endif
