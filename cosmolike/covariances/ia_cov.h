#ifndef COSMOLIKE_IA_COV_H
#define COSMOLIKE_IA_COV_H
#include "spectra_cov.h"
#ifdef __cplusplus
extern "C" {
#endif

// Add the TATT terms beyond NLA to supplied E spectra. Fill B spectra
// from zero. Both outputs use triangular all-field pairs as in spectra_cov.
void tatt_spectra_cov(
    const struct radial_cov* radial, // NLA windows and common quadrature
    const int nell,                 // requested multipole count
    const double* ell,              // multipoles >= 1
    double* const* ee,              // NLA input, complete TATT E output
    double* const* bb               // TATT B output, zero for galaxy legs
  );
#ifdef __cplusplus
}
#endif
#endif
