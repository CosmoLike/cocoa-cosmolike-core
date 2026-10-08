#ifndef COSMOLIKE_IA_COV_H
#define COSMOLIKE_IA_COV_H
#include "spectra_cov.h"
#ifdef __cplusplus
extern "C" {
#endif

// Add the TATT terms beyond NLA to supplied E spectra. Fill B spectra
// from zero. Both outputs use triangular all-field pairs as in spectra_cov.
// ee must come from limber_spectra_cov on the same snapshot, built with
// include_ia = 1 and without RSD. B modes enter the real-space estimators
// with opposite signs: xi+ measures EE+BB and xi- measures EE-BB
// (assembly_cov.c applies the signs). Call outside an OpenMP region.
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
