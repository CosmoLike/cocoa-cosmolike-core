#ifndef COSMOLIKE_OPERATORS_COV_H
#define COSMOLIKE_OPERATORS_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// Stateless geometry builders. The caller owns each padded output row.
// Real-space rows use enum probe_cov: xi+, xi-, gamma_t, w, then bin.
// See operators_cov.c for spin conventions, quadrature and mode removal.
void realspace_operator_cov(
    const int nbin,            // number of angular bins
    const double* edges_rad,   // [nbin+1], strictly increasing in [0, pi]
    const int ell_max,         // highest included integer multipole, >= 2
    const int nquad,           // tabulated GL rule per bin; caller refines
    double* const* kernel      // [4*nbin][ell_max+1], overwritten
  );

void bandpower_operator_cov(
    const int nband,           // number of Fourier bands
    const int ell_min,         // first integer multipole of output columns
    const int nell,            // number of consecutive integer multipoles
    const int* first,          // [nband], first included multipole
    const int* last,           // [nband], last included multipole
    double* const* kernel      // [nband][nell], overwritten
  );

#ifdef __cplusplus
}
#endif
#endif
