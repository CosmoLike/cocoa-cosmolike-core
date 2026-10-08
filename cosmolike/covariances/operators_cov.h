#ifndef COSMOLIKE_OPERATORS_COV_H
#define COSMOLIKE_OPERATORS_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// Stateless geometry builders. The caller owns each padded output row.
// Real-space rows use enum probe_cov: xi+, xi-, gamma_t, w, then bin.
// See operators_cov.c for spin conventions, quadrature and mode removal.
//
// kernel[probe*nbin+bin][ell] is (2 ell+1)/(4 pi) times the area average
// over the bin of the probe's Wigner kernel d^ell (d22, d2-2, d20, d00),
// so a binned correlation is sum_ell kernel[row][ell] C_ell. Spin rows are
// zero at ell = 0, 1. A wide bin is split into panels: nquad counts the
// Gauss-Legendre nodes of each panel (64, 96, 128, 256, 512 or 1024), not
// of the whole bin.
void realspace_operator_cov(
    const int nbin,            // number of angular bins
    const double* edges_rad,   // [nbin+1], strictly increasing in [0, pi]
    const int ell_max,         // highest included integer multipole, >= 2
    const int nquad,           // tabulated GL rule per bin; caller refines
    double* const* kernel      // [4*nbin][ell_max+1], overwritten
  );

// Give each integer ell inside a band its normalized (2 ell+1) weight,
// (2 ell+1)/N_band with N_band = (last-first+1)(last+first+1); all
// columns outside that band are zero. Bounds are inclusive absolute
// multipoles, while output column zero corresponds to ell_min.
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
