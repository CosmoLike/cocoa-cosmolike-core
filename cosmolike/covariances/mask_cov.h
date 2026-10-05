#ifndef COSMOLIKE_MASK_COV_H
#define COSMOLIKE_MASK_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// Integrate the two-position survey footprint over each angular annulus.
// The raw mask has C_0=area_sr^2/(4 pi); retain its monopole and dipole.
// scalar_kernel is the bin-averaged w operator from operators_cov.c,
// including (2L+1)/(4 pi). No density or ellipticity factor is inserted.
// Inputs stay read-only. The caller owns the overwritten output array.
void mask_pair_area_cov(
    const int nbin,                     // number of angular bins
    const int nmask,                    // mask multipoles 0..nmask-1
    const double area_sr,               // integral of the common mask, sr
    const double* edges_rad,            // [nbin+1], increasing in [0, pi]
    const double* mask_cl,              // [nmask], raw mask power spectrum
    const double* const* scalar_kernel, // [nbin][nmask], w-bin operator
    double* pair_area                   // [nbin], ordered-pair area in sr^2
  );

#ifdef __cplusplus
}
#endif
#endif
