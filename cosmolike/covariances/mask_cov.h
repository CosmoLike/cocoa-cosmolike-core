#ifndef COSMOLIKE_MASK_COV_H
#define COSMOLIKE_MASK_COV_H

#ifdef __cplusplus
extern "C" {
#endif

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
