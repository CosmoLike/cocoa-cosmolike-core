#ifndef COSMOLIKE_MASK_COV_H
#define COSMOLIKE_MASK_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// Integrate the two-position survey footprint over each angular annulus:
//   pair_area[b] = 8 pi^2 Delta_x,b sum_L C_L^W scalar_kernel[b][L],
// in sr^2, with Delta_x,b = cos(theta_low,b) - cos(theta_high,b). A
// full-sky mask gives 8 pi^2 Delta_x,b; mask_cov.c has the derivation.
// The raw mask has C_0=area_sr^2/(4 pi); retain its monopole and dipole.
// scalar_kernel is the bin-averaged w operator from operators_cov.c,
// including (2L+1)/(4 pi). No density or ellipticity factor is inserted:
// pass pair_area[b] unchanged to gaussian_noise_pair_cov, which supplies
// the catalog and shear-component factors.
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
