#ifndef COSMOLIKE_ASSEMBLY_COV_H
#define COSMOLIKE_ASSEMBLY_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// Both interfaces call these C assemblers. Rows are caller-owned and may
// have padding; the last physical axis is always contiguous. Output rows
// do not overlap inputs. Wrappers check shapes and finite values first.
// See assembly_cov.c for the physical sums and parallel work ownership.
// Covariance rows and columns are indexed observable*nbin + bin.
//
// Gaussian: Cov[(r,i),(s,j)] = sum_ell K_(p_r,i)(ell) G_rs(ell)
// K_(p_s,j)(ell), plus exact pair-count pure noise on equal real-space
// bins. realspace = 1 selects that real-space split; 0 means Fourier
// bands, which keep all noise in the harmonic sum. fsky = area_sr/(4 pi).
void gaussian_matrix_cov(
    const int nell,                       // consecutive integer multipoles
    const int nfield,                     // lens plus source catalogs
    const int nobs,                       // measured catalog pairs
    const int nbin,                       // angular or Fourier bins
    const int* rows,                      // flat [nobs,3] (probe,A,B)
    const double* const* spectra,         // [nfield*nfield,nell], signal
    const double* const* b_spectra,       // optional BB rows, NULL for E only
    const double* noise,                  // [nfield], white noise powers
    const double* const* kernels,         // [4*nbin,nell], or [nbin,nell]
    const int ell_min,                    // first input multipole
    const double area_sr,                 // common footprint area
    const double* pair_area,              // [nbin] sr^2, NULL for Fourier
    const int realspace,                  // exact real-space pure noise
    double* const* output                 // [nobs*nbin,nobs*nbin]
  );

// Connected: C[(r,i),(s,j)] = sum_node W_r W_s T[p_r*nbin+i][p_s*nbin+j]
// measure, with W_r = W_A*W_B of observable r and T the angularly projected
// matter trispectrum, read only where p_r <= p_s (i <= j if p_r = p_s).
void connected_matrix_cov(
    const int nobs,                       // measured catalog pairs
    const int nbin,                       // bins for each observable
    const int nnode,                      // radial integration nodes
    const int* probes,                    // [nobs], statistic IDs 0..3
    const double* const* pair_window,     // [nobs,nnode], W_A*W_B
    const double* const* projected,       // [(4*nbin)^2,nnode], matter T
    const double* measure,                // [nnode], dchi/(area*f_K^6)
    double* const* output                 // [nobs*nbin,nobs*nbin]
  );

#ifdef __cplusplus
}
#endif
#endif
