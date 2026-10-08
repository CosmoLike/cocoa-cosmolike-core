#ifndef COSMOLIKE_SSC_COV_H
#define COSMOLIKE_SSC_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// Strength of the survey-averaged long-wavelength density at each radial
// node, in the long-mode Limber approximation, from a supplied raw angular
// mask spectrum:
//
//   sigma2[p] = sum_L (2L+1) mask_cl[L] power[p][L]
//               / (area_sr^2 distance[p]^2),
//
// with power[p][L] = P_lin((L+1/2)/f_K[p], a[p]) in length^3. A raw mask
// spectrum has C_0 = area_sr^2/(4 pi); any other normalization stops the
// run with an error.
// sigma2[p] has units of length: it multiplies the radial Dirac delta in
// <delta_b(chi) delta_b(chi')> = delta_D(chi-chi') sigma_b^2(chi), so it is
// not the dimensionless variance of a finite-width shell. No radial
// integration weight dchi is included here; the SSC integral applies it.
void ssc_mask_variance_cov(
    const int nnode,                    // radial samples
    const int nmask,                    // mask multipoles L=0,...,nmask-1
    const double area_sr,              // integral of the mask, steradians
    const double* mask_cl,             // raw mask spectrum [nmask]
    const double* distance,            // positive f_K [nnode]
    const double* const* power,        // linear power [nnode][nmask]
    double* sigma2                     // background strength [nnode]
  );

// Functional response Phi[row](chi), defined by the first-order change of
// one measured spectrum, delta C_row = integral dchi Phi[row](chi)
// delta_b(chi). A row is one field pair AB at one multipole ell:
//
//   Phi[row][j] = pair_window[row][j] power_response[row][j]/distance[j]^2
//                 - mean_window[row][j] signal[row].
//
// pair_window is W_A W_B (length^-2); power_response is the dimensional
// D = dP/d(delta_b) at k=(ell+1/2)/f_K (length^3); mean_window is U_A+U_B
// (length^-1, zero for a field not divided by a catalog mean); signal is
// the dimensionless C_AB(ell) of the mean model whose survey-mean
// normalization the estimator adopts. Phi has units length^-1.
// The caller chooses the matter response and the observed-mean convention.
// Inputs are finite; output is separate, and rows may have padded strides.
void ssc_shell_response_cov(
    const int nrow,                     // pair/multipole combinations
    const int nnode,                    // common radial samples
    const double* distance,            // positive f_K [nnode]
    const double* signal,              // mean-model C_AB [nrow]
    const double* const* pair_window,  // W_A W_B [nrow][nnode]
    const double* const* mean_window,  // U_A+U_B [nrow][nnode]
    const double* const* power_response, // dP/d(delta_b), length^3
    double* const* response             // Phi [nrow][nnode], length^-1
  );

#ifdef __cplusplus
}
#endif
#endif
