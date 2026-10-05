#ifndef COSMOLIKE_SSC_COV_H
#define COSMOLIKE_SSC_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// Long-mode Limber approximation for a supplied, raw angular mask spectrum.
// power[p][L] is P_lin((L+1/2)/f_K[p], a[p]), in length^3. The mask has
// C_0 = area_sr^2/(4 pi). sigma2[p] has units of LENGTH, not variance of a
// finite-width shell. No radial integration weight is included here.
void ssc_mask_variance_cov(
    const int nnode,                    // radial samples
    const int nmask,                    // mask multipoles L=0,...,nmask-1
    const double area_sr,              // integral of the mask over the sky
    const double* mask_cl,             // raw mask spectrum [nmask]
    const double* distance,            // positive f_K [nnode]
    const double* const* power,        // linear power [nnode][nmask]
    double* sigma2                     // background strength [nnode]
  );

// Functional response Phi_i(chi), where delta x_i = integral dchi
// Phi_i(chi) delta_b(chi). A row can be one pair and one multipole.
// The caller chooses the matter response and the observed-mean convention.
// Inputs are finite; output is separate, and rows may have padded strides.
void ssc_shell_response_cov(
    const int nrow,                     // pair/multipole combinations
    const int nnode,                    // common radial samples
    const double* distance,            // positive f_K [nnode]
    const double* signal,              // full angular C_AB for each row
    const double* const* pair_window,  // W_A W_B [nrow][nnode]
    const double* const* mean_window,  // response of the two catalog means
    const double* const* power_response, // dP/d(delta_b), length^3
    double* const* response             // Phi [nrow][nnode], length^-1
  );

#ifdef __cplusplus
}
#endif
#endif
