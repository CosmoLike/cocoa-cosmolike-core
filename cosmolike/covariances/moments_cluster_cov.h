#ifndef COSMOLIKE_MOMENTS_CLUSTER_COV_H
#define COSMOLIKE_MOMENTS_CLUSTER_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// Selected halo mass integrals on caller-supplied quadrature nodes.
// weight[a][selection][mass] includes dlnM, dn/dlnM and a single factor
// of the probability of membership in an observed category, in L^-3.
// profile[a][k][mass] is (M/rho)*u(k|M), in L^3; bias[a][mass] is the
// linear halo bias.
// Arrays are finite; weight >= 0. Output rows do not overlap any input.
// No mass-function choice, normalization or angular projection is added.
//
// Flatten a,selection as row=a*nselection+selection in these outputs:
// density[2][row] = integral dn S, integral dn S b, both in L^-3.
// single[2][row][k] = J01, J11, both dimensionless.
// pair[3][row][p] = J02(K,Q), J03(K,K,Q), J03(K,Q,Q), in L^3,L^6,L^6.
// p follows (0,0),(0,1),...,(1,1),... in the upper triangle of k.
// J_beta_mu = integral dn S b_beta product(profile), b_0=1, b_1=b.
// A halo carries one label, so its indicator obeys I^2 = I: a same-halo
// moment contains S once, and two exclusive bins share no halo.
// Multiplying two selection probabilities is not the moment of a shared
// halo assigned to exclusive observed bins. See the C physics derivation.
void moments_cluster_cov(
    const int na,                      // independent radial states
    const int nselection,              // observed selection categories
    const int nk,                      // profile samples per state
    const int nmass,                   // common mass-node count
    const double* const* const* weight, // selected mass measures
    const double* const* bias,         // linear halo bias at mass nodes
    const double* const* const* profile, // mass-weighted Fourier profiles
    double* const* density,             // two abundance moments
    double** const* single,             // two one-profile moments
    double** const* pair                // three two/three-profile moments
  );

#ifdef __cplusplus
}
#endif
#endif
