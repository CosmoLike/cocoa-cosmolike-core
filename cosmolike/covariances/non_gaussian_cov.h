#ifndef COSMOLIKE_NON_GAUSSIAN_COV_H
#define COSMOLIKE_NON_GAUSSIAN_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// Assemble the halo power and its dimensional background response,
//   P_halo = I11^2 P_lin + I02,
//   D_halo = (growth - dilation*slope) I11^2 P_lin + I12.
// inputs[6][point]: P_lin, P_target, I11, I02(k,k), I12(k,k), log slope.
// output[2][point]: P_halo and D=dP_target/d(delta_b) when fractional=1;
// fractional=0 instead returns the absolute halo-model response.
// The caller explicitly selects both coefficients and the slope's spectrum:
// (47/21, 1/3) with the slope of I11^2 P_lin is the published isotropic
// halo response; (17/7, 1/2) with the slope of P_lin is the distinct
// planar (flat-sky projected) tree-level construction.
void halo_response_cov(
    const int npoint,                // number of independent (k,a) points
    const double growth_coefficient, // constant part of the two-halo response
    const double dilation_coefficient, // coefficient of the supplied log slope
    const int fractional,           // 1 rescales D_halo/P_halo by P_target
    const double* const* inputs,    // six input rows
    double* const* output            // two output rows
  );

// Assemble the five halo contributions to the angle-averaged cNG trispectrum.
// pk[2][point], i11[2][point] refer to K,Q. moments uses the five roles in
// halo_cov.h; tree uses the three roles in perturbation_cov.h. All refer
// to the same field, scale factor and (K,Q). No survey factors are added.
// terms[5][point] = 1h, 2h(1+3), 2h(2+2), 3h, 4h, each in length^9:
// I04, 2[P_K I11(K) I13(K,Q,Q) + P_Q I11(Q) I13(K,K,Q)], 2 I12^2 AvgP,
// 4 I12 I11(K) I11(Q) AvgB and [I11(K) I11(Q)]^2 AvgT.
void halo_trispectrum_cov(
    const int npoint,                // independent K,Q,a combinations
    const double* const* pk,        // two linear-power rows
    const double* const* i11,       // two one-profile moment rows
    const double* const* moments,   // five pair-moment rows
    const double* const* tree,      // planar P/B/T averages
    double* const* terms             // five separated halo contributions
  );

#ifdef __cplusplus
}
#endif
#endif
