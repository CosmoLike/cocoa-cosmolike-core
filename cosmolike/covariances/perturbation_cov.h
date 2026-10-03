#ifndef COSMOLIKE_PERTURBATION_COV_H
#define COSMOLIKE_PERTURBATION_COV_H

#ifdef __cplusplus
extern "C" {
#endif

// Planar tree-level averages for supplied (K,Q) pairs and linear power.
// k and pk have two rows, K/Q and P(K)/P(Q). At each angular node supply
// corner=1+cos(theta), evaluated as 2*sin((pi-theta)/2)^2 near pi, and
// ps=P(sqrt((K-Q)^2+2*K*Q*corner)). Weights integrate dtheta/pi.
// average has three rows: <P>, <B_tree>, <T_tree>, with powers L^3,L^6,L^9.
// Zero-internal-momentum SSC channels are excluded analytically.
void tree_averages_cov(
    const int npair,                 // number of supplied K,Q combinations
    const int nangle,                // common angular quadrature size
    const double* const* k,         // [2][npair], positive wavenumbers
    const double* const* pk,        // [2][npair], linear power at K and Q
    const double* corner,           // [nangle], finite 0 < 1+cos(theta) <= 2
    const double* weight,           // [nangle], positive normalized measure
    const double* const* ps,        // [npair][nangle], linear power at |k+q|
    double* const* average           // [3][npair], overwritten
  );

#ifdef __cplusplus
}
#endif
#endif
