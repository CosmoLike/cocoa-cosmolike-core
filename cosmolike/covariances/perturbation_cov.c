#include <math.h>
#include <stdlib.h>

#include "perturbation_cov.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

typedef simde__m128d v2d; // two independent K,Q pairs

// ---------------------------------------------------------------------------
// Analytic planar average of the EdS kernel F3(k,-k,q).
//
// F3 is the cubic density response to three linear modes. Starting from
// the n=3 recursion of Bernardeau et al. (2002), astro-ph/0112551,
// Eqs. 43-45, symmetrize over all six argument orders. The inner pair
// (k,-k) contributes zero: its quadratic kernel vanishes faster than
// the mode-coupling pole diverges. The other two cyclic terms remain.
// Their integral over dtheta/pi can be evaluated analytically, giving
// the two branches below (study report 10, Section 2.3-2.4).
//
// paired is |k|, other is |q|. Exchanging them changes which mode occurs
// twice in F3; this function is therefore not symmetric in its arguments.
// Both branches meet at -4/63 when the magnitudes coincide. They approach
// -r^2/14 for small r and -r^2/12 for large r, where r=other/paired.
// The caller validates positive finite wavenumbers. No power is included.
// ---------------------------------------------------------------------------
static double paired_f3_cov(
    const double paired, // magnitude of k and -k
    const double other   // magnitude of the third mode
  )
{
  const double ratio = other/paired;
  const double ratio2 = ratio*ratio;
  if (ratio <= 1.0) {
    return -ratio2*(9.0-ratio2)/126.0;
  }
  return -(21.0*ratio2-12.0+7.0/ratio2)/252.0;
}


// ---------------------------------------------------------------------------
// Integrate the three perturbative averages used by the halo trispectrum.
//
// Let K=|k|, Q=|q|, mu=cos(theta), s=|k+q| and P_X=P_lin(X,a), at one
// common scale factor. The 3D EdS kernels are averaged in the transverse
// PLANE, using <F> = integral_0^pi dtheta F / pi. This is not the 3D
// solid-angle measure dmu/2. In particular <F2(k,q)> = 6/7, not 17/21.
//
// Combining the two F2 terms with the shared internal momentum gives
//
//   G = F2(k+q,-q) P_Q + F2(k+q,-k) P_K.
//
// Each term separately contains large ratios near k=-q. Combine them
// algebraically BEFORE evaluation (study report 10, Section 2.4):
//
//   G = -(P_K+P_Q)/28 - mu (K P_Q/Q + Q P_K/K)/2
//       + { (2/7)[(Q+K mu)^2 P_Q + (K+Q mu)^2 P_K]
//           - (Q^2-K^2)(P_Q-P_K)/4 } / s^2.
//
// To preserve the small angle near pi, use c=1+mu as supplied below.
// Then s^2=(K-Q)^2+2 K Q c, Q+K mu=(Q-K)+K c, and analogously for the
// other term. None subtracts two nearly equal squared magnitudes. On
// the exact diagonal G approaches 13 P_K/14 as c tends to zero.
// Quadrature nodes exclude c=0, so no zero denominator is evaluated.
//
// The Wick contractions reduce to
//
//   AvgP = <P_s>,
//   AvgB = (12/7) P_K P_Q + 2 <P_s G>,
//   AvgT = 12 <F3(k,-k,q)> P_K^2 P_Q
//          +12 <F3(q,-q,k)> P_Q^2 P_K + 8 <P_s G^2>.
//
// For AvgT, the factors 12 come from four choices of the cubic leg and
// its 3! contractions; the 8 combines the two nonzero exchange channels.
// The exactly zero internal-momentum channel is reserved for SSC, not
// evaluated and then repaired. See Takada & Hu (2013), arXiv:1302.6994,
// Section III; the independent reference enumerates all Wick diagrams
// and uses the Bernardeau recursion rather than these reduced formulas.
//
// Inputs and units:
//   k[0/1] - positive K,Q, both in the same inverse-length unit
//   pk[0/1] - finite, nonnegative P_K,P_Q, in length^3
//   corner - 1+cos(theta); compute as 2 sin^2((pi-theta)/2) near pi
//   weight - positive dtheta/pi weights, summing to one
//   ps - finite, nonnegative P_s, from the same P_lin and scale factor
//   average - caller-owned rows AvgP,AvgB,AvgT, in length^3,^6,^9
// npair and nangle describe the physical sizes; row strides may be padded.
// Power samples, angle nodes and halo moments are prepared by the caller.
// A graded angular rule is needed when P_s is sharply peaked near pi;
// this integrator does not silently choose or increase its resolution.
//
// No allocation, table lookup or static cache occurs here. Each worker
// owns two pair outputs. Each SIMD lane accumulates one pair in increasing
// angular-node order, so thread count does not reorder any sum. Inputs
// cannot overlap output; output rows must not overlap one another.
// ---------------------------------------------------------------------------
void tree_averages_cov(
    const int npair,                 // number of K,Q pairs
    const int nangle,                // number of angular nodes
    const double* const* k,         // two rows of wavenumbers
    const double* const* pk,        // power at those wavenumbers
    const double* corner,           // stable 1+cos(theta)
    const double* weight,           // dtheta/pi quadrature weights
    const double* const* ps,        // P at each internal momentum
    double* const* average           // three output averages
  )
{
  if (npair < 1
      || nangle < 1) {
    log_fatal("tree_averages_cov needs positive pair and angle counts");
    exit(1);
  }
  double normalization = 0.0;
  for (int node=0; node<nangle; node++) {
    if (!isfinite(corner[node])
        || corner[node] <= 0.0
        || corner[node] > 2.0
        || !isfinite(weight[node])
        || weight[node] <= 0.0) {
      log_fatal("tree_averages_cov: invalid angle or weight at node %d",
                node);
      exit(1);
    }
    normalization += weight[node];
  }
  if (fabs(normalization-1.0) > 1.e-10) {
    log_fatal("tree_averages_cov needs weights summing to 1, got %g",
              normalization);
    exit(1);
  }
  for (int pair=0; pair<npair; pair++) {
    for (int role=0; role<2; role++) {
      if (!isfinite(k[role][pair])
          || k[role][pair] <= 0.0) {
        log_fatal("tree_averages_cov: invalid k in role %d, pair %d",
                  role, pair);
        exit(1);
      }
    }
  }

  #pragma omp parallel for schedule(static)
  for (int pair=0; pair<npair; pair+=2) {
    const int next = pair+1 < npair ? pair+1 : pair;
    const v2d vk = simde_mm_set_pd(k[0][next], k[0][pair]);
    const v2d vq = simde_mm_set_pd(k[1][next], k[1][pair]);
    const v2d vpk = simde_mm_set_pd(pk[0][next], pk[0][pair]);
    const v2d vpq = simde_mm_set_pd(pk[1][next], pk[1][pair]);
    const double* restrict power0 = ps[pair];
    const double* restrict power1 = ps[next];

    // Quantities independent of theta are computed once per pair.
    const v2d vhalf = simde_mm_set1_pd(0.5);
    const v2d vtwo_sevenths = simde_mm_set1_pd(2.0/7.0);
    const v2d vdiff = simde_mm_sub_pd(vq, vk);
    const v2d vdiff2 = simde_mm_mul_pd(vdiff, vdiff);
    const v2d vtwokq = simde_mm_mul_pd(simde_mm_set1_pd(2.0),
                                     simde_mm_mul_pd(vk, vq));
    const v2d vbase = simde_mm_mul_pd(simde_mm_set1_pd(-1.0/28.0),
                                    simde_mm_add_pd(vpk, vpq));
    const v2d vslope = simde_mm_mul_pd(vhalf, simde_mm_add_pd(
        simde_mm_div_pd(simde_mm_mul_pd(vk, vpq), vq),
        simde_mm_div_pd(simde_mm_mul_pd(vq, vpk), vk)));
    const v2d vpower_diff = simde_mm_sub_pd(vpq, vpk);
    const v2d vsquare_diff = simde_mm_mul_pd(vdiff,
                                           simde_mm_add_pd(vq, vk));
    const v2d vcorrection = simde_mm_mul_pd(simde_mm_set1_pd(0.25),
        simde_mm_mul_pd(vsquare_diff, vpower_diff));
    v2d vsum_p = simde_mm_setzero_pd();
    v2d vsum_b = simde_mm_setzero_pd();
    v2d vsum_t = simde_mm_setzero_pd();

    for (int node=0; node<nangle; node++) {
      const v2d vc = simde_mm_set1_pd(corner[node]);
      const v2d vmu = simde_mm_set1_pd(corner[node]-1.0);
      const v2d vs2 = simde_mm_fmadd_pd(vtwokq, vc, vdiff2);

      // Numerator of the final fraction in G: two squared projections
      // weighted by power, followed by the finite power-difference term.
      const v2d vproj_q = simde_mm_fmadd_pd(vk, vc, vdiff);
      const v2d vproj_k = simde_mm_sub_pd(simde_mm_mul_pd(vq, vc), vdiff);
      const v2d vterm_q = simde_mm_mul_pd(vpq,
          simde_mm_mul_pd(vproj_q, vproj_q));
      const v2d vterm_k = simde_mm_mul_pd(vpk,
          simde_mm_mul_pd(vproj_k, vproj_k));
      const v2d vnumerator = simde_mm_fmsub_pd(vtwo_sevenths,
          simde_mm_add_pd(vterm_q, vterm_k), vcorrection);
      const v2d vregular = simde_mm_fnmadd_pd(vmu, vslope, vbase);
      const v2d vg = simde_mm_add_pd(vregular,
                                    simde_mm_div_pd(vnumerator, vs2));

      // One read of P_s serves all three averages. The two vector lanes
      // are different K,Q pairs; no horizontal reduction is needed.
      const v2d vps = simde_mm_set_pd(power1[node], power0[node]);
      const v2d vw = simde_mm_set1_pd(weight[node]);
      const v2d vweighted_p = simde_mm_mul_pd(vw, vps);
      const v2d vg2 = simde_mm_mul_pd(vg, vg);
      vsum_p = simde_mm_add_pd(vsum_p, vweighted_p);
      vsum_b = simde_mm_fmadd_pd(vweighted_p, vg, vsum_b);
      vsum_t = simde_mm_fmadd_pd(vweighted_p, vg2, vsum_t);
    }

    // Add the angle-independent F2 and F3 contributions after integration.
    // A duplicated final lane supplies an odd pair count without padding.
    double sums[3][2];
    simde_mm_storeu_pd(sums[0], vsum_p);
    simde_mm_storeu_pd(sums[1], vsum_b);
    simde_mm_storeu_pd(sums[2], vsum_t);
    for (int lane=0; lane<2; lane++) {
      const int index = pair+lane;
      if (index < npair) {
        const double p = pk[0][index];
        const double q = pk[1][index];
        const double first = paired_f3_cov(k[0][index], k[1][index]);
        const double second = paired_f3_cov(k[1][index], k[0][index]);
        average[0][index] = sums[0][lane];
        average[1][index] = (12.0/7.0)*p*q+2.0*sums[1][lane];
        average[2][index] = 12.0*first*p*p*q+12.0*second*q*q*p
                            +8.0*sums[2][lane];
      }
    }
  }
}
