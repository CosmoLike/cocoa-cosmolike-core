#include <math.h>
#include <stdlib.h>

#include "perturbation_cov.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

// SIMD (single instruction, multiple data) applies one operation to two
// doubles at once. The two positions, called lanes, hold independent (K,Q)
// pairs, each with its own complete angle sum; no instruction combines
// numbers from different pairs. v2d stores the two doubles, lane 0 and
// lane 1. SIMDe is a header-only portability library: each simde_mm_* call
// compiles to the matching native instruction (SSE2/FMA on x86, NEON on
// ARM processors such as Apple Silicon), or to plain C where none exists.
//
// A fused multiply-add (FMA) evaluates a*b+c with one rounding when
// supported directly by the processor. This differs from rounding a*b
// first and then adding c; the calls below retain the chosen operations.
// fmadd (a*b+c) and fnmadd (c-a*b) are single-rounding instructions on
// ARM64 NEON and on x86 with FMA enabled. fmsub (a*b-c) is one fused
// instruction only on x86 with FMA: elsewhere SIMDe writes it as a
// multiply followed by a subtraction, and clang on ARM64 (Apple Silicon)
// rounds the two separately.
// Unaligned loads/stores accept addresses that are not multiples of 16
// bytes. They still require two valid adjacent doubles in the array.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Analytic planar average of the EdS kernel F3(k,-k,q).
//
// F3 is the cubic density response to three linear modes. Starting from
// the n=3 recursion of Bernardeau et al. (2002), astro-ph/0112551,
// Eqs. 43-45, symmetrize over all six argument orders. The result is a
// sum of three cyclic terms; each singles out one leg and pairs the other
// two. In F3(k,-k,q) the term that pairs (k,-k) contributes zero: if the
// pair's sum e tends to zero, its quadratic kernels F2 and G2 vanish like
// e^2 while the mode-coupling factors diverge at most like 1/e, so the
// term vanishes like e. The other two cyclic terms remain.
//
// Put k = K(1,0), q = Q(cos theta, sin theta) and r = Q/K. The remaining
// terms are rational functions of cos theta. After partial fractions,
// only planar averages over dtheta/pi are needed:
//
//   <cos theta> = <cos^3 theta> = 0,   <cos^2 theta> = 1/2,
//   <1/(1+r^2-2 r cos theta)> = 1/|1-r^2|.
//
// The absolute value is why the result has one branch for r <= 1 and
// another for r >= 1 (study report 10, Section 2.3-2.4):
//
//   <F3(k,-k,q)> = -r^2 (9-r^2)/126            for r <= 1,
//                = -(21 r^2 - 12 + 7/r^2)/252  for r >= 1.
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
  // r = other/paired and r^2, the only combination the average depends on.
  const double ratio = other/paired;
  const double ratio2 = ratio*ratio;

  // Branch r <= 1. Both branches give -4/63 at r = 1, so which one owns
  // exactly r = 1 does not change the result.
  if (ratio <= 1.0) {
    return -ratio2*(9.0-ratio2)/126.0;
  }
  // Branch r > 1, where |1-r^2| = r^2-1.
  return -(21.0*ratio2-12.0+7.0/ratio2)/252.0;
}


// ---------------------------------------------------------------------------
// Integrate the three perturbative averages used by the halo trispectrum.
//
// Let K=|k|, Q=|q|, mu=cos(theta), s=|k+q| and P_X=P_lin(X,a), at one
// common scale factor. The 3D EdS kernels are averaged in the transverse
// plane, using <F> = integral_0^pi dtheta F / pi. Under Limber every 3D
// wavevector lies in the plane perpendicular to the line of sight
// (k = ell/f_K, with ell the 2D multipole vector), and the covariance of
// two narrow ell bins averages over the angle between them in that plane.
// The integrand depends on cos(theta) only, so the half circle 0..pi gives
// the full-circle average. This is not the 3D solid-angle measure dmu/2.
// In particular <F2(k,q)> = 6/7, not 17/21.
//
// Combining the two F2 terms with the shared internal momentum gives
//
//   G = F2(k+q,-q) P_Q + F2(k+q,-k) P_K.
//
// Each term separately contains ratios such as (s.q)/s^2 that grow without
// bound as s -> 0 near k=-q; their sum stays finite because
// s.k + s.q = s^2. Combine them algebraically before evaluation (study
// report 10, Section 2.4):
//
//   G = -(P_K+P_Q)/28 - mu (K P_Q/Q + Q P_K/K)/2
//       + { (2/7)[(Q+K mu)^2 P_Q + (K+Q mu)^2 P_K]
//           - (Q^2-K^2)(P_Q-P_K)/4 } / s^2.
//
// Near theta = pi, 1+cos(theta) computed from cos(theta) loses its
// relative accuracy, so the caller supplies c = 1+mu = 2 sin^2((pi-theta)/2)
// directly. Then s^2=(K-Q)^2+2 K Q c, Q+K mu=(Q-K)+K c, and K+Q mu =
// Q c-(Q-K). None subtracts two nearly equal squared magnitudes; Q-K
// itself is exact in floating point when K and Q are within a factor of
// two (Sterbenz lemma). On the exact diagonal (K=Q, P_K=P_Q) G approaches
// 13 P_K/14 as c tends to zero.
// Quadrature nodes exclude c=0, so no zero denominator is evaluated.
//
// The Wick contractions reduce to
//
//   AvgP = <P_s>,
//   AvgB = (12/7) P_K P_Q + 2 <P_s G>,
//   AvgT = 12 <F3(k,-k,q)> P_K^2 P_Q
//          +12 <F3(q,-q,k)> P_Q^2 P_K + 8 <P_s G^2>.
//
// In AvgB, 12/7 = 2 <F2(k,q)>. In AvgT, the four choices of the cubic leg
// times its 3! contractions give 24 terms: the cubic leg q or -q gives
// F3(k,-k,q) P_K^2 P_Q (equal by parity) and the leg k or -k gives
// F3(q,-q,k) P_Q^2 P_K, hence 12 each. The 8 combines the two nonzero
// exchange channels, with internal momenta k+q and k-q, each contributing
// 4 P G^2; the planar average makes them equal, because k-q is k+q at the
// angle pi-theta. The exactly zero internal-momentum channel k+(-k)=0
// becomes the super-sample (beat-coupling) term once the survey window is
// kept, so it is reserved for SSC, not evaluated and then repaired. See
// Takada & Hu (2013), arXiv:1302.6994, Section III; the independent
// reference enumerates all Wick diagrams and uses the Bernardeau
// recursion rather than these reduced formulas.
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

  // A planar angular average uses normalized dtheta/pi weights. Verify
  // their normalization separately from the allowed range of each node.
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

  // The stable formulas below divide by K and Q, so neither may vanish.
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

  // Fixing K and Q does not fix the internal mode |k+q|: it also depends
  // on the angle between the two vectors. The covariance needs an average
  // over that angle. All nontrivial angular dependence enters through
  // P_s, P_s*G and P_s*G^2, so accumulate those three integrals together.
  // The remaining F2/F3 contributions are analytic and are added afterward.
  // Each worker treats two (K,Q) pairs. SIMD shares the angle samples but
  // keeps one pair's complete integrals in each lane; adding lanes would
  // mix two different choices of the external wavenumbers.
  // One iteration of this loop handles pairs pair and pair+1; the static
  // schedule gives each thread a contiguous block of such pairs.
  #pragma omp parallel for schedule(static)
  for (int pair=0; pair<npair; pair+=2) {
    // --- 1. PACK TWO INDEPENDENT WAVENUMBER PAIRS ---

    // scalar: K = k[0][j], Q = k[1][j], PK = pk[0][j] and PQ = pk[1][j],
    // for j = pair in lane 0 and j = next in lane 1.

    // An odd final pair is duplicated for safe reads. Only the genuine
    // pair's result is written in the output loop below.
    const int next = pair+1 < npair ? pair+1 : pair;

    // vk = [k[0][pair], k[0][next]] = [K of pair, K of next]. set_pd takes
    // the high lane first: lane 0 gets K for pair and lane 1 gets K for
    // next. A lane is a whole pair, not one of K and Q.
    const v2d vk = simde_mm_set_pd(k[0][next], k[0][pair]);

    // vq = [k[1][pair], k[1][next]]: Q for pair/next in lanes 0/1. The
    // reversed set_pd argument order keeps these Q values matched to the
    // K values above.
    const v2d vq = simde_mm_set_pd(k[1][next], k[1][pair]);

    // vpk = [pk[0][pair], pk[0][next]]: P_K in pair/next order; set_pd
    // assigns its last argument to lane 0. These are powers, with units of
    // length cubed.
    const v2d vpk = simde_mm_set_pd(pk[0][next], pk[0][pair]);

    // vpq = [pk[1][pair], pk[1][next]]: P_Q with pair in lane 0 and next in
    // lane 1, again supplying the high-lane value as the first set_pd
    // argument.
    const v2d vpq = simde_mm_set_pd(pk[1][next], pk[1][pair]);

    // Rows of P_s = P_lin(|k+q|) over the angular nodes, one per pair.
    // restrict promises the compiler that nothing writes to these rows
    // while they are read. Both may point to the same row at an odd
    // endpoint; that is allowed because both only read.
    const double* restrict power0 = ps[pair];
    const double* restrict power1 = ps[next];

    // --- 2. PRECOMPUTE THE ANGLE-INDEPENDENT PARTS OF G ---

    // scalar: for each pair j=pair (lane 0), next (lane 1), where
    // K=k[0][j], Q=k[1][j], PK=pk[0][j] and PQ=pk[1][j]:
    //   diff = Q-K;
    //   diff2 = diff*diff;
    //   twoKQ = 2*(K*Q);
    //   base = (-1.0/28.0)*(PK+PQ);
    //   slope = 0.5*((K*PQ)/Q+(Q*PK)/K);
    //   correction = 0.25*((diff*(Q+K))*(PQ-PK));
    // These pieces of the combined perturbation kernel are independent
    // of angle. Computing them once avoids repeating work at every
    // quadrature node. Each SIMD lane holds all pieces for one pair.

    // vhalf = [0.5, 0.5]: set1 copies 1/2 into both lanes, for the
    // coefficient of cos(theta) in G.
    const v2d vhalf = simde_mm_set1_pd(0.5);

    // vtwo_sevenths = [2/7, 2/7], for G's squared-projection numerator.
    // 2.0/7.0 is one rounded double, the same at every angle.
    const v2d vtwo_sevenths = simde_mm_set1_pd(2.0/7.0);

    // scalar: diff = Q-K; diff2 = diff*diff; twoKQ = 2*(K*Q)

    // diff = Q-K in each pair. Subtracting the magnitudes themselves is
    // exact when they are within a factor of two, so this remains
    // accurate near equal K,Q.
    const v2d vdiff = simde_mm_sub_pd(vq, vk);

    // diff2 = (Q-K)^2 for the stable |k+q|^2 expression below.
    const v2d vdiff2 = simde_mm_mul_pd(vdiff, vdiff);

    // vtwo = [2, 2]: set1 copies 2 to both lanes for the 2*K*Q term in
    // |k+q|^2.
    const v2d vtwo = simde_mm_set1_pd(2.0);

    // K*Q, formed separately within each pair, not across lanes.
    const v2d vkq = simde_mm_mul_pd(vk, vq);

    // twoKQ = 2*(K*Q), the coefficient of c = 1+cos(theta) in |k+q|^2.
    const v2d vtwokq = simde_mm_mul_pd(vtwo, vkq);

    // scalar: base = (-1.0/28.0)*(PK+PQ)

    // vminus_one_28 = [-1/28, -1/28]: set1 copies the constant
    // multiplying P_K+P_Q into both lanes.
    const v2d vminus_one_28 = simde_mm_set1_pd(-1.0/28.0);

    // P_K+P_Q within each pair, for G's constant contribution.
    const v2d vpower_sum = simde_mm_add_pd(vpk, vpq);

    // base = -(P_K+P_Q)/28, formed separately in the two lanes.
    const v2d vbase = simde_mm_mul_pd(vminus_one_28, vpower_sum);

    // scalar: slope = 0.5*((K*PQ)/Q+(Q*PK)/K)
    // The coefficient of -mu is (K*P_Q/Q + Q*P_K/K)/2. Keep its
    // multiplication and division order explicit to preserve rounding.

    // K*P_Q within each pair, the numerator of the first ratio.
    const v2d vk_pq = simde_mm_mul_pd(vk, vpq);

    // (K*P_Q)/Q: that numerator divided by its own Q in each lane.
    const v2d vk_pq_over_q = simde_mm_div_pd(vk_pq, vq);

    // Q*P_K within each pair, the numerator of the second ratio.
    const v2d vq_pk = simde_mm_mul_pd(vq, vpk);

    // (Q*P_K)/K: divided by the matching K in each lane.
    const v2d vq_pk_over_k = simde_mm_div_pd(vq_pk, vk);

    // K*P_Q/Q + Q*P_K/K for each pair; no sum between independent lanes.
    const v2d vratio_sum = simde_mm_add_pd(vk_pq_over_q, vq_pk_over_k);

    // slope = 0.5*ratio_sum, G's -mu coefficient in each lane.
    const v2d vslope = simde_mm_mul_pd(vhalf, vratio_sum);

    // scalar: correction = 0.25*((diff*(Q+K))*(PQ-PK))
    // The remaining correction is (Q^2-K^2)*(P_Q-P_K)/4.

    // P_Q-P_K: the two external powers subtracted within each pair.
    const v2d vpower_diff = simde_mm_sub_pd(vpq, vpk);

    // Q+K in each lane, for the difference-of-squares identity.
    const v2d vsum_kq = simde_mm_add_pd(vq, vk);

    // (Q-K)*(Q+K) = Q^2-K^2 avoids subtracting nearly equal squared
    // magnitudes.
    const v2d vsquare_diff = simde_mm_mul_pd(vdiff, vsum_kq);

    // vquarter = [1/4, 1/4]: set1 copies the common factor into both pair
    // lanes.
    const v2d vquarter = simde_mm_set1_pd(0.25);

    // (Q^2-K^2)*(P_Q-P_K): the squared-magnitude difference times the
    // power difference.
    const v2d vcorrection_product = simde_mm_mul_pd(vsquare_diff, vpower_diff);

    // correction = 0.25*product, separately in each lane.
    const v2d vcorrection = simde_mm_mul_pd(vquarter, vcorrection_product);

    // --- 3. INTEGRATE P_s, P_s*G AND P_s*G^2 OVER ANGLE ---

    // scalar: integrand and updates for one pair j at angle node n,
    // starting from sumP = sumB = sumT = 0:
    //   c = corner[n];
    //   mu = c-1;
    //   s2 = fma(twoKQ, c, diff2);
    //   projectionQ = fma(K, c, diff);
    //   projectionK = Q*c-diff;
    //   numerator = (2.0/7.0)*(PQ*(projectionQ*projectionQ)
    //                          +PK*(projectionK*projectionK))-correction;
    //   G = fma(-mu, slope, base)+numerator/s2;
    //   weightedP = weight[n]*ps[j][n];
    //   sumP += weightedP;
    //   sumB = fma(weightedP, G, sumB);
    //   sumT = fma(weightedP, G*G, sumT);
    // The numerator line is one fused fma(2.0/7.0, ..., -correction) on
    // x86 with FMA; on ARM64 the product is rounded before the subtraction
    // (see fmsub below). These updates integrate three powers of the
    // coupling kernel, 1, G and G^2, weighted by P_s. SIMD follows the
    // same angular order for two pairs; no term multiplies values from
    // different pairs.

    // vsum_p = [0, 0]: both pairs' P_s integrals start at zero (setzero
    // sets every lane to 0.0).
    v2d vsum_p = simde_mm_setzero_pd();

    // vsum_b = [0, 0]: both pairs' P_s*G integrals start at zero.
    v2d vsum_b = simde_mm_setzero_pd();

    // vsum_t = [0, 0]: both pairs' P_s*G^2 integrals start at zero.
    v2d vsum_t = simde_mm_setzero_pd();

    // Nearly opposite vectors with K close to Q give a very small internal
    // magnitude s. Compute s^2 from (K-Q)^2+2*K*Q*(1+cos(theta)) to avoid
    // subtracting large, nearly equal squared magnitudes. The combined G
    // below likewise cancels its large terms algebraically before division.
    // At each angle, its quadrature weight turns P_s, P_s*G and P_s*G^2
    // into contributions to the three averages. SIMD updates both pairs,
    // each in its own lane, visiting the angles in the same fixed order.
    // One iteration is one angle theta_n, shared by the two pairs.
    for (int node=0; node<nangle; node++) {
      // vc = [c, c] with c = corner[node] = 1+cos(theta): both pairs use
      // this same angle, so set1 copies c to both lanes.
      const v2d vc = simde_mm_set1_pd(corner[node]);

      // vmu = [mu, mu] with mu = cos(theta) = c-1, computed in scalar code
      // and copied to both lanes for G's regular term.
      const v2d vmu = simde_mm_set1_pd(corner[node]-1.0);

      // scalar: s2 = fma(twoKQ, c, diff2)
      // s^2 = 2*K*Q*c+(Q-K)^2, separately for each pair. fmadd computes
      // vtwokq*vc + vdiff2, product and addition fused into one
      // native-FMA rounding.
      const v2d vs2 = simde_mm_fmadd_pd(vtwokq, vc, vdiff2);

      // Numerator of the final fraction in G: two squared projections
      // weighted by power, followed by the finite power-difference term.

      // scalar: projectionQ = fma(K, c, diff)
      // Q+K*mu = K*c+(Q-K) per lane. fmadd computes vk*vc + vdiff and
      // retains one rounding on native FMA hardware, even when this
      // projection is very small.
      const v2d vproj_q = simde_mm_fmadd_pd(vk, vc, vdiff);

      // scalar: projectionK = Q*c-diff
      // Q*c in each pair. Keep its rounding separate from subtraction.
      const v2d vqc = simde_mm_mul_pd(vq, vc);

      // K+Q*mu = Q*c-(Q-K), independently in each lane.
      const v2d vproj_k = simde_mm_sub_pd(vqc, vdiff);

      // scalar: PQ*(projectionQ*projectionQ)+PK*(projectionK*projectionK)

      // (Q+K*mu)^2, separately for each pair.
      const v2d vproj_q2 = simde_mm_mul_pd(vproj_q, vproj_q);

      // P_Q*(Q+K*mu)^2: that squared projection weighted by the pair's own
      // P_Q.
      const v2d vterm_q = simde_mm_mul_pd(vpq, vproj_q2);

      // (K+Q*mu)^2, separately for each pair.
      const v2d vproj_k2 = simde_mm_mul_pd(vproj_k, vproj_k);

      // P_K*(K+Q*mu)^2: weighted by the matching P_K.
      const v2d vterm_k = simde_mm_mul_pd(vpk, vproj_k2);

      // projection_sum: the two power-weighted squared projections added
      // within each pair.
      const v2d vprojection_sum = simde_mm_add_pd(vterm_q, vterm_k);

      // scalar: numerator = (2.0/7.0)*projection_sum-correction
      // fmsub computes vtwo_sevenths*vprojection_sum - vcorrection, product
      // minus its third argument, in each lane. On x86 with FMA this is
      // one fused rounding, fma(2.0/7.0, projection_sum, -correction). On
      // ARM64 SIMDe's portable form is a multiply then a subtraction, which
      // clang rounds separately: the product is rounded first.
      const v2d vnumerator = simde_mm_fmsub_pd(vtwo_sevenths,
                                             vprojection_sum, vcorrection);

      // scalar: G = fma(-mu, slope, base)+numerator/s2

      // regular = base-mu*slope. fnmadd negates the product vmu*vslope
      // before adding its third argument, vbase, fused into one native-FMA
      // rounding per lane.
      const v2d vregular = simde_mm_fnmadd_pd(vmu, vslope, vbase);

      // numerator/s^2: each numerator divided by the matching internal
      // magnitude squared, which is positive because every c > 0.
      const v2d vfraction = simde_mm_div_pd(vnumerator, vs2);

      // G = regular + fraction, completed separately for each pair.
      const v2d vg = simde_mm_add_pd(vregular, vfraction);

      // scalar: weightedP = weight[n]*ps[j][n]

      // One read of P_s serves all three averages. The two vector lanes
      // are different K,Q pairs; no horizontal reduction is needed.
      // vps = [power0[node], power1[node]] = [P_s(pair), P_s(next)]:
      // set_pd takes lane 1 first. The internal magnitude |k+q| differs
      // between pairs, so the two lanes read different rows.
      const v2d vps = simde_mm_set_pd(power1[node], power0[node]);

      // vw = [weight[node], weight[node]]: set1 copies this node's
      // dtheta/pi weight to both pair lanes.
      const v2d vw = simde_mm_set1_pd(weight[node]);

      // weightedP = weight*P_s, the common angle weight times each pair's
      // own P_s.
      const v2d vweighted_p = simde_mm_mul_pd(vw, vps);

      // G^2 within each lane, for the connected four-point term.
      const v2d vg2 = simde_mm_mul_pd(vg, vg);

      // scalar: sumP += weightedP; sumB = fma(weightedP, G, sumB);
      //         sumT = fma(weightedP, G*G, sumT)

      // sumP += weightedP: the weighted P_s added to its own pair's AvgP
      // sum, lane by lane.
      vsum_p = simde_mm_add_pd(vsum_p, vweighted_p);

      // sumB = fma(weightedP, G, sumB): weighted P_s times G added to each
      // pair's bispectrum integral. fmadd computes vweighted_p*vg + vsum_b
      // with one native-FMA rounding in each lane.
      vsum_b = simde_mm_fmadd_pd(vweighted_p, vg, vsum_b);

      // sumT = fma(weightedP, G*G, sumT): weighted P_s times G^2 added to
      // each pair's trispectrum integral, again one fused native-FMA
      // rounding per lane without mixing pairs.
      vsum_t = simde_mm_fmadd_pd(vweighted_p, vg2, vsum_t);
    }

    // --- 4. COMPLETE THE ANALYTIC PARTS AND WRITE VALID PAIRS ---

    // scalar: sums[r][lane] = the angular integral r (P_s, P_s*G or
    // P_s*G^2) of pair+lane, then the AvgP/AvgB/AvgT formulas.
    // Add the angle-independent F2 and F3 contributions after integration.
    // A duplicated final lane supplies an odd pair count without padding.
    double sums[3][2];

    // sums[0][0..1] = AvgP subtotals for pair/next: storeu writes lane 0
    // to sums[0][0] and lane 1 to sums[0][1]. storeu does not require
    // special vector alignment for this ordinary stack array.
    simde_mm_storeu_pd(sums[0], vsum_p);

    // sums[1][0..1] = the two P_s*G integrals, in the same lane order.
    // storeu needs two valid doubles, but no vector-aligned address.
    simde_mm_storeu_pd(sums[1], vsum_b);

    // sums[2][0..1] = the two P_s*G^2 integrals. storeu likewise accepts
    // the ordinary array address without extra alignment.
    simde_mm_storeu_pd(sums[2], vsum_t);

    // For each valid pair lane, add the angle-independent F2/F3 pieces
    // and write all three averages. An odd final pair leaves a duplicate
    // lane whose result is deliberately not written.
    for (int lane=0; lane<2; lane++) {
      const int index = pair+lane;
      if (index < npair) {
        // p and q here are P_K and P_Q, not the wavenumber magnitudes.
        // first = <F3(k,-k,q)> = F3bar(K,Q) and second = <F3(q,-q,k)> =
        // F3bar(Q,K): the paired magnitude is paired_f3_cov's first
        // argument.
        const double p = pk[0][index];
        const double q = pk[1][index];
        const double first = paired_f3_cov(k[0][index], k[1][index]);
        const double second = paired_f3_cov(k[1][index], k[0][index]);

        // Restore the angle-independent terms and their Wick multiplicities
        // from the scalar AvgP/AvgB/AvgT equations in the function header:
        // (12/7) p q = 2 <F2(k,q)> P_K P_Q, and 12 per paired F3 term.
        average[0][index] = sums[0][lane];
        average[1][index] = (12.0/7.0)*p*q+2.0*sums[1][lane];
        average[2][index] = 12.0*first*p*p*q+12.0*second*q*q*p
                            +8.0*sums[2][lane];
      }
    }
  }
}
