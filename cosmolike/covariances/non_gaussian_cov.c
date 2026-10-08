#include <math.h>
#include <stdlib.h>

#include "non_gaussian_cov.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

// SIMD (single instruction, multiple data) applies one operation to
// several numbers at once. A v2d holds two doubles, in positions called
// lanes 0 and 1. Every operation below acts separately on two (k,a)
// points (the response) or two (K,Q,a) configurations (the trispectrum):
// lane 0 holds the whole calculation for one of them, lane 1 for the
// other, and the two lanes are never combined.
//
// SIMDe turns each simde_mm_* call into the vector instructions of the
// build machine (SSE2 or AVX on x86, NEON on arm64). set_pd(x1, x0) puts
// x0 in lane 0 and x1 in lane 1: the high lane is written first.
// set1_pd(x) copies one number into both lanes. A fused multiply-add
// (FMA) evaluates a*b+c with the product kept exact and a single rounding
// of the result. SIMDe emits one fused instruction on arm64 and on x86
// builds with FMA enabled (the optimized -march=native build on FMA
// hardware); its portable fallback for other x86 builds rounds a*b first
// and then adds c. The two can differ in the last bit.
// Unaligned stores (storeu) accept addresses that are not multiples of
// 16 bytes. They still require two valid adjacent doubles in the array.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Assemble a dimensional halo-model response with explicit model choices.
//
// The two-halo power is P_2h = I11^2 P_lin; the one-halo power is I02.
// A background density mode delta_b, a fluctuation longer than the
// survey, changes their amplitudes. In the adopted halo approximation
// the one-halo derivative is I12, while the two-halo term contains growth
// and a dilation (rescaling of wavenumber):
//
//   P_halo = P_2h + I02,
//   D_halo = (growth_coefficient - dilation_coefficient * slope) P_2h
//            + I12.
//
// I02 and I12 are pair moments at K = Q = k. The supplied slope is
// dlnP_X/dlnk, without the k^3 factor. Published isotropic halo response:
// growth=47/21, dilation=1/3, P_X=P_2h. This is Takada & Hu (2013),
// arXiv:1302.6994v3, corrected Eq. 44 (in the note added after the
// acknowledgments), derived in Li, Hu & Takada (2014), arXiv:1401.0385,
// Section II C:
// 68/21 - (1/3) dln(k^3 P_2h)/dlnk = 47/21 - (1/3) dlnP_2h/dlnk.
//
// PHYSICAL DERIVATION & LOGIC FLOW (separate universe, linear regime)
//   1. Growth: structure grows faster inside the long mode; the local
//      linear power gains a factor 1 + (26/21) delta_b.
//   2. Reference density: contrasts are measured against the global
//      mean density, which multiplies the power by 1 + 2 delta_b.
//   3. Dilation: the region contracts by delta_b/3 in each dimension, so
//      a local wavenumber appears at a different global k. Relabelling k
//      leaves the dimensionless k^3 P unchanged; for the two-halo power
//      this adds -(1/3) dln(k^3 P_2h)/dlnk = -1 - (1/3) dlnP_2h/dlnk.
//   4. Sum: 26/21 + 2 - 1 = 47/21 multiplies P_2h, 1/3 multiplies its
//      slope. One halo: the number density of halos of mass M changes by
//      the fraction b(M) delta_b, so dI02/d(delta_b) = I12; the Eulerian
//      bias b already contains the dilation of the halo positions.
//
// A planar squeezed-tree construction instead uses growth=17/7,
// dilation=1/2 and P_X=P_lin. At tree level, the response of P_lin(k) to
// a long mode whose direction makes cosine mu with k is
// dP_lin/d(delta_b) = [R1 + RK (mu^2 - 1/3)] P_lin, with isotropic part
// R1 = 47/21 - n/3, tidal part RK = 8/7 - n and n = dlnP_lin/dlnk. A 3D
// average over directions, <mu^2> = 1/3, leaves R1. Under flat-sky Limber
// projection both modes lie in the plane of the sky, where
// <mu^2> = <cos^2 theta> = 1/2, giving R1 + RK/6 = 17/7 - n/2.
// In the linear limit (I11 -> 1, I12 -> 0) the planar halo response is
// this R1+RK/6 with tree-level responses, as in Barreira, Krause &
// Schmidt (2018), arXiv:1711.07467, Sections 2 and 4.1. This is not a
// calibrated nonlinear tidal response. These choices are deliberately
// explicit inputs; the code does not present the planar form as a paper
// erratum or silently replace the published two-halo slope.
//
// If fractional=1, preserve the halo model's fractional response but use
// a supplied target power (for example Halofit): D=(D_halo/P_halo) P_target.
// This replacement is a modeling choice, not an identity of perturbation
// theory. If fractional=0, return D_halo. Galaxy survey-mean subtraction
// belongs later in ssc_shell_response_cov, not in this matter response.
//
// Inputs are six rows: P_lin, P_target, I11, I02, I12, slope. All must
// refer to the same density field and (k,a). Power and D have length^3;
// I11, slope and the coefficients are dimensionless. P_halo must be
// positive. output contains P_halo and D; it must not overlap inputs.
// No table lookup, allocation, cache, noise or survey window is added.
// Workers own pairs of points; an odd final lane is read but not stored.
// ---------------------------------------------------------------------------
void halo_response_cov(
    const int npoint,                // independent k,a points
    const double growth_coefficient, // constant response coefficient
    const double dilation_coefficient, // logarithmic-slope coefficient
    const int fractional,           // rescale fractional halo response
    const double* const* inputs,    // six physical input rows
    double* const* output            // halo power and dimensional response
  )
{
  if (npoint < 1
      || !isfinite(growth_coefficient)
      || !isfinite(dilation_coefficient)
      || (fractional != 0
          && fractional != 1)) {
    log_fatal("halo_response_cov needs points, finite coefficients, "
              "and fractional = 0 or 1");
    exit(1);
  }
  for (int point=0; point<npoint; point++) {
    const double p2h = inputs[2][point]*inputs[2][point]*inputs[0][point];
    const double phalo = p2h+inputs[3][point];
    if (!isfinite(phalo)
        || phalo <= 0.0) {
      log_fatal("halo_response_cov needs positive halo power at point %d",
                point);
      exit(1);
    }
  }

  // --- 1. COEFFICIENTS SHARED BY EVERY WAVENUMBER ---

  // The response prescription separates a change in clustering strength
  // from a shift in physical length scale. Its two coefficients apply to
  // every point; the power and its slope carry the dependence on k and a.
  // Giving both lanes the same coefficients lets the following loop apply
  // that prescription to two different points at once.

  // set1_pd copies growth_coefficient (47/21 in the published response)
  // to both lanes, so each point receives the same response prescription
  // without mixing their power spectra.
  const v2d vgrowth = simde_mm_set1_pd(growth_coefficient);

  // set1_pd copies dilation_coefficient (1/3 in the published response)
  // to both lanes; each lane will multiply it by its own logarithmic
  // power-spectrum slope.
  const v2d vdilation = simde_mm_set1_pd(dilation_coefficient);

  // A long-wavelength overdensity changes both the number of halos and
  // their clustering. I12 supplies the one-halo change; growth and the
  // rescaling of k supply the two-halo change. Their sum is D_halo.
  // When requested, D_halo/P_halo transfers the fractional change to the
  // supplied target spectrum. Each worker evaluates two independent (k,a)
  // points: SIMD applies the same formula in both lanes, retaining each
  // point's own power, slope and response. There is no sum between points.
  #pragma omp parallel for schedule(static)
  for (int point=0; point<npoint; point+=2) {
    // An odd final point is repeated in lane 1 to keep reads valid.
    // The output loop below discards that duplicate lane.
    const int next = point+1 < npoint ? point+1 : point;
    v2d values[6];

    // scalar: the same calculation at one point, j = point (lane 0) or
    // j = next (lane 1):
    //   p2h = (inputs[2][j]*inputs[2][j])*inputs[0][j];
    //   phalo = p2h+inputs[3][j];
    //   factor = fma(-dilation_coefficient, inputs[5][j],
    //                growth_coefficient);
    //   response = fma(factor, p2h, inputs[4][j]);
    //   if (fractional) response = (response/phalo)*inputs[1][j];
    //   output[0][j] = phalo;
    //   output[1][j] = response;
    // This is a response of one power spectrum at one (k,a) point.
    // The SIMD code below carries out two such responses side by side.

    // --- 2. LOAD THE SIX PHYSICAL INPUTS AT BOTH POINTS ---

    // A response needs both the unperturbed power and how it changes.
    // P_lin, I11 and I02 build the halo power; I12 and the supplied slope
    // describe its response. P_target is used only for fractional rescaling.
    // Each iteration selects one quantity and copies its values at the two
    // points into separate lanes, keeping all six quantities aligned.
    for (int role=0; role<6; role++) {
      // set_pd takes lane 1 first: lane 0 gets inputs[role][point],
      // lane 1 gets inputs[role][next]. Roles 0..5 are P_lin, P_target,
      // I11, I02(k,k), I12(k,k) and the slope dlnP_X/dlnk.
      values[role] = simde_mm_set_pd(inputs[role][next], inputs[role][point]);
    }

    // --- 3. P_halo = I11^2 P_lin + I02 ---

    // Two density factors can belong to the same halo or to separate halos.
    // I02 includes their common halo profile and abundance in the first
    // case. In the second, P_lin correlates the halo positions, with one
    // bias-weighted profile factor I11 for each halo. Adding both cases
    // gives the power whose fractional change is needed below.

    // mul_pd squares I11 lane by lane, values[2]*values[2]: I11^2 at
    // point in lane 0 and at next in lane 1; no cross-point product.
    const v2d vi11_squared = simde_mm_mul_pd(values[2], values[2]);

    // mul_pd multiplies each I11^2 by its own P_lin (values[0]) to obtain
    // the two-halo power P_2h = (I11*I11)*P_lin in each lane.
    const v2d vp2h = simde_mm_mul_pd(vi11_squared, values[0]);

    // add_pd adds the one-halo power I02(k,k) (values[3]) to P_2h in each
    // lane: P_halo = P_2h + I02.
    const v2d vhalo = simde_mm_add_pd(vp2h, values[3]);

    // --- 4. D_halo = (growth - dilation*slope) P_2h + I12 ---

    // D_halo is the derivative of power with respect to a background
    // overdensity, evaluated at zero overdensity. The slope term accounts
    // for shifting k along a nonconstant spectrum; the constant term
    // collects faster growth, the reference density and the k^3 part of
    // the dilation. I12 adds the response of the one-halo contribution.
    // The optional rescaling keeps D_halo/P_halo while replacing the power
    // amplitude by P_target. This is a model choice, not another halo term.

    // fnmadd(a,b,c) means -(a*b)+c: in each lane, coefficient =
    // fma(-dilation_coefficient, slope, growth_coefficient), that is
    // growth - dilation*slope with the product kept exact and one rounding
    // (fused instruction; see the file header).
    const v2d vcoefficient = simde_mm_fnmadd_pd(vdilation, values[5], vgrowth);

    // fmadd: response = fma(coefficient, P_2h, I12(k,k)) separately at
    // both points, coefficient*P_2h+I12 with one rounding (fused).
    v2d vresponse = simde_mm_fmadd_pd(vcoefficient, vp2h, values[4]);

    if (fractional) {
      // div_pd divides each D_halo by its own P_halo, lane by lane: the
      // dimensionless fractional response dlnP/d(delta_b), not a ratio
      // between the two points.
      const v2d vfraction = simde_mm_div_pd(vresponse, vhalo);

      // mul_pd multiplies by P_target (values[1]) at the matching point
      // to restore length^3: D = (D_halo/P_halo)*P_target.
      vresponse = simde_mm_mul_pd(vfraction, values[1]);
    }

    // --- 5. WRITE THE TWO PHYSICAL OUTPUT ROWS ---

    // Return power and response separately so the caller can inspect their
    // ratio and form the survey's SSC response. The two SIMD lanes describe
    // different points, so they become adjacent entries rather than a sum.

    double result[2][2];

    // storeu writes the halo powers: lane 0 to result[0][0], P_halo at
    // point, and lane 1 to result[0][1], P_halo at next. It accepts this
    // ordinary stack array without special vector alignment.
    simde_mm_storeu_pd(result[0], vhalo);

    // storeu writes the responses D in the same point order:
    // result[1][0] at point, result[1][1] at next. It requires two valid
    // doubles, but no vector-aligned address.
    simde_mm_storeu_pd(result[1], vresponse);

    // Row 0 of output receives P_halo and row 1 the response. Lane 1 is
    // written only when next is a genuine point, discarding a duplicate.
    for (int role=0; role<2; role++) {
      output[role][point] = result[role][0];
      if (point+1 < npoint) {
        output[role][next] = result[role][1];
      }
    }
  }
}


// ---------------------------------------------------------------------------
// Assemble the five connected halo contributions at each (K,Q,a).
//
// Here K=|k| and Q=|q| are wavenumber magnitudes; a is the scale factor.
// A density "leg" means one of the four Fourier factors delta(k),
// delta(-k), delta(q), delta(-q) in the power-spectrum covariance.
// Four density legs can occupy one, two, three or four halos. Two halos
// have two different partitions: one leg plus three, or two plus two.
// For the covariance parallelogram (k,-k,q,-q), averaging over the angle
// between k and q in the plane of the sky (dtheta/pi, as flat-sky Limber
// projection requires) gives
//
//   T_1h  = I04(K,K,Q,Q),
//   T_13  = 2 [P_K I11(K) I13(K,Q,Q) + P_Q I11(Q) I13(K,K,Q)],
//   T_22  = 2 I12(K,Q)^2 AvgP,
//   T_3h  = 4 I12(K,Q) I11(K) I11(Q) AvgB,
//   T_4h  = [I11(K) I11(Q)]^2 AvgT.
//
// P_K and P_Q are linear powers at K and Q. AvgP, AvgB and AvgT are the
// angle averages of the linear power at |k+q|, the tree bispectrum and
// the tree trispectrum that correlate the positions of separate halos.
//
// In T_13 the isolated leg can be k, -k, q or -q: four choices. The
// first two have the same magnitudes, as do the last two. Thus each
// displayed term has coefficient two, not one. This follows directly
// from the four set partitions ("+3 perm.") of T^2h_13 in Takada & Hu
// (2013), Eqs. 28-29, arXiv:1302.6994. CosmoCov's tri_2h_13_cov keeps
// one copy of each product, half of this term.
// T_22: of the three ways to split the legs into two pairs, (k,-k)(q,-q)
// exchanges zero momentum, a super-sample (SSC) channel excluded here;
// (k,q)(-k,-q) and (k,-q)(-k,q) exchange |k+q| and |k-q|, whose angle
// averages are equal, hence the factor two. T_3h: of the six choices of
// the two legs that share a halo, (k,-k) and (q,-q) again exchange zero
// momentum and belong to SSC; the four others pair one K leg with one Q
// leg, hence the factor four. tree_averages_cov supplies the linear
// power, bispectrum and trispectrum averages with the exact
// zero-internal-momentum SSC channels already excluded.
//
// The moments follow halo_cov.h: rows I02, I12, I13(K,Q,Q), I13(K,K,Q),
// I04. I02 is not used here; keeping the same row layout avoids a second
// moment-table convention. pk and i11 have K and Q rows. tree has AvgP,
// AvgB, AvgT rows. All inputs are finite and use the same density field,
// redshift, wavenumbers and length unit. Every output has dimension L^9.
//
// terms stores these five contributions separately. The caller can sum
// them and apply the projection integral dchi W_A W_B W_C W_D T
// / (area*f_K^6). No field windows, spin factors, shot-noise trispectrum
// or survey geometry enter this routine. A tree/halo approximation need
// not be a positive-semidefinite covariance at every node; never clip its
// eigenvalues or replace a negative contribution with zero.
//
// All arrays are caller-owned. Output rows are disjoint from one another
// and all inputs. No cache, allocation or cross-thread sum is used.
// SIMD lanes 0 and 1 hold two different (K,Q,a) configurations, point and
// next; an odd final configuration is read twice and its copy discarded.
// ---------------------------------------------------------------------------
void halo_trispectrum_cov(
    const int npoint,                // independent K,Q,a combinations
    const double* const* pk,        // P_lin(K), P_lin(Q)
    const double* const* i11,       // I11(K), I11(Q)
    const double* const* moments,   // five pair moments
    const double* const* tree,      // three planar tree averages
    double* const* terms             // five halo contributions
  )
{
  if (npoint < 1) {
    log_fatal("halo_trispectrum_cov needs at least one point");
    exit(1);
  }

  // set1_pd copies the multiplicity 2.0 to both configuration lanes. It
  // counts two different things below: the two signs of the isolated leg
  // in T_13 and the two nonzero exchange channels in T_22.
  const v2d vtwo = simde_mm_set1_pd(2.0);

  // set1_pd copies the multiplicity 4.0 to both configuration lanes: the
  // four ways for one K leg and one Q leg to share the two-leg halo of
  // T_3h.
  const v2d vfour = simde_mm_set1_pd(4.0);

  // The four density factors of a connected four-point function can live
  // in one, two, three or four halos. The two-halo case splits again into
  // 1+3 and 2+2 factors, giving the five terms assembled below. Mass moments
  // describe factors in the same halo; the supplied power and tree averages
  // connect different halos. Each worker combines them for two (K,Q,a)
  // points. SIMD keeps one complete point per lane, and each term has its
  // own output row so its contribution can be examined before summation.
  #pragma omp parallel for schedule(static)
  for (int point=0; point<npoint; point+=2) {
    // Lane 0 owns point; lane 1 owns next. Repeat an odd final point
    // only for safe reads, and discard its second result when writing.
    const int next = point+1 < npoint ? point+1 : point;
    v2d vp[2];
    v2d vi[2];
    v2d vm[5];
    v2d vt[3];

    // scalar: one configuration, j = point (lane 0) or j = next (lane 1):
    //   product = i11[0][j]*i11[1][j];
    //   left13 = (pk[0][j]*i11[0][j])*moments[2][j];
    //   right13 = (pk[1][j]*i11[1][j])*moments[3][j];
    //   terms[0][j] = moments[4][j];
    //   terms[1][j] = 2*(left13+right13);
    //   terms[2][j] = (2*(moments[1][j]*moments[1][j]))*tree[0][j];
    //   terms[3][j] = (4*(moments[1][j]*product))*tree[1][j];
    //   terms[4][j] = (product*product)*tree[2][j];
    // These are the five allocations of density factors among halos.
    // SIMD evaluates both configurations without multiplying across them;
    // K and Q remain distinct inputs within each configuration.

    // --- 1. PACK THE INPUTS WITHOUT MIXING K AND Q ---

    // One trispectrum value combines two physical scales, K and Q, at the
    // same time a. Both scales must stay inside one SIMD lane. Lane 0 holds
    // the whole configuration at point; lane 1 holds the one at next.
    // For example, vp[0] contains [P_K(point), P_K(next)], while vp[1]
    // contains [P_Q(point), P_Q(next)]; K and Q are not the two lanes.
    // This arrangement evaluates the same physics twice without coupling
    // unrelated configurations. The three loops collect its ingredients:
    // single-leg factors, shared-halo moments, and correlations among halos.

    // P_lin describes the correlation linking two halo positions. I11 adds
    // a halo's bias-weighted density profile for one leg. Read both at K,
    // then both at Q, so the 1+3 terms can attach the correct isolated leg.
    for (int role=0; role<2; role++) {
      // set_pd takes lane 1 first, so lane 0 gets pk[role][point] and
      // lane 1 pk[role][next]: P_lin at K (role 0) or Q (role 1). role
      // selects K or Q; the lanes select the configurations point, next.
      vp[role] = simde_mm_set_pd(pk[role][next], pk[role][point]);

      // Pack I11 at the same magnitude, i11[role][point] in lane 0 and
      // i11[role][next] in lane 1. The reversed argument order of set_pd
      // keeps it aligned with the powers above.
      vi[role] = simde_mm_set_pd(i11[role][next], i11[role][point]);
    }

    // A mass moment combines density profiles whose legs share one halo,
    // weighted by halo abundance and, where needed, bias. Preserve the
    // supplied row order: I02, I12, I13(K,Q,Q), I13(K,K,Q), I04. The two
    // I13 rows differ because the halo can contain two Q legs or two K legs.
    // vm[0] (I02) is packed with the others, but no term below reads it.
    for (int role=0; role<5; role++) {
      // Pack one halo-moment role for both points. set_pd puts its last
      // argument in lane 0; vm[role] is [moment(point), moment(next)].
      vm[role] = simde_mm_set_pd(moments[role][next], moments[role][point]);
    }

    // Separate halos still need correlations between their positions.
    // AvgP, AvgB and AvgT supply the two-, three- and connected four-point
    // correlations, already averaged over the relative angle of k and q.
    // This loop aligns each average with its own K,Q configuration; it
    // performs no new angular integral and never averages the SIMD lanes.
    for (int role=0; role<3; role++) {
      // Pack AvgP, AvgB or AvgT in point/next lane order. set_pd receives
      // the next-point value first because it fills the high lane first.
      vt[role] = simde_mm_set_pd(tree[role][next], tree[role][point]);
    }

    // --- 2. ONE-HALO TERM AND 1+3 TWO-HALO TERM ---

    // If all four legs share a halo, I04 already contains the full term.
    // For two halos with one leg in one and three in the other, a linear
    // power links I11 for the isolated leg to I13 for the remaining legs.
    // Isolating k leaves magnitudes K,Q,Q; isolating q leaves K,K,Q.
    // Isolating -k or -q gives the same respective contribution, explaining
    // the factor two on each group. Thus there are four partitions, grouped
    // into two products. Every product stays within its configuration's lane.

    // mul_pd forms product = I11(K)*I11(Q) within each configuration's
    // lane, for the later 3h and 4h terms.
    const v2d vi_product = simde_mm_mul_pd(vi[0], vi[1]);

    // mul_pd forms P_K*I11(K) separately for the two configurations.
    const v2d vbiased_pk = simde_mm_mul_pd(vp[0], vi[0]);

    // mul_pd attaches the three-leg halo to the isolated K leg:
    // left13 = (P_K*I11(K))*I13(K,Q,Q) in each lane.
    const v2d vleft13 = simde_mm_mul_pd(vbiased_pk, vm[2]);

    // mul_pd forms P_Q*I11(Q) in each lane for the other isolated leg.
    const v2d vbiased_pq = simde_mm_mul_pd(vp[1], vi[1]);

    // mul_pd attaches I13(K,K,Q) to that isolated Q leg at the matching
    // point: right13 = (P_Q*I11(Q))*I13(K,K,Q).
    const v2d vright13 = simde_mm_mul_pd(vbiased_pq, vm[3]);

    v2d vterms[5];

    // All four density legs in one halo give I04 directly (a plain copy
    // of the packed vector, not an intrinsic).
    vterms[0] = vm[4];

    // add_pd adds the K-isolated and Q-isolated partitions within each
    // lane: left13 + right13.
    const v2d vpartitions13 = simde_mm_add_pd(vleft13, vright13);

    // mul_pd by 2.0: each isolated magnitude has two signs (k or -k,
    // q or -q), so T_13 = 2*(left13+right13) in each lane.
    vterms[1] = simde_mm_mul_pd(vtwo, vpartitions13);

    // --- 3. TWO HALOS WITH TWO LEGS EACH ---

    // Each halo now contains one K leg and one Q leg, giving two factors
    // of I12(K,Q). Their positions correlate through the exchanged power.
    // Pairing k with q or with -q gives two channels with the same angular
    // average AvgP, hence 2*I12^2*AvgP. The zero-exchange pairing (k,-k)
    // with (q,-q) is excluded from this cNG term; survey-background effects
    // are handled separately in SSC.

    // mul_pd squares I12(K,Q) independently in each configuration's lane.
    const v2d vi12_squared = simde_mm_mul_pd(vm[1], vm[1]);

    // mul_pd by 2.0 counts the two nonzero exchange channels:
    // 2*I12^2 in each lane.
    const v2d vtwo_i12_squared = simde_mm_mul_pd(vtwo, vi12_squared);

    // mul_pd weights those channels by their averaged exchanged power:
    // T_22 = (2*I12^2)*AvgP at each point.
    vterms[2] = simde_mm_mul_pd(vtwo_i12_squared, vt[0]);

    // --- 4. THREE HALOS AND FOUR HALOS ---

    // With three halos, one contains two legs and the others one each.
    // Their mass weights are I12*I11(K)*I11(Q); AvgB correlates the three
    // halo positions. Four surviving choices of the paired legs give the
    // multiplicity four: the pairs (k,-k) and (q,-q) carry zero momentum
    // and belong to SSC. With four halos, every leg has its own I11 factor,
    // giving I11(K)^2*I11(Q)^2. AvgT then supplies the connected correlation
    // between those four halos. Neither step mixes different SIMD points.

    // mul_pd joins the two-leg halo I12 to the two single-leg I11
    // factors: I12*(I11(K)*I11(Q)) in each lane.
    const v2d vthree_halo_weight = simde_mm_mul_pd(vm[1], vi_product);

    // mul_pd by 4.0 includes the four choices of the paired legs,
    // separately per lane.
    const v2d vfour_partitions = simde_mm_mul_pd(vfour, vthree_halo_weight);

    // mul_pd by the averaged bispectrum gives T_3h at each point:
    // (4*(I12*product))*AvgB.
    vterms[3] = simde_mm_mul_pd(vfour_partitions, vt[1]);

    // mul_pd squares product: four separate halos supply
    // [I11(K)*I11(Q)]^2, lane by lane.
    const v2d vfour_halo_weight = simde_mm_mul_pd(vi_product, vi_product);

    // mul_pd by AvgT gives T_4h = (product*product)*AvgT at the matching
    // point.
    vterms[4] = simde_mm_mul_pd(vfour_halo_weight, vt[2]);

    // --- 5. KEEP EACH HALO CONTRIBUTION IN ITS OWN OUTPUT ROW ---

    // Keep the five physical contributions separate so their signs and
    // relative sizes can be checked before survey projection. One iteration
    // writes a halo term's two point values into the matching output row:
    // 1h, 2h(1+3), 2h(2+2), 3h or 4h. These lanes are never added together;
    // if the last point was duplicated for reading, write it only once.
    for (int role=0; role<5; role++) {
      double result[2];

      // storeu writes this halo term's lane 0 (configuration point) to
      // result[0] and lane 1 (configuration next) to result[1]. It needs
      // no vector alignment on this two-double stack array.
      simde_mm_storeu_pd(result, vterms[role]);

      terms[role][point] = result[0];
      if (point+1 < npoint) {
        terms[role][next] = result[1];
      }
    }
  }
}
