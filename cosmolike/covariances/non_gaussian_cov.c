#include <math.h>
#include <stdlib.h>

#include "non_gaussian_cov.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

// A vector holds two doubles in positions called lanes. Every operation
// below acts separately on two (k,a) points or two (K,Q,a) combinations.
// A fused multiply-add (FMA) evaluates a*b+c with one rounding when
// supported directly by the processor. This differs from rounding a*b
// first and then adding c; the calls below retain the chosen operations.
// Unaligned loads/stores accept addresses that are not multiples of 16
// bytes. They still require two valid adjacent doubles in the array.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Assemble a dimensional halo-model response with explicit model choices.
//
// The two-halo power is P_2h = I11^2 P_lin; the one-halo power is I02.
// A background density mode changes their amplitudes. In the adopted
// halo approximation the one-halo derivative is I12, while the two-halo
// term contains growth and a dilation (rescaling of wavenumber):
//
//   P_halo = P_2h + I02,
//   D_halo = (growth_coefficient - dilation_coefficient * slope) P_2h
//            + I12.
//
// The supplied slope is dlnP_X/dlnk, WITHOUT the k^3 factor. Published
// isotropic halo response: growth=47/21, dilation=1/3, P_X=P_2h.
// This is Takada & Hu (2013), arXiv:1302.6994v3, corrected Eq. 44:
// 68/21 - (1/3) dln(k^3 P_2h)/dlnk = 47/21 - (1/3) dlnP_2h/dlnk.
//
// A planar squeezed-tree construction instead uses growth=17/7,
// dilation=1/2 and P_X=P_lin (study report 10, Section 3.1). Its linear
// limit is R1+RK/6 with tree-level responses, as in Barreira, Krause &
// Schmidt (2018), arXiv:1711.07467, Sections 2 and 4.1. This is not a
// calibrated nonlinear tidal response. These choices are deliberately
// explicit inputs; the code does not label a study-derived form a paper
// erratum or silently replace the published two-halo slope.
//
// If fractional=1, preserve the halo model's FRACTIONAL response but use
// a supplied target power (for example Halofit): D=(D_halo/P_halo) P_target.
// If fractional=0, return D_halo. This replacement is a modeling choice,
// not an identity of perturbation theory. Galaxy survey-mean subtraction
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

  // Copy the growth coefficient to both lanes, so each point receives
  // the same response prescription without mixing their power spectra.
  const v2d vgrowth = simde_mm_set1_pd(growth_coefficient);

  // Copy the dilation coefficient to both lanes; each will multiply its
  // own logarithmic power-spectrum slope.
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

    // Scalar equivalent at either point j=point,next:
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
      // lane 1 gets inputs[role][next]. Role order is defined above.
      values[role] = simde_mm_set_pd(inputs[role][next], inputs[role][point]);
    }

    // --- 3. P_halo = I11^2 P_lin + I02 ---

    // Two density factors can belong to the same halo or to separate halos.
    // I02 includes their common halo profile and abundance in the first
    // case. In the second, P_lin correlates the halo positions, with one
    // bias-weighted profile factor I11 for each halo. Adding both cases
    // gives the power whose fractional change is needed below.

    // Square I11 independently at each point; no cross-point product.
    const v2d vi11_squared = simde_mm_mul_pd(values[2], values[2]);

    // Multiply each I11^2 by its own P_lin to obtain P_2h.
    const v2d vp2h = simde_mm_mul_pd(vi11_squared, values[0]);

    // Add the one-halo power I02 to P_2h in each lane.
    const v2d vhalo = simde_mm_add_pd(vp2h, values[3]);

    // --- 4. D_halo = (growth - dilation*slope) P_2h + I12 ---

    // D_halo is the derivative of power with respect to a background
    // overdensity, evaluated at zero overdensity. The slope term accounts
    // for shifting k along a nonconstant spectrum; the growth term changes
    // its amplitude. I12 adds the response of the one-halo contribution.
    // The optional rescaling keeps D_halo/P_halo while replacing the power
    // amplitude by P_target. This is a model choice, not another halo term.

    // fnmadd means -(first*second)+third: here growth-dilation*slope.
    // Each lane uses one fused rounding on native FMA hardware.
    const v2d vcoefficient = simde_mm_fnmadd_pd(vdilation, values[5], vgrowth);

    // Form coefficient*P_2h+I12 separately at both points. fmadd fuses
    // the multiplication and addition into one native-FMA rounding.
    v2d vresponse = simde_mm_fmadd_pd(vcoefficient, vp2h, values[4]);

    if (fractional) {
      // Divide each D_halo by its own P_halo: this is the dimensionless
      // fractional response, not a ratio between the two points.
      const v2d vfraction = simde_mm_div_pd(vresponse, vhalo);

      // Multiply by P_target at the matching point to restore length^3.
      vresponse = simde_mm_mul_pd(vfraction, values[1]);
    }

    // --- 5. WRITE THE TWO PHYSICAL OUTPUT ROWS ---

    // Return power and response separately so the caller can inspect their
    // ratio and form the survey's SSC response. The two SIMD lanes describe
    // different points, so they become adjacent entries rather than a sum.

    double result[2][2];

    // Copy halo powers from lanes 0/1 to result[0][0/1]. storeu accepts
    // this ordinary stack array without special vector alignment.
    simde_mm_storeu_pd(result[0], vhalo);

    // Copy responses to result[1][0/1] in the same point order.
    // storeu requires two valid doubles, but no vector-aligned address.
    simde_mm_storeu_pd(result[1], vresponse);

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
// For the covariance parallelogram (k,-k,q,-q), angular averaging gives
//
//   T_1h  = I04(K,K,Q,Q),
//   T_13  = 2 [P_K I11(K) I13(K,Q,Q) + P_Q I11(Q) I13(K,K,Q)],
//   T_22  = 2 I12(K,Q)^2 AvgP,
//   T_3h  = 4 I12(K,Q) I11(K) I11(Q) AvgB,
//   T_4h  = [I11(K) I11(Q)]^2 AvgT.
//
// In T_13 the isolated leg can be k, -k, q or -q: FOUR choices. The
// first two have the same magnitudes, as do the last two. Thus each
// displayed term has coefficient two, not one. This follows directly
// from the four set partitions in Takada & Hu (2013), Eqs. 28-29,
// arXiv:1302.6994; the study's old CosmoCov implementation has only one.
// The two nonzero exchange channels give T_22's factor two; four choices
// of the paired halo give T_3h's factor four. tree_averages_cov supplies
// the linear power, bispectrum and trispectrum averages with the exact
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
// Two SIMD lanes process different pairs; the final odd lane is discarded.
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

  // Copy the two exchange-channel multiplicity to both point lanes.
  const v2d vtwo = simde_mm_set1_pd(2.0);

  // Copy the four choices of the paired halo to both point lanes.
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

    // Scalar equivalent for one configuration j=point or next:
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
      // set_pd takes lane 1 first, so lane 0 gets P at point and lane 1
      // P at next. role selects K or Q; lanes select independent pairs.
      vp[role] = simde_mm_set_pd(pk[role][next], pk[role][point]);

      // Pack I11 with point in lane 0 and next in lane 1. The reversed
      // argument order of set_pd keeps it aligned with the powers above.
      vi[role] = simde_mm_set_pd(i11[role][next], i11[role][point]);
    }

    // A mass moment combines density profiles whose legs share one halo,
    // weighted by halo abundance and, where needed, bias. Preserve the
    // supplied row order: I02, I12, I13(K,Q,Q), I13(K,K,Q), I04. The two
    // I13 rows differ because the halo can contain two Q legs or two K legs.
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

    // Multiply I11(K)*I11(Q) within each pair for the later 3h/4h terms.
    const v2d vi_product = simde_mm_mul_pd(vi[0], vi[1]);

    // Form P_K*I11(K) separately for the two independent pairs.
    const v2d vbiased_pk = simde_mm_mul_pd(vp[0], vi[0]);

    // Attach the three-leg halo I13(K,Q,Q) to the isolated K leg.
    const v2d vleft13 = simde_mm_mul_pd(vbiased_pk, vm[2]);

    // Form P_Q*I11(Q) in each lane for the other isolated leg.
    const v2d vbiased_pq = simde_mm_mul_pd(vp[1], vi[1]);

    // Attach I13(K,K,Q) to that isolated Q leg at the matching point.
    const v2d vright13 = simde_mm_mul_pd(vbiased_pq, vm[3]);

    v2d vterms[5];

    // All four density legs in one halo give I04 directly.
    vterms[0] = vm[4];

    // Add the K-isolated and Q-isolated partitions within each lane.
    const v2d vpartitions13 = simde_mm_add_pd(vleft13, vright13);

    // Each isolated magnitude has two signs, giving twice that sum.
    vterms[1] = simde_mm_mul_pd(vtwo, vpartitions13);

    // --- 3. TWO HALOS WITH TWO LEGS EACH ---

    // Each halo now contains one K leg and one Q leg, giving two factors
    // of I12(K,Q). Their positions correlate through the exchanged power.
    // Pairing k with q or with -q gives two channels with the same angular
    // average AvgP, hence 2*I12^2*AvgP. The zero-exchange pairing (k,-k)
    // with (q,-q) is excluded from this cNG term; survey-background effects
    // are handled separately in SSC.

    // Square I12(K,Q) independently for each pair of external magnitudes.
    const v2d vi12_squared = simde_mm_mul_pd(vm[1], vm[1]);

    // Count the two nonzero exchange channels: 2*I12^2 in each lane.
    const v2d vtwo_i12_squared = simde_mm_mul_pd(vtwo, vi12_squared);

    // Weight those channels by their averaged exchanged power AvgP.
    vterms[2] = simde_mm_mul_pd(vtwo_i12_squared, vt[0]);

    // --- 4. THREE HALOS AND FOUR HALOS ---

    // With three halos, one contains two legs and the others one each.
    // Their mass weights are I12*I11(K)*I11(Q); AvgB correlates the three
    // halo positions. Four surviving choices of the paired legs give the
    // multiplicity four. With four halos, every leg has its own I11 factor,
    // giving I11(K)^2*I11(Q)^2. AvgT then supplies the connected correlation
    // between those four halos. Neither step mixes different SIMD points.

    // Join the two-leg halo I12 to the two single-leg I11 factors.
    const v2d vthree_halo_weight = simde_mm_mul_pd(vm[1], vi_product);

    // Include the four choices of the paired halo, separately per lane.
    const v2d vfour_partitions = simde_mm_mul_pd(vfour, vthree_halo_weight);

    // Multiply by the averaged bispectrum to obtain T_3h at each point.
    vterms[3] = simde_mm_mul_pd(vfour_partitions, vt[1]);

    // Four separate halos supply [I11(K)*I11(Q)]^2, lane by lane.
    const v2d vfour_halo_weight = simde_mm_mul_pd(vi_product, vi_product);

    // Multiply by AvgT to obtain T_4h at the matching point.
    vterms[4] = simde_mm_mul_pd(vfour_halo_weight, vt[2]);

    // --- 5. KEEP EACH HALO CONTRIBUTION IN ITS OWN OUTPUT ROW ---

    // Keep the five physical contributions separate so their signs and
    // relative sizes can be checked before survey projection. One iteration
    // writes a halo term's two point values into the matching output row:
    // 1h, 2h(1+3), 2h(2+2), 3h or 4h. These lanes are never added together;
    // if the last point was duplicated for reading, write it only once.
    for (int role=0; role<5; role++) {
      double result[2];

      // Copy lanes 0/1 to result[0/1], preserving point/next order.
      // storeu needs no vector alignment on this two-double stack array.
      simde_mm_storeu_pd(result, vterms[role]);

      terms[role][point] = result[0];
      if (point+1 < npoint) {
        terms[role][next] = result[1];
      }
    }
  }
}
