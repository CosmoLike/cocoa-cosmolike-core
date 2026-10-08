#include <stdlib.h>

#include "moments_cluster_cov.h"
#include "cosmolike/basics.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

// SIMD (single instruction, multiple data) applies one instruction to
// several numbers at once. A v2d, SIMDe's simde__m128d, holds two
// doubles; each position is called a lane, lane 0 the low and lane 1 the
// high double. Here the two lanes always carry separate integrals, never
// two halves of one mass sum. Each integral adds its masses in increasing
// order, so changing the OpenMP thread count cannot change that order.
//
// simde_mm_fmadd_pd(a, b, c) returns a*b + c in each lane. With an FMA
// instruction (x86 built with FMA support, or arm64 NEON) the product is
// kept exact and only the sum is rounded, like the C function fma(a, b,
// c). Without one, SIMDe's portable code rounds the product and the sum
// separately.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Integrate the halo population selected into an observed cluster category.
//
// A mass bin contributes dn = (dn/dlnM) dlnM halos per comoving volume.
// Let S_i(M,a) be the probability that a halo enters observed category i.
// The caller supplies weight = dn*S_i at each quadrature node, including
// richness scatter, completeness and any redshift selection being used.
// Summing weight gives the selected abundance n_i. Multiplying by the
// linear halo bias b before summing gives dn_i/d(delta_b), provided S_i
// itself is held fixed in the background overdensity delta_b.
//
// A halo also contributes p(k|M) = (M/rho)*u(k|M) to the matter field.
// Here u is its dimensionless Fourier profile; p has units of volume.
// One matter leg requires one p, two legs require two p factors, etc.
// In J_beta_mu, beta says whether halo bias is present and mu counts the
// matter-profile factors. The routine returns the unnormalized selected
// integrals
//
//   J01(K)     = integral dn S_i p(K),
//   J11(K)     = integral dn S_i b p(K),
//   J02(K,Q)   = integral dn S_i p(K) p(Q),
//   J03(K,K,Q) = integral dn S_i p(K)^2 p(Q),
//   J03(K,Q,Q) = integral dn S_i p(K) p(Q)^2.
//
// Their uses explain why these moments share one profile table:
// * J01/n_i is the one-halo cluster-matter power in To et al. (2021),
//   arXiv:2008.10757, Eqs. 20--21. J11/n_i supplies its abundance response,
//   before any observed-mean subtraction for a measured two-point function.
//   This is the abundance contribution only: a changing halo profile or
//   the growth/dilation of different-halo clustering needs separate terms.
// * J02(K,K) + 2 P_lin(K) I11(K) J11(K) is the non-SSC count-matter-power
//   kernel of Schaan, Takada & Spergel (2014), arXiv:1406.3330, Eq. 35:
//   their one-halo sum n_i p_i^1h is J02(K,K), and their two-halo sum
//   2 n_i n_j p_ij^2h, with i selected and j any halo, is the second term.
//   I11(K) = integral dn b p(K) runs over all halos, without S_i; it is
//   not a selected J11.
// * J02 and J03 supply same-halo terms with respectively two or one
//   cluster legs in the connected cluster-lensing covariance. Each cluster
//   leg's abundance normalization belongs to the later assembly.
//
// An observed cluster is assigned to one category. Its membership
// indicator obeys I_i^2=I_i, so the same-halo expectation contains S_i
// once, not S_i^2. Two different exclusive categories cannot receive the
// same halo even if their true-mass distributions overlap (I_i I_j = 0
// for i != j). Do not multiply their selection probabilities to construct
// that same-halo cross term. Correlations of different halos and their
// SSC are not excluded by this rule. Those cross-category contributions
// must be retained separately.
//
// This routine chooses no abundance, bias, profile or selection model.
// There is no low-mass completion: an unresolved matter contribution does
// not represent additional detected clusters. An environmental selection
// derivative needs an additional supplied model, outside this fixed-S
// interpretation. No survey mask, shot noise, catalog-mean subtraction,
// SSC or full trispectrum is assembled here.
//
// The three stages below compute abundances, single-profile moments, then
// pair moments. All k pairs share the supplied profiles; none is evaluated
// again inside a mass integral. Every output owns its increasing-mass sum.
// OpenMP collapses independent state, selection and k/pair indices so a
// small number of categories does not limit the available workers.
//
// Parameters (length unit L throughout):
//   na         - number of independent states (scale factors), >= 1
//   nselection - number of observed categories i, >= 1
//   nk         - number of wavenumbers per state, >= 1
//   nmass      - number of mass quadrature nodes, >= 1
//   weight     - [na][nselection][nmass] dn S_i = dlnM (dn/dlnM) S_i,
//                in L^-3, nonnegative
//   bias       - [na][nmass] linear halo bias b(M)
//   profile    - [na][nk][nmass] p(k|M) = (M/rho) u(k|M), in L^3
//
// Outputs (row = state*nselection + selection; every entry is written):
//   density    - [2][row]: n_i and dn_i/d(delta_b), in L^-3
//   single     - [2][row][nk]: J01(k) and J11(k), dimensionless
//   pair       - [3][row][nk(nk+1)/2]: J02(K,Q) in L^3, J03(K,K,Q) and
//                J03(K,Q,Q) in L^6, for each upper-triangle pair of
//                wavenumber indices, index(K) <= index(Q)
// ---------------------------------------------------------------------------
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
  )
{
  if (na < 1
      || nselection < 1
      || nk < 1
      || nmass < 1) {
    log_fatal("moments_cluster_cov needs positive state, selection, "
              "profile and mass counts");
    exit(1);
  }

  // --- 1. ABUNDANCE AND ITS FIXED-SELECTION RESPONSE ---

  // Integrate every selected population independently. One iteration
  // handles one population, a (state, selection) row, over all masses in
  // increasing order. Lane 0 counts its halos, n_i = integral dn S_i,
  // while lane 1 weights those same halos by their linear response to
  // delta_b, dn_i/d(delta_b) = integral dn S_i b. The two lanes stay
  // separate sums; keeping them together reuses the selection read.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int state=0; state<na; state++) {
    for (int bin=0; bin<nselection; bin++) {
      const int row = state*nselection+bin; // flattened output population
      const double* restrict measure = weight[state][bin]; // dn*S_i
      const double* restrict halo_bias = bias[state]; // b(M) at this a

      // scalar: with number = response = 0 initially,
      //   for (int mass=0; mass<nmass; mass++) {
      //     number = fma(measure[mass], 1.0, number);
      //     response = fma(measure[mass], halo_bias[mass], response);
      //   }
      // The first sum counts the selected halos per volume; bias weights
      // how their abundance changes with background density. SIMD assigns
      // one lane to number and one to response, not to different masses.
      // fma(x, 1.0, s) is s + x rounded once, an ordinary addition,
      // because x*1.0 is exact.
      // setzero sets both lanes to 0.0: lane 0 starts n_i and lane 1
      // dn_i/d(delta_b); no mass node has contributed yet.
      v2d sum = simde_mm_setzero_pd();

      // Adding each mass node counts the objects it contributes and the
      // change of that contribution under a unit background overdensity.
      for (int mass=0; mass<nmass; mass++) {
        // set1 copies measure[mass] = dn S_i at this mass node into both
        // lanes: both integrals use the same selected number measure.
        const v2d vweight = simde_mm_set1_pd(measure[mass]);

        // set_pd(high, low) lists the high lane first: lane 0 gets 1.0,
        // which counts the halos, and lane 1 gets halo_bias[mass] = b(M),
        // which weights them by their response.
        const v2d factor = simde_mm_set_pd(halo_bias[mass], 1.0);

        // fmadd_pd(a, b, c) = a*b + c: lane 0 becomes fma(measure[mass],
        // 1.0, number) and lane 1 fma(measure[mass], halo_bias[mass],
        // response). Each lane is rounded once with an FMA instruction
        // (see the note at v2d).
        sum = simde_mm_fmadd_pd(vweight, factor, sum);
      }
      double result[2]; // n_i and its abundance response, both in L^-3

      // storeu writes lane 0, n_i, to result[0] and lane 1,
      // dn_i/d(delta_b), to result[1]. The ordinary stack array needs no
      // vector alignment.
      simde_mm_storeu_pd(result, sum);
      density[0][row] = result[0];
      density[1][row] = result[1];
    }
  }

  // --- 2. SINGLE-PROFILE MOMENTS ---

  // A selected halo contributes p(k) to its surrounding matter field.
  // Integrate it once with the abundance weight and once with the biased
  // abundance weight. Each SIMD lane owns a different k; the two moments
  // use separate accumulators so a response never mixes with a mean.
  // One iteration integrates the wavenumbers mode (lane 0) and next
  // (lane 1) of one population over all masses in increasing order.
  // mode advances by two. When nk is odd, the last iteration repeats mode
  // in lane 1 (next = mode), and that duplicate is discarded when
  // storing. Every iteration writes distinct output entries.
  #pragma omp parallel for collapse(3) schedule(static)
  for (int state=0; state<na; state++) {
    for (int bin=0; bin<nselection; bin++) {
      for (int mode=0; mode<nk; mode+=2) {
        const int next = mode+1 < nk ? mode+1 : mode; // valid second lane
        const int row = state*nselection+bin; // flattened population
        const double* restrict measure = weight[state][bin]; // dn*S_i
        const double* restrict halo_bias = bias[state]; // b(M)
        const double* restrict first = profile[state][mode]; // p(K|M)
        const double* restrict second = profile[state][next]; // p(Q|M)

        // scalar: for either mode j = mode, next,
        //   mean = 0;
        //   response = 0;
        //   for (int mass=0; mass<nmass; mass++) {
        //     p = profile[state][j][mass];
        //     mean = fma(measure[mass], p, mean);
        //     response = fma(measure[mass]*halo_bias[mass], p, response);
        //   }
        // The profile gives the matter associated with each selected halo;
        // bias weights the change in their abundance. SIMD keeps one mode
        // per lane and separate accumulators for the mean and its response.
        // In the response line, measure*halo_bias is rounded before the fma.
        // setzero sets both lanes of mean to 0.0: lane 0 will hold J01 at
        // K = k_mode and lane 1 at Q = k_next; neither has a mass yet.
        v2d mean = simde_mm_setzero_pd();

        // setzero starts the bias-weighted integrals J11 at k_mode (lane
        // 0) and k_next (lane 1) at 0.0 too.
        v2d response = simde_mm_setzero_pd();

        // Reuse each selected mass weight for both k values. Profiles were
        // evaluated by the caller, so this loop only multiplies and sums.
        for (int mass=0; mass<nmass; mass++) {
          // set_pd(high, low) lists the high lane first: lane 0 gets
          // first[mass] = p(k_mode|M) and lane 1 second[mass] =
          // p(k_next|M), the halo's mass-weighted profile at both k.
          const v2d p = simde_mm_set_pd(second[mass], first[mass]);

          // set1 copies measure[mass] = dn S_i into both lanes: both
          // wavenumbers see the same selected halo population.
          const v2d w = simde_mm_set1_pd(measure[mass]);

          // set1 copies the scalar product measure[mass]*halo_bias[mass]
          // = dn S_i b, rounded once before the copy, into both lanes:
          // the linear abundance response, shared by both k.
          const v2d wb = simde_mm_set1_pd(measure[mass]*halo_bias[mass]);

          // fmadd_pd(a, b, c) = a*b + c: lane s becomes
          // fma(measure[mass], p_s, mean_s), adding this mass node's
          // selected matter contribution to J01 at its own wavenumber;
          // one rounding with an FMA instruction (see the note at v2d).
          mean = simde_mm_fmadd_pd(w, p, mean);

          // fmadd_pd: lane s becomes fma(measure[mass]*halo_bias[mass],
          // p_s, response_s), the change of that contribution, added to
          // J11 at the same wavenumber; one rounding.
          response = simde_mm_fmadd_pd(wb, p, response);
        }
        double result[2][2]; // [mean/response][K/Q], dimensionless

        // storeu writes lane 0, J01 at k_mode, to result[0][0] and lane 1,
        // J01 at k_next, to result[0][1]. The stack array needs no vector
        // alignment.
        simde_mm_storeu_pd(result[0], mean);

        // storeu writes the J11 lanes to result[1][0] (k_mode) and
        // result[1][1] (k_next), in the same order.
        simde_mm_storeu_pd(result[1], response);

        // Write mode's moments, and next's only when it is a distinct
        // wavenumber: for odd nk the final group repeated mode in lane 1,
        // and that duplicate is discarded here.
        single[0][row][mode] = result[0][0];
        single[1][row][mode] = result[1][0];
        if (mode+1 < nk) {
          single[0][row][next] = result[0][1];
          single[1][row][next] = result[1][1];
        }
      }
    }
  }

  // --- 3. TWO- AND THREE-PROFILE MOMENTS ---

  const int npair = nk*(nk+1)/2; // number of unordered wavenumber pairs
  int** modes = (int**) malloc2d_int(2, npair); // each pair's K and Q IDs
  int index = 0; // next triangular output index

  // The upper triangle avoids evaluating symmetric pairs twice. The
  // two J03 outputs retain which member of that pair appears twice.
  for (int first=0; first<nk; first++) {
    for (int second=first; second<nk; second++) {
      modes[0][index] = first;
      modes[1][index] = second;
      index++;
    }
  }

  // Same-halo correlations contain products of its profiles at K and Q.
  // Evaluate three products with shared reads: pK*pQ, pK^2*pQ, pK*pQ^2.
  // The lanes own different (K,Q) pairs. Every population and pair can
  // therefore be assigned to a worker without changing a mass-sum order.
  // One iteration integrates the wavenumber pairs item (lane 0) and next
  // (lane 1) of one population over all masses in increasing order. item
  // advances by two. When npair is odd, the last iteration repeats item
  // in lane 1 (next = item), and that duplicate is discarded when
  // storing.
  #pragma omp parallel for collapse(3) schedule(static)
  for (int state=0; state<na; state++) {
    for (int bin=0; bin<nselection; bin++) {
      for (int item=0; item<npair; item+=2) {
        const int next = item+1 < npair ? item+1 : item; // valid lane 1
        const int row = state*nselection+bin; // flattened population
        const double* restrict measure = weight[state][bin]; // dn*S_i
        const double* restrict k0 = profile[state][modes[0][item]]; // K0
        const double* restrict q0 = profile[state][modes[1][item]]; // Q0
        const double* restrict k1 = profile[state][modes[0][next]]; // K1
        const double* restrict q1 = profile[state][modes[1][next]]; // Q1

        // scalar: at mass m, for either pair j = item, next,
        //   k = profile[state][modes[0][j]][m];
        //   q = profile[state][modes[1][j]][m];
        //   product = k*q;
        //   sum2 = fma(measure[m], product, sum2);
        //   sum3k = fma(measure[m], product*k, sum3k);
        //   sum3q = fma(measure[m], product*q, sum3q);
        // Start these three sums at zero and integrate all masses in order.
        // Each product is rounded once before its fma.
        // The profiles describe several matter legs in one selected halo,
        // so its membership probability appears only once in measure[m].
        // SIMD evaluates two complete pairs without mixing their profiles.
        // setzero sets both lanes of sum2 to 0.0: lane 0 will hold J02 of
        // pair item and lane 1 of pair next; no mass has contributed yet.
        v2d sum2 = simde_mm_setzero_pd();

        // setzero starts sum3k, the J03(K,K,Q) of the two pairs, at 0.0.
        v2d sum3k = simde_mm_setzero_pd();

        // setzero starts sum3q, the J03(K,Q,Q) of the two pairs, at 0.0.
        v2d sum3q = simde_mm_setzero_pd();

        // One selected halo supplies all legs of a same-halo moment, so
        // its probability occurs once in every product, never squared.
        // One iteration adds one mass node to all six sums.
        for (int mass=0; mass<nmass; mass++) {
          // scalar: k = p(K|M), q = p(Q|M) of each lane's pair

          // set_pd(high, low) lists the high lane first: lane 0 gets
          // k0[mass] = p(K|M) of pair item, lane 1 k1[mass] of pair next.
          const v2d k = simde_mm_set_pd(k1[mass], k0[mass]);

          // set_pd(high, low): lane 0 gets q0[mass] = p(Q|M) of pair
          // item, lane 1 q1[mass] of pair next, matching the K lanes.
          const v2d q = simde_mm_set_pd(q1[mass], q0[mass]);

          // set1 copies measure[mass] = dn S_i into both lanes; both pair
          // integrals run over the same selected halos.
          const v2d w = simde_mm_set1_pd(measure[mass]);

          // scalar: product = k*q

          // mul_pd, lane by lane: p(K|M) p(Q|M) of each pair, rounded
          // once. Profiles are multiplied within a pair, never across
          // SIMD lanes.
          const v2d product2 = simde_mm_mul_pd(k, q);

          // scalar: sum2 = fma(measure[m], product, sum2)

          // fmadd_pd(a, b, c) = a*b + c: lane s becomes
          // fma(measure[mass], product_s, sum2_s), adding dn S_i p(K)
          // p(Q) to that pair's J02; one rounding with an FMA
          // instruction (see the note at v2d).
          sum2 = simde_mm_fmadd_pd(w, product2, sum2);

          // scalar: sum3k = fma(measure[m], product*k, sum3k)

          // mul_pd: a third matter leg at K multiplies each pair by its
          // own p(K), giving p(K)^2 p(Q), rounded once.
          const v2d product3k = simde_mm_mul_pd(product2, k);

          // fmadd_pd: lane s becomes fma(measure[mass], product3k_s,
          // sum3k_s), the J03(K,K,Q) term of that pair; one rounding.
          sum3k = simde_mm_fmadd_pd(w, product3k, sum3k);

          // scalar: sum3q = fma(measure[m], product*q, sum3q)

          // mul_pd: a third leg at Q instead multiplies each pair by its
          // own p(Q), giving p(K) p(Q)^2, rounded once.
          const v2d product3q = simde_mm_mul_pd(product2, q);

          // fmadd_pd: lane s becomes fma(measure[mass], product3q_s,
          // sum3q_s), the J03(K,Q,Q) term in its separate accumulator;
          // one rounding.
          sum3q = simde_mm_fmadd_pd(w, product3q, sum3q);
        }
        double result[3][2]; // [J02/J03KKQ/J03KQQ][pair 0/pair 1]

        // storeu writes lane 0, J02 of pair item, to result[0][0] and
        // lane 1, J02 of pair next, to result[0][1]. The ordinary stack
        // array needs no vector alignment.
        simde_mm_storeu_pd(result[0], sum2);

        // storeu writes the J03(K,K,Q) lanes to result[1][0] (item) and
        // result[1][1] (next).
        simde_mm_storeu_pd(result[1], sum3k);

        // storeu writes the J03(K,Q,Q) lanes to result[2][0] (item) and
        // result[2][1] (next).
        simde_mm_storeu_pd(result[2], sum3q);

        // An odd final pair was duplicated into the spare SIMD lane.
        // Discard that lane rather than writing beyond the output row.
        for (int role=0; role<3; role++) {
          pair[role][row][item] = result[role][0];
          if (item+1 < npair) {
            pair[role][row][next] = result[role][1];
          }
        }
      }
    }
  }

  free(modes);
}
