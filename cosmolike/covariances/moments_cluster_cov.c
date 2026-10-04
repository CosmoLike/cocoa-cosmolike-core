#include <stdlib.h>

#include "moments_cluster_cov.h"
#include "cosmolike/basics.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

// Each SIMD value holds two doubles, called lanes. We use the lanes for
// separate integrals, never to split a mass sum. Thus changing the OpenMP
// thread count cannot change the order in which a moment adds its masses.
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
// matter-profile factors. We keep the unnormalized selected integrals
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
//   kernel of Schaan, Takada & Spergel (2014), arXiv:1406.3330, Eq. 35.
//   I11 is the all-halo biased one-profile moment, NOT a selected J11.
// * J02 and J03 supply same-halo terms with respectively two or one
//   cluster legs in the connected cluster-lensing covariance. Each cluster
//   leg's abundance normalization belongs to the later assembly.
//
// An observed cluster is assigned to one category. Its membership
// indicator obeys I_i^2=I_i, so the same-halo expectation contains S_i
// once, not S_i^2. Two different exclusive categories cannot receive the
// same halo even if their true-mass distributions overlap. Do not multiply
// their selection probabilities to construct that same-halo cross term.
// Correlations of DIFFERENT halos and their SSC are not excluded by this
// rule. Those cross-category contributions must be retained separately.
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

  // Integrate every selected population independently. Lane 0 counts its
  // halos, while lane 1 weights those same halos by their linear response
  // to delta_b. Keeping the two sums together reuses the selection read.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int state=0; state<na; state++) {
    for (int bin=0; bin<nselection; bin++) {
      const int row = state*nselection+bin; // flattened output population
      const double* restrict measure = weight[state][bin]; // dn*S_i
      const double* restrict halo_bias = bias[state]; // b(M) at this a

      // Both integrals start at zero: no mass node has contributed yet.
      v2d sum = simde_mm_setzero_pd();

      // Adding each mass node counts the objects it contributes and the
      // change of that contribution under a unit background overdensity.
      for (int mass=0; mass<nmass; mass++) {
        // Both integrals use the same selected number measure dn*S_i.
        const v2d vweight = simde_mm_set1_pd(measure[mass]);

        // set_pd takes the HIGH lane first: lane 0 gets 1, lane 1 gets b.
        const v2d factor = simde_mm_set_pd(halo_bias[mass], 1.0);

        // Add dn*S_i and dn*S_i*b to their separate sums. FMA computes
        // each multiplication plus addition with one native-FMA rounding.
        sum = simde_mm_fmadd_pd(vweight, factor, sum);
      }
      double result[2]; // n_i and its abundance response, both in L^-3

      // Store both lanes in an ordinary stack array, without an alignment
      // requirement. Lane 0 becomes result[0], lane 1 becomes result[1].
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

        // Lane 0 integrates K, lane 1 integrates Q; neither has a mass yet.
        v2d mean = simde_mm_setzero_pd();

        // Start the corresponding bias-weighted integrals at zero too.
        v2d response = simde_mm_setzero_pd();

        // Reuse each selected mass weight for both k values. Profiles were
        // evaluated by the caller, so this loop only multiplies and sums.
        for (int mass=0; mass<nmass; mass++) {
          // Pack p(K|M) in lane 0 and p(Q|M) in lane 1 (high lane first).
          const v2d p = simde_mm_set_pd(second[mass], first[mass]);

          // Both wavenumbers see the same selected halo population.
          const v2d w = simde_mm_set1_pd(measure[mass]);

          // Its linear abundance response is dn*S_i*b, shared by both k.
          const v2d wb = simde_mm_set1_pd(measure[mass]*halo_bias[mass]);

          // Each lane adds one selected halo's matter contribution.
          mean = simde_mm_fmadd_pd(w, p, mean);

          // Each lane separately adds the change in that contribution.
          response = simde_mm_fmadd_pd(wb, p, response);
        }
        double result[2][2]; // [mean/response][K/Q], dimensionless

        // storeu writes K then Q without requiring aligned stack storage.
        simde_mm_storeu_pd(result[0], mean);

        // Store the bias-weighted pair in the second row in the same order.
        simde_mm_storeu_pd(result[1], response);

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

        // No mass has contributed to the two-profile moments yet.
        v2d sum2 = simde_mm_setzero_pd();

        // Initialize the three-profile moments with K repeated to zero.
        v2d sum3k = simde_mm_setzero_pd();

        // Likewise initialize the moments with Q repeated to zero.
        v2d sum3q = simde_mm_setzero_pd();

        // One selected halo supplies all legs of a same-halo moment, so
        // its probability occurs once in every product, never squared.
        for (int mass=0; mass<nmass; mass++) {
          // Pack the first member of pair 0/1 into lane 0/1 (high first).
          const v2d k = simde_mm_set_pd(k1[mass], k0[mass]);

          // Pack each pair's second member in matching lane order.
          const v2d q = simde_mm_set_pd(q1[mass], q0[mass]);

          // Duplicate the selected halo measure for the two pair integrals.
          const v2d w = simde_mm_set1_pd(measure[mass]);

          // Multiply profiles WITHIN each pair, not between SIMD lanes.
          const v2d product2 = simde_mm_mul_pd(k, q);

          // Add one weighted two-leg contribution to each pair's sum.
          sum2 = simde_mm_fmadd_pd(w, product2, sum2);

          // A third matter leg at K multiplies each pair by its own p(K).
          const v2d product3k = simde_mm_mul_pd(product2, k);

          // Integrate these three-leg contributions for both pairs.
          sum3k = simde_mm_fmadd_pd(w, product3k, sum3k);

          // A third leg at Q instead multiplies each pair by its own p(Q).
          const v2d product3q = simde_mm_mul_pd(product2, q);

          // Integrate the Q-repeated moments in their separate accumulator.
          sum3q = simde_mm_fmadd_pd(w, product3q, sum3q);
        }
        double result[3][2]; // [J02/J03KKQ/J03KQQ][pair 0/pair 1]

        // Store the two J02 values in lane order; stack alignment is free.
        simde_mm_storeu_pd(result[0], sum2);

        // Store the K-repeated J03 pair in the second ordinary stack row.
        simde_mm_storeu_pd(result[1], sum3k);

        // Store the Q-repeated pair in the third row with the same ordering.
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
