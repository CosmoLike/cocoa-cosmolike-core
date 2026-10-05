#include <math.h>
#include <stdlib.h>

#include "spectra_cluster_cov.h"
#include "cosmolike/basics.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

// Two doubles occupy positions called lanes. Each lane below integrates
// a different field pair; the radial nodes of either sum stay in order.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Add all cluster spectra to a galaxy/shear covariance calculation.
//
// A normalized cluster window q_c obeys integral q_c dchi = 1. For an
// abundance-selected catalog, q_c is proportional to f_K^2 phi_c n_c:
// shell volume times selection probability times comoving abundance.
// It does NOT include the cluster bias b_c. Keeping those factors
// separate is essential because a cluster's own mass profile is not
// multiplied by its large-scale bias.
//
// The supported three-dimensional mean model is
//   P_cc' = b_c b_c' P_NL,
//   P_cg  = b_c b_g P_NL,
//   P_cm  = b_c P_NL + P_cm^1h.
// P_cm^1h is the selected halo's mass profile, averaged within a richness
// category. See To et al. (2021), arXiv:2008.10757, Secs. 4.1.2--4.1.3.
// The galaxy model is linear bias on these scales; it has no additional
// cluster-galaxy one-halo term. No claim of a complete joint halo model
// follows from using this mean-spectrum approximation.
//
// Limber projection converts these spectra into angular correlations:
//   C_cc' = integral dchi q_c q_c' b_c b_c' P_NL / f_K^2,
//   C_cg  = integral dchi q_c b_c W_g P_NL / f_K^2,
//   C_cs  = F_ell integral dchi q_c W_s (b_c P_NL + P_cm^1h)/f_K^2.
// Here W_g already contains galaxy bias, W_s is the lensing window,
// and F_ell = sqrt[(ell-1)ell(ell+1)(ell+2)]/(ell+1/2)^2 is the core
// harmonic shear convention. Real-space operators need the same extra
// source conversion as the galaxy/shear covariance. Noise is separate.
//
// This boundary takes supplied arrays so selection and numerical tables
// are explicit. It assumes zero IA, magnification and RSD; adding any
// of them requires separate field contributions, not merely changing W_s.
// No data-vector pair cut is imposed. Gaussian covariance can require
// cross-redshift and crossed cluster-source spectra absent from the mean.
// The caller must check the noise-inclusive field matrix and final total
// covariance for positivity; this routine never repairs a negative mode.
//
// The calculation has two stages. First enumerate pairs and combine
// q_c with b_c once per shell. Then distribute (ell,pair-group) outputs
// over OpenMP workers. SIMDe integrates two different spectra together,
// sharing matter power and radial measure without changing either sum's
// order. Scratch is grouped, caller inputs are immutable, and no global
// state or lazy table is accessed. Calls at this entry point are serial.
// ---------------------------------------------------------------------------
void limber_cluster_cov(
    const int nell,                   // multipole count
    const double* ell,                // multipoles
    const int nnode,                  // radial-node count
    const double* distance,           // transverse distances
    const double* dchi,               // radial quadrature weights
    const int nbase,                  // galaxy and source fields
    const int nlens,                  // number of galaxy fields
    const double* const* base,        // galaxy/lensing windows
    const int ncluster,               // cluster categories
    const double* const* window,      // normalized cluster windows
    const double* const* bias,        // selected cluster biases
    const double* const* power,       // common nonlinear matter powers
    const double* const* const* p1h,  // selected one-halo profiles
    const int* richness,              // profile index per cluster field
    double* const* spectra            // all requested pair spectra
  )
{
  if (nell < 1
      || nnode < 1
      || nbase < 1
      || ncluster < 1
      || nlens < 0
      || nlens > nbase) {
    log_fatal("limber_cluster_cov needs positive sizes and "
              "0 <= nlens <= nbase");
    exit(1);
  }

  // --- 1. PAIR MAP AND BIASED CLUSTER WINDOWS ---

  const int ncross = ncluster*nbase;
  const int npair = ncross+ncluster*(ncluster+1)/2;
  int** pairs = (int**) malloc2d_int(2, npair);
  double** weighted = (double**) malloc2d(ncluster+1, nnode);

  // The last row holds dchi/f_K^2, common to every spectrum. The other
  // rows hold q_c b_c. These arrays have different physical roles but
  // the same node axis, so one allocation owns their scratch storage.
  double* restrict measure = weighted[ncluster];
  for (int node=0; node<nnode; node++) {
    measure[node] = dchi[node]/(distance[node]*distance[node]);
  }

  // A density perturbation changes the cluster density by b_c times
  // its amplitude. Precompute q_c b_c because every multipole and every
  // partner field reuses it. The work is independent across bins/nodes.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int cluster=0; cluster<ncluster; cluster++) {
    for (int node=0; node<nnode; node+=2) {
      const double* restrict q = window[cluster];
      const double* restrict b = bias[cluster];
      double* restrict qb = weighted[cluster];

      // Scalar equivalent for the two shells j=node,node+1:
      //   qb[j] = q[j]*b[j];
      // The radial probability q places the catalog in distance; bias b
      // sets its response to matter. SIMD retains one shell in each lane.
      if (node+1 < nnode) {
        // Load two adjacent q_c samples into low/high lanes. loadu does
        // not require alignment, but both node addresses must be valid.
        const v2d vwindow = simde_mm_loadu_pd(q+node);

        // Load b_c at the same shells in the same lane order.
        const v2d vbias = simde_mm_loadu_pd(b+node);

        // Multiply each shell's window by its own bias, independently.
        const v2d vweighted = simde_mm_mul_pd(vwindow, vbias);

        // Store the two products in adjacent doubles, without imposing
        // vector alignment on the caller-independent scratch row.
        simde_mm_storeu_pd(qb+node, vweighted);
      } else {
        // The unmatched final shell uses the same one multiplication.
        qb[node] = q[node]*b[node];
      }
    }
  }

  // Every cluster is paired with every galaxy and source. Indexing the
  // right side by nbase+cluster distinguishes cluster-cluster pairs.
  int pair = 0;
  for (int cluster=0; cluster<ncluster; cluster++) {
    for (int field=0; field<nbase; field++) {
      pairs[0][pair] = cluster;
      pairs[1][pair] = field;
      pair++;
    }
  }
  for (int first=0; first<ncluster; first++) {
    for (int second=first; second<ncluster; second++) {
      pairs[0][pair] = first;
      pairs[1][pair] = nbase+second;
      pair++;
    }
  }

  // --- 2. COMMON-SHELL INTEGRALS FOR EVERY PAIR AND MULTIPOLE ---

  // Each shell supplies correlations of structures at the same distance.
  // The large-scale term uses q_c b_c P for every pair. Only a source
  // partner receives the cluster's own mass-profile term q_c P_cm^1h.
  // Collapse both output indices so a short multipole array still offers
  // enough independent spectra for eight workers. SIMD lanes own two
  // complete radial sums; neither threads nor lanes share a reduction.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int index=0; index<nell; index++) {
    for (int first_pair=0; first_pair<npair; first_pair+=2) {
      const int next_pair = first_pair+1 < npair ? first_pair+1 : npair-1;
      const int pair_id[2] = {first_pair, next_pair};
      const double* q[2];      // unbiased left cluster windows
      const double* qb[2];     // biased left cluster windows
      const double* right[2];  // partner windows, including its bias
      const double* own[2];    // one-halo profile, or NULL for density
      double factor[2];        // shear transfer, or unity for density
      const double* restrict pk = power[index];

      // Fix each lane's physical role before visiting shells. A cluster
      // partner reads q_c' b_c'; a galaxy reads its biased W_g; a source
      // reads W_s and additionally receives the selected halo's profile.
      for (int lane=0; lane<2; lane++) {
        const int left = pairs[0][pair_id[lane]];
        const int field = pairs[1][pair_id[lane]];
        q[lane] = window[left];
        qb[lane] = weighted[left];
        own[lane] = NULL;
        factor[lane] = 1.0;
        if (field >= nbase) {
          right[lane] = weighted[field-nbase];
        } else {
          right[lane] = base[field];
          if (field >= nlens) {
            const double l = ell[index];
            const double shifted = l+0.5;
            own[lane] = p1h[richness[left]][index];
            factor[lane] = sqrt((l-1.0)*l*(l+1.0)*(l+2.0))
                           /(shifted*shifted);
          }
        }
      }

      // Explicit row pointers keep the node loop free of pointer-array
      // lookups. restrict states that writes to the output cannot alter
      // these read-only rows while the two sums are accumulated.
      const double* restrict q0 = q[0];
      const double* restrict q1 = q[1];
      const double* restrict qb0 = qb[0];
      const double* restrict qb1 = qb[1];
      const double* restrict right0 = right[0];
      const double* restrict right1 = right[1];
      const double* restrict own0 = own[0];
      const double* restrict own1 = own[1];

      // Scalar equivalent for either pair lane s=0,1:
      //   total = 0;
      //   for (int node=0; node<nnode; node++) {
      //     halo = own[s] == NULL ? 0 : own[s][node];
      //     left = qb[s][node]*pk[node]+q[s][node]*halo;
      //     product = left*right[s][node];
      //     total = fma(product, measure[node], total);
      //   }
      //   result[s] = total;
      // The halo's own mass contributes only for source partners. Its
      // weight is q, while the correlated matter term has weight q*b.
      // SIMD carries the two projected spectra in distinct lanes; the
      // shear conversion factor is applied when storing the results.
      // Both pair sums start at zero. setzero does not mix the lanes.
      v2d vtotal = simde_mm_setzero_pd();

      // Multiply the two fields' shell contributions by dchi/f_K^2.
      // Radial order is fixed for each sum, so changing the thread count
      // cannot change the order of floating-point additions.
      for (int node=0; node<nnode; node++) {
        const double halo0 = own0 == NULL ? 0.0 : own0[node];
        const double halo1 = own1 == NULL ? 0.0 : own1[node];

        // set_pd takes the high lane first: lane 0 represents first_pair
        // and lane 1 next_pair. Pack their biased cluster windows.
        const v2d vbiased = simde_mm_set_pd(qb1[node], qb0[node]);

        // Both pairs sample the same matter power at this ell and shell.
        const v2d vpower = simde_mm_set1_pd(pk[node]);

        // Form q_c b_c P independently for the two pairs.
        const v2d vlarge = simde_mm_mul_pd(vbiased, vpower);

        // Pack the unbiased cluster windows in the same lane order.
        const v2d vwindow = simde_mm_set_pd(q1[node], q0[node]);

        // Density partners have zero here. Source partners receive their
        // selected richness profile, which can differ between the lanes.
        const v2d vprofile = simde_mm_set_pd(halo1, halo0);

        // The cluster's own mass is weighted by q_c, without b_c.
        const v2d vhalo = simde_mm_mul_pd(vwindow, vprofile);

        // Add the one-halo and biased matter terms within each pair only.
        const v2d vleft = simde_mm_add_pd(vlarge, vhalo);

        // Pack the two partner windows in that same order. There is no
        // product between different field pairs or different distances.
        const v2d vright = simde_mm_set_pd(right1[node], right0[node]);

        // Multiply each cluster amplitude by its own partner window.
        const v2d vproduct = simde_mm_mul_pd(vleft, vright);

        // The two pairs share this shell's positive dchi/f_K^2 measure.
        const v2d vmeasure = simde_mm_set1_pd(measure[node]);

        // Accumulate each integral separately. Native FMA rounds the
        // product-plus-addition once; radial summation order is unchanged.
        vtotal = simde_mm_fmadd_pd(vproduct, vmeasure, vtotal);
      }

      double result[2]; // one integrated scalar per lane

      // Copy low/high lane sums to result[0/1]; storeu accepts an ordinary
      // stack array without a special vector-alignment requirement.
      simde_mm_storeu_pd(result, vtotal);
      spectra[first_pair][index] = factor[0]*result[0];
      if (first_pair+1 < npair) {
        spectra[first_pair+1][index] = factor[1]*result[1];
      }
    }
  }

  free(weighted);
  free(pairs);
}
