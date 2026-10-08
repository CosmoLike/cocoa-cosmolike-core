#include <math.h>
#include <stdlib.h>

#include "spectra_cluster_cov.h"
#include "cosmolike/basics.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

// SIMD (single instruction, multiple data) applies one instruction to
// several numbers at once. A v2d, SIMDe's simde__m128d, holds two
// doubles; each position is called a lane, lane 0 the low and lane 1 the
// high double. In the q_c b_c loop the lanes hold two adjacent radial
// shells. In the projection loop they hold two different field pairs:
// each lane is a complete radial sum, visiting the nodes in order.
//
// simde_mm_fmadd_pd(a, b, c) returns a*b + c in each lane. With an FMA
// instruction (x86 built with FMA support, or arm64 NEON) the product is
// kept exact and only the sum is rounded, like the C function fma(a, b,
// c). Without one, SIMDe's portable code rounds the product and the sum
// separately.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Add all cluster spectra to a galaxy/shear covariance calculation.
//
// A normalized cluster window q_c obeys integral q_c dchi = 1. For an
// abundance-selected catalog, q_c is proportional to f_K^2 phi_c n_c:
// shell volume times selection probability times comoving abundance.
// It does not include the cluster bias b_c. Keeping those factors
// separate is essential because a cluster's own mass profile is not
// multiplied by its large-scale bias.
//
// The supported three-dimensional mean model is
//   P_cc' = b_c b_c' P_NL,
//   P_cg  = b_c b_g P_NL,
//   P_cm  = b_c P_NL + P_cm^1h.
// P_cm^1h is the matter of the selected halos themselves: the mean of
// (M/rho_m) u(k|M) over the halos selected into one richness category,
// J01/n in moments_cluster_cov.c. It is the cluster's own mass, not
// matter correlated with it through large-scale structure, so it carries
// no halo bias. See To et al. (2021), arXiv:2008.10757, Secs.
// 4.1.2--4.1.3 and Eqs. 20--21.
// The galaxy model is linear bias on these scales; it has no additional
// cluster-galaxy one-halo term. No claim of a complete joint halo model
// follows from using this mean-spectrum approximation.
//
// Limber projection converts these spectra into angular correlations.
// Only structures at the same distance correlate, and the shell at chi
// supplies each P at k = (ell+1/2)/f_K(chi) and that shell's redshift:
//   C_cc' = integral dchi q_c q_c' b_c b_c' P_NL / f_K^2,
//   C_cg  = integral dchi q_c b_c W_g P_NL / f_K^2,
//   C_cs  = F_ell integral dchi q_c W_s (b_c P_NL + P_cm^1h)/f_K^2.
// Here W_g already contains galaxy bias, W_s is the lensing window,
// and F_ell = sqrt[(ell-1)ell(ell+1)(ell+2)]/(ell+1/2)^2 is the core
// harmonic shear convention: the spin-2 factor that turns the lensing
// potential into shear, divided by the (ell+1/2)^2 with which the Limber
// window W_s turns that potential into convergence. F_ell tends to 1 at
// large ell. Real-space operators need the same extra source conversion
// as the galaxy/shear covariance. Noise is separate.
//
// This boundary takes supplied arrays so selection and numerical tables
// are explicit. It assumes zero IA, magnification and RSD; adding any
// of them requires separate field contributions, not merely changing W_s.
// No data-vector pair cut is imposed. Gaussian covariance can require
// cross-redshift and crossed cluster-source spectra absent from the mean.
// Two exclusive observed categories c != c' still have a nonzero signal
// C_cc' wherever their true-redshift windows overlap, because both trace
// the same matter. Their catalog noise, added elsewhere, is diagonal:
// 1/nbar_c per steradian for c = c' only, since no cluster carries two
// labels. The caller must check the noise-inclusive field matrix and
// final total covariance for positivity; this routine never repairs a
// negative mode.
//
// The calculation has two stages. First enumerate pairs and combine
// q_c with b_c once per shell. Then distribute (ell,pair-group) outputs
// over OpenMP workers. SIMDe integrates two different spectra together,
// sharing matter power and radial measure without changing either sum's
// order. Scratch is grouped, caller inputs are immutable, and no global
// state or lazy table is accessed. Calls at this entry point are serial.
//
// Parameters (length unit L throughout):
//   nell     - number of multipoles, >= 1
//   ell      - [nell] multipoles, ell >= 2 so that F_ell is real
//   nnode    - number of common radial nodes chi_j, >= 1
//   distance - [nnode] f_K(chi_j) in L, positive
//   dchi     - [nnode] radial quadrature weights in L, positive
//   nbase    - number of galaxy plus source fields
//   nlens    - base fields 0..nlens-1 are galaxies, the rest sources
//   base     - [nbase][nnode] W_g (bias included) or W_s, in L^-1
//   ncluster - number of observed cluster categories c
//   window   - [ncluster][nnode] normalized q_c, in L^-1
//   bias     - [ncluster][nnode] selected cluster bias b_c
//   power    - [nell][nnode] P_NL at ((ell+1/2)/f_K, z(chi_j)), in L^3
//   p1h      - [nrichness][nell][nnode] P_cm^1h at the same points, L^3
//   richness - [ncluster] p1h row of each cluster category
//
// Output (every entry is written; no catalog noise is included):
//   spectra  - [npair][nell] dimensionless C_ell. Rows
//              0..ncluster*nbase-1 hold (cluster, base) pairs,
//              cluster-major. The remaining ncluster*(ncluster+1)/2 rows
//              hold the cluster upper triangle (0,0),(0,1),...,(1,1),...
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

  // A matter perturbation delta_m changes the selected cluster density
  // by b_c delta_m, so the large-scale cluster window is q_c b_c.
  // Precompute it because every multipole and every partner field reuses
  // it. One iteration forms q_c b_c of one category at two adjacent
  // shells: SIMD lane 0 holds node and lane 1 node+1, never added. node
  // advances by two; an odd final node takes the scalar branch. Every
  // (cluster, node pair) writes its own entries, so the work is
  // independent across workers.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int cluster=0; cluster<ncluster; cluster++) {
    for (int node=0; node<nnode; node+=2) {
      const double* restrict q = window[cluster];
      const double* restrict b = bias[cluster];
      double* restrict qb = weighted[cluster];

      // scalar: for the two shells j = node and j = node+1,
      //   qb[j] = q[j]*b[j];
      // The radial probability q places the catalog in distance; bias b
      // sets its response to matter. SIMD retains one shell in each lane
      // and performs the same single rounded product, so each lane equals
      // the scalar result bitwise.
      if (node+1 < nnode) {
        // loadu puts q[node] = q_c(chi_node) in lane 0 and q[node+1] in
        // lane 1. It requires no vector-aligned address, but both
        // elements must exist, which node+1 < nnode guarantees.
        const v2d vwindow = simde_mm_loadu_pd(q+node);

        // loadu puts b[node] = b_c(chi_node) in lane 0 and b[node+1] in
        // lane 1: the bias at the same shells, in the same lane order.
        const v2d vbias = simde_mm_loadu_pd(b+node);

        // mul_pd multiplies lane by lane: q_c b_c at shell node (lane 0)
        // and at node+1 (lane 1). No product mixes two shells.
        const v2d vweighted = simde_mm_mul_pd(vwindow, vbias);

        // storeu writes lane 0 to qb[node] and lane 1 to qb[node+1] in
        // the scratch row weighted[cluster]; no alignment is required.
        simde_mm_storeu_pd(qb+node, vweighted);
      } else {
        // The unmatched final shell (node = nnode-1) uses the same one
        // multiplication.
        qb[node] = q[node]*b[node];
      }
    }
  }

  // Every cluster is paired with every galaxy and source. Indexing the
  // right side by nbase+cluster distinguishes cluster-cluster pairs.
  // pairs[0][p] is the left cluster category of output row p, and
  // pairs[1][p] its partner: a base field below nbase, or nbase plus a
  // cluster category. The order is the output row order of the header.
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

  // In Limber's approximation each shell supplies correlations of
  // structures at the same distance: the shell at chi contributes its
  // two field windows times the 3D power at k = (ell+1/2)/f_K(chi),
  // weighted by the measure dchi/f_K^2. The large-scale term uses
  // q_c b_c P for every pair. Only a source partner receives the
  // cluster's own mass-profile term q_c P_cm^1h.
  //
  // One iteration computes the spectra of two consecutive pairs at one
  // multipole: first_pair in SIMD lane 0 and next_pair in lane 1. Each
  // lane is a complete radial sum over all nodes in increasing order;
  // the two lanes are never added. first_pair advances by two. When npair
  // is odd, the last iteration repeats pair npair-1 in lane 1, and that
  // duplicate is discarded when storing. Collapse both output indices so
  // a short multipole array still offers enough independent spectra for
  // eight workers. Neither threads nor lanes share a reduction, so the
  // result does not depend on the thread count.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int index=0; index<nell; index++) {
    for (int first_pair=0; first_pair<npair; first_pair+=2) {
      // Lane 1's pair: the next row, or a repeat of the last row when
      // first_pair is already the final pair of an odd npair.
      const int next_pair = first_pair+1 < npair ? first_pair+1 : npair-1;
      const int pair_id[2] = {first_pair, next_pair};
      const double* q[2];      // unbiased left cluster windows
      const double* qb[2];     // biased left cluster windows
      const double* right[2];  // partner windows, including its bias
      const double* own[2];    // one-halo profile, or NULL for density
      double factor[2];        // shear transfer, or unity for density
      const double* restrict pk = power[index];

      // Fix each lane's physical role before visiting shells. A cluster
      // partner (field >= nbase) reads q_c' b_c'; a galaxy reads its
      // biased W_g; a source (nlens <= field < nbase) reads W_s and
      // additionally receives the selected halo's profile P_cm^1h of the
      // left cluster's richness row and the shear factor F_ell.
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
      // lookups. restrict promises that no pointer used in this iteration
      // modifies these rows. The node loop only reads them; the single
      // write per lane, to the output after the loop, touches disjoint
      // memory.
      const double* restrict q0 = q[0];
      const double* restrict q1 = q[1];
      const double* restrict qb0 = qb[0];
      const double* restrict qb1 = qb[1];
      const double* restrict right0 = right[0];
      const double* restrict right1 = right[1];
      const double* restrict own0 = own[0];
      const double* restrict own1 = own[1];

      // scalar: for either pair lane s = 0, 1 (each line rounds once;
      // only the fma is fused),
      //   total = 0;
      //   for (int node=0; node<nnode; node++) {
      //     halo = own[s] == NULL ? 0 : own[s][node];
      //     large = qb[s][node]*pk[node];
      //     one_halo = q[s][node]*halo;
      //     left = large+one_halo;
      //     product = left*right[s][node];
      //     total = fma(product, measure[node], total);
      //   }
      //   spectra[pair_id[s]][index] = factor[s]*total;
      // The halo's own mass contributes only for source partners. Its
      // weight is q, while the correlated matter term has weight q*b.
      // SIMD carries the two projected spectra in distinct lanes; the
      // shear conversion factor is applied when storing the results.
      // setzero sets both lanes to 0.0: lane 0 starts the integral of
      // first_pair and lane 1 that of next_pair. It does not mix lanes.
      v2d vtotal = simde_mm_setzero_pd();

      // One node adds one shell's contribution, multiplied by its measure
      // dchi/f_K^2, to each lane's sum. Radial order is fixed for each
      // sum, so changing the thread count cannot change the order of
      // floating-point additions.
      for (int node=0; node<nnode; node++) {
        // scalar: halo = own[s] == NULL ? 0 : own[s][node]. A density
        // partner has no own-profile term; a source partner reads
        // P_cm^1h of its lane's cluster category at this shell.
        const double halo0 = own0 == NULL ? 0.0 : own0[node];
        const double halo1 = own1 == NULL ? 0.0 : own1[node];

        // scalar: large = qb[s][node]*pk[node]

        // set_pd(high, low) takes the high lane first: lane 0 gets
        // qb0[node], the q_c b_c of first_pair's cluster at this shell,
        // and lane 1 gets qb1[node], that of next_pair's cluster.
        const v2d vbiased = simde_mm_set_pd(qb1[node], qb0[node]);

        // set1 copies pk[node] = P_NL((ell+1/2)/f_K, z) of this shell
        // into both lanes: both pairs sample the same matter power.
        const v2d vpower = simde_mm_set1_pd(pk[node]);

        // mul_pd, lane by lane: q_c b_c P_NL, the large-scale cluster
        // amplitude of each pair, rounded once.
        const v2d vlarge = simde_mm_mul_pd(vbiased, vpower);

        // scalar: one_halo = q[s][node]*halo

        // set_pd(high, low): lane 0 gets q0[node] and lane 1 q1[node],
        // the unbiased windows q_c of the two pairs' clusters.
        const v2d vwindow = simde_mm_set_pd(q1[node], q0[node]);

        // set_pd(high, low): lane 0 gets halo0 and lane 1 halo1. Density
        // partners have zero here. Source partners receive their selected
        // richness profile, which can differ between the lanes.
        const v2d vprofile = simde_mm_set_pd(halo1, halo0);

        // mul_pd, lane by lane: q_c P_cm^1h. The cluster's own mass is
        // weighted by q_c, without b_c; it is zero for density partners.
        const v2d vhalo = simde_mm_mul_pd(vwindow, vprofile);

        // scalar: left = large+one_halo

        // add_pd, lane by lane: q_c (b_c P_NL + P_cm^1h), the one-halo
        // and biased matter terms of each pair added within that pair
        // only, rounded once.
        const v2d vleft = simde_mm_add_pd(vlarge, vhalo);

        // scalar: product = left*right[s][node]

        // set_pd(high, low): lane 0 gets right0[node] and lane 1
        // right1[node], each pair's partner window at this shell:
        // q_c' b_c' for a cluster, W_g for a galaxy, W_s for a source.
        // There is no product between different field pairs or
        // different distances.
        const v2d vright = simde_mm_set_pd(right1[node], right0[node]);

        // mul_pd: each cluster amplitude times its own partner window,
        // rounded once.
        const v2d vproduct = simde_mm_mul_pd(vleft, vright);

        // scalar: total = fma(product, measure[node], total)

        // set1 copies measure[node] = dchi/f_K^2 of this shell into both
        // lanes; the two pairs share this positive Limber measure.
        const v2d vmeasure = simde_mm_set1_pd(measure[node]);

        // fmadd_pd(a, b, c) = a*b + c: lane s becomes total_s =
        // fma(product_s, measure[node], total_s). Each lane adds this
        // shell to its own pair's integral, rounded once with an FMA
        // instruction (see the note at v2d). Radial summation order is
        // unchanged.
        vtotal = simde_mm_fmadd_pd(vproduct, vmeasure, vtotal);
      }

      double result[2]; // one integrated scalar per lane

      // storeu copies lane 0, the integral of first_pair, to result[0]
      // and lane 1, that of next_pair, to result[1]. It accepts an
      // ordinary stack array without a vector-alignment requirement.
      simde_mm_storeu_pd(result, vtotal);

      // Apply each pair's transfer factor, F_ell for a source partner and
      // 1 otherwise, and store. When npair is odd the final group has
      // next_pair = first_pair; skipping lane 1 then discards that
      // duplicate instead of writing a nonexistent row.
      spectra[first_pair][index] = factor[0]*result[0];
      if (first_pair+1 < npair) {
        spectra[first_pair+1][index] = factor[1]*result[1];
      }
    }
  }

  free(weighted);
  free(pairs);
}
