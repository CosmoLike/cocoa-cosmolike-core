#include <math.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "halo_cov.h"
#include "cosmolike/basics.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/halo.h"
#include "cosmolike/structs.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

// A SIMD vector applies the same operation to two doubles in positions
// called lanes. Here each lane owns a different k or (K,Q) mass integral.
// A fused multiply-add (FMA) evaluates a*b+c with one rounding when
// supported directly by the processor. This differs from rounding a*b
// first and then adding c; the calls below retain the chosen operations.
// Unaligned loads/stores accept addresses that are not multiples of 16
// bytes. They still require two valid adjacent doubles in the array.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Build the mass moments shared by halo-model SSC and connected covariance.
//
// A halo of mass M contributes its normalized Fourier profile u(k|M).
// Correlating mu density legs in the same halo gives one profile per leg:
//
//   I_mu^beta(k1,...,kmu;a) = integral dlnM (dn/dlnM)
//                            b_beta(M,a) (M/rho_cb)^mu product_i u(ki|M),
//   b_0 = 1, b_1 = linear halo bias.
//
// This is the halo-moment definition of Takada & Hu (2013), Eq. 25,
// arXiv:1302.6994, applied to the cb field. Halos count collapsed cold
// matter plus baryons, so their peak height and abundance use
//
//   nu = 1.686/sigma_cb(M,a),
//   dn/dlnM = (rho_cb/M) f(nu,a) nu dlnnu/dlnM.
//
// See Ichiki & Takada (2012), arXiv:1108.4688, and Castorina et al.
// (2014), arXiv:1311.1212, for the cold-field mass-function prescription.
// There is no total-matter halo-statistics switch. sigma2, fnu, hb1nu,
// conc and u_nfw_c are the public core readers. Concentration inherits
// D_cb(M,a)=sigma_cb(M,a)/sigma_cb(M,1); the M200m radius still uses the
// core's 200*rho_m mass definition. Reusing readers does not change them.
//
// The moment's M/rho_cb factor makes this a cb-density moment. A later
// total-matter observable needs its cb fractions and neutrino cross terms;
// do not silently interpret this as a full massive-neutrino matter model.
// For the pinned massless test configuration cb and total matter coincide.
//
// Only I11 receives an unresolved-low-mass completion. Define
// A = 1 - sum_mass dlnM (dn/dlnM) b M/rho_cb on this SAME quadrature.
// Add A*u(k|M_min) to I11. Thus I11(0)=1 at every a, independent of the
// finite lower mass cutoff. Higher moments are left uncorrected, as in
// the core halo-model convention (Mead et al. 2020, arXiv:2005.00009,
// Appendix A). No division by a bias-normalization table is applied.
//
// Computation proceeds in four visible stages:
// 1. Build one mass rule shared by all a and k, with covariance-owned
//    panels and node counts. The core Ntable settings are never modified.
// 2. Compute each mass/scale-factor abundance, bias and concentration once.
//    Store the five mass weights needed by all later sums together.
// 3. Compute u(k|M) once per (a,k,M), shared by every pair involving k.
// 4. Sum I11 and the five pair moments. Two SIMDe lanes own independent
//    results and keep increasing mass-node order, without a thread reduction.
//
// Parameters:
//   a[na], k[na][nk] - current scale factors and their wavenumber grids
//   lnm_edges[npanel+1], nquad - mass panels and GL nodes per panel
//   i11 - caller-owned [na][nk], dimensionless
//   moments - caller-owned [5][na][npair], npair=nk*(nk+1)/2
// Pair order is (0,0),(0,1),...,(1,1),... in the supplied k grid.
// Roles are I02(K,Q), I12(K,Q), I13(K,Q,Q), I13(K,K,Q), I04(K,K,Q,Q).
// Their dimensions are L^3,L^3,L^6,L^6,L^9 with L=c/H0.
// Passing NULL for moments requests I11 alone. The SSC logarithmic
// slope needs I11 at two nearby wavenumbers, but its higher moments
// are already known at the central wavenumber. In that case no pair
// table is allocated and no two-, three- or four-profile sum is done.
//
// Calls are serial at the entry point. Lazy core tables are warmed before
// workers read them. There is no covariance static cache; grouped scratch
// is released before return. Writable rows cannot alias inputs or each
// other. Public reader table precision remains part of the supplied core
// state and must be included in the eventual covariance accuracy scan.
// ---------------------------------------------------------------------------
void halo_moments_cov(
    const int na,                  // scale-factor count
    const double* a,              // scale factors
    const int nk,                  // wavenumber count per scale factor
    const double* const* k,       // wavenumber rows
    const int npanel,              // logarithmic mass-panel count
    const double* lnm_edges,      // increasing logarithmic mass edges
    const int nquad,               // quadrature nodes per panel
    double* const* i11,            // one-profile biased moment
    double** const* moments        // five roles, or NULL for I11 only
  )
{
  if (na < 1
      || nk < 1
      || npanel < 1
      || like.halo_model[3] != HALO_PROFILE_NFW) {
    log_fatal("halo_moments_cov needs positive sizes and NFW profiles");
    exit(1);
  }
  if (nquad != 64
      && nquad != 96
      && nquad != 128
      && nquad != 256
      && nquad != 512
      && nquad != 1024) {
    log_fatal("halo_moments_cov: unsupported tabulated mass rule %d", nquad);
    exit(1);
  }
  // Check each scale factor and its k grid before any shared tables are
  // constructed. All samples must lie in the public readers' domain.
  for (int row=0; row<na; row++) {
    if (!isfinite(a[row])
        || a[row] < limits.a_min
        || a[row] >= 1.0) {
      log_fatal("halo_moments_cov: a[%d] must be in [%g,1)",
                row, limits.a_min);
      exit(1);
    }
    for (int node=0; node<nk; node++) {
      if (!isfinite(k[row][node])
          || k[row][node] < 0.0) {
        log_fatal("halo_moments_cov needs finite k >= 0; row %d, node %d",
                  row, node);
        exit(1);
      }
    }
  }
  const double lnm_min = log(limits.halo_m[RANGE_MIN]);
  const double lnm_max = log(limits.halo_m[RANGE_MAX]);

  for (int edge=0; edge<=npanel; edge++) {
    if (!isfinite(lnm_edges[edge])
        || lnm_edges[edge] < lnm_min
        || lnm_edges[edge] > lnm_max
        || (edge > 0
            && lnm_edges[edge] <= lnm_edges[edge-1])) {
      log_fatal("halo_moments_cov needs increasing lnM edges inside "
                "the core sigma range [%g,%g]", lnm_min, lnm_max);
      exit(1);
    }
  }

  // --- 1. COMMON MASS QUADRATURE AND PAIR ORDER ---

  const int nmass = npanel*nquad;
  const int npair = moments == NULL ? 0 : nk*(nk+1)/2; // requested pairs
  const double rho_cb = cosmology.rho_crit*omega_halo_field();

  if (!isfinite(rho_cb)
      || rho_cb <= 0.0) {
    log_fatal("halo_moments_cov needs a positive initialized cb density");
    exit(1);
  }

  // mass roles are M, the volume V=M/rho_cb, and dlnM/V. The last role
  // supplies the density prefactor of dn/dlnM before f(nu)*nu*dlnnu/dlnM.
  const double mass_min = exp(lnm_edges[0]);
  double** mass = (double**) malloc2d(3, nmass);
  int** pairs = NULL; // no pair indices are needed for I11 alone

  if (npair > 0) {
    pairs = (int**) malloc2d_int(2, npair);
  }

  // Map the same Gaussian integration rule onto each logarithmic interval.
  gsl_integration_glfixed_table* rule = malloc_gslint_glfixed(nquad);

  // Each mass panel contributes nquad samples to the common integration
  // rule. Save their physical masses, cb volumes and number-density weights.
  for (int panel=0; panel<npanel; panel++) {
    // Map one Gaussian node into this panel and store its three mass-only
    // quantities at the global index used by every redshift and k row.
    for (int node=0; node<nquad; node++) {
      const int index = panel*nquad+node;
      double lnm;      // logarithmic mass abscissa
      double measure;  // positive dlnM quadrature measure

      gsl_integration_glfixed_point(lnm_edges[panel], lnm_edges[panel+1],
                                    node, &lnm, &measure, rule);

      // Store physical mass, cb volume, and the weighted number prefactor
      // separately: subsequent redshift rows share these same mass nodes.
      mass[0][index] = exp(lnm);
      mass[1][index] = mass[0][index]/rho_cb;
      mass[2][index] = measure/mass[1][index];
    }
  }

  gsl_integration_glfixed_table_free(rule);

  // Enumerate the upper triangle once. Each output pair can then locate
  // its two profile rows directly, without searching the wavenumber grid.
  if (npair > 0) {
    int pair = 0; // position in the common upper-triangle pair list

    for (int first=0; first<nk; first++) {
      for (int second=first; second<nk; second++) {
        pairs[0][pair] = first;
        pairs[1][pair] = second;
        pair++;
      }
    }
  }

  // --- 2. HALO STATISTICS AT EACH (a,M), WITH SERIAL TABLE WARMUP ---

  // Public readers may build tables on their first call. Complete that
  // setup here so the parallel loops below only read initialized tables.
  (void) sigma2(mass_min, a[0]);
  (void) dlognudlogm(mass_min, a[0]);
  (void) fnu(1.0, a[0]);
  (void) hb1nu(1.0, a[0]);
  (void) u_nfw_c(conc(mass_min, a[0]), 1.0, mass_min, a[0]);

  // The weights already include dlnM. Their role is the power of volume
  // and whether a bias is present: bV, V^2, bV^2, bV^3, V^4 times dn.
  double*** weights = (double***) malloc3d(5, na, nmass);
  double** concentration = (double**) malloc2d(na, nmass+1);
  double* completion = malloc(sizeof(double)*na);

  if (completion == NULL) {
    log_fatal("halo_moments_cov: cannot allocate completion weights");
    exit(1);
  }

  // A halo moment sums contributions from halos of every mass. Each mass
  // interval contributes its expected number density times profile, volume
  // and bias factors. The abundance, volume and bias do not depend on k,
  // so prepare their products once for all later profile integrals.
  // Every (scale factor, mass) weight is independent. Distribute both
  // indices so a call with only one scale factor still has enough work
  // for eight or ten threads. The completion sum below is separate:
  // sharing its additions among workers would change their rounding order.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int row=0; row<na; row++) {
    // At each mass, turn the peak height into a weighted halo abundance.
    // Combine it with bias and volume powers for the later profile sums.
    // Each worker writes a different node; no shared sum is updated here.
    for (int node=0; node<nmass; node++) {
      const double m = mass[0][node];
      const double volume = mass[1][node];
      const double nu = 1.686/sqrt(sigma2(m, a[row]));
      const double bias = hb1nu(nu, a[row]);

      // number is (dn/dlnM)*dlnM at this mass and scale factor. Attach
      // powers of the halo volume and bias to form the five moment weights.
      const double number = mass[2][node]*fnu(nu, a[row])*nu
                            *dlognudlogm(m, a[row]);
      const double volume2 = volume*volume;

      weights[0][row][node] = number*bias*volume;
      weights[1][row][node] = number*volume2;
      weights[2][row][node] = number*bias*volume2;
      weights[3][row][node] = weights[2][row][node]*volume;
      weights[4][row][node] = weights[1][row][node]*volume2;

      // Every wavenumber at this mass uses the same concentration.
      concentration[row][node] = conc(m, a[row]);
    }
  }

  // At k=0 every normalized halo profile is one, so the biased mass
  // weights alone give the resolved part of I11(0). Their integral must
  // equal one when all matter is included. Assign the missing fraction
  // to the minimum-mass profile. One worker owns the entire sum at each
  // scale factor and adds masses in order, preserving the original result.
  #pragma omp parallel for schedule(static)
  for (int row=0; row<na; row++) {
    double resolved = 0.0; // resolved biased mass fraction at this a

    for (int node=0; node<nmass; node++) {
      resolved += weights[0][row][node];
    }

    // Assign the missing k=0 weight to a profile at the minimum mass.
    completion[row] = 1.0-resolved;
    concentration[row][nmass] = conc(mass_min, a[row]);
  }

  // --- 3. SHARED PROFILES, INCLUDING THE M_min COMPLETION PROFILE ---

  double*** profile = (double***) malloc3d(na, nk, nmass+1);

  // At a fixed (a,k), every halo mass has an independent NFW profile.
  // Whole mass rows let the worker reuse a and k. If fewer rows exist
  // than workers, divide each row into contiguous mass chunks so a small
  // request can still occupy the thread team. Larger requests keep one
  // chunk per row and avoid a parallel-loop index calculation per mass.
  int threads = 1; // available workers; serial builds use one
  #ifdef _OPENMP
  threads = omp_get_max_threads();
  #endif

  const int nprofile = na*nk; // independent (scale factor, wavenumber) rows
  const int nchunk = (threads+nprofile-1)/nprofile; // chunks per profile row

  // For example, two profile rows and eight workers give four chunks
  // per row: all eight workers can evaluate different masses. With at
  // least eight rows, each task instead evaluates one complete mass row.
  // No mass sum is split here; the later moment integrals keep their order.
  #pragma omp parallel for collapse(3) schedule(static)
  for (int row=0; row<na; row++) {
    // Each k uses this scale factor's shared mass/concentration samples.
    for (int index=0; index<nk; index++) {
      // Chunks cover disjoint parts of the same profile row. A chunk stops
      // before end; its neighbor starts there, so each mass is written once.
      for (int chunk=0; chunk<nchunk; chunk++) {
        const int begin = (nmass+1)*chunk/nchunk; // first owned mass slot
        const int end = (nmass+1)*(chunk+1)/nchunk; // first unowned mass slot

        // Evaluate each mass in the owned interval. The final physical
        // slot belongs to the minimum-mass completion profile; it is
        // included in the partition just like every resolved halo mass.
        for (int node=begin; node<end; node++) {
          const double m = node < nmass ? mass[0][node] : mass_min;
          double value = 1.0; // normalized profile at zero wavenumber

          if (k[row][index] > 0.0) {
            value = u_nfw_c(concentration[row][node], k[row][index], m,
                            a[row]);
          }

          profile[row][index][node] = value;
        }
      }
    }
  }

  // --- 4a. I11: ONE PROFILE AND ONE BIAS, PLUS UNRESOLVED MASS ---

  // I11 describes how the halo population contributes to a large-scale
  // density fluctuation. A halo supplies its profile u(k|M), weighted by
  // its mass fraction and bias; integrating over mass gives I11(k).
  // Add the unresolved fraction through the chosen minimum-mass profile.
  // For one scale factor, each worker computes two such integrals. SIMD
  // shares their mass weights, but each lane keeps a different k and its
  // complete mass sum. Adding lanes would mix different physical scales.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int row=0; row<na; row++) {
    // Integrate two wavenumbers at this scale factor and write their I11
    // values. SIMD lanes 0/1 own index/next, including their completion terms.
    for (int index=0; index<nk; index+=2) {
      // Lane 0 owns index, lane 1 owns next. Repeat the last valid profile
      // for an odd nk, then discard that duplicate output below.
      const int next = index+1 < nk ? index+1 : index;
      const double* restrict u0 = profile[row][index];
      const double* restrict u1 = profile[row][next];
      const double* restrict weight = weights[0][row];

      // Scalar equivalent for either wavenumber j=index,next:
      //   sum = 0;
      //   for (int node=0; node<nmass; node++) {
      //     sum = fma(profile[row][j][node], weight[node], sum);
      //   }
      //   i11[row][j] = sum+completion[row]*profile[row][j][nmass];
      // The weighted halo profiles describe the resolved mass population;
      // the last term supplies the missing low-mass response. SIMD keeps
      // these complete mass integrals separate for the two wavenumbers.
      // Start the two I11 mass sums at zero, without mixing wavenumbers.
      v2d vsum = simde_mm_setzero_pd();

      // For each mass, update both k-specific sums with u(k|M)*weight.
      // SIMD shares the scalar weight but keeps the two integrals separate.
      for (int node=0; node<nmass; node++) {
        // set_pd takes the high lane first. Pack [u(index,M),u(next,M)]
        // into lanes 0 and 1 at this common mass node.
        const v2d vu = simde_mm_set_pd(u1[node], u0[node]);

        // Copy the common (dn/dlnM)*dlnM*b*M/rho_cb weight into both lanes.
        const v2d vw = simde_mm_set1_pd(weight[node]);

        // Each k adds u(k|M)*weight to its own I11. fmadd fuses product
        // and addition into one rounding on native FMA hardware.
        vsum = simde_mm_fmadd_pd(vu, vw, vsum);
      }

      double result[2];

      // Copy I11 sums for index/next into result[0/1]. storeu permits
      // this two-double stack array without special vector alignment.
      simde_mm_storeu_pd(result, vsum);

      // Add the unresolved-mass contribution at each physical k only once.
      i11[row][index] = result[0]+completion[row]*u0[nmass];
      if (index+1 < nk) {
        i11[row][next] = result[1]+completion[row]*u1[nmass];
      }
    }
  }

  // --- 4b. FIVE PAIR MOMENTS FROM THE SAME FOUR PROFILE READS ---

  // Several factors of the density field can come from the same halo.
  // Each factor supplies a profile u and a volume M/rho_cb; a moment sums
  // their product over the halo abundance, with bias when required. For a
  // fixed (K,Q), the five moments reuse the same two profiles but combine
  // them in different powers. Compute them together to reuse those reads.
  // Each worker handles two pairs at one scale factor. SIMD lane 0 owns
  // the first pair's mass integrals and lane 1 the second pair's; they
  // remain separate because they describe different covariance entries.
  // An I11-only request has no pairs: avoid starting an empty team.
  #pragma omp parallel for collapse(2) schedule(static) if(npair > 0)
  for (int row=0; row<na; row++) {
    // At this scale factor, compute and store five moments for each pair
    // group. SIMD handles two pairs together without adding their results.
    for (int index=0; index<npair; index+=2) {
      // Each lane owns one full (K,Q) pair, not one of the two magnitudes.
      // An odd final pair is repeated for safe reads and written only once.
      const int next = index+1 < npair ? index+1 : index;
      const double* restrict uk0 = profile[row][pairs[0][index]];
      const double* restrict uq0 = profile[row][pairs[1][index]];
      const double* restrict uk1 = profile[row][pairs[0][next]];
      const double* restrict uq1 = profile[row][pairs[1][next]];
      // Scalar updates at one mass node for either pair j=index,next:
      //   uk = profile[row][pairs[0][j]][node];
      //   uq = profile[row][pairs[1][j]][node];
      //   product = uk*uq;
      //   sum[0] = fma(product, weights[1][row][node], sum[0]);
      //   sum[1] = fma(product, weights[2][row][node], sum[1]);
      //   sum[2] = fma(product*uq, weights[3][row][node], sum[2]);
      //   sum[3] = fma(product*uk, weights[3][row][node], sum[3]);
      //   sum[4] = fma(product*product, weights[4][row][node], sum[4]);
      // Initialize all five sums to zero, then visit every mass in order.
      // Each extra profile represents another density factor in one halo.
      // SIMD carries two copies of these five sums for independent pairs.
      v2d vsums[5];

      for (int role=0; role<5; role++) {
        // Initialize this moment's two independent pair sums to [0,0].
        vsums[role] = simde_mm_setzero_pd();
      }

      // At each mass, form the needed two-, three- and four-leg profile
      // products and add their weighted contributions to all five moments.
      // Lanes remain separate pairs; dn denotes the weighted abundance
      // (dn/dlnM)*dlnM and V denotes M/rho_cb in the comments below.
      for (int node=0; node<nmass; node++) {
        // --- PROFILES AT THIS MASS, WITH PAIRS KEPT IN SEPARATE LANES ---

        // set_pd takes lane 1 first: vk holds the K-profile for index
        // in lane 0 and for next in lane 1.
        const v2d vk = simde_mm_set_pd(uk1[node], uk0[node]);

        // Pack the matching Q-profiles in the same index/next lane order;
        // set_pd again receives the high-lane value first.
        const v2d vq = simde_mm_set_pd(uq1[node], uq0[node]);

        // Multiply matching lanes to form u(K)*u(Q) for each pair.
        const v2d vproduct = simde_mm_mul_pd(vk, vq);

        // Append another Q leg: u(K)*u(Q)^2 in each pair lane.
        const v2d vkqq = simde_mm_mul_pd(vproduct, vq);

        // Append another K leg: u(K)^2*u(Q) in each pair lane.
        const v2d vkkq = simde_mm_mul_pd(vproduct, vk);

        // Square the two-profile product to get u(K)^2*u(Q)^2 per pair.
        const v2d vkkqq = simde_mm_mul_pd(vproduct, vproduct);

        // --- MASS WEIGHTS SHARED BY THE TWO PAIRS ---

        // Copy the unbiased two-leg weight dn*V^2 into both lanes.
        const v2d vw02 = simde_mm_set1_pd(weights[1][row][node]);

        // Copy the biased two-leg weight dn*b*V^2 into both lanes.
        const v2d vw12 = simde_mm_set1_pd(weights[2][row][node]);

        // Copy the biased three-leg weight dn*b*V^3 into both lanes.
        const v2d vw13 = simde_mm_set1_pd(weights[3][row][node]);

        // Copy the unbiased four-leg weight dn*V^4 into both lanes.
        const v2d vw04 = simde_mm_set1_pd(weights[4][row][node]);

        // --- ADD THIS MASS NODE TO EACH OF THE FIVE MOMENTS ---

        // I02 += u(K)*u(Q)*dn*V^2 in each lane. fmadd uses a single
        // native-FMA rounding for the product and running-sum addition.
        vsums[0] = simde_mm_fmadd_pd(vproduct, vw02, vsums[0]);

        // I12 += u(K)*u(Q)*dn*b*V^2 for each pair, with one fused rounding
        // on native FMA hardware, preserving the order of mass nodes.
        vsums[1] = simde_mm_fmadd_pd(vproduct, vw12, vsums[1]);

        // I13(K,Q,Q) += u(K)*u(Q)^2*dn*b*V^3. fmadd accumulates each
        // lane separately with one native-FMA rounding.
        vsums[2] = simde_mm_fmadd_pd(vkqq, vw13, vsums[2]);

        // I13(K,K,Q) += u(K)^2*u(Q)*dn*b*V^3. fmadd fuses product and
        // addition once per lane on native FMA hardware.
        vsums[3] = simde_mm_fmadd_pd(vkkq, vw13, vsums[3]);

        // I04 += u(K)^2*u(Q)^2*dn*V^4. fmadd retains the two independent
        // mass sums and a single native-FMA rounding for each update.
        vsums[4] = simde_mm_fmadd_pd(vkkqq, vw04, vsums[4]);
      }

      // Store each moment's two pair results, discarding a duplicated
      // second lane when the total number of pairs is odd.
      for (int role=0; role<5; role++) {
        double result[2];

        // Copy this moment's index/next sums into result[0/1]. storeu
        // requires two valid doubles but no vector-aligned stack address.
        simde_mm_storeu_pd(result, vsums[role]);

        moments[role][row][index] = result[0];
        if (index+1 < npair) {
          moments[role][row][next] = result[1];
        }
      }
    }
  }

  // The caller keeps only i11 and moments; all construction tables end here.
  free(profile);
  free(completion);
  free(concentration);
  free(weights);
  free(pairs);
  free(mass);
}
