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

// SIMD (single instruction, multiple data) applies one operation to
// several numbers at once. A v2d holds two doubles, in positions called
// lanes 0 and 1. In this file each lane owns one complete mass integral:
// in the I11 sums the lanes are two wavenumbers of one scale factor, and
// in the pair sums they are two different (K,Q) pairs. Lanes are never
// added to each other, because they belong to different outputs.
//
// SIMDe turns each simde_mm_* call into the vector instructions of the
// build machine (SSE2 or AVX on x86, NEON on arm64). set_pd(x1, x0) puts
// x0 in lane 0 and x1 in lane 1: the high lane is written first.
// set1_pd(x) copies one number into both lanes; setzero_pd() gives [0,0].
// A fused multiply-add (FMA) evaluates a*b+c with the product kept exact
// and a single rounding of the sum. SIMDe emits one fused instruction on
// arm64 and on x86 builds with FMA enabled (the optimized -march=native
// build on FMA hardware); its portable fallback for other x86 builds
// rounds a*b first and then adds c. The two can differ in the last bit.
// Unaligned stores (storeu) accept addresses that are not multiples of
// 16 bytes. They still require two valid adjacent doubles in the array.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Estimate the limit of eleven integrals with progressively smaller M_min.
//
// The integrals omit a slowly decreasing low-mass remainder. Wynn's epsilon
// algorithm removes successive geometric parts of that remainder without
// changing the halo abundance or bias. Its recurrence is
//
//   epsilon[-1,n] = 0, epsilon[0,n] = partial[n],
//   epsilon[p+1,n] = epsilon[p-1,n+1]
//                    + 1/(epsilon[p,n+1]-epsilon[p,n]).
//
// Even columns estimate the integral; odd columns are auxiliary reciprocal
// differences. Eleven partial sums permit five extrapolation steps. Here
// the lower mass bounds are 1, 10^-4, ..., 10^-40 M_sun/h. They are equally
// spaced in lnM, the integration coordinate used to construct the sequence.
// partial[n] integrates from the lower bound M_n = 10^(-4n) M_sun/h upward.
//
// Why the remainder is geometric: suppose the integrand below these bounds
// is a power law, dI/dlnM = c M^alpha with alpha > 0. The part still
// missing below M_n is then (c/alpha) M_n^alpha = (c/alpha) r^n, with the
// ratio r = 10^(-4 alpha). Equal steps in lnM make the remainder shrink by
// the same factor from one partial sum to the next. Column 2p of the
// epsilon table equals Shanks' transform of the sequence, which is exact
// when the remainder is a sum of p such geometric terms.
//
// A zero difference gives no further information. Stop at the last finite
// even column in that case, rather than dividing by zero. This guard is
// needed for already-converged sequences, not a substitute for checking
// convergence with more accurate halo tables and quadrature rules.
// Increasing extrapolation order is useful only while its corrections
// shrink. A growing correction can signal a pole in the extrapolation:
// then a tiny change in a partial sum produces a large change in I11.
// Keep the previous estimate when this occurs, instead of always taking
// the highest finite order. This does not rescale any halo mass or bias.
// ---------------------------------------------------------------------------
static double halo_wynn_cov(
    const double* partial // eleven integrals with decreasing lower mass
  )
{
  double epsilon[12][11] = {{0.0}}; // row 0 represents column -1
  double estimate = partial[10];   // finite integral before extrapolation
  double last_change = INFINITY;   // accept the first finite correction

  for (int node=0; node<11; node++) {
    epsilon[1][node] = partial[node];
  }

  // Build one shorter column at a time. Array row column stores the
  // mathematical epsilon column column-1, because row 0 holds epsilon_-1.
  // Thus odd array rows contain integral estimates. Select their last
  // entry to retain the newest, deepest partial integral at every order.
  for (int column=2; column<12; column++) {
    // A zero difference means the sequence no longer changes at this
    // order; a non-finite entry means the reciprocal overflowed. Either
    // way, return the last accepted estimate.
    for (int node=0; node<12-column; node++) {
      const double difference = epsilon[column-1][node+1]
                                -epsilon[column-1][node];
      if (difference == 0.0) {
        return estimate;
      }
      epsilon[column][node] = epsilon[column-2][node+1]+1.0/difference;
      if (!isfinite(epsilon[column][node])) {
        return estimate;
      }
    }

    // An odd array row is an even mathematical column, a new estimate of
    // the integral. Accept its deepest entry only if the change from the
    // previous estimate has not grown; otherwise keep that estimate.
    if (column % 2 == 1) {
      const double change = fabs(epsilon[column][11-column]-estimate);
      if (change > last_change) {
        return estimate;
      }
      estimate = epsilon[column][11-column];
      last_change = change;
    }
  }
  return estimate;
}

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
// Array names write beta first and mu second: I11 = I_1^1 (one profile,
// one bias), I02 = I_2^0, I12 = I_2^1, I13 = I_3^1 and I04 = I_4^0.
// Each leg brings the volume V = M/rho_cb that the halo mass would fill
// at the mean cb density. With one number density per halo, I_mu^beta
// has units of L^(3mu-3), L = c/H0: I11 is dimensionless and I04 is L^9.
//
// This is the halo-moment definition of Takada & Hu (2013), Eq. 25,
// arXiv:1302.6994, applied to the cb field. Halos count collapsed cold
// matter plus baryons, so their peak height and abundance use
//
//   nu = 1.686/sigma_cb(M,a),
//   dn/dlnM = (rho_cb/M) f(nu,a) nu dlnnu/dlnM.
//
// 1.686 is delta_c, the linear density contrast at which a spherical
// overdensity collapses (the value halo.c uses). The multiplicity f
// obeys (M/rho_cb) dn = f(nu) dnu: f(nu) dnu is the mass density in halos
// with peak heights between nu and nu+dnu, divided by the mean cb density.
// The core normalizes the fitted f by integral b f dnu = 1 over all nu
// (matter is unbiased with respect to itself), not by integral f dnu = 1.
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
// With ordinary finite mass panels only I11 receives a completion. By the
// normalization above, integral dn b M/rho_cb = 1 over all masses, so
// I11(k=0) should be one: the 2-halo power then tends to P_lin at large
// scales. A finite lower cutoff M_min misses part of that integral. Define
// A = 1 - sum_mass dlnM (dn/dlnM) b M/rho_cb on this same quadrature,
// and add A*u(k|M_min) to I11: the missing biased mass is assigned to
// halos of mass exactly M_min (Mead et al. 2020, arXiv:2005.00009,
// Appendix A). Thus I11(0)=1 at every a, independent of the finite lower
// mass cutoff. Higher moments are left uncorrected, as in the core
// halo-model convention; Mead et al. note that halos below M_min add very
// little to the one-halo integral, which carries an extra mass factor.
// No division by a bias-normalization table is applied.
//
// The default panels extend to 10^-40 with eleven four-decade intervals
// below 10^4. For these panels, extrapolate the raw I11 partial integrals
// with Wynn first. Extrapolate their zero-k bias weights by the same rule
// and add only [1-Wynn(I11(0))]*u(k|M_min). This retains the exact large-
// scale normalization while reducing the weight assigned to that profile.
// In one formula, I11(k) = E(k) + [1-E(0)] u(k|M_min), where E(k) is the
// Wynn limit of the I11 partial integrals at k (with other panel layouts,
// E is the finite integral itself).
// Higher moments converge quickly because of their extra mass factors;
// they use the finite integrals without extrapolation or completion.
//
// Computation proceeds in four visible stages:
// 1. Build one mass rule shared by all a and k, with covariance-owned
//    panels and node counts. The core Ntable settings are never modified.
// 2. Compute each mass/scale-factor abundance, bias and concentration once.
//    Store the five mass weights needed by all later sums together.
// 3. Compute u(k|M) once per (a,k,M), shared by every pair involving k.
// 4. Sum I11 and the five pair moments. In the I11 sums, SIMDe lanes 0
//    and 1 hold the mass integrals at two wavenumbers, k[row][index] and
//    k[row][next]; in the pair sums, they hold the five moments of two
//    different (K,Q) pairs. Each lane adds its mass nodes in a fixed
//    order (I11: nodes above 10^4 first, then the tail panels downward;
//    pairs: increasing mass). No sum is split between threads.
//
// Code map: mass[0..2][node] hold M, V = M/rho_cb and dlnM/V at each
// quadrature node; weights[0..4][row][node] hold dn b V, dn V^2,
// dn b V^2, dn b V^3 and dn V^4 with dn = (dn/dlnM) dlnM at a[row];
// profile[row][index][node] is u(k[row][index]|M), and its extra slot
// node = nmass belongs to the M_min completion halo, whose weight
// 1 - E(0) is completion[row].
//
// Parameters:
//   a[na], k[na][nk] - current scale factors and their wavenumber grids
//   lnm_edges[npanel+1], nquad - mass panels and GL nodes per panel
//   i11 - caller-owned [na][nk], dimensionless
//   moments - caller-owned [5][na][npair], npair=nk*(nk+1)/2
// Pair order is (0,0),(0,1),...,(1,1),... in the supplied k grid: pair
// (first,second), first <= second, has K = k[row][first] and
// Q = k[row][second].
// Roles are I02(K,Q), I12(K,Q), I13(K,Q,Q), I13(K,K,Q), I04(K,K,Q,Q).
// Their dimensions are L^3,L^3,L^6,L^6,L^9 with L=c/H0.
// Passing NULL for moments requests I11 alone. The SSC logarithmic
// slope needs I11 at two nearby wavenumbers, but its higher moments
// are already known at the central wavenumber. In that case no pair
// table is allocated and no two-, three- or four-profile sum is done.
//
// Calls are serial at the entry point: one thread enters this function.
// Lazy core tables are warmed before workers read them (stage 2). There
// is no covariance static cache; grouped scratch is released before
// return. Writable rows cannot alias inputs or each other. Public reader
// table precision remains part of the supplied core state and must be
// included in the eventual covariance accuracy scan.
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
  const double lnm_min = log(limits.halo_sigma_min);
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

  // Recognize the documented tail sequence: edges 0..11 must equal
  // ln(10^(-40+4*edge)), the bounds 10^-40, 10^-36, ..., 10^4 M_sun/h,
  // to 1e-12 in lnM. Other supplied panel layouts remain ordinary finite
  // integrals, useful for independent comparisons.
  int ntail = 0; // panels used by the eleven-term extrapolation
  if (npanel >= 11) {
    ntail = 11;
    for (int edge=0; edge<=11; edge++) {
      if (fabs(lnm_edges[edge]-(-40.0+4.0*edge)*log(10.0)) > 1.e-12) {
        ntail = 0;
      }
    }
  }
  // Tiny-halo profiles vary very slowly within each four-decade panel.
  // The tested tail rule starts at 32 nodes; ordinary mass panels retain
  // at least 64. Refining integration accuracy raises both rules: main
  // rules 64 and 96 pair with 32 tail nodes, larger ones with half.
  const int tail_nquad = nquad <= 96 ? 32 : nquad/2;
  const int tail_nodes = ntail*tail_nquad;
  const int nmass = tail_nodes+(npanel-ntail)*nquad;
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

  // Two precomputed GSL Gauss-Legendre rules: nquad nodes for each ordinary
  // panel and tail_nquad nodes for each tail panel. Each rule is mapped
  // onto every logarithmic interval of its kind.
  gsl_integration_glfixed_table* rule = malloc_gslint_glfixed(nquad);
  gsl_integration_glfixed_table* tail_rule =
      malloc_gslint_glfixed(tail_nquad);

  // Low-mass tail panels use the smaller rule; every panel above 10^4
  // uses the main nquad rule. Both parts store the same physical
  // quantities, so the later halo and SIMD loops do not need separate
  // integration formulas. GSL returns a panel's nodes in increasing lnM,
  // so the global node index runs through the masses in increasing order:
  // the tail panels first (indices below tail_nodes), then the others.
  for (int panel=0; panel<npanel; panel++) {
    const int count = panel < ntail ? tail_nquad : nquad;
    const int start = panel < ntail ? panel*tail_nquad
                                    : tail_nodes+(panel-ntail)*nquad;
    // Map one Gaussian node into this panel and store its three mass-only
    // quantities at the global index used by every redshift and k row.
    for (int node=0; node<count; node++) {
      const int index = start+node;
      double lnm;      // logarithmic mass abscissa
      double measure;  // positive dlnM quadrature measure

      gsl_integration_glfixed_point(lnm_edges[panel], lnm_edges[panel+1],
                                    node, &lnm, &measure,
                                    panel < ntail ? tail_rule : rule);

      // Store physical mass, cb volume, and the weighted number prefactor
      // separately: subsequent redshift rows share these same mass nodes.
      mass[0][index] = exp(lnm);
      mass[1][index] = mass[0][index]/rho_cb;
      mass[2][index] = measure/mass[1][index];
    }
  }

  gsl_integration_glfixed_table_free(rule);
  gsl_integration_glfixed_table_free(tail_rule);

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
  // No table depends on the a or M of the call that builds it, so one
  // serial call per reader suffices: sigma2 builds the shared (lnM,a)
  // variance and mass-slope tables (matter and cb), which dlognudlogm and
  // conc also read; fnu builds the table of its normalization alpha(a);
  // u_nfw_c builds the NFW transform table. hb1nu reads fixed Tinker
  // coefficients and is called so that every reader used below has run
  // once serially. (void) discards the values: only the setup matters.
  (void) sigma2(mass_min, a[0]);
  (void) dlognudlogm(mass_min, a[0]);
  (void) fnu(1.0, a[0]);
  (void) hb1nu(1.0, a[0]);
  (void) u_nfw_c(conc(mass_min, a[0]), 1.0, mass_min, a[0]);

  // The weights already include dlnM. Their role is the power of volume
  // and whether a bias is present: bV, V^2, bV^2, bV^3, V^4 times dn.
  // Roles 0..4 therefore feed I11, I02, I12, both I13 and I04. The
  // concentration rows have one extra slot, index nmass, for the M_min
  // halo of the completion; completion[row] holds that halo's weight.
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
    // The peak height is nu = delta_c/sigma_cb(M,a), delta_c = 1.686.
    for (int node=0; node<nmass; node++) {
      const double m = mass[0][node];
      const double volume = mass[1][node];
      const double nu = 1.686/sqrt(sigma2(m, a[row]));
      const double bias = hb1nu(nu, a[row]);

      // number = (dn/dlnM)*dlnM at this mass and scale factor. mass[2]
      // holds dlnM*rho_cb/M, so number = (rho_cb/M) f(nu) nu
      // (dlnnu/dlnM) dlnM: the header's mass function times the
      // quadrature measure. Attach powers of the halo volume and bias to
      // form the five moment weights.
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
  // equal one when all matter is included. For the default tail, estimate
  // that integral's limit before assigning the residual to the minimum-
  // mass profile. One worker owns every partial sum at each scale factor.
  #pragma omp parallel for schedule(static)
  for (int row=0; row<na; row++) {
    double resolved = 0.0; // resolved biased mass fraction at this a

    for (int node=tail_nodes; node<nmass; node++) {
      resolved += weights[0][row][node];
    }

    // Begin with the integral above 10^4. Append successively lower panels
    // to obtain bounds 1, 10^-4, ..., 10^-40. Extrapolation estimates the
    // unintegrated tail; any small remaining normalization error is kept
    // explicit in completion, rather than rescaling the fitted bias.
    // partial[n] is the integral down to 10^(-4n), the Wynn sequence. The
    // I11 loop of stage 4a adds the same nodes in this same order, so at
    // k=0, where u=1, its partial sums equal these exactly.
    if (ntail > 0) {
      double partial[11]; // cumulative bias-weighted integrals
      for (int panel=ntail-1; panel>=0; panel--) {
        for (int node=panel*tail_nquad;
             node<(panel+1)*tail_nquad; node++) {
          resolved += weights[0][row][node];
        }
        partial[ntail-1-panel] = resolved;
      }
      resolved = halo_wynn_cov(partial);
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
  // least as many rows as workers, nchunk is one and each task instead
  // evaluates one complete mass row. No mass sum is split here; the later
  // moment integrals keep their order.
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
          // u(0|M) = 1 exactly by normalization; u_nfw_c needs k > 0.
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
  // The default low-mass panels give a sequence of partial integrals.
  // Extrapolate that sequence and add only its residual zero-k completion.
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

      // scalar: one I11 sum for wavenumber j = index (lane 0) or j = next
      // (lane 1), over the nodes above the tail, in increasing mass:
      //   double sum = 0.0;
      //   for (int node=tail_nodes; node<nmass; node++) {
      //     sum = fma(profile[row][j][node], weight[node], sum);
      //   }
      // weight[node] is the cb mass fraction in halos of this mass
      // interval, (dn/dlnM) dlnM M/rho_cb, times their bias. Multiplying
      // by u(k_j|M) gives those halos' share of I11(k_j), the factor that
      // a density leg in a separate halo brings to the two-halo terms.
      // Tail panels are appended to these sums below, before extrapolation
      // and completion. The SIMD loop runs the two scalar loops side by
      // side, one per lane, never adding the lanes.

      // setzero_pd returns [0,0]: lane 0 starts the I11 sum at
      // k[row][index], lane 1 the sum at k[row][next].
      v2d vsum = simde_mm_setzero_pd();

      // For each mass, update both k-specific sums with u(k|M)*weight.
      // SIMD shares the scalar weight but keeps the two integrals separate.
      for (int node=tail_nodes; node<nmass; node++) {
        // set_pd takes the high lane first: lane 0 receives u0[node] =
        // u(k[row][index]|M) and lane 1 receives u1[node] =
        // u(k[row][next]|M), both at this common mass node.
        const v2d vu = simde_mm_set_pd(u1[node], u0[node]);

        // set1_pd copies weight[node] = (dn/dlnM)*dlnM*b*M/rho_cb into
        // both lanes: the two wavenumbers see the same halos.
        const v2d vw = simde_mm_set1_pd(weight[node]);

        // fmadd: in each lane l, vsum[l] = fma(vu[l], vw[l], vsum[l]),
        // i.e. sum += u(k_l|M)*weight with the product kept exact and the
        // sum rounded once where SIMDe emits a fused instruction (header).
        vsum = simde_mm_fmadd_pd(vu, vw, vsum);
      }

      double result[2];

      // storeu writes lane 0 to result[0] and lane 1 to result[1]: the
      // I11 integrals above the tail at k[row][index] and k[row][next]
      // (the complete finite integrals when there is no tail). This stack
      // array needs no 16-byte alignment, only its two valid doubles.
      simde_mm_storeu_pd(result, vsum);

      // Continue the same two sums through the tail, one four-decade panel
      // at a time, from [1,10^4] down to [10^-40,10^-36] M_sun/h. After
      // each panel, save the running sums: partial[lane][n] is the I11
      // integral down to the lower bound 10^(-4n), the sequence that
      // halo_wynn_cov extrapolates. Each wavenumber keeps its own sequence.
      //
      // scalar: lane = 0 (j = index) or 1 (j = next), continuing sum:
      //   for (int panel=ntail-1; panel>=0; panel--) {
      //     for (int node=panel*tail_nquad;
      //          node<(panel+1)*tail_nquad; node++) {
      //       sum = fma(profile[row][j][node], weight[node], sum);
      //     }
      //     partial[lane][ntail-1-panel] = sum;
      //   }
      //   result[lane] = halo_wynn_cov(partial[lane]);
      if (ntail > 0) {
        double partial[2][11]; // one eleven-term sequence per wavenumber

        // Panels run downward in mass, nodes upward inside each panel: the
        // order of the zero-k sums in the completion loop of stage 2.
        for (int panel=ntail-1; panel>=0; panel--) {
          for (int node=panel*tail_nquad;
               node<(panel+1)*tail_nquad; node++) {
            // set_pd takes the high lane first: lane 0 receives u0[node]
            // (wavenumber index) and lane 1 u1[node] (wavenumber next) at
            // this tail mass, the same lane order as above the tail.
            const v2d vu = simde_mm_set_pd(u1[node], u0[node]);

            // set1_pd copies this tail node's I11 weight
            // (dn/dlnM)*dlnM*b*M/rho_cb into both lanes.
            const v2d vw = simde_mm_set1_pd(weight[node]);

            // fmadd: vsum[l] = fma(vu[l], vw[l], vsum[l]) in each lane,
            // sum += u*weight, rounded once where SIMDe emits a fused
            // instruction (file header).
            vsum = simde_mm_fmadd_pd(vu, vw, vsum);
          }

          // storeu writes the two running sums, now integrated down to this
          // panel's lower edge 10^(-40+4*panel), to result[0] (index) and
          // result[1] (next). No alignment is needed on this stack array.
          simde_mm_storeu_pd(result, vsum);
          partial[0][ntail-1-panel] = result[0];
          partial[1][ntail-1-panel] = result[1];
        }

        // Extrapolate each wavenumber's own sequence to zero lower mass:
        // result[0] = E(k[row][index]) and result[1] = E(k[row][next]).
        result[0] = halo_wynn_cov(partial[0]);
        result[1] = halo_wynn_cov(partial[1]);
      }

      // Add the unresolved-mass contribution at each physical k only once:
      // I11 = E(k) + completion*u(k|M_min). Slot nmass of each profile row
      // is the M_min completion halo. Lane 1 is written only when next is
      // a genuine wavenumber, which discards an odd nk's duplicated lane.
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

      // scalar: the five sums of one pair, j = index (lane 0) or j = next
      // (lane 1), over every mass node in increasing mass:
      //   double sum[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
      //   for (int node=0; node<nmass; node++) {
      //     uk = profile[row][pairs[0][j]][node];
      //     uq = profile[row][pairs[1][j]][node];
      //     product = uk*uq;
      //     sum[0] = fma(product, weights[1][row][node], sum[0]);
      //     sum[1] = fma(product, weights[2][row][node], sum[1]);
      //     sum[2] = fma(product*uq, weights[3][row][node], sum[2]);
      //     sum[3] = fma(product*uk, weights[3][row][node], sum[3]);
      //     sum[4] = fma(product*product, weights[4][row][node], sum[4]);
      //   }
      //   moments[role][row][j] = sum[role] for role = 0..4.
      // Each extra profile represents another density factor in one halo.
      // SIMD carries two copies of these five sums, one per pair lane.
      v2d vsums[5];

      for (int role=0; role<5; role++) {
        // setzero_pd gives [0,0]: lane 0 starts this moment for pair
        // index, lane 1 the same moment for pair next.
        vsums[role] = simde_mm_setzero_pd();
      }

      // At each mass, form the needed two-, three- and four-leg profile
      // products and add their weighted contributions to all five moments.
      // Lanes remain separate pairs; dn denotes the weighted abundance
      // (dn/dlnM)*dlnM and V denotes M/rho_cb in the comments below.
      for (int node=0; node<nmass; node++) {
        // --- PROFILES AT THIS MASS, WITH PAIRS KEPT IN SEPARATE LANES ---

        // set_pd takes lane 1 first: lane 0 receives uk0[node], the
        // profile u(K|M) of pair index, and lane 1 uk1[node], u(K|M) of
        // pair next.
        const v2d vk = simde_mm_set_pd(uk1[node], uk0[node]);

        // Pack the matching Q-profiles in the same index/next lane order:
        // lane 0 uq0[node] = u(Q|M) of pair index, lane 1 uq1[node] of
        // pair next. set_pd again receives the high-lane value first.
        const v2d vq = simde_mm_set_pd(uq1[node], uq0[node]);

        // mul_pd multiplies matching lanes: product = u(K)*u(Q) for each
        // pair, two legs of one halo (an ordinary rounded product).
        const v2d vproduct = simde_mm_mul_pd(vk, vq);

        // Append another Q leg: product*u(Q) = u(K)*u(Q)^2 in each lane.
        const v2d vkqq = simde_mm_mul_pd(vproduct, vq);

        // Append another K leg: product*u(K) = u(K)^2*u(Q) in each lane.
        const v2d vkkq = simde_mm_mul_pd(vproduct, vk);

        // Square the two-profile product: product*product = u(K)^2*u(Q)^2
        // per pair, the four legs of the one-halo term.
        const v2d vkkqq = simde_mm_mul_pd(vproduct, vproduct);

        // --- MASS WEIGHTS SHARED BY THE TWO PAIRS ---

        // set1_pd copies weights[1][row][node], the unbiased two-leg
        // weight dn*V^2 of this mass, into both lanes.
        const v2d vw02 = simde_mm_set1_pd(weights[1][row][node]);

        // set1_pd copies weights[2][row][node], the biased two-leg weight
        // dn*b*V^2, into both lanes.
        const v2d vw12 = simde_mm_set1_pd(weights[2][row][node]);

        // set1_pd copies weights[3][row][node], the biased three-leg
        // weight dn*b*V^3 shared by both I13 roles, into both lanes.
        const v2d vw13 = simde_mm_set1_pd(weights[3][row][node]);

        // set1_pd copies weights[4][row][node], the unbiased four-leg
        // weight dn*V^4, into both lanes.
        const v2d vw04 = simde_mm_set1_pd(weights[4][row][node]);

        // --- ADD THIS MASS NODE TO EACH OF THE FIVE MOMENTS ---

        // Each fmadd below computes, in each pair lane l,
        // vsums[role][l] = fma(legs[l], weight, vsums[role][l]): the
        // product of the profile legs and the mass weight is kept exact
        // and the running sum is rounded once where SIMDe emits a fused
        // instruction (file header). Mass nodes are added in increasing
        // order.

        // I02(K,Q): vsums[0] = fma(vproduct, vw02, vsums[0]), that is
        // sum += u(K)*u(Q)*dn*V^2, two legs in one unbiased halo.
        vsums[0] = simde_mm_fmadd_pd(vproduct, vw02, vsums[0]);

        // I12(K,Q): vsums[1] = fma(vproduct, vw12, vsums[1]), that is
        // sum += u(K)*u(Q)*dn*b*V^2, two legs in one biased halo.
        vsums[1] = simde_mm_fmadd_pd(vproduct, vw12, vsums[1]);

        // I13(K,Q,Q): vsums[2] = fma(vkqq, vw13, vsums[2]), that is
        // sum += u(K)*u(Q)^2*dn*b*V^3, one K leg and two Q legs.
        vsums[2] = simde_mm_fmadd_pd(vkqq, vw13, vsums[2]);

        // I13(K,K,Q): vsums[3] = fma(vkkq, vw13, vsums[3]), that is
        // sum += u(K)^2*u(Q)*dn*b*V^3, two K legs and one Q leg.
        vsums[3] = simde_mm_fmadd_pd(vkkq, vw13, vsums[3]);

        // I04(K,K,Q,Q): vsums[4] = fma(vkkqq, vw04, vsums[4]), that is
        // sum += u(K)^2*u(Q)^2*dn*V^4, all four legs in one halo.
        vsums[4] = simde_mm_fmadd_pd(vkkqq, vw04, vsums[4]);
      }

      // Store each moment's two pair results, discarding a duplicated
      // second lane when the total number of pairs is odd.
      for (int role=0; role<5; role++) {
        double result[2];

        // storeu writes lane 0 to result[0], this moment for pair index,
        // and lane 1 to result[1], the same moment for pair next. The
        // stack array needs two valid doubles but no 16-byte alignment.
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
