#include <math.h>
#include <stdlib.h>

#include "halo_cluster_cov.h"
#include "cosmolike/basics.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/halo.h"
#include "cosmolike/halo_cluster.h"
#include "cosmolike/structs.h"
#include "cosmolike/structs_cluster.h"
#include "simde/x86/sse2.h"

// SIMD (single instruction, multiple data) applies one instruction to
// several numbers at once. A v2d, SIMDe's simde__m128d, holds two
// doubles; each position is called a lane, lane 0 the low and lane 1 the
// high double. Here lane 0 holds mass node and lane 1 mass node+1:
// adjacent mass samples, never two parts of the same sum. Only
// multiplications occur, so each lane equals its scalar product bitwise.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Prepare selected halo measures for cluster covariance mass integrals.
//
// A richness catalog selects a halo with probability S_lambda(M,z).
// Its expected abundance in a logarithmic mass interval is
//
//   dn_selected = dlnM (rho_cb/M) f(nu,a) nu (dlnnu/dlnM) S_lambda,
//   nu = 1.686/sigma_cb(M,a).
//
// A matter-density leg contributes p(k|M)=(M/rho_m) u_NFW(k|M), the
// halo's mass-weighted Fourier profile (p in moments_cluster_cov.c).
// Multiplying dn_selected by p, b_h*p or several p factors gives the
// selected moments used by cluster lensing, count cross covariance and
// SSC. For example, an overdensity changes the abundance by b_h times
// that overdensity. Its one-profile response therefore contains
// integral dn_selected b_h p, whereas the mean uses integral dn_selected p.
// The downstream moment integrator forms both from these same samples.
//
// This entry point requires massless neutrinos, so rho_cb=rho_m. It reads
// sigma_cb, the linear halo bias, concentration and NFW profile through
// public core functions. The initialized lognormal richness relation
// supplies S_lambda; no new fit or mass-observable convention is chosen.
// The selection is fixed at given M,z under a background perturbation.
// An environmental selection response needs extra physics, not a change
// to the quadrature weights. Photo-z membership is also a separate input.
// See Tinker et al. (2010), arXiv:1001.3162, Eq. 6 (bias), Eq. 7
// (normalization) and Eqs. 8--12 (multiplicity), and To et al. (2021),
// arXiv:2008.10757, Eqs. 20--21.
//
// The HMF amplitude alpha follows the initialized cluster convention
// (cluster.hmf_alpha_mode). The Tinker multiplicity is
//
//   f(nu) = alpha [1 + (beta nu)^(-2 phi)] nu^(2 eta) exp(-gamma nu^2/2).
//
// In fixed mode (CLUSTER_HMF_ALPHA_FIXED, the DES convention) alpha =
// 0.368, the z = 0 value of Table 4, at every redshift. In normalized mode
// (CLUSTER_HMF_ALPHA_NORMALIZED) alpha(a) is the value that halo.c's fnu
// uses, set at each redshift by Eq. 7, integral b(nu) f(nu) dnu = 1
// (0.3684 at z = 0, falling with z). Both modes share beta, gamma, phi and
// eta, so the two f(nu) differ only by the constant factor 0.368/alpha(a)
// at each scale factor. That factor is evaluated once per scale factor at
// nu=1, and every mass then uses the public fnu reader times it. No
// duplicate mass-dependent fit is needed. This identity holds only for
// the Tinker et al. (2010) fit in halo.c's fnu; the binding requires
// HMF_TINKER_2010. The amplitude changes the abundance and its moments,
// but not abundance-weighted averages such as the selected bias.
//
// There is no low-mass completion for an observed cluster catalog.
// The caller supplies the mass quadrature, keeping covariance resolution
// independent of data-vector tables. Results retain the mass axis so the
// existing selected-moment integrator can be checked or reused directly.
//
// Calculation:
// 1. Warm lazy core readers serially; prepare mass-only quantities.
// 2. Read abundance, halo bias and concentration once at each (a,M).
//    Apply richness selection once to the abundance, using two mass lanes.
// 3. Reuse concentration for all k; multiply each NFW value by its halo
//    volume with SIMDe. All three independent indices are OpenMP work.
// No worker performs a shared reduction or initializes a core table.
//
// Parameters (core units: M in Msun/h, lengths in c/H0):
//   na       - number of scale factors (states), >= 1
//   a        - [na] scale factors, limits.a_min <= a < 1
//   nk       - number of wavenumbers per state, >= 1
//   k        - [na][nk] wavenumbers in (c/H0)^-1, k >= 0
//   nmass    - number of mass quadrature nodes, >= 1
//   lnm      - [nmass] ln(M/[Msun/h]) inside the core sigma mass range
//   dlnm     - [nmass] positive quadrature weights in ln M
//
// Outputs (caller-owned; every entry is written):
//   weight   - [na][cluster.richness_nbin][nmass] selected dn =
//              dlnM (dn/dlnM) S_lambda, in (c/H0)^-3
//   bias     - [na][nmass] linear halo bias b_h(nu), dimensionless
//   profile  - [na][nk][nmass] p(k|M) = (M/rho_m) u_NFW(k|M), in
//              (c/H0)^3
//
// Cache ownership: the core variance, alpha(a) and NFW tables are owned
// by the core and only warmed here. Two scratch arrays, work and
// amplitude, are allocated and freed within the call.
// ---------------------------------------------------------------------------
void halo_samples_cluster_cov(
    const int na,                      // independent scale factors
    const double* a,                   // scale factors
    const int nk,                      // wavenumbers per scale factor
    const double* const* k,            // wavenumber rows
    const int nmass,                   // logarithmic mass nodes
    const double* lnm,                 // logarithmic masses
    const double* dlnm,                // positive integration measures
    double** const* weight,            // selected abundance measures
    double* const* bias,               // linear halo bias
    double** const* profile            // volume-weighted profiles
  )
{
  // --- 1. SERIAL READER WARMUP AND SHARED MASS QUANTITIES ---

  // rho_m = rho_crit Omega_m, in (Msun/h) per (c/H0)^3; massless
  // neutrinos make it the rho_cb of the mass function as well. first_mass
  // is only a valid sample point for the warm calls below.
  const double rho = cosmology.rho_crit*cosmology.Omega_m;
  const double first_mass = exp(lnm[0]);
  const int nrichness = cluster.richness_nbin;

  // Several core readers build a lazy table on their first call, or after
  // a change of the settings that table depends on, and that build is not
  // thread-safe. Each call below runs on this single thread, before any
  // OpenMP region, so the parallel loops only read finished tables. The
  // returned values are discarded.
  //   sigma2      builds the cold variance and mass-slope table shared by
  //               sigma2, dlognudlogm and conc (conc reads it at a and at
  //               a = 1);
  //   dlognudlogm reads that same table;
  //   fnu         builds the alpha(a) table of Tinker Eq. 7;
  //   hb1nu       has no table; the call checks the halo-bias model once;
  //   u_nfw_c     builds the NFW f, G table, which depends on neither mass
  //               nor concentration; its argument conc(first_mass, a[0])
  //               checks the concentration model once;
  //   prob_richness_bin_given_m has no table; the call validates the
  //               lognormal richness model once.
  // An unsupported model setting therefore stops here, on one thread.
  (void) sigma2(first_mass, a[0]);
  (void) dlognudlogm(first_mass, a[0]);
  (void) fnu(1.0, a[0]);
  (void) hb1nu(1.0, a[0]);
  (void) u_nfw_c(conc(first_mass, a[0]), 1.0, first_mass, a[0]);
  (void) prob_richness_bin_given_m(lnm[0], 1.0/a[0]-1.0, 0);

  // The first two rows hold M and M/rho_m; row 2+state holds c(M,a) of
  // that state, filled in stage 2. One allocation owns their common mass
  // axis and call lifetime.
  double** work = (double**) malloc2d(2+na, nmass);
  for (int node=0; node<nmass; node++) {
    work[0][node] = exp(lnm[node]);
    work[1][node] = work[0][node]/rho;
  }

  // Only the fixed-amplitude convention needs a correction to fnu. At
  // nu=1 every power of nu is one, so eta drops out and
  //
  //   f(1) = alpha [1 + beta^(-2 phi)] exp(-gamma/2).
  //
  // beta, gamma and phi follow Tinker's redshift evolution (Eqs. 9--12)
  // at aa = max(a, 0.25): the fit is held at its z = 3 values beyond its
  // fitted range, as in halo.c. fixed is f(1) with alpha = 0.368, and
  // fnu(1, a) is f(1) with halo.c's alpha(a), so their ratio is
  // 0.368/alpha(a). In normalized mode the factor stays 1.
  double* amplitude = (double*) malloc1d(na);
  for (int state=0; state<na; state++) {
    amplitude[state] = 1.0;
    if (cluster.hmf_alpha_mode == CLUSTER_HMF_ALPHA_FIXED) {
      const double aa = fmax(a[state], 0.25);
      const double beta = 0.589*pow(aa, -0.2);
      const double gamma = 0.864*pow(aa, 0.01);
      const double phi = -0.729*pow(aa, 0.08);
      const double fixed = 0.368*(1.0+pow(beta, -2.0*phi))*exp(-gamma/2.0);
      amplitude[state] = fixed/fnu(1.0, a[state]);
    }
  }

  // --- 2. SELECTED ABUNDANCE, BIAS AND CONCENTRATION ---

  // The expensive halo quantities depend on a and M but not k. Every
  // richness category reuses them. One iteration handles two adjacent
  // mass nodes, node and node+1, at one scale factor: it reads their
  // unselected abundance, bias and concentration, then multiplies the
  // abundance by each richness category's probability, with SIMD lane 0
  // holding node and lane 1 node+1. node advances by two; when nmass is
  // odd the last iteration has count = 1 and uses the scalar branch.
  // Every iteration writes distinct entries, so collapsing state and mass
  // pairs keeps (nmass+1)/2 independent items per state even when na=1.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int state=0; state<na; state++) {
    for (int node=0; node<nmass; node+=2) {
      const double* restrict mass = work[0];
      const double* restrict volume = work[1];
      double* restrict concentration = work[2+state];
      double* restrict halo_bias = bias[state];
      double number[2] = {0.0, 0.0}; // dn before the richness selection
      const int count = node+1 < nmass ? 2 : 1;

      // Scalar public readers supply each mass's fit values. They read
      // already-warmed tables; the independent weights use SIMD below.
      // nu = delta_c/sigma_cb(M,a) with delta_c = 1.686, and the
      // unselected abundance of this mass node is
      //   number = dlnM (rho_m/M) f(nu) amplitude nu dln nu/dln M,
      // where fnu(nu, a)*amplitude is f(nu) in the chosen alpha mode.
      // hb1nu gives the linear bias b_h(nu) and conc the concentration,
      // stored for the profile stage.
      for (int lane=0; lane<count; lane++) {
        const int index = node+lane;
        const double nu = 1.686/sqrt(sigma2(mass[index], a[state]));
        number[lane] = dlnm[index]/volume[index]*fnu(nu, a[state])
                       *amplitude[state]*nu
                       *dlognudlogm(mass[index], a[state]);
        halo_bias[index] = hb1nu(nu, a[state]);
        concentration[index] = conc(mass[index], a[state]);
      }

      // Assigning a halo to a richness category multiplies its number
      // measure by one probability, S_lambda = P(bin|M, z) at z = 1/a-1.
      // Do not square this probability when several later covariance
      // legs belong to that same halo: its membership indicator obeys
      // I^2 = I.
      for (int bin=0; bin<nrichness; bin++) {
        double* restrict selected = weight[state][bin];
        const double first = prob_richness_bin_given_m(
            lnm[node], 1.0/a[state]-1.0, bin);
        if (count == 2) {
          const double second = prob_richness_bin_given_m(
              lnm[node+1], 1.0/a[state]-1.0, bin);

          // scalar: for the two adjacent mass samples,
          //   selected[node] = number[0]*first;
          //   selected[node+1] = number[1]*second;
          // Each membership probability selects objects from its own mass
          // interval. SIMD performs these two multiplications together;
          // it does not combine probabilities from different masses.
          // loadu puts number[0], the unselected dn of mass node, in lane
          // 0 and number[1], that of node+1, in lane 1. The two-element
          // stack array needs no alignment.
          const v2d vnumber = simde_mm_loadu_pd(number);

          // set_pd(high, low) lists the high lane first: lane 0 gets
          // first = P(bin|M_node) and lane 1 second = P(bin|M_node+1),
          // each probability next to its own mass's number measure.
          const v2d vselection = simde_mm_set_pd(second, first);

          // mul_pd, lane by lane: dn S_lambda, each mass's abundance times
          // its own category probability, rounded once.
          const v2d vweight = simde_mm_mul_pd(vnumber, vselection);

          // storeu writes lane 0 to selected[node] and lane 1 to
          // selected[node+1]. The output need not have a vector-aligned
          // address; both elements exist because count == 2.
          simde_mm_storeu_pd(selected+node, vweight);
        } else {
          // A final unpaired mass node (odd nmass): the same single
          // multiplication.
          selected[node] = number[0]*first;
        }
      }
    }
  }

  // --- 3. MASS-WEIGHTED PROFILES SHARED BY ALL RICHNESS CATEGORIES ---

  // A profile sample depends on a, k and M. The matter leg of one halo is
  // p(k|M) = (M/rho_m) u(k|M): its mass in units of the mean density (a
  // volume) times its normalized Fourier profile. One iteration fills two
  // adjacent mass nodes, node (SIMD lane 0) and node+1 (lane 1), at one
  // (a, k); an odd final node uses the scalar branch. Distribute all
  // three axes so small batches can use eight or ten workers, while
  // retaining contiguous pairs of masses. No integration or
  // thread-dependent summation occurs.
  #pragma omp parallel for collapse(3) schedule(static)
  for (int state=0; state<na; state++) {
    for (int mode=0; mode<nk; mode++) {
      for (int node=0; node<nmass; node+=2) {
        const double* restrict mass = work[0];
        const double* restrict volume = work[1];
        const double* restrict concentration = work[2+state];
        double* restrict output = profile[state][mode];

        // At k=0 the Fourier phase exp(i k.r) is one at every radius, so
        // u(0|M) is the mass inside r_Delta divided by M: exactly one for
        // the truncated NFW profile, and the sample is M/rho_m. u_nfw_c is
        // read only for positive k: it takes ln(k r_s), which diverges at
        // k = 0 and would give NaN there.
        if (k[state][mode] == 0.0) {
          output[node] = volume[node];
          if (node+1 < nmass) {
            output[node+1] = volume[node+1];
          }
          continue;
        }

        // u_nfw_c returns the normalized NFW transform u(k|M) of each
        // mass at this k, using the concentration stored in stage 2.
        const double first = u_nfw_c(concentration[node], k[state][mode],
                                     mass[node], a[state]);
        if (node+1 < nmass) {
          const double second = u_nfw_c(concentration[node+1],
              k[state][mode], mass[node+1], a[state]);

          // scalar: for the two adjacent mass samples,
          //   output[node] = volume[node]*first;
          //   output[node+1] = volume[node+1]*second;
          // Here volume=M/rho and first/second are normalized NFW profiles
          // at those masses. Each product is one halo's density factor.
          // SIMD evaluates both products with one mass in each lane.
          // loadu puts volume[node] = M_node/rho_m in lane 0 and
          // volume[node+1] in lane 1. These halo volumes convert the
          // normalized u into a density leg. No alignment is required, and
          // both elements exist because node+1 < nmass.
          const v2d vvolume = simde_mm_loadu_pd(volume+node);

          // set_pd(high, low) takes the high lane first: lane 0 gets first
          // = u(k|M_node) and lane 1 second = u(k|M_node+1), matching the
          // volumes.
          const v2d vprofile = simde_mm_set_pd(second, first);

          // mul_pd, lane by lane: each lane becomes its own
          // (M/rho)*u(k|M), with units L^3, rounded once.
          const v2d vweighted = simde_mm_mul_pd(vvolume, vprofile);

          // storeu writes lane 0 to output[node] and lane 1 to
          // output[node+1] in the caller-owned mass row, without an
          // alignment requirement.
          simde_mm_storeu_pd(output+node, vweighted);
        } else {
          // A final unpaired mass node (odd nmass): the same single
          // multiplication.
          output[node] = volume[node]*first;
        }
      }
    }
  }

  free(amplitude);
  free(work);
}
