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

// Two doubles occupy independent SIMD positions, called lanes. They
// represent adjacent mass samples, never two parts of the same sum.
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
// A matter-density leg contributes v(k|M)=(M/rho_m) u_NFW(k|M).
// Multiplying dn_selected by v, b_h*v or several v factors gives the
// selected moments used by cluster lensing, count cross covariance and
// SSC. For example, an overdensity changes the abundance by b_h times
// that overdensity. Its one-profile response therefore contains
// integral dn_selected b_h v, whereas the mean uses integral dn_selected v.
// The downstream moment integrator forms both from these same samples.
//
// This entry point requires massless neutrinos, so rho_cb=rho_m. It reads
// sigma_cb, the linear halo bias, concentration and NFW profile through
// public core functions. The initialized lognormal richness relation
// supplies S_lambda; no new fit or mass-observable convention is chosen.
// The selection is fixed at given M,z under a background perturbation.
// An environmental selection response needs extra physics, not a change
// to the quadrature weights. Photo-z membership is also a separate input.
// See Tinker et al. (2010), arXiv:1001.3162, Eqs. 8--12, and
// To et al. (2021), arXiv:2008.10757, Eqs. 20--21.
//
// The HMF amplitude follows the initialized cluster convention. In fixed
// mode it is 0.368; in normalized mode it is halo.c's fnu. Since these
// functions have identical nu dependence, their ratio can be evaluated
// once per scale factor at nu=1. Every mass then uses the public fnu
// reader times that ratio. No duplicate mass-dependent fit is needed.
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

  const double rho = cosmology.rho_crit*cosmology.Omega_m;
  const double first_mass = exp(lnm[0]);
  const int nrichness = cluster.richness_nbin;

  (void) sigma2(first_mass, a[0]);
  (void) dlognudlogm(first_mass, a[0]);
  (void) fnu(1.0, a[0]);
  (void) hb1nu(1.0, a[0]);
  (void) u_nfw_c(conc(first_mass, a[0]), 1.0, first_mass, a[0]);
  (void) prob_richness_bin_given_m(lnm[0], 1.0/a[0]-1.0, 0);

  // The first two rows hold M and M/rho; later rows hold c(M,a).
  // One allocation owns their common mass axis and call lifetime.
  double** work = (double**) malloc2d(2+na, nmass);
  for (int node=0; node<nmass; node++) {
    work[0][node] = exp(lnm[node]);
    work[1][node] = work[0][node]/rho;
  }

  // Only the fixed-amplitude convention needs a correction to fnu.
  // At nu=1 the powers of nu are unity. The four remaining parameters
  // follow Tinker's redshift fit, held at z=3 beyond its fitted range.
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
  // richness category reuses them. Each worker owns two adjacent masses;
  // collapsing state and mass groups also fills eight cores when na=1.
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
      // measure by ONE probability. Do not square this probability when
      // several later covariance legs belong to that same halo.
      for (int bin=0; bin<nrichness; bin++) {
        double* restrict selected = weight[state][bin];
        const double first = prob_richness_bin_given_m(
            lnm[node], 1.0/a[state]-1.0, bin);
        if (count == 2) {
          const double second = prob_richness_bin_given_m(
              lnm[node+1], 1.0/a[state]-1.0, bin);

          // Scalar equivalent for the two adjacent mass samples:
          //   selected[node] = number[0]*first;
          //   selected[node+1] = number[1]*second;
          // Each membership probability selects objects from its own mass
          // interval. SIMD performs these two multiplications together;
          // it does not combine probabilities from different masses.
          // Low/high lanes contain neighboring masses' unselected dn.
          // loadu accepts this stack array without alignment constraints.
          const v2d vnumber = simde_mm_loadu_pd(number);

          // set_pd lists the high lane first: each probability is paired
          // with its own mass's number measure in vnumber.
          const v2d vselection = simde_mm_set_pd(second, first);

          // Multiply each mass's abundance by its category probability.
          const v2d vweight = simde_mm_mul_pd(vnumber, vselection);

          // Store the two selected measures into adjacent output nodes.
          // The output need not have a vector-aligned address.
          simde_mm_storeu_pd(selected+node, vweight);
        } else {
          selected[node] = number[0]*first;
        }
      }
    }
  }

  // --- 3. MASS-WEIGHTED PROFILES SHARED BY ALL RICHNESS CATEGORIES ---

  // A profile sample depends on a, k and M. Distribute all three axes so
  // small batches can use eight or ten workers, while retaining contiguous
  // pairs of masses. No integration or thread-dependent summation occurs.
  #pragma omp parallel for collapse(3) schedule(static)
  for (int state=0; state<na; state++) {
    for (int mode=0; mode<nk; mode++) {
      for (int node=0; node<nmass; node+=2) {
        const double* restrict mass = work[0];
        const double* restrict volume = work[1];
        const double* restrict concentration = work[2+state];
        double* restrict output = profile[state][mode];

        // At k=0 the normalized Fourier profile is exactly one: every
        // mass element has the same Fourier phase. The core expression
        // contains a logarithmic limit and is read only for positive k.
        if (k[state][mode] == 0.0) {
          output[node] = volume[node];
          if (node+1 < nmass) {
            output[node+1] = volume[node+1];
          }
          continue;
        }

        const double first = u_nfw_c(concentration[node], k[state][mode],
                                     mass[node], a[state]);
        if (node+1 < nmass) {
          const double second = u_nfw_c(concentration[node+1],
              k[state][mode], mass[node+1], a[state]);

          // Scalar equivalent for the two adjacent mass samples:
          //   output[node] = volume[node]*first;
          //   output[node+1] = volume[node+1]*second;
          // Here volume=M/rho and first/second are normalized NFW profiles
          // at those masses. Each product is one halo's density factor.
          // SIMD evaluates both products with one mass in each lane.
          // Read adjacent M/rho values into low/high lanes. These are
          // halo volumes, converting normalized u into a density leg.
          const v2d vvolume = simde_mm_loadu_pd(volume+node);

          // Put the matching NFW values in the same lane order. set_pd
          // takes the high-mass neighbor first, not the low lane first.
          const v2d vprofile = simde_mm_set_pd(second, first);

          // Each lane becomes its own (M/rho)*u(k|M), with units L^3.
          const v2d vweighted = simde_mm_mul_pd(vvolume, vprofile);

          // Write those two density legs to the caller-owned mass row.
          simde_mm_storeu_pd(output+node, vweighted);
        } else {
          output[node] = volume[node]*first;
        }
      }
    }
  }

  free(amplitude);
  free(work);
}
