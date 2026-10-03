#include <math.h>
#include <stdlib.h>

#include "halo_cov.h"
#include "cosmolike/basics.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/halo.h"
#include "cosmolike/structs.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

typedef simde__m128d v2d; // two independent wavenumbers or wavenumber pairs

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
    double** const* moments        // five pair-moment roles
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
  const int npair = nk*(nk+1)/2;
  const double rho_cb = cosmology.rho_crit*omega_halo_field();
  if (!isfinite(rho_cb)
      || rho_cb <= 0.0) {
    log_fatal("halo_moments_cov needs a positive initialized cb density");
    exit(1);
  }
  const double mass_min = exp(lnm_edges[0]);
  double** mass = (double**) malloc2d(3, nmass);
  int** pairs = (int**) malloc2d_int(2, npair);
  gsl_integration_glfixed_table* rule = malloc_gslint_glfixed(nquad);
  for (int panel=0; panel<npanel; panel++) {
    for (int node=0; node<nquad; node++) {
      const int index = panel*nquad+node;
      double lnm;      // logarithmic mass abscissa
      double measure;  // positive dlnM quadrature measure
      gsl_integration_glfixed_point(lnm_edges[panel], lnm_edges[panel+1],
                                    node, &lnm, &measure, rule);
      mass[0][index] = exp(lnm);
      mass[1][index] = mass[0][index]/rho_cb;
      mass[2][index] = measure/mass[1][index];
    }
  }
  gsl_integration_glfixed_table_free(rule);
  int pair = 0;
  for (int first=0; first<nk; first++) {
    for (int second=first; second<nk; second++) {
      pairs[0][pair] = first;
      pairs[1][pair] = second;
      pair++;
    }
  }

  // --- 2. HALO STATISTICS AT EACH (a,M), WITH SERIAL TABLE WARMUP ---

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
  #pragma omp parallel for schedule(static)
  for (int row=0; row<na; row++) {
    double resolved = 0.0;
    for (int node=0; node<nmass; node++) {
      const double m = mass[0][node];
      const double volume = mass[1][node];
      const double nu = 1.686/sqrt(sigma2(m, a[row]));
      const double bias = hb1nu(nu, a[row]);
      const double number = mass[2][node]*fnu(nu, a[row])*nu
                            *dlognudlogm(m, a[row]);
      const double volume2 = volume*volume;
      weights[0][row][node] = number*bias*volume;
      weights[1][row][node] = number*volume2;
      weights[2][row][node] = number*bias*volume2;
      weights[3][row][node] = weights[2][row][node]*volume;
      weights[4][row][node] = weights[1][row][node]*volume2;
      concentration[row][node] = conc(m, a[row]);
      resolved += weights[0][row][node];
    }
    completion[row] = 1.0-resolved;
    concentration[row][nmass] = conc(mass_min, a[row]);
  }

  // --- 3. SHARED PROFILES, INCLUDING THE M_min COMPLETION PROFILE ---

  double*** profile = (double***) malloc3d(na, nk, nmass+1);
  #pragma omp parallel for collapse(2) schedule(static)
  for (int row=0; row<na; row++) {
    for (int index=0; index<nk; index++) {
      for (int node=0; node<=nmass; node++) {
        const double m = node < nmass ? mass[0][node] : mass_min;
        double value = 1.0;
        if (k[row][index] > 0.0) {
          value = u_nfw_c(concentration[row][node], k[row][index], m,
                          a[row]);
        }
        profile[row][index][node] = value;
      }
    }
  }

  // --- 4a. I11: ONE PROFILE AND ONE BIAS, PLUS UNRESOLVED MASS ---

  #pragma omp parallel for collapse(2) schedule(static)
  for (int row=0; row<na; row++) {
    for (int index=0; index<nk; index+=2) {
      const int next = index+1 < nk ? index+1 : index;
      const double* restrict u0 = profile[row][index];
      const double* restrict u1 = profile[row][next];
      const double* restrict weight = weights[0][row];
      v2d vsum = simde_mm_setzero_pd();
      for (int node=0; node<nmass; node++) {
        const v2d vu = simde_mm_set_pd(u1[node], u0[node]);
        const v2d vw = simde_mm_set1_pd(weight[node]);
        vsum = simde_mm_fmadd_pd(vu, vw, vsum);
      }
      double result[2];
      simde_mm_storeu_pd(result, vsum);
      i11[row][index] = result[0]+completion[row]*u0[nmass];
      if (index+1 < nk) {
        i11[row][next] = result[1]+completion[row]*u1[nmass];
      }
    }
  }

  // --- 4b. FIVE PAIR MOMENTS FROM THE SAME FOUR PROFILE READS ---

  #pragma omp parallel for collapse(2) schedule(static)
  for (int row=0; row<na; row++) {
    for (int index=0; index<npair; index+=2) {
      const int next = index+1 < npair ? index+1 : index;
      const double* restrict uk0 = profile[row][pairs[0][index]];
      const double* restrict uq0 = profile[row][pairs[1][index]];
      const double* restrict uk1 = profile[row][pairs[0][next]];
      const double* restrict uq1 = profile[row][pairs[1][next]];
      v2d vsums[5];
      for (int role=0; role<5; role++) {
        vsums[role] = simde_mm_setzero_pd();
      }
      for (int node=0; node<nmass; node++) {
        const v2d vk = simde_mm_set_pd(uk1[node], uk0[node]);
        const v2d vq = simde_mm_set_pd(uq1[node], uq0[node]);
        const v2d vproduct = simde_mm_mul_pd(vk, vq);
        const v2d vkqq = simde_mm_mul_pd(vproduct, vq);
        const v2d vkkq = simde_mm_mul_pd(vproduct, vk);
        const v2d vkkqq = simde_mm_mul_pd(vproduct, vproduct);
        const v2d vw02 = simde_mm_set1_pd(weights[1][row][node]);
        const v2d vw12 = simde_mm_set1_pd(weights[2][row][node]);
        const v2d vw13 = simde_mm_set1_pd(weights[3][row][node]);
        const v2d vw04 = simde_mm_set1_pd(weights[4][row][node]);
        vsums[0] = simde_mm_fmadd_pd(vproduct, vw02, vsums[0]);
        vsums[1] = simde_mm_fmadd_pd(vproduct, vw12, vsums[1]);
        vsums[2] = simde_mm_fmadd_pd(vkqq, vw13, vsums[2]);
        vsums[3] = simde_mm_fmadd_pd(vkkq, vw13, vsums[3]);
        vsums[4] = simde_mm_fmadd_pd(vkkqq, vw04, vsums[4]);
      }
      for (int role=0; role<5; role++) {
        double result[2];
        simde_mm_storeu_pd(result, vsums[role]);
        moments[role][row][index] = result[0];
        if (index+1 < npair) {
          moments[role][row][next] = result[1];
        }
      }
    }
  }
  free(profile);
  free(completion);
  free(concentration);
  free(weights);
  free(pairs);
  free(mass);
}
