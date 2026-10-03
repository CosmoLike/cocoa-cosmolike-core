#include <math.h>
#include <stdlib.h>

#include "mask_cov.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

typedef simde__m128d v2d; // two independent angular-bin sums

// ---------------------------------------------------------------------------
// Ordered-pair angular area from a common survey footprint.
//
// PHYSICAL DERIVATION
// For an unclustered catalog with density n per steradian, the expected
// number of objects in dOmega is n W(nhat) dOmega. W is the survey mask.
// Integrating two such positions with separation inside a bin gives
//
//   N_pair = n_A n_B A_pair,
//   A_pair = integral dOmega_1 dOmega_2 W_1 W_2 1_bin(theta_12).
//
// The addition theorem for spherical harmonics turns the angular delta
// function enforcing theta_12=theta into a Legendre series. For a common
// mask with raw power C_L^W = sum_M |W_LM|^2/(2L+1), this gives
//
//   A_pair = 2 pi sum_L (2L+1) C_L^W
//                       integral_bin sin(theta) dtheta P_L(cos(theta)).
//
// See Friedrich et al., arXiv:2012.08568, Appendix C, Eqs. 104-108.
// The supplied scalar-bin operator is
//   K_bin,L = (2L+1)/(4 pi Delta_x) integral_bin sin(theta) dtheta P_L,
// so A_pair = 8 pi^2 Delta_x sum_L C_L^W K_bin,L.
// Delta_x = cos(theta_low)-cos(theta_high) is evaluated with two sines.
//
// RAW MASK, NOT SSC NORMALIZATION
// The same raw spectrum can feed SSC and this function. Its monopole is
// C_0^W = area_sr^2/(4 pi). Do not divide it by C_0, area_sr or f_sky here.
// A full-sky mask has C_0=4 pi and all other modes zero; the result is
// 4 pi * 2 pi Delta_x, the uniform ordered-pair area. For a finite mask
// the angular boundary removes pairs, increasing the pure-noise variance.
// The input K must retain L=0 and L=1 even if the signal estimator removes
// those modes: these L describe the FOOTPRINT, not the cosmological signal.
//
// The result goes directly into gaussian_noise_pair_cov. That function
// accounts for catalog Kronecker factors and shape variance per component;
// no additional factor of two for unordered pairs belongs here.
// This contract assumes a common binary footprint, independent catalogs,
// and uniform noise within it. Different catalog masks and general object
// weights require the corresponding cross/weighted pair counts explicitly.
// The mask band limit must be refined, especially at small separations.
// A nonpositive reconstructed pair area stops; it is never clipped.
//
// OWNERSHIP AND THREADS
// All input arrays are finite and caller-owned. Each scalar-kernel row
// must represent its supplied angular bin and include nmask multipoles.
// Output is overwritten in sr^2. Each worker owns two bins, and each SIMD
// lane retains its complete L sum in the same order for every thread count.
// There is no allocation, cache, BLAS call or hidden mask renormalization.
//
// Cache invalidation:
// No static state. The caller may retain the result until its mask spectrum,
// area, angular bins or scalar-bin quadrature changes.
// ---------------------------------------------------------------------------
void mask_pair_area_cov(
    const int nbin,                     // number of angular bins
    const int nmask,                    // raw mask modes 0..nmask-1
    const double area_sr,               // integral of the common mask
    const double* edges_rad,            // [nbin+1], increasing radians
    const double* mask_cl,              // [nmask], raw mask C_L
    const double* const* scalar_kernel, // [nbin][nmask], full scalar operator
    double* pair_area                   // [nbin], ordered area in sr^2
  )
{
  if (nbin < 1
      || nmask < 1
      || !isfinite(area_sr)
      || area_sr <= 0.0
      || area_sr > 4.0*M_PI) {
    log_fatal("mask_pair_area_cov needs positive sizes and area in (0,4pi]");
    exit(1);
  }
  for (int ell=0; ell<nmask; ell++) {
    if (!isfinite(mask_cl[ell])
        || mask_cl[ell] < 0.0) {
      log_fatal("mask_pair_area_cov: invalid raw mask power at L=%d", ell);
      exit(1);
    }
  }
  const double monopole = area_sr*area_sr/(4.0*M_PI);
  if (fabs(mask_cl[0]/monopole-1.0) > 1.e-8) {
    log_fatal("mask_pair_area_cov needs raw C0=area^2/(4pi)");
    exit(1);
  }
  for (int edge=0; edge<=nbin; edge++) {
    if (!isfinite(edges_rad[edge])
        || edges_rad[edge] < 0.0
        || edges_rad[edge] > M_PI) {
      log_fatal("mask_pair_area_cov: angular edge %d outside [0,pi]", edge);
      exit(1);
    }
    if (edge > 0
        && edges_rad[edge] <= edges_rad[edge-1]) {
      log_fatal("mask_pair_area_cov needs strictly increasing edges");
      exit(1);
    }
  }

  #pragma omp parallel for schedule(static)
  for (int bin=0; bin<nbin; bin+=2) {
    // The last group may duplicate its read-only row. Store that bin once.
    const int next = bin+1 < nbin ? bin+1 : bin;
    const double* restrict kernel0 = scalar_kernel[bin];
    const double* restrict kernel1 = scalar_kernel[next];
    v2d vsum = simde_mm_setzero_pd();
    for (int ell=0; ell<nmask; ell++) {
      const v2d vkernel = simde_mm_set_pd(kernel1[ell], kernel0[ell]);
      const v2d vmask = simde_mm_set1_pd(mask_cl[ell]);
      vsum = simde_mm_fmadd_pd(vkernel, vmask, vsum);
    }
    double sum[2];
    simde_mm_storeu_pd(sum, vsum);
    for (int lane=0; lane<2; lane++) {
      const int row = bin+lane;
      if (row < nbin) {
        const double lower = edges_rad[row];
        const double upper = edges_rad[row+1];
        const double width = 2.0*sin((upper+lower)/2.0)
                               *sin((upper-lower)/2.0);
        const double area = 8.0*M_PI*M_PI*width*sum[lane];
        if (!isfinite(area)
            || area <= 0.0) {
          log_fatal("mask_pair_area_cov: bin %d has no positive pair area; "
                    "check footprint, bins and mask resolution", row);
          exit(1);
        }
        pair_area[row] = area;
      }
    }
  }
}
