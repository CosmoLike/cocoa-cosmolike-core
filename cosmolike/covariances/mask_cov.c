#include <math.h>
#include <stdlib.h>

#include "mask_cov.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

// SIMD (single instruction, multiple data) applies one arithmetic
// operation to several numbers at once. v2d is a 128-bit vector of two
// doubles; its two positions, lanes 0 and 1, hold separate angular-bin
// sums throughout this file and are never added to each other.
//
// SIMDe calls are named simde_mm_<operation>_pd: "mm" marks a 128-bit
// vector and "pd" means packed doubles. SIMDe translates each call into
// the matching x86 SSE2/FMA instruction, the NEON instruction on 64-bit
// ARM (Apple silicon), or portable C when neither is available.
//
// A fused multiply-add (FMA) evaluates a*b+c with one rounding when
// supported directly by the processor. This differs from rounding a*b
// first and then adding c, which is what SIMDe does when the build target
// has no FMA instruction. The calls below retain the chosen operations.
// Unaligned stores (the "u" in storeu) accept addresses that are not
// multiples of 16 bytes. They still require two valid adjacent doubles.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Ordered-pair angular area from a common survey footprint.
//
// PHYSICAL DERIVATION & LOGIC FLOW
// For an unclustered catalog with density n per steradian, the expected
// number of objects in dOmega is n W(nhat) dOmega. W is the survey mask,
// 1 inside the footprint and 0 outside. Integrating two such positions
// with separation inside a bin gives
//
//   N_pair = n_A n_B A_pair,
//   A_pair = integral dOmega_1 dOmega_2 W_1 W_2 1_bin(theta_12).
//
// Two standard steps turn this into a sum over mask multipoles. First,
// expand the bin indicator in Legendre polynomials of cos(theta_12):
//
//   1_bin(theta_12) = sum_L (2L+1)/2 P_L(cos(theta_12))
//                     integral_bin sin(theta) dtheta P_L(cos(theta)).
//
// Second, the addition theorem for spherical harmonics,
// P_L(cos(theta_12)) = 4 pi/(2L+1) sum_M Y_LM(nhat_1) Y*_LM(nhat_2),
// separates the two positions, so each position integral becomes a mask
// harmonic W_LM = integral dOmega W Y*_LM. For a common mask with raw
// power C_L^W = sum_M |W_LM|^2/(2L+1), this gives
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
// C_0^W = |W_00|^2 = area_sr^2/(4 pi), because Y_00 = 1/sqrt(4 pi) gives
// W_00 = area_sr/sqrt(4 pi). Do not divide it by C_0, area_sr or f_sky
// here. A full-sky mask has C_0=4 pi and all other modes zero; with
// K_bin,0 = 1/(4 pi) the result is 4 pi * 2 pi Delta_x = 8 pi^2 Delta_x,
// the uniform ordered-pair area. For a finite mask the angular boundary
// removes pairs relative to area_sr * 2 pi Delta_x, the edge-free value
// of annulus_pair_area_cov, which increases the pure-noise variance.
// The input K must retain L=0 and L=1 even if the signal estimator removes
// those modes: these L describe the footprint geometry, not the
// cosmological signal.
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
// Output is overwritten in sr^2. Each loop iteration owns two adjacent
// bins; static scheduling gives every worker a contiguous run of such
// iterations. Each SIMD lane retains its complete L sum in the same order
// for every thread count. There is no allocation, cache, BLAS call or
// hidden mask renormalization, and no lazy table is read, so no serial
// warm-up is needed. The loop always starts its own parallel region: it
// has no if(!omp_in_parallel()) clause, unlike gaussian_wick_cov and
// gaussian_project_cov, so it is meant to be called from serial code.
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
  // --- 1. CHECK THE RAW MASK AND THE ANGULAR BINS ---

  if (nbin < 1
      || nmask < 1
      || !isfinite(area_sr)
      || area_sr <= 0.0
      || area_sr > 4.0*M_PI) {
    log_fatal("mask_pair_area_cov needs positive sizes and area in (0,4pi]");
    exit(1);
  }

  // Validate the raw spectrum before interpreting its monopole as area.
  // A power spectrum is a sum of squared moduli, so it cannot be negative.
  // Here the loop index ell is the mask multipole L.
  for (int ell=0; ell<nmask; ell++) {
    if (!isfinite(mask_cl[ell])
        || mask_cl[ell] < 0.0) {
      log_fatal("mask_pair_area_cov: invalid raw mask power at L=%d", ell);
      exit(1);
    }
  }

  // Raw-mask guard: C_0^W = |W_00|^2 = area_sr^2/(4 pi), as derived in the
  // function header. Dividing a raw spectrum by C_0, area_sr or f_sky
  // changes its monopole (unless that divisor is 1), so a normalized
  // spectrum, or one computed for a different footprint area, is rejected.
  // The comparison allows a 1e-8 relative difference for rounding.
  const double monopole = area_sr*area_sr/(4.0*M_PI);
  if (fabs(mask_cl[0]/monopole-1.0) > 1.e-8) {
    log_fatal("mask_pair_area_cov needs raw C0=area^2/(4pi)");
    exit(1);
  }

  // Ordered boundaries are needed for a positive spherical annulus area.
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

  // --- 2. MASK SUM AND PAIR AREA, TWO BINS AT A TIME ---

  // A pair is observable only when both positions lie inside the footprint.
  // The mask spectrum describes how the availability of two positions varies
  // with separation: A(theta) = sum_L (2L+1) C_L^W P_L(cos(theta)) is the
  // footprint area weighted by the fraction of each point's ring of radius
  // theta that also lies inside (area_sr at theta = 0, 4 pi on the full
  // sky). Its sum against K_bin,L, sum_L K_bin,L C_L^W, is the average of
  // A(theta)/(4 pi) over the bin's area element sin(theta) dtheta, and the
  // factor 8*pi^2*Delta_x below converts the average to the ordered-pair
  // area. Each iteration computes two bins. SIMD accumulates their
  // separate multipole sums, one per lane, so bins are never mixed; each
  // lane adds L = 0, 1, ..., nmask-1 in increasing order.
  #pragma omp parallel for schedule(static)
  for (int bin=0; bin<nbin; bin+=2) {
    // The last group may duplicate its read-only row. Store that bin once.
    // With an odd nbin the final iteration has next = bin, so lane 1
    // repeats lane 0 and is discarded before any store to pair_area.
    const int next = bin+1 < nbin ? bin+1 : bin;
    const double* restrict kernel0 = scalar_kernel[bin];
    const double* restrict kernel1 = scalar_kernel[next];

    // Scalar equivalent for either angular bin b=bin,next:
    //   sum = 0;
    //   for (int ell=0; ell<nmask; ell++) {
    //     sum = fma(scalar_kernel[b][ell], mask_cl[ell], sum);
    //   }
    // The later area factor converts this mask-correlation average into
    // an ordered-pair area. SIMD accumulates two bin averages separately:
    // lane 0 holds the sum of bin and lane 1 the sum of next.

    // scalar: sum_bin = 0.0; sum_next = 0.0;
    // setzero_pd returns a v2d with 0.0 in both lanes: lane 0 starts the
    // sum of K_bin,L*C_L^W for bin, lane 1 the sum for next.
    v2d vsum = simde_mm_setzero_pd();

    // Add one mask multipole to both bin sums. SIMD reuses C_L^W across
    // lanes, but multiplies it by a different bin kernel in each lane.
    // scalar: sum_bin  = fma(kernel0[ell], mask_cl[ell], sum_bin);
    //         sum_next = fma(kernel1[ell], mask_cl[ell], sum_next);
    for (int ell=0; ell<nmask; ell++) {
      // set_pd(high, low) takes lane 1 first: lane 0 = kernel0[ell] =
      // K_bin,L and lane 1 = kernel1[ell] = K_next,L, the scalar operators
      // of two different angular bins at this mask multipole.
      const v2d vkernel = simde_mm_set_pd(kernel1[ell], kernel0[ell]);

      // set1_pd copies the one mask power C_L^W = mask_cl[ell] into both
      // lanes: both bins see the same footprint.
      const v2d vmask = simde_mm_set1_pd(mask_cl[ell]);

      // fmadd_pd(x, y, z) evaluates x*y + z lane by lane, with one rounding
      // on native FMA hardware (x86 FMA, ARM NEON): lane 0 is
      // fma(kernel0[ell], mask_cl[ell], sum_bin) and lane 1 is
      // fma(kernel1[ell], mask_cl[ell], sum_next). Each bin adds
      // K_bin,L*C_L^W to its own running sum; lanes are never combined.
      vsum = simde_mm_fmadd_pd(vkernel, vmask, vsum);
    }

    double sum[2]; // sum_L K_bin,L C_L^W for bin [0] and next [1]

    // scalar: sum[0] = sum_bin; sum[1] = sum_next;
    // storeu_pd writes lane 0 to sum[0] and lane 1 to sum[1]; the two bin
    // sums are not added to each other. The unaligned form accepts this
    // two-double stack array without 16-byte alignment.
    simde_mm_storeu_pd(sum, vsum);

    // Recover each bin's pair area from its harmonic sum. The row check
    // discards the repeated lane when there is an odd number of bins.
    // scalar: pair_area[row] = 8 pi^2 Delta_x(row) sum[lane] for each
    // real row = bin + lane.
    for (int lane=0; lane<2; lane++) {
      const int row = bin+lane;
      if (row < nbin) {
        const double lower = edges_rad[row];
        const double upper = edges_rad[row+1];

        // width = cos(lower)-cos(upper) is the annulus area divided by
        // 2 pi. The sine identity retains precision for narrow annuli.
        const double width = 2.0*sin((upper+lower)/2.0)
                               *sin((upper-lower)/2.0);

        // 8 pi^2 width = (4 pi width)(2 pi). The first factor undoes the
        // scalar operator's 1/(4 pi width) normalization; the 2 pi comes
        // from integrating the second object's azimuth around the first.
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
