#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <stddef.h>
#include <stdlib.h>

#include "gaussian_cov.h"
#include "log.c/src/log.h"

#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

// v2d holds two doubles in positions called lanes. SIMD applies the same
// operation to both, here two covariance columns with separate ell sums.
// A fused multiply-add (FMA) evaluates a*b+c with one rounding when
// supported directly by the processor. This differs from rounding a*b
// first and then adding c; the calls below retain the chosen operations.
// Unaligned loads/stores accept addresses that are not multiples of 16
// bytes. They still require two valid adjacent doubles in the array.
typedef simde__m128d v2d;

// ============================================================================
// GAUSSIAN COVARIANCE AT INTEGER MULTIPOLES
// ============================================================================

// ---------------------------------------------------------------------------
// Covariance of two angular power spectra C_AB and C_CD at the same ell.
//
// PHYSICAL DERIVATION & LOGIC FLOW
// A Gaussian four-point expectation separates into products of two-point
// expectations. Subtracting <C_AB><C_CD> leaves two pairings: AC with BD,
// and AD with BC. For one integer ell, their covariance is
//
//   G_AB,CD(ell) = [(C_AC + N_AC)(C_BD + N_BD)
//                + (C_AD + N_AD)(C_BC + N_BC)] / [(2 ell + 1) fsky].
//
// There are 2 ell + 1 spherical-harmonic modes on the full sky. The
// fsky approximation reduces their number in proportion to survey area;
// it does not describe general mask-induced coupling between multipoles.
// This is Krause & Eifler (2017), arXiv:1601.05779, Appendix A, with
// integer multipoles (Delta ell = 1; Eq. 28 in the arXiv v1 text).
//
// C is signal only, with any shear transfer factors already applied.
// N is white noise in the observed field: 1/n for number density and
// sigma_component^2/n for shear, where n is per steradian. Noise has no
// extra shear transfer factor. For independent catalogs N_AC vanishes
// unless A and C denote the same field. All four signal spectra are
// required, even if some pairs are excluded from the data vector.
//
// A real-space calculation requests only CC + CN + NC here and adds
// pure noise from pair counts. Summing white-noise NN to a finite ell
// cutoff poorly represents its sharply localized angular covariance.
// Expand the products explicitly: subtracting NN from (C+N)(C+N) loses
// small signal terms through rounding when noise dominates.
//
// Parameters:
//   cross_spectra - rows AC, BD, AD, BC; each has nell consecutive nodes
//                  starting at ell_min. Signed cross-spectra are allowed.
//   cross_noise   - white-noise spectra in the same four-row order
//   include_noise_noise - 0 omits NN; 1 includes NN for harmonic outputs
//   gaussian      - output [nell], overwritten, distinct from all inputs
//   ell_min, nell, fsky - multipole grid and positive survey sky fraction
//
// Cache invalidation:
// No cache or allocation. The caller supplies current spectra and noise.
// Thread safety:
// No global state and no lazy table reads. A standalone call distributes
// multipoles among workers. Inside an existing parallel region, the calling
// worker computes this entire block, using its own output. No nested team
// is started; other workers can calculate different observable blocks.
// ---------------------------------------------------------------------------
void gaussian_wick_cov(
    const int ell_min,                  // first integer multipole
    const int nell,                     // number of consecutive multipoles
    const double fsky,                  // survey area / (4 pi)
    const double* const* cross_spectra, // rows AC, BD, AD, BC
    const double* cross_noise,          // white noise in the same order
    const int include_noise_noise,      // 0 omits NN, 1 includes NN
    double* gaussian                    // output harmonic covariance
  )
{
  if (ell_min < 0
      || nell < 1
      || !isfinite(fsky)
      || fsky <= 0.0
      || fsky > 1.0
      || (include_noise_noise != 0
          && include_noise_noise != 1)) {
    log_fatal("gaussian_wick_cov needs ell_min >= 0, nell > 0, "
              "finite 0 < fsky <= 1, and include_noise_noise = 0 or 1");
    exit(1);
  }

  // --- 1. NAME THE FOUR SPECTRA USED BY THE TWO WICK PAIRINGS ---

  // The order is AC, BD, AD, BC; these can include cross-bin spectra
  // absent from the data vector. Each row spans the same ell grid.
  const double* restrict cl_ac = cross_spectra[0];
  const double* restrict cl_bd = cross_spectra[1];
  const double* restrict cl_ad = cross_spectra[2];
  const double* restrict cl_bc = cross_spectra[3];

  // White noise is independent of ell and vanishes for distinct catalogs.
  const double noise_ac = cross_noise[0];
  const double noise_bd = cross_noise[1];
  const double noise_ad = cross_noise[2];
  const double noise_bc = cross_noise[3];

  // Fourier covariance retains NN; real-space covariance adds its exact
  // pair-count contribution later instead of summing a truncated NN tail.
  double pure_noise = 0.0;
  if (include_noise_noise) {
    pure_noise = noise_ac*noise_bd + noise_ad*noise_bc;
  }

  // --- 2. COMBINE SIGNAL AND NOISE AT EACH MULTIPOLE ---

  // Gaussian fluctuations connect the two measured spectra through AC*BD
  // and AD*BC. Noise contributes to those same pairings when fields coincide.
  // Averaging more independent modes reduces the covariance, which explains
  // the division by fsky*(2*ell+1). In this approximation different ell
  // values do not couple, so one worker can compute each ell independently.
  // The four input rows may coincide for auto spectra; the output is separate.
  #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
  for (int node=0; node<nell; node++) {
    // Keep CC and the two CN+NC contributions explicit. This avoids
    // recovering small signal terms by subtracting a large NN afterward.
    const double signal = cl_ac[node]*cl_bd[node] + cl_ad[node]*cl_bc[node];
    const double mixed_ac_bd = cl_ac[node]*noise_bd + noise_ac*cl_bd[node];
    const double mixed_ad_bc = cl_ad[node]*noise_bc + noise_ad*cl_bc[node];

    // The survey samples approximately fsky*(2 ell+1) independent modes.
    const double mode_count = (2.0*(ell_min + (double) node) + 1.0)*fsky;

    gaussian[node] = (signal + mixed_ac_bd + mixed_ad_bc + pure_noise)
                    /mode_count;
  }
}



// ============================================================================
// PROJECTION INTO ANGULAR BINS OR MULTIPOLE BANDS
// ============================================================================

// ---------------------------------------------------------------------------
// Apply a left and a right linear binning operator to diagonal G(ell).
//
// If a measured row is x_i = sum_ell K_i(ell) C(ell), linearity gives
//
//   Cov(x_i, y_j) = sum_ell K_left[i][ell] G(ell) K_right[j][ell].
//
// For real-space correlations K contains the bin-averaged spherical
// kernel and its (2 ell + 1)/(4 pi) normalization. For Fourier bands K
// contains normalized band weights. No extra weights or ell spacing are
// inserted here. In particular this is a sum over integers, not an
// integral on a logarithmic ell grid. The spherical real-space operators
// are described by Friedrich et al. (2021), arXiv:2012.08568, Section 4.
//
// First form weighted_left[i][ell] = K_left[i][ell] G(ell). This product
// is shared by every right-hand bin and need only be computed once.
// Then each matrix element is one dot product of two contiguous rows.
//
// Parameters:
//   nleft, nright, nell - strictly positive array dimensions
//   kernel_left, kernel_right - row pointers; physical rows have nell values
//   gaussian - harmonic covariance [nell], including whichever noise terms
//              the caller requested from gaussian_wick_cov
//   weighted_left - scratch [nleft][nell], owned and reused by the caller
//   covariance - output [nleft][nright], overwritten
//
// All rows must use the same multipole grid, including its first ell.
// Scratch and output must not overlap each other or any input. Their rows
// must also be disjoint: different workers can write different rows.
// Read-only input rows may coincide. Row pointers support padded strides.
// Cache invalidation:
// No static cache. Geometry owners retain kernels and scratch across calls.
// Thread safety:
// A standalone call distributes output tiles among workers. If a caller
// already distributes observable blocks, each worker instead handles all
// tiles of its own block with private scratch. No nested team is started.
// The increasing-ell sum never crosses workers in either arrangement.
// ---------------------------------------------------------------------------
void gaussian_project_cov(
    const int nleft,                    // left bin count
    const int nright,                   // right bin count
    const int nell,                     // multipole count
    const double* const* kernel_left,   // left projection rows
    const double* const* kernel_right,  // right projection rows
    const double* gaussian,             // harmonic covariance
    double* const* weighted_left,       // caller-owned weighted rows
    double* const* covariance           // output block
  )
{
  if (nleft < 1
      || nright < 1
      || nell < 1) {
    log_fatal("gaussian_project_cov needs positive nleft, nright and nell");
    exit(1);
  }

  // --- 1. WEIGHT EACH LEFT OPERATOR ONCE ---

  // Each covariance entry sums K_left*G*K_right over ell. The product
  // K_left*G is identical for all right bins, so compute it once per left
  // row and reuse it. Workers write separate scratch rows; G is shared.
  #pragma omp parallel for if(!omp_in_parallel()) schedule(static)
  for (int left=0; left<nleft; left++) {
    const double* restrict kernel = kernel_left[left];
    double* restrict weighted = weighted_left[left];

    for (int node=0; node<nell; node++) {
      weighted[node] = kernel[node]*gaussian[node];
    }
  }

  // --- 2. PROJECT GROUPS OF FOUR LEFT AND FOUR RIGHT BINS ---

  // A four-by-four group shares kernel reads among sixteen outputs. Each
  // vector holds two right-bin results, not pieces of a single ell sum.
  // Thus every result still adds ell=0,1,2,... in the scalar order. There
  // is no sum across vector lanes and no reduction across OpenMP workers.
  enum { tile_rows = 4 }; // rows per group; columns are two pairs of lanes

  // Transforming both observables requires every left/right bin pairing.
  // Group four bins on each side so a loaded kernel value can contribute to
  // several entries before moving to the next ell. One worker owns all
  // sixteen sums in that group. Each SIMD vector accumulates two distinct
  // covariance entries, one per lane; adding lanes would incorrectly mix
  // different measured angular bins. Each lane therefore keeps its own sum.
  #pragma omp parallel for collapse(2) if(!omp_in_parallel()) schedule(static)
  for (int left=0; left<nleft; left+=tile_rows) {
    // Pair this group of left bins with every group of right bins. The
    // SIMD lanes produce distinct covariance columns, never a combined sum.
    for (int right=0; right<nright; right+=4) {
      // The last group can have fewer than four columns. Repeating its
      // final valid kernel keeps every read in bounds; unused results are
      // discarded below. The same rule covers a partial group of rows.
      const int right1 = right+1 < nright ? right+1 : nright-1;
      const int right2 = right+2 < nright ? right+2 : nright-1;
      const int right3 = right+3 < nright ? right+3 : nright-1;
      const double* restrict kernel0 = kernel_right[right];
      const double* restrict kernel1 = kernel_right[right1];
      const double* restrict kernel2 = kernel_right[right2];
      const double* restrict kernel3 = kernel_right[right3];

      // One left weight multiplies all four right kernels at this ell.
      // Each row therefore needs two independent two-column accumulators.
      const double* weighted_rows[tile_rows];
      v2d vtotals_low[tile_rows];
      v2d vtotals_high[tile_rows];

      for (int row=0; row<tile_rows; row++) {
        const int index = left+row < nleft ? left+row : nleft-1;
        weighted_rows[row] = weighted_left[index];

        // Start the sums for right-bin columns 0 and 1 at zero. The two
        // lanes will keep independent ell sums for this left-bin row.
        vtotals_low[row] = simde_mm_setzero_pd();

        // Start separate sums for columns 2 and 3 at zero as well.
        vtotals_high[row] = simde_mm_setzero_pd();
      }

      // Walk the multipoles once for this block. Read its four right
      // kernels, then update all left rows; each SIMD lane accumulates
      // weighted_left*kernel_right for its own covariance column.
      for (int node=0; node<nell; node++) {
        // Each 128-bit vector holds two doubles. set_pd puts its last
        // argument in lane 0: low holds columns 0,1 and high holds 2,3.
        // Keeping the pairs separate also avoids repeatedly splitting and
        // joining a 256-bit value on machines with 128-bit vector registers.
        const v2d vkernels_low = simde_mm_set_pd(
          kernel1[node], kernel0[node]);

        // set_pd takes lane 1 first: pack the kernels of columns 2 and 3
        // into lanes 0 and 1 of the second vector, in that order.
        const v2d vkernels_high = simde_mm_set_pd(
          kernel3[node], kernel2[node]);

        // At this ell, one left-bin weight updates four covariance entries.
        // Two SIMD vectors hold columns 0/1 and 2/3, with independent sums.
        for (int row=0; row<tile_rows; row++) {
          const double* restrict weighted = weighted_rows[row];

          // Scalar equation, once for each right bin:
          //   total += weighted_left[left+row][ell] * kernel_right[right][ell].
          // set1_pd repeats the left-bin weight on both lanes.
          const v2d vweight = simde_mm_set1_pd(weighted[node]);

          // fmadd evaluates weight*kernel + total with one rounding on
          // native FMA/NEON, matching the scalar fma rather than rounding
          // the product first. No lane is added to a different lane.
          vtotals_low[row] = simde_mm_fmadd_pd(
            vweight, vkernels_low, vtotals_low[row]);

          // Apply the same weight*kernel+total update to columns 2 and 3.
          // fmadd again has one rounding on native FMA hardware and keeps
          // the two column sums separate.
          vtotals_high[row] = simde_mm_fmadd_pd(
            vweight, vkernels_high, vtotals_high[row]);
        }
      }

      // Copy the completed SIMD sums into the valid output rows and
      // columns. Repeated edge inputs contributed only discarded lanes.
      for (int row=0;
           row<tile_rows
           && left+row<nleft;
           row++) {
        double results[4];

        // Copy columns 0 and 1 to results[0] and results[1]. storeu needs
        // two valid doubles, but no special vector alignment of the array.
        simde_mm_storeu_pd(results, vtotals_low[row]);

        // Copy the other two lanes into results[2] and results[3]. storeu
        // also allows this offset address without vector alignment.
        simde_mm_storeu_pd(results+2, vtotals_high[row]);

        for (int column=0;
             column<4
             && right+column<nright;
             column++) {
          covariance[left+row][right+column] = results[column];
        }
      }
    }
  }
}



// ============================================================================
// PURE NOISE FROM ANGULAR PAIR COUNTS
// ============================================================================

// ---------------------------------------------------------------------------
// Angular area available for ordered pairs in a uniform survey.
//
// A ring between theta_low and theta_high has solid angle
//   A_ring = 2 pi (cos(theta_low) - cos(theta_high)).
// Multiplying by survey area gives a pair area in sr^2. With densities
// n_A and n_B per steradian, n_A n_B times this area is the expected
// ordered pair count. An auto-catalog pair occurs twice in this count.
// This neglects survey edges. A mask or measured random-pair count must
// replace this area at the caller, not be silently normalized here.
//
// The equivalent sine product avoids subtracting two numbers close to
// one for small angular separations: cos(a)-cos(b) =
// 2 sin((a+b)/2) sin((b-a)/2). This matters for narrow, small-angle bins.
//
// Parameters: area_sr in (0,4 pi]; 0 <= theta_low_rad < theta_high_rad <= pi.
// Returns: pair area in sr^2. Cache invalidation: no cache.
// Thread safety: pure scalar calculation, callable from any thread.
// ---------------------------------------------------------------------------
double annulus_pair_area_cov(
    const double area_sr,               // survey solid angle in sr
    const double theta_low_rad,         // lower separation in radians
    const double theta_high_rad         // upper separation in radians
  )
{
  if (!isfinite(area_sr)
      || area_sr <= 0.0
      || area_sr > 4.0*M_PI
      || !isfinite(theta_low_rad)
      || !isfinite(theta_high_rad)
      || theta_low_rad < 0.0
      || theta_high_rad > M_PI
      || theta_high_rad <= theta_low_rad) {
    log_fatal("annulus_pair_area_cov needs finite 0 < area_sr <= 4 pi "
              "and 0 <= theta_low_rad < theta_high_rad <= pi");
    exit(1);
  }

  const double midpoint = 0.5*(theta_high_rad + theta_low_rad);
  const double half_width = 0.5*(theta_high_rad - theta_low_rad);
  return area_sr*4.0*M_PI*sin(midpoint)*sin(half_width);
}


// ---------------------------------------------------------------------------
// Pure-noise covariance of two correlation estimators in the same theta bin.
//
// Catalogs are independent, though their redshift distributions can overlap.
// fields[] contains catalog IDs in order A,B,C,D. Give each catalog one
// unique ID across both lens and source samples; reuse that ID whenever
// the same catalog appears again. For gamma_t, A and C must be lens
// catalogs and B and D source catalogs.
// N_g = 1/n_g and N_s = sigma_component^2/n_s, with densities per sr.
//
// For w_AB and w_CD the two surviving noise pairings give
//   (delta_AC delta_BD + delta_AD delta_BC) N_A N_B / pair_area.
// For gamma_t only one shear component is measured and only the direct
// lens-lens/source-source pairing exists. For xi_+ or xi_-, the tangential
// and cross components each contribute the same variance: multiply the
// w expression by two. In xi_+ versus xi_-, these two contributions cancel.
// Other probe combinations have zero pure noise for independent catalogs.
//
// This is the pair-count form of the sampling terms in Friedrich et al.
// (2021), arXiv:2012.08568, Sections 6.10.1--6.10.3. For auto shear it
// gives 4 sigma_component^4 / N_ordered, not the expression for a
// dispersion defined as the sum of both ellipticity-component variances.
//
// Parameters:
//   probe_left, probe_right - one of the four probe_cov enum values
//   fields - [A,B,C,D] catalog IDs, all nonnegative
//   noise_ab - [N_A,N_B], finite and nonnegative
//   pair_area_sr2 - positive area in sr^2; may include a mask correction
// Returns: covariance for the same angular bin. For disjoint angular bins
// the caller inserts zero. Overlapping angular bins need their overlap
// pair counts and are not covered by this diagonal-bin convention.
// Cache invalidation: no cache. Thread safety: no shared state.
// ---------------------------------------------------------------------------
double gaussian_noise_pair_cov(
    const enum probe_cov probe_left,    // left estimator
    const enum probe_cov probe_right,   // right estimator
    const int* fields,                  // A, B, C, D catalog IDs
    const double* noise_ab,             // N_A and N_B
    const double pair_area_sr2          // angular ordered-pair area
  )
{
  if (probe_left < XI_PLUS_COV
      || probe_left > W_THETA_COV
      || probe_right < XI_PLUS_COV
      || probe_right > W_THETA_COV
      || !isfinite(pair_area_sr2)
      || pair_area_sr2 <= 0.0
      || !isfinite(noise_ab[0])
      || !isfinite(noise_ab[1])
      || noise_ab[0] < 0.0
      || noise_ab[1] < 0.0
      || fields[0] < 0
      || fields[1] < 0
      || fields[2] < 0
      || fields[3] < 0) {
    log_fatal("gaussian_noise_pair_cov needs supported probes, nonnegative "
              "field IDs, finite nonnegative noise and positive pair area");
    exit(1);
  }

  if (probe_left != probe_right) {
    return 0.0;
  }

  // Noise correlates repeated measurements of the same objects. Each
  // boolean below is a product of two catalog Kronecker deltas: the
  // direct pairing checks A=C and B=D; the exchanged one checks A=D, B=C.
  const int direct = fields[0] == fields[2]
                     && fields[1] == fields[3];
  const int exchanged = fields[0] == fields[3]
                        && fields[1] == fields[2];
  const double pair_variance = noise_ab[0]*noise_ab[1]/pair_area_sr2;

  if (probe_left == GAMMA_T_COV) {
    return direct*pair_variance;
  }
  if (probe_left == W_THETA_COV) {
    return (direct + exchanged)*pair_variance;
  }
  return 2.0*(direct + exchanged)*pair_variance;
}
