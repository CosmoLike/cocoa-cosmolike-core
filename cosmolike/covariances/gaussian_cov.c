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

// SIMD (single instruction, multiple data) applies one arithmetic
// operation to several numbers at once. Each number occupies a vector
// position called a lane. v2d is a 128-bit vector of two doubles, lane 0
// and lane 1. In this file the two lanes always hold two different
// covariance columns, each with its own ell sum; lanes are never added.
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
// Code map, at ell = ell_min + node:
//   signal                    = C_AC C_BD + C_AD C_BC              (CC)
//   mixed_ac_bd + mixed_ad_bc = C_AC N_BD + N_AC C_BD
//                             + C_AD N_BC + N_AD C_BC         (CN + NC)
//   pure_noise                = N_AC N_BD + N_AD N_BC                (NN)
//                               (zero when include_noise_noise = 0)
//   mode_count                = (2 ell + 1) fsky
//
// Parameters:
//   ell_min, nell - the multipoles ell = ell_min, ..., ell_min + nell - 1;
//                   array index node holds ell = ell_min + node
//   fsky          - survey area / (4 pi), in (0, 1]
//   cross_spectra - rows AC, BD, AD, BC, each [nell] on that ell grid.
//                   Signed cross-spectra are allowed.
//   cross_noise   - [4] white-noise powers N_AC, N_BD, N_AD, N_BC in the
//                   normalization of C (1/n or sigma_component^2/n)
//   include_noise_noise - 0 omits NN (real-space split); 1 includes NN
//                   for harmonic band powers
//   gaussian      - output [nell], overwritten, distinct from all inputs,
//                   in the squared units of the input spectra
//
// Cache invalidation:
// No cache or allocation. The caller supplies current spectra and noise.
// Thread safety:
// No global state and no lazy table reads, so no serial warm-up is needed.
// A standalone call distributes multipoles among workers. Inside an active
// parallel region (a team of more than one thread, where omp_in_parallel()
// is true), the calling worker computes this entire block, using its own
// output. No nested team is started; other workers can calculate different
// observable blocks. An inactive enclosing region, for example one whose
// if clause was false, still lets this call start its own team.
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
  // One iteration handles ell = ell_min + node and writes only
  // gaussian[node]; nothing is summed across iterations. The if clause
  // opens a team only when no active parallel region encloses this call.
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
// PHYSICAL DERIVATION & LOGIC FLOW
// If measured rows are x_i = sum_ell K_left[i][ell] C(ell) and
// y_j = sum_ell' K_right[j][ell'] C'(ell'), linearity gives
//
//   Cov(x_i, y_j) = sum_ell sum_ell' K_left[i][ell]
//                   Cov(C(ell), C'(ell')) K_right[j][ell'].
//
// gaussian_wick_cov supplies Cov(C(ell), C'(ell')) = G(ell) delta_ell,ell'
// (different multipoles are uncorrelated), so the Kronecker delta removes
// the sum over ell':
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
// The contraction needs only a shared index and a diagonal weight, so it
// serves other terms too: connected_matrix_cov (assembly_cov.c) passes
// radial shells in place of ell, catalog pair windows as the operators
// and the projected trispectrum times its radial measure as the weight.
// "ell" and "multipole" below then mean that shared node index.
//
// Parameters:
//   nleft, nright, nell - strictly positive array dimensions
//   kernel_left, kernel_right - row pointers; physical rows have nell values
//   gaussian - harmonic covariance [nell], including whichever noise terms
//              the caller requested from gaussian_wick_cov
//   weighted_left - scratch [nleft][nell], owned and reused by the caller
//   covariance - output [nleft][nright], overwritten, in the units of G
//                times those of both operators (the real-space and band
//                operators are dimensionless)
//
// All rows must use the same multipole grid, including its first ell.
// Scratch and output must not overlap each other or any input. Their rows
// must also be disjoint: different workers can write different rows.
// Read-only input rows may coincide. Row pointers support padded strides.
// Cache invalidation:
// No static cache. Geometry owners retain kernels and scratch across calls.
// Thread safety:
// No global state and no lazy table reads, so no serial warm-up is needed.
// A standalone call distributes output tiles among workers. If the call
// sits inside an active parallel region that distributes observable
// blocks, each worker instead handles all tiles of its own block with
// private scratch. No nested team is started. In the standalone case the
// first loop ends with the implicit barrier of its parallel region, so
// every weighted row is complete before any tile reads it.
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
  // One iteration fills the complete row weighted_left[left][0..nell-1]
  // with independent products; nothing is summed in this stage.
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
  // Thus every result still adds its terms in increasing node order,
  // node = 0, 1, 2, ..., starting at the grid's first multipole, exactly
  // as the scalar loop does. There is no sum across vector lanes and no
  // reduction across OpenMP workers.
  enum { tile_rows = 4 }; // rows per group; columns are two pairs of lanes

  // Transforming both observables requires every left/right bin pairing.
  // Group four bins on each side so a loaded kernel value can contribute to
  // several entries before moving to the next ell. One worker owns all
  // sixteen sums in that group. Each SIMD vector accumulates two distinct
  // covariance entries, one per lane; adding lanes would incorrectly mix
  // different measured angular bins. Each lane therefore keeps its own sum.
  // collapse(2) merges the left-group and right-group loops into one list
  // of (left group, right group) tiles; schedule(static) hands each worker
  // a contiguous share of that list.
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
      const double* weighted_rows[tile_rows]; // weighted left rows of group
      v2d vtotals_low[tile_rows];  // per row: sums of columns 0 and 1
      v2d vtotals_high[tile_rows]; // per row: sums of columns 2 and 3

      // Scalar equivalent for one valid output entry (i,j) in this block,
      // with i = left+row and j = right+column (row, column = 0..3):
      //   total = 0;
      //   for (int node=0; node<nell; node++) {
      //     total = fma(weighted_left[i][node], kernel_right[j][node],
      //                 total);
      //   }
      //   covariance[i][j] = total;
      // The left weight already contains the harmonic covariance. Thus
      // this sum transforms the other measurement into its bin as well.
      // The SIMD block evaluates sixteen such entries together, keeping
      // one sum per lane and the same multipole order for every entry:
      // vtotals_low[row] holds the totals of columns 0 and 1 in lanes 0
      // and 1, and vtotals_high[row] those of columns 2 and 3.

      // scalar: total[row][column] = 0.0 for every row and column, and
      // record the weighted_left row that feeds each tile row.
      for (int row=0; row<tile_rows; row++) {
        // Row left+row, or the last valid row when the group is partial;
        // a repeated row is computed but never stored.
        const int index = left+row < nleft ? left+row : nleft-1;
        weighted_rows[row] = weighted_left[index];

        // setzero_pd returns a v2d with 0.0 in both lanes. Lane 0 starts
        // the ell sum of entry (left+row, right), lane 1 that of entry
        // (left+row, right+1): independent sums for this left-bin row.
        vtotals_low[row] = simde_mm_setzero_pd();

        // setzero_pd again: lanes 0 and 1 start the separate sums of
        // entries (left+row, right+2) and (left+row, right+3).
        vtotals_high[row] = simde_mm_setzero_pd();
      }

      // Walk the multipoles once for this block. Read its four right
      // kernels, then update all left rows; each SIMD lane accumulates
      // weighted_left*kernel_right for its own covariance column.
      // scalar: at each node, for row = 0..3 and column = 0..3,
      //   total[row][column] = fma(weighted_left[left+row][node],
      //                            kernel_right[right+column][node],
      //                            total[row][column]);
      for (int node=0; node<nell; node++) {
        // scalar: k[column] = kernel_right[right+column][node] for
        // column = 0..3; kernel0..kernel3 point to those four rows (an
        // edge group repeats its last valid row).
        // set_pd(high, low) builds a v2d from two scalars and puts its
        // last argument in lane 0: lane 0 = kernel0[node] and lane 1 =
        // kernel1[node], the right-bin operators of columns 0 and 1 at
        // this multipole. Keeping column pairs in two 128-bit vectors
        // avoids repeatedly splitting and joining a 256-bit value on
        // machines with 128-bit vector registers.
        const v2d vkernels_low = simde_mm_set_pd(
          kernel1[node], kernel0[node]);

        // set_pd again, lane 1 first: lane 0 = kernel2[node] and lane 1 =
        // kernel3[node], the right-bin operators of columns 2 and 3.
        const v2d vkernels_high = simde_mm_set_pd(
          kernel3[node], kernel2[node]);

        // At this ell, one left-bin weight updates four covariance entries.
        // Two SIMD vectors hold columns 0/1 and 2/3, with independent sums.
        for (int row=0; row<tile_rows; row++) {
          const double* restrict weighted = weighted_rows[row];

          // scalar: w = weighted[node] = K_left[left+row][node] G[node],
          // the left operator already multiplied by the harmonic
          // covariance. set1_pd copies this one number into both lanes,
          // because both columns of a vector share the same left row.
          const v2d vweight = simde_mm_set1_pd(weighted[node]);

          // scalar, for columns 0 and 1 (lanes 0 and 1):
          //   total[row][column] = fma(w, kernel_right[right+column][node],
          //                            total[row][column]);
          // fmadd_pd(x, y, z) evaluates x*y + z lane by lane, with one
          // rounding on native FMA hardware (x86 FMA, ARM NEON), exactly
          // as the scalar fma. No lane is added to a different lane.
          vtotals_low[row] = simde_mm_fmadd_pd(
            vweight, vkernels_low, vtotals_low[row]);

          // scalar, for columns 2 and 3 (lanes 0 and 1 of the second
          // vector):
          //   total[row][column] = fma(w, kernel_right[right+column][node],
          //                            total[row][column]);
          // fmadd_pd again evaluates x*y + z lane by lane, with one rounding
          // per lane on native FMA hardware; the two column sums stay apart.
          vtotals_high[row] = simde_mm_fmadd_pd(
            vweight, vkernels_high, vtotals_high[row]);
        }
      }

      // Copy the completed SIMD sums into the valid output rows and
      // columns. Repeated edge inputs contributed only discarded lanes.
      // scalar: covariance[left+row][right+column] = total[row][column]
      // for every row < nleft and column < nright; the loop bounds below
      // drop the repeated edge rows and columns.
      for (int row=0;
           row<tile_rows
           && left+row<nleft;
           row++) {
        double results[4]; // the four column totals of this left row

        // storeu_pd writes lane 0 to results[0] and lane 1 to results[1]:
        // the entries (left+row, right) and (left+row, right+1). The
        // unaligned form needs two valid doubles but no 16-byte alignment.
        simde_mm_storeu_pd(results, vtotals_low[row]);

        // storeu_pd writes the second vector to results[2] and results[3]:
        // the entries (left+row, right+2) and (left+row, right+3). The
        // offset address results+2 needs no vector alignment either.
        simde_mm_storeu_pd(results+2, vtotals_high[row]);

        // Keep only the columns that exist in the output matrix.
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
// PHYSICAL DERIVATION & LOGIC FLOW
// Place the first object anywhere in the footprint, of solid angle
// area_sr. Its partners at separations between theta_low and theta_high
// fill a ring of solid angle
//   A_ring = 2 pi (cos(theta_low) - cos(theta_high)) = 2 pi Delta_x,
// where 2 pi comes from the partner's azimuth around the first object.
// Multiplying by survey area gives the pair area A_pair = area_sr A_ring
// in sr^2. With densities n_A and n_B per steradian, n_A n_B A_pair is the
// expected ordered pair count. An auto-catalog pair occurs twice in this
// count. On the full sky (area_sr = 4 pi), A_pair = 8 pi^2 Delta_x;
// mask_pair_area_cov returns the same value for a full-sky mask.
// This neglects survey edges, where part of the ring falls outside the
// footprint. A mask or measured random-pair count must replace this area
// at the caller, not be silently normalized here.
//
// The equivalent sine product avoids subtracting two numbers close to
// one for small angular separations: cos(a)-cos(b) =
// 2 sin((a+b)/2) sin((b-a)/2). This matters for narrow, small-angle bins.
// Code map, with a = theta_low_rad and b = theta_high_rad:
// midpoint = (a+b)/2 and half_width = (b-a)/2, so
//   area_sr*4 pi sin(midpoint) sin(half_width) = area_sr*2 pi*Delta_x.
//
// Parameters:
//   area_sr        - survey solid angle in sr, in (0, 4 pi]
//   theta_low_rad  - lower separation in radians, >= 0
//   theta_high_rad - upper separation in radians, in (theta_low_rad, pi]
// Returns: ordered-pair area in sr^2. Cache invalidation: no cache.
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

  // Delta_x = 2 sin(midpoint) sin(half_width). The factor 4 pi below is
  // the ring's azimuthal 2 pi times the 2 of this identity.
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
// PHYSICAL DERIVATION & LOGIC FLOW
// Without clustering, the estimator of one bin averages over the
// N_pair = n_A n_B A_pair expected ordered pairs, A_pair = pair_area_sr2.
// For w, Poisson statistics give the pair count a variance equal to its
// mean, so the noise variance is 1/N_pair. For shear, each pair term
// contains randomly oriented ellipticity components, each of variance
// sigma_component^2, and averaging N_pair such terms divides their
// variance by N_pair. Written with N = 1/n or sigma_component^2/n, both
// cases give, per measured shear component,
//
//   pair_variance = N_A N_B / A_pair.
//
// With the catalog Kronecker deltas direct = delta_AC delta_BD and
// exchanged = delta_AD delta_BC (the two booleans in the code), this gives
//
//   w,       w       : (direct + exchanged) pair_variance
//   gamma_t, gamma_t :  direct              pair_variance
//   xi_+,    xi_+    : 2 (direct + exchanged) pair_variance
//   xi_-,    xi_-    : the same as xi_+, xi_+
//   any other pair   : 0
//
// The direct pairing (A=C, B=D) finds the same pairs in both estimators;
// the exchanged pairing (A=D, B=C) finds them with the roles reversed.
// For an auto-correlation both hold: the two orderings of one physical
// pair are one measurement, and the factor two converts the ordered count
// into the number of distinct pairs, N_unordered = N_ordered/2.
// For gamma_t only one shear component is measured and only the direct
// lens-lens/source-source pairing exists, because a lens catalog never
// shares an ID with a source catalog. For xi_+ or xi_-, the tangential
// and cross components each contribute the same variance: multiply the
// w expression by two. In xi_+ versus xi_-, these two contributions enter
// with opposite signs and cancel. Every other mixed pair of estimators
// contains a randomly oriented, zero-mean ellipticity with no partner in
// the other estimator, so it has zero pure noise for independent catalogs.
//
// This is the pair-count form of the sampling terms in Friedrich et al.
// (2021), arXiv:2012.08568, Sections 6.10.1--6.10.3. For auto shear it
// gives 4 sigma_component^4 / N_ordered. A paper that defines the
// dispersion sigma_e^2 = 2 sigma_component^2 as the sum of both component
// variances writes the same number as sigma_e^4 / N_ordered; N_s here
// must always use the per-component value.
//
// Parameters:
//   probe_left, probe_right - one of the four probe_cov enum values
//   fields - [A,B,C,D] catalog IDs, all nonnegative
//   noise_ab - [N_A,N_B], finite and nonnegative: 1/n_g for a lens
//              catalog, sigma_component^2/n_s for a source catalog
//   pair_area_sr2 - positive ordered-pair area in sr^2, from
//                   annulus_pair_area_cov or mask_pair_area_cov; it must
//                   not be halved for unordered pairs
// Returns: dimensionless covariance for the same angular bin. For disjoint
// angular bins the caller inserts zero. Overlapping angular bins need
// their overlap pair counts and are not covered by this diagonal-bin
// convention.
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

  // Different estimators share no pure noise: xi_+ against xi_- cancels,
  // and every other mixed pair has an unpaired zero-mean ellipticity.
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

  // N_A N_B / A_pair: the variance of one pair term (1 for counts and
  // sigma_component^2 for each ellipticity factor) divided by the expected
  // number of ordered pairs n_A n_B A_pair, per measured shear component.
  const double pair_variance = noise_ab[0]*noise_ab[1]/pair_area_sr2;

  // gamma_t: one shear component and only the direct lens/source pairing.
  if (probe_left == GAMMA_T_COV) {
    return direct*pair_variance;
  }

  // w: both catalog pairings and no shear component.
  if (probe_left == W_THETA_COV) {
    return (direct + exchanged)*pair_variance;
  }

  // xi_+ or xi_-: both pairings, times the two shear components.
  return 2.0*(direct + exchanged)*pair_variance;
}
