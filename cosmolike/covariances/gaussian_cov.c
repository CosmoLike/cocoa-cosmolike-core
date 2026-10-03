#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include "gaussian_cov.h"
#include "log.c/src/log.h"

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
// No global state and no lazy table reads. Call outside parallel regions;
// each OpenMP worker writes different multipoles, with no shared sum.
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
  if (ell_min < 0 || nell < 1 || !isfinite(fsky)
      || fsky <= 0.0 || fsky > 1.0
      || (include_noise_noise != 0 && include_noise_noise != 1)) {
    log_fatal("gaussian_wick_cov needs ell_min >= 0, nell > 0, "
              "finite 0 < fsky <= 1, and include_noise_noise = 0 or 1");
    exit(1);
  }

  const double* restrict cl_ac = cross_spectra[0];
  const double* restrict cl_bd = cross_spectra[1];
  const double* restrict cl_ad = cross_spectra[2];
  const double* restrict cl_bc = cross_spectra[3];
  const double noise_ac = cross_noise[0];
  const double noise_bd = cross_noise[1];
  const double noise_ad = cross_noise[2];
  const double noise_bc = cross_noise[3];

  double pure_noise = 0.0;
  if (include_noise_noise) {
    pure_noise = noise_ac*noise_bd + noise_ad*noise_bc;
  }

  // Only the output is written. The four read-only input rows may coincide
  // for auto spectra; none may overlap the output array.
  #pragma omp parallel for schedule(static)
  for (int node=0; node<nell; node++) {
    const double signal = cl_ac[node]*cl_bd[node] + cl_ad[node]*cl_bc[node];
    const double mixed_ac_bd = cl_ac[node]*noise_bd + noise_ac*cl_bd[node];
    const double mixed_ad_bc = cl_ad[node]*noise_bc + noise_ad*cl_bc[node];
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
// Scratch and output must not overlap each other or any input. Input rows
// may coincide. Row pointers support the house allocators' padded strides.
// Cache invalidation:
// No static cache. Geometry owners retain kernels and scratch across calls.
// Thread safety:
// Call outside parallel regions. Each output belongs to one worker; the
// increasing-ell sum never crosses workers, regardless of thread count.
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
  if (nleft < 1 || nright < 1 || nell < 1) {
    log_fatal("gaussian_project_cov needs positive nleft, nright and nell");
    exit(1);
  }

  #pragma omp parallel for schedule(static)
  for (int left=0; left<nleft; left++) {
    const double* restrict kernel = kernel_left[left];
    double* restrict weighted = weighted_left[left];

    for (int node=0; node<nell; node++) {
      weighted[node] = kernel[node]*gaussian[node];
    }
  }

  // Local pointers tell the compiler that writes cannot change a kernel.
  // Access through these pointers also walks adjacent ell values in memory.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int left=0; left<nleft; left++) {
    for (int right=0; right<nright; right++) {
      const double* restrict weighted = weighted_left[left];
      const double* restrict kernel = kernel_right[right];
      double total = 0.0;

      for (int node=0; node<nell; node++) {
        total += weighted[node]*kernel[node];
      }
      covariance[left][right] = total;
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
  if (!isfinite(area_sr) || area_sr <= 0.0 || area_sr > 4.0*M_PI
      || !isfinite(theta_low_rad) || !isfinite(theta_high_rad)
      || theta_low_rad < 0.0 || theta_high_rad > M_PI
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
// fields[] contains globally distinct catalog IDs in order A,B,C,D. For
// gamma_t, A and C must be lens catalogs and B and D source catalogs.
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
  if (probe_left < XI_PLUS_COV || probe_left > W_THETA_COV
      || probe_right < XI_PLUS_COV || probe_right > W_THETA_COV
      || !isfinite(pair_area_sr2) || pair_area_sr2 <= 0.0
      || !isfinite(noise_ab[0]) || !isfinite(noise_ab[1])
      || noise_ab[0] < 0.0 || noise_ab[1] < 0.0
      || fields[0] < 0 || fields[1] < 0 || fields[2] < 0 || fields[3] < 0) {
    log_fatal("gaussian_noise_pair_cov needs supported probes, nonnegative "
              "field IDs, finite nonnegative noise and positive pair area");
    exit(1);
  }

  if (probe_left != probe_right) {
    return 0.0;
  }

  const int direct = fields[0] == fields[2] && fields[1] == fields[3];
  const int exchanged = fields[0] == fields[3] && fields[1] == fields[2];
  const double pair_variance = noise_ab[0]*noise_ab[1]/pair_area_sr2;

  if (probe_left == GAMMA_T_COV) {
    return direct*pair_variance;
  }
  if (probe_left == W_THETA_COV) {
    return (direct + exchanged)*pair_variance;
  }
  return 2.0*(direct + exchanged)*pair_variance;
}
