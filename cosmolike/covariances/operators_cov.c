#include <math.h>
#include <omp.h>
#include <stdlib.h>

#include "operators_cov.h"
#include "gaussian_cov.h"
#include "cosmolike/basics.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

typedef simde__m128d v2d; // two angular quadrature nodes

// ---------------------------------------------------------------------------
// Area-averaged full-sky angular operators for observed scalar/shear fields.
//
// PHYSICS AND NORMALIZATION
// A harmonic spectrum predicts a correlation through
//
//   X(theta) = sum_ell (2 ell+1)/(4 pi) d_ell(theta) C_ell.
//
// The four kernels d are the rotation-matrix elements d^ell_(2,2),
// d^ell_(2,-2), d^ell_(2,0), and d^ell_(0,0) for xi+, xi-, gamma_t,
// and w respectively. Rotating the shear basis between the two objects
// accounts for its spin two: these are not all scalar Legendre kernels.
// Tangential shear has the positive P_ell^2 convention of Friedrich et al.,
// arXiv:2012.08568, Appendices A/B, Eqs. 95 and 98-100.
//
// Here C is the UNIT-NORMALIZED observed-shear spectrum. To reproduce the
// existing core real-space transformation convention, multiply each source
// leg of its supplied spectrum by sqrt[(ell-1)(ell+2)/(ell(ell+1))] first.
// This is a convention conversion, not a correction to apply to spectra
// already defined for observed shear. White noise stays sigma_component^2/n.
// Xi+ acts on EE+BB, xi- on EE-BB. B-mode bookkeeping belongs to the caller.
//
// With x=cos(theta), s=sin(theta/2), c=cos(theta/2), and n=ell-2,
// the unit-normalized rotation elements can be written as
//
//   d_(2, 2) = c^4 P_n^(0,4)(x),
//   d_(2,-2) = s^4 P_n^(4,0)(x),
//   d_(2, 0) = s^2 c^2 sqrt[(ell+2)(ell+1)/(ell(ell-1))] P_n^(2,2)(x).
//
// P_n^(alpha,beta) is a Jacobi polynomial. For w use P_ell^(0,0)=P_ell.
// The half-angle factors retain tiny-angle precision, especially xi-:
// subtracting nearly equal endpoint antiderivatives can erase this signal.
//
// NUMERICAL STEPS
// A bin represents the spherical area average integral sin(theta) dtheta,
// divided by cos(theta_low)-cos(theta_high). Tabulated Gauss-Legendre nodes
// integrate this smooth finite interval in theta. This is a numerical bin
// average, not a point approximation or an exact antiderivative. The caller
// must refine nquad against ell_max and the widest bin before using it.
// The cosine difference is evaluated as two sines to avoid cancellation.
//
// At every angular node, Jacobi polynomials follow the three-term relation
//   P_(n+1) = (A_n x+B_n) P_n - C_n P_(n-1),
// starting from P_0=1 and P_(-1)=0. DLMF 18.9.1-2 gives the coefficients.
// They depend only on degree and spin; construct them once for all bins.
// Two rolling rows suffice, rather than a node-by-multipole work table.
//
// OWNERSHIP, MODES AND THREADS
// Output row probe*nbin+bin has ell=0..ell_max. Spin rows have ell<2 zero.
// The scalar monopole/dipole are retained; a survey estimator that removes
// them must remove the same modes from its signal/noise treatment explicitly.
// All angles are radians and the operators are dimensionless.
//
// Each OpenMP worker owns one complete (probe,bin) row. SIMDe advances two
// quadrature nodes and their two partial sums; the final two-lane addition
// is fixed and independent of thread count. Grouped scratch is allocated
// before the parallel region. Output rows must be disjoint. No BLAS call.
//
// Cache invalidation:
// No static cache. Rebuild only when bin edges, ell_max or nquad change;
// the caller can retain these geometry-only operators across cosmologies.
// ---------------------------------------------------------------------------
void realspace_operator_cov(
    const int nbin,            // number of angular bins
    const double* edges_rad,   // [nbin+1], increasing angles in radians
    const int ell_max,         // highest included multipole
    const int nquad,           // angular quadrature nodes per bin
    double* const* kernel      // [4*nbin][ell_max+1], caller-owned rows
  )
{
  if (nbin < 1
      || ell_max < 2) {
    log_fatal("realspace_operator_cov needs nbin>=1 and ell_max>=2");
    exit(1);
  }
  if (nquad != 64
      && nquad != 96
      && nquad != 128
      && nquad != 256
      && nquad != 512
      && nquad != 1024) {
    log_fatal("realspace_operator_cov: unsupported GL size %d", nquad);
    exit(1);
  }
  for (int edge=0; edge<=nbin; edge++) {
    if (!isfinite(edges_rad[edge])
        || edges_rad[edge] < 0.0
        || edges_rad[edge] > M_PI) {
      log_fatal("realspace_operator_cov: angle %d is outside [0,pi]", edge);
      exit(1);
    }
    if (edge > 0
        && edges_rad[edge] <= edges_rad[edge-1]) {
      log_fatal("realspace_operator_cov: angular edges must increase");
      exit(1);
    }
  }

  // --- 1. ANGULAR NODES AND DEGREE-ONLY RECURRENCE COEFFICIENTS ---

  // Geometry roles: x, normalized area measure, sin^2(theta/2), cos^2.
  double*** geometry = (double***) malloc3d(4, nbin, nquad);
  gsl_integration_glfixed_table* rule = malloc_gslint_glfixed(nquad);
  for (int bin=0; bin<nbin; bin++) {
    const double lower = edges_rad[bin];
    const double upper = edges_rad[bin+1];
    const double width = 2.0*sin((upper+lower)/2.0)*sin((upper-lower)/2.0);
    if (width <= 0.0) {
      log_fatal("realspace_operator_cov: bin %d has zero angular area", bin);
      exit(1);
    }
    for (int node=0; node<nquad; node++) {
      double theta;   // angle at this Gaussian node, in radians
      double measure; // integration weight for dtheta
      gsl_integration_glfixed_point(lower, upper, node, &theta, &measure,
                                    rule);
      const double sine = sin(theta/2.0);
      const double cosine = cos(theta/2.0);
      geometry[0][bin][node] = cos(theta);
      geometry[1][bin][node] = measure*sin(theta)/width;
      geometry[2][bin][node] = sine*sine;
      geometry[3][bin][node] = cosine*cosine;
    }
  }
  gsl_integration_glfixed_table_free(rule);

  // [probe][A/B/C/normalization][degree]. Norm includes the mode density
  // (2 ell+1)/(4 pi), and the additional normalized d_(2,0) factor.
  const double alpha[4] = {0.0, 4.0, 2.0, 0.0};
  const double beta[4] = {4.0, 0.0, 2.0, 0.0};
  double*** coefficient = (double***) malloc3d(4, 4, ell_max+1);
  for (int probe=0; probe<4; probe++) {
    const double sum = alpha[probe]+beta[probe];
    const double difference = alpha[probe]-beta[probe];
    const int first_ell = probe == W_THETA_COV ? 0 : 2;
    for (int degree=0; degree<=ell_max-first_ell; degree++) {
      const double n = degree;
      const double twice = 2.0*n+sum;
      const double denominator = 2.0*(n+1.0)*(n+sum+1.0);
      coefficient[probe][0][degree] = (twice+1.0)*(twice+2.0)/denominator;
      if (degree == 0) {
        coefficient[probe][1][degree] = difference/2.0;
        coefficient[probe][2][degree] = 0.0;
      } else {
        coefficient[probe][1][degree] = difference*sum*(twice+1.0)
                                         /(denominator*twice);
        coefficient[probe][2][degree] = 2.0*(n+alpha[probe])*(n+beta[probe])
                                         *(twice+2.0)/(denominator*twice);
      }
      const double ell = degree+first_ell;
      double normalization = (2.0*ell+1.0)/(4.0*M_PI);
      if (probe == GAMMA_T_COV) {
        normalization *= sqrt((ell+2.0)*(ell+1.0)/(ell*(ell-1.0)));
      }
      coefficient[probe][3][degree] = normalization;
    }
  }

  // --- 2. SPIN POLYNOMIALS AND AREA AVERAGES ---

  // [worker][previous/current/weighted prefactor][node]. Every worker
  // reuses its rows for the next bin; no allocation occurs inside OpenMP.
  const int nthreads = omp_get_max_threads();
  double*** work = (double***) malloc3d(nthreads, 3, nquad);
  #pragma omp parallel for collapse(2) schedule(static)
  for (int probe=0; probe<4; probe++) {
    for (int bin=0; bin<nbin; bin++) {
      const int thread = omp_get_thread_num();
      const int first_ell = probe == W_THETA_COV ? 0 : 2;

      // These rows occupy different allocations or different padded rows;
      // writes cannot change another worker's input or output.
      double* restrict previous = work[thread][0];
      double* restrict current = work[thread][1];
      double* restrict weight = work[thread][2];
      double* restrict output = kernel[probe*nbin+bin];
      const double* restrict cosine = geometry[0][bin];
      for (int node=0; node<nquad; node++) {
        const double sine2 = geometry[2][bin][node];
        const double cosine2 = geometry[3][bin][node];
        double prefactor = 1.0;
        if (probe == XI_PLUS_COV) {
          prefactor = cosine2*cosine2;
        } else if (probe == XI_MINUS_COV) {
          prefactor = sine2*sine2;
        } else if (probe == GAMMA_T_COV) {
          prefactor = sine2*cosine2;
        }
        previous[node] = 0.0;
        current[node] = 1.0;
        weight[node] = geometry[1][bin][node]*prefactor;
      }
      for (int ell=0; ell<first_ell; ell++) {
        output[ell] = 0.0;
      }

      // Sum P_n first, then advance to P_(n+1). Every supported Gaussian
      // rule is even, so both SIMD lanes correspond to real angular nodes.
      for (int degree=0; degree<=ell_max-first_ell; degree++) {
        const v2d va = simde_mm_set1_pd(coefficient[probe][0][degree]);
        const v2d vb = simde_mm_set1_pd(coefficient[probe][1][degree]);
        const v2d vc = simde_mm_set1_pd(coefficient[probe][2][degree]);
        v2d vsum = simde_mm_setzero_pd();
        for (int node=0; node<nquad; node+=2) {
          const v2d vcurrent = simde_mm_loadu_pd(current+node);
          const v2d vprevious = simde_mm_loadu_pd(previous+node);
          const v2d vx = simde_mm_loadu_pd(cosine+node);
          const v2d vweight = simde_mm_loadu_pd(weight+node);
          vsum = simde_mm_fmadd_pd(vweight, vcurrent, vsum);
          const v2d vlinear = simde_mm_fmadd_pd(va, vx, vb);
          const v2d vnext = simde_mm_fmsub_pd(vlinear, vcurrent,
                                           simde_mm_mul_pd(vc, vprevious));
          simde_mm_storeu_pd(previous+node, vcurrent);
          simde_mm_storeu_pd(current+node, vnext);
        }
        double sum[2];
        simde_mm_storeu_pd(sum, vsum);
        output[degree+first_ell] = (sum[0]+sum[1])
                                  *coefficient[probe][3][degree];
      }
    }
  }
  free(work);
  free(coefficient);
  free(geometry);
}

// ---------------------------------------------------------------------------
// Exact discrete Fourier-band averaging on an integer multipole grid.
//
// Each ell has 2 ell+1 angular modes. An unbiased mode-weighted band is
//   C_band = sum_(ell=first)^last (2 ell+1) C_ell / N_band,
//   N_band = (last+1)^2-first^2 = (last-first+1)(last+first+1).
// The factored form avoids subtracting two close squared integers.
//
// Multiplying these operators on both sides of a covariance performs the
// band average. For Gaussian covariance proportional to 1/(2 ell+1), a
// constant Wick numerator gives exactly its value divided by fsky*N_band.
// Non-Gaussian covariance gets no extra mode-count division: its two band
// sums already contain both normalizations. See Krause & Eifler,
// arXiv:1601.05779, Appendix A. Overlapping bands are allowed and correlated.
//
// Parameters: first/last are inclusive absolute multipoles, not column IDs.
// Every band must fit within the supplied integer grid. All rows, including
// columns outside each band, are overwritten. No allocation or static cache.
// Each worker owns one band; two SIMD lanes fill adjacent multipoles.
// ---------------------------------------------------------------------------
void bandpower_operator_cov(
    const int nband,           // number of Fourier bands
    const int ell_min,         // first integer multipole of output columns
    const int nell,            // number of consecutive multipoles
    const int* first,          // [nband], inclusive lower edges
    const int* last,           // [nband], inclusive upper edges
    double* const* kernel      // [nband][nell], caller-owned rows
  )
{
  if (nband < 1
      || ell_min < 0
      || nell < 1) {
    log_fatal("bandpower_operator_cov: invalid band count or ell grid");
    exit(1);
  }
  for (int band=0; band<nband; band++) {
    if (first[band] < ell_min
        || last[band] < first[band]
        || (double) last[band] >= (double) ell_min+nell) {
      log_fatal("bandpower_operator_cov: band %d outside ell grid", band);
      exit(1);
    }
  }
  #pragma omp parallel for schedule(static)
  for (int band=0; band<nband; band++) {
    double* restrict row = kernel[band]; // this band owns its writable row
    const double lower = first[band];
    const double upper = last[band];
    const double modes = (upper-lower+1.0)*(upper+lower+1.0);
    const v2d vnorm = simde_mm_set1_pd(1.0/modes);
    for (int node=0; node<nell; node++) {
      row[node] = 0.0;
    }
    int node = first[band]-ell_min;
    const int end = last[band]-ell_min;
    for (; node+1<=end; node+=2) {
      const double mode = 2.0*((double) ell_min+node)+1.0;
      const v2d vmode = simde_mm_set_pd(mode+2.0, mode);
      simde_mm_storeu_pd(row+node, simde_mm_mul_pd(vmode, vnorm));
    }
    if (node <= end) {
      row[node] = (2.0*((double) ell_min+node)+1.0)/modes;
    }
  }
}
