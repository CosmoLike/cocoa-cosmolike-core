#include <math.h>
#include <omp.h>
#include <stdlib.h>

#include "operators_cov.h"
#include "gaussian_cov.h"
#include "cosmolike/basics.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

// SIMD performs the same operation on several values at once. Here v2d
// holds two doubles; their positions are called lane 0 and lane 1.
// The real-space loop uses two angular nodes, the band loop two ell values.
// A fused multiply-add (FMA) evaluates a*b+c with one rounding when
// supported directly by the processor. This differs from rounding a*b
// first and then adding c; the calls below retain the chosen operations.
// Unaligned loads/stores accept addresses that are not multiples of 16
// bytes. They still require two valid adjacent doubles in the array.
typedef simde__m128d v2d;

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
// integrate finite panels in theta. At large ell the kernel oscillates
// roughly as cos(ell*theta), so a wide bin contains many sign changes.
// One low-order rule across the whole bin could miss those oscillations.
// Split wide bins so ell_max*panel_width <= 128 radians, then use the same
// precomputed rule in each panel. Even the minimum 64-node rule therefore
// samples the fastest oscillations generously; refine nquad to test this.
// To see the margin, map a panel to t in [-1,1]. Its oscillatory phase
// varies as at most 64*t, whereas a 64-node Gaussian rule integrates
// every polynomial through degree 127 exactly. This gives room to resolve
// the oscillation's polynomial expansion. The analytic scalar-bin integral
// supplies an independent high-ell check; the margin is not an error bound.
// Panel boundaries depend on bin geometry and ell_max, not on nquad, so
// changing integration accuracy tests the same panel integrals.
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
    const int nquad,           // precomputed GSL nodes per angular panel
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
  // Validate each angular boundary before allocating tables: bins must
  // have positive width and lie within physical separations from 0 to pi.
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

  // --- 1. SAMPLE THE GEOMETRY OF EACH ANGULAR BIN ---

  // More panels resolve more oscillations without constructing a new GSL
  // rule. A narrow bin needs only one panel. Store each bin's node count
  // and allocate enough shared scratch for the widest bin's calculation.
  int* nnode = (int*) malloc1d_int(nbin); // total nodes in each angular bin
  int max_nodes = 0; // largest count, setting the padded workspace size

  for (int bin=0; bin<nbin; bin++) {
    const double phase = ell_max*(edges_rad[bin+1]-edges_rad[bin]);
    const int npanel = (int) fmax(1.0, ceil(phase/128.0)); // resolved panels
    nnode[bin] = npanel*nquad;
    if (nnode[bin] > max_nodes) {
      max_nodes = nnode[bin];
    }
  }

  // Allocate all four geometry tables together. Their indices are
  // [quantity][angular bin][quadrature node]. The quantities are cos(theta),
  // normalized area weight, sin^2(theta/2), and cos^2(theta/2).
  double*** geometry = (double***) malloc3d(4, nbin, max_nodes);

  // A Gauss-Legendre rule supplies sample angles and integration weights.
  // Reuse this rule for every bin, mapping it onto that bin's boundaries.
  gsl_integration_glfixed_table* rule = malloc_gslint_glfixed(nquad);

  // Build the geometry of one angular annulus per iteration. Its boundaries
  // set the area normalization; Gaussian nodes supply the angles and
  // weights stored for reuse by every probe and multipole.
  for (int bin=0; bin<nbin; bin++) {
    // These are the two angular boundaries of the measured annulus.
    const double lower = edges_rad[bin];
    const double upper = edges_rad[bin+1];

    // The spherical area element is 2 pi sin(theta) dtheta. Dividing an
    // annulus integral by its area cancels 2 pi, leaving the normalization
    // width = integral sin(theta) dtheta = cos(lower)-cos(upper).
    // The sine-product identity avoids subtracting two cosines near one.
    const double width = 2.0*sin((upper+lower)/2.0)*sin((upper-lower)/2.0);

    // A zero-area annulus cannot define an average correlation.
    if (width <= 0.0) {
      log_fatal("realspace_operator_cov: bin %d has zero angular area", bin);
      exit(1);
    }

    // Each panel contributes part of the SAME measured annulus; its
    // weights use the full annulus area above, not a separate panel area.
    const int npanel = nnode[bin]/nquad; // fixed across integration levels
    const double step = (upper-lower)/npanel; // angular panel width

    // The flat node index visits panels in angle order. Its quotient
    // selects a panel; its remainder selects the precomputed Gaussian
    // node inside that panel. Every later multipole reuses this geometry.
    for (int node=0; node<nnode[bin]; node++) {
      // GSL returns theta and its weight for integral f(theta) dtheta.
      // The spherical sin(theta) factor is added explicitly below.
      double theta;   // angle at this Gaussian node, in radians
      double measure; // integration weight for dtheta

      const int panel = node/nquad; // panel containing this angular node
      const double begin = lower+panel*step; // lower panel boundary
      const double end = lower+(panel+1)*step; // upper panel boundary
      gsl_integration_glfixed_point(begin, end, node % nquad,
                                    &theta, &measure, rule);

      // The spin-two rotation kernels use powers of these half angles.
      // Evaluating them directly retains precision at tiny separations.
      const double sine = sin(theta/2.0);
      const double cosine = cos(theta/2.0);

      // Jacobi polynomials are evaluated at x = cos(theta).
      geometry[0][bin][node] = cos(theta);

      // Convert the dtheta weight to a normalized spherical-area weight.
      // Summing these weights approximates the area average of unity.
      geometry[1][bin][node] = measure*sin(theta)/width;

      // Cache the squared half angles: every multipole reuses them.
      geometry[2][bin][node] = sine*sine;
      geometry[3][bin][node] = cosine*cosine;
    }
  }

  // All sample angles and weights have been copied into geometry.
  gsl_integration_glfixed_table_free(rule);

  // --- 2. PRECOMPUTE THE POLYNOMIAL RECURRENCE ---

  // [probe][A/B/C/normalization][degree]. Norm includes the mode density
  // (2 ell+1)/(4 pi), and the additional normalized d_(2,0) factor.
  const double alpha[4] = {0.0, 4.0, 2.0, 0.0};
  const double beta[4] = {4.0, 0.0, 2.0, 0.0};
  double*** coefficient = (double***) malloc3d(4, 4, ell_max+1);

  // Each probe has its own Jacobi-polynomial parameters. Build its degree
  // coefficients and harmonic normalizations once, shared by all bins.
  for (int probe=0; probe<4; probe++) {
    const double sum = alpha[probe]+beta[probe];
    const double difference = alpha[probe]-beta[probe];
    const int first_ell = probe == W_THETA_COV ? 0 : 2;

    // Store A_n, B_n and C_n for advancing this probe's polynomial by one
    // degree, plus the normalization mapping that degree to physical ell.
    for (int degree=0; degree<=ell_max-first_ell; degree++) {
      const double n = degree;
      const double twice = 2.0*n+sum;
      const double denominator = 2.0*(n+1.0)*(n+sum+1.0);

      // A multiplies x P_n. B multiplies P_n and C multiplies P_(n-1).
      // At n=0 the special form avoids division by zero when alpha+beta=0.
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

      // Convert the area-averaged polynomial into the harmonic operator.
      // Gamma_t also needs the normalization of the d_(2,0) element.
      const double ell = degree+first_ell;
      double normalization = (2.0*ell+1.0)/(4.0*M_PI);
      if (probe == GAMMA_T_COV) {
        normalization *= sqrt((ell+2.0)*(ell+1.0)/(ell*(ell-1.0)));
      }
      coefficient[probe][3][degree] = normalization;
    }
  }

  // --- 3. SPIN POLYNOMIALS AND AREA AVERAGES ---

  // [worker][previous/current/weighted prefactor][node]. Every worker
  // reuses its rows for the next bin; no allocation occurs inside OpenMP.
  const int nthreads = omp_get_max_threads();
  double*** work = (double***) malloc3d(nthreads, 3, max_nodes);

  // A measured angular bin averages correlations over an annulus, rather
  // than observing the correlation at just its center. For each multipole
  // we therefore integrate the appropriate spin kernel over the sample
  // angles, using the area weights prepared above. The polynomial recurrence
  // obtains the next multipole from the previous two without starting over.
  // One worker builds a complete (probe,bin) row. SIMD evaluates two angles
  // together: the lanes accumulate even/odd-node contributions to the SAME
  // bin integral, so their subtotals are added after visiting all angles.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int probe=0; probe<4; probe++) {
    // For this probe, use each bin's own geometry to fill its ell row.
    // The two SIMD lanes always refer to two angles inside that same bin.
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

      // Start with P_(-1)=0 and P_0=1. Attach the probe's half-angle
      // factor to the area weight; it does not change with degree.
      for (int node=0; node<nnode[bin]; node++) {
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

      // A spin-two field has no ell=0 or ell=1 contribution.
      for (int ell=0; ell<first_ell; ell++) {
        output[ell] = 0.0;
      }

      // Each degree produces one harmonic-operator entry: sum the current
      // polynomial over angles, then advance its recurrence. SIMD keeps
      // separate even/odd-node sums until their final addition. All allowed
      // rules are even, so each two-node load has two real angular samples.
      for (int degree=0; degree<=ell_max-first_ell; degree++) {
        // Scalar equivalent for one angular node j, with coefficients
        // A=coefficient[probe][0][degree], and similarly B (1) and C (2):
        //   subtotal[j%2] = fma(weight[j], current[j], subtotal[j%2]);
        //   linear = fma(A, cosine[j], B);
        //   next = fma(linear, current[j], -(C*previous[j]));
        //   previous[j] = current[j];
        //   current[j] = next;
        // The first line integrates the current polynomial over the bin.
        // The other lines prepare the next polynomial degree. Even and odd
        // nodes have separate subtotals, initialized to zero for this degree;
        // they are added only after all angles have contributed. SIMD uses
        // one lane for each parity, preserving that scalar summation order.
        // Copy A_n to both lanes: the recurrence coefficient is the same
        // at the two angles, even though the polynomial values differ.
        const v2d va = simde_mm_set1_pd(coefficient[probe][0][degree]);

        // Copy B_n to both lanes for the angle-independent additive term.
        const v2d vb = simde_mm_set1_pd(coefficient[probe][1][degree]);

        // Copy C_n to both lanes for the previous-degree contribution.
        const v2d vc = simde_mm_set1_pd(coefficient[probe][2][degree]);

        // Start two partial bin integrals at zero: lane 0 will sum nodes
        // 0,2,4,... and lane 1 nodes 1,3,5,..., always in this fixed order.
        v2d vsum = simde_mm_setzero_pd();

        // Process two angles together: accumulate weight*P_n, advance both
        // polynomials to P_(n+1), and save them for the next degree. Lane 0
        // owns node and lane 1 owns node+1; their sums stay separate here.
        for (int node=0; node<nnode[bin]; node+=2) {
          // Load P_n at node and node+1 into lanes 0 and 1. loadu accepts
          // an address without special vector alignment; both nodes exist.
          const v2d vcurrent = simde_mm_loadu_pd(current+node);

          // Load P_(n-1) at the same two nodes, in the same lane order.
          // loadu does not require vector-aligned storage.
          const v2d vprevious = simde_mm_loadu_pd(previous+node);

          // Load cos(theta) for nodes node and node+1. loadu permits
          // ordinary double-array addresses without vector alignment.
          const v2d vx = simde_mm_loadu_pd(cosine+node);

          // Load the area-times-spin weights for the same two angles.
          // loadu imposes no extra alignment on weight+node.
          const v2d vweight = simde_mm_loadu_pd(weight+node);

          // Each lane adds weight*P_n to its own bin-integral subtotal.
          // fmadd performs multiply-plus-add with one rounding on native
          // FMA hardware; it does not add lane 0 to lane 1.
          vsum = simde_mm_fmadd_pd(vweight, vcurrent, vsum);

          // Form A_n*cos(theta)+B_n separately at the two angles.
          // fmadd fuses that multiplication and addition into one rounding
          // on native FMA hardware.
          const v2d vlinear = simde_mm_fmadd_pd(va, vx, vb);

          // Multiply C_n by P_(n-1) lane by lane. This product is rounded
          // before entering the fused subtraction in the next step.
          const v2d vbackward = simde_mm_mul_pd(vc, vprevious);

          // At each angle compute P_(n+1) = vlinear*P_n - vbackward.
          // fmsub means multiply-minus-third-argument, fused into one
          // rounding on native FMA hardware.
          const v2d vnext = simde_mm_fmsub_pd(vlinear, vcurrent, vbackward);

          // Copy lanes 0 and 1 into previous[node] and previous[node+1].
          // storeu needs two valid doubles, but no vector-aligned address.
          simde_mm_storeu_pd(previous+node, vcurrent);

          // Save P_(n+1) at current[node] and current[node+1] for the next
          // degree. storeu accepts the ordinary double-array address.
          simde_mm_storeu_pd(current+node, vnext);
        }

        double sum[2];

        // Copy the even-node and odd-node subtotals into sum[0] and sum[1].
        // storeu allows this stack array without special vector alignment.
        simde_mm_storeu_pd(sum, vsum);

        // This is the only addition between lanes. Its order is fixed,
        // then the harmonic normalization converts the integral to K_ell.
        output[degree+first_ell] = (sum[0]+sum[1])
                                  *coefficient[probe][3][degree];
      }
    }
  }

  free(work);
  free(coefficient);
  free(geometry);
  free(nnode);
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
//
// Cache invalidation:
// No static state. The caller can reuse these weights until the integer
// multipole grid or inclusive band limits change.
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

  // A Fourier band combines many multipoles. An ell with more independent
  // modes should contribute more to this average, hence its weight 2*ell+1.
  // Dividing by the band's total mode count makes the weights sum to one.
  // Each worker fills one band's row, leaving zero outside its limits.
  // SIMD computes weights for two neighboring ell values; these are separate
  // operator entries, so they are stored separately rather than added here.
  #pragma omp parallel for schedule(static)
  for (int band=0; band<nband; band++) {
    double* restrict row = kernel[band]; // this band owns its writable row
    const double lower = first[band];
    const double upper = last[band];

    // The sum of 2 ell+1 over this inclusive band is the mode count.
    const double modes = (upper-lower+1.0)*(upper+lower+1.0);

    // Both adjacent multipoles use the same band normalization. set1_pd
    // copies 1/N_band into lane 0 and lane 1: [1/N_band, 1/N_band].
    const v2d vnorm = simde_mm_set1_pd(1.0/modes);

    // Multipoles outside this band must contribute zero to its average.
    for (int node=0; node<nell; node++) {
      row[node] = 0.0;
    }

    // Convert absolute ell limits into indices of the supplied row.
    int node = first[band]-ell_min;
    const int end = last[band]-ell_min;

    // Scalar equivalent for j=node,node+1 inside this band:
    //   mode = 2*(ell_min+j)+1;
    //   row[j] = mode*(1.0/modes);
    // The numerator counts full-sky modes at that multipole; the common
    // denominator makes their weights sum to one across the whole band.
    // SIMD writes two separate weights, not their sum.
    // Fill two adjacent multipoles together only when both are in the band.
    // The scalar operation for each is row[node] = (2 ell+1)/N_band.
    for (; node+1<=end; node+=2) {
      // Increasing ell by one adds two modes: (2(ell+1)+1) = mode+2.
      const double mode = 2.0*((double) ell_min+node)+1.0;

      // set_pd takes the HIGH lane first. Its second argument becomes
      // lane 0, so the vector is [mode, mode+2] in increasing ell order.
      const v2d vmode = simde_mm_set_pd(mode+2.0, mode);

      // Multiply matching lanes: [(2 ell+1)/N_band,
      // (2(ell+1)+1)/N_band]. The two weights are independent.
      const v2d vband_weight = simde_mm_mul_pd(vmode, vnorm);

      // Write lane 0 to row[node] and lane 1 to row[node+1]. storeu allows
      // an ordinary double-array address without special vector alignment.
      simde_mm_storeu_pd(row+node, vband_weight);
    }

    // An odd number of multipoles leaves one final weight. Compute it
    // scalarly so no two-value store writes beyond the band boundary.
    if (node <= end) {
      row[node] = (2.0*((double) ell_min+node)+1.0)/modes;
    }
  }
}
