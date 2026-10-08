#include <math.h>
#include <omp.h>
#include <stdlib.h>

#include "operators_cov.h"
#include "gaussian_cov.h"
#include "cosmolike/basics.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

// SIMD (single instruction, multiple data) applies one arithmetic operation
// to several numbers at once; each number sits in a vector position called
// a lane. A v2d holds two doubles, lane 0 and lane 1, in one 128-bit
// register: SSE2 on x86-64, NEON on arm64. SIMDe translates the x86
// intrinsic names used below into the native instructions of either
// processor, so one source serves both. Vector variables carry a v prefix.
//
// The real-space loop puts two neighboring quadrature angles of one bin in
// the two lanes; the band loop puts two neighboring integer multipoles.
//
// A fused multiply-add evaluates a*b + c exactly and rounds once, while a
// separate multiply and add rounds twice; the two answers can differ in
// the last bit. The two fused calls of this file round as follows:
//   simde_mm_fmadd_pd(a, b, c) = a*b + c: one rounding with native x86 FMA
//     (_mm_fmadd_pd) and on arm64 (vfmaq_f64). An x86 build without FMA
//     rounds the product and then the sum.
//   simde_mm_fmsub_pd(a, b, c) = a*b - c: one rounding only with native
//     x86 FMA. Elsewhere, arm64 included, SIMDe evaluates it as
//     simde_mm_sub_pd(simde_mm_mul_pd(a, b), c): the product is rounded,
//     then the difference.
// The last bits of the operators can therefore differ between x86 and
// arm64. On one machine they do not depend on the number of threads.
//
// loadu/storeu (u for unaligned) accept the address of any double, not
// only multiples of 16 bytes. Each still reads or writes two consecutive
// doubles, and both must lie inside the array.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Area-averaged full-sky angular operators for observed scalar/shear fields.
//
// PHYSICS AND NORMALIZATION
// A harmonic spectrum C_ell predicts a two-point correlation through
//
//   X(theta) = sum_ell (2 ell+1)/(4 pi) d_ell(theta) C_ell.
//
// The factor (2 ell+1)/(4 pi) is the number of harmonic modes of degree
// ell, 2 ell+1, per steradian of the full sky. The kernel d_ell is an
// element of Wigner's small rotation matrix d^ell_(m',m)(theta). Shear has
// spin two: rotating the local axes by an angle psi multiplies
// gamma_1 + i gamma_2 by exp(2 i psi). A correlation function measures
// shear in the tangential and cross components defined by the line joining
// the two objects, and d^ell_(m',m) rotates spin-m' and spin-m harmonic
// modes into that frame. With both spins zero it is the Legendre
// polynomial P_ell(cos theta). The four rows of this operator are
//
//   xi+     = <gamma_t gamma_t> + <gamma_x gamma_x>:  d^ell_(2,2),  EE+BB
//   xi-     = <gamma_t gamma_t> - <gamma_x gamma_x>:  d^ell_(2,-2), EE-BB
//   gamma_t = <delta_g gamma_t>:                      d^ell_(2,0),  gE
//   w       = <delta_g delta_g>:                      d^ell_(0,0),  gg
//
// Tangential shear has the positive P_ell^2 convention of Friedrich et al.,
// arXiv:2012.08568, Appendices A/B, Eqs. 95 and 98-100; for example
// d^2_(2,0) = sqrt(3/8) sin^2(theta) > 0. These are also the full-sky
// kernels of the Core Cosmology Library (arXiv:1812.05995) for spin-2
// E/B spectra.
//
// Here C_ell is the spectrum of the observed fields: galaxy overdensity
// and the spin-2 E and B modes of the measured shear, normalized so that
// white shape noise is sigma_component^2/n. The core real-space
// transformation uses different kernels: its xi+- kernels are
// (ell-1)(ell+2)/(ell(ell+1)) times d^ell_(2,+-2), and its gamma_t kernel
// is the square root of that factor times d^ell_(2,0). To reproduce it,
// multiply each source (shear) leg of the core spectrum by
// sqrt[(ell-1)(ell+2)/(ell(ell+1))] first: twice for source-source, once
// for lens-source, never for lens-lens and never for the white noise,
// which already describes the measured shear. The core's Limber shear
// spectra already carry the curved-sky factor
// sqrt[(ell-1)ell(ell+1)(ell+2)]/(ell+1/2)^2 per source leg, which contains
// this spin factor once; the conversion applies it a second time, as the
// core kernels do. Which convention a survey should adopt needs its own
// validation; these operators cannot settle it. Xi+ acts on EE+BB, xi- on
// EE-BB; B-mode bookkeeping belongs to the caller.
//
// With x = cos(theta), s = sin(theta/2), c = cos(theta/2) and n = ell-2,
// the rotation elements, normalized so that d^ell_(m,m)(0) = 1, are
//
//   d^ell_(2, 2) = c^4 P_n^(0,4)(x),
//   d^ell_(2,-2) = s^4 P_n^(4,0)(x),
//   d^ell_(2, 0) = s^2 c^2 sqrt[(ell+2)(ell+1)/(ell(ell-1))] P_n^(2,2)(x),
//   d^ell_(0, 0) = P_ell^(0,0)(x) = P_ell(x).
//
// P_n^(alpha,beta) is the Jacobi polynomial of degree n: a polynomial in
// x, orthogonal on [-1,1] with weight (1-x)^alpha (1+x)^beta. The
// half-angle factors carry the small-angle behavior explicitly. For xi-,
// d^ell_(2,-2) is of order theta^4 at small theta, and s^4 is computed
// from sin(theta/2) directly. Writing s^2 as (1-x)/2 with x near one, or
// subtracting two nearly equal antiderivatives at the bin edges as an
// analytic bin average does, would lose that small signal to cancellation.
//
// NUMERICAL STEPS
// A measured bin averages the correlation over the annulus between its
// edges, weighting each separation by its spherical area
// 2 pi sin(theta) dtheta. The average is the integral of f sin(theta)
// dtheta divided by the annulus measure cos(theta_low)-cos(theta_high),
// which is evaluated as two sines to avoid cancellation. Mask-dependent
// pair-separation weighting is not included.
//
// At large ell the kernel oscillates roughly as cos(ell*theta), so a wide
// bin contains many sign changes, and one low-order rule across the whole
// bin could miss those oscillations. Each bin is therefore split into
// equal panels with ell_max*panel_width <= 128 radians, and the same
// precomputed nquad-node Gauss-Legendre rule integrates every panel.
// Across one panel the fastest kernel completes at most 128/(2 pi), about
// 20, oscillations, so even the smallest 64-node rule puts about three
// nodes on each. To see the margin another way, map a panel to t in
// [-1,1]: its oscillatory phase varies as at most 64*t, whereas a 64-node
// Gaussian rule integrates every polynomial through degree 127 exactly,
// leaving room to resolve the oscillation's polynomial expansion. This is
// numerical averaging, not an exact endpoint formula, and the margin is
// not an error bound: refine nquad to test it. The analytic scalar-bin
// integral supplies an independent high-ell check. Panel boundaries depend
// on bin geometry and ell_max, not on nquad, so changing integration
// accuracy tests the same panel integrals.
//
// At every angular node, Jacobi polynomials follow the three-term relation
//   P_(n+1) = (A_n x + B_n) P_n - C_n P_(n-1),
// starting from P_0 = 1 and P_(-1) = 0. DLMF 18.9.1-2 gives the
// coefficients. They depend only on degree and on (alpha,beta), so they
// are constructed once for all bins. Each step needs only the two previous
// degrees: two rolling rows over the nodes replace a node-by-multipole
// work table.
//
// PHYSICAL DERIVATION & LOGIC FLOW
//   1. For each bin and node: x = cos(theta), the area weight
//      w = dtheta sin(theta)/[cos(theta_low)-cos(theta_high)], s^2, c^2.
//   2. For each probe and degree: A_n, B_n, C_n of P_n^(alpha,beta), and
//      norm = (2 ell+1)/(4 pi), times the d^ell_(2,0) square root for
//      gamma_t.
//   3. For each (probe,bin) row and degree n, at ell = n+2 (ell = n for w):
//        kernel[probe*nbin+bin][ell] = norm * sum_node w h P_n(x),
//      with half-angle prefactor h = c^4, s^4, s^2 c^2 or 1 for xi+, xi-,
//      gamma_t, w; then every node advances from P_n to P_(n+1).
//
// OWNERSHIP, MODES AND THREADS
// Output row probe*nbin+bin has ell = 0..ell_max. Spin rows (xi+, xi-,
// gamma_t) are zero at ell = 0, 1, where no spin-2 harmonic exists. The
// scalar w row retains its monopole and dipole; a survey estimator that
// removes them must remove the same modes from its signal/noise treatment
// explicitly. All angles are radians and the operators are dimensionless.
//
// Each OpenMP worker computes complete (probe,bin) rows, one degree after
// another. Inside a row, SIMDe advances two quadrature nodes and their two
// partial sums; the final two-lane addition is fixed, so the result is
// independent of the thread count. Grouped scratch is allocated before
// the parallel region. Output rows must be disjoint. No BLAS call.
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

  // Count the panels of each bin. ell_max*(theta_high - theta_low) is the
  // phase, in radians, by which the fastest kernel cos(ell_max theta)
  // advances across the bin; ceil(phase/128) equal panels keep the advance
  // within one panel at or below 128 radians, and a narrow bin keeps a
  // single panel. For example, a 200-250 arcmin bin at ell_max = 50000
  // spans about 727 radians and gets 6 panels. More panels resolve more
  // oscillations without constructing a new GSL rule. Store each bin's
  // node count and size the shared scratch for the widest bin.
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

  // Allocate all four geometry tables together, indexed
  // [quantity][angular bin][quadrature node]. The quantities are
  //   0: x = cos(theta), the argument of the Jacobi polynomials,
  //   1: the normalized area weight of the node,
  //   2: s^2 = sin^2(theta/2),
  //   3: c^2 = cos^2(theta/2).
  // A bin with fewer than max_nodes nodes leaves the end of its rows unused.
  double*** geometry = (double***) malloc3d(4, nbin, max_nodes);

  // A Gauss-Legendre rule approximates an integral over an interval by a
  // weighted sum of the integrand at nquad special nodes; it is exact for
  // polynomials through degree 2 nquad - 1. GSL tabulates nodes and weights
  // once; every panel of every bin maps them onto its own boundaries.
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
    // The identity cos(a)-cos(b) = 2 sin((a+b)/2) sin((b-a)/2) avoids
    // subtracting two cosines near one in a narrow small-angle bin.
    const double width = 2.0*sin((upper+lower)/2.0)*sin((upper-lower)/2.0);

    // A zero-area annulus cannot define an average correlation.
    if (width <= 0.0) {
      log_fatal("realspace_operator_cov: bin %d has zero angular area", bin);
      exit(1);
    }

    // Each panel contributes part of the same measured annulus; its
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

      // Convert the dtheta weight to a normalized spherical-area weight,
      // w = measure sin(theta)/width. Summed over the bin's nodes these
      // weights approximate the area average of unity, which is one.
      geometry[1][bin][node] = measure*sin(theta)/width;

      // Cache the squared half angles: every multipole reuses them.
      geometry[2][bin][node] = sine*sine;
      geometry[3][bin][node] = cosine*cosine;
    }
  }

  // All sample angles and weights have been copied into geometry.
  gsl_integration_glfixed_table_free(rule);

  // --- 2. PRECOMPUTE THE POLYNOMIAL RECURRENCE ---

  // Jacobi parameters (alpha, beta) of each probe in enum probe_cov order:
  // xi+ (0,4), xi- (4,0), gamma_t (2,2) and w (0,0), the Legendre case.
  // coefficient[probe][role][degree] holds role 0: A_n, 1: B_n, 2: C_n,
  // and 3: the harmonic normalization (2 ell+1)/(4 pi), which for gamma_t
  // also includes the square root of the d^ell_(2,0) representation.
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

      // A multiplies x P_n, B multiplies P_n and C multiplies P_(n-1).
      // DLMF 18.9.2 with a = alpha, b = beta and t = twice = 2n+a+b:
      //   A_n = (t+1)(t+2) / [2(n+1)(n+a+b+1)],
      //   B_n = (a^2-b^2)(t+1) / [2(n+1)(n+a+b+1) t],
      //   C_n = (n+a)(n+b)(t+2) / [(n+1)(n+a+b+1) t],
      // with a^2-b^2 = difference*sum; the 2 in the code's C_n numerator
      // cancels the 2 of the shared denominator. At n = 0 the general B_n
      // is 0/0 for w (t = a+b = 0), so every probe uses the special form:
      // P_1 = A_0 x + (a-b)/2 gives B_0 = (a-b)/2, and C_0 multiplies
      // P_(-1) = 0, so it is set to zero.
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

      // Convert the area-averaged polynomial into the harmonic operator at
      // ell = degree + first_ell. Gamma_t also needs the square root of
      // the d^ell_(2,0) representation; ell >= 2 keeps ell(ell-1) > 0.
      const double ell = degree+first_ell;
      double normalization = (2.0*ell+1.0)/(4.0*M_PI);
      if (probe == GAMMA_T_COV) {
        normalization *= sqrt((ell+2.0)*(ell+1.0)/(ell*(ell-1.0)));
      }
      coefficient[probe][3][degree] = normalization;
    }
  }

  // --- 3. SPIN POLYNOMIALS AND AREA AVERAGES ---

  // Per-worker scratch work[worker][row][node], three rows per worker:
  //   row 0, previous: P_(n-1) at every node of the current bin;
  //   row 1, current:  P_n at every node;
  //   row 2, weight:   the node's area weight times the probe's half-angle
  //                    prefactor, which does not change with degree.
  // Every worker reuses its rows for the next bin; no allocation occurs
  // inside OpenMP.
  const int nthreads = omp_get_max_threads();
  double*** work = (double***) malloc3d(nthreads, 3, max_nodes);

  // A measured angular bin averages correlations over an annulus, rather
  // than observing the correlation at just its center. For each multipole
  // we therefore integrate the appropriate spin kernel over the sample
  // angles, using the area weights prepared above. The polynomial recurrence
  // obtains the next multipole from the previous two without starting over.
  // One iteration of the collapsed (probe,bin) loop builds one complete
  // output row; rows are independent, so static scheduling hands whole rows
  // to workers and no row is shared. SIMD evaluates two angles together:
  // the lanes accumulate even/odd-node contributions to the same bin
  // integral, so their subtotals are added after visiting all angles.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int probe=0; probe<4; probe++) {
    // For this probe, use each bin's own geometry to fill its ell row.
    // The two SIMD lanes always refer to two angles inside that same bin.
    for (int bin=0; bin<nbin; bin++) {
      const int thread = omp_get_thread_num();
      const int first_ell = probe == W_THETA_COV ? 0 : 2;

      // Local restrict pointers. The three scratch rows are distinct padded
      // rows of this worker, the output row belongs to this (probe,bin)
      // alone, and the geometry is read-only, so a store through one
      // pointer never changes what another reads. restrict promises the
      // compiler exactly this, so it need not reload values after each
      // store. Writes cannot change another worker's input or output.
      double* restrict previous = work[thread][0];
      double* restrict current = work[thread][1];
      double* restrict weight = work[thread][2];
      double* restrict output = kernel[probe*nbin+bin];
      const double* restrict cosine = geometry[0][bin];

      // Start the recurrence at every node with P_(-1) = 0 in previous and
      // P_0 = 1 in current. Attach the probe's half-angle prefactor h to
      // the area weight: c^4 for xi+, s^4 for xi-, s^2 c^2 for gamma_t and
      // 1 for w. It does not change with degree, so it is applied once.
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

      // d^ell_(2,m) exists only for ell >= 2: a spin-two field has no
      // ell=0 or ell=1 contribution, so the spin rows start with two zeros.
      // The w row has first_ell = 0 and keeps its monopole and dipole.
      for (int ell=0; ell<first_ell; ell++) {
        output[ell] = 0.0;
      }

      // Each degree n produces one harmonic-operator entry, at
      // ell = n + first_ell: sum the current polynomial over angles, then
      // advance its recurrence. SIMD keeps separate even/odd-node sums until
      // their final addition. All allowed rules are even, and nnode[bin] is
      // a multiple of nquad, so each two-node load has two real angular
      // samples of this bin.
      for (int degree=0; degree<=ell_max-first_ell; degree++) {
        // scalar: one degree, with A = coefficient[probe][0][degree] and B,
        // C in roles 1, 2, visiting the nodes j in increasing order:
        //   double subtotal[2] = {0.0, 0.0};
        //   for (int j=0; j<nnode[bin]; j++) {
        //     subtotal[j%2] = fma(weight[j], current[j], subtotal[j%2]);
        //     const double linear = fma(A, cosine[j], B);
        //     const double next = fma(linear, current[j], -(C*previous[j]));
        //     previous[j] = current[j];
        //     current[j] = next;
        //   }
        //   output[degree+first_ell] = (subtotal[0]+subtotal[1])
        //                              *coefficient[probe][3][degree];
        // The first statement in the loop integrates the current polynomial
        // over the bin; the other four advance it by one degree. The fma
        // for next is what x86 with FMA computes; on arm64 SIMDe rounds
        // linear*current[j] before subtracting C*previous[j] (see the top
        // of this file). Lane 0 carries subtotal[0] (nodes 0, 2, 4, ...)
        // and lane 1 subtotal[1] (nodes 1, 3, 5, ...), each accumulated in
        // increasing node order, so the vector loop preserves this scalar
        // summation order.
        //
        // Rolling rows. On entry to degree n, previous[j] = P_(n-1)(x_j)
        // and current[j] = P_n(x_j) at every node j (0 and 1 when n = 0).
        // A vector step at nodes node, node+1 reads both rows there and
        // overwrites them with P_n and P_(n+1); nodes not yet visited still
        // hold P_(n-1) and P_n. After the node loop previous holds P_n and
        // current holds P_(n+1) at every node, ready for degree n+1. The
        // last degree also computes a P_(n+1) beyond ell_max; it is unused.

        // A_n in lanes 0 and 1 (set1_pd copies one double into both lanes):
        // the recurrence coefficient is the same at the two angles, even
        // though the polynomial values differ.
        const v2d va = simde_mm_set1_pd(coefficient[probe][0][degree]);

        // B_n in both lanes (set1_pd): the angle-independent additive term
        // of A_n x + B_n.
        const v2d vb = simde_mm_set1_pd(coefficient[probe][1][degree]);

        // C_n in both lanes (set1_pd): the weight of the previous-degree
        // contribution C_n P_(n-1).
        const v2d vc = simde_mm_set1_pd(coefficient[probe][2][degree]);

        // [0.0, 0.0] (setzero_pd clears both lanes): two partial bin
        // integrals of this degree. Lane 0 will sum nodes 0,2,4,... and
        // lane 1 nodes 1,3,5,..., always in this fixed order.
        v2d vsum = simde_mm_setzero_pd();

        // Process two angles together: accumulate weight*P_n, advance both
        // polynomials to P_(n+1), and save them for the next degree. Lane 0
        // owns node and lane 1 owns node+1; their sums stay separate here.
        // node advances by two up to the even nnode[bin]: no remainder.
        for (int node=0; node<nnode[bin]; node+=2) {
          // vcurrent = [P_n at node, P_n at node+1]: loadu reads
          // current[node] into lane 0 and current[node+1] into lane 1. It
          // needs no vector alignment; both nodes exist in this bin.
          const v2d vcurrent = simde_mm_loadu_pd(current+node);

          // vprevious = [P_(n-1) at node, at node+1], from previous[node]
          // and previous[node+1] (loadu, no alignment needed).
          const v2d vprevious = simde_mm_loadu_pd(previous+node);

          // vx = [cos(theta) at node, at node+1], from cosine[node] and
          // cosine[node+1] (loadu): each lane's polynomial argument x.
          const v2d vx = simde_mm_loadu_pd(cosine+node);

          // vweight = [w h at node, at node+1], from weight[node] and
          // weight[node+1] (loadu): area weight times half-angle prefactor.
          const v2d vweight = simde_mm_loadu_pd(weight+node);

          // vsum[lane] = fma(weight, P_n, vsum[lane]): each lane adds
          // weight*P_n of its own node to its own bin-integral subtotal,
          // rounded once on x86 with FMA and on arm64. Lane 0 is never
          // added to lane 1 here.
          vsum = simde_mm_fmadd_pd(vweight, vcurrent, vsum);

          // vlinear[lane] = fma(A_n, x, B_n) = A_n*cos(theta) + B_n at each
          // lane's angle, rounded once on x86 with FMA and on arm64.
          const v2d vlinear = simde_mm_fmadd_pd(va, vx, vb);

          // vbackward[lane] = C_n*P_(n-1), lane by lane. This product is
          // rounded before it enters the subtraction in the next step.
          const v2d vbackward = simde_mm_mul_pd(vc, vprevious);

          // vnext[lane] = vlinear*P_n - vbackward = P_(n+1) at each angle.
          // fmsub means multiply-minus-third-argument. With native x86 FMA
          // it is fma(vlinear, P_n, -vbackward), one rounding; on arm64
          // SIMDe rounds vlinear*P_n first and then the difference.
          const v2d vnext = simde_mm_fmsub_pd(vlinear, vcurrent, vbackward);

          // previous[node] and previous[node+1] = P_n of lanes 0 and 1:
          // the degree just summed becomes the P_(n-1) of the next degree.
          // storeu needs two valid doubles, but no vector-aligned address.
          simde_mm_storeu_pd(previous+node, vcurrent);

          // current[node] and current[node+1] = P_(n+1) of lanes 0 and 1,
          // the polynomial of the next degree (storeu, no alignment). Both
          // rows have now advanced at these two nodes; later nodes still
          // hold P_(n-1) and P_n until the loop reaches them.
          simde_mm_storeu_pd(current+node, vnext);
        }

        // The two lane subtotals of this degree, read out of the register.
        double sum[2];

        // sum[0] = lane 0, the even-node subtotal, and sum[1] = lane 1, the
        // odd-node subtotal. storeu allows this stack array without special
        // vector alignment.
        simde_mm_storeu_pd(sum, vsum);

        // This is the only addition between lanes. Its order is fixed (even
        // plus odd) and independent of the thread count. The normalization
        // (2 ell+1)/(4 pi), with the d^ell_(2,0) square root for gamma_t,
        // then converts the bin average of h P_n into K_ell at
        // ell = degree + first_ell.
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
// Each ell has 2 ell+1 angular modes (m = -ell..ell). A mode-weighted band
// counts each of its modes once:
//   C_band = sum_(ell=first)^last (2 ell+1) C_ell / N_band,
//   N_band = (last+1)^2-first^2 = (last-first+1)(last+first+1).
// The factored form avoids subtracting two close squared integers. A
// spectrum that is constant across the band returns that constant.
//
// Multiplying these operators on both sides of a covariance, O Cov O^T,
// performs the band average. For Gaussian covariance proportional to
// 1/(2 ell+1), G(ell) = X/[(2 ell+1) fsky], a constant Wick numerator X
// gives exactly its value divided by fsky*N_band:
//   sum_ell [(2 ell+1)/N_band]^2 X/[(2 ell+1) fsky] = X/(fsky N_band).
// Non-Gaussian covariance gets no extra mode-count division: it has no
// 1/(2 ell+1) of its own, its two band sums already contain both
// normalizations, and a constant matrix stays constant. See Krause &
// Eifler, arXiv:1601.05779, Appendix A. Overlapping bands are allowed and
// correlated.
//
// Parameters: first/last are inclusive absolute multipoles, not column IDs;
// column j of a row is multipole ell_min + j. Every band must fit within
// the supplied integer grid. All rows, including columns outside each
// band, are overwritten. No allocation or static cache.
// Each worker owns whole band rows; within a row the two SIMD lanes fill
// two adjacent multipoles, separate entries that are never added.
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
  // One iteration fills one band's row, leaving zero outside its limits;
  // rows are independent, so static scheduling hands whole rows to workers.
  // SIMD computes weights for two neighboring ell values; these are separate
  // operator entries, so they are stored separately rather than added here.
  #pragma omp parallel for schedule(static)
  for (int band=0; band<nband; band++) {
    double* restrict row = kernel[band]; // this band owns its writable row
    const double lower = first[band];
    const double upper = last[band];

    // The sum of 2 ell+1 over this inclusive band is the mode count
    // N_band, a product of two integers that is exact in double precision.
    const double modes = (upper-lower+1.0)*(upper+lower+1.0);

    // Both adjacent multipoles use the same band normalization. set1_pd
    // copies 1/N_band into lane 0 and lane 1: [1/N_band, 1/N_band]. The
    // reciprocal is rounded once here and shared by every vector step.
    const v2d vnorm = simde_mm_set1_pd(1.0/modes);

    // Multipoles outside this band must contribute zero to its average.
    for (int node=0; node<nell; node++) {
      row[node] = 0.0;
    }

    // Convert absolute ell limits into indices of the supplied row.
    int node = first[band]-ell_min;
    const int end = last[band]-ell_min;

    // scalar: for each column j from node to end,
    //   const double mode = 2.0*((double) ell_min+j)+1.0;
    //   row[j] = mode*(1.0/modes);
    // The numerator counts full-sky modes at that multipole; the common
    // denominator makes their weights sum to one across the whole band.
    // The vector loop takes j = node (lane 0) and node+1 (lane 1) per step,
    // only while both are in the band, and writes two separate weights,
    // not their sum. Mathematically each is (2 ell+1)/N_band; the scalar
    // remainder below divides directly, which can differ from
    // mode*(1/N_band) in the last bit.
    for (; node+1<=end; node+=2) {
      // mode = 2 ell+1 at ell = ell_min+node, an exact integer in double
      // precision. Increasing ell by one adds two modes:
      // (2(ell+1)+1) = mode+2, also exact.
      const double mode = 2.0*((double) ell_min+node)+1.0;

      // set_pd(high, low) takes lane 1 first. Its second argument becomes
      // lane 0, so the vector is [mode, mode+2] in increasing ell order.
      const v2d vmode = simde_mm_set_pd(mode+2.0, mode);

      // Multiply matching lanes, each product rounded once:
      // [(2 ell+1)/N_band, (2(ell+1)+1)/N_band], the weights of columns
      // node and node+1. The two weights are independent.
      const v2d vband_weight = simde_mm_mul_pd(vmode, vnorm);

      // Write lane 0 to row[node] and lane 1 to row[node+1]. storeu allows
      // an ordinary double-array address without special vector alignment;
      // node+1 <= end keeps both writes inside the band.
      simde_mm_storeu_pd(row+node, vband_weight);
    }

    // An odd number of multipoles leaves one final weight, node == end.
    // Compute it scalarly so no two-value store writes beyond the band
    // boundary. This direct division can differ from the vector lanes'
    // mode*(1/N_band) in the last bit.
    if (node <= end) {
      row[node] = (2.0*((double) ell_min+node)+1.0)/modes;
    }
  }
}
