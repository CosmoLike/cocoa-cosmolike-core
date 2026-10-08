#include <math.h>
#include <stdlib.h>

#include "spectra_cov.h"
#include "cosmolike/basics.h"
#include "cosmolike/bias.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/IA.h"
#include "cosmolike/radial_weights.h"
#include "cosmolike/redshift_spline.h"
#include "cosmolike/structs.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

// SIMD applies one operation to two doubles in positions called lanes.
// The efficiency loop uses lanes for integrals A and B; the spectrum
// loop uses them for two independent field-pair integrals.
// A fused multiply-add (FMA) evaluates a*b+c with one rounding when
// supported directly by the processor. This differs from rounding a*b
// first and then adding c; the calls below retain the chosen operations.
// Unaligned loads/stores accept addresses that are not multiples of 16
// bytes. They still require two valid adjacent doubles in the array.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Prepare matter power for independent covariance integration rows.
//
// In a trispectrum integral, each row represents a pair K,Q. Its columns
// contain |K+Q| at the sampled relative angles. Every sample has the same
// redshift, so the core reader reuses one redshift interpolation bracket
// within a row. Different rows only read the initialized cosmology tables
// and write separate outputs; they can therefore be assigned to workers.
//
// The existing reader still performs every interpolation. This changes
// neither the power model nor its arithmetic, and introduces no cache or
// reduction. Wavenumbers have units (c/H0)^-1 and power has units (c/H0)^3.
// Call outside an OpenMP region, after initializing the core power tables.
// ---------------------------------------------------------------------------
void power_rows_cov(
    const double a,                  // shared scale factor
    const int nrow,                  // independent wavenumber rows
    const int ncol,                  // samples per row
    const double* const* k,          // positive physical wavenumbers
    const int linear,               // linear or configured nonlinear power
    double* const* power            // caller-owned output rows
  )
{
  if (nrow < 1
      || ncol < 1) {
    log_fatal("power_rows_cov needs positive row and column counts");
    exit(1);
  }

  // The nonlinear reader shares the core's run-mode latch (cosmo3D.c,
  // pdelta_dispatch), a static the first Pdelta-family call writes. On
  // the production path this function can be that first call, so touch
  // the dispatch serially here; the workers below then only read it.
  // The linear reader is stateless, and one extra sample costs nothing.
  if (linear) {
    (void) p_lin(k[0][0], a);
  } else {
    (void) Pdelta(k[0][0], a);
  }

  // Each worker reads a complete row at the common redshift. Its output
  // does not depend on any other row, so no synchronization is needed
  // between samples and no sum changes with the number of workers.
  #pragma omp parallel for schedule(static)
  for (int row=0; row<nrow; row++) {
    if (linear) {
      p_lin_at_a(a, k[row], ncol, power[row]);
    } else {
      Pdelta_at_a(a, k[row], ncol, power[row]);
    }
  }
}

// ---------------------------------------------------------------------------
// Integrate each catalog's lensing efficiency on a covariance-owned grid.
//
// In a flat universe a foreground shell at chi lenses a source at chi'
// with efficiency (1-chi/chi'). Average over the normalized source density:
//
//   g(chi) = integral_chi^infinity dchi' n_chi(chi') (1-chi/chi')
//          = A(chi) - chi B(chi),
//   A = integral_a_min^a da' n_z(z(a'))/a'^2,
//   B = integral_a_min^a da' n_z(z(a'))/[a'^2 chi(a')].
//
// The two cumulative integrals avoid a separate nested integral for every
// foreground shell. This is the same factorization as g_tomo/g_lens in
// redshift_spline.c, with a covariance-owned grid and no foreground cut.
// Their upper endpoint is a=1. This implementation assumes n_z/chi tends
// to zero at the observer, as for a catalog with a positive lower-redshift
// cutoff. It assigns that limit explicitly. Checking n_z(0)=0 below is a
// necessary endpoint check; it does not establish the limiting behavior.
//
// Parameters: amin is the far end of the integration volume; nwindow is
// the number of uniform-a samples; nfield is lenses followed by sources.
// Returns [field][nwindow] g, allocated by malloc2d; the caller frees it.
// Linear interpolation of g uses direct arithmetic on this uniform grid.
// The two SIMD lanes are A and B, not parts of one sum: cumulative order
// is preserved. Each worker owns one field. No core table is changed.
// ---------------------------------------------------------------------------
static double** lensing_efficiency_cov(
    const double amin, // far endpoint of the supplied radial volume
    const int nwindow, // covariance grid size
    const int nlens,   // number of lens catalogs
    const int nfield   // total number of catalogs
  )
{
  const double da = (1.0-amin)/(nwindow-1);

  // Geometry rows hold scale factor and distance on the same uniform grid.
  // They are shared across fields; each efficiency row belongs to one field.
  double** geometry = (double**) malloc2d(2, nwindow);
  double** efficiency = (double**) malloc2d(nfield, nwindow);

  for (int node=0; node<nwindow; node++) {
    const double a = node == nwindow-1 ? 1.0 : amin+node*da;
    geometry[0][node] = a;
    geometry[1][node] = chi(a);
  }

  // The endpoint distance vanishes exactly; it is never used as a divisor.
  geometry[1][nwindow-1] = 0.0;

  // Matter at distance chi lenses only galaxies farther away, at chi'>chi.
  // Each such galaxy contributes the geometric factor 1-chi/chi'. Averaging
  // this factor over a catalog gives two simpler integrals: A is the
  // fraction of galaxies behind chi; B is that same fraction weighted by
  // 1/chi'. Their combination A-chi*B is the lensing efficiency g(chi).
  //
  // The grid begins at the far boundary and moves toward us as a increases.
  // Each step includes another slice of galaxies in A and B. Keeping the
  // accumulated values avoids reintegrating all the farther slices at every
  // distance. One loop iteration constructs this whole g row for one catalog,
  // so different OpenMP workers can own different catalogs independently.
  // SIMD lane 0 accumulates A and lane 1 accumulates B: both use the same
  // integration steps, but their integrands differ by the factor 1/chi'.
  #pragma omp parallel for schedule(static)
  for (int field=0; field<nfield; field++) {
    double* restrict row = efficiency[field];

    // Scalar equivalent, starting with A=B=previous_A=previous_B=0:
    //   current_A = density;
    //   current_B = density*inverse_distance;
    //   if (node > 0) {
    //     A = fma(da/2, previous_A+current_A, A);
    //     B = fma(da/2, previous_B+current_B, B);
    //   }
    //   row[node] = A-distance*B;
    //   previous_A = current_A;
    //   previous_B = current_B;
    // Here density and inverse_distance are evaluated below at each node.
    // A counts sources behind the lens; B weights them by inverse distance.
    // The two lanes hold these different integrals at the same radial step.
    // Initialize the two previous integrands to zero. Lane 0 represents
    // n_z/a^2 for A; lane 1 represents n_z/(a^2 chi) for B.
    v2d vprevious = simde_mm_setzero_pd();

    // Begin both cumulative integrals at zero at the far radial boundary.
    v2d vintegral = simde_mm_setzero_pd();

    // Copy da/2 to both lanes: each trapezoid is half the step size times
    // the sum of the integrand at its two endpoints.
    const v2d vhalf_step = simde_mm_set1_pd(da/2.0);

    // Moving to the next a sample adds the galaxies in one new radial slice.
    // Its contribution to A is the integral of n_z/a^2 across that interval;
    // for B the integrand is n_z/(a^2*chi). The 1/a^2 converts the catalog's
    // density per unit redshift into a density per unit scale factor.
    //
    // We approximate each integrand as a straight line between its two
    // endpoint samples. The area below that line is the interval width da
    // times the mean endpoint value: da*(previous+current)/2. This is the
    // trapezoidal rule. The first node has no preceding interval, so both
    // integrals remain zero there. Later nodes add one such area per lane.
    // SIMD applies this same rule to A and B together without mixing them.
    // At each node, A-chi*B then gives g for matter at that node's distance.
    for (int node=0; node<nwindow; node++) {
      const double a = geometry[0][node];
      const double distance = geometry[1][node];
      const double z = 1.0/a-1.0;
      double density;

      // Change variables from redshift to scale factor: |dz/da| = 1/a^2.
      if (field < nlens) {
        density = nz_lens_photoz(z, field)/(a*a);
      } else {
        density = nz_source_photoz(z, field-nlens)/(a*a);
      }

      // At the observer chi=0, direct division would give 0/0. Use the
      // assumed zero limit of n_z/chi, and reject nonzero endpoint density.
      double inverse_distance = 0.0;
      if (node < nwindow-1) {
        inverse_distance = 1.0/distance;
      } else if (density != 0.0) {
        log_fatal("lensing_efficiency_cov needs n(z=0)=0; field %d "
                  "has density %g", field, density);
        exit(1);
      }

      // set_pd takes the high lane first. Lane 0 gets the A integrand
      // density; lane 1 gets the B integrand density/chi.
      const v2d vcurrent = simde_mm_set_pd(density*inverse_distance,
                                         density);

      if (node > 0) {
        // Add old and new endpoints within each integrand lane. This is
        // the trapezoid's endpoint sum, not an addition of A to B.
        const v2d vendpoints = simde_mm_add_pd(vprevious, vcurrent);

        // Add (da/2)*endpoint_sum to each cumulative integral. fmadd has
        // one native-FMA rounding for each multiply-plus-add operation.
        vintegral = simde_mm_fmadd_pd(vhalf_step, vendpoints, vintegral);
      }

      double integral[2];

      // Copy A into integral[0] and B into integral[1]. storeu accepts
      // this stack array without requiring a vector-aligned address.
      simde_mm_storeu_pd(integral, vintegral);

      // Recombine the two integrals into the lensing efficiency g=A-chi*B.
      row[node] = integral[0]-distance*integral[1];

      // The current integrands become the next trapezoid's old endpoints.
      vprevious = vcurrent;
    }
  }

  free(geometry);
  return efficiency;
}


// -----------------------------------------------------------------------
// Populate density, lensing and signed NLA windows at supplied a samples.
//
// Both Gaussian quadrature and FFTLog need the same physical windows.
// Their sample positions differ, but the efficiency interpolation and
// catalog conventions must not differ. geometry[0] already holds a;
// geometry[3] initially holds da weights (unused by the FFTLog consumer).
// The helper converts those weights to dchi and fills the other rows.
// -----------------------------------------------------------------------
static void fill_radial_cov(
    struct radial_cov* radial, // owner of geometry and window arrays
    const double amin,         // far edge for cumulative efficiencies
    const int nwindow,         // uniform-a efficiency samples
    const int include_ia       // include the signed linear alignment field
  )
{
  const int nfield = radial->nlens+radial->nsource;
  // --- 2. INITIALIZE CORE READERS BEFORE PARALLEL SAMPLING ---

  // Warm the core's geometry and each per-bin window before workers read
  // them. This is setup work, so an explicit serial pass is preferable to
  // locks around individual table reads in the integration loops.
  const double a_first = radial->geometry[0][0];
  const struct chis distance = chi_all(a_first);
  const double hubble_first = hoverh0v2(a_first, distance.dchida);
  const double growth_first = growfac(a_first);

  // Visit each catalog once to trigger any lazy core table construction.
  // This serial pass prepares density, bias and IA readers for the later
  // parallel sampling; it does not compute the covariance windows yet.
  for (int field=0; field<nfield; field++) {
    if (field < radial->nlens) {
      (void) W_gal(a_first, field, hubble_first);
      (void) gb1(1.0/a_first-1.0, field);
      (void) gbmag(0.0, field);
    } else {
      const int source = field-radial->nlens;
      (void) W_source(a_first, source, hubble_first);
      if (include_ia) {
        (void) IA_A1_Z1(a_first, growth_first, source);
      }
    }
  }

  // --- 3. BUILD THE EFFICIENCY GRID AND COMMON INTEGRATION NODES ---

  // Efficiency uses uniform a spacing for direct interpolation; radial
  // integration uses Gaussian nodes on the caller's supplied panels.
  double** efficiency = lensing_efficiency_cov(amin, nwindow,
                                               radial->nlens, nfield);
  const double inv_da = (nwindow-1)/(1.0-amin);

  // --- 4. CONVERT THE MEASURE AND SAMPLE EACH FIELD WINDOW ---

  // Projected correlations add contributions along the line of sight,
  // weighted by how strongly each catalog responds at that distance.
  // These windows include local galaxy density, lensing by foreground
  // matter, and intrinsic alignment. They must refer to the same shells
  // so their products later describe correlations of the same matter.
  // Each worker fills all windows at one common sample. Convert its da
  // weight with |dchi/da|: increasing a moves toward us, but the physical
  // integration measure must remain a positive distance interval.
  #pragma omp parallel for schedule(static)
  for (int node=0; node<radial->nnode; node++) {
    const double a = radial->geometry[0][node];
    const double z = 1.0/a-1.0;
    const struct chis distance = chi_all(a);
    const double fk = f_K(distance.chi);

    if (!(fk > 0.0)
        || !(distance.dchida > 0.0)) {
      log_fatal("radial_inputs_cov: nonpositive distance or measure at "
                "a=%g; check the distance table and panel endpoints", a);
      exit(1);
    }

    const double hubble = hoverh0v2(a, distance.dchida);
    const double growth = growfac(a);

    // chi_all returns the positive magnitude |dchi/da|. Thus multiplying
    // the positive Gaussian da weight gives a positive dchi measure.
    radial->geometry[1][node] = distance.chi;
    radial->geometry[2][node] = fk;
    radial->geometry[3][node] *= distance.dchida;

    // Direct indexing on the uniform efficiency grid: the integer part
    // selects a left endpoint, and the fraction blends the two values.
    const double position = (a-amin)*inv_da;
    const int left = (int) position;
    const double fraction = position-left;
    const double prefactor = 1.5*cosmology.Omega_m*fk/a;

    // At this fixed distance, interpolate each catalog's efficiency and
    // form its physical windows. Store contributions separately because
    // their ell-dependent factors are supplied by limber_spectra_cov.
    for (int field=0; field<nfield; field++) {
      // Interpolate g between its two bracketing uniform-a samples.
      const double g = (1.0-fraction)*efficiency[field][left]
                       +fraction*efficiency[field][left+1];

      if (field < radial->nlens) {
        // Galaxy density and magnification have different ell factors;
        // keep their radial windows separate until the spectrum stage.
        radial->window[0][field][node] = gb1(z, field)
                                       *W_gal(a, field, hubble);
        radial->window[1][field][node] = gbmag(z, field)
                                       *prefactor*g;
      } else {
        // Store the source density for audits, the lensing efficiency
        // for shear, and the negative NLA term in its own window role.
        const int source = field-radial->nlens;
        radial->window[0][field][node] = W_source(a, source, hubble);
        radial->window[1][field][node] = prefactor*g;
        if (include_ia) {
          radial->window[2][field][node] = -W_source(a, source, hubble)
                                         *IA_A1_Z1(a, growth, source);
        }
      }
    }
  }

  free(efficiency);
}


// ---------------------------------------------------------------------------
// Sample the radial windows on a common, positive quadrature rule.
//
// A projected field is A(n) = integral dchi W_A(chi) delta(chi*n).
// In the Limber approximation its cross spectrum with B is
//
//   C_AB(ell) = integral dchi W_A W_B P((ell+1/2)/f_K, a)/f_K^2.
//
// See Krause & Eifler, arXiv:1601.05779, Eqs. 4-7. Their flat-sky
// expression is supplemented by the core's spin and magnification
// transfer factors in limber_spectra_cov below.
//
// Use the same chi nodes for every pair, including pairs absent from the
// measured data vector. At each node W_A W_B is an outer product. With
// positive integration weights and P >= 0, their sum is a positive
// semidefinite field matrix. A negative cross spectrum is allowed: two
// windows can have opposite signs because of magnification or alignment.
//
// The input intervals are in scale factor, which runs in the opposite
// direction to distance. geometry[3] stores da * |dchi/da|, a positive
// distance measure. It does not contain f_K^-2: SSC and cNG need different
// distance powers and reuse this same geometry.
//
// Separate window contributions have distinct multipole dependence:
//   lens:   window[0] = b1 n_l(z) H/H0; window[1] = b_mag W_mag;
//   source: window[1] = W_kappa; window[2] = -W_source A1.
// Source window[0] also stores the unbiased source density for input
// audits; it does not enter the shear spectrum. All windows have units of
// inverse distance. This builder supplies the linear/NLA part of IA;
// higher-order TATT correlations are added separately by ia_cov.c.
// They cannot be represented by a single deterministic field window. This function deliberately uses linear bias.
//
// Parameters and ownership:
//   npanel, a_edges - common integration intervals, strictly inside (0,1)
//   nquad - positive, tabulated GSL rule size per interval
//   nwindow - number of uniform-a samples for cumulative efficiencies
//   include_ia - include the core's NLA amplitude when 1; otherwise omit IA
// Returns a heap-owned snapshot; release it with free_radial_cov.
//
// There is no persistent covariance cache. Recreate the snapshot after a
// cosmology, photo-z, bias, IA or redshift-distribution change. The public
// core readers still have lazy tables: initialize them serially before
// the parallel fill. Call this entry outside any OpenMP region.
// ---------------------------------------------------------------------------
struct radial_cov* radial_inputs_cov(
    const int npanel,       // number of common integration panels
    const double* a_edges,  // increasing scale-factor edges
    const int nquad,        // tabulated nodes per panel
    const int nwindow,      // cumulative lensing-efficiency grid size
    const int include_ia    // whether to include NLA
  )
{
  if (npanel < 1
      || nwindow < 2
      || redshift.clustering_nbin < 1
      || redshift.shear_nbin < 1
      || (include_ia != 0
          && include_ia != 1)) {
    log_fatal("radial_inputs_cov needs panels, nwindow >= 2, bins, "
              "and include_ia = 0 or 1");
    exit(1);
  }
  if (fabs(cosmology.Omega_m+cosmology.Omega_v-1.0) > 1.e-10) {
    log_fatal("radial_inputs_cov lensing efficiency requires flat geometry");
    exit(1);
  }
  if (nquad != 64
      && nquad != 96
      && nquad != 128
      && nquad != 256
      && nquad != 512
      && nquad != 1024) {
    log_fatal("radial_inputs_cov: nquad=%d is not a supported "
              "tabulated rule (64,96,128,256,512,1024)", nquad);
    exit(1);
  }
  if (include_ia
      && nuisance.IA_MODEL != IA_MODEL_NLA
      && nuisance.IA_MODEL != IA_MODEL_TATT) {
    log_fatal("radial_inputs_cov supports NLA/TATT linear windows; IA_MODEL=%d",
              nuisance.IA_MODEL);
    exit(1);
  }
  for (int edge=0; edge<=npanel; edge++) {
    if (!isfinite(a_edges[edge])
        || a_edges[edge] <= 0.0
        || a_edges[edge] >= 1.0
        || (edge > 0
            && a_edges[edge] <= a_edges[edge-1])) {
      log_fatal("radial_inputs_cov: edge %d = %g; supply strictly "
                "increasing scale factors inside (0,1)",
                edge, a_edges[edge]);
      exit(1);
    }
  }

  // --- 1. ALLOCATE THE SHARED RADIAL SNAPSHOT ---

  // Allocate by physical role, rather than one allocation per field.
  // House multidimensional arrays have padded rows: free each parent
  // once, and never flatten them for a memset or a memcpy.
  struct radial_cov* radial = malloc(sizeof(*radial));
  if (radial == NULL) {
    log_fatal("radial_inputs_cov: cannot allocate the snapshot");
    exit(1);
  }

  radial->nnode = npanel*nquad;
  radial->nlens = redshift.clustering_nbin;
  radial->nsource = redshift.shear_nbin;
  const int nfield = radial->nlens + radial->nsource;

  // Geometry rows are a, chi, f_K and dchi weight. Window roles are
  // density, lensing/magnification and signed intrinsic alignment.
  radial->geometry = (double**) malloc2d(4, radial->nnode);
  radial->window = (double***) malloc3d(3, nfield, radial->nnode);
  zero3d(radial->window, 3, nfield, radial->nnode);

  gsl_integration_glfixed_table* rule = malloc_gslint_glfixed(nquad);

  // Map the Gaussian rule onto each scale-factor panel, then concatenate
  // its samples into geometry's common a and da-weight rows.
  for (int panel=0; panel<npanel; panel++) {
    // One node supplies an a value and its positive integration weight;
    // its global index identifies the same shell in every field window.
    for (int node=0; node<nquad; node++) {
      const int index = panel*nquad+node;
      double a;       // quadrature abscissa in this scale-factor interval
      double weight;  // positive quadrature weight, including interval width

      gsl_integration_glfixed_point(a_edges[panel], a_edges[panel+1],
                                    node, &a, &weight, rule);

      // Retain the da weight here; the next stage multiplies by |dchi/da|.
      radial->geometry[0][index] = a;
      radial->geometry[3][index] = weight;
    }
  }

  gsl_integration_glfixed_table_free(rule);

  fill_radial_cov(radial, a_edges[0], nwindow, include_ia);
  return radial;
}


// -----------------------------------------------------------------------
// Sample the same field windows uniformly in ln(chi) for FFTLog.
//
// The lower distance is positive because ln(0) is undefined. Its omitted
// foreground must be tested by lowering chi_min; it is not the zero guard
// of the FFT. The far distance corresponds to amin, which must enclose
// both catalogs. Increasing nchi by intervals preserves old grid nodes.
// The shared fill keeps the Limber and non-Limber window models identical.
// -----------------------------------------------------------------------
struct radial_cov* radial_logchi_cov(
    const double amin,      // far boundary in scale factor
    const double chi_min,   // positive near distance, in c/H0
    const int nchi,         // physical samples, including both endpoints
    const int nwindow,      // uniform-a efficiency grid
    const int include_ia    // signed NLA source contribution
  )
{
  const double chi_max = chi(amin);
  if (nchi < 3
      || chi_min <= 0.0
      || chi_min >= chi_max
      || nwindow < 2) {
    log_fatal("radial_logchi_cov: invalid distance or window grid");
    exit(1);
  }
  (void) a_chi(chi_min);
  struct radial_cov* radial = malloc(sizeof(*radial));
  if (radial == NULL) {
    log_fatal("radial_logchi_cov: workspace allocation failed");
    exit(1);
  }
  radial->nnode = nchi;
  radial->nlens = redshift.clustering_nbin;
  radial->nsource = redshift.shear_nbin;
  const int nfield = radial->nlens+radial->nsource;
  radial->geometry = (double**) malloc2d(4, nchi);
  radial->window = (double***) malloc3d(3, nfield, nchi);
  zero3d(radial->window, 3, nfield, nchi);
  const double step = log(chi_max/chi_min)/(nchi-1);

  // FFTLog integrates in log distance itself, so these samples carry no
  // Gaussian da measure. Store zero in that unused row explicitly.
  for (int node=0; node<nchi; node++) {
    const double distance = chi_min*exp(node*step);
    radial->geometry[0][node] = node == nchi-1 ? amin : a_chi(distance);
    radial->geometry[3][node] = 0.0;
  }
  fill_radial_cov(radial, amin, nwindow, include_ia);
  return radial;
}


// Release the grouped arrays and their owner. No other object owns a row.
void free_radial_cov(struct radial_cov* radial)
{
  free(radial->window);
  free(radial->geometry);
  free(radial);
}



// ---------------------------------------------------------------------------
// Build all Limber field spectra, with common windows and radial nodes.
//
// The scalar equation for each output is
//   C_AB = sum_p [dchi_p P_p/f_K,p^2] W_A,p W_B,p.
//
// First precompute the spectrum and complete field windows. This hoists
// power-spectrum reads and RSD evaluations out of the field-pair loop.
// Then integrate pairs in groups of two. The two SIMD lanes accumulate
// two different spectra, each in increasing radial-node order. Threads
// own different (ell, pair-group) outputs; there is no cross-thread sum.
//
// Lens windows contain density + ell(ell+1)/(ell+1/2)^2 magnification.
// If requested, the same RSD window is added to each lens wherever that
// field appears, including lens-source pairs. Using RSD for gg but not
// gs would describe two different random fields with the same name and
// would lose the positive-semidefinite construction.
// Source windows contain lensing minus NLA, multiplied by
//   sqrt[(ell-1) ell (ell+1) (ell+2)]/(ell+1/2)^2.
// These are the core C_ell conventions. Converting to observed shear
// spectra for unit-normalized spin kernels is a separate per-field
// rescaling; never apply that signal rescaling to white shape noise.
//
// Parameters:
//   radial - current snapshot from radial_inputs_cov
//   nell, ell - caller's multipole grid (does not alter Ntable)
//   linear - select linear total-matter P or the core's run-mode Pdelta
//   include_rsd - one common lens-field choice, 0 or 1
//   spectra - caller-owned triangular pair rows, each of length nell
//
// Output and inputs must not overlap. Array entries are overwritten.
// Scratch is grouped by role and released before return. This function
// reads the current core P tables, so the snapshot and core must describe
// the same cosmology. There is no covariance cache or hidden boost.
// ---------------------------------------------------------------------------
void limber_spectra_cov(
    const struct radial_cov* radial, // shared radial rule and base windows
    const int nell,                 // number of multipole nodes
    const double* ell,              // supplied multipoles
    const int linear,               // linear or run-mode matter power
    const int include_rsd,          // common lens RSD choice
    double* const* spectra          // triangular pair output
  )
{
  if (nell < 1
      || (linear != 0
          && linear != 1
          && linear != 2)
      || (include_rsd != 0
          && include_rsd != 1)) {
    log_fatal("limber_spectra_cov needs nell > 0, power mode 0..2 and RSD 0/1");
    exit(1);
  }
  for (int index=0; index<nell; index++) {
    if (!isfinite(ell[index])
        || ell[index] < 1.0) {
      log_fatal("limber_spectra_cov: ell[%d]=%g; need finite ell >= 1",
                index, ell[index]);
      exit(1);
    }
  }

  const int nnode = radial->nnode;
  const int nfield = radial->nlens+radial->nsource;
  const int npair = nfield*(nfield+1)/2;

  // --- 1. ENUMERATE ALL FIELD PAIRS AND PREPARE SHARED READERS ---

  // Include pairs excluded from the data vector: the Wick contractions
  // of retained observables can still need those cross spectra.
  int** pairs = (int**) malloc2d_int(2, npair);
  int pair = 0;

  for (int first=0; first<nfield; first++) {
    for (int second=first; second<nfield; second++) {
      pairs[0][pair] = first;
      pairs[1][pair] = second;
      pair++;
    }
  }

  // The two power roles share an allocation. Role 0 holds k during the
  // batched P(k,a) read, then becomes dchi*P/f_K^2; role 1 stores P.
  // Node-major rows let Pdelta_at_a locate the redshift bracket once per
  // row. Windows are ell-major for contiguous integration over distance.
  double*** power = (double***) malloc3d(2, nnode, nell);
  double*** window = (double***) malloc3d(nell, nfield, nnode);

  // Complete lazy power and RSD table construction before worker reads.
  if (linear) {
    (void) p_lin(1.0, radial->geometry[0][0]);
  } else {
    (void) Pdelta(1.0, radial->geometry[0][0]);
  }
  if (include_rsd) {
    const double a = radial->geometry[0][0];
    (void) a_chi(radial->geometry[1][0]);
    (void) W_RSD(ell[0]+0.5, a, a, 0);
  }

  const double chi_max = cosmology.chi[1][cosmology.chi_nz-1]
                         /cosmology.coverH0;

  // --- 2. READ POWER ONCE PER NODE AND FORM COMPLETE FIELD WINDOWS ---

  // Under Limber, an angular mode ell samples the matter spectrum near
  // k=(ell+1/2)/f_K. At a fixed distance this P(k,a) is common to every
  // catalog pair; only their windows differ. Read all these powers once
  // per distance, and attach the projection weight dchi/f_K^2.
  // The spin, magnification and RSD factors then turn the base windows
  // into complete fields at each ell. Each worker owns one distance;
  // the next loop reuses its results across every catalog pairing.
  #pragma omp parallel for schedule(static)
  for (int node=0; node<nnode; node++) {
    const double a = radial->geometry[0][node];
    const double fk = radial->geometry[2][node];
    double* restrict k = power[0][node];
    double* restrict pk = power[1][node];

    // All k in this batch share one redshift bracket in the power reader.
    for (int index=0; index<nell; index++) {
      k[index] = (ell[index]+0.5)/fk;
    }

    if (linear == 2) {
      // Subtract precisely the separable field used by FFTLog: its
      // anchor spectrum at a=1 times the same supplied growth squared.
      // Using p_lin(k,a) instead would leave a scale-dependent residual.
      p_lin_at_a(1.0, k, nell, pk);
      const double growth = growfac(a);
      for (int index=0; index<nell; index++) {
        pk[index] *= growth*growth;
      }
    } else if (linear) {
      p_lin_at_a(a, k, nell, pk);
    } else {
      Pdelta_at_a(a, k, nell, pk);
    }

    // For each ell at this distance, combine its geometric spin factors
    // with the base windows and optional RSD. Retain dchi*P/f_K^2 once
    // per ell, instead of recomputing it for every pair of catalogs.
    for (int index=0; index<nell; index++) {
      const double l = ell[index];
      const double ell_shift = l+0.5;
      const double magnification = l*(l+1.0)/(ell_shift*ell_shift);
      const double shear = sqrt((l-1.0)*l*(l+1.0)*(l+2.0))
                           /(ell_shift*ell_shift);

      // The temporary k row is no longer needed. Reuse it for the common
      // positive integration factor dchi*P/f_K^2 at each multipole.
      k[index] = radial->geometry[3][node]*pk[index]/(fk*fk);

      // The shifted-distance RSD approximation samples a second shell.
      // If that shell is beyond the supplied distance table, stop rather
      // than silently discarding its contribution or extrapolating a_chi.
      double a_shift = a;
      if (include_rsd) {
        const double chi_shift = radial->geometry[1][node]
                                 *(ell_shift+1.0)/ell_shift;
        if (chi_shift > chi_max) {
          log_fatal("limber_spectra_cov: RSD distance %g exceeds %g; "
                    "extend the distance table", chi_shift, chi_max);
          exit(1);
        }
        a_shift = a_chi(chi_shift);
      }

      // Combine contributions to one observed field before pairing fields.
      // This retains density-magnification and lensing-IA cross terms.
      for (int field=0; field<nfield; field++) {
        if (field < radial->nlens) {
          double value = radial->window[0][field][node]
                         +magnification*radial->window[1][field][node];
          if (include_rsd) {
            value += W_RSD(ell_shift, a, a_shift, field);
          }
          window[index][field][node] = value;
        } else {
          window[index][field][node] = shear
              *(radial->window[1][field][node]
                +radial->window[2][field][node]);
        }
      }
    }
  }

  // --- 3. INTEGRATE EACH FIELD PAIR ON THE SAME RADIAL RULE ---

  // In Limber, a shared shell contributes W_A*W_B*P*dchi/f_K^2 to C_AB.
  // Multiplying the two windows selects matter to which both fields
  // respond; summing over shells gives their projected cross spectrum.
  // Use the same positive integration rule for every pair so this remains
  // a consistent matrix of field correlations, including unmeasured pairs
  // needed by the covariance. One worker handles an ell and two pairs.
  // SIMD shares the matter power and radial weight, but each lane keeps
  // its own pair of windows and its own complete spectrum through storage.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int index=0; index<nell; index++) {
    // At this ell, process two catalog pairs together using the prepared
    // windows. Lane 0 and lane 1 each produce one independent spectrum.
    for (int first_pair=0; first_pair<npair; first_pair+=2) {
      // Repeat the last valid pair in an unused lane of an odd-sized
      // block. Only valid outputs are stored, so no padding is required.
      const int next_pair = first_pair+1 < npair ? first_pair+1 : npair-1;
      const double* restrict left0 = window[index][pairs[0][first_pair]];
      const double* restrict right0 = window[index][pairs[1][first_pair]];
      const double* restrict left1 = window[index][pairs[0][next_pair]];
      const double* restrict right1 = window[index][pairs[1][next_pair]];

      // Scalar equivalent for either catalog pair p:
      //   sum = 0;
      //   for (int node=0; node<nnode; node++) {
      //     product = window[index][pairs[0][p]][node]
      //               *window[index][pairs[1][p]][node];
      //     sum = fma(product, power[0][node][index], sum);
      //   }
      //   spectra[p][index] = sum;
      // The shared power table already includes dchi/f_K^2. SIMD carries
      // these sums for p=first_pair,next_pair in separate lanes.
      // Start two independent C_AB sums at zero. Lane 0 owns first_pair;
      // lane 1 owns next_pair. Neither lane is part of the other's sum.
      v2d vtotal = simde_mm_setzero_pd();

      // Walk all radial nodes in increasing order. SIMD updates the two
      // spectra together using the same dchi*P/f_K^2, but their own W_A W_B.
      for (int node=0; node<nnode; node++) {
        // set_pd takes the high lane first: pack the left-field windows
        // as [left(first_pair),left(next_pair)] in lanes 0 and 1.
        const v2d vleft = simde_mm_set_pd(left1[node], left0[node]);

        // Pack the right-field windows in the same pair order. set_pd's
        // last argument goes to lane 0, keeping the fields matched.
        const v2d vright = simde_mm_set_pd(right1[node], right0[node]);

        // Copy this node's dchi*P/f_K^2 weight to both pair lanes.
        const v2d vmeasure = simde_mm_set1_pd(power[0][node][index]);

        // Multiply W_A*W_B within each pair, with no cross-lane products.
        const v2d vproduct = simde_mm_mul_pd(vleft, vright);

        // Add W_A*W_B*dchi*P/f_K^2 to each spectrum's own radial sum.
        // fmadd fuses product and addition with one native-FMA rounding.
        vtotal = simde_mm_fmadd_pd(vproduct, vmeasure, vtotal);
      }

      double result[2];

      // Copy first_pair/next_pair spectra into result[0/1]. storeu permits
      // this ordinary double array without special vector alignment.
      simde_mm_storeu_pd(result, vtotal);

      // Store only real pairs; discard a duplicate final lane if present.
      spectra[first_pair][index] = result[0];
      if (first_pair+1 < npair) {
        spectra[first_pair+1][index] = result[1];
      }
    }
  }

  free(window);
  free(power);
  free(pairs);
}
