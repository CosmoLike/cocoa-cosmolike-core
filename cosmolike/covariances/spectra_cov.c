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

typedef simde__m128d v2d; // two independent field-pair sums

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
// Their upper endpoint is a=1. When a catalog has no objects at z=0,
// n_z/chi is assigned its zero limiting value there. A catalog nonzero at
// z=0 needs a different endpoint treatment and is rejected explicitly.
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
  double** geometry = (double**) malloc2d(2, nwindow);
  double** efficiency = (double**) malloc2d(nfield, nwindow);
  for (int node=0; node<nwindow; node++) {
    const double a = node == nwindow-1 ? 1.0 : amin+node*da;
    geometry[0][node] = a;
    geometry[1][node] = chi(a);
  }
  // The endpoint distance vanishes exactly; it is never used as a divisor.
  geometry[1][nwindow-1] = 0.0;

  #pragma omp parallel for schedule(static)
  for (int field=0; field<nfield; field++) {
    double* restrict row = efficiency[field];
    v2d vprevious = simde_mm_setzero_pd();
    v2d vintegral = simde_mm_setzero_pd();
    const v2d vhalf_step = simde_mm_set1_pd(da/2.0);
    for (int node=0; node<nwindow; node++) {
      const double a = geometry[0][node];
      const double distance = geometry[1][node];
      const double z = 1.0/a-1.0;
      double density;
      if (field < nlens) {
        density = nz_lens_photoz(z, field)/(a*a);
      } else {
        density = nz_source_photoz(z, field-nlens)/(a*a);
      }
      double inverse_distance = 0.0;
      if (node < nwindow-1) {
        inverse_distance = 1.0/distance;
      } else if (density != 0.0) {
        log_fatal("lensing_efficiency_cov needs n(z=0)=0; field %d "
                  "has density %g", field, density);
        exit(1);
      }
      const v2d vcurrent = simde_mm_set_pd(density*inverse_distance,
                                         density);
      if (node > 0) {
        const v2d vendpoints = simde_mm_add_pd(vprevious, vcurrent);
        vintegral = simde_mm_fmadd_pd(vhalf_step, vendpoints, vintegral);
      }
      double integral[2];
      simde_mm_storeu_pd(integral, vintegral);
      row[node] = integral[0]-distance*integral[1];
      vprevious = vcurrent;
    }
  }
  free(geometry);
  return efficiency;
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
// inverse distance. NLA is the only supported IA model in this builder;
// higher-order IA and galaxy-bias terms cannot be represented by these
// single-field windows. This function deliberately uses linear bias.
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
      && nuisance.IA_MODEL != IA_MODEL_NLA) {
    log_fatal("radial_inputs_cov supports NLA only; IA_MODEL=%d",
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
  radial->geometry = (double**) malloc2d(4, radial->nnode);
  radial->window = (double***) malloc3d(3, nfield, radial->nnode);
  zero3d(radial->window, 3, nfield, radial->nnode);

  // Warm the core's geometry and each per-bin window before workers read
  // them. This is setup work, so an explicit serial pass is preferable to
  // locks around individual table reads in the integration loops.
  const double a_first = (a_edges[0] + a_edges[1])/2.0;
  const struct chis distance = chi_all(a_first);
  const double hubble_first = hoverh0v2(a_first, distance.dchida);
  const double growth_first = growfac(a_first);
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

  double** efficiency = lensing_efficiency_cov(a_edges[0], nwindow,
                                               radial->nlens, nfield);
  const double inv_da = (nwindow-1)/(1.0-a_edges[0]);

  gsl_integration_glfixed_table* rule = malloc_gslint_glfixed(nquad);
  for (int panel=0; panel<npanel; panel++) {
    for (int node=0; node<nquad; node++) {
      const int index = panel*nquad+node;
      double a;       // quadrature abscissa in this scale-factor interval
      double weight;  // positive quadrature weight, including interval width
      gsl_integration_glfixed_point(a_edges[panel], a_edges[panel+1],
                                    node, &a, &weight, rule);
      radial->geometry[0][index] = a;
      radial->geometry[3][index] = weight;
    }
  }
  gsl_integration_glfixed_table_free(rule);

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
    radial->geometry[1][node] = distance.chi;
    radial->geometry[2][node] = fk;
    radial->geometry[3][node] *= distance.dchida;

    // Direct indexing on the uniform efficiency grid: the integer part
    // selects a left endpoint, and the fraction blends the two values.
    const double position = (a-a_edges[0])*inv_da;
    const int left = (int) position;
    const double fraction = position-left;
    const double prefactor = 1.5*cosmology.Omega_m*fk/a;

    for (int field=0; field<nfield; field++) {
      const double g = (1.0-fraction)*efficiency[field][left]
                       +fraction*efficiency[field][left+1];
      if (field < radial->nlens) {
        radial->window[0][field][node] = gb1(z, field)
                                       *W_gal(a, field, hubble);
        radial->window[1][field][node] = gbmag(z, field)
                                       *prefactor*g;
      } else {
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
          && linear != 1)
      || (include_rsd != 0
          && include_rsd != 1)) {
    log_fatal("limber_spectra_cov needs nell > 0 and binary switches");
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

  #pragma omp parallel for schedule(static)
  for (int node=0; node<nnode; node++) {
    const double a = radial->geometry[0][node];
    const double fk = radial->geometry[2][node];
    double* restrict k = power[0][node];
    double* restrict pk = power[1][node];
    for (int index=0; index<nell; index++) {
      k[index] = (ell[index]+0.5)/fk;
    }
    if (linear) {
      p_lin_at_a(a, k, nell, pk);
    } else {
      Pdelta_at_a(a, k, nell, pk);
    }
    for (int index=0; index<nell; index++) {
      const double l = ell[index];
      const double ell_shift = l+0.5;
      const double magnification = l*(l+1.0)/(ell_shift*ell_shift);
      const double shear = sqrt((l-1.0)*l*(l+1.0)*(l+2.0))
                           /(ell_shift*ell_shift);
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

  #pragma omp parallel for collapse(2) schedule(static)
  for (int index=0; index<nell; index++) {
    for (int first_pair=0; first_pair<npair; first_pair+=2) {
      // Repeat the last valid pair in an unused lane of an odd-sized
      // block. Only valid outputs are stored, so no padding is required.
      const int next_pair = first_pair+1 < npair ? first_pair+1 : npair-1;
      const double* restrict left0 = window[index][pairs[0][first_pair]];
      const double* restrict right0 = window[index][pairs[1][first_pair]];
      const double* restrict left1 = window[index][pairs[0][next_pair]];
      const double* restrict right1 = window[index][pairs[1][next_pair]];
      v2d vtotal = simde_mm_setzero_pd();
      for (int node=0; node<nnode; node++) {
        const v2d vleft = simde_mm_set_pd(left1[node], left0[node]);
        const v2d vright = simde_mm_set_pd(right1[node], right0[node]);
        const v2d vmeasure = simde_mm_set1_pd(power[0][node][index]);
        const v2d vproduct = simde_mm_mul_pd(vleft, vright);
        vtotal = simde_mm_fmadd_pd(vproduct, vmeasure, vtotal);
      }
      double result[2];
      simde_mm_storeu_pd(result, vtotal);
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
