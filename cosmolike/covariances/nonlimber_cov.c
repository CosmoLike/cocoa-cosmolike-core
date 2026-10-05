#include <math.h>
#include <stdlib.h>

#include "nonlimber_cov.h"
#include "fftlog_cov.h"
#include "cosmolike/basics.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/structs.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

typedef simde__m128d v2d;

// -----------------------------------------------------------------------
// Project one common separable linear field without the Limber shortcut.
//
// Write delta(k,a)=D(a)*delta(k,1). Every catalog then has a transfer F,
// and its cross spectrum is (2/pi) integral dlnk k^3 P(k,1) F_A F_B.
// A density transfer integrates chi*W_density*D against j_l(k*chi).
// A lensing/IA transfer integrates chi*(W_lensing+W_IA)*D against
// j_l(k*chi)/(k*chi)^2 and receives an angular derivative factor.
// For galaxy magnification this factor is l(l+1); for shear it is
// sqrt((l-1)l(l+1)(l+2)). Windows include all bias and IA amplitudes.
//
// The caller forms the hybrid of Fang et al. (arXiv:1911.11947):
//   nonlinear Limber + exact separable linear - matched linear Limber.
// Both linear terms MUST use P(k,1) and the same D(a). Using the supplied
// p_lin(k,a) only in the subtraction would spoil cancellation whenever
// those tables have scale-dependent growth. This prescription does not
// implement general unequal-time massive-neutrino evolution.
//
// One common k measure keeps every exact linear pair part of a Gram
// matrix. It includes crossed bins absent from the measured data vector.
// Positivity of this linear matrix does not prove positivity of a hybrid
// matrix; the latter must be checked after its Limber residual is added.
//
// The routine supports flat geometry, zero massive neutrinos, density,
// magnification and linear/NLA alignment, but no RSD transfer. It returns
// ss as a diagnostic too; callers choosing gg/gs corrections may leave
// their shear-shear model Limber. SSC and cNG never call this routine.
// -----------------------------------------------------------------------
void nonlimber_spectra_cov(
    const struct radial_cov* radial, // shared quadrature and windows
    const double amin,              // far radial boundary
    const int nwindow,               // efficiency samples
    const int include_ia,            // include linear alignment
    const int lmax,                  // last integer multipole
    const int nchi,                  // physical logarithmic samples
    const double chi_min,            // near radial boundary in c/H0
    double* const* exact,            // exact-linear triangular rows
    double* const* matched           // matched-Limber triangular rows
  )
{
  const int intervals = nchi-1;
  if (lmax < 2
      || intervals < 64
      || (intervals & (intervals-1)) != 0
      || !isfinite(chi_min)
      || chi_min <= 0.0
      || cosmology.Omega_nu != 0.0
      || fabs(cosmology.Omega_m+cosmology.Omega_v-1.0) > 1.e-10) {
    log_fatal("nonlimber_spectra_cov needs flat massless cosmology, "
              "lmax >= 2, nchi=2^n+1 >= 65 and positive chi_min");
    exit(1);
  }

  // --- 1. PREPARE THE MATCHING LINEAR FIELD ON TWO RADIAL GRIDS ---

  // The quadrature grid supplies the Limber subtraction. FFTLog samples
  // the same physical windows on log distance, independently of that
  // integration rule. Forward transforms are computed only once below.
  const int nfield = radial->nlens+radial->nsource;
  const int ncomponent = radial->nlens+nfield;
  const int npair = nfield*(nfield+1)/2;
  double* multipoles = (double*) malloc1d(lmax-1);
  for (int index=0; index<lmax-1; index++) {
    multipoles[index] = index+2.0;
  }
  limber_spectra_cov(radial, lmax-1, multipoles, 2, 0, matched);
  free(multipoles);

  struct radial_cov* logarithmic = radial_logchi_cov(
      amin, chi_min, nchi, nwindow, include_ia);
  const double dlnchi = log(chi(amin)/chi_min)/intervals;
  double** inputs = (double**) malloc2d(ncomponent, nchi);
  int* powers = (int*) malloc1d_int(ncomponent);
  (void) p_lin(1.0, 1.0);

  // Components are lens densities first, then one spin kernel per field.
  // A spin kernel means magnification for a lens and lensing+IA for a
  // source. Multiplying by chi converts dchi to FFTLog's dlnchi measure.
  #pragma omp parallel for schedule(static)
  for (int component=0; component<ncomponent; component++) {
    const int density = component < radial->nlens;
    const int field = density ? component : component-radial->nlens;
    powers[component] = density ? 0 : 2;
    for (int node=0; node<nchi; node++) {
      const double a = logarithmic->geometry[0][node];
      const double distance = chi_min*exp(node*dlnchi);
      double window = logarithmic->window[0][field][node];
      if (!density) {
        window = logarithmic->window[1][field][node]
                 +logarithmic->window[2][field][node];
      }
      inputs[component][node] = distance*window*growfac(a);
    }
  }
  free_radial_cov(logarithmic);

  // Empty guards each span the physical logarithmic interval. Reading
  // an extra quarter interval at each reciprocal end retains low-k power
  // that a grid beginning at (ell+1)/chi_max would otherwise miss.
  // These physical widths stay fixed when the interval count doubles.
  const int padding = intervals;
  const int extra = intervals/4;
  const int nk = nchi+2*extra;
  struct fftlog_workspace_cov* work = fftlog_create_cov(
      nchi, padding, 4*intervals, extra, chi_min, dlnchi,
      ncomponent, powers, (const double* const*) inputs);
  free(inputs);
  free(powers);

  // --- 2. TRANSFORM BOUNDED MULTIPOLE BLOCKS AND COMBINE FIELD PARTS ---

  // A block supplies enough independent transforms for eight workers
  // without retaining an ell-by-field-by-k cube for the entire survey.
  // Geometry and matter power are shared by every pair at this ell.
  double*** transfer = (double***) malloc3d(16, ncomponent, nk);
  double*** waves = (double***) malloc3d(2, 16, nk);
  int** pairs = (int**) malloc2d_int(2, npair);
  int pair = 0;
  for (int left=0; left<nfield; left++) {
    for (int right=left; right<nfield; right++) {
      pairs[0][pair] = left;
      pairs[1][pair] = right;
      pair++;
    }
  }

  for (int first=2; first<=lmax; first+=16) {
    const int count = lmax-first+1 < 16 ? lmax-first+1 : 16;
    fftlog_execute_cov(work, first, count, waves[0], transfer);

    // Each worker completes an ell. Combining components in increasing
    // field order safely reuses the first nfield transfer rows: every
    // overwritten component has already been consumed at that point.
    #pragma omp parallel for schedule(static)
    for (int index=0; index<count; index++) {
      const double ell = first+index;
      p_lin_at_a(1.0, waves[0][index], nk, waves[1][index]);
      for (int node=0; node<nk; node++) {
        const double k = waves[0][index][node];
        double measure = (2.0/M_PI)*dlnchi*k*k*k;
        if (node == 0
            || node == nk-1) {
          measure *= 0.5;
        }
        waves[1][index][node] *= measure;
      }

      // Scalar equivalent: F = F_density+l(l+1)*F_magnification for
      // lenses, and F = sqrt((l-1)l(l+1)(l+2))*F_spin for sources.
      // This construction combines contributions before pairing fields.
      for (int field=0; field<nfield; field++) {
        const double* spin = transfer[index][radial->nlens+field];
        double* output = transfer[index][field];
        double factor = ell*(ell+1.0);
        if (field >= radial->nlens) {
          factor = sqrt((ell-1.0)*ell*(ell+1.0)*(ell+2.0));
        }
        for (int node=0; node<nk; node++) {
          const double density = field < radial->nlens ? output[node] : 0;
          output[node] = density+factor*spin[node];
        }
      }
    }

    // --- 3. CONTRACT ALL PAIRS WITH A SHARED POSITIVE K MEASURE ---

    // Scalar equivalent for pair (A,B): C=sum_k measure[k]*F_A[k]*F_B[k].
    // Two SIMD lanes accumulate different pairs, in the same k order.
    // Collapsing ell and pair tasks keeps small-bin surveys parallel too.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int index=0; index<count; index++) {
      for (int pair=0; pair<npair; pair+=2) {
        const int next = pair+1 < npair ? pair+1 : pair;
        // setzero starts both independent spectrum sums at zero.
        v2d total = simde_mm_setzero_pd();
        for (int node=0; node<nk; node++) {
          // set_pd takes the high lane first. Lane 0 gets pair's left
          // transfer; lane 1 gets next's left transfer at the same k.
          const v2d left = simde_mm_set_pd(
              transfer[index][pairs[0][next]][node],
              transfer[index][pairs[0][pair]][node]);
          // Keep the right transfer in the same pair order as the left.
          const v2d right = simde_mm_set_pd(
              transfer[index][pairs[1][next]][node],
              transfer[index][pairs[1][pair]][node]);
          // set1 copies the shared dlnk*k^3*P*2/pi measure to both lanes.
          const v2d weight = simde_mm_set1_pd(waves[1][index][node]);
          // mul forms F_A*F_B independently for the two catalog pairs.
          const v2d product = simde_mm_mul_pd(left, right);
          // fmadd accumulates each weighted product, without mixing pairs.
          total = simde_mm_fmadd_pd(weight, product, total);
        }
        double values[2];
        // storeu copies the lane sums to an ordinary, unaligned array.
        simde_mm_storeu_pd(values, total);
        exact[pair][first+index-2] = values[0];
        if (pair+1 < npair) {
          exact[next][first+index-2] = values[1];
        }
      }
    }
  }

  fftlog_free_cov(work);
  free(pairs);
  free(waves);
  free(transfer);
}


// Keep the hybrid addition shared between production and notebook APIs.
// Its inputs have already been validated by their public boundary.
void apply_nonlimber_cov(
    const struct radial_cov* radial, // existing quadrature snapshot
    const double amin,              // far radial boundary
    const int nwindow,               // lensing-efficiency samples
    const int include_ia,            // signed linear alignment
    const int lmax,                  // correction cutoff
    const int nchi,                  // logarithmic radial samples
    const double chi_min,            // near distance in c/H0
    const int nell,                  // requested output multipoles
    const double* ell,               // requested integer ell below cutoff
    double* const* spectra           // nonlinear Limber input, hybrid output
  )
{
  const int nfield = radial->nlens+radial->nsource;
  const int npair = nfield*(nfield+1)/2;
  double*** correction = (double***) malloc3d(2, npair, lmax-1);
  nonlimber_spectra_cov(radial, amin, nwindow, include_ia, lmax, nchi,
      chi_min, correction[0], correction[1]);

  // Only a pair with at least one density field receives the correction.
  // Retain the caller's existing shear-shear spectra, including their IA.
  int pair = 0;
  for (int first=0; first<nfield; first++) {
    for (int second=first; second<nfield; second++) {
      if (first < radial->nlens) {
        for (int index=0; index<nell; index++) {
          const double value = ell[index];
          if (value >= 2.0
              && value <= lmax) {
            const int offset = (int) value-2;
            spectra[pair][index] += correction[0][pair][offset]
                                   -correction[1][pair][offset];
          }
        }
      }
      pair++;
    }
  }
  free(correction);
}
