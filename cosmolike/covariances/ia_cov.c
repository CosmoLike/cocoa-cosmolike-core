#include <math.h>
#include <stdlib.h>

#include "ia_cov.h"
#include "cosmolike/basics.h"
#include "cosmolike/cosmo3D.h"
#include "cosmolike/IA.h"
#include "cosmolike/pt_cfastpt.h"
#include "cosmolike/structs.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"
#include "simde/x86/fma.h"

typedef simde__m128d v2d;

// -----------------------------------------------------------------------
// Complete the Gaussian spectra with tidal alignment and tidal torquing.
//
// NLA already contributes GG, GI+IG and II through the signed source
// window W_lensing-W_source*C1. Here add only the higher-order pieces of
// Blazek et al. (arXiv:1708.09247), using the core FAST-PT kernels and
// amplitude convention also used in cosmo2D.c. C1 is positive for a
// positive A1; the minus sign enters the matter-intrinsic correlation.
// C2 here omits the conventional factor 5, supplied explicitly below.
//
// The density-weighted tidal field has amplitude C1*bTA. The quadratic
// tidal field has amplitude -5*C2 in this positive-C1 convention. Their
// auto/cross powers create both E and B modes. Parity forbids gB and EB,
// so only source-source pairs have nonzero BB. Shot/shape noise is not
// part of these spectra and must not receive an IA or spin factor.
//
// Limber associates k=(ell+1/2)/chi with a radial shell. Compute the ten
// one-loop kernels once per (ell,shell), then reuse them for all pairs.
// The core table was constructed with cubic upsampling; the hot lookup
// here is direct-index linear interpolation on that dense log-k grid.
// Its z=0 kernels evolve with D^4, because each loop contains two linear
// powers. The shared growth convention is the same one used by IA.c.
//
// Inputs and outputs have the axes in ia_cov.h. Call serially after the
// radial builder; initialize FAST-PT before workers read its cached table.
// No SSC/cNG response or halo approximation is changed by this function.
// -----------------------------------------------------------------------
void tatt_spectra_cov(
    const struct radial_cov* radial, // common rule and NLA window snapshot
    const int nell,                 // multipole samples
    const double* ell,              // positive multipoles
    double* const* ee,              // NLA E input, TATT E output
    double* const* bb               // TATT B output
  )
{
  if (nuisance.IA_MODEL != IA_MODEL_TATT
      || nell < 1) {
    log_fatal("tatt_spectra_cov requires TATT and at least one multipole");
    exit(1);
  }
  const int nlens = radial->nlens;
  const int nsource = radial->nsource;
  const int nfield = nlens+nsource;
  const int npair = nfield*(nfield+1)/2;
  const int nnode = radial->nnode;
  double*** amplitude = (double***) malloc3d(3, nsource, nnode);
  double*** loop = (double***) malloc3d(nell, nnode, 10);
  int** pairs = (int**) malloc2d_int(2, npair);
  int pair = 0;
  for (int first=0; first<nfield; first++) {
    for (int second=first; second<nfield; second++) {
      pairs[0][pair] = first;
      pairs[1][pair] = second;
      pair++;
    }
  }

  // --- 1. SHARE THE AMPLITUDES AND ONE-LOOP POWER AT EACH SHELL ---

  // FAST-PT has persistent core state, including FFTW plans. Build it
  // outside OpenMP before workers perform read-only interpolation.
  get_FPT_IA();
  const double lnk_min = log(FPTIA.krange[RANGE_MIN]);
  const double lnk_max = log(FPTIA.krange[RANGE_MAX]);
  const double inv_step = FPTIA.N/(lnk_max-lnk_min);

  // One shell supplies all source amplitudes and every requested k.
  // Hoisting these quantities avoids catalog-dependent power reads.
  #pragma omp parallel for schedule(static)
  for (int node=0; node<nnode; node++) {
    const double a = radial->geometry[0][node];
    const double growth = growfac(a);
    const double growth4 = growth*growth*growth*growth;
    for (int source=0; source<nsource; source++) {
      amplitude[0][source][node] = IA_A1_Z1(a, growth, source);
      amplitude[1][source][node] = IA_A2_Z1(a, growth, source);
      amplitude[2][source][node] = IA_BTA_Z1(a, growth, source);
    }
    for (int index=0; index<nell; index++) {
      const double lnk = log((ell[index]+0.5)/radial->geometry[2][node]);
      // Match the core's finite FAST-PT support: no one-loop term is
      // extrapolated outside that range. The NLA matter power remains.
      for (int role=0; role<10; role++) {
        loop[index][node][role] = 0.0;
      }
      if (lnk >= lnk_min
          && lnk <= lnk_max) {
        const double position = (lnk-lnk_min)*inv_step;
        int left = (int) position;
        double fraction = position-left;
        // The core convention places its final support edge one cell
        // beyond the table's samples and uses the penultimate value
        // there. Retain this explicit convention for model agreement.
        if (left+1 >= FPTIA.N) {
          left = FPTIA.N-2;
          fraction = 0.0;
        }
        for (int role=0; role<10; role++) {
          const double* values = FPTIA.tab[role];
          loop[index][node][role] = growth4
              *((1.0-fraction)*values[left]+fraction*values[left+1]);
        }
      }
    }
  }

  // --- 2. PROJECT GI/IG AND II CORRECTIONS FOR EVERY FIELD PAIR ---

  // Density-source pairs acquire gI; source-source pairs acquire both GI
  // orderings and II. Retain all crosses, even if a data-vector pair map
  // excludes them. Each worker owns an ell and a complete catalog pair.
  // SIMD lanes keep E and B integrals separate while sharing their shell
  // weight; neither parity channel can leak into the other.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int index=0; index<nell; index++) {
    for (int pair=0; pair<npair; pair++) {
      const int first = pairs[0][pair];
      const int second = pairs[1][pair];
      bb[pair][index] = 0.0;
      if (second < nlens) {
        continue;
      }
      const double l = ell[index];
      const double shift2 = (l+0.5)*(l+0.5);
      const double spin = sqrt((l-1)*l*(l+1)*(l+2))/shift2;
      const int source2 = second-nlens;

      // Scalar equivalent: E += measure*delta_E; B += measure*delta_B.
      // setzero initializes the independent E (lane 0) and B (lane 1) sums.
      v2d integral = simde_mm_setzero_pd();
      for (int node=0; node<nnode; node++) {
        const double* power = loop[index][node];
        const double ta_delta = power[2]+power[3];
        const double mix_delta = power[6]+power[7];
        const double c12 = amplitude[0][source2][node];
        const double c22 = amplitude[1][source2][node];
        const double b2 = amplitude[2][source2][node];
        const double n2 = radial->window[0][second][node];
        const double gi2 = c12*b2*ta_delta-5.0*c22*mix_delta;
        const double distance = radial->geometry[2][node];
        const double measure = radial->geometry[3][node]/(distance*distance);
        double delta_e;
        double delta_b = 0.0;

        if (first < nlens) {
          // gI correlates the lens density/magnification with the source
          // alignment. The minus sign follows the positive-C1 convention.
          const double galaxy = radial->window[0][first][node]
              +l*(l+1)/shift2*radial->window[1][first][node];
          delta_e = -spin*galaxy*n2*gi2;
        } else {
          const int source1 = first-nlens;
          const double c11 = amplitude[0][source1][node];
          const double c21 = amplitude[1][source1][node];
          const double b1 = amplitude[2][source1][node];
          const double n1 = radial->window[0][first][node];
          const double lens1 = radial->window[1][first][node];
          const double lens2 = radial->window[1][second][node];
          const double gi1 = c11*b1*ta_delta-5.0*c21*mix_delta;

          // II combines density-weighted alignment, linear/quadratic
          // cross terms and quadratic auto power. The E expression
          // excludes C11*C12*P, already present in the NLA input.
          const double ii_e = c11*c12
              *(b1*b2*power[4]+(b1+b2)*ta_delta)
              -5.0*(c11*c22+c12*c21)*mix_delta
              -5.0*(c11*b1*c22+c12*b2*c21)*power[8]
              +25.0*c21*c22*power[0];
          const double ii_b = c11*c12*b1*b2*power[5]
              -5.0*(c11*b1*c22+c12*b2*c21)*power[9]
              +25.0*c21*c22*power[1];
          delta_e = spin*spin
              *(-n1*lens2*gi1-n2*lens1*gi2+n1*n2*ii_e);
          delta_b = spin*spin*n1*n2*ii_b;
        }

        // set_pd takes its high lane first: lane 0 receives delta_E,
        // lane 1 delta_B. These are different parity channels, not bins.
        const v2d values = simde_mm_set_pd(delta_b, delta_e);
        // set1 gives both channels the common dchi/f_K^2 measure.
        const v2d weight = simde_mm_set1_pd(measure);
        // fmadd appends one weighted shell to each channel independently.
        integral = simde_mm_fmadd_pd(weight, values, integral);
      }
      double values[2];
      // storeu copies E and B sums to adjacent ordinary doubles.
      simde_mm_storeu_pd(values, integral);
      ee[pair][index] += values[0];
      bb[pair][index] = values[1];
    }
  }
  free(pairs);
  free(loop);
  free(amplitude);
}
