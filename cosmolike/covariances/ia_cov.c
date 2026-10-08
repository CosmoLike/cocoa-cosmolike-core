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

// SIMD vocabulary. A v2d holds two doubles side by side, in positions
// called lanes 0 and 1, and every call below acts on each lane separately.
// Here lane 0 carries the E-mode integral of one field pair and lane 1 its
// B-mode integral. simde_mm_setzero_pd() sets both lanes to 0.0;
// simde_mm_set1_pd(x) copies x into both; simde_mm_set_pd(high, low) puts
// low in lane 0 and high in lane 1. simde_mm_fmadd_pd(a, b, c) returns
// a*b + c in each lane, rounded once where the processor has a fused
// instruction (NEON on arm64, FMA-enabled x86) and twice otherwise.
// simde_mm_storeu_pd(p, v) writes lane 0 to p[0] and lane 1 to p[1]; the
// address p need not be a multiple of 16 bytes.
typedef simde__m128d v2d;

// ---------------------------------------------------------------------------
// Complete the Gaussian spectra with tidal alignment and tidal torquing.
//
// The tidal alignment and tidal torquing model (TATT; Blazek et al.,
// arXiv:1708.09247) expands the intrinsic shape of a source in the tidal
// field s_ij, the matter density delta and their products. With the core's
// positive amplitudes (IA.c)
//
//   C1   = IA_A1_Z1  = A1(z) Omega_m c1rhocrit_ia/D(a),
//   C2   = IA_A2_Z1  = A2(z) Omega_m c1rhocrit_ia/D(a)^2,
//   b_TA = IA_BTA_Z1,
//
// the intrinsic shape field correlated here is
//
//   I = -[C1 s + C1 b_TA (delta s) - 5 C2 (s s)],
//
// where (delta s) is the density-weighted tidal field and (s s) the
// trace-free quadratic tidal field; their E and B projections on the sky
// enter the spectra. C1 is positive for a positive A1, so the overall
// minus sign enters every matter-intrinsic correlation once and cancels in
// II. C2 here omits the conventional factor 5 of Blazek et al., supplied
// explicitly below (25 where two quadratic fields meet).
//
// NLA already contributes GG, GI+IG and II through the signed source
// window W_kappa - C1 n_s H/H0 of spectra_cov.c. Here add only the
// higher-order pieces, using the core FAST-PT kernels and amplitude
// convention also used in cosmo2D.c. Their auto/cross powers create both
// E and B modes. Parity forbids gB and EB, so only source-source pairs
// have nonzero BB. Shot/shape noise is not part of these spectra and must
// not receive an IA or spin factor.
//
// FAST-PT one-loop rows (FPTIA.tab at z=0, in (c/H0)^3):
//   0/1: (s s) auto power, E/B        2+3: delta with (delta s), E
//   4/5: (delta s) auto power, E/B    6+7: delta with (s s), E
//   8/9: (delta s) with (s s), E/B
// The E part of the linear tidal field is proportional to delta in
// Fourier space, so rows 2+3 and 6+7 also give the s x (delta s) and
// s x (s s) terms of II.
//
// Limber associates k=(ell+1/2)/f_K with a radial shell; f_K = chi in the
// flat geometry the radial builder requires. Compute the ten one-loop
// kernels once per (ell,shell), then reuse them for all pairs. By default
// the core fills its table by cubic-spline upsampling from a coarser
// convolution grid (pt_cfastpt.c, fpt_regrid); the hot lookup here is
// direct-index linear interpolation in ln k on that table, as in
// cosmo2D.c. Its z=0 kernels evolve with D^4, because each loop contains
// two linear powers. The growth growfac(a), with D(1)=1, is the one used
// for the amplitudes and by the data-vector TATT path. Outside the table's
// k support the one-loop terms are zero, as in the core; the NLA matter
// power remains.
//
// Parameters:
//   radial - snapshot from radial_inputs_cov: lens windows, source density
//            window[0], lensing window[1], f_K and dchi weights
//   nell   - number of multipoles, at least 1
//   ell    - multipoles >= 1, the grid limber_spectra_cov already checked
//   ee     - [npair][nell] E spectra from limber_spectra_cov on the same
//            snapshot with include_ia = 1; the corrections are added in
//            place
//   bb     - [npair][nell] B spectra, overwritten; zero for every pair
//            with a lens field
//
// The lens leg is density plus magnification with linear bias. RSD is not
// included, and both C++ entries reject TATT with RSD before calling this
// function. Call serially: FAST-PT is prepared before the OpenMP loops,
// and each (ell, pair) output belongs to one worker, with no cross-thread
// sum. No SSC/cNG response or halo approximation is changed by this
// function.
// ---------------------------------------------------------------------------
void tatt_spectra_cov(
    const struct radial_cov* radial, // common rule and NLA window snapshot
    const int nell,                 // multipole samples
    const double* ell,              // multipoles >= 1
    double* const* ee,              // NLA E input, TATT E output
    double* const* bb               // TATT B output
  )
{
  // Only TATT has terms beyond NLA; reject any other model, and an empty
  // multipole grid.
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

  // amplitude[0/1/2][source][node] = C1, C2 and b_TA of each source catalog
  // at each radial node (dimensionless). loop[index][node][role] = D^4
  // times one-loop row number role at that node's k = (ell+1/2)/f_K, in
  // (c/H0)^3. Stage 1 fills both; stage 2 reads them.
  double*** amplitude = (double***) malloc3d(3, nsource, nnode);
  double*** loop = (double***) malloc3d(nell, nnode, 10);

  // Store each unordered field pair once. Symmetric spectra need the
  // same physical integral for (A,B) and (B,A), including unmeasured pairs.
  // The i-major order matches the ee and bb rows of limber_spectra_cov.
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
  // get_FPT_IA computes the C-FAST-PT table, or reuses it while cosmology
  // and settings are unchanged. Unlike cosmo2D.c, this call does not test
  // nuisance.IA_code: a table installed from Python (IA_code = 1) is
  // replaced by the C-FAST-PT one.
  get_FPT_IA();

  // The table samples ln k uniformly: sample i (0 <= i < N) sits at
  // lnk_min + i*(lnk_max-lnk_min)/N, so lnk_max lies one step beyond the
  // last sample. inv_step converts a distance in ln k into table steps.
  const double lnk_min = log(FPTIA.krange[RANGE_MIN]);
  const double lnk_max = log(FPTIA.krange[RANGE_MAX]);
  const double inv_step = FPTIA.N/(lnk_max-lnk_min);

  // One iteration is one radial node (shell). It evaluates every source's
  // TATT amplitudes there and interpolates the ten one-loop rows at each
  // multipole's Limber wavenumber. Neither depends on the catalog pair, so
  // the pair loop of stage 2 reuses them instead of repeating table reads.
  // Each worker writes only its own node's entries.
  #pragma omp parallel for schedule(static)
  for (int node=0; node<nnode; node++) {
    const double a = radial->geometry[0][node];

    // growfac(a) is D(a) with D(1)=1. The table was computed from
    // p_lin(k, a=1); each one-loop term is quadratic in the linear power,
    // so it scales as D^4.
    const double growth = growfac(a);
    const double growth4 = growth*growth*growth*growth;

    // C1, C2 and b_TA of every source catalog at this shell (IA.c).
    for (int source=0; source<nsource; source++) {
      amplitude[0][source][node] = IA_A1_Z1(a, growth, source);
      amplitude[1][source][node] = IA_A2_Z1(a, growth, source);
      amplitude[2][source][node] = IA_BTA_Z1(a, growth, source);
    }

    // For each multipole, interpolate the ten rows at k = (ell+1/2)/f_K.
    for (int index=0; index<nell; index++) {
      const double lnk = log((ell[index]+0.5)/radial->geometry[2][node]);

      // Match the core's finite FAST-PT support: outside [lnk_min, lnk_max]
      // no one-loop term is extrapolated, and all ten entries stay zero.
      // The NLA matter power in ee is unaffected.
      for (int role=0; role<10; role++) {
        loop[index][node][role] = 0.0;
      }
      if (lnk >= lnk_min
          && lnk <= lnk_max) {
        // position counts table steps from lnk_min; its integer part is
        // the left sample and its fraction the interpolation weight.
        const double position = (lnk-lnk_min)*inv_step;
        int left = (int) position;
        double fraction = position-left;

        // The core convention places its final support edge one cell
        // beyond the table's samples. For lnk in that last cell, from the
        // final sample up to lnk_max, left+1 would step past the table;
        // like cosmo2D.c's hold-last-node clamp, use left = N-2 with zero
        // fraction there, which reads the penultimate sample. Retain this
        // explicit convention for model agreement.
        if (left+1 >= FPTIA.N) {
          left = FPTIA.N-2;
          fraction = 0.0;
        }

        // Linear interpolation in ln k, times D^4. The core's LERP writes
        // the same straight line as values[left] + fraction*(values[left+1]
        // - values[left]), so the two agree up to rounding.
        for (int role=0; role<10; role++) {
          const double* values = FPTIA.tab[role];
          loop[index][node][role] = growth4
              *((1.0-fraction)*values[left]+fraction*values[left+1]);
        }
      }
    }
  }

  // --- 2. PROJECT gI, GI/IG AND II CORRECTIONS FOR EVERY FIELD PAIR ---

  // PHYSICAL DERIVATION & LOGIC FLOW
  //   1. Observed fields: a lens is b1 delta plus magnification; a source
  //      E mode is the lensing shear G plus the alignment I of the header;
  //      a source B mode is I alone, since lensing makes no B mode here.
  //   2. At one shell a Limber spectrum adds W_X W_Y P_XY dchi/f_K^2. Lens
  //      and lensing windows respond to delta; the alignment of a source
  //      enters through its density window n_s and its amplitudes. With
  //      gi = C1 b_TA (rows 2+3) - 5 C2 (rows 6+7), the delta-I power
  //      beyond NLA is -gi.
  //   3. Corrections beyond NLA at one shell, before dchi/f_K^2:
  //        lens-source    dE = -spin W_g n_s2 gi_2
  //        source-source  dE = spin^2 (-n_s1 W_kappa2 gi_1
  //                                    -n_s2 W_kappa1 gi_2 + n_s1 n_s2 ii_E)
  //                       dB = spin^2 n_s1 n_s2 ii_B
  //      with W_g = b1 n_l H/H0 + [ell(ell+1)/(ell+1/2)^2] b_mag W_mag and
  //      spin = sqrt((ell-1) ell (ell+1) (ell+2))/(ell+1/2)^2, one factor
  //      per source leg.
  //   4. Sum over shells, add E to the NLA input and store B. Lens-lens
  //      pairs receive nothing.
  //
  // Retain all crosses, even if a data-vector pair map excludes them.
  // collapse(2) lets OpenMP divide the combined (ell, pair) iterations
  // among workers; each iteration owns one complete output pair at one
  // ell. SIMD lanes keep E and B integrals separate while sharing their
  // shell weight; neither parity channel can leak into the other.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int index=0; index<nell; index++) {
    for (int pair=0; pair<npair; pair++) {
      // first <= second are the two fields of this output row; indices
      // below nlens are lenses.
      const int first = pairs[0][pair];
      const int second = pairs[1][pair];

      // B starts from zero for every pair. A lens-lens pair has no source
      // leg, so its E row stays the NLA input and its B row stays zero.
      bb[pair][index] = 0.0;
      if (second < nlens) {
        continue;
      }

      // Angular factors: shift2 = (ell+1/2)^2 and the spin-2 factor of one
      // source leg (ell >= 1 keeps the square root real).
      const double l = ell[index];
      const double shift2 = (l+0.5)*(l+0.5);
      const double spin = sqrt((l-1)*l*(l+1)*(l+2))/shift2;
      const int source2 = second-nlens;

      // scalar: for this pair, starting from E = B = 0,
      //   for (int node=0; node<nnode; node++) {
      //     ... delta_e and delta_b of the node, as computed below ...
      //     E = fma(measure, delta_e, E);
      //     B = fma(measure, delta_b, B);
      //   }
      //   ee[pair][index] += E;
      //   bb[pair][index] = B;
      // measure = dchi/f_K^2 = geometry[3][node]/geometry[2][node]^2. The
      // vector code keeps E in lane 0 and B in lane 1; it never adds them.

      // integral = [E, B] = [0, 0] (setzero sets both lanes to 0.0).
      v2d integral = simde_mm_setzero_pd();

      // Each shell contributes its local shape correlations, weighted by
      // the number of sources there and by lensing geometry (step 3 of the
      // derivation). The two SIMD sums integrate E and B over exactly the
      // same shells, in increasing node order.
      for (int node=0; node<nnode; node++) {
        const double* power = loop[index][node];

        // FAST-PT rows follow the shared core convention:
        //   0/1: quadratic tidal auto power, E/B;
        //   2+3: density with density-weighted alignment;
        //   4/5: density-weighted alignment auto power, E/B;
        //   6+7: density with the quadratic tidal field;
        //   8/9: weighted alignment with the quadratic tidal field, E/B.
        // These rows contain D^4 but no source-bin amplitudes yet.
        // ta_delta and mix_delta add the two E rows of each cross power.
        const double ta_delta = power[2]+power[3];
        const double mix_delta = power[6]+power[7];

        // The second field is always a source here. c12, c22 and b2 are its
        // C1, C2 and b_TA (b2 is not a galaxy bias); n2 is its density
        // window n_s2; gi2 = C1 b_TA (rows 2+3) - 5 C2 (rows 6+7) is its
        // delta-I power beyond NLA, without the minus sign.
        const double c12 = amplitude[0][source2][node];
        const double c22 = amplitude[1][source2][node];
        const double b2 = amplitude[2][source2][node];
        const double n2 = radial->window[0][second][node];
        const double gi2 = c12*b2*ta_delta-5.0*c22*mix_delta;

        // Shell weight dchi/f_K^2: geometry[2] holds f_K and geometry[3]
        // the positive dchi weight.
        const double distance = radial->geometry[2][node];
        const double measure = radial->geometry[3][node]/(distance*distance);
        double delta_e;
        double delta_b = 0.0;

        if (first < nlens) {
          // gI correlates the lens density/magnification with the source
          // alignment. galaxy is W_g of step 3, with the magnification
          // factor ell(ell+1)/(ell+1/2)^2. The minus sign follows the
          // positive-C1 convention; one source leg gives one spin factor.
          const double galaxy = radial->window[0][first][node]
              +l*(l+1)/shift2*radial->window[1][first][node];
          delta_e = -spin*galaxy*n2*gi2;
        } else {
          // Source-source: c11, c21 and b1 are C1, C2 and b_TA of the first
          // source (b1 is not a galaxy bias); n1 is its density window,
          // lens1 and lens2 are the two lensing windows W_kappa, and gi1 is
          // the first source's delta-I power beyond NLA.
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
          // excludes C11*C12*P, already present in the NLA input. The two
          // -[...] brackets of I multiply to a plus sign; -5 and 25 carry
          // the quadratic-field normalization.
          const double ii_e = c11*c12
              *(b1*b2*power[4]+(b1+b2)*ta_delta)
              -5.0*(c11*c22+c12*c21)*mix_delta
              -5.0*(c11*b1*c22+c12*b2*c21)*power[8]
              +25.0*c21*c22*power[0];
          // B has no linear term: the linear tidal field has no B mode.
          const double ii_b = c11*c12*b1*b2*power[5]
              -5.0*(c11*b1*c22+c12*b2*c21)*power[9]
              +25.0*c21*c22*power[1];

          // GI and IG each pair one source's lensing with the other's
          // alignment, with the minus sign of the positive-C1 convention;
          // two source legs give spin^2.
          delta_e = spin*spin
              *(-n1*lens2*gi1-n2*lens1*gi2+n1*n2*ii_e);
          delta_b = spin*spin*n1*n2*ii_b;
        }

        // values = [delta_e, delta_b]: set_pd takes lane 1 first and lane 0
        // last. These are different parity channels, not bins.
        const v2d values = simde_mm_set_pd(delta_b, delta_e);

        // weight = [measure, measure]: set1 copies the common dchi/f_K^2
        // into both lanes.
        const v2d weight = simde_mm_set1_pd(measure);

        // scalar: E = fma(measure, delta_e, E); B = fma(measure, delta_b, B).
        // fmadd computes weight*values + integral in each lane, appending
        // one weighted shell to each channel independently, with one
        // rounding per lane where the processor has a fused instruction.
        integral = simde_mm_fmadd_pd(weight, values, integral);
      }

      double values[2];

      // values[0..1] = [E, B]: storeu writes lane 0 to values[0] and lane 1
      // to values[1]. This ordinary two-double array needs no special
      // vector alignment; it is a different variable from the vector
      // values inside the node loop, whose scope ended with that loop.
      simde_mm_storeu_pd(values, integral);

      // scalar: ee[pair][index] += E; bb[pair][index] = B.
      ee[pair][index] += values[0];
      bb[pair][index] = values[1];
    }
  }

  // Release the scratch: one free per grouped array.
  free(pairs);
  free(loop);
  free(amplitude);
}
