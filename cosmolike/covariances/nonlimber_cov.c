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

// SIMD applies one operation to several numbers at once; each number sits
// in a vector position called a lane. A v2d holds two doubles, lanes 0
// and 1 (one SSE2 register on x86-64, one NEON register on arm64). SIMDe
// translates the x86-named simde_mm_* calls to either instruction set.
typedef simde__m128d v2d;

// -----------------------------------------------------------------------
// Project one common separable linear field without the Limber shortcut.
//
// Write delta(k,a)=D(a)*delta(k,1), with D(1)=1 (growfac). Every catalog
// then has a transfer F, and its cross spectrum is
//   C_AB(l) = (2/pi) integral dlnk k^3 P(k,1) F_A(l,k) F_B(l,k).
// A density transfer integrates chi*W_density*D against j_l(k*chi) in
// dlnchi, i.e. integral dchi W_density D j_l(k*chi).
// A lensing/IA transfer integrates chi*(W_lensing+W_IA)*D against
// j_l(k*chi)/(k*chi)^2 and receives an angular derivative factor.
// For galaxy magnification this factor is l(l+1); for shear it is
// sqrt((l-1)l(l+1)(l+2)). Windows include all bias and IA amplitudes.
//
// The caller forms the hybrid of Fang et al. (arXiv:1911.11947, Sec. 2.2):
//   nonlinear Limber + exact separable linear - matched linear Limber.
// Both linear terms must use P(k,1) and the same D(a). Using the supplied
// p_lin(k,a) only in the subtraction would spoil cancellation whenever
// those tables have scale-dependent growth. This prescription does not
// implement general unequal-time massive-neutrino evolution.
//
// PHYSICAL DERIVATION & LOGIC FLOW
// 1. A projected field is A(n) = integral dchi W_A(chi) delta(chi n, a).
//    The plane-wave expansion
//      exp(i k.x) = 4 pi sum_lm i^l j_l(k chi) Y*_lm(k/|k|) Y_lm(n)
//    gives its multipoles
//      a_lm = 4 pi i^l integral d^3k/(2 pi)^3 delta(k,1) Y*_lm(k/|k|)
//             F_A(l,k),    F_A(l,k) = integral dchi W_A D j_l(k chi).
// 2. Statistical isotropy, <delta(k,1) delta*(k',1)> = (2 pi)^3
//    delta_D(k-k') P(k,1), and the orthonormal Y_lm leave the factor
//    (4 pi)^2/(2 pi)^3 = 2/pi of C_AB above.
// 3. Magnification, shear and the NLA tidal field are angular second
//    derivatives of the potential. Poisson's 1/k^2, paired with the chi^2
//    of the projection geometry, gives the kernel j_l(k chi)/(k chi)^2;
//    the derivatives on the sphere give l(l+1) (magnification) or
//    sqrt((l+2)!/(l-2)!) (spin-2 E mode), applied after the transform.
// 4. Limber limit: j_l(x) ~ sqrt(pi/(2l+1)) delta_D(l+1/2-x) reduces C_AB
//    to integral dchi W_A W_B D^2 P((l+1/2)/chi,1)/chi^2, each lensing
//    leg keeping its angular factor of item 3 times 1/(l+1/2)^2.
//    limber_spectra_cov(linear=2) is that matched term, so exact-matched
//    tends to zero at high l and keeps the low-l correction to Limber of
//    the linear field.
// Code map:
//    stage 1  matched = limber_spectra_cov(..., 2, 0, ...); radial inputs
//             chi*W*D on the ln(chi) grid; forward FFTs saved once
//    stage 2  per block of 16 multipoles: component transfers
//             (fftlog_execute_cov), the k weights (2/pi) dlnk k^3 P(k,1)
//             and the observed-field transfers with their angular factors
//    stage 3  exact[p][l-2] = sum_k weight F_A F_B for every pair p
//    caller   spectra += exact - matched (apply_nonlimber_cov)
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
//
// Parameters:
//   radial     - quadrature snapshot from radial_inputs_cov; supplies the
//                matched Limber term and the lens/source counts
//   amin       - far boundary of the ln(chi) grid: the far edge a_edges[0]
//                used to build radial, enclosing both catalogs
//   nwindow    - uniform-a efficiency samples, as used for radial
//   include_ia - 1 adds the signed NLA window to the source fields, as
//                used for radial
//   lmax       - last integer multipole; outputs cover l = 2..lmax
//   nchi       - ln(chi) samples including both ends, 2^n+1 >= 65
//   chi_min    - near boundary in c/H0; the exact term omits closer
//                shells, so lower it to test that foreground
//   exact, matched - caller-owned rows [npair][lmax-1], overwritten, in
//                the i-major triangular pair order of spectra_cov.h
// Every temporary (ln(chi) snapshot, FFTLog workspace, transfers and k
// rows) is created and released here. Call outside OpenMP regions: the
// loops start their own teams. Each output is summed by one worker in a
// fixed order, so the result does not depend on the thread count.
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
  // Supported model only; anything else stops instead of being
  // approximated. Flat geometry: f_K(chi) = chi, and j_l(k chi) are the
  // radial modes of flat space (curved space needs hyperspherical
  // Bessel functions). Massless neutrinos: massive ones make the linear
  // growth scale dependent, which D(a) delta(k,1) cannot represent.
  // intervals = nchi-1 a power of two, at least 64: doubling it keeps
  // the old nodes, the FFT length 4*intervals has only the factor 2, and
  // the reciprocal extension intervals/4 is a whole number of samples.
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
  // nfield observed fields, lenses then sources; ncomponent radial
  // inputs, one density per lens plus one spin input per field; npair
  // unordered field pairs, crosses included.
  const int nfield = radial->nlens+radial->nsource;
  const int ncomponent = radial->nlens+nfield;
  const int npair = nfield*(nfield+1)/2;

  // matched[p][l-2]: the Limber projection at l = 2..lmax of the same
  // separable field, power mode 2 = D(a)^2 p_lin(k,1), without RSD (0).
  double* multipoles = (double*) malloc1d(lmax-1);
  for (int index=0; index<lmax-1; index++) {
    multipoles[index] = index+2.0;
  }
  limber_spectra_cov(radial, lmax-1, multipoles, 2, 0, matched);
  free(multipoles);

  // The same windows at chi_i = chi_min*exp(i*dlnchi), i = 0..nchi-1, up
  // to chi(amin); dlnchi is the step radial_logchi_cov used for them.
  // inputs[component][node] receives chi*W*D and powers[component] the
  // kernel power, 0 (density) or 2 (spin).
  struct radial_cov* logarithmic = radial_logchi_cov(
      amin, chi_min, nchi, nwindow, include_ia);
  const double dlnchi = log(chi(amin)/chi_min)/intervals;
  double** inputs = (double**) malloc2d(ncomponent, nchi);
  int* powers = (int*) malloc1d_int(ncomponent);

  // Touch the linear power reader serially, before workers read it in
  // the block loop below. The covariance code builds any table a shared
  // reader might create on first use outside OpenMP; the current p_lin
  // only reads the tables installed by the cosmology interface, so this
  // call is a safeguard rather than a required initialization.
  (void) p_lin(1.0, 1.0);

  // Components are lens densities first, then one spin kernel per field.
  // A spin kernel means magnification for a lens and lensing+IA for a
  // source. Multiplying by chi converts dchi to FFTLog's dlnchi measure.
  // D(a) carries the time dependence of the separable field, so the k
  // integral later needs only the a=1 power. Component c < nlens is the
  // density of lens c; component nlens+f is the spin input of field f.
  // One iteration fills one component's radial input; components are
  // independent.
  #pragma omp parallel for schedule(static)
  for (int component=0; component<ncomponent; component++) {
    const int density = component < radial->nlens;
    const int field = density ? component : component-radial->nlens;
    powers[component] = density ? 0 : 2;
    for (int node=0; node<nchi; node++) {
      const double a = logarithmic->geometry[0][node];
      const double distance = chi_min*exp(node*dlnchi);

      // window[0]: b1 times the lens distribution per unit distance.
      // window[1]+window[2]: the lensing window (times b_mag for a lens)
      // plus the signed NLA term, which is zero for lenses.
      double window = logarithmic->window[0][field][node];
      if (!density) {
        window = logarithmic->window[1][field][node]
                 +logarithmic->window[2][field][node];
      }
      inputs[component][node] = distance*window*growfac(a);
    }
  }
  free_radial_cov(logarithmic);

  // The zero guard below chi_min is as wide in ln(chi) as the physical
  // interval (padding = intervals samples). The FFT length 4*intervals
  // leaves 2*intervals-1 zero samples above chi_max, so each guard spans
  // at least that width. Reading extra = intervals/4 samples, a quarter
  // of the physical log width, at each reciprocal end retains low-k power
  // that a grid beginning at (ell+1)/chi_max would otherwise miss, and
  // part of the high-k tail beyond (ell+1)/chi_min.
  // These physical widths stay fixed when the interval count doubles,
  // and so does the Fourier period 4*intervals*dlnchi =
  // 4 ln(chi_max/chi_min): refinement adds nodes without moving the old
  // ones.
  const int padding = intervals;
  const int extra = intervals/4;
  const int nk = nchi+2*extra;

  // The workspace copies inputs and saves their forward transforms, so
  // the radial inputs can be released at once.
  struct fftlog_workspace_cov* work = fftlog_create_cov(
      nchi, padding, 4*intervals, extra, chi_min, dlnchi,
      ncomponent, powers, (const double* const*) inputs);
  free(inputs);
  free(powers);

  // --- 2. TRANSFORM BOUNDED MULTIPOLE BLOCKS AND COMBINE FIELD PARTS ---

  // A block supplies enough independent transforms for eight workers
  // without retaining an ell-by-field-by-k cube for the entire survey.
  // Geometry and matter power are shared by every pair at this ell.
  // Work arrays for one block of at most 16 multipoles, index = l-first:
  //   transfer[index][component][node] - F_l(k) of each radial component;
  //       after the combination step, rows 0..nfield-1 hold the complete
  //       transfers of the observed fields
  //   waves[0][index][node] - the reciprocal grid k of that multipole
  //   waves[1][index][node] - the k weight (2/pi) dlnk k^3 P(k,1)
  //   pairs[0][p], pairs[1][p] - the two fields of pair p
  double*** transfer = (double***) malloc3d(16, ncomponent, nk);
  double*** waves = (double***) malloc3d(2, 16, nk);
  int** pairs = (int**) malloc2d_int(2, npair);

  // Unordered pairs left <= right in i-major triangular order,
  // (0,0),(0,1),...,(1,1),..., the row order of exact and matched.
  int pair = 0;
  for (int left=0; left<nfield; left++) {
    for (int right=left; right<nfield; right++) {
      pairs[0][pair] = left;
      pairs[1][pair] = right;
      pair++;
    }
  }

  // Each block covers up to 16 consecutive multipoles, the capacity of
  // the workspace's Mellin-kernel table. Per block: transform every
  // component (all share one k grid per multipole), attach the k weights,
  // assemble the observed-field transfers, then sum every pair. The
  // forward FFTs of stage 1 serve every block; count is 16 except
  // possibly in the last block.
  for (int first=2; first<=lmax; first+=16) {
    const int count = lmax-first+1 < 16 ? lmax-first+1 : 16;
    fftlog_execute_cov(work, first, count, waves[0], transfer);

    // Each worker completes an ell. Combining components in increasing
    // field order safely reuses the first nfield transfer rows: every
    // overwritten component has already been consumed at that point.
    // One iteration first forms the k weights of this multipole, then
    // assembles its observed-field transfers. Multipoles are independent.
    #pragma omp parallel for schedule(static)
    for (int index=0; index<count; index++) {
      const double ell = first+index;

      // The pair integral (2/pi) integral dlnk k^3 P(k,1) F_A F_B uses the
      // reciprocal nodes themselves. Their log step equals dlnchi, so the
      // trapezoid rule weights node q by dlnchi, halved at the two ends
      // (the area under a straight line between neighbouring samples).
      // waves[1] holds P(k,1) times that complete weight, shared by every
      // pair at this multipole. P is the a=1 anchor of the separable
      // field; the growth D(a) already sits in the transfers.
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

      // Combine the parts of each observed field before pairing fields:
      //   lens f:   F = F_density + l(l+1) F_magnification,
      //   source f: F = sqrt((l-1)l(l+1)(l+2)) F_lensing+IA,
      // so density-magnification and lensing-IA cross terms are kept.
      // Field f's result overwrites row f. Row f < nlens is lens f's own
      // density, read and replaced node by node. Row f >= nlens is the
      // spin input of field f-nlens, already used at iteration
      // f-nlens < f. The spin input read now, row nlens+f > f, has not
      // been overwritten yet.
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

    // The exact spectrum of pair (A,B) is the k integral of its two
    // transfers, C = sum_k measure[k]*F_A[k]*F_B[k], where measure[k] =
    // (2/pi) dlnk k^3 P(k,1) is the same for every pair. Since it is
    // non-negative and common, the spectra at one l form F W F^T, a Gram
    // matrix: positive semidefinite, crossed bins included. One task is
    // one multipole and two consecutive pairs. Two SIMD lanes accumulate
    // different pairs, in the same k order: lane 0 carries pair and lane 1
    // next, and the lanes are never added together. Collapsing ell and
    // pair tasks keeps small-bin surveys parallel too.
    //
    // scalar, for one pair p with fields A = pairs[0][p], B = pairs[1][p]:
    //   double sum = 0.0;
    //   for (int node=0; node<nk; node++) {
    //     const double product = transfer[index][A][node]
    //                            *transfer[index][B][node];
    //     sum = fma(waves[1][index][node], product, sum);
    //   }
    //   exact[p][first+index-2] = sum;
    // Lane 0 computes this for p = pair and lane 1 for p = next.
    #pragma omp parallel for collapse(2) schedule(static)
    for (int index=0; index<count; index++) {
      for (int pair=0; pair<npair; pair+=2) {
        // Two pairs per task. For an odd npair the final task repeats
        // pair in lane 1 (next = pair); that duplicate is discarded below.
        const int next = pair+1 < npair ? pair+1 : pair;

        // setzero starts both independent spectrum sums at zero:
        // lane 0 for pair, lane 1 for next.
        v2d total = simde_mm_setzero_pd();

        // Each step adds the trapezoid term of one k node to both sums.
        for (int node=0; node<nk; node++) {
          // set_pd takes the high lane first. Lane 0 gets pair's left
          // transfer; lane 1 gets next's left transfer at the same k.
          // These are F_A(k) of the two pairs, from two transfer rows.
          const v2d left = simde_mm_set_pd(
              transfer[index][pairs[0][next]][node],
              transfer[index][pairs[0][pair]][node]);

          // Keep the right transfer in the same pair order as the left:
          // set_pd again puts its last argument, F_B(k) of pair, in lane
          // 0 and F_B(k) of next in lane 1.
          const v2d right = simde_mm_set_pd(
              transfer[index][pairs[1][next]][node],
              transfer[index][pairs[1][pair]][node]);

          // set1 copies the shared dlnk*k^3*P*2/pi measure to both lanes:
          // waves[1][index][node], trapezoid weight included.
          const v2d weight = simde_mm_set1_pd(waves[1][index][node]);

          // mul forms F_A*F_B independently for the two catalog pairs,
          // each product rounded once; no product crosses lanes.
          const v2d product = simde_mm_mul_pd(left, right);

          // fmadd accumulates each weighted product, without mixing pairs:
          // total = weight*product + total in each lane, the scalar
          // sum = fma(weight, product, sum). It is fused (one rounding)
          // where FMA is native: arm64 NEON, or x86 built with FMA.
          // SIMDe's fallback without FMA rounds the product and the sum
          // separately.
          total = simde_mm_fmadd_pd(weight, product, total);
        }

        double values[2];

        // storeu copies the lane sums to an ordinary, unaligned array:
        // values[0] = lane 0 (pair), values[1] = lane 1 (next).
        simde_mm_storeu_pd(values, total);

        // Store pair's spectrum at l = first+index, row offset l-2.
        // Store next's only when it is a real pair, discarding the
        // duplicate lane of an odd final task.
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


// -----------------------------------------------------------------------
// Turn nonlinear Limber spectra into the hybrid spectra, in place:
//
//   C_hybrid = C_Limber,nonlin + C_exact,lin - C_Limber,matched-lin.
//
// The two linear terms project the same separable field, so where Limber
// is accurate (high l) they cancel and C_hybrid -> C_Limber,nonlin. At
// low l the exact term replaces the Limber projection of the linear
// part, while the nonlinear excess over linear power stays Limber.
//
// Keep the hybrid addition shared between production and notebook APIs.
// Its inputs have already been validated by their public boundary:
// integer ell inside [2,lmax], nchi = 2^n+1 >= 65, chi_min > 0, no RSD
// and massless neutrinos. Parameters as in nonlimber_spectra_cov, plus
// the caller's nell multipoles ell[nell] and spectra[npair][nell], the
// nonlinear Limber input that becomes the hybrid output for gg and gs.
// -----------------------------------------------------------------------
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
  // correction[0][pair][l-2] = exact and correction[1][pair][l-2] =
  // matched, for l = 2..lmax, in the same pair order as spectra.
  const int nfield = radial->nlens+radial->nsource;
  const int npair = nfield*(nfield+1)/2;
  double*** correction = (double***) malloc3d(2, npair, lmax-1);
  nonlimber_spectra_cov(radial, amin, nwindow, include_ia, lmax, nchi,
      chi_min, correction[0], correction[1]);

  // Only a pair with at least one galaxy (lens) field receives the
  // correction. Retain the caller's shear-shear spectra, including IA.
  // Lens fields come first, so a pair holds a lens exactly when its first
  // field is a lens. The counter pair walks the triangular order of the
  // spectra rows. For a requested ell inside [2,lmax], an integer here,
  // offset = ell-2 selects the correction at that multipole.
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
