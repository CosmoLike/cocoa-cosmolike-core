// <complex.h> precedes <fftw3.h> on purpose: FFTW then defines
// fftw_complex as the C99 double complex type, so the Fourier
// coefficients below are multiplied, divided and conjugated with
// ordinary complex arithmetic (*=, conj) instead of re/im array pairs.
#include <complex.h>
#include <math.h>
#include <stdlib.h>
#include <fftw3.h>
#include <gsl/gsl_sf_gamma.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "fftlog_cov.h"
#include "cosmolike/basics.h"
#include "log.c/src/log.h"
#include "simde/x86/sse2.h"

// SIMD applies one operation to several numbers at once; each number sits
// in a vector position called a lane. A v2d holds two doubles, lanes 0
// and 1 (one SSE2 register on x86-64, one NEON register on arm64). SIMDe
// translates the x86-named simde_mm_* calls to either instruction set.
typedef simde__m128d v2d;

// A recipe is an FFTW plan: FFTW's precomputed sequence of steps for one
// transform length and memory layout. Executing an existing plan on new
// arrays is thread safe; creating or destroying one is not.
// The caller owns one recipe and one set of radial Fourier coefficients.
// Workers borrow that recipe, but only write their own scratch rows,
// indexed by OpenMP thread number. A real input of length nfft has
// nfreq = nfft/2+1 independent Fourier coefficients, frequencies
// 0..nfft/2; the negative frequencies are their complex conjugates.
struct fftlog_workspace_cov {
  int nchi;                    // number of physical distance samples
  int padding;                 // zero guard before the physical interval
  int nfft;                    // full, padded transform length
  int extra;                   // extra reciprocal-grid samples per end
  int nfield;                  // number of radial functions
  int nthreads;                // allocated worker count
  int* inverse_power;          // [field], density 0 or lensing 2
  double chi_min;              // first physical distance
  double dlnchi;               // uniform logarithmic spacing
  double** real;               // [thread][nfft], input or inverse output
  fftw_complex** scratch;      // [thread][nfft/2+1], inverse coefficients
  fftw_complex** forward;      // [field][nfft/2+1], reusable forward FFT
  fftw_complex** kernel;       // [ell in block][frequency], density Mellin
  fftw_plan inverse;           // single shared complex-to-real recipe
};

// -----------------------------------------------------------------------
// Prepare the ell-independent part of a spherical-Bessel projection.
//
// A projected field contains integral dlnchi f(chi) j_l(k chi). FFTLog
// represents f as complex powers of chi, whose Bessel integrals are
// known analytically. The powers come from an ordinary discrete Fourier
// transform in ln(chi): on a uniform ln(chi) grid the Fourier mode
// exp(i eta ln chi) is the power chi^(i eta).
//
// We use bias nu=1: Fourier transform f(chi)/chi, so each mode of f is
// chi^(1+i eta), then restore the removed power through the Mellin kernel
// and 1/k (fftlog_execute_cov). The Mellin integrals converge for
// -l < Re(s) < 2 (density) and 2-l < Re(s) < 4 (lensing, j_l(t)/t^2);
// Re(s) = nu = 1 lies inside both for every l >= 2.
// This is the single-Bessel construction of Fang et al. (1911.11947).
//
// Array layout: real[m] = f(chi_m)/chi_m, with
//   chi_m = chi_min*exp((m-padding)*dlnchi),  m = 0..nfft-1.
// The physical interval occupies indices padding..padding+nchi-1.
// Everything else is zero: those empty samples separate periodic copies
// of the input, rather than extrapolating the survey to other distances.
// A frequency taper suppresses the highest quarter of Fourier modes to
// reduce ringing at interval edges. Radial support and taper accuracy
// must still be tested against direct integrals for each survey.
//
// Only two plans are made, serially, because FFTW's planner is not thread
// safe. FFTW_ESTIMATE picks each recipe by heuristics, without trial
// transforms, so planning neither overwrites the arrays nor depends on
// timing measurements. Every field uses the same forward recipe with new
// arrays; it is destroyed after the coefficients are saved. The inverse
// recipe survives for every later multipole block.
// Dimensions and alignment are identical for the worker scratch rows:
// new-array execution needs the planned length, the same out-of-place
// layout and the same alignment, and the grouped allocations give every
// row the 64-byte alignment of row 0, on which the plans were made.
//
// Parameters (see fftlog_cov.h for the grid contract):
//   nchi, chi_min, dlnchi - physical samples chi_min*exp(i*dlnchi)
//   padding, nfft         - zero guard below chi_min; padded FFT length
//   extra                 - reciprocal samples read beyond each end
//   nfield, inverse_power - number of inputs; kernel power 0 or 2 each
//   radial                - [field][nchi] values f = chi*W, copied
// Returns a heap-owned workspace; release it with fftlog_free_cov.
// -----------------------------------------------------------------------
struct fftlog_workspace_cov* fftlog_create_cov(
    const int nchi,                  // radial samples before padding
    const int padding,               // lower zero guard
    const int nfft,                  // padded transform length
    const int extra,                 // output extension in log k
    const double chi_min,            // positive minimum distance
    const double dlnchi,             // positive log-distance interval
    const int nfield,                // independent radial inputs
    const int* inverse_power,        // kernel choice for each input
    const double* const* radial      // [field][nchi] signed radial values
  )
{
  if (nchi < 3
      || padding < 1
      || nfft < 8
      || nfft < nchi+2*padding
      || nfft%2 != 0
      || extra < 0
      || extra > padding
      || nfield < 1
      || !isfinite(chi_min)
      || chi_min <= 0.0
      || !isfinite(dlnchi)
      || dlnchi <= 0.0) {
    log_fatal("fftlog_create_cov: invalid radial grid or FFT shape");
    exit(1);
  }

  // FFTW splits these lengths into short transforms of 2, 3, 5 or 7
  // samples. A length with a large prime factor needs a different, more
  // expensive decomposition. The caller rounds its base size once and
  // doubles it with refinement so the Fourier period remains fixed.
  // The factor of a short transform is called its radix: a length
  // N = 2^a 3^b 5^c 7^d is a chain of radix-2, 3, 5 and 7 steps, each
  // done by a small, hard-coded FFTW kernel. The loop below divides out
  // each allowed factor as often as it divides; any remainder above 1 is
  // a larger prime factor, which is rejected.
  int remaining = nfft;
  const int factors[4] = {2, 3, 5, 7};
  for (int factor=0; factor<4; factor++) {
    while (remaining%factors[factor] == 0) {
      remaining /= factors[factor];
    }
  }
  if (remaining != 1) {
    log_fatal("fftlog_create_cov: FFT length needs only 2/3/5/7 factors");
    exit(1);
  }

  struct fftlog_workspace_cov* work = malloc(sizeof(*work));
  if (work == NULL) {
    log_fatal("fftlog_create_cov: workspace allocation failed");
    exit(1);
  }
  work->nchi = nchi;
  work->padding = padding;
  work->nfft = nfft;
  work->extra = extra;
  work->nfield = nfield;
  work->chi_min = chi_min;
  work->dlnchi = dlnchi;

  // Worker rows are indexed by omp_get_thread_num(), so allocate one for
  // every thread of the largest team a later parallel region can start.
  work->nthreads = 1;
#ifdef _OPENMP
  work->nthreads = omp_get_max_threads();
#endif

  // Group arrays by role: a real row and a complex scratch row for each
  // worker, the nfreq saved forward coefficients of every field, and the
  // Mellin kernels of at most 16 multipoles, one block of the caller.
  const int nfreq = nfft/2+1;
  work->inverse_power = (int*) malloc1d_int(nfield);
  work->real = (double**) malloc2d(work->nthreads, nfft);
  work->scratch = (fftw_complex**)
      malloc2d_fftwc(work->nthreads, nfreq);
  work->forward = (fftw_complex**) malloc2d_fftwc(nfield, nfreq);
  work->kernel = (fftw_complex**) malloc2d_fftwc(16, nfreq);

  // Plan serially on worker row 0. The real-to-complex (r2c) recipe maps
  // nfft doubles to nfreq coefficients; the complex-to-real (c2r) recipe
  // maps them back. FFTW lets a c2r transform overwrite its complex input,
  // so the inverse only ever reads a worker's scratch copy, never the
  // saved forward coefficients (fftlog_execute_cov).
  fftw_plan forward = fftw_plan_dft_r2c_1d(nfft, work->real[0],
      work->scratch[0], FFTW_ESTIMATE);
  work->inverse = fftw_plan_dft_c2r_1d(nfft, work->scratch[0],
      work->real[0], FFTW_ESTIMATE);
  if (forward == NULL
      || work->inverse == NULL) {
    log_fatal("fftlog_create_cov: FFTW plan creation failed");
    exit(1);
  }

  // Each field has its own radial selection. Save its forward transform
  // once: changing ell later changes only the Bessel integration kernel.
  // One iteration prepares one field on one worker: it lays out f/chi on
  // the padded ln(chi) array, expands it in Fourier modes (the powers of
  // chi of the header) and tapers the highest frequencies. Fields are
  // independent, and each writes only its own coefficient row.
  // A worker fully clears its row before inserting the physical samples,
  // including the unused tail left by rounding the FFT size upward.
  #pragma omp parallel for schedule(static)
  for (int field=0; field<nfield; field++) {
    int id = 0;
#ifdef _OPENMP
    id = omp_get_thread_num();
#endif
    if (inverse_power[field] != 0
        && inverse_power[field] != 2) {
      log_fatal("fftlog_create_cov: kernel power must be 0 or 2");
      exit(1);
    }
    work->inverse_power[field] = inverse_power[field];
    for (int node=0; node<nfft; node++) {
      work->real[id][node] = 0.0;
    }

    // Physical sample node sits at chi = chi_min*exp(node*dlnchi), array
    // index padding+node. Dividing f by chi applies the bias nu = 1.
    for (int node=0; node<nchi; node++) {
      if (!isfinite(radial[field][node])) {
        log_fatal("fftlog_create_cov: radial input must be finite");
        exit(1);
      }
      const double chi = chi_min*exp(node*dlnchi);
      work->real[id][padding+node] = radial[field][node]/chi;
    }

    // New-array execution shares the recipe, not its original buffers.
    // Each field writes separate coefficients; a later inverse cannot
    // overwrite them because it uses the worker's scratch allocation.
    // FFTW's forward sign: X_j = sum_m real[m]*exp(-2 pi i j m/nfft),
    // stored for j = 0..nfft/2.
    fftw_execute_dft_r2c(forward, work->real[id], work->forward[field]);

    // Taper the top quarter of the stored frequencies: width = nfft/8 of
    // the nfft/2 frequency steps (the FAST-PT window of McEwen et al.,
    // arXiv:1603.04826, appendix on edge effects). x = position falls
    // from 1 at the first tapered node to 0 at the Nyquist frequency
    // nfft/2, and the taper is
    //   W(x) = x - sin(2 pi x)/(2 pi),  W' = 1 - cos(2 pi x).
    // W, W' and W'' join the untouched modes (W = 1) continuously and all
    // vanish at the Nyquist end. W = 0 there also removes the Nyquist
    // coefficient, whose frequency +nfft/2 is the same discrete mode as
    // -nfft/2; once multiplied by a complex kernel, its imaginary part
    // would be discarded by FFTW's c2r inverse.
    const int width = nfft/8;
    for (int node=nfreq-1-width; node<nfreq; node++) {
      const double position = (double) (nfreq-1-node)/width;
      const double taper = position-sin(2.0*M_PI*position)/(2.0*M_PI);
      work->forward[field][node] *= taper;
    }
  }

  // Every field's coefficients are saved; the forward recipe is no longer
  // needed. The inverse recipe stays in the workspace.
  fftw_destroy_plan(forward);
  return work;
}


// -----------------------------------------------------------------------
// Integrate each radial Fourier mode against a spherical Bessel function.
//
// One FFTLog transform has three steps: expand the radial input in
// powers of distance (the forward FFT saved by fftlog_create_cov),
// integrate each power against the Bessel kernel analytically (multiply
// its coefficient by a Mellin kernel), and add the integrated powers at
// every output k (one inverse FFT per field and multipole).
//
// A Fourier mode of f/chi becomes chi^s in f, where s=1+i eta.
// Substitution t=k chi gives k^-s times the Mellin integral
//   M_l(s) = integral_0^infinity dt t^(s-1) j_l(t)
//          = sqrt(pi) 2^(s-2) Gamma((l+s)/2)/Gamma((l+3-s)/2).
// Multiplying the Fourier coefficients by M_l performs this integral
// for all modes; the inverse FFT adds their contributions at each k.
//
// Compute gamma ratios for the first even and odd ell only. The identity
// Gamma(z+1)=z Gamma(z) advances each sequence by two multipoles:
//   M_(l+2)(s) = (l+s)/(l+3-s) M_l(s).
// For lensing, the radial input stays f = chi*W, as for density: the
// 1/(k chi)^2 stays inside the Bessel kernel, j_l(t)/t^2, instead of an
// external k^-2 with the input divided by chi^2. Its Mellin kernel is
// M_l(s-2) = M_l(s)/[(l+s-2)(l+3-s)], again from Gamma(z+1)=z Gamma(z).
// This avoids amplifying the observer endpoint by chi^-2 (N5K,
// arXiv:2212.04291, Eq. 24).
//
// The two padded origins set the Fourier phase. FFT index padding is
// anchored at k0=(ell+1)/chi_max; reading extra earlier/later indices
// extends the integration range without changing that phase anchor.
//
// PHYSICAL DERIVATION & LOGIC FLOW
// Notation: N = nfft, L = N*dlnchi (the ln(chi) period of the array),
// eta_j = 2 pi j/L, s_j = 1 + i eta_j, p = inverse_power (0 or 2).
//
// 1. Expansion (fftlog_create_cov). Array index m holds
//      chi_m = chi_origin*exp(m*dlnchi),
//      chi_origin = chi_min*exp(-padding*dlnchi),
//    and X_j = sum_m [f/chi](chi_m) exp(-2 pi i j m/N). Inverting,
//      f(chi) = (1/N) sum_j X_j chi_origin^(-i eta_j) chi^(s_j),
//    where a negative j carries the complex conjugate of X_|j|.
// 2. Bessel integral of one power, with t = k chi:
//      integral dlnchi chi^s j_l(k chi)/(k chi)^p = k^(-s) M_l(s-p).
// 3. Sum over powers:
//      F_l(k) = (1/(N k)) sum_j X_j M_l(s_j-p) (chi_origin k)^(-i eta_j).
// 4. Output grid. Inverse index n holds k_n = k_origin*exp(n*dlnchi),
//      k_origin = k0*exp(-padding*dlnchi),
//    so index padding holds k0 = (l+1)/chi_max. Then
//      (chi_origin k_n)^(-i eta_j) = phase_j exp(-2 pi i j n/N),
//      phase_j = (chi_origin k_origin)^(-i eta_j),
//    and chi_origin*k_origin = origin*k0 in the code: the product of the
//    two padded origins.
// 5. Sign. FFTW's c2r inverse sums exp(+2 pi i j n/N). The sum of step 4
//    is real, so it equals the FFTW inverse of conjugated coefficients:
//      scratch_j = conj(X_j * kernel_j * phase_j),
//    kernel_j = M_l(s_j) for p = 0, M_l(s_j-2) for p = 2.
// 6. Normalization: transfer = inverse/(N k). FFTW does not divide by
//    N, and 1/k = k^(-Re s) is the non-oscillating factor of k^(-s); its
//    oscillating factor k^(-i eta) went into the phase of step 4.
//
// Parameters (see fftlog_cov.h): work, the prepared workspace, of which
// only the worker rows and the kernel table are written; the block
// first, ..., first+nell-1; outputs wave[nell][nk] and
// transfer[nell][field][nk], nk = nchi+2*extra.
// -----------------------------------------------------------------------
void fftlog_execute_cov(
    struct fftlog_workspace_cov* work, // owned transform preparation
    const int first,                  // first ell in this block
    const int nell,                   // block count, at most 16
    double* const* wave,              // [nell][nchi+2*extra]
    double** const* transfer          // [nell][field][nchi+2*extra]
  )
{
  // The kernel table holds 16 multipoles. l >= 2: the lensing integral
  // of j_l(t)/t^2 converges at Re(s) = 1 only for l >= 2, and spin-2
  // fields have no l < 2 modes.
  if (first < 2
      || nell < 1
      || nell > 16) {
    log_fatal("fftlog_execute_cov needs ell >= 2 and 1..16 multipoles");
    exit(1);
  }

  // Worker rows are indexed by thread number: a team larger than the one
  // counted at creation would index past the allocated rows.
#ifdef _OPENMP
  if (omp_get_max_threads() > work->nthreads) {
    log_fatal("fftlog_execute_cov: recreate workspace for a larger team");
    exit(1);
  }
#endif

  // nfreq stored frequencies and nk output samples per transform. The
  // array period L = nfft*dlnchi in ln(chi) sets eta_j = 2 pi j/L.
  // chi_max is the last physical sample. origin*k0 = chi_origin*k_origin
  // is the product of the two padded origins of the derivation (step 4),
  // each padding samples below its first physical value.
  const int nfreq = work->nfft/2+1;
  const int nk = work->nchi+2*work->extra;
  const double period = work->nfft*work->dlnchi;
  const double chi_max = work->chi_min
                         *exp((work->nchi-1)*work->dlnchi);
  const double origin = work->chi_min
                        *exp(-2*work->padding*work->dlnchi);

  // Tabulate the density kernels M_l(s_j) of the whole block; every field
  // shares them. At one frequency the even and the odd multipoles form two
  // independent recurrences, so one task is a (parity, frequency) pair: it
  // evaluates M_l exactly for the first multipole of its parity and steps
  // up by two. Distribute frequencies as well as parity so the table
  // uses the whole team: 2*nfreq tasks, whatever the field count.
  // Workers store disjoint kernels; no Fourier sum is split among them.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int parity=0; parity<2; parity++) {
    for (int node=0; node<nfreq; node++) {
      const double eta = 2.0*M_PI*node/period;
      const double complex s = 1.0+I*eta;

      // Seed l = first+parity from log-gamma: Gamma((l+1)/2) alone
      // overflows a double above l ~ 342, and |Gamma| underflows at large
      // eta (it decays like exp(-pi |eta|/4)), while the ratio of the two
      // Gammas stays moderate. GSL returns ln Gamma(z) = ln|Gamma(z)| +
      // i arg Gamma(z) as two results, for z = (l+s)/2 = (l+1)/2 + i eta/2
      // and z = (l+3-s)/2 = (l+2)/2 - i eta/2.
      gsl_sf_result numerator;       // logarithmic gamma amplitude
      gsl_sf_result numerator_phase; // gamma phase in radians
      gsl_sf_result denominator;     // logarithmic gamma amplitude
      gsl_sf_result denominator_phase; // gamma phase in radians
      gsl_sf_lngamma_complex_e((first+parity+1.0)/2.0, eta/2.0,
          &numerator, &numerator_phase);
      gsl_sf_lngamma_complex_e((first+parity+2.0)/2.0, -eta/2.0,
          &denominator, &denominator_phase);

      // M_l(s) = sqrt(pi) 2^(s-2) Gamma((l+s)/2)/Gamma((l+3-s)/2), with
      // 2^(s-2) = (1/2) exp(i eta ln 2): sqrt(pi)/2 is the real prefactor,
      // eta ln 2 joins the gamma phases, and the amplitudes subtract.
      double complex mellin = sqrt(M_PI)/2.0
          *cexp(numerator.val-denominator.val
                +I*(eta*log(2.0)+numerator_phase.val
                    -denominator_phase.val));

      // Store M_l for each multipole of this parity in the block, then
      // advance to l+2 with M_(l+2)(s) = (l+s)/(l+3-s) M_l(s). A block of
      // 16 takes at most seven steps from the exact seed.
      for (int index=parity; index<nell; index+=2) {
        const double ell = first+index;
        work->kernel[index][node] = mellin;
        mellin *= (ell+s)/(ell+3.0-s);
      }
    }
  }

  // The reciprocal grid of each multipole. FFTLog's output has the same
  // logarithmic step as its input: k_q = k0*exp((q-extra)*dlnchi). Node
  // extra holds k0 = (l+1)/chi_max. Below k0 even the farthest shell has
  // k chi < l+1, before the first peak of j_l, which falls toward zero
  // like (k chi)^l there; node nk-1-extra holds (l+1)/chi_min. The extra
  // nodes at each end extend the k range beyond these two anchors.
  for (int index=0; index<nell; index++) {
    const double k0 = (first+index+1.0)/chi_max;
    for (int node=0; node<nk; node++) {
      wave[index][node] = k0*exp((node-work->extra)*work->dlnchi);
    }
  }

  // A transform belongs to one field and one ell: it applies steps 3-6
  // of the derivation to that field's saved coefficients. Collapse both
  // axes so the number of parallel tasks is not limited by the catalog
  // count. Each task writes its entire transfer row and borrows one
  // worker's scratch. The saved forward coefficients and plan remain
  // read-only.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int index=0; index<nell; index++) {
    for (int field=0; field<work->nfield; field++) {
      int id = 0;
#ifdef _OPENMP
      id = omp_get_thread_num();
#endif
      // phase_j = (origin*k0)^(-i eta_j) = exp(i j phase_step), with
      // phase_step = -2 pi ln(origin*k0)/L (step 4 of the derivation).
      // rotation = exp(i phase_step) turns phase_j into phase_(j+1).
      const double ell = first+index;
      const double k0 = (ell+1.0)/chi_max;
      const double phase_step = -2.0*M_PI*log(origin*k0)/period;
      const double complex rotation = cexp(I*phase_step);
      double complex phase = 1.0;

      // Successive Fourier frequencies differ by a constant. Rotate the
      // phase rather than recomputing trigonometry for every coefficient;
      // reset it every 64 nodes to bound roundoff in that recurrence.
      // One iteration prepares the inverse coefficient of frequency node.
      for (int node=0; node<nfreq; node++) {
        if (node%64 == 0) {
          phase = cexp(I*(node*phase_step));
        }

        // Density (p = 0) uses M_l(s) from the shared table. Lensing
        // (p = 2) divides by (l+s-2)(l+3-s), which turns it into
        // M_l(s-2), the Mellin integral of j_l(t)/t^2.
        double complex kernel = work->kernel[index][node];
        if (work->inverse_power[field] == 2) {
          const double complex s = 1.0+I*(2.0*M_PI*node/period);
          kernel /= (ell+s-2.0)*(ell+3.0-s);
        }

        // Step 5: the conjugate lets FFTW's exp(+i...) inverse evaluate
        // the exp(-i...) sum of step 4. The product goes to this worker's
        // scratch row; the saved coefficients are only read.
        work->scratch[id][node] =
            conj(work->forward[field][node]*kernel*phase);
        phase *= rotation;
      }

      // New-array execution of the shared inverse recipe: it reads this
      // worker's scratch row, which a c2r transform may overwrite, and
      // writes the worker's real row: real[n] = sum over frequencies.
      fftw_execute_dft_c2r(work->inverse, work->scratch[id], work->real[id]);

      // Step 6: normalize and store F_l(k).
      //
      // The transfer at output node q is the inverse sum at index
      // padding-extra+q divided by N*k_q. N supplies the 1/N that FFTW's
      // unnormalized inverse omits; 1/k is the k^(-1) left by the bias
      // nu=1. Output node extra is inverse index padding, the anchor k0.
      // The two lanes hold two adjacent output nodes, q and q+1, of the
      // same field and multipole. Nothing is summed across lanes, and each
      // lane performs the scalar operations below (one multiply and one
      // divide, each rounded once), so SIMD and scalar results are bitwise
      // equal.
      //
      // scalar:
      //   for (int node=0; node<nk; node++) {
      //     output[node] = input[node]/(work->nfft*wave[index][node]);
      //   }
      const double* input = work->real[id]+work->padding-work->extra;
      double* output = transfer[index][field];

      // set1 duplicates N = nfft, converted to double, into both lanes so
      // both samples use the same normalization.
      const v2d vlength = simde_mm_set1_pd(work->nfft);

      // Two output nodes per step while node and node+1 are both valid;
      // an odd nk leaves the last node to the scalar statement below.
      int node = 0;
      for (; node+1<nk; node+=2) {
        // loadu reads input[node], input[node+1], the unnormalized
        // inverse sums at k[node] and k[node+1], into lanes 0 and 1; it
        // does not require the shifted physical interval to start at a
        // 16-byte aligned address.
        const v2d vinput = simde_mm_loadu_pd(input+node);

        // loadu reads wave[index][node], wave[index][node+1], the two
        // wavenumbers, into lanes 0 and 1, matching vinput lane by lane.
        const v2d vwave = simde_mm_loadu_pd(wave[index]+node);

        // mul forms N*k separately for the two wavenumbers:
        // scalar nfft*wave[index][q], rounded once.
        const v2d vscale = simde_mm_mul_pd(vlength, vwave);

        // div applies the normalization independently in both lanes:
        // input[q]/(N*k_q), rounded once, the transfer F_l(k_q).
        const v2d vvalue = simde_mm_div_pd(vinput, vscale);

        // storeu writes both results to ordinary adjacent output doubles,
        // output[node] and output[node+1], i.e. transfer[index][field]
        // [node..node+1]; both exist since node+1 < nk, and no alignment
        // is required.
        simde_mm_storeu_pd(output+node, vvalue);
      }

      // Scalar remainder: the last node when nk is odd (always so for the
      // nonlimber_cov.c grid, nk = nchi+2*extra with nchi odd).
      if (node < nk) {
        output[node] = input[node]/(work->nfft*wave[index][node]);
      }
    }
  }
}


// Release serially after all executions finish. No output borrows storage
// from the workspace, so returned transfer arrays remain valid afterwards.
// Each grouped array is one allocation, so one free releases it; the plan
// is released through FFTW.
void fftlog_free_cov(struct fftlog_workspace_cov* work)
{
  fftw_destroy_plan(work->inverse);
  free(work->inverse_power);
  free(work->real);
  free(work->scratch);
  free(work->forward);
  free(work->kernel);
  free(work);
}
