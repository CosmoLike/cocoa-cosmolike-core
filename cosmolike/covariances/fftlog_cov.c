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

typedef simde__m128d v2d;

// The caller owns one recipe and one set of radial Fourier coefficients.
// Workers borrow that recipe, but only write their own scratch rows.
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
// known analytically. We use bias nu=1: Fourier transform f(chi)/chi,
// then restore the removed power through the Mellin kernel and 1/k.
// This is the single-Bessel construction of Fang et al. (1911.11947).
//
// The physical interval occupies indices padding..padding+nchi-1.
// Everything else is zero: those empty samples separate periodic copies
// of the input, rather than extrapolating the survey to other distances.
// A frequency taper suppresses the highest quarter of Fourier modes to
// reduce ringing at interval edges. Radial support and taper accuracy
// must still be tested against direct integrals for each survey.
//
// Only two plans are made, serially. Every field uses the same forward
// recipe with new arrays; it is destroyed after the coefficients are
// saved. The inverse recipe survives for every later multipole block.
// Dimensions and alignment are identical for the worker scratch rows.
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
  work->nthreads = 1;
#ifdef _OPENMP
  work->nthreads = omp_get_max_threads();
#endif
  const int nfreq = nfft/2+1;
  work->inverse_power = (int*) malloc1d_int(nfield);
  work->real = (double**) malloc2d(work->nthreads, nfft);
  work->scratch = (fftw_complex**)
      malloc2d_fftwc(work->nthreads, nfreq);
  work->forward = (fftw_complex**) malloc2d_fftwc(nfield, nfreq);
  work->kernel = (fftw_complex**) malloc2d_fftwc(16, nfreq);

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
    fftw_execute_dft_r2c(forward, work->real[id], work->forward[field]);
    const int width = nfft/8;
    for (int node=nfreq-1-width; node<nfreq; node++) {
      const double position = (double) (nfreq-1-node)/width;
      const double taper = position-sin(2.0*M_PI*position)/(2.0*M_PI);
      work->forward[field][node] *= taper;
    }
  }
  fftw_destroy_plan(forward);
  return work;
}


// -----------------------------------------------------------------------
// Integrate each radial Fourier mode against a spherical Bessel function.
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
// For lensing, keep chi^2 in the radial input and integrate j_l(t)/t^2.
// Its Mellin kernel is M_l(s)/[(l+s-2)(l+3-s)]. This avoids amplifying
// the observer endpoint by chi^-2 (N5K, arXiv:2212.04291, Eq. 24).
//
// The two padded origins set the Fourier phase. FFT index padding is
// anchored at k0=(ell+1)/chi_max; reading extra earlier/later indices
// extends the integration range without changing that phase anchor.
// -----------------------------------------------------------------------
void fftlog_execute_cov(
    struct fftlog_workspace_cov* work, // owned transform preparation
    const int first,                  // first ell in this block
    const int nell,                   // block count, at most 16
    double* const* wave,              // [nell][nchi+2*extra]
    double** const* transfer          // [nell][field][nchi+2*extra]
  )
{
  if (first < 2
      || nell < 1
      || nell > 16) {
    log_fatal("fftlog_execute_cov needs ell >= 2 and 1..16 multipoles");
    exit(1);
  }
#ifdef _OPENMP
  if (omp_get_max_threads() > work->nthreads) {
    log_fatal("fftlog_execute_cov: recreate workspace for a larger team");
    exit(1);
  }
#endif
  const int nfreq = work->nfft/2+1;
  const int nk = work->nchi+2*work->extra;
  const double period = work->nfft*work->dlnchi;
  const double chi_max = work->chi_min
                         *exp((work->nchi-1)*work->dlnchi);
  const double origin = work->chi_min
                        *exp(-2*work->padding*work->dlnchi);

  // Each frequency has two independent multipole sequences. Distribute
  // frequencies as well as parity so even a single field uses the team.
  // Workers store disjoint kernels; no Fourier sum is split among them.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int parity=0; parity<2; parity++) {
    for (int node=0; node<nfreq; node++) {
      const double eta = 2.0*M_PI*node/period;
      const double complex s = 1.0+I*eta;
      gsl_sf_result numerator;       // logarithmic gamma amplitude
      gsl_sf_result numerator_phase; // gamma phase in radians
      gsl_sf_result denominator;     // logarithmic gamma amplitude
      gsl_sf_result denominator_phase; // gamma phase in radians
      gsl_sf_lngamma_complex_e((first+parity+1.0)/2.0, eta/2.0,
          &numerator, &numerator_phase);
      gsl_sf_lngamma_complex_e((first+parity+2.0)/2.0, -eta/2.0,
          &denominator, &denominator_phase);
      double complex mellin = sqrt(M_PI)/2.0
          *cexp(numerator.val-denominator.val
                +I*(eta*log(2.0)+numerator_phase.val
                    -denominator_phase.val));

      for (int index=parity; index<nell; index+=2) {
        const double ell = first+index;
        work->kernel[index][node] = mellin;
        mellin *= (ell+s)/(ell+3.0-s);
      }
    }
  }

  for (int index=0; index<nell; index++) {
    const double k0 = (first+index+1.0)/chi_max;
    for (int node=0; node<nk; node++) {
      wave[index][node] = k0*exp((node-work->extra)*work->dlnchi);
    }
  }

  // A transform belongs to one field and one ell. Collapse both axes
  // so the number of parallel tasks is not limited by the catalog count.
  // Each task writes its entire transfer row and borrows one worker's
  // scratch. The saved forward coefficients and plan remain read-only.
  #pragma omp parallel for collapse(2) schedule(static)
  for (int index=0; index<nell; index++) {
    for (int field=0; field<work->nfield; field++) {
      int id = 0;
#ifdef _OPENMP
      id = omp_get_thread_num();
#endif
      const double ell = first+index;
      const double k0 = (ell+1.0)/chi_max;
      const double phase_step = -2.0*M_PI*log(origin*k0)/period;
      const double complex rotation = cexp(I*phase_step);
      double complex phase = 1.0;

      // Successive Fourier frequencies differ by a constant. Rotate the
      // phase rather than recomputing trigonometry for every coefficient;
      // reset it every 64 nodes to bound roundoff in that recurrence.
      for (int node=0; node<nfreq; node++) {
        if (node%64 == 0) {
          phase = cexp(I*(node*phase_step));
        }
        double complex kernel = work->kernel[index][node];
        if (work->inverse_power[field] == 2) {
          const double complex s = 1.0+I*(2.0*M_PI*node/period);
          kernel /= (ell+s-2.0)*(ell+3.0-s);
        }
        work->scratch[id][node] =
            conj(work->forward[field][node]*kernel*phase);
        phase *= rotation;
      }

      fftw_execute_dft_c2r(work->inverse, work->scratch[id], work->real[id]);

      // Scalar equivalent: output[q] = inverse[padding-extra+q]/(N*k[q]).
      // N removes FFTW's inverse-transform normalization; 1/k restores
      // the real bias nu=1. SIMD processes two adjacent k samples at once.
      const double* input = work->real[id]+work->padding-work->extra;
      double* output = transfer[index][field];
      // set1 duplicates N so both samples use the same normalization.
      const v2d vlength = simde_mm_set1_pd(work->nfft);
      int node = 0;
      for (; node+1<nk; node+=2) {
        // loadu reads adjacent doubles into lanes 0 and 1; it does not
        // require the shifted physical interval to start at alignment.
        const v2d vinput = simde_mm_loadu_pd(input+node);
        const v2d vwave = simde_mm_loadu_pd(wave[index]+node);
        // mul forms N*k separately for the two wavenumbers.
        const v2d vscale = simde_mm_mul_pd(vlength, vwave);
        // div applies the normalization independently in both lanes.
        const v2d vvalue = simde_mm_div_pd(vinput, vscale);
        // storeu writes both results to ordinary adjacent output doubles.
        simde_mm_storeu_pd(output+node, vvalue);
      }
      if (node < nk) {
        output[node] = input[node]/(work->nfft*wave[index][node]);
      }
    }
  }
}


// Release serially after all executions finish. No output borrows storage
// from the workspace, so returned transfer arrays remain valid afterwards.
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
