#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <gsl/gsl_math.h>
#include "utils_complex_cfastpt.h"
#include "utils_cfastpt.h"

#include "../log.c/src/log.h"

void c_window(double complex *out, double c_window_width, long halfN) {
	// 'out' is (halfN+1) complex array
	long Ncut;
	Ncut = (long)(halfN * c_window_width);
	long i;
	double W;
	for(i=0; i<=Ncut; i++) { // window for right-side
		W = (double)(i)/Ncut - 1./(2.*M_PI) * sin(2.*i*M_PI/Ncut);
		out[halfN-i] *= W;
	}
}


void fftconvolve(fftw_complex *in1, fftw_complex *in2, long N, fftw_complex *out) {
	long i;
	fftw_complex *a, *b;
	fftw_complex *a1, *b1;
	fftw_complex *c;
	fftw_plan pa, pb, pc;

	long Ntotal;
	if(N%2==1) {
		Ntotal = 2*N - 1;
	}else {
		log_fatal("This fftconvolve doesn't support even size input arrays"); 
		exit(1);
		Ntotal = 2*N;
	}

	a = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal );
	b = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal );
	a1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal );
	b1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal );

	c = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal );

	for(i=0;i<N;i++){
		a[i] = in1[i];
		b[i] = in2[i];
	}
	for( ;i<Ntotal;i++){
		a[i] = 0.;
		b[i] = 0.;
	}

	pa = fftw_plan_dft_1d(Ntotal, a, a1, FFTW_FORWARD, FFTW_ESTIMATE);
	pb = fftw_plan_dft_1d(Ntotal, b, b1, FFTW_FORWARD, FFTW_ESTIMATE);
	#pragma omp parallel
	#pragma omp single 
	{
		#pragma omp task
		fftw_execute(pa);
		#pragma omp task
		fftw_execute(pb);
		#pragma omp taskwait
	}

	for(i=0;i<Ntotal;i++){
		a1[i] *= b1[i];
	}
	pc = fftw_plan_dft_1d(Ntotal, a1, c, FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(pc);

	for(i=0;i<Ntotal; i++){
		out[i] = c[i]/(double complex)Ntotal;
	}

	fftw_destroy_plan(pa);
	fftw_destroy_plan(pb);
	fftw_destroy_plan(pc);
	fftw_free(a);
	fftw_free(b);
	fftw_free(a1);
	fftw_free(b1);
	fftw_free(c);
}

void fftconvolve_optimize(fftw_complex *in1, fftw_complex *in2, long N, fftw_complex *out, fftw_complex *a, 
fftw_complex *b, fftw_complex *a1, fftw_complex *b1, fftw_complex *c, fftw_plan pa, fftw_plan pb, fftw_plan pc) 
{
	long i;

	long Ntotal;
	if(N%2==1) 
	{
		Ntotal = 2*N - 1;
	}
	else 
	{
		log_fatal("This fftconvolve doesn't support even size input arrays"); 
		exit(1);
		Ntotal = 2*N;
	}

	for(i=0;i<N;i++){
		a[i] = in1[i];
		b[i] = in2[i];
	}
	for( ;i<Ntotal;i++){
		a[i] = 0.;
		b[i] = 0.;
	}

//	#pragma omp parallel
//	#pragma omp single 
	{
//		#pragma omp task
		fftw_execute(pa);
//		#pragma omp task
		fftw_execute(pb);
//		#pragma omp taskwait
	}

	for(i=0;i<Ntotal;i++)
	{
		a1[i] *= b1[i];
	}
	
	fftw_execute(pc);

	for(i=0;i<Ntotal; i++)
	{
		out[i] = c[i]/(double complex)Ntotal;
	}
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -- NEW IMPLEMENTATION -------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

static const double LANCZOS_P[] = {
  0.99999999999980993227684700473478,
  676.520368121885098567009190444019,
  -1259.13921672240287047156078755283,
  771.3234287776530788486528258894,
  -176.61502916214059906584551354,
  12.507343278686904814458936853,
  -0.13857109526572011689554707,
  9.984369578019570859563e-6,
  1.50563273514931155834e-7
};

static inline double complex lngamma_lanczos(double complex z) {
  if (creal(z) < 0.5)
    return clog(M_PI) - clog(csin(M_PI*z)) - lngamma_lanczos(1. - z);
  z -= 1;
  double complex x = LANCZOS_P[0];
  for (int n = 1; n < 9; n++)
    x += LANCZOS_P[n] / (z + (double)n);
  double complex t = z + 7.5;
  return 0.5 * log(2*M_PI) + (z+0.5)*clog(t) - t + clog(x);
}

// compute lngamma(a) - lngamma(b) directly, sharing loop work
// assumes Re(a) >= 0.5 and Re(b) >= 0.5 (no reflection needed)
static inline double complex lngamma_ratio(double complex a, double complex b) {
  a -= 1;
  b -= 1;
  double complex xa = LANCZOS_P[0];
  double complex xb = LANCZOS_P[0];
  for (int n = 1; n < 9; n++) {
    xa += LANCZOS_P[n] / (a + (double)n);
    xb += LANCZOS_P[n] / (b + (double)n);
  }
  double complex ta = a + 7.5;
  double complex tb = b + 7.5;
  return (a+0.5)*clog(ta) - (b+0.5)*clog(tb) - ta + tb + clog(xa) - clog(xb);
}

void g_m_vals(double mu, double q_real, double *q_imag, double complex *gm, long N) {
  const double a_real = (mu + 1. + q_real) / 2.;
  const double b_real = (mu + 1. - q_real) / 2.;
  #pragma omp parallel for
  for (long i = 0; i < N; i++) {
    double qi = q_imag[i] / 2.;
    gm[i] = cexp(lngamma_ratio(a_real + I*qi, b_real - I*qi));
  }
}

void gamma_ratios(double l, double nu, double *eta, double complex *gl, long N) {
  const double a_real = (l + nu) / 2.;
  const double b_real = (3. + l - nu) / 2.;
  #pragma omp parallel for
  for (long i = 0; i < N; i++) {
    double ei = eta[i] / 2.;
    gl[i] = cexp(lngamma_ratio(a_real + I*ei, b_real - I*ei));
  }
}

void f_z(double z_real, double *z_imag, double complex *fz, long N) {
  g_m_vals(0.5, z_real - 0.5, z_imag, fz, N);
  const double sqrt_pi_2 = sqrt(M_PI) / 2.;
  const double ln2 = log(2.);
  #pragma omp parallel for
  for (long i = 0; i < N; i++) {
    double complex z = z_real + I*z_imag[i];
    fz[i] *= sqrt_pi_2 * cexp(z * ln2);
  }
}

// ---------------------------------------------------------------------------
// fftconvolve_real: Real-valued FFT convolution of in1 (length N1) and
// in2 (length N2), result in out (length N1+N2-1).
//
// OPTIMIZATIONS:
// 1. Static caching of memory blocks and FFTW plans (N1 and N2 are always
//    Nk and 2*Nk-1, so sizes don't change across calls).
// 2. Three template plans instead of creating/destroying 3 plans per call.
// 3. Pre-zero-pad tails once at allocation time.
// 4. Eliminated separate output array c — write directly into out via
//    fftw_execute_dft_c2r with out as the output buffer.
// ---------------------------------------------------------------------------
void fftconvolve_real(double *in1, double *in2, long N1, long N2, double *out) {
  static long s_N1 = 0, s_N2 = 0;
  static long s_Ntotal = 0, s_Ncomplex = 0;
  static double s_inv_Ntotal = 0;
  static double *s_a = NULL, *s_b = NULL;
  static fftw_complex *s_a1 = NULL, *s_b1 = NULL;
  static fftw_plan s_pa = NULL, s_pb = NULL, s_pc = NULL;

  // Rebuild if sizes changed
  if (s_N1 != N1 || s_N2 != N2) {
    if (s_a) {
      fftw_destroy_plan(s_pa);
      fftw_destroy_plan(s_pb);
      fftw_destroy_plan(s_pc);
      fftw_free(s_a); fftw_free(s_b);
      fftw_free(s_a1); fftw_free(s_b1);
    }

    s_N1 = N1;
    s_N2 = N2;
    s_Ntotal = N1 + N2 - 1;
    s_Ncomplex = (s_Ntotal % 2 == 1) ? (s_Ntotal+1)/2 : s_Ntotal/2+1;
    s_inv_Ntotal = 1.0 / (double)s_Ntotal;

    s_a  = fftw_malloc(sizeof(double) * s_Ntotal);
    s_b  = fftw_malloc(sizeof(double) * s_Ntotal);
    s_a1 = fftw_malloc(sizeof(fftw_complex) * s_Ncomplex);
    s_b1 = fftw_malloc(sizeof(fftw_complex) * s_Ncomplex);

    // Template plans — applied to different data via fftw_execute_dft_r2c/c2r
    s_pa = fftw_plan_dft_r2c_1d(s_Ntotal, s_a, s_a1, FFTW_ESTIMATE);
    s_pb = fftw_plan_dft_r2c_1d(s_Ntotal, s_b, s_b1, FFTW_ESTIMATE);
    s_pc = fftw_plan_dft_c2r_1d(s_Ntotal, s_a1, s_a, FFTW_ESTIMATE);

    // Pre-zero-pad tails (only written once; per-call copies don't touch them)
    memset(s_a + N1, 0, sizeof(double) * (s_Ntotal - N1));
    memset(s_b + N2, 0, sizeof(double) * (s_Ntotal - N2));
  }

  // Copy inputs (tails stay zero from initialization)
  memcpy(s_a, in1, sizeof(double) * N1);
  memcpy(s_b, in2, sizeof(double) * N2);

  // Forward FFTs
  fftw_execute(s_pa);
  fftw_execute(s_pb);

  // Pointwise multiply in Fourier space
  for (long i = 0; i < s_Ncomplex; i++)
    s_a1[i] *= s_b1[i];

  // Backward FFT + normalize into out
  fftw_execute_dft_c2r(s_pc, s_a1, out);
  for (long i = 0; i < s_Ntotal; i++)
    out[i] *= s_inv_Ntotal;
}
