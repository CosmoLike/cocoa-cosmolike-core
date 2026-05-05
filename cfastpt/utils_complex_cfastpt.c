#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <gsl/gsl_math.h>
#include "utils_complex_cfastpt.h"
#include "utils_cfastpt.h"

#include <gsl/gsl_math.h>
#include "../log.c/src/log.h"

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

// ---------------------------------------------------------------------------
// lngamma_lanczos: Compute ln(Gamma(z)) for complex z using the Lanczos
// approximation with g=7 and 9 coefficients.
//
// The Lanczos approximation gives:
//   Gamma(z) ≈ sqrt(2*pi) * (z + g - 0.5)^(z - 0.5) * exp(-(z + g - 0.5)) * x(z)
// where x(z) = P[0] + sum_{n=1}^{8} P[n]/(z-1+n).
//
// Taking the logarithm:
//   ln(Gamma(z)) = 0.5*ln(2*pi) + (z-0.5)*ln(z+g-0.5) - (z+g-0.5) + ln(x(z))
//
// For Re(z) < 0.5, the Euler reflection formula is used:
//   Gamma(z) * Gamma(1-z) = pi / sin(pi*z)
//   => ln(Gamma(z)) = ln(pi) - ln(sin(pi*z)) - ln(Gamma(1-z))
// This recurses once with (1-z), which has Re(1-z) > 0.5.
//
// NOTE: This function is only used standalone for rare cases
// The hot path uses lngamma_ratio instead, which computes ln(Gamma(a)) - ln(Gamma(b))
// in a single fused pass for better performance and numerical stability.
//
// Parameters:
//   z:  complex argument
// Returns:
//   ln(Gamma(z)) as a double complex
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// lngamma_ratio: Compute ln(Gamma(a)) - ln(Gamma(b)) in a single pass.
//
// Instead of two independent lngamma_lanczos calls (each doing 9 complex
// divisions, a clog, etc.), this fused version shares the Lanczos series
// loop: both xa and xb are accumulated in the same iteration, halving
// the number of complex divisions.
//
// The Lanczos approximation (g=7) gives:
//   ln(Gamma(z+1)) = 0.5*ln(2*pi) + (z+0.5)*ln(z+g+0.5) - (z+g+0.5) + ln(x(z))
// where x(z) = P[0] + sum_{n=1}^{8} P[n]/(z+n).
//
// For the ratio ln(Gamma(a)) - ln(Gamma(b)), the 0.5*ln(2*pi) terms cancel,
// leaving only the difference of the remaining terms.
//
// ASSUMPTION: Re(a) >= 0.5 and Re(b) >= 0.5, so the Lanczos reflection
// formula (for Re(z) < 0.5) is not needed. This holds for all FAST-PT
// calls where a and b come from (mu+1+q)/2 with mu >= 0.5.
//
// Parameters:
//   a, b:  complex arguments with Re(a) >= 0.5 and Re(b) >= 0.5
// Returns:
//   ln(Gamma(a)) - ln(Gamma(b))  as a double complex
// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// g_m_vals: Compute the g_m kernel values for the FAST-PT algorithm.
//
// Evaluates the ratio of Gamma functions:
//   g_m(mu, q, q_imag) = Gamma((mu+1+q+I*q_imag)/2) / Gamma((mu+1-q-I*q_imag)/2)
// for each frequency mode i, where q_imag = q_imag[i].
//
// This kernel appears in the Mellin-space representation of the FAST-PT
// convolution integrals. The mu parameter controls the spherical Bessel function 
// order (mu = J + 0.5 for the J_abJ1J2Jk_ar terms), and q_real is the biasing exponent.
//
// Uses lngamma_ratio for numerical stability: the log of the Gamma ratio
// is computed in a single Lanczos pass (no overflow), then exponentiated.
//
// Parameters:
//   mu:      order parameter (typically J + 0.5)
//   q_real:  real part of the biasing exponent
//   q_imag:  array of imaginary parts (Fourier frequencies), length N
//   gm:      output array of complex g_m values, length N
//   N:       number of frequency modes
// ---------------------------------------------------------------------------
void g_m_vals(double mu, double q_real, double *q_imag, double complex *gm, long N) {
  const double a_real = (mu + 1. + q_real) / 2.;
  const double b_real = (mu + 1. - q_real) / 2.;
  #pragma omp parallel for
  for (long i = 0; i < N; i++) {
    double qi = q_imag[i] / 2.;
    gm[i] = cexp(lngamma_ratio(a_real + I*qi, b_real - I*qi));
  }
}

// ---------------------------------------------------------------------------
// gamma_ratios: Compute ratios of Gamma functions for the FFT-PT kernels.
//
// Evaluates g_l(nu, eta) = Gamma((l+nu+I*eta)/2) / Gamma((3+l-nu-I*eta)/2)
// for each frequency mode i, where eta = eta[i].
//
// These ratios appear in the spherical Bessel function decomposition of the
// FAST-PT integrals. The l parameter is the order of the spherical Bessel 
// function, and nu is the biasing exponent.
//
// The computation uses lngamma_ratio to evaluate the log of the ratio
// in a single pass (shared Lanczos series loop, no overflow risk), then
// exponentiates once. This is both faster and more numerically stable
// than computing two separate lngamma calls.
//
// Parameters:
//   l:    spherical Bessel function order (integer, but passed as double)
//   nu:   biasing exponent
//   eta:  array of Fourier frequencies, length N
//   gl:   output array of complex Gamma ratios, length N
//   N:    number of frequency modes
// ---------------------------------------------------------------------------
void gamma_ratios(double l, double nu, double *eta, double complex *gl, long N) {
  const double a_real = (l + nu) * 0.5;
  const double b_real = (3. + l - nu) * 0.5;

  #pragma omp parallel for
  for (long i = 0; i < N; i++) {
    double ei = eta[i] * 0.5;
    gl[i] = cexp(lngamma_ratio(a_real + I*ei, b_real - I*ei));
  }
}

// ---------------------------------------------------------------------------
// f_z: Compute the f(z) kernel for the FFT-PT algorithm.
//
// Computes f(z) = sqrt(pi)/2 * 2^z * g_m(0.5, z_real - 0.5, z_imag)
// where z = z_real + I*z_imag[i] for each frequency mode i.
//
// This function evaluates the Fourier-space kernel needed by the FAST-PT
// convolution integrals. It combines two pieces:
//   1. g_m_vals(mu=0.5, q=z_real-0.5): ratio of Gamma functions
//      Gamma((mu+1+q+I*q_imag)/2) / Gamma((mu+1-q-I*q_imag)/2)
//      evaluated at each imaginary frequency z_imag[i].
//   2. The factor sqrt(pi)/2 * 2^z, where the power 2^z = exp(z * ln2)
//      is computed for complex z via cexp.
//
// The relationship between f_z and the original gamma_ratios formulation:
//   f_z encodes g_l = exp(z*ln2 + lngamma((l+z)/2) - lngamma((3+l-z)/2))
//   specialized to l=0, rewritten in terms of g_m_vals for efficiency.
//
// Parameters:
//   z_real:  real part of z (the biasing exponent nu)
//   z_imag:  array of imaginary parts (Fourier frequencies), length N
//   fz:      output array of complex f(z) values, length N
//   N:       number of frequency modes
// ---------------------------------------------------------------------------
void f_z(double z_real, double *z_imag, double complex *fz, long N) {

  // Step 1: Compute the Gamma-function ratio part.
  // g_m_vals with mu=0.5 and q_real=z_real-0.5 gives:
  //   fz[i] = Gamma((1.5 + z_real - 0.5 + I*z_imag[i]) / 2)
  //         / Gamma((1.5 - z_real + 0.5 - I*z_imag[i]) / 2)
  //         = Gamma((1 + z_real + I*z_imag[i]) / 2)
  //         / Gamma((2 - z_real - I*z_imag[i]) / 2)
  g_m_vals(0.5, z_real - 0.5, z_imag, fz, N);

  // Step 2: Multiply by the prefactor sqrt(pi)/2 * 2^z.
  // The 2^z = exp(z * ln2) factor arises from the change of variables
  // in the Hankel-like transform underlying FFT-PT.
  // M_LN2 = ln(2) from gsl/gsl_math.h, avoids recomputing log(2).
  const double sqrt_pi_2 = sqrt(M_PI) / 2.;

  #pragma omp parallel for
  for (long i = 0; i < N; i++) {
    double complex z = z_real + I*z_imag[i];
    fz[i] *= sqrt_pi_2 * cexp(z * M_LN2);
  }
}

// ---------------------------------------------------------------------------
// fftconvolve_real: Real-valued FFT convolution of in1 (length N1) and
// in2 (length N2), result in out (length N1+N2-1).
//
// Computes the linear (non-circular) convolution:
//   out[k] = sum_j  in1[j] * in2[k-j]
// using the convolution theorem: FFT(in1) * FFT(in2) -> IFFT -> out.
//
// Used by IA_ta (deltaE2 term) and IA_mix (B term) for the direct
// convolution integrals that don't go through J_abJ1J2Jk_ar.
//
// OPTIMIZATIONS (vs original fftconvolve_real):
// 1. Static caching of memory blocks and FFTW plans. The sizes N1 and N2
//    are always Nk and 2*Nk-1 (set by the IA kernel), so they never change
//    across calls. Plans and buffers are created once and reused.
// 2. Two template r2c plans + one c2r plan instead of creating/destroying
//    3 plans per call. The r2c plans use fftw_execute (same buffers), the
//    c2r plan uses fftw_execute_dft_c2r to write directly into 'out'.
// 3. Pre-zero-pad tails once at allocation time. Per-call memcpy only
//    writes the data portion; the zero-padded tail is untouched.
// 4. Eliminated the separate output array 'c' — the backward FFT writes
//    directly into the caller's 'out' buffer via fftw_execute_dft_c2r.
// ---------------------------------------------------------------------------

// Not thread-safe: uses static buffers/plans.
void fftconvolve_real(double *in1, double *in2, long N1, long N2, double *out) {
  //---------------------------------------------------------------------------
  // STATIC CACHE: buffers and plans persist across calls.
  // Rebuilt only if N1 or N2 change (which in practice never happens).
  // ---------------------------------------------------------------------------
  static long s_N1 = 0, s_N2 = 0;       // cached input sizes
  static long s_Ntotal = 0;              // N1 + N2 - 1 (output/FFT size)
  static long s_Ncomplex = 0;            // number of complex bins in r2c output
  static double s_inv_Ntotal = 0;        // 1.0 / Ntotal (IFFT normalization)
  static double *s_a = NULL, *s_b = NULL;           // real input buffers
  static fftw_complex *s_a1 = NULL, *s_b1 = NULL;  // complex FFT output buffers
  static fftw_plan s_pa = NULL;          // r2c plan: s_a -> s_a1
  static fftw_plan s_pb = NULL;          // r2c plan: s_b -> s_b1
  static fftw_plan s_pc = NULL;          // c2r plan: s_a1 -> out (template)

  // Rebuild if sizes changed
  if (s_N1 != N1 || s_N2 != N2) {
    
    // Free old buffers and plans if they exist
    if (s_a) {
      fftw_destroy_plan(s_pa);
      fftw_destroy_plan(s_pb);
      fftw_destroy_plan(s_pc);
      fftw_free(s_a);
      fftw_free(s_b);
      fftw_free(s_a1);
      fftw_free(s_b1);
    }

    // Cache sizes and normalization constant
    s_N1 = N1;
    s_N2 = N2;
    s_Ntotal = N1 + N2 - 1;
    // Number of complex bins in r2c output: floor(Ntotal/2) + 1
    s_Ncomplex = (s_Ntotal % 2 == 1) ? (s_Ntotal + 1) / 2 : s_Ntotal / 2 + 1;
    s_inv_Ntotal = 1.0 / (double)s_Ntotal;

    // Allocate zero-padded real buffers (length Ntotal each).
    // in1 has N1 data points + (Ntotal-N1) zeros at the tail.
    // in2 has N2 data points + (Ntotal-N2) zeros at the tail.
    s_a = fftw_malloc(sizeof(double) * s_Ntotal);
    s_b = fftw_malloc(sizeof(double) * s_Ntotal);

    // Allocate complex buffers for r2c FFT output
    s_a1 = fftw_malloc(sizeof(fftw_complex) * s_Ncomplex);
    s_b1 = fftw_malloc(sizeof(fftw_complex) * s_Ncomplex);

    // Create FFTW plans:
    //   s_pa, s_pb: forward r2c, always applied to s_a/s_b -> s_a1/s_b1
    //   s_pc: backward c2r template, created with s_a1 -> s_a but applied
    //         via fftw_execute_dft_c2r to write into the caller's 'out'
    s_pa = fftw_plan_dft_r2c_1d(s_Ntotal, s_a, s_a1, FFTW_ESTIMATE);
    s_pb = fftw_plan_dft_r2c_1d(s_Ntotal, s_b, s_b1, FFTW_ESTIMATE);
    s_pc = fftw_plan_dft_c2r_1d(s_Ntotal, s_a1, s_a, FFTW_ESTIMATE);

    // Pre-zero-pad the tails once. Per-call memcpy only overwrites the
    // data portion (indices 0..N1-1 and 0..N2-1), leaving zeros intact.
    memset(s_a + N1, 0, sizeof(double) * (s_Ntotal - N1));
    memset(s_b + N2, 0, sizeof(double) * (s_Ntotal - N2));
  }

  // Copy input data into the cached buffers (zero-padded tails untouched)
  memcpy(s_a, in1, sizeof(double) * N1);
  memcpy(s_b, in2, sizeof(double) * N2);

  // Forward r2c FFTs: real input -> complex frequency domain
  fftw_execute(s_pa);
  fftw_execute(s_pb);

  // Pointwise multiply in frequency domain (convolution theorem)
  for (long i = 0; i < s_Ncomplex; i++) {
    s_a1[i] *= s_b1[i];
  }

  // Backward c2r FFT: complex frequency domain -> real output.
  // Uses fftw_execute_dft_c2r to write directly into caller's 'out'
  // buffer instead of an intermediate array.
  fftw_execute_dft_c2r(s_pc, s_a1, out);
  
  // FFTW's backward transform is unnormalized — divide by Ntotal
  for (long i = 0; i < s_Ntotal; i++) {
    out[i] *= s_inv_Ntotal;
  }
}
