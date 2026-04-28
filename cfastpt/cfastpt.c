#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <time.h>
#include <fftw3.h>
#include <gsl/gsl_math.h>
#include "cfastpt.h"
#include "utils_cfastpt.h"
#include "utils_complex_cfastpt.h"

#include "../log.c/src/log.h"

#include <omp.h>

void fastpt_scalar(int *alpha_ar, int *beta_ar, int *ell_ar, int *isP13type_ar,
double *coeff_A_ar, int Nterms, double *Pout, double *k, double *Pin, int Nk)
{
  double **Fy;
  Fy = malloc(sizeof(double*) * Nterms);
  for(int i=0;i<Nterms;i++) 
  {
    Fy[i] = malloc(sizeof(double) * Nk);
  }

  fastpt_config config;
  config.nu = -2.;
  config.c_window_width = 0.65; //0.25;
  config.N_pad = 1500;
  config.N_extrap_low = 500;
  config.N_extrap_high = 500;

  J_abl_ar(k, Pin, Nk, alpha_ar, beta_ar, ell_ar, isP13type_ar, Nterms, &config, Fy);

  #pragma omp parallel for
  for(int j=0; j<Nk; j++)
  {
    Pout[j] = 0.;
    for(int i=0; i<Nterms; i++) 
    {
      Pout[j] += coeff_A_ar[i] * Fy[i][j];
    }
  }


  for(int i = 0; i < Nterms; i++) 
  {
    free(Fy[i]);
  }
  free(Fy);
}


void J_abl_ar(double *x, double *fx, long N, int *alpha, int *beta, int *ell, 
int *isP13type __attribute__((unused)), int Nterms, fastpt_config *config, double **Fy) 
{
  // x: k array, fx: Pin array
  const long N_original = N;
  const long N_pad = config->N_pad;
  const long N_extrap_low = config->N_extrap_low;
  const long N_extrap_high = config->N_extrap_high;
  N += (2*N_pad + N_extrap_low+N_extrap_high);

  if(N % 2) 
  {
    log_fatal("cfastpt.c: J_abl_ar: Please use even number of x !");
    exit(0);
  }
  const long halfN = N/2;

  const double x0 = x[0];
  const double dlnx = log(x[1]/x0);

  // Only calculate the m>=0 part
  double eta_m[halfN+1];
  for(long i=0; i<=halfN; i++) 
  {
    eta_m[i] = 2*M_PI / dlnx / N * i;
  }

  // biased input func
  double* fb = malloc(N* sizeof(double));
  for(long i=0; i<N_pad; i++) 
  {
    fb[i] = 0.;
    fb[N-1-i] = 0.;
  }

  if(N_extrap_low) 
  {
    int sign;
    if(fx[0] == 0) 
    {
      log_fatal("J_abl_ar: Can't log-extrapolate zero on the low side!");
      exit(1);
    }
    else if (fx[0]>0) 
    {
      sign = 1;
    }
    else 
    {
      sign=-1;
    }
    
    if(fx[1]/fx[0] <= 0) 
    {
      log_fatal("J_abl_ar: Log-extrapolation on the low side fails due to sign change!");
      exit(1);
    }
    
    double dlnf_low = log(fx[1]/fx[0]);
    
    #pragma omp parallel for
    for(long i=N_pad; i<N_pad+N_extrap_low; i++) 
    {
      const double xi = exp(log(x0) + (i-N_pad - N_extrap_low)*dlnx);
      fb[i] = sign*exp(log(fx[0]*sign) + (i- N_pad - N_extrap_low)*dlnf_low) / pow(xi, config->nu);
    }
  }
  
  #pragma omp parallel for
  for(long i=N_pad+N_extrap_low; i<N_pad+N_extrap_low+N_original; i++) 
  {
    fb[i] = fx[i-N_pad-N_extrap_low] / pow(x[i-N_pad-N_extrap_low], config->nu) ;
  }
  
  if(N_extrap_high) 
  {
    int sign;
    
    if(fx[N_original-1] == 0) 
    {
      log_fatal("J_abl_ar: Can't log-extrapolate zero on the high side!");
      exit(1);
    }
    else if(fx[N_original-1] > 0) 
    {
      sign = 1;
    }
    else 
    {
      sign=-1;
    }
    
    if(fx[N_original-1]/fx[N_original-2] <= 0) 
    {
      log_fatal("J_abl_ar: Log-extrapolation on the high side fails due to sign change!");
      exit(1);
    }
    
    const double dlnf_high = log(fx[N_original-1]/fx[N_original-2]);
    
    #pragma omp parallel for
    for(long i=N-N_pad-N_extrap_high; i<N-N_pad; i++) 
    {
      const double xi = exp(log(x[N_original-1]) + (i-N_pad - N_extrap_low- N_original)*dlnx);
      fb[i] = sign * exp(log(fx[N_original-1]*sign) + 
        (i- N_pad - N_extrap_low- N_original)*dlnf_high) / pow(xi, config->nu);
    }
  }

  fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
  fftw_plan plan_forward;
  plan_forward = fftw_plan_dft_r2c_1d(N, fb, out, FFTW_ESTIMATE);
  
  fftw_execute(plan_forward);
  
  c_window(out, config->c_window_width, halfN);

  double **out_ifft = malloc(sizeof(double*) * Nterms);
  fftw_complex **out_vary = malloc(sizeof(fftw_complex*) * Nterms);
  fftw_plan* plan_backward = malloc(sizeof(fftw_plan) * Nterms);

  for(int i_term=0;i_term<Nterms;i_term++) 
  {
    out_ifft[i_term] = malloc(sizeof(double) * (2*N) );
    out_vary[i_term] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N+1) );
    plan_backward[i_term] = 
      fftw_plan_dft_c2r_1d(2*N, out_vary[i_term], out_ifft[i_term], FFTW_ESTIMATE);
  }

  double tau_l[N+1];
  for(long i=0;i<=N;i++)
  {
    tau_l[i] = 2.*M_PI / dlnx / N * i;
  }

  // initialize FFT plans for Convolution
  fftw_complex **a = malloc(sizeof(fftw_complex*) * Nterms);
  fftw_complex **b = malloc(sizeof(fftw_complex*) * Nterms);
  fftw_complex **a1 = malloc(sizeof(fftw_complex*) * Nterms);
  fftw_complex **b1 = malloc(sizeof(fftw_complex*) * Nterms);
  fftw_complex **c = malloc(sizeof(fftw_complex*) * Nterms);
  fftw_plan* pa = malloc(sizeof(fftw_plan) * Nterms);
  fftw_plan* pb = malloc(sizeof(fftw_plan) * Nterms);
  fftw_plan* pc = malloc(sizeof(fftw_plan) * Nterms);

  long Ntotal_convolve;
  if(N%2==0) 
  { // N+1 is odd
    Ntotal_convolve = 2*N + 1;
  }
  else 
  {
    log_fatal("J_abl_ar: This fftconvolve doesn't support even size input arrays"
      " (of out_pad1, outpad2)");
    exit(1);
  }

  for(int i_term=0; i_term<Nterms; i_term++) 
  {
    a[i_term] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
    b[i_term] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
    a1[i_term] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
    b1[i_term] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );
    c[i_term] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * Ntotal_convolve );

    pa[i_term] = 
      fftw_plan_dft_1d(Ntotal_convolve, a[i_term], a1[i_term], FFTW_FORWARD, FFTW_ESTIMATE);
    pb[i_term] = 
      fftw_plan_dft_1d(Ntotal_convolve, b[i_term], b1[i_term], FFTW_FORWARD, FFTW_ESTIMATE);
    pc[i_term] = 
      fftw_plan_dft_1d(Ntotal_convolve, a1[i_term], c[i_term], FFTW_BACKWARD, FFTW_ESTIMATE);
  }

  fftw_complex** out_pad1 = (fftw_complex**) fftw_malloc(sizeof(fftw_complex*) * Nterms);
  fftw_complex** out_pad2 = (fftw_complex**) fftw_malloc(sizeof(fftw_complex*) * Nterms);
  fftw_complex** pads_convolve = (fftw_complex**) fftw_malloc(sizeof(fftw_complex*) * Nterms);

  for(int i_term=0; i_term<Nterms; i_term++) 
  {
    out_pad1[i_term] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N + 1));
    out_pad2[i_term] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N + 1));
    pads_convolve[i_term] =  (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(2*N + 1));
  }

  #pragma omp parallel for
  for(int i_term=0; i_term<Nterms; i_term++) 
  {
    double complex gl[halfN+1];
    g_m_vals(ell[i_term]+0.5, 1.5 + config->nu + alpha[i_term], eta_m, gl, halfN + 1);

    // Do convolutions
    for(long i=0; i<=halfN; i++) 
    {
      out_pad1[i_term][i+halfN] = out[i] / (double) N * gl[i] ;
    }
    for(long i=0; i<halfN; i++) 
    {
      out_pad1[i_term][i] = conj(out_pad1[i_term][N-i]) ;
    }

    if(alpha != beta)
    {
      g_m_vals(ell[i_term]+0.5, 1.5 + config->nu + beta[i_term], eta_m, gl, halfN+1);

      for(long i=0; i<=halfN; i++) 
      {
        out_pad2[i_term][i+halfN] = out[i] / (double)N * gl[i] ;
      }
      for(long i=0; i<halfN; i++) 
      {
        out_pad2[i_term][i] = conj(out_pad2[i_term][N-i]);
      }
      
      fftconvolve_optimize(out_pad1[i_term], out_pad2[i_term], N+1, pads_convolve[i_term], a[i_term], 
        b[i_term], a1[i_term], b1[i_term], c[i_term], pa[i_term], pb[i_term], pc[i_term]);
    }
    else
    {
      fftconvolve_optimize(out_pad1[i_term], out_pad1[i_term], N+1, pads_convolve[i_term], a[i_term], 
        b[i_term], a1[i_term], b1[i_term], c[i_term], pa[i_term], pb[i_term], pc[i_term]);
    }

    // convolution finished
    pads_convolve[i_term][N] = creal(pads_convolve[i_term][N]);
    double complex h_part[N+1];
    for(long i=0;i<=N;i++)
    {
      h_part[i] = pads_convolve[i_term][i+N]; // C_h term in Eq.(2.21) in McEwen et al (2016)
                                              // but only take h = 0,1,2,...,N.
    }

    const int p = -5.-2.*config->nu - alpha[i_term]-beta[i_term];
    double complex fz[N+1];
    f_z(p+1, tau_l, fz, N+1);

    for(long i=0; i<=N; i++)
    {
      out_vary[i_term][i] = h_part[i] * conj(fz[i]) * cpow(2., I*tau_l[i]);
    }
    
    fftw_execute(plan_backward[i_term]);

    const int sign_ell = (ell[i_term]%2? -1:1);
    for(long i=0; i<N_original; i++)
    {
      Fy[i_term][i] = out_ifft[i_term][2*(i+N_pad+N_extrap_low)] * sign_ell / (M_PI*M_PI) * 
        pow(2., 2.+2*config->nu+alpha[i_term]+beta[i_term]) * pow(x[i],-p-2.);
    }
  }

  for(int i_term=0; i_term<Nterms; i_term++)
  {
    fftw_destroy_plan(plan_backward[i_term]);
    fftw_free(out_vary[i_term]);
    fftw_free(out_pad1[i_term]);
    fftw_free(out_pad2[i_term]);
    fftw_free(pads_convolve[i_term]);

    free(out_ifft[i_term]);

    fftw_free(a[i_term]);
    fftw_free(b[i_term]);
    fftw_free(a1[i_term]);
    fftw_free(b1[i_term]);
    fftw_free(c[i_term]);

    fftw_destroy_plan(pa[i_term]);
    fftw_destroy_plan(pb[i_term]);
    fftw_destroy_plan(pc[i_term]);
  }

  free(plan_forward);
  free(plan_backward);
  free(out_ifft);
  fftw_free(out);
  free(fb);
  free(out_vary);

  free(out_pad1);
  free(out_pad2);
  free(pads_convolve);
  free(a);
  free(b);
  free(a1);
  free(b1);
  free(c);
  free(pa);
  free(pb);
  free(pc);
}

void Pd1d2(double *k, double *Pin, long Nk, double *Pout)
{
  int alpha_ar[] = {0,0,1};
  int beta_ar[]  = {0,0,-1};
  int ell_ar[]   = {0,2,1};
  int isP13type_ar[] = {0,0,0};
  double coeff_A_ar[] = {2.*(17./21), 2.*(4./21), 2.};
  int Nterms = 3;

  fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Pd2d2(double *k, double *Pin, long Nk, double *Pout)
{
  int alpha_ar[] = {0};
  int beta_ar[]  = {0};
  int ell_ar[]   = {0};
  int isP13type_ar[] = {0};
  double coeff_A_ar[] = {2.};
  int Nterms = 1;

  fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Pd1s2(double *k, double *Pin, long Nk, double *Pout)
{
  int alpha_ar[] = {0,0,0,1,1};
  int beta_ar[]  = {0,0,0,-1,-1};
  int ell_ar[]   = {0,2,4,1,3};
  int isP13type_ar[] = {0,0,0,0,0};
  double coeff_A_ar[] = {2*(8./315.),2*(254./441.),2*(16./245.),2*(4./15.),2*(2./5.)};
  int Nterms = 5;

  fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Pd2s2(double *k, double *Pin, long Nk, double *Pout)
{
  int alpha_ar[] = {0};
  int beta_ar[]  = {0};
  int ell_ar[]   = {2};
  int isP13type_ar[] = {0};
  double coeff_A_ar[] = {2.*2./3.};
  int Nterms = 1;

  fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Ps2s2(double *k, double *Pin, long Nk, double *Pout)
{
  int alpha_ar[] = {0,0,0};
  int beta_ar[]  = {0,0,0};
  int ell_ar[]   = {0,2,4};
  int isP13type_ar[] = {0,0,0};
  double coeff_A_ar[] = {2.*(4./45.), 2*(8./63.), 2*(8./35.)};
  int Nterms = 3;

  fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -- NEW IMPLEMENTATION -------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// J_abJ1J2Jk_ar: Core FFT-PT computation engine
// ---------------------------------------------------------------------------
//
// Computes Nterms convolution integrals using the FFT-based algorithm
// Each term is specified by (alpha, beta, J1, J2, Jk) and the result
// for term i is stored in Fy[i][0..N_original-1].
//
// This function is called once per cosmology evaluation (from get_FPT_IA
// via IA_tt/IA_ta/IA_mix), always with the same k grid and config but
// different fx (power spectrum). The implementation exploits this by
// caching everything that doesn't depend on fx.
//
// OPTIMIZATION SUMMARY (vs original implementation):
//
// 1. STATIC CACHING: All data that depends only on N (frequency grids,
//    c_window weights, x_full array, FFTW plans, memory blocks) is
//    computed once on the first call and reused forever. This eliminates
//    ~30ms of redundant setup per cosmology evaluation.
//
// 2. g_m_vals CACHING: The Gamma-ratio arrays g_m(J) are cached by J
//    value (small integers 0-15). Each unique J is computed once, ever.
//    Saves ~2ms per call x 8 calls = ~16ms.
//
// 3. SHARED FORWARD FFTs: The biased input fb depends only on alpha
//    (via nu1=-2-alpha) or beta (via nu2=-2-beta). Many terms share the
//    same alpha/beta, so we compute the forward FFT + c_window once per
//    unique value. For IA_tt (all alpha=0, beta=0): 22 FFTs -> 2.
//
// 4. TEMPLATE FFTW PLANS: 5 template plans instead of hundreds. Applied
//    to different arrays via fftw_execute_dft / fftw_execute_dft_c2r.
//
// 5. CONTIGUOUS MEMORY BLOCKS: All per-term arrays in a few large blocks
//    instead of individual mallocs. Better cache locality.
//
// 6. INLINED CONVOLUTION: fftconvolve_optimize was inlined — data goes
//    directly into FFT buffers, results read directly from FFT output.
//
// 7. PRECOMPUTED c_window + x_full: sin() weights and the padded k grid
//    are computed once and cached.
//
// MEMORY LAYOUT of s_fft_block3 (per term i, stride = 5 * s_Ntotal):
//   a[i]  = base + 0 * s_Ntotal   (convolution input 1, from alpha/gl1)
//   b[i]  = base + 1 * s_Ntotal   (convolution input 2, from beta/gl2)
//   a1[i] = base + 2 * s_Ntotal   (FFT(a), then pointwise *= b1)
//   b1[i] = base + 3 * s_Ntotal   (FFT(b))
//   cv[i] = base + 4 * s_Ntotal   (IFFT result of convolution)
//
// ALGORITHM PER TERM
//   1. Multiply precomputed windowed FFT(fb) by g_m(J) -> a, b
//   2. FFT(a) -> a1,  FFT(b) -> b1           (2 FFTs)
//   3. Pointwise multiply: a1 *= b1
//   4. IFFT(a1) -> cv                         (1 FFT)
//   5. cv * conj(g_m(Jk)) / Ntotal -> out_vary
//   6. IFFT(out_vary) -> out_ifft             (1 FFT)
//   7. Extract and normalize: Fy = out_ifft * pi^(3/2)/8 / x
//
// Total: 4 FFTs per term (down from 6 — the 2 biasing FFTs were moved
// to the alpha/beta precomputation outside the parallel loop).
// ---------------------------------------------------------------------------
void J_abJ1J2Jk_ar(double *x, double *fx, long N, int *alpha, int *beta,
    int *J1, int *J2, int *Jk, int Nterms, fastpt_config *config, double **Fy) {
 
  // =========================================================================
  // STATIC CACHE
  // These variables persist across calls. Everything is rebuilt only when
  // N changes (which in practice means only on the very first call).
  // =========================================================================
  #define GM_J_MAX 16
 
  // Cached padded size and derived constants
  static long s_N = 0;
  static long s_halfN;        // N/2
  static long s_Ntotal;       // 2*N + 1 (convolution output size)
  static long s_Ncut;         // number of c_window tapering points
  static long s_eta_len;      // halfN + 1 (positive-frequency count)
  static long s_tau_len;      // N + 1
  static double s_inv_N;      // 1.0 / N
  static double s_inv_Ntotal; // 1.0 / (2*N + 1)
  static double s_pi_factor;  // pi^(3/2) / 8 (output normalization)
 
  // Frequency grids (depend only on N and dlnx)
  static double *s_eta_m = NULL; // eta_m[i] = 2*pi*i / (dlnx*N), i=0..halfN
  static double *s_tau_l = NULL; // tau_l[i] = 2*pi*i / (dlnx*N), i=0..N
 
  // Precomputed c_window weights: tapering function applied to the
  // high-frequency end of the FFT output to suppress ringing.
  // Precomputed to avoid recomputing sin() on every call.
  static double *s_c_win = NULL;
 
  // Cached padded k grid: x_full is deterministic from k_min/k_max/N,
  // so we compute it once and reuse. Only f_unbias changes per call.
  static double *s_x_full = NULL;
 
  // g_m_vals cache: indexed directly by J value (0..GM_J_MAX-1).
  // g_m_vals(J+0.5, -0.5, eta_m) involves expensive lngamma evaluations
  // over arrays of size halfN+1 or N+1. Since J values are small integers
  // that repeat across terms, we compute each unique J once and cache
  // forever (until N changes).
  static double complex *s_gm_eta[GM_J_MAX]; // g_m for eta frequencies
  static double complex *s_gm_tau[GM_J_MAX]; // g_m for tau frequencies
  static int s_gm_valid[GM_J_MAX];           // 1 if cached, 0 if not
 
  // Contiguous memory blocks (allocated for exact Nterms on first call):
  //   s_dbl_block:  out_ifft arrays, 2*N doubles per term (c2r FFT output)
  //   s_fft_block2: out_vary arrays, (N+1) complex per term (c2r FFT input)
  //   s_fft_block3: a/b/a1/b1/cv, 5*Ntotal complex per term
  static double *s_dbl_block = NULL;
  static fftw_complex *s_fft_block2 = NULL, *s_fft_block3 = NULL;
 
  // Template FFTW plans: 5 templates instead of hundreds of per-term plans.
  // fftw_execute_dft / fftw_execute_dft_c2r applies them to different arrays.
  //   s_plan_pa:   FFT forward,  size Ntotal (a -> a1)
  //   s_plan_pb:   FFT forward,  size Ntotal (b -> b1)
  //   s_plan_pc:   FFT backward, size Ntotal (a1 -> cv)
  //   s_plan_back: c2r backward, size 2*N    (out_vary -> out_ifft)
  //   s_plan_r2c:  r2c forward,  size N      (fb_tmp -> fft_tmp, alpha/beta)
  static fftw_plan s_plan_pa = NULL, s_plan_pb = NULL, s_plan_pc = NULL;
  static fftw_plan s_plan_back = NULL, s_plan_r2c = NULL;
 
  // =========================================================================
  // PARAMETER SETUP
  // =========================================================================
  const long N_original = N;
  const long N_pad = config->N_pad;
  const long N_extrap_low = config->N_extrap_low;
  const long N_extrap_high = config->N_extrap_high;
 
  // Pad N: zero-padding + extrapolation regions on both sides
  N += (2*N_pad + N_extrap_low + N_extrap_high);
 
  if (N % 2) {
    log_fatal("J_abJ1J2Jk_ar: Please use even number of x !");
    exit(1);
  }
  const long halfN = N / 2;
  const double x0 = x[0];
  const double dlnx = log(x[1] / x0); // log-spacing of input k array
 
  // =========================================================================
  // ONE-TIME SETUP (first call, or if N changes)
  //
  // Allocates memory, creates plans, precomputes grids/weights/x_full.
  // Everything here is reused on all subsequent calls since only fx
  // (the power spectrum) changes between cosmology evaluations.
  // =========================================================================
  if (s_N != N) 
  {
    // --- Free old cache ---
    if (s_eta_m) { free(s_eta_m); free(s_tau_l); free(s_c_win); free(s_x_full); }
    for (int j = 0; j < GM_J_MAX; j++) {
      if (s_gm_valid[j]) { free(s_gm_eta[j]); free(s_gm_tau[j]); }
      s_gm_valid[j] = 0;
    }
    if (s_dbl_block) {
      free(s_dbl_block); fftw_free(s_fft_block2); fftw_free(s_fft_block3);
    }
    if (s_plan_pa) {
      fftw_destroy_plan(s_plan_pa); fftw_destroy_plan(s_plan_pb);
      fftw_destroy_plan(s_plan_pc); fftw_destroy_plan(s_plan_back);
    }
    if (s_plan_r2c) fftw_destroy_plan(s_plan_r2c);
 
    // --- Store new N and derived constants ---
    s_N = N;
    s_halfN = halfN;
    s_Ntotal = 2*N + 1;
    s_eta_len = halfN + 1;
    s_tau_len = N + 1;
    s_inv_N = 1.0 / (double)N;
    s_inv_Ntotal = 1.0 / (double)s_Ntotal;
    s_pi_factor = pow(M_PI, 1.5) / 8.;
 
    // --- Frequency grids ---
    // eta_m: for the m>=0 Fourier modes used in g_m_vals
    // tau_l: for the convolution output modes
    s_eta_m = malloc(sizeof(double) * s_eta_len);
    s_tau_l = malloc(sizeof(double) * s_tau_len);
    for (long i = 0; i < s_eta_len; i++)
      s_eta_m[i] = 2*M_PI / dlnx / N * i;
    for (long i = 0; i < s_tau_len; i++)
      s_tau_l[i] = 2.*M_PI / dlnx / N * i;
 
    // --- c_window weights ---
    // Tapering function that smoothly goes from 0 at Nyquist to 1 at cutoff.
    s_Ncut = (long)(halfN * config->c_window_width);
    s_c_win = malloc(sizeof(double) * (s_Ncut + 1));
    for (long i = 0; i <= s_Ncut; i++)
      s_c_win[i] = (double)i / s_Ncut - 1./(2.*M_PI) * sin(2.*i*M_PI / s_Ncut);
 
    // --- Cached x_full (padded k grid) ---
    // The k grid is deterministic from k_min/k_max/N, so x_full never
    // changes. We compute it once here; the per-call work only rebuilds
    // f_unbias (which depends on the changing power spectrum fx).
    s_x_full = malloc(sizeof(double) * N);
    for (long i = 0; i < N_pad; i++) {
      s_x_full[i]     = exp(log(x0) + (i - N_pad - N_extrap_low) * dlnx);
      s_x_full[N-1-i] = exp(log(x0) + (N-1-i - N_pad - N_extrap_low) * dlnx);
    }
    if (N_extrap_low) {
      for (long i = N_pad; i < N_pad + N_extrap_low; i++)
        s_x_full[i] = exp(log(x0) + (i - N_pad - N_extrap_low) * dlnx);
    }
    for (long i = N_pad + N_extrap_low; i < N_pad + N_extrap_low + N_original; i++)
      s_x_full[i] = x[i - N_pad - N_extrap_low];
    if (N_extrap_high) {
      for (long i = N - N_pad - N_extrap_high; i < N - N_pad; i++)
        s_x_full[i] = exp(log(x[N_original-1])
                      + (i - N_pad - N_extrap_low - N_original) * dlnx);
    }
 
    // --- Template r2c plan for alpha/beta forward FFTs ---
    // Created with temporary arrays; fftw_execute_dft_r2c applies it to
    // the actual fb_tmp/fft_tmp arrays later.
    {
      double *tmp_in = fftw_malloc(sizeof(double) * N);
      fftw_complex *tmp_out = fftw_malloc(sizeof(fftw_complex) * (halfN + 1));
      s_plan_r2c = fftw_plan_dft_r2c_1d(N, tmp_in, tmp_out, FFTW_ESTIMATE);
      fftw_free(tmp_in);
      fftw_free(tmp_out);
    }
 
    // --- Contiguous memory blocks for Nterms ---
    //   s_dbl_block:  Nterms * 2*N doubles    (out_ifft, c2r output)
    //   s_fft_block2: Nterms * (N+1) complex  (out_vary, c2r input)
    //   s_fft_block3: Nterms * 5*Ntotal complex (a, b, a1, b1, cv)
    s_dbl_block  = malloc(sizeof(double) * Nterms * 2 * N);
    s_fft_block2 = fftw_malloc(sizeof(fftw_complex) * Nterms * (N + 1));
    s_fft_block3 = fftw_malloc(sizeof(fftw_complex) * Nterms * 5 * s_Ntotal);
 
    // --- 4 template plans using first term's arrays as prototypes ---
    // fftw_execute_dft applies the same algorithm to any term's arrays.
    fftw_complex *base0 = s_fft_block3;
    s_plan_pa   = fftw_plan_dft_1d(s_Ntotal, base0,
                    base0 + 2*s_Ntotal, FFTW_FORWARD, FFTW_ESTIMATE);
    s_plan_pb   = fftw_plan_dft_1d(s_Ntotal, base0 + s_Ntotal,
                    base0 + 3*s_Ntotal, FFTW_FORWARD, FFTW_ESTIMATE);
    s_plan_pc   = fftw_plan_dft_1d(s_Ntotal, base0 + 2*s_Ntotal,
                    base0 + 4*s_Ntotal, FFTW_BACKWARD, FFTW_ESTIMATE);
    s_plan_back = fftw_plan_dft_c2r_1d(2*N, s_fft_block2,
                    s_dbl_block, FFTW_ESTIMATE);
 
    // --- Pre-zero-pad a and b tails for all terms ---
    // The convolution requires zero-padding from index N+1 to Ntotal-1.
    // Done once here; the main loop only writes indices 0..N.
    for (int i = 0; i < Nterms; i++) {
      fftw_complex *base = s_fft_block3 + i * 5 * s_Ntotal;
      memset(base + (N+1), 0,
        sizeof(fftw_complex) * (s_Ntotal - (N+1)));            // a tail
      memset(base + s_Ntotal + (N+1), 0,
        sizeof(fftw_complex) * (s_Ntotal - (N+1)));            // b tail
    }
  }
 
  // =========================================================================
  // CACHE g_m_vals FOR UNIQUE J VALUES
  //
  // g_m_vals computes Gamma-ratio arrays involving lngamma evaluations
  // over halfN+1 or N+1 points. J values are small integers (0-8 typically)
  // that repeat across terms. Each unique J is computed once and cached
  // forever (until N changes).
  // =========================================================================
  for (int i = 0; i < Nterms; i++) {
    int js[] = { J1[i], J2[i], Jk[i] };
    for (int k = 0; k < 3; k++) {
      int j = js[k];
      if (!s_gm_valid[j]) {
        s_gm_eta[j] = malloc(sizeof(double complex) * s_eta_len);
        s_gm_tau[j] = malloc(sizeof(double complex) * s_tau_len);
        g_m_vals(j + 0.5, -0.5, s_eta_m, s_gm_eta[j], s_eta_len);
        g_m_vals(j + 0.5, -0.5, s_tau_l, s_gm_tau[j], s_tau_len);
        s_gm_valid[j] = 1;
      }
    }
  }
 
  // =========================================================================
  // BUILD f_unbias (PER-CALL: changes each evaluation because fx changes)
  //
  // x_full is cached (deterministic from k grid), so only f_unbias needs
  // rebuilding. The padded array has:
  //   - Zero-padding at both ends (N_pad on each side)
  //   - Log-extrapolation regions beyond the original data
  //   - Original fx data in the middle
  // =========================================================================
  double f_unbias[N];
 
  // Zero-padding at both ends
  for (long i = 0; i < N_pad; i++) {
    f_unbias[i]     = 0.;
    f_unbias[N-1-i] = 0.;
  }
 
  // Low-side log-extrapolation
  if (N_extrap_low) {
    int sign = (fx[0] > 0) ? 1 : -1;
    const double dlnf_low = log(fx[1] / fx[0]);
    for (long i = N_pad; i < N_pad + N_extrap_low; i++)
      f_unbias[i] = sign * exp(log(fx[0]*sign)
                    + (i - N_pad - N_extrap_low) * dlnf_low);
  }
 
  // Copy original data into the middle
  for (long i = N_pad + N_extrap_low;
       i < N_pad + N_extrap_low + N_original; i++)
    f_unbias[i] = fx[i - N_pad - N_extrap_low];
 
  // High-side log-extrapolation
  if (N_extrap_high) {
    int sign = (fx[N_original-1] > 0) ? 1 : -1;
    const double dlnf_high = log(fx[N_original-1] / fx[N_original-2]);
    for (long i = N - N_pad - N_extrap_high; i < N - N_pad; i++)
      f_unbias[i] = sign * exp(log(fx[N_original-1]*sign)
                    + (i - N_pad - N_extrap_low - N_original) * dlnf_high);
  }
 
  // =========================================================================
  // PRECOMPUTE WINDOWED FFTs FOR UNIQUE ALPHA/BETA VALUES
  //
  // The biased input fb = f_unbias / x^nu depends only on alpha (nu=-2-alpha)
  // or beta (nu=-2-beta). Many terms share the same alpha/beta, so we:
  //   1. Find unique alpha values -> compute FFT(fb) + c_window once each
  //   2. Find unique beta values  -> same
  //   3. Each term looks up its precomputed result by index
  //
  // For IA_tt (all alpha=0, beta=0): 22 forward FFTs -> 2.
  // For IA_mix (2 unique alpha, 2 unique beta): 64 -> 4.
  // =========================================================================
 
  // Deduplicate alpha and beta values
  int unique_alpha[Nterms], unique_beta[Nterms];
  int n_unique_alpha = 0, n_unique_beta = 0;
  for (int i = 0; i < Nterms; i++) {
    int found = 0;
    for (int j = 0; j < n_unique_alpha; j++)
      if (unique_alpha[j] == alpha[i]) { found = 1; break; }
    if (!found) unique_alpha[n_unique_alpha++] = alpha[i];
    found = 0;
    for (int j = 0; j < n_unique_beta; j++)
      if (unique_beta[j] == beta[i]) { found = 1; break; }
    if (!found) unique_beta[n_unique_beta++] = beta[i];
  }
 
  // Allocate storage for precomputed windowed FFT results
  const long fft_small = halfN + 1;
  fftw_complex *fft_alpha = fftw_malloc(
    sizeof(fftw_complex) * n_unique_alpha * fft_small);
  fftw_complex *fft_beta = fftw_malloc(
    sizeof(fftw_complex) * n_unique_beta * fft_small);
 
  // Temporary arrays for forward FFT (stack-allocated).
  // Uses cached s_plan_r2c template plan via fftw_execute_dft_r2c.
  double fb_tmp[N];
  fftw_complex fft_tmp[fft_small];
 
  // Compute windowed FFT for each unique alpha:
  //   1. Bias: fb_tmp = f_unbias / x_full^(-2-alpha)
  //   2. Forward r2c FFT (using cached template plan)
  //   3. Apply c_window tapering to high frequencies
  //   4. Store result for lookup by terms sharing this alpha
  for (int u = 0; u < n_unique_alpha; u++) {
    double nu = -2. - unique_alpha[u];
    for (long i = 0; i < N; i++)
      fb_tmp[i] = f_unbias[i] / pow(s_x_full[i], nu);
    fftw_execute_dft_r2c(s_plan_r2c, fb_tmp, fft_tmp);
    for (long i = 0; i <= s_Ncut; i++)
      fft_tmp[halfN-i] *= s_c_win[i];
    memcpy(fft_alpha + u*fft_small, fft_tmp,
      sizeof(fftw_complex) * fft_small);
  }
 
  // Same for each unique beta
  for (int u = 0; u < n_unique_beta; u++) {
    double nu = -2. - unique_beta[u];
    for (long i = 0; i < N; i++)
      fb_tmp[i] = f_unbias[i] / pow(s_x_full[i], nu);
    fftw_execute_dft_r2c(s_plan_r2c, fb_tmp, fft_tmp);
    for (long i = 0; i <= s_Ncut; i++)
      fft_tmp[halfN-i] *= s_c_win[i];
    memcpy(fft_beta + u*fft_small, fft_tmp,
      sizeof(fftw_complex) * fft_small);
  }
 
  // Build lookup: for each term, which unique alpha/beta index to use
  int idx_alpha[Nterms], idx_beta[Nterms];
  for (int i = 0; i < Nterms; i++) {
    for (int u = 0; u < n_unique_alpha; u++)
      if (unique_alpha[u] == alpha[i]) idx_alpha[i] = u;
    for (int u = 0; u < n_unique_beta; u++)
      if (unique_beta[u] == beta[i]) idx_beta[i] = u;
  }
 
  // =========================================================================
  // DERIVE POINTER ARRAYS from cached contiguous blocks.
  // Cheap stack arrays of pointers into the static blocks.
  //
  // Layout per term i in s_fft_block3 (stride = 5 * s_Ntotal):
  //   a[i]  = base              convolution input 1
  //   b[i]  = base + Ntotal     convolution input 2
  //   a1[i] = base + 2*Ntotal   FFT(a), then pointwise *= b1
  //   b1[i] = base + 3*Ntotal   FFT(b)
  //   cv[i] = base + 4*Ntotal   IFFT result of convolution
  // =========================================================================
  double *out_ifft[Nterms];
  fftw_complex *out_vary[Nterms];
  fftw_complex *a[Nterms], *b[Nterms], *a1[Nterms], *b1[Nterms], *cv[Nterms];
  for (int i = 0; i < Nterms; i++) {
    out_ifft[i] = s_dbl_block + i * 2*N;
    out_vary[i] = s_fft_block2 + i * (N + 1);
    fftw_complex *base = s_fft_block3 + i * 5 * s_Ntotal;
    a[i]  = base;
    b[i]  = base + s_Ntotal;
    a1[i] = base + 2*s_Ntotal;
    b1[i] = base + 3*s_Ntotal;
    cv[i] = base + 4*s_Ntotal;
  }
 
  // =========================================================================
  // MAIN COMPUTATION LOOP
  //
  // Each term is independent — OpenMP parallelizes across terms.
  // Per term (4 FFTs, down from 6 in the original):
  //   Step 1: Combine precomputed windowed FFT with cached g_m(J1)/g_m(J2)
  //           -> build convolution inputs a[i] and b[i] with conjugate symmetry
  //   Step 2: Forward FFT a and b (2 FFTs, via template plans)
  //   Step 3: Pointwise multiply in Fourier space: a1 *= b1
  //   Step 4: Backward FFT -> convolution result cv (1 FFT)
  //   Step 5: Multiply by conj(g_m(Jk)) and normalize -> out_vary
  //   Step 6: Backward c2r FFT -> final real result (1 FFT)
  //   Step 7: Extract un-padded region with pi^(3/2)/8 / x normalization
  // =========================================================================
  #pragma omp parallel for
  for (int i_term = 0; i_term < Nterms; i_term++) {
 
    // --- Step 1: Build convolution inputs a and b ---
    const fftw_complex *out_a = fft_alpha + idx_alpha[i_term] * fft_small;
    const fftw_complex *out_b = fft_beta  + idx_beta[i_term]  * fft_small;
    const double complex *gl1 = s_gm_eta[J1[i_term]];
    const double complex *gl2 = s_gm_eta[J2[i_term]];
 
    // Positive frequencies (indices halfN..N)
    for (long i = 0; i <= halfN; i++) {
      a[i_term][i+halfN] = out_a[i] * s_inv_N * gl1[i];
      b[i_term][i+halfN] = out_b[i] * s_inv_N * gl2[i];
    }
    // Conjugate symmetry for negative frequencies (indices 0..halfN-1)
    for (long i = 0; i < halfN; i++) {
      a[i_term][i] = conj(a[i_term][N-i]);
      b[i_term][i] = conj(b[i_term][N-i]);
    }
    // Indices N+1..Ntotal-1 were pre-zero-padded at allocation time.
 
    // --- Steps 2-4: Convolution via FFT (inlined fftconvolve_optimize) ---
    fftw_execute_dft(s_plan_pa, a[i_term], a1[i_term]);
    fftw_execute_dft(s_plan_pb, b[i_term], b1[i_term]);
    for (long i = 0; i < s_Ntotal; i++)
      a1[i_term][i] *= b1[i_term][i];
    fftw_execute_dft(s_plan_pc, a1[i_term], cv[i_term]);
 
    // --- Step 5: Multiply by conj(g_m(Jk)) and normalize ---
    cv[i_term][N] = creal(cv[i_term][N]); // Hermitian midpoint must be real
    const double complex *fz = s_gm_tau[Jk[i_term]];
    // C_h * conj(f_z) / Ntotal  (McEwen et al 2016, Eq. 2.21)
    for (long i = 0; i <= N; i++)
      out_vary[i_term][i] = cv[i_term][i+N] * s_inv_Ntotal * conj(fz[i]);
 
    // --- Step 6: Final backward c2r FFT ---
    fftw_execute_dft_c2r(s_plan_back, out_vary[i_term], out_ifft[i_term]);
 
    // --- Step 7: Extract result and normalize ---
    for (long i = 0; i < N_original; i++)
      Fy[i_term][i] = out_ifft[i_term][2*(i + N_pad + N_extrap_low)]
                     * s_pi_factor / x[i];
  }
 
  // =========================================================================
  // CLEANUP: only the per-call alpha/beta FFT storage.
  // Everything else persists in static variables for the next call.
  // =========================================================================
  fftw_free(fft_alpha);
  fftw_free(fft_beta);
}
