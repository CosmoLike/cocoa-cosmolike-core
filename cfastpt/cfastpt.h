#ifndef __CFASTPT_CFASTPT_H
#define __CFASTPT_CFASTPT_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct fastpt_config 
{
  double nu; // only used in scalar; in tensor, nu1,nu2 are computed by alpha,beta
  double c_window_width;
  long N_pad;
  long N_extrap_low;
  long N_extrap_high;
} fastpt_config;

void J_abl(
    const double *restrict x,      // input k grid, length N (log-spaced wavenumbers)
    const double *restrict fx,     // input power spectrum P(k), length N
    long N,                        // number of input k points (before padding)
    const int *restrict alpha,     // biasing exponent per term: nu1 = 1.5 + nu + alpha[i]
    const int *restrict beta,      // biasing exponent per term: nu2 = 1.5 + nu + beta[i]
    const int *restrict ell,       // angular momentum per term (half-integer ell+0.5 in g_m)
    const int *restrict isP13type, // unused, kept for API compatibility
    int Nterms,                    // number of terms to compute
    const fastpt_config *restrict config, // padding/windowing config
    double **restrict Fy           // output: Fy[i][j] = result for term i at k-point j
  );

void J_abJ1J2Jk(
    const double *restrict x,      // input k grid, length N (log-spaced wavenumbers)
    const double *restrict fx,     // input power spectrum P(k), length N (changes each call)
    long N,                        // number of input k points (before padding)
    const int *restrict alpha,     // biasing exponent 1 per term: nu1 = -2 - alpha[i]
    const int *restrict beta,      // biasing exponent 2 per term: nu2 = -2 - beta[i]
    const int *restrict J1,        // angular momentum coupling 1 per term (indexes g_m cache)
    const int *restrict J2,        // angular momentum coupling 2 per term (indexes g_m cache)
    const int *restrict Jk,        // angular momentum coupling 3 per term (indexes g_m cache)
    int Nterms,                    // number of terms to compute
    const fastpt_config *restrict config, // padding/windowing config (N_pad, c_window_width, etc.)
    double **restrict Fy           // output: Fy[i][j] = result for term i at k-point j
  );

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
