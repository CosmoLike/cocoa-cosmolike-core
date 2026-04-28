#include <fftw3.h>
#ifndef __CFFTLOG_CFFTLOG_H
#define __CFFTLOG_CFFTLOG_H
#ifdef __cplusplus
extern "C" {
#endif

typedef struct config 
{
  double nu;
  double c_window_width;
  int derivative;
  long N_pad;
  long N_extrap_low;
  long N_extrap_high;
} config;

void cfftlog(double* x, double* fx, long N, config* config, int ell, double* y, double* Fy);

void cfftlog_ells(double* x, double* fx, long N, config* config, int* ell, long Nell, 
double** y, double** Fy);

void cfftlog_ells_increment(double* x, double* fx, long N, config* config, int* ell, long Nell, 
double** y, double** Fy);

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// begin of new experimental version
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

void cfftlog_ells_cocoa0(
  double* const x,
  double* const* const* const fx,
  int const Nx,
  config* const cfg,
  fftw_complex* const* const toutfwd,
  double* const* const eta_m,
  int const N[][3],
  int const Nmax,
  int const SIZE1,
  int const SIZE2
);

/*
void cfftlog_ells_cocoa( // 1D array: 1st dim: zbins, 2nd dim: three FFT per bin 
    double* const x,      // assume all fx have the same x-array
    double* const* const* const fx,
    int const Nx,                    // assume all bins and FFTs have the same N's
    config* const cfg,           // assume all guns have the same config
    int* const* const ell,
    int* const LMAX,           
    double* const* const* const y, 
    double* const* const* const* const Fy,
    int const SIZE1,
    int const SIZE2
  ); 
*/

void cfftlog_ells_cocoa(
    double* const x,
    int const Nx,
    config* const cfg,
    const int* const ell,
    int const LMAX,
    double* const* const* const y,
    double* const* const* const* const Fy,
    fftw_complex* const* const toutfwd,
    double* const* const eta_m,
    int const N[][3],
    int const Nmax,
    int const ks, // k start
    int const ke, // k end
    int const SIZE1,
    int const SIZE2
  );

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
