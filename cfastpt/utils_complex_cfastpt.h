#include <complex.h>
#include <fftw3.h>

#ifndef __CFASTPT_UTILS_COMPLEX_CFASTPT_H
#define __CFASTPT_UTILS_COMPLEX_CFASTPT_H
#ifdef __cplusplus
extern "C" {
#endif

void f_z(double z_real, double *z_imag, double complex *fz, long N);

void c_window(double complex *out, double c_window_width, long halfN);

void gamma_ratios(double l, double nu, double *eta, double complex *gl, long N);

void g_m_vals(double mu, double q_real, double *q_imag, double complex *gm, long N);

void fftconvolve_real(double *in1, double *in2, long N1, long N2, double *out);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
