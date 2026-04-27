#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <string.h>
#include <time.h>
#include <fftw3.h>
#include <gsl/gsl_math.h>

#include "cosmolike/basics.h"
#include "cfftlog.h"
#include "utils.h"
#include "utils_complex.h"

#include "log.c/src/log.h"

void cfftlog(double *x, double *fx, long N, config *config, int ell, double *y, double *Fy) {
	long N_original = N;
	long N_pad = config->N_pad;
	N += 2*N_pad;

	if(N % 2) {
		printf("Please use even number of x !\n"); 
		exit(0);
	}
	long halfN = N/2;

	double x0, y0;
	x0 = x[0];

	double dlnx;
	dlnx = log(x[1]/x0);

	// Only calculate the m>=0 part
	double eta_m[halfN+1];
	long i;
	for(i=0; i<=halfN; i++) {eta_m[i] = 2*M_PI / dlnx / N * i;}

	double complex gl[halfN+1];

	switch(config->derivative) {
		case 0: g_l_cfft((double)ell, config->nu, eta_m, gl, halfN+1); break;
		case 1: g_l_1_cfft((double)ell, config->nu, eta_m, gl, halfN+1); break;
		case 2: g_l_2_cfft((double)ell, config->nu, eta_m, gl, halfN+1); break;
		default: printf("Integral Not Supported! Please choose config->derivative from [0,1,2].\n");
	}

	// calculate y arrays
	for(i=0; i<N_original; i++) {
		y[i] = (ell+1.) / x[N_original-1-i];
	}
	y0 = y[0];

	// biased input func
	double *fb;
	fb = malloc(N* sizeof(double));
	for(i=0; i<N_pad; i++) {
		fb[i] = 0.;
		fb[N-1-i] = 0.;
	}
	for(i=N_pad; i<N_pad+N_original; i++) {
		fb[i] = fx[i-N_pad] / pow(x[i-N_pad], config->nu) ;
	}

	fftw_complex *out;
	fftw_plan plan_forward, plan_backward;
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	plan_forward = fftw_plan_dft_r2c_1d(N, fb, out, FFTW_ESTIMATE);

	fftw_execute(plan_forward);

	c_window_cfft(out, config->c_window_width, halfN);

	for(i=0; i<=halfN; i++) {
		out[i] *= cpow(x0*y0/exp(2*N_pad*dlnx), -I*eta_m[i]) * gl[i] ;
		out[i] = conj(out[i]);
	}

	double *out_ifft;
	out_ifft = malloc(sizeof(double) * N );
	plan_backward = fftw_plan_dft_c2r_1d(N, out, out_ifft, FFTW_ESTIMATE);

	fftw_execute(plan_backward);

	for(i=0; i<N_original; i++) {
		Fy[i] = out_ifft[i-N_pad] * sqrt(M_PI) / (4.*N * pow(y[i], config->nu));
	}

	fftw_destroy_plan(plan_forward);
	fftw_destroy_plan(plan_backward);
	fftw_free(out);
	free(out_ifft);
	free(fb);
}

void cfftlog_ells(double *x, double *fx, long N, config *config, int* ell, long Nell, double **y, double **Fy) {
	long N_original = N;
	long N_pad = config->N_pad;
	long N_extrap_low = config->N_extrap_low;
	long N_extrap_high = config->N_extrap_high;
	N += (2*N_pad + N_extrap_low+N_extrap_high);

	if(N % 2) {
		printf("Please use even number of x !\n");
		exit(0);
	}
	long halfN = N/2;

	const double x0 = x[0];
	const double dlnx = log(x[1]/x0);

	// Only calculate the m>=0 part
	double eta_m[halfN+1];
	for(int i=0; i<=halfN; i++) {
		eta_m[i] = 2*M_PI / dlnx / N * i;
	}

	// biased input func
	double *fb = malloc(N* sizeof(double));
	for(int i=0; i<N_pad; i++) {
		fb[i] = 0.;
		fb[N-1-i] = 0.;
	}
	double xi;
	int sign;
	if(N_extrap_low) {
		if(fx[0]==0) {
			printf("Can't log-extrapolate zero on the low side!\n");
			exit(1);
		}
		else if(fx[0]>0) {
			sign = 1;
		}
		else {
			sign=-1;
		}
		if(fx[1]/fx[0]<=0) {
			printf("Log-extrapolation on the low side fails due to sign change!\n");
			exit(1);
		}
		double dlnf_low = log(fx[1]/fx[0]);
		for(int i=N_pad; i<N_pad+N_extrap_low; i++) {
			xi = exp(log(x0) + (i-N_pad - N_extrap_low)*dlnx);
			fb[i] = sign * exp(log(fx[0]*sign) + (i- N_pad - N_extrap_low)*dlnf_low) / pow(xi, config->nu);
		}
	}
	for(int i=N_pad+N_extrap_low; i<N_pad+N_extrap_low+N_original; i++) {
		fb[i] = fx[i-N_pad-N_extrap_low] / pow(x[i-N_pad-N_extrap_low], config->nu) ;
	}
	if(N_extrap_high) {
		if(fx[N_original-1]==0) {
			printf("Can't log-extrapolate zero on the high side!\n");
			exit(1);
		}
		else if(fx[N_original-1]>0) {
			sign = 1;
		}
		else {
			sign=-1;
		}
		if(fx[N_original-1]/fx[N_original-2]<=0) {
			printf("Log-extrapolation on the high side fails due to sign change!\n");
			exit(1);
		}
		double dlnf_high = log(fx[N_original-1]/fx[N_original-2]);
		for(int i=N-N_pad-N_extrap_high; i<N-N_pad; i++) {
			xi = exp(log(x[N_original-1]) + (i-N_pad - N_extrap_low- N_original)*dlnx);
			fb[i] = sign * exp(log(fx[N_original-1]*sign) + (i- N_pad - N_extrap_low- N_original)*dlnf_high)/pow(xi, config->nu);
		}
	}

	fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	fftw_plan plan_forward;

	plan_forward = fftw_plan_dft_r2c_1d(N, fb, out, FFTW_ESTIMATE);
	fftw_execute(plan_forward);
	c_window_cfft(out, config->c_window_width, halfN);


	double **out_ifft = malloc(sizeof(double*) * Nell);
	fftw_complex **out_vary = malloc(sizeof(fftw_complex*) * Nell);
	fftw_plan* plan_backward = malloc(sizeof(fftw_plan*) * Nell);
	for(int j=0; j<Nell; j++) {
		out_ifft[j] = malloc(sizeof(double) * N);
		out_vary[j] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
		plan_backward[j] = fftw_plan_dft_c2r_1d(N, out_vary[j], out_ifft[j], FFTW_ESTIMATE);
	}

	#pragma omp parallel for
	for (int j=0; j<Nell; j++) {
		double complex gl[halfN+1];

		switch(config->derivative) {
			case 0: g_l_cfft((double)ell[j], config->nu, eta_m, gl, halfN+1); break;
			case 1: g_l_1_cfft((double)ell[j], config->nu, eta_m, gl, halfN+1); break;
			case 2: g_l_2_cfft((double)ell[j], config->nu, eta_m, gl, halfN+1); break;
			default: printf("Integral Not Supported! Please choose config->derivative from [0,1,2].\n");
		}

		// calculate y arrays
		for(int i=0; i<N_original; i++) {
			y[j][i] = (ell[j]+1.) / x[N_original-1-i];
		}
		const double y0 = y[j][0];

		for(int i=0; i<=halfN; i++) {
			out_vary[j][i] = conj(out[i] * cpow(x0*y0/exp((N-N_original)*dlnx), -I*eta_m[i]) * gl[i]) ;
		}

		fftw_execute(plan_backward[j]);

		for(int i=0; i<N_original; i++) {
			Fy[j][i] = out_ifft[j][i+N_pad+N_extrap_high] * sqrt(M_PI) / (4.*N * pow(y[j][i], config->nu));
		}
	}

	for (int j=0; j<Nell; j++) {
		fftw_destroy_plan(plan_backward[j]);
		fftw_free(out_vary[j]);
		free(out_ifft[j]);
	}
	free(plan_backward);
	free(out_vary);
	free(out_ifft);
	fftw_destroy_plan(plan_forward);
	fftw_free(out);
	free(fb);
}

void cfftlog_ells_increment(double *x, double *fx, long N, config *config, int* ell, long Nell, double **y, double **Fy) {

	long N_original = N;
	long N_pad = config->N_pad;
	long N_extrap_low = config->N_extrap_low;
	long N_extrap_high = config->N_extrap_high;
	N += (2*N_pad + N_extrap_low+N_extrap_high);

	if(N % 2) {
		printf("Please use even number of x !\n"); exit(0);
	}
	long halfN = N/2;

	double x0;
	x0 = x[0];

	double dlnx;
	dlnx = log(x[1]/x0);

	// Only calculate the m>=0 part
	double eta_m[halfN+1];
	for(int i=0; i<=halfN; i++) {
		eta_m[i] = 2*M_PI / dlnx / N * i;
	}

	// biased input func
	double *fb;
	fb = malloc(N* sizeof(double));
	for(int i=0; i<N_pad; i++) {
		fb[i] = 0.;
		fb[N-1-i] = 0.;
	}
	double xi;
	int sign;
	if(N_extrap_low) {
		if(fx[0]==0) {
			printf("Can't log-extrapolate zero on the low side!\n");
			exit(1);
		}
		else if(fx[0]>0) {sign = 1;}
		else {sign=-1;}
		if(fx[1]/fx[0]<=0) {printf("Log-extrapolation on the low side fails due to sign change!\n"); exit(1);}
		double dlnf_low = log(fx[1]/fx[0]);
		for(int i=N_pad; i<N_pad+N_extrap_low; i++) {
			xi = exp(log(x0) + (i-N_pad - N_extrap_low)*dlnx);
			fb[i] = sign * exp(log(fx[0]*sign) + (i- N_pad - N_extrap_low)*dlnf_low) / pow(xi, config->nu);
		}
	}
	for(int i=N_pad+N_extrap_low; i<N_pad+N_extrap_low+N_original; i++) {
		fb[i] = fx[i-N_pad-N_extrap_low] / pow(x[i-N_pad-N_extrap_low], config->nu) ;
	}
	if(N_extrap_high) {
		if(fx[N_original-1]==0) {
			printf("Can't log-extrapolate zero on the high side!\n");
			exit(1);
		}
		else if(fx[N_original-1]>0) {sign = 1;}
		else {sign=-1;}
		if(fx[N_original-1]/fx[N_original-2]<=0) {printf("Log-extrapolation on the high side fails due to sign change!\n"); exit(1);}
		double dlnf_high = log(fx[N_original-1]/fx[N_original-2]);
		for(int i=N-N_pad-N_extrap_high; i<N-N_pad; i++) {
			xi = exp(log(x[N_original-1]) + (i-N_pad - N_extrap_low- N_original)*dlnx);
			fb[i] = sign * exp(log(fx[N_original-1]*sign) + (i- N_pad - N_extrap_low- N_original)*dlnf_high) / pow(xi, config->nu);
		}
	}

	fftw_complex *out;
	fftw_plan plan_forward;
	out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (halfN+1) );
	plan_forward = fftw_plan_dft_r2c_1d(N, fb, out, FFTW_ESTIMATE);
	fftw_execute(plan_forward);

	c_window_cfft(out, config->c_window_width, halfN);

	double **out_ifft = malloc(sizeof(double*) * Nell);
	fftw_complex **out_vary = malloc(sizeof(fftw_complex*) * Nell);
	fftw_plan* plan_backward = malloc(sizeof(fftw_plan*) * Nell);
	for(int j=0; j<Nell; j++) {
		out_ifft[j] = malloc(sizeof(double) * N);
		out_vary[j] = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*(halfN+1));
		plan_backward[j] =
			fftw_plan_dft_c2r_1d(N, out_vary[j], out_ifft[j], FFTW_ESTIMATE);
	}

	#pragma omp parallel for
	for(int j=0; j<Nell; j++){
		double complex gl[halfN+1];
		switch(config->derivative) {
			case 0: g_l_cfft((double)ell[j], config->nu, eta_m, gl, halfN+1); break;
			case 1: g_l_1_cfft((double)ell[j], config->nu, eta_m, gl, halfN+1); break;
			case 2: g_l_2_cfft((double)ell[j], config->nu, eta_m, gl, halfN+1); break;
			default: printf("Integral Not Supported! Please choose config->derivative from [0,1,2].\n");
		}

		// calculate y arrays
		for(int i=0; i<N_original; i++) {
			y[j][i] = (ell[j]+1.) / x[N_original-1-i];
		}
		const double y0 = y[j][0];

		for(int i=0; i<=halfN; i++) {
			out_vary[j][i] =
			conj(out[i] * cpow(x0*y0/exp((N-N_original)*dlnx), -I*eta_m[i]) * gl[i]);
		}

		fftw_execute(plan_backward[j]);

		for(int i=0; i<N_original; i++) {
			Fy[j][i] += out_ifft[j][i+N_pad+N_extrap_high] * sqrt(M_PI) /
				(4.*N * pow(y[j][i], config->nu));
		}
	}

	for (int j=0; j<Nell; j++) {
		fftw_destroy_plan(plan_backward[j]);
		fftw_free(out_vary[j]);
		free(out_ifft[j]);
	}
	free(plan_backward);
	free(out_vary);
	free(out_ifft);
	fftw_destroy_plan(plan_forward);
	fftw_free(out);
	free(fb);
}

// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// begin of new version
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

typedef fftw_complex fftwc;
typedef fftw_plan fftwp;

static void**** malloc4d_complex(const long nx, const long ny,
                                 const long nz, const long nw)
{
  const size_t align = 64;

  size_t nxp = nx * sizeof(double complex***);
  if (nxp % align != 0) nxp = nxp + (align - nxp % align);
  nxp = nxp / sizeof(double complex***);

  size_t s2p = nx * ny * sizeof(double complex**);
  if (s2p % align != 0) s2p = s2p + (align - s2p % align);
  s2p = s2p / sizeof(double complex**);

  size_t s3p = nx * ny * nz * sizeof(double complex*);
  if (s3p % align != 0) s3p = s3p + (align - s3p % align);
  s3p = s3p / sizeof(double complex*);

  size_t nwp = nw * sizeof(double complex);
  if (nwp % align != 0) nwp = nwp + (align - nwp % align);
  nwp = nwp / sizeof(double complex);

  void* raw_block = NULL;
  if (posix_memalign(&raw_block, align,
                     sizeof(double complex***) * nxp +
                     sizeof(double complex**)  * s2p +
                     sizeof(double complex*)   * s3p +
                     sizeof(double complex)    * nx * ny * nz * nwp) != 0) {
    log_fatal("array allocation failed (malloc4d_complex)"); exit(1);
  }

  double complex**** tab = (double complex****) raw_block;
  double complex***  lvl3 = (double complex***)
    ((char*) raw_block + sizeof(double complex***) * nxp);
  double complex**   lvl2 = (double complex**)
    ((char*) raw_block + sizeof(double complex***) * nxp +
                         sizeof(double complex**)  * s2p);
  double complex* data = (double complex*)
    ((char*) raw_block + sizeof(double complex***) * nxp +
                         sizeof(double complex**)  * s2p +
                         sizeof(double complex*)   * s3p);

  for (int i = 0; i < nx; i++) {
    tab[i] = lvl3 + i * ny;
    for (int j = 0; j < ny; j++) {
      tab[i][j] = lvl2 + (ny * i + j) * nz;
      for (int k = 0; k < nz; k++) {
        tab[i][j][k] = data + ((long)(ny * i + j) * nz + k) * nwp;
      }
    }
  }
  return (void****) tab;
}

static void**** malloc3d_ptr(const long nx, const long ny, const long nz)
{
  const size_t align = 64;

  size_t nxp = nx * sizeof(double***);
  if (nxp % align != 0) nxp = nxp + (align - nxp % align);
  nxp = nxp / sizeof(double***);

  size_t s2p = nx * ny * sizeof(double**);
  if (s2p % align != 0) s2p = s2p + (align - s2p % align);
  s2p = s2p / sizeof(double**);

  void* raw_block = NULL;
  if (posix_memalign(&raw_block, align,
                     sizeof(double***) * nxp +
                     sizeof(double**)  * s2p +
                     sizeof(double*)   * nx * ny * nz) != 0) {
    log_fatal("array allocation failed (malloc3d_ptr)"); exit(1);
  }

  double**** tab = (double****) raw_block;
  double***  lvl2 = (double***)
    ((char*) raw_block + sizeof(double***) * nxp);
  double**   lvl1 = (double**)
    ((char*) raw_block + sizeof(double***) * nxp +
                         sizeof(double**)  * s2p);

  for (int i = 0; i < nx; i++) {
    tab[i] = lvl2 + i * ny;
    for (int j = 0; j < ny; j++) {
      tab[i][j] = lvl1 + (ny * i + j) * nz;
    }
  }
  return (void****) tab;
}

static void**** malloc4d_fftwc(const long nx, const long ny,
                               const long nz, const long nw)
{
  const size_t align = 64;

  size_t nxp = nx * sizeof(fftwc***);
  if (nxp % align != 0) nxp = nxp + (align - nxp % align);
  nxp = nxp / sizeof(fftwc***);

  size_t s2p = nx * ny * sizeof(fftwc**);
  if (s2p % align != 0) s2p = s2p + (align - s2p % align);
  s2p = s2p / sizeof(fftwc**);

  size_t s3p = nx * ny * nz * sizeof(fftwc*);
  if (s3p % align != 0) s3p = s3p + (align - s3p % align);
  s3p = s3p / sizeof(fftwc*);

  size_t nwp = nw * sizeof(fftwc);
  if (nwp % align != 0) nwp = nwp + (align - nwp % align);
  nwp = nwp / sizeof(fftwc);

  void* raw_block = NULL;
  if (posix_memalign(&raw_block, align,
                     sizeof(fftwc***) * nxp +
                     sizeof(fftwc**)  * s2p +
                     sizeof(fftwc*)   * s3p +
                     sizeof(fftwc)    * nx * ny * nz * nwp) != 0) {
    log_fatal("array allocation failed (malloc4d_fftwc)"); exit(1);
  }

  fftwc**** tab = (fftwc****) raw_block;
  fftwc***  lvl3 = (fftwc***)
    ((char*) raw_block + sizeof(fftwc***) * nxp);
  fftwc**   lvl2 = (fftwc**)
    ((char*) raw_block + sizeof(fftwc***) * nxp +
                         sizeof(fftwc**)  * s2p);
  fftwc*    data = (fftwc*)
    ((char*) raw_block + sizeof(fftwc***) * nxp +
                         sizeof(fftwc**)  * s2p +
                         sizeof(fftwc*)   * s3p);

  for (int i = 0; i < nx; i++) {
    tab[i] = lvl3 + i * ny;
    for (int j = 0; j < ny; j++) {
      tab[i][j] = lvl2 + (ny * i + j) * nz;
      for (int k = 0; k < nz; k++) {
        tab[i][j][k] = data + ((long)(ny * i + j) * nz + k) * nwp;
      }
    }
  }
  return (void****) tab;
}

static void*** malloc3d_fftwp(const long nx, const long ny, const long nz)
{
  const size_t align = 64;

  size_t nxp = nx * sizeof(fftwp**);
  if (nxp % align != 0) nxp = nxp + (align - nxp % align);
  nxp = nxp / sizeof(fftwp**);

  size_t s2p = nx * ny * sizeof(fftwp*);
  if (s2p % align != 0) s2p = s2p + (align - s2p % align);
  s2p = s2p / sizeof(fftwp*);

  void* raw_block = NULL;
  if (posix_memalign(&raw_block, align,
                     sizeof(fftwp**) * nxp +
                     sizeof(fftwp*)  * s2p +
                     sizeof(fftwp)   * nx * ny * nz) != 0) {
    log_fatal("array allocation failed (malloc3d_fftwp)");
    exit(1);
  }

  fftwp*** tab = (fftwp***) raw_block;
  fftwp**  lvl2 = (fftwp**)
    ((char*) raw_block + sizeof(fftwp**) * nxp);
  fftwp*   lvl1 = (fftwp*)
    ((char*) raw_block + sizeof(fftwp**) * nxp +
                         sizeof(fftwp*)  * s2p);

  for (int i = 0; i < nx; i++) {
    tab[i] = lvl2 + i * ny;
    for (int j = 0; j < ny; j++) {
      tab[i][j] = lvl1 + (ny * i + j) * nz;
    }
  }
  return (void***) tab;
}

void cfftlog_ells_cocoa( // 1D array: 1st dim: zbins, 2nd dim: three FFT per bin 
		double* const x, 	 		// assume all fx have the same x-array
		double* const* const* const fx,
		int const Nx, 				 					 // assume all bins and FFTs have the same N's
		config* const cfg, 					 // assume all guns have the same config
		int* const* const ell,
		int* const LMAX, 					 
    double* const* const* const y, 
    double* const* const* const* const Fy,
	  int const SIZE1,
		int const SIZE2
	) 
{
	int Nmax = 0;
	int N[SIZE2][3];
	for(int j=0; j<SIZE2; j++) {
		N[j][0] = cfg[j].N_pad;
		N[j][1] = Nx;
		N[j][2] = 2*N[j][0] + N[j][1];
		if(N[j][2] % 2) {
			log_fatal("Please use even number of x"); 
			exit(1);
		}
		if (N[j][2] > Nmax) 
			Nmax = N[j][2];	
	}
	int imax, Nellmax = 0;
	for(int i=0; i<SIZE1; i++) {
		if (LMAX[i] > Nellmax) {
			Nellmax = LMAX[i];
			imax = i;
		}
	}
	const double sqrtpi = sqrt(M_PI);
	const double ln2 = log(2.);
	const double x0   = x[0];
	const double dlnx = log(x[1]/x[0]);
	const double complex clogpi = clog(M_PI);
	const double ln2pio2 = 0.5*log(2*M_PI);
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------	
	#pragma omp parallel for collapse(2) schedule(static,1)
	for(int i=0; i<SIZE1; i++) {
		for(int q=0; q<Nx; q++) { // q < Nx
			for (int k=0; k<LMAX[i]; k++) {
				y[i][k][q] = (ell[i][k] + 1.) / x[Nx -1 -q];
			}
		}
	}	
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------	
	double*** fb = (double***) malloc3d(SIZE1, SIZE2, Nmax); // biased input func
	#pragma omp parallel for collapse(2) schedule(static,1)
	for(int i=0; i<SIZE1; i++) {
		for(int j=0; j<SIZE2; j++) {
			for(int k=0; k<N[j][0]; k++) {
				fb[i][j][k] = 0.; // padding
			}
			for(int k=N[j][0]; k<N[j][0]+N[j][1]; k++) {
				const int q = k - N[j][0];
				if (q < 0 || q > Nx-1) {
					log_fatal("logical error on the array indexes"); exit(1);
				}
				fb[i][j][k] = fx[i][j][q] / pow(x[q], cfg[j].nu) ;
			}
			for(int k=N[j][0]+N[j][1]; k<N[j][2]; k++) {
				fb[i][j][k] = 0.; // padding
			}
		}
	}
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------	
	fftwc* toutfwd[SIZE1][SIZE2]; fftwp planf[SIZE1][SIZE2];
	for(int i=0; i<SIZE1; i++) {
		for(int j=0; j<SIZE2; j++) {
			toutfwd[i][j] = (fftwc*) fftw_malloc(sizeof(fftwc)*(N[j][2]/2+1));
			planf[i][j] = fftw_plan_dft_r2c_1d(N[j][2], fb[i][j], toutfwd[i][j], FFTW_ESTIMATE);
		}
	}
	#pragma omp parallel for collapse(2) schedule(static,1)
	for(int i=0; i<SIZE1; i++) {
		for(int j=0; j<SIZE2; j++) {
			fftw_execute(planf[i][j]);
			// c_window_cfft function begins -----------------------------------------
			const double cww = cfg[j].c_window_width;
			if( !(cww > 0) || !(cww	< 1)) {
				log_fatal("improper window width"); exit(1);
			}
			const int halfN = N[j][2]/2;
			const int kmax = (int) (halfN * cww);
			for(int k=0; k<(kmax+1); k++) { // window for right-side
				const double tmp = (double)(k)/kmax;
				const double W = (double)(k)/kmax - sin(2.*M_PI*k/kmax)/(2.*M_PI);
				toutfwd[i][j][N[j][2]/2-k] *= W;
			}
		}
	}
	fftwc* const* const* const* const outfwd = 
		(fftwc* const* const* const* const) malloc4d_fftwc(SIZE1,SIZE2,Nellmax,Nmax/2+1);
	double eta_m[SIZE2][Nmax/2+1];
	for(int j=0; j<SIZE2; j++) {
		#pragma omp parallel for schedule(static,1)
		for(int q=0; q<N[j][2]/2+1; q++) {
			eta_m[j][q] = (2.0*M_PI/(dlnx * N[j][2])) * q;	
		}
	}
	#pragma omp parallel for collapse(2) schedule(static,1)
	for(int i=0; i<SIZE1; i++) {
		for(int j=0; j<SIZE2; j++) {
			for (int k=0; k<LMAX[i]; k++) {
				for(int q=0; q<(N[j][2]/2+1); q++) { 
					outfwd[i][j][k][q] = toutfwd[i][j][q];
				}
			}
		}
	}
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	double complex* const* const* const* const gl = 
		(double complex* const* const* const* const) malloc4d_complex(SIZE1, SIZE2, Nellmax, Nmax/2+1);
	double pfac[] = {0.99999999999980993227684700473478,
									 676.520368121885098567009190444019,
									-1259.13921672240287047156078755283,
									 771.3234287776530788486528258894,
									-176.61502916214059906584551354,
									 12.507343278686904814458936853,
									-0.13857109526572011689554707,
									 9.984369578019570859563e-6,
									 1.50563273514931155834e-7};
	{
		int i = imax;
		for(int j=0; j<SIZE2; j++) {
			const double nu = cfg[j].nu;
			switch(cfg[j].derivative) 
			{	
				case 0: 
				{
					#pragma omp parallel for collapse(2) schedule(static,1)
					for (int k=0; k<LMAX[i]; k++) {
						for(int q=0; q<N[j][2]/2+1; q++) 
						{
							const double complex z = nu + I*eta_m[j][q];
							double complex part1;
							{
								const double complex a = 0.5*(ell[i][k] + z);
								if(creal(a) < 0.5) {
									double complex tmp = pfac[0];
									for(int w=1; w<9; w++) tmp += pfac[w] / ((-a) + w);
									const double complex t = (-a) + 7.5;
									part1 = clogpi - clog(csin(M_PI*a)) - 
													(ln2pio2 + ((-a) + 0.5)*clog(t) - t + clog(tmp));
								}
								else {
									double complex tmp = pfac[0];
									for(int w=1; w<9; w++) tmp += pfac[w] / ((a-1) + w);
									const double complex t = (a-1) + 7.5;
									part1 = ln2pio2 + ((a-1) + 0.5)*clog(t) - t + clog(tmp);
								}
							}
							double complex part2;
							{
								const double complex a = 0.5*(3 + ell[i][k] - z);
								if(creal(a) < 0.5) {
									double complex tmp = pfac[0];
									for(int w=1; w<9; w++) tmp += pfac[w] / ((-a) + w);
									const double complex t = (-a) + 7.5;
									part2 = clogpi - clog(csin(M_PI*a)) - 
													(ln2pio2 + ((-a) + 0.5)*clog(t) - t + clog(tmp));
								}
								else {
									double complex tmp = pfac[0];
									for(int w=1; w<9; w++) tmp += pfac[w] / ((a-1) + w);
									const double complex t = (a-1) + 7.5;
									part2 = ln2pio2 + ((a-1) + 0.5)*clog(t) - t + clog(tmp);
								}
							}
							gl[i][j][k][q] = cexp(z*ln2 + part1 - part2);	
						}
					}
					break;
				}
				case 1: 
				{
					#pragma omp parallel for collapse(2) schedule(static,1)
					for (int k=0; k<LMAX[i]; k++) {
						for(int q=0; q<N[j][2]/2+1; q++) {
							const double complex z = nu + I*eta_m[j][q];
							double complex part1;
							{
								const double complex a = 0.5*(ell[i][k] + z - 1.);
								if(creal(a) < 0.5) {
									double complex tmp = pfac[0];
									for(int w=1; w<9; w++) tmp += pfac[w] / ((-a) + w);
									const double complex t = (-a) + 7.5;
									part1 = clogpi - clog(csin(M_PI*a)) - 
													(ln2pio2 + ((-a) + 0.5)*clog(t) - t + clog(tmp));
								}
								else {
									double complex tmp = pfac[0];
									for(int w=1; w<9; w++) tmp += pfac[w] / ((a-1) + w);
									const double complex t = (a-1) + 7.5;
									part1 = ln2pio2 + ((a-1) + 0.5)*clog(t) - t + clog(tmp);
								}
							}
							double complex part2;
							{
								const double complex a = 0.5*(4 + ell[i][k] - z);
								double complex tmp = pfac[0];
								if(creal(a) < 0.5) {
									double complex tmp = pfac[0];
									for(int w=1; w<9; w++) tmp += pfac[w] / ((-a) + w);
									const double complex t = (-a) + 7.5;
									part2 = clogpi - clog(csin(M_PI*a)) - 
													(ln2pio2 + ((-a) + 0.5)*clog(t) - t + clog(tmp));
								}
								else {
									double complex tmp = pfac[0];
									for(int w=1; w<9; w++) tmp += pfac[w] / ((a-1) + w);
									const double complex t = (a-1) + 7.5;
									part2 = ln2pio2 + ((a-1) + 0.5)*clog(t) - t + clog(tmp);
								}
							}
							gl[i][j][k][q] = -(z-1)*cexp((z-1)*ln2 + part1 - part2);
						}
					}
					break;
				}
				case 2: 
				{
					#pragma omp parallel for collapse(2) schedule(static,1)
					for (int k=0; k<LMAX[i]; k++) {
						for(int q=0; q<N[j][2]/2+1; q++) {
							const double complex z = nu + I*eta_m[j][q];
							double complex part1;
							{
								const double complex a = 0.5*(ell[i][k] + z - 2);
								if(creal(a) < 0.5) {
									double complex tmp = pfac[0];
									for(int w=1; w<9; w++) tmp += pfac[w] / ((-a) + w);
									const double complex t = (-a) + 7.5;
									part1 = clogpi - clog(csin(M_PI*a)) - 
													(ln2pio2 + ((-a) + 0.5)*clog(t) - t + clog(tmp));
								}
								else {
									double complex tmp = pfac[0];
									for(int w=1; w<9; w++) tmp += pfac[w] / ((a-1) + w);
									const double complex t = (a-1) + 7.5;
									part1 = ln2pio2 + ((a-1) + 0.5)*clog(t) - t + clog(tmp);
								}
							}
							double complex part2;
							{
								const double complex a = 0.5*(5 + ell[i][k] - z);
								if(creal(a) < 0.5) {
									double complex tmp = pfac[0];
									for(int w=1; w<9; w++) tmp += pfac[w] / ((-a) + w);
									const double complex t = (-a) + 7.5;
									part2 = clogpi - clog(csin(M_PI*a)) - 
													(ln2pio2 + ((-a) + 0.5)*clog(t) - t + clog(tmp));
								}
								else {
									double complex tmp = pfac[0];
									for(int w=1; w<9; w++) tmp += pfac[w] / ((a-1) + w);
									const double complex t = (a-1) + 7.5;
									part2 = ln2pio2 + ((a-1) + 0.5)*clog(t) - t + clog(tmp);
								}
							}
							gl[i][j][k][q] = (z-1)*(z-2)*cexp((z-2)*ln2+part1-part2);
						}
					}
					break;
				}
			}
		}
	}
	for(int i=0; i<imax; i++) {
		for(int j=0; j<SIZE2; j++) {
			#pragma omp parallel for collapse(2) schedule(static,1)
			for (int k=0; k<LMAX[i]; k++) {
				for(int q=0; q<N[j][2]/2+1; q++) {
					gl[i][j][k][q] = gl[imax][j][k][q]; 
				}
			}
		}
	}
	if (imax+1 < SIZE1) {
		for(int i=imax+1; i<SIZE1; i++) {
			for(int j=0; j<SIZE2; j++) {
				#pragma omp parallel for collapse(2) schedule(static,1)
				for (int k=0; k<LMAX[i]; k++) {
					for(int q=0; q<N[j][2]/2+1; q++) {
						gl[i][j][k][q] = gl[imax][j][k][q]; 
					}
				}
			}
		}
	}
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	double**** outbcw = (double****) malloc3d_ptr(SIZE1, SIZE2, Nellmax);
	fftwp*** planb = (fftwp***) malloc3d_fftwp(SIZE1, SIZE2, Nellmax);
	for(int i=0; i<SIZE1; i++) {
		for(int j=0; j<SIZE2; j++) {
			for(int k=0; k<LMAX[i]; k++) {
				outbcw[i][j][k] = (double*) fftw_malloc(sizeof(double)*N[j][2]);
				planb[i][j][k] = fftw_plan_dft_c2r_1d(N[j][2], 
																							outfwd[i][j][k], 
																							outbcw[i][j][k], 
																							FFTW_ESTIMATE);
			}
		}
	}
	for(int i=0; i<SIZE1; i++) {
		for(int j=0; j<SIZE2; j++) {
			#pragma omp parallel for collapse(2) schedule(static,1)
			for (int k=0; k<LMAX[i]; k++) {
				for(int q=0; q<(N[j][2]/2+1); q++) { 
					const double y0 = y[i][k][0];			
					outfwd[i][j][k][q] *= cpow(x0*y0/exp(2*N[j][0]*dlnx),-I*eta_m[j][q]);
					outfwd[i][j][k][q] *= gl[i][j][k][q];
					outfwd[i][j][k][q]  = conj(outfwd[i][j][k][q]);
				}
			}
		}
	}
	for(int i=0; i<SIZE1; i++) {
		#pragma omp parallel for collapse(2) schedule(static,1)
		for(int j=0; j<SIZE2; j++) {
			for (int k=0; k<LMAX[i]; k++) {
				fftw_execute(planb[i][j][k]);
			}
		}
	}
	for(int i=0; i<SIZE1; i++) {
		#pragma omp parallel for collapse(3) schedule(static,1)
		for(int j=0; j<SIZE2; j++) {
			for (int k=0; k<LMAX[i]; k++) {
				for(int q=0; q<Nx; q++) {
					Fy[i][j][k][q] = outbcw[i][j][k][N[j][0]+q] * sqrtpi / 
													 (4.*N[j][2] * pow(y[i][k][q], cfg[j].nu));
				}
			}
		}
	}
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	for(int i=0; i<SIZE1; i++) {
		for(int j=0; j<SIZE2; j++) {
			free(toutfwd[i][j]);
			fftw_destroy_plan(planf[i][j]);
			for(int k=0; k<LMAX[i]; k++) {
				free(outbcw[i][j][k]);
				fftw_destroy_plan(planb[i][j][k]);
			}
		}
	}
	free((void*) outbcw); 
	free((void*) outfwd); 
	free((void*) gl); 
	free((void*) fb);
	free((void*) planb);
	return;
}