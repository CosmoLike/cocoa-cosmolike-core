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
#include <omp.h>

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
// begin of new experimental version
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------

typedef fftw_complex fftwc;
typedef fftw_plan fftwp;

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
)
{
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
	fftwp planf[SIZE2];
	for (int j=0; j<SIZE2; j++) {
	  planf[j] = fftw_plan_dft_r2c_1d(N[j][2], 
	  																fb[0][j], 
	  																toutfwd[0*SIZE2+j],
	                                  FFTW_ESTIMATE);
	}
	#pragma omp parallel for collapse(2) schedule(static,1)
	for(int i=0; i<SIZE1; i++) {
		for(int j=0; j<SIZE2; j++) {
			fftw_execute_dft_r2c(planf[j], fb[i][j], toutfwd[i*SIZE2+j]);
			// c_window_cfft function begins -----------------------------------------
			const double cww = cfg[j].c_window_width;
			if( !(cww > 0) || !(cww	< 1)) {
				log_fatal("improper window width"); exit(1);
			}
			const int halfN = N[j][2]/2;
			const int kmax = (int) (halfN * cww);
			for(int k=0; k<(kmax+1); k++) { // window for right-side
				const double W = (double)(k)/kmax - sin(2.*M_PI*k/kmax)/(2.*M_PI);
				toutfwd[i*SIZE2+j][N[j][2]/2-k] *= W;
			}
		}
	}
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	const double dlnx = log(x[1]/x[0]);
	for(int j=0; j<SIZE2; j++) {
		#pragma omp parallel for schedule(static,1)
		for(int q=0; q<N[j][2]/2+1; q++) {
			eta_m[j][q] = (2.0*M_PI/(dlnx * N[j][2])) * q;	
		}
	}
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	for (int j=0; j<SIZE2; j++) {
    fftw_destroy_plan(planf[j]);
  }
  free((void*) fb);
}

void cfftlog_ells_cocoa(
  double* const x,
  int const Nx,
  config* const cfg,
  const int* const ell,
  int const LMAX,
  double* const* const* const y,
  double* const* const* const* const Fy,
  fftwc* const* const toutfwd,
  double* const* const eta_m,
  int const N[][3],
  int const Nmax,
  int const ks, // k start
  int const ke, // k end
  int const SIZE1,
  int const SIZE2
) 
{
	const int kmax = (ke < LMAX) ? ke : LMAX;

	if (SIZE1 < 1 || SIZE2 < 1) {
    log_fatal("SIZE1 and SIZE2 must be >= 1");
    exit(1);
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
			for (int k=ks; k<kmax; k++) {
				y[i][k][q] = (ell[k] + 1.) / x[Nx -1 -q];
			}
		}
	}	
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	double complex* const* const* const gl = 
		(double complex* const* const* const) malloc3d_complex(SIZE2,ke-ks,Nmax/2+1);
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
		const int i = 0; // now LMAX is a scalar
		for(int j=0; j<SIZE2; j++) {
			const double nu = cfg[j].nu;
			switch(cfg[j].derivative) 
			{	
				case 0: 
				{
					#pragma omp parallel for collapse(2) schedule(static,1)
					for (int k=ks; k<ke; k++) {
						for(int q=0; q<N[j][2]/2+1; q++) 
						{
							const double complex z = nu + I*eta_m[j][q];
							double complex part1;
							{
								const double complex a = 0.5*(ell[k] + z);
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
								const double complex a = 0.5*(3 + ell[k] - z);
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
							gl[j][k-ks][q] = cexp(z*ln2 + part1 - part2);	
						}
					}
					break;
				}
				case 1: 
				{
					#pragma omp parallel for collapse(2) schedule(static,1)
					for (int k=ks; k<ke; k++) {
						for(int q=0; q<N[j][2]/2+1; q++) {
							const double complex z = nu + I*eta_m[j][q];
							double complex part1;
							{
								const double complex a = 0.5*(ell[k] + z - 1.);
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
								const double complex a = 0.5*(4 + ell[k] - z);
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
							gl[j][k-ks][q] = -(z-1)*cexp((z-1)*ln2 + part1 - part2);
						}
					}
					break;
				}
				case 2: 
				{
					#pragma omp parallel for collapse(2) schedule(static,1)
					for (int k=ks; k<ke; k++) {
						for(int q=0; q<N[j][2]/2+1; q++) {
							const double complex z = nu + I*eta_m[j][q];
							double complex part1;
							{
								const double complex a = 0.5*(ell[k] + z - 2);
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
								const double complex a = 0.5*(5 + ell[k] - z);
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
							gl[j][k-ks][q] = (z-1)*(z-2)*cexp((z-2)*ln2+part1-part2);
						}
					}
					break;
				}
			}
		}
	}
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------	
	const int NTHREADS = omp_get_max_threads();
	fftwc** outfwd = (fftwc**) malloc2d_fftwc(NTHREADS, Nmax/2+1);
	double** outbcw = (double**) malloc2d(NTHREADS, Nmax);

	fftwp planb[SIZE2];
	for(int j=0; j<SIZE2; j++) {
		planb[j] = fftw_plan_dft_c2r_1d(N[j][2],
																	  outfwd[0], 
																	  outbcw[0], 
																		FFTW_ESTIMATE);
	}

  double base_j[SIZE2];
  for(int j=0; j<SIZE2; j++) {
  	base_j[j] = x0 / exp(2 * N[j][0] * dlnx);
  }
	for(int i=0; i<SIZE1; i++) {
		#pragma omp parallel for collapse(2) schedule(static,1)
		for(int j=0; j<SIZE2; j++) {
			for (int k=ks; k<kmax; k++) {	
			  const int id = omp_get_thread_num();	
			  const double lnbase = log(base_j[j] * y[i][k][0]);		
				for(int q=0; q<(N[j][2]/2+1); q++) { 
					fftwc val = toutfwd[i*SIZE2+j][q];	
					const double phase = -eta_m[j][q] * lnbase;
					val *= cos(phase) + I * sin(phase);
					val *= gl[j][k-ks][q];
					outfwd[id][q] = conj(val);
				}
				
				fftw_execute_dft_c2r(planb[j], outfwd[id], outbcw[id]);
				
				for(int q=0; q<Nx; q++) {
					Fy[i][j][k][q] = outbcw[id][N[j][0]+q] * sqrtpi / 
													 (4.*N[j][2] * pow(y[i][k][q], cfg[j].nu));
				}
			}
		}
	}
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	// ---------------------------------------------------------------------------
	for(int j=0; j<SIZE2; j++) {
		fftw_destroy_plan(planb[j]);
	}
	free((void*) outbcw); 
	free((void*) outfwd); 
	free((void*) gl); 
	return;
}