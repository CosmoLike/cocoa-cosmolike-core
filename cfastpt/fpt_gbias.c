#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utils_complex_cfastpt.h"
#include "cfastpt.h"
#include "fpt_gbias.h"

void Pd1d2(double *k, double *Pin, long Nk, double *Pout){
	int alpha_ar[]      = {0,0,1};
	int beta_ar[]       = {0,0,-1};
	int ell_ar[]        = {0,2,1};
	int isP13type_ar[]  = {0,0,0};
	double coeff_A_ar[] = {2.*(17./21), 2.*(4./21), 2.};
	int Nterms          = 3;

	fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Pd2d2(double *k, double *Pin, long Nk, double *Pout){
	int alpha_ar[]      = {0};
	int beta_ar[]       = {0};
	int ell_ar[]        = {0};
	int isP13type_ar[]  = {0};
	double coeff_A_ar[] = {2.};
	int Nterms          = 1;

	fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Pd1s2(double *k, double *Pin, long Nk, double *Pout){
	int alpha_ar[]      = {0,0,0,1,1};
	int beta_ar[]       = {0,0,0,-1,-1};
	int ell_ar[]        = {0,2,4,1,3};
	int isP13type_ar[]  = {0,0,0,0,0};
	double coeff_A_ar[] = {2*(8./315.),2*(254./441.),2*(16./245.),2*(4./15.),2*(2./5.)};
	int Nterms          = 5;

	fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Pd2s2(double *k, double *Pin, long Nk, double *Pout){
	int alpha_ar[]      = {0};
	int beta_ar[]       = {0};
	int ell_ar[]        = {2};
	int isP13type_ar[]  = {0};
	double coeff_A_ar[] = {2.*2./3.};
	int Nterms          = 1;

	fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Ps2s2(double *k, double *Pin, long Nk, double *Pout){
	int alpha_ar[]      = {0,0,0};
	int beta_ar[]       = {0,0,0};
	int ell_ar[]        = {0,2,4};
	int isP13type_ar[]  = {0,0,0};
	double coeff_A_ar[] = {2.*(4./45.), 2*(8./63.), 2*(8./35.)};
	int Nterms          = 3;

	fastpt_scalar(alpha_ar, beta_ar, ell_ar, isP13type_ar, coeff_A_ar, Nterms, Pout, k, Pin, Nk);
}

void Pd1d3nl(double *k, double *Pin, long Nk, double *Pout){
	int i;

	double exps[2*Nk-1], f[2*Nk-1];
	double dL = log(k[1]/k[0]);
	long Ncut = floor(3./dL);
	double r;
	for(i=0; i<2*Nk-1; i++){
		exps[i] = exp(-dL*(i-Nk+1));
	}

	for(i=0; i<Nk-1-Ncut; i++){
		r = exps[i];
		f[i] = r* (128./315 + 128./(2297295.*pow(r,12)) + 128./(945945.*pow(r,10)) + 128./(315315.*pow(r,8)) + 128./(72765.*pow(r,6)) + 128./(6615.*pow(r,4)) + 128./(735.*r*r) );
	}
	for( ; i<Nk-1; i++){
		r = exps[i];
		f[i] = r* ( (-6. *r + 22. *pow(r,3) + 22 * pow(r,5) - 6* pow(r,7)) - 3.*pow(r*r-1, 4) *log(fabs(r-1.)/(r+1.)) ) /126./pow(r,3);
	}
	for(i=Nk; i<Nk-1+Ncut; i++){
		r = exps[i];
		f[i] = r* ( (-6. *r + 22. *pow(r,3) + 22 * pow(r,5) - 6* pow(r,7)) - 3.*pow(r*r-1, 4) *log(fabs(r-1.)/(r+1.)) ) /126./pow(r,3);
	}
	for( ; i<2*Nk-1; i++){
		r = exps[i];
		f[i] = r* ( 128./315.*r*r - 128./735.*pow(r,4) + 128./6615.*pow(r,6) + 128./72765.*pow(r,8) + 128./315315.*pow(r,10) + 128./945945.*pow(r,12) );
	}
	f[Nk-1] = 16./63.;
	double g[3*Nk-2];
	fftconvolve_real(Pin, f, Nk, 2*Nk-1, g);
	for(i=0; i<Nk; i++){
		Pout[i] = pow(k[i],3)/(4.*M_PI*M_PI) * Pin[i] * g[Nk-1+i] * dL; 
	}
}
