#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "utils_complex_cfastpt.h"
#include "cfastpt.h"
#include "fpt_ia.h"


void IA_tt_P_E(double *k, double *Pin, long Nk, double *Pout){
	int Nterms          = 11;
	int alpha_ar[]      = {0,0,0,0,0,0,0,0,0,0,0};
	int beta_ar[]       = {0,0,0,0,0,0,0,0,0,0,0};
	int l1_ar[]         = {0,2,4,2,1,3,0,2,2,1,0};
	int l2_ar[]         = {0,0,0,2,1,1,0,0,2,1,0};
	int l_ar[]          = {0,0,0,0,1,1,2,2,2,3,4};
	double coeff_A_ar[] = {2.*(16./81.), 2*(713./1134.), 2*(38./315.), 2*(95./162), 2*(-107./60),\
							 2*(-19./15.), 2*(239./756.),  2*(11./9.),   2*(19./27.), 2*(-7./10.), \
							 2*(3./35)};
	run_fastpt_tensor(k, Pin, Nk, Pout, Nterms, alpha_ar, beta_ar, l1_ar, l2_ar, l_ar, coeff_A_ar);
}

void IA_tt_P_B(double *k, double *Pin, long Nk, double *Pout){
	int Nterms          = 10;
	int alpha_ar[]      = {0,0,0,0,0,0,0,0,0,0};
	int beta_ar[]       = {0,0,0,0,0,0,0,0,0,0};
	int l1_ar[]         = {0,2,4,2,1,3,0,2,2,1};
	int l2_ar[]         = {0,0,0,2,1,1,0,0,2,1};
	int l_ar[]          = {0,0,0,0,1,1,2,2,2,3};
	double coeff_A_ar[] = {2.*(-41./405), 2*(-298./567), 2*(-32./315), 2*(-40./81), 2*(59./45),\
						   2*(16./15.),   2*(-2./9.),    2*(-20./27.), 2*(-16./27), 2*(2./5.)};
	run_fastpt_tensor(k, Pin, Nk, Pout, Nterms, alpha_ar, beta_ar, l1_ar, l2_ar, l_ar, coeff_A_ar);
}

void IA_tt(double *k, double *Pin, long Nk, double *P_E, double *P_B){
	IA_tt_P_E(k, Pin, Nk, P_E);
	IA_tt_P_B(k, Pin, Nk, P_B);
}


void IA_deltaE1(double *k, double *Pin, long Nk, double *Pout){
	int Nterms          = 4;
	int alpha_ar[]      = {0, 0, 1, -1};
	int beta_ar[]       = {0, 0,-1,  1};
	int l1_ar[]         = {0, 0, 0,  0};
	int l2_ar[]         = {2, 2, 2,  2};
	int l_ar[]          = {0, 2, 1,  1};
	double coeff_A_ar[] = {2.*(17./21.), 2*(4./21.), 1., 1.};
	run_fastpt_tensor(k, Pin, Nk, Pout, Nterms, alpha_ar, beta_ar, l1_ar, l2_ar, l_ar, coeff_A_ar);
}

void IA_0E0E(double *k, double *Pin, long Nk, double *Pout){
	int Nterms          = 4;
	int alpha_ar[]      = {0,0,0,0};
	int beta_ar[]       = {0,0,0,0};
	int l1_ar[]         = {0,2,2,0};
	int l2_ar[]         = {0,0,2,4};
	int l_ar[]          = {0,0,0,0};
	double coeff_A_ar[] = {29./90., 5./63., 19./18., 19./35};
	run_fastpt_tensor(k, Pin, Nk, Pout, Nterms, alpha_ar, beta_ar, l1_ar, l2_ar, l_ar, coeff_A_ar);
}

void IA_0B0B(double *k, double *Pin, long Nk, double *Pout){
	int Nterms          = 5;
	int alpha_ar[]      = {0,0,0,0,0};
	int beta_ar[]       = {0,0,0,0,0};
	int l1_ar[]         = {0,2,2,0,1};
	int l2_ar[]         = {0,0,2,4,1};
	int l_ar[]          = {0,0,0,0,1};
	double coeff_A_ar[] = {2./45, -44./63, -8./9, -16./35, 2.};
	run_fastpt_tensor(k, Pin, Nk, Pout, Nterms, alpha_ar, beta_ar, l1_ar, l2_ar, l_ar, coeff_A_ar);
}

void IA_deltaE2(double *k, double *Pin, long Nk, double *Pout){
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
		f[i] = r* ( 768./7 - 256/(7293.*pow(r,10)) - 256/(3003.*pow(r,8)) - 256/(1001.*pow(r,6)) - 256/(231.*pow(r,4)) - 256/(21.*r*r)  );
	}
	for( ; i<Nk-1; i++){
		r = exps[i];
		f[i] = r* ( 30. + 146*r*r - 110*pow(r,4) + 30*pow(r,6) + log(fabs(r-1.)/(r+1.))*(15./r - 60.*r + 90*pow(r,3) - 60*pow(r,5) + 15*pow(r,7))  );
	}
	for(i=Nk; i<Nk-1+Ncut; i++){
		r = exps[i];
		f[i] = r* ( 30. + 146*r*r - 110*pow(r,4) + 30*pow(r,6) + log(fabs(r-1.)/(r+1.))*(15./r - 60.*r + 90*pow(r,3) - 60*pow(r,5) + 15*pow(r,7))  );
	}
	for( ; i<2*Nk-1; i++){
		r = exps[i];
		f[i] = r* ( 256*r*r - 256*pow(r,4) + (768*pow(r,6))/7. - (256*pow(r,8))/21. - (256*pow(r,10))/231. - (256*pow(r,12))/1001. - (256*pow(r,14))/3003.  );
	}
	f[Nk-1] = 96.;
	double g[3*Nk-2];
	fftconvolve_real(Pin, f, Nk, 2*Nk-1, g);
	for(i=0; i<Nk; i++){
		Pout[i] = 2.* pow(k[i],3)/(896.*M_PI*M_PI) * Pin[i] * g[Nk-1+i] * dL; 
	}
}	

void IA_ta(double *k, double *Pin, long Nk, double *P_dE1, double *P_dE2, double *P_0E0E, double *P_0B0B){
	IA_deltaE1(k, Pin, Nk, P_dE1);
	IA_deltaE2(k, Pin, Nk, P_dE2);
	IA_0E0E(k, Pin, Nk, P_0E0E);
	IA_0B0B(k, Pin, Nk, P_0B0B);
}

void IA_mix_P_A(double *k, double *Pin, long Nk, double *Pout){
	int Nterms          = 13;
	int alpha_ar[]      = {0,0,0,0,0,0,0, 1,1,1,1,1,1};
	int beta_ar[]       = {0,0,0,0,0,0,0, -1,-1,-1,-1,-1,-1};
	int l1_ar[]         = {0,2,0,2,1,1,0, 0,2,1,1,0,0};
	int l2_ar[]         = {0,0,0,0,1,1,0, 0,0,1,1,2,0};
	int l_ar[]          = {0,0,2,2,1,3,4, 1,1,0,2,1,3};
	double coeff_A_ar[] = {2.*(-31./210.), 2*(-34./63), 2*(-47./147), 2*(-8./63),2*(93./70), 2*(6./35), 2*(-8./245),\
						   2.*(-3./10),2.*(-1./3),2.*(1./2),2.*(1.),2.*(-1./3),2.*(-1./5)};
	run_fastpt_tensor(k, Pin, Nk, Pout, Nterms, alpha_ar, beta_ar, l1_ar, l2_ar, l_ar, coeff_A_ar);
}

void IA_mix_P_B(double *k, double *Pin, long Nk, double *Pout){
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
		f[i] = r* (-16./147 - 16/(415701.*pow(r,12)) - 32/(357357.*pow(r,10)) - 16/(63063.*pow(r,8)) - 64/(63063.*pow(r,6)) - 16/(1617.*pow(r,4)) + 32/(441.*r*r) )/2.;
	}
	for( ; i<Nk-1; i++){
		r = exps[i];
		f[i] = r* ((2.* r * (225.- 600.* r*r + 1198.* pow(r,4) - 600.* pow(r,6) + 225.* pow(r,8)) + \
    				225.* pow((r*r - 1.),4) * (r*r + 1.) * log(fabs(r-1)/(r+1)) )/(20160.* pow(r,3)) - 29./315*r*r )/2.;
	}
	for(i=Nk; i<Nk-1+Ncut; i++){
		r = exps[i];
		f[i] = r* ((2.* r * (225.- 600.* r*r + 1198.* pow(r,4) - 600.* pow(r,6) + 225.* pow(r,8)) + \
    				225.* pow((r*r - 1.),4) * (r*r + 1.) * log(fabs(r-1)/(r+1)) )/(20160.* pow(r,3)) - 29./315*r*r )/2.;
	}
	for( ; i<2*Nk-1; i++){
		r = exps[i];
		f[i] = r* ( (-16*pow(r,4))/147. + (32*pow(r,6))/441. - (16*pow(r,8))/1617. - (64*pow(r,10))/63063. - 16*pow(r,12)/63063. - (32*pow(r,14))/357357. - (16*pow(r,16))/415701. )/2.;
	}
	f[Nk-1] = -1./42.;
	double g[3*Nk-2];
	fftconvolve_real(Pin, f, Nk, 2*Nk-1, g);
	for(i=0; i<Nk; i++){
		Pout[i] = 4.* pow(k[i],3)/(2.*M_PI*M_PI) * Pin[i] * g[Nk-1+i] * dL; 
	}
}

void IA_D_EE(double *k, double *Pin, long Nk, double *Pout){
	int Nterms          = 8;
	int alpha_ar[]      = {0,0,0,0, 0,0,0,0};
	int beta_ar[]       = {0,0,0,0, 0,0,0,0};
	int l1_ar[]         = {0,2,4,0, 2,1,3,2};
	int l2_ar[]         = {0,0,0,0, 0,1,1,2};
	int l_ar[]          = {0,0,0,2, 2,1,1,0};
	double coeff_A_ar[] = {2.*(-43./540), 2*(-167./756), 2*(-19./105), 2*(1./18),\
						   2*(-7./18), 2*(11./20), 2*(19./20), 2.*(-19./54)};
	run_fastpt_tensor(k, Pin, Nk, Pout, Nterms, alpha_ar, beta_ar, l1_ar, l2_ar, l_ar, coeff_A_ar);
}

void IA_D_BB(double *k, double *Pin, long Nk, double *Pout){
	int Nterms          = 8;
	int alpha_ar[]      = {0,0,0,0, 0,0,0,0};
	int beta_ar[]       = {0,0,0,0, 0,0,0,0};
	int l1_ar[]         = {0,2,4,0, 2,1,3,2};
	int l2_ar[]         = {0,0,0,0, 0,1,1,2};
	int l_ar[]          = {0,0,0,2, 2,1,1,0};
	double coeff_A_ar[] = {2.*(13./135), 2*(86./189), 2*(16./105), 2*(2./9),\
						   2*(4./9), 2*(-13./15), 2*(-4./5), 2.*(8./27)};
	run_fastpt_tensor(k, Pin, Nk, Pout, Nterms, alpha_ar, beta_ar, l1_ar, l2_ar, l_ar, coeff_A_ar);
}

void IA_mix(double *k, double *Pin, long Nk, double *P_A, double *P_B, double *P_DEE, double *P_DBB){
	IA_mix_P_A(k, Pin, Nk, P_A);
	IA_mix_P_B(k, Pin, Nk, P_B);
	IA_D_EE(k, Pin, Nk, P_DEE);
	IA_D_BB(k, Pin, Nk, P_DBB);
}
