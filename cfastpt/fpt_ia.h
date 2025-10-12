#ifndef __CFASTPT_CFASTPT_H
#define __CFASTPT_CFASTPT_H
#ifdef __cplusplus
extern "C" {
#endif

void IA_tt_P_E(double *k, double *Pin, long Nk, double *Pout);
void IA_tt_P_B(double *k, double *Pin, long Nk, double *Pout);
void IA_deltaE1(double *k, double *Pin, long Nk, double *Pout);
void IA_deltaE2(double *k, double *Pin, long Nk, double *Pout);
void IA_0E0E(double *k, double *Pin, long Nk, double *Pout);
void IA_0B0B(double *k, double *Pin, long Nk, double *Pout);
void IA_mix_P_A(double *k, double *Pin, long Nk, double *Pout);
void IA_mix_P_B(double *k, double *Pin, long Nk, double *Pout);
void IA_D_EE(double *k, double *Pin, long Nk, double *Pout);
void IA_D_BB(double *k, double *Pin, long Nk, double *Pout);

void IA_tt(double *k, double *Pin, long Nk, double *P_E, double *P_B);
void IA_ta(double *k, double *Pin, long Nk, double *P_dE1, double *P_dE2, double *P_0E0E, double *P_0B0B);
void IA_mix(double *k, double *Pin, long Nk, double *P_A, double *P_B, double *P_DEE, double *P_DBB);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD