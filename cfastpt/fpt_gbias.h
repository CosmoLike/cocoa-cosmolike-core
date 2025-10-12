#ifndef __CFASTPT_CFASTPT_H
#define __CFASTPT_CFASTPT_H
#ifdef __cplusplus
extern "C" {
#endif

void Pd1d2(double *k, double *Pin, long Nk, double *Pout);
void Pd2d2(double *k, double *Pin, long Nk, double *Pout);
void Pd1s2(double *k, double *Pin, long Nk, double *Pout);
void Pd2s2(double *k, double *Pin, long Nk, double *Pout);
void Ps2s2(double *k, double *Pin, long Nk, double *Pout);
void Pd1d3nl(double *k, double *Pin, long Nk, double *Pout);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD