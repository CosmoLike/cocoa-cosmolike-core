#ifndef __CFASTPT_UTILS_CFASTPT_H
#define __CFASTPT_UTILS_CFASTPT_H
#ifdef __cplusplus
extern "C" {
#endif

int J_table(int Ncols, int Nterms, int (*terms)[Ncols], double *coeff_A,
                int (*out)[Ncols], double *coeff_out);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
