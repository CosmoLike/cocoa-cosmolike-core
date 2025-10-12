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

typedef struct fastpt_todo 
{
  int isScalar;
  double* alpha;
  double* beta;
  double* ell;
  int* isP13type;
  double* coeff_ar;
  int Nterms;
} fastpt_todo;

typedef struct fastpt_todolist 
{
  fastpt_todo *fastpt_todo;
  int N_todo;
} fastpt_todolist;

void fastpt_scalar(int* alpha_ar, int* beta_ar, int* ell_ar, int* isP13type_ar, 
double* coeff_A_ar, int Nterms, double* Pout, double* k, double* Pin, int Nk);

void J_abl_ar(double* x, double* fx, long N, int* alpha, int* beta, int* ell, 
int* isP13type, int Nterms, fastpt_config* config, double** Fy);

void J_abl(double* x, double* fx, int alpha, int beta, long N, 
fastpt_config* config, int ell, double* Fy);

void fastpt_tensor(int* alpha_ar, int* beta_ar, int* J1_ar, int* J2_ar, 
int* Jk_ar, double* coeff_AB_ar, int Nterms, double* Pout, double* k, double* Pin, int Nk);

void J_abJ1J2Jk_ar(double* x, double* fx, long N, int* alpha, int* beta, 
int* J1, int* J2, int* Jk, int Nterms, fastpt_config* config, double** Fy);

void run_fastpt_tensor(double *k, double *Pin, long Nk, double *Pout,
int Nterms, int *alpha_ar, int *beta_ar, int *l1_ar, int *l2_ar, int *l_ar, double *coeff_A_ar);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD