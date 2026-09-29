#ifndef __COSMOLIKE_HALO_H
#define __COSMOLIKE_HALO_H
#ifdef __cplusplus
extern "C" {
#endif

// HALO BIAS OPTIONS ---------------------------
#define HALO_BIAS_TINKER_2010 0

// HMF OPTIONS ---------------------------------
#define HMF_TINKER_2010 0

// CONCENTRATION OPTIONS -----------------------
#define CONCENTRATION_BHATTACHARYA_2013 0

// HALO PROFILE OPTIONS OPTIONS -----------------------
#define HALO_PROFILE_NFW 0

double hb1nu(const double nu, const double a);

double fnu(const double nu, const double a);

double conc(const double m, const double growfac_a);

void bias_norm_work(const double* a, const int na, double* out);

double bias_norm_nointerp(const double a);

double bias_norm(const double a);

double dlognudlogm(const double M);

double u_nfw_c(const double c, const double k, const double m, const double a);

double u_KS(double c, double k, const double rv);

double ngal_nointerp(const int ni, const double a, const int init);

double ngal(const int ni, const double a);

double mmean_nointerp(const int ni, const double a, const int init);

double fsat_nointerp(const int ni, const double a, const int init);

double bgal_nointerp(const int ni, const double a, const int init);

double bgal(const int ni, const double a);

double p_mm(const double k, const double a);

double p_gm(const double k, const double a, const int ni);

double p_gg(const double k, const double a, const int ni, const int nj);

double p_my(const double k, const double a);

double p_yy(const double k, const double a);

void set_HOD(const int ni);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
