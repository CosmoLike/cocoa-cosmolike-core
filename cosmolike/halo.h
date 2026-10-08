#ifndef __COSMOLIKE_HALO_H
#define __COSMOLIKE_HALO_H
#ifdef __cplusplus
extern "C" {
#endif

// HALO BIAS OPTIONS ---------------------------
#define HALO_BIAS_TINKER_2010 0
#define HALO_BIAS_SHETH_MO_TORMEN_2001 1

// HMF OPTIONS ---------------------------------
#define HMF_TINKER_2010 0
#define HMF_TINKER_2008 1

// CONCENTRATION OPTIONS -----------------------
#define CONCENTRATION_BHATTACHARYA_2013 0
#define CONCENTRATION_DUFFY_2008 1

// HALO PROFILE OPTIONS OPTIONS -----------------------
#define HALO_PROFILE_NFW 0

// FIELDS EXPOSED BY sigma2_field(M,a,field) -------------------------
// Both variances are available for diagnostics. Halo statistics always
// use CB: P_cb and rho_crit (Omega_m - Omega_nu), excluding neutrinos.
// MATTER uses P_lin and rho_crit Omega_m for total-matter comparisons.
#define HALO_FIELD_MATTER 0
#define HALO_FIELD_CB 1

double hb1nu(const double nu, const double a);

double fnu(const double nu, const double a);

double conc(const double m, const double a);

double bias_norm(const double a);

double dlognudlogm(const double M, const double a);

double u_nfw_c(const double c, const double k, const double m, const double a);

double ngal(const int ni, const double a);

double bgal(const int ni, const double a);

double p_gm(const double k, const double a, const int ni);

double p_gg(const double k, const double a, const int ni, const int nj);

void set_HOD(const int ni);

// halo-model intrinsic alignment (Fortuna et al. 2021), one IA
// population over the source redshift range; k in (c/H0)^-1
double ia_f_red_central(const double a);

double ia_p1h_dI(const double k, const double a);

double ia_p1h_II(const double k, const double a);

double ia_window_2h(const double k);

#ifdef __cplusplus
}
#endif
#endif // HEADER GUARD
